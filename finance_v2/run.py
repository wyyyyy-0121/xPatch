"""Sequential AAPL v2 walk-forward runs with signatures, resumable receipts and calibration."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from .data import confirmation_split, load_market_data, quarterly_splits, training_dates
from .io import ROOT, digest, read_json, source_hashes, write_json
from .metrics import calibrate, calibrated, forecast_metrics
from .models import ALL_MODELS, NEURAL, PRETRAINED, STOCHASTIC, Pretrained, verify_model_lock
from .training import (fit_baseline, load_neural, neural_predictions, predict_baseline,
                       setup_runtime, synchronize, train_neural)


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def seeds_for(name, config):
    return config["seeds"] if name in STOCHASTIC else [0]


def run_path(output, split, feature_set, name, seed):
    return Path(output) / split["name"] / feature_set / f"{name}_seed{seed}"


def signature_of(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def validate_config(config):
    if config["schema_version"] != 1 or set(config["models"]) - ALL_MODELS:
        raise ValueError("Invalid v2 config/model")
    if config["horizons_minutes"] != [1, 5, 12] or config["lookback_returns"] != 240:
        raise ValueError("Protocol fixes lookback 240 and cumulative horizons 1/5/12")
    if config["feature_sets"] != ["close", "ohlcv"]:
        raise ValueError("Protocol fixes close and OHLCV panels")
    for key in ("origin_stride_minutes", "purge_sessions"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"Invalid {key}")
    if not config["seeds"] or len(config["seeds"]) != len(set(config["seeds"])):
        raise ValueError("Unique seeds required")
    splits = quarterly_splits(config)
    if len(splits) != 10 or splits[0]["name"] != "2023Q1" or splits[-1]["name"] != "2025Q2":
        raise ValueError("Expected original ten-quarter protocol")
    for split in splits:
        training_dates(split["train_dates"], config)


def identity_for(config):
    path = ROOT / config["data"]
    sha = digest(path)
    if sha != config["data_sha256"]:
        raise ValueError("Data hash differs from pre-registration")
    versions = {p: importlib.metadata.version(p) for p in (
        "torch", "numpy", "pandas", "scikit-learn", "exchange-calendars", "lightgbm",
        "chronos-forecasting", "transformers", "huggingface-hub", "accelerate", "einops", "safetensors")}
    return {"config": config, "data_sha256": sha, "sources": source_hashes(),
            "runtime_versions": versions, "python_version": platform.python_version(),
            "protocol_sha256": digest(ROOT / "PROTOCOL.md"),
            "model_lock_sha256": digest(ROOT / config["pretrained"]["lock"]),
            "requirements_sha256": digest(ROOT / "requirements-finance-v2.txt")}


def prepare_manifest(config, output):
    validate_config(config)
    identity = identity_for(config)
    signature = signature_of(identity)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "manifest.json"
    if target.exists():
        manifest = read_json(target)
        if manifest["signature"] != signature:
            raise RuntimeError("Signed source/config/data/protocol changed; use a new output directory")
    else:
        verify_model_lock(config)
        manifest = {"signature": signature, "identity": identity, "created_at_utc": utc_now(),
                    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                                    "numpy": np.__version__, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None},
                    "scope": "retrospective; previously exposed historical confirmation; no live orders or profitability claim"}
        write_json(target, manifest)
        (output / "environment.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    return manifest


def save_predictions(path, data, origins, pred, quantiles):
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, origins=origins, dates=data.dates[origins].astype(str),
                        timestamps=data.timestamps[origins].astype(str), y=data.targets(origins),
                        pred=pred, quantiles=quantiles)
    temporary.replace(path)


def check_receipt(directory, signature):
    result = read_json(directory / "result.json")
    if result["signature"] != signature:
        raise RuntimeError(f"Run signature mismatch: {directory}")
    for filename, sha in result["files"].items():
        if digest(directory / filename) != sha:
            raise RuntimeError(f"Changed or incomplete artifact: {directory / filename}")
    return result


def run_one(name, seed, config, data, split, output, signature, fixed_epochs=None,
            pretrained=None):
    directory = run_path(output, split, data.feature_set, name, seed)
    if (directory / "result.json").exists():
        result = check_receipt(directory, signature)
        print(f"RESUME {directory}", flush=True)
        return result
    directory.mkdir(parents=True, exist_ok=True)
    started_utc = utc_now()
    device = setup_runtime(config)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    stride = config["origin_stride_minutes"]
    dates = training_dates(split["train_dates"], config)
    samples = {k: data.samples(dates[k], stride) for k in ("early_fit", "epoch_selection", "final_fit", "calibration")}
    origins = data.samples(split["eval_dates"], stride)
    if not len(origins) or any(not len(v) for v in samples.values()):
        raise ValueError(f"No usable samples: {directory}")
    scale = data.scaler(dates["final_fit"])
    write_json(directory / "scaler.json", scale)
    write_json(directory / "split.json", {**split, "training_partition": dates})
    print(f"START {directory} fit={len(samples['final_fit'])} cal={len(samples['calibration'])} eval={len(origins)}", flush=True)
    start = time.perf_counter()
    details = {}
    if name in NEURAL:
        if fixed_epochs is None:
            inner_scale = data.scaler(dates["early_fit"])
            details["selection"] = train_neural(name, data, samples["early_fit"], samples["epoch_selection"],
                                                  inner_scale, config, seed, directory, phase="selection")
            fixed_epochs = details["selection"]["epoch"]
        details["final_fit"] = train_neural(name, data, samples["final_fit"], None,
                                             scale, config, seed, directory, fixed_epochs, "final")
        model, payload = load_neural(directory / "final.pt", device)
        details["parameters"] = sum(p.numel() for p in model.parameters())
        batch_size = config["runtime"]["batch_size_inference"]

        def predict_at(which):
            steps = neural_predictions(model, data.inputs(which, scale), batch_size, scale["target_std"])
            return np.cumsum(steps, axis=1)[:, np.asarray(data.horizons) - 1]
    elif name in PRETRAINED:
        if pretrained is None:
            pretrained = Pretrained(name, config, device)

        def predict_at(which):
            return pretrained.predict(data, which, seed)
    else:
        if name != "zero":
            details["final_fit"] = fit_baseline(name, data, samples["final_fit"], scale, config, directory)

        def predict_at(which):
            return predict_baseline(name, data, which, scale, directory)
    calibration_pred = predict_at(samples["calibration"])
    calibration = calibrate(data.targets(samples["calibration"]), calibration_pred,
                            config["calibration"]["quantile_levels"])
    calibration["dates"] = dates["calibration"]
    write_json(directory / "calibration.json", calibration)
    save_predictions(directory / "calibration_predictions.npz", data, samples["calibration"],
                     calibration_pred, calibrated(calibration_pred, calibration))
    synchronize(device)
    inference_started = time.perf_counter()
    prediction = predict_at(origins)
    synchronize(device)
    inference_seconds = time.perf_counter() - inference_started
    quantiles = calibrated(prediction, calibration)
    save_predictions(directory / "predictions.npz", data, origins, prediction, quantiles)
    if name in NEURAL:
        # Reconstruct weights in a fresh object and check a deterministic batch.
        reloaded, _ = load_neural(directory / "final.pt", device)
        batch = origins[:min(len(origins), config["runtime"]["batch_size_inference"])]
        steps = neural_predictions(reloaded, data.inputs(batch, scale), batch_size, scale["target_std"])
        check = np.cumsum(steps, axis=1)[:, np.asarray(data.horizons) - 1]
        np.testing.assert_array_equal(check, prediction[:len(batch)])
        details["checkpoint_sample_exact_reproduction"] = True
        del model, reloaded
    result = {"signature": signature, "model": name, "seed": seed, "feature_set": data.feature_set,
              "split": split["name"], "kind": split["kind"], "samples": len(origins),
              "calibration_samples": len(calibration_pred), "started_at_utc": started_utc,
              "completed_at_utc": utc_now(), "total_seconds": time.perf_counter() - start,
              "eval_prediction_seconds_including_data_transfers": inference_seconds,
              "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None,
              "details": details, "metrics": forecast_metrics(data.targets(origins), prediction,
                  data.horizons, quantiles, calibration["quantile_levels"])}
    if split["name"] == "historical_confirmation":
        result["freeze_sha256"] = digest(Path(output) / "frozen.json")
    files = sorted(p for p in directory.iterdir() if p.is_file() and not p.name.endswith(".tmp"))
    result["files"] = {p.name: digest(p) for p in files if p.name != "result.json"}
    write_json(directory / "result.json", result)
    print(f"DONE {directory} 12m_mse={result['metrics']['12']['mse']:.8g} seconds={result['total_seconds']:.1f}", flush=True)
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def load_panels(config):
    return {feature: load_market_data(ROOT / config["data"], config["lookback_returns"],
                tuple(config["horizons_minutes"]), feature,
                config["periods"]["development_start"], config["periods"]["historical_confirmation_end"])
            for feature in config["feature_sets"]}


def validation_runs(config, output, signature):
    results = []
    for split in quarterly_splits(config):
        for feature in config["feature_sets"]:
            for name in config["models"]:
                for seed in seeds_for(name, config):
                    results.append(check_receipt(run_path(output, split, feature, name, seed), signature))
    return results


def freeze(config, output, signature):
    results = validation_runs(config, output, signature)
    acceptance = read_json(output / "acceptance.json")
    if not (acceptance["passed"] and acceptance["signature"] == signature and acceptance["all_model_checks_requested"]):
        raise RuntimeError("Independent full model acceptance required before freezing")
    from .report import generate_report
    report = generate_report(output, "development")
    selected, originals, epochs = {}, {}, {}
    for feature in config["feature_sets"]:
        scores = report["panels"][feature]["models"]
        originals[feature] = min(NEURAL, key=lambda n: scores[n]["metrics"]["12"]["mse"])
        eligible = [name for name in config["models"] if name != "zero" and
                    report["panels"][feature]["vs_zero"][name]["development_gate_passed"]]
        selected[feature] = min(eligible, key=lambda n: scores[n]["metrics"]["12"]["mse"]) if eligible else None
        for name in NEURAL:
            values = [r["details"]["final_fit"]["epoch"] for r in results
                      if r["feature_set"] == feature and r["model"] == name]
            epochs[f"{feature}/{name}"] = max(1, int(np.floor(np.median(values) + 0.5)))
    frozen = {"signature": signature, "frozen_at_utc": utc_now(), "selected": selected,
              "selected_original": originals, "final_epochs": epochs,
              "development_report_sha256": digest(output / "development_summary.json"),
              "confirmation_is_independent": False,
              "epoch_rule": "rounded median of per-quarter and per-seed inner-selection epochs",
              "opening_rule": "only panels with a candidate passing all predeclared development gates"}
    path = output / "frozen.json"
    if path.exists():
        previous = read_json(path)
        for key in ("signature", "selected", "selected_original", "final_epochs", "development_report_sha256"):
            if previous[key] != frozen[key]:
                raise RuntimeError("Frozen decision changed")
        return previous
    write_json(path, frozen)
    return frozen


def run_validation(config, output, manifest, panels, model_subset=None):
    start = time.perf_counter()
    limit = config["runtime"]["max_validation_runtime_hours"] * 3600
    previous_seconds = sum(read_json(p).get("total_seconds", 0) for p in output.glob("20*Q*/*/*/result.json"))
    for split in quarterly_splits(config):
        for feature, data in panels.items():
            for name in config["models"]:
                if model_subset and name not in model_subset:
                    continue
                pretrained = None
                for seed in seeds_for(name, config):
                    if previous_seconds + time.perf_counter() - start > limit:
                        raise RuntimeError("Validation runtime budget reached; completed artifacts preserved")
                    artifact_bytes = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
                    if artifact_bytes > config["runtime"]["max_total_artifact_gib"] * 2**30:
                        raise RuntimeError("Artifact storage budget reached; completed files preserved")
                    if name in PRETRAINED and pretrained is None:
                        pretrained = Pretrained(name, config, setup_runtime(config))
                    run_one(name, seed, config, data, split, output, manifest["signature"], pretrained=pretrained)
                del pretrained


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/aapl_v2.json")
    parser.add_argument("--output", default="artifacts/aapl_v2/main")
    parser.add_argument("--stage", choices=["audit", "validate", "freeze", "confirm", "report", "verify", "all"], default="audit")
    parser.add_argument("--models", help="Comma-separated subset for staged validation only; protocol/manifest unchanged")
    args = parser.parse_args()
    config, output = read_json(args.config), Path(args.output)
    subset = set(args.models.split(",")) if args.models else None
    if subset and (args.stage != "validate" or subset - set(config["models"])):
        raise ValueError("Model subsets only allowed for validation and must be configured")
    manifest = prepare_manifest(config, output)
    panels = load_panels(config)
    write_json(output / "splits.json", quarterly_splits(config))
    write_json(output / "audit.json", {k: d.audit for k, d in panels.items()})
    print(f"Protocol signature={manifest['signature']} stride={config['origin_stride_minutes']}", flush=True)
    if args.stage in {"validate", "all"}:
        if (output / "frozen.json").exists():
            validation_runs(config, output, manifest["signature"])
        else:
            run_validation(config, output, manifest, panels, subset)
    if args.stage in {"freeze", "all"}:
        if args.stage == "all":
            from .verify import verify
            verify(output)
        freeze(config, output, manifest["signature"])
    if args.stage in {"confirm", "all"}:
        frozen = read_json(output / "frozen.json")
        if frozen["signature"] != manifest["signature"]:
            raise RuntimeError("Frozen signature mismatch")
        acceptance = read_json(output / "acceptance.json")
        if not (acceptance["passed"] and acceptance["signature"] == manifest["signature"] and acceptance["all_model_checks_requested"]):
            raise RuntimeError("Independent acceptance required before opening confirmation")
        split = confirmation_split(config)
        for feature, selected in frozen["selected"].items():
            if selected is None:
                print(f"GATE_STOP {feature}: no stable candidate; historical confirmation skipped", flush=True)
                continue
            for name in dict.fromkeys(["zero", "ridge", frozen["selected_original"][feature], selected]):
                for seed in seeds_for(name, config):
                    epochs = frozen["final_epochs"].get(f"{feature}/{name}")
                    run_one(name, seed, config, panels[feature], split, output, manifest["signature"], epochs)
    if args.stage in {"report", "all"}:
        from .report import generate_report
        generate_report(output, "development")
        if (output / "historical_confirmation").exists():
            generate_report(output, "historical_confirmation")
    if args.stage in {"verify", "all"}:
        from .verify import verify
        verify(output)


if __name__ == "__main__":
    main()
