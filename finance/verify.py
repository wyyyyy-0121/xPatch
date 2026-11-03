"""Independent CSV label/metric recomputation and saved-model prediction replay."""
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import Calibration
from .features import FitState
from .io import ROOT, read_json, sha, stable_hash, write_json
from .jobs import receipt
from .models import ChronosModel, load_model, raw_predictions


def reference_targets(closes, origin_timestamps):
    """31 actual source closes give 30 adjacent returns, independent of RVData."""
    if isinstance(origin_timestamps, pd.DatetimeIndex):
        if origin_timestamps.tz is None:
            raise ValueError("Independent target timestamps must be timezone aware")
        timestamps = origin_timestamps.as_unit("ns").asi8
    else:
        timestamps = np.asarray(origin_timestamps, dtype=np.int64)
    if np.any(timestamps < pd.Timestamp("2000-01-01", tz="UTC").value):
        raise ValueError("Integer target timestamps must use nanoseconds since the epoch")
    parts = []
    for start in range(0, len(timestamps), 4000):
        batch = timestamps[start:start + 4000]
        starts = batch[:, None] + np.arange(-1, 30) * 60 * 10**9
        prices = closes.reindex(pd.DatetimeIndex(starts.reshape(-1), tz="UTC")).to_numpy().reshape(-1, 31)
        if not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError("Independent target has missing/nonpositive source closes")
        parts.append(np.sum(np.diff(np.log(prices), axis=1) ** 2, axis=1))
    return np.concatenate(parts)


def independent_scores(y, p, dates):
    y = np.maximum(y, 1e-12)
    p = np.maximum(p, 1e-12)
    frame = pd.DataFrame({"date": dates, "qlike": y / p + np.log(p) - np.log(y) - 1,
        "log_rv_mse": (np.log(y) - np.log(p)) ** 2})
    grouped = frame.groupby("date", sort=True)[["qlike", "log_rv_mse"]].mean()
    return {key: float(grouped[key].mean()) for key in grouped}


def replay_indices(size, batch, pretrained):
    if not pretrained:
        return np.arange(size)
    # Preserve original batch boundaries, including the last partial batch.
    last = ((size - 1) // batch) * batch
    return np.unique(np.r_[np.arange(min(size, batch)), np.arange(last, size)])


def verify_one(context, request, directory):
    source = ROOT / request["source_directory"]
    original = receipt(source)
    if original is None or original["status"] != "complete":
        raise ValueError("Replay source is not a verified completed job")
    original_request = read_json(source / "request.json")
    if original["request_hash"] != stable_hash(original_request):
        raise ValueError("Replay source request fingerprint changed")
    if request.get("source_result_sha256", sha(source / "result.json")) != sha(source / "result.json"):
        raise ValueError("Replay source result changed")
    for key in ("signature", "model", "panel", "params", "config_id", "seed", "partition"):
        if stable_hash(request[key]) != stable_hash(original_request[key]):
            raise ValueError("Replay request differs from source model identity")
    config, data, store = context["config"], context["data"], context["store"]
    if original_request["phase"] not in ("outer", "confirmation", "delivery", "smoke"):
        raise ValueError("Replay only completed evaluation jobs")
    state_saved = FitState(**read_json(source / "fit_state.json"))
    state = FitState.fit(data, original_request["partition"]["fit_dates"], config["seasonality"])
    if state_saved.dictionary() != state.dictionary():
        raise ValueError("Fit-only transforms could not be rebuilt")
    if "reference_closes" not in context:
        entry = read_json(ROOT / config["data_manifest"])["assets"]["AAPL"]
        if sha(ROOT / entry["csv"]) != entry["csv_sha256"]:
            raise ValueError("Independent target source changed")
        frame = pd.read_csv(ROOT / entry["csv"], usecols=["timestamp", "close"])
        context["reference_closes"] = pd.Series(frame.close.to_numpy(), index=pd.DatetimeIndex(pd.to_datetime(frame.timestamp, utc=True)))
    with np.load(source / "predictions.npz", allow_pickle=False) as saved:
        values = {key: saved[key] for key in saved.files}
    origins, cal_origins = values["origins"], values["calibration_origins"]
    validate_prediction_grid(data, config, original_request, values)
    rebuilt_y = reference_targets(context["reference_closes"], values["timestamps"])
    rebuilt_cal_y = reference_targets(context["reference_closes"], data.timestamps[cal_origins].asi8)
    np.testing.assert_allclose(rebuilt_y, values["y"], rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(rebuilt_cal_y, values["calibration_y"], rtol=1e-8, atol=1e-14)
    recomputed = independent_scores(rebuilt_y, values["mean"], values["dates"])
    for key in recomputed:
        np.testing.assert_allclose(recomputed[key], original["score"][key], rtol=1e-8, atol=1e-10)
    name, params, panel = request["model"], request["params"], request["panel"]
    if name == "chronos2" and context["chronos"] is None:
        context["chronos"] = ChronosModel(config, context["device"])
    model = load_model(name, source, params, config, context["device"], context["chronos"])
    batch = config["search"]["chronos2"]["batch_size"] if name == "chronos2" else params.get("batch_size", 512)
    indices = replay_indices(len(origins), batch, name == "chronos2")
    cal_indices = replay_indices(len(cal_origins), batch, name == "chronos2")
    # First and last pretrained batches are called separately, as in original prediction.
    def replay(selected):
        if name != "chronos2":
            return raw_predictions(name, model, data, store, state, selected, params, panel, config, context["device"])
        return np.concatenate([raw_predictions(name, model, data, store, state, selected[start:start + batch], params, panel,
            config, context["device"]) for start in range(0, len(selected), batch)])
    raw, cal_raw = replay(origins[indices]), replay(cal_origins[cal_indices])
    np.testing.assert_allclose(raw, values["raw"][indices], rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(cal_raw, values["calibration_raw"][cal_indices], rtol=1e-6, atol=1e-12)
    calibration = Calibration(**read_json(source / "calibration.json"))
    rebuilt_cal = Calibration.fit(rebuilt_cal_y, values["calibration_raw"], data.dates[cal_origins], calibration.levels)
    np.testing.assert_allclose(rebuilt_cal.log_factor, calibration.log_factor, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(rebuilt_cal.residual_quantiles, calibration.residual_quantiles, rtol=1e-7, atol=1e-8)
    mean, quantiles = calibration.predict(raw)
    np.testing.assert_allclose(mean, values["mean"][indices], rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(quantiles, values["quantiles"][indices], rtol=1e-6, atol=1e-12)
    result = {"passed": True, "source_directory": request["source_directory"], "source_result_sha256": sha(source / "result.json"),
        "verification_scope": request.get("verification_scope", "smoke"), "round": request.get("verification_round"),
        "independent_labels": len(origins), "independent_calibration_labels": len(cal_origins),
        "replayed_origins": len(indices), "replayed_calibration_origins": len(cal_indices),
        "scope": "first_and_last_fixed_batches" if name == "chronos2" else "full",
        "max_raw_absolute_error": float(np.max(np.abs(raw - values["raw"][indices]))), "independent_scores": recomputed}
    write_json(Path(directory) / "verification.json", result)
    if name == "xpatch":
        import gc
        import torch
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def validate_prediction_grid(data, config, request, values):
    """A correct score on a truncated or substituted time grid is insufficient."""
    split, phase = request["partition"], request["phase"]
    stride = config["sampling"]["outer_stride_minutes"] if phase in ("outer", "confirmation", "delivery") else config["sampling"]["inner_validation_stride_minutes"]
    origins = data.origins(split["evaluation_dates"], stride=stride)
    calibration = data.origins(split["calibration_dates"], stride=config["sampling"]["calibration_stride_minutes"])
    if request.get("smoke_limit_prediction") is not None:
        if phase != "smoke":
            raise ValueError("Prediction limits cannot be applied to formal verification")
        origins, calibration = origins[:request["smoke_limit_prediction"]], calibration[:request["smoke_limit_prediction"]]
    for key, expected in (("origins", origins), ("calibration_origins", calibration),
            ("timestamps", data.timestamps[origins].asi8), ("dates", data.dates[origins])):
        if not np.array_equal(values[key], expected):
            raise ValueError(f"Stored {key} differs from the registered prediction grid")


SCOPES = ("development", "confirmation", "delivery")


def verification_directory(directory, round_number, scope):
    if round_number not in (1, 2) or scope not in SCOPES:
        raise ValueError("Verification needs a registered scope and round 1 or 2")
    parent = Path(directory) if scope == "development" else Path(directory) / ("confirmation" if scope == "confirmation" else "delivery_verification")
    return parent / f"verify_{round_number}"


def expected_sources(config, run_directory, scope):
    """Derive exact model/seed coverage, rather than trusting a summary count."""
    directory = Path(run_directory)
    setup = read_json(directory / "setup.json")
    if setup["config"] != config:
        raise ValueError("Verification setup differs from the registered config")
    expected = {}

    def add(path, identity):
        relative = str(path.relative_to(ROOT))
        request = read_json(path / "request.json")
        result = receipt(path, stable_hash(request))
        if result is None or result["status"] != "complete":
            raise ValueError(f"Verification source is incomplete: {relative}")
        for key, value in {"signature": setup["signature"], "directory": relative, **identity}.items():
            if stable_hash(request.get(key)) != stable_hash(value):
                raise ValueError(f"Verification source {key} differs: {relative}")
        for key in ("signature", "model", "panel", "config_id", "params", "seed", "phase", "directory", "job_id"):
            if stable_hash(result.get(key)) != stable_hash(request[key]):
                raise ValueError(f"Source receipt has another {key}: {relative}")
        if not {"request.json", "predictions.npz", "fit_state.json", "calibration.json"}.issubset(result["files"]):
            raise ValueError("Source receipt omits required prediction artifacts")
        if relative in expected:
            raise ValueError("Duplicate verification source")
        expected[relative] = request

    if scope == "development":
        for split in setup["partitions"]:
            for panel in config["feature_sets"]:
                for name in config["models"]:
                    group = directory / f"{split['name']}_{panel}_{name}"
                    experiment = read_json(group / "experiment.json")
                    selection = read_json(group / "selection.json")
                    registered = {row["id"]: row["params"] for row in setup["search"]["models"][name]}
                    params = registered[selection["config_id"]]
                    seeds = config["search"]["seeds"] if name in config["search"]["stochastic_models"] else [2026]
                    paths = [group / "outer" / selection["config_id"] / split["name"] / f"s{seed}" for seed in seeds]
                    if experiment["status"] != "complete" or sorted(experiment["outer_jobs"]) != sorted(str(p.relative_to(ROOT)) for p in paths):
                        raise ValueError("Development sources omit or substitute a registered model/seed")
                    for path, seed in zip(paths, seeds):
                        identity = {"phase": "outer", "partition": split, "model": name, "panel": panel,
                            "config_id": selection["config_id"], "params": params, "seed": seed}
                        if name in ("xpatch", "lightgbm"):
                            identity["fixed_epochs" if name == "xpatch" else "fixed_iterations"] = selection["fixed_training_length"]
                        add(path, identity)
    elif scope == "confirmation":
        from .cache import load_data, prepare_data
        from .splits import partition
        development = read_json(directory / "evaluation.json")
        confirmation = read_json(directory / "confirmation/evaluation.json")
        allowed = development["passed_candidates"]
        if not allowed or confirmation.get("allowed_by_development") != allowed:
            raise ValueError("Historical confirmation scope is not admitted")
        cache, identity = prepare_data(config)
        data = load_data(cache, identity)
        start, end = config["periods"]["historical_confirmation_start"], config["periods"]["historical_confirmation_end"]
        split = partition("historical_confirmation", [str(d) for d in data.days if d < start],
            [str(d) for d in data.days if start <= d <= end], config["splits"]["calibration_sessions"], config["splits"]["purge_sessions"]).dictionary()
        needed = sorted(set(allowed) | {key.split("/")[0] + "/har" for key in allowed})
        for key in needed:
            panel, name = key.split("/")
            selection = read_json(directory / f"2025Q2_{panel}_{name}/selection.json")
            params = next(row["params"] for row in setup["search"]["models"][name] if row["id"] == selection["config_id"])
            seeds = config["search"]["seeds"] if name in config["search"]["stochastic_models"] else [2026]
            for seed in seeds:
                source = directory / "confirmation" / panel / name / f"s{seed}"
                identity = {"phase": "confirmation", "partition": split, "model": name, "panel": panel,
                    "params": params, "config_id": selection["config_id"], "seed": seed}
                if name in ("xpatch", "lightgbm"):
                    identity["fixed_epochs" if name == "xpatch" else "fixed_iterations"] = selection["fixed_training_length"]
                add(source, identity)
        if sorted(confirmation["job_directories"]) != sorted(expected):
            raise ValueError("Confirmation sources omit or substitute a registered model/seed")
    elif scope == "delivery":
        from .cache import load_data, prepare_data
        from .splits import partition
        manifest = read_json(ROOT / "delivery/model.json")
        selection = read_json(directory / "2025Q2_aapl_har/selection.json")
        cache, identity = prepare_data(config)
        data = load_data(cache, identity)
        split = partition("delivery_baseline", data.days[:-1].tolist(), data.days[-1:].tolist(),
            config["splits"]["calibration_sessions"], config["splits"]["purge_sessions"]).dictionary()
        path = directory / "delivery_baseline"
        if manifest["source_directory"] != str(path.relative_to(ROOT)) or manifest["source_result_sha256"] != sha(path / "result.json"):
            raise ValueError("Delivery model differs from its replay source")
        if receipt(path)["started_utc"] < manifest["prerequisites"]["not_before_utc"]:
            raise ValueError("Delivery fitting began before independent verification completed")
        add(path, {"phase": "delivery", "partition": split, "model": "har", "panel": "aapl",
            "config_id": selection["config_id"], "params": selection["params"], "seed": 2026})
    else:
        raise ValueError("Unknown verification scope")
    if not expected:
        raise ValueError("An empty verification scope cannot pass")
    return expected


def verification_requests(directory, round_number, scope, sources):
    destination = verification_directory(directory, round_number, scope)
    return [{**{key: source[key] for key in ("signature", "model", "panel", "config_id", "params", "seed", "partition")},
        "job_id": f"verify{round_number}/{scope}/{source['job_id']}",
        "directory": str((destination / source["job_id"]).relative_to(ROOT)), "phase": "verify",
        "verification_scope": scope, "verification_round": round_number, "source_directory": path,
        "source_result_sha256": sha(ROOT / path / "result.json")} for path, source in sorted(sources.items())]


def verification_evidence(config, run_directory, round_number, scope="development"):
    sources = expected_sources(config, run_directory, scope)
    requests = verification_requests(run_directory, round_number, scope, sources)
    path = verification_directory(run_directory, round_number, scope) / "summary.json"
    summary = read_json(path)
    signature = read_json(Path(run_directory) / "setup.json")["signature"]
    source_digests = {r["source_directory"]: r["source_result_sha256"] for r in requests}
    if (summary.get("signature") != signature or summary.get("scope") != scope or summary.get("round") != round_number or
            summary.get("passed") is not True or summary.get("jobs") != len(requests) or summary.get("sources") != source_digests or
            sorted(summary.get("job_directories", [])) != sorted(r["directory"] for r in requests)):
        raise ValueError(f"Verification summary coverage or identity differs: {scope} round {round_number}")
    finished = []
    for request in requests:
        result = receipt(ROOT / request["directory"], stable_hash(request))
        if result is None or result["status"] != "complete":
            raise ValueError("Missing or failed independent verification receipt")
        finished.append(result["finished_utc"])
        if result["started_utc"] < receipt(ROOT / request["source_directory"])["finished_utc"]:
            raise ValueError("Replay began before its source was complete")
        verification = result["verification"]
        if (verification.get("passed") is not True or verification.get("source_directory") != request["source_directory"] or
                verification.get("source_result_sha256") != request["source_result_sha256"] or
                verification.get("verification_scope") != scope or verification.get("round") != round_number or
                read_json(ROOT / request["directory"] / "request.json") != request or
                "verification.json" not in result["files"] or
                read_json(ROOT / request["directory"] / "verification.json") != verification):
            raise ValueError("Independent verification receipt differs from its source/scope/round")
    if summary.get("completed_utc") != max(finished):
        raise ValueError("Independent verification completion time differs from its receipts")
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def require_verifications(config, run_directory, scope="development"):
    return {str(number): verification_evidence(config, run_directory, number, scope) for number in (1, 2)}


def verify_run(config, run_directory, round_number, scope="development"):
    from .cache import prepare_data
    from .provenance import verify_freeze
    from .scheduler import Scheduler
    freeze = verify_freeze(config)
    directory = Path(run_directory)
    setup = read_json(directory / "setup.json")
    if setup["signature"] != freeze["signature"]:
        raise ValueError("Verification run belongs to another freeze")
    destination = verification_directory(directory, round_number, scope)
    if round_number == 2:
        verification_evidence(config, directory, 1, scope)
    # Preserve the exact hashes used as prerequisites by downstream stages.
    if (destination / "summary.json").exists():
        verification_evidence(config, directory, round_number, scope)
        return read_json(destination / "summary.json")
    cache, identity = prepare_data(config)
    requests = verification_requests(directory, round_number, scope, expected_sources(config, directory, scope))
    with Scheduler(config, cache, identity, destination) as scheduler:
        results = scheduler.map(requests)
    summary = {"signature": freeze["signature"], "scope": scope, "round": round_number,
        "passed": all(r["status"] == "complete" and r["verification"]["passed"] for r in results),
        "jobs": len(results), "job_directories": [r["directory"] for r in results],
        "completed_utc": max(r["finished_utc"] for r in results),
        "sources": {r["source_directory"]: r["source_result_sha256"] for r in requests},
        "pretrained_scope": "first_and_last_fixed_batches_of_each_calibration_and_evaluation_job",
        "supervised_scope": "full_calibration_and_evaluation_predictions"}
    write_json(destination / "summary.json", summary)
    if summary["passed"]:
        verification_evidence(config, directory, round_number, scope)
    return summary
