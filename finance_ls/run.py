"""CLI: validation -> frozen protocol -> fixed-epoch held-out evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import torch
from sklearn.linear_model import Ridge

from .data import chronological_splits, load_market_data
from .metrics import forecast_metrics
from .models import ALL_MODELS, NEURAL_MODELS, make_model


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def source_hashes():
    paths = list((ROOT / "finance_ls").glob("*.py"))
    paths += [ROOT / p for p in ("layers/network.py", "layers/ema.py", "layers/decomp.py",
                                "layers/revin.py", "models/xPatch.py")]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def predict(model, x, batch_size):
    model.eval()
    return torch.cat([model(x[i:i + batch_size]) for i in range(0, len(x), batch_size)])


def save_predictions(path, data, origins, y, pred):
    np.savez_compressed(path, origins=origins,
                        timestamps=data.timestamps[origins].astype(str),
                        dates=data.dates[origins].astype(str),
                        y=np.asarray(y, dtype=np.float64), pred=np.asarray(pred, dtype=np.float64))


def load_checkpoint(path, device="cpu"):
    """Construct a model solely from its checkpoint; no training CSV is needed."""
    payload = torch.load(path, map_location=device, weights_only=True)
    model = make_model(payload["model"], payload["config"]).to(device)
    model.load_state_dict(payload["state_dict"])
    return model.eval(), payload


def fit_neural(name, config, seed, train, evaluate, scale, directory, fixed_epochs=None):
    device = torch.device(config["device"])
    seed_everything(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    x_train, y_train = (torch.from_numpy(a).to(device) for a in train)
    x_eval = torch.from_numpy(evaluate[0]).to(device)
    model = make_model(name, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    batch_size = config["batch_size"]
    checkpoint = directory / "checkpoint.pt"
    best, best_epoch, bad_epochs = float("inf"), 0, 0
    history = []
    sync(device)
    started = time.perf_counter()
    epochs = fixed_epochs if fixed_epochs is not None else config["max_epochs"]
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(x_train), device=device)
        sum_loss = torch.zeros((), device=device)
        for i in range(0, len(x_train), batch_size):
            indices = permutation[i:i + batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_train[indices])
            loss = torch.mean((prediction - y_train[indices]) ** 2)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss in {name}, epoch {epoch}")
            loss.backward()
            optimizer.step()
            sum_loss += loss.detach() * len(indices)
        entry = {"epoch": epoch, "train_scaled_mse": float(sum_loss / len(x_train))}
        if fixed_epochs is None:
            prediction = predict(model, x_eval, batch_size).cpu().numpy()[..., 0].astype(np.float64)
            actual = evaluate[1][..., 0].astype(np.float64)
            score = float(np.mean(((prediction - actual).sum(axis=1) * scale["std"]) ** 2))
            entry["validation_cumulative12_mse"] = score
            improved = score < best
            if improved:
                best, best_epoch, bad_epochs = score, epoch, 0
            else:
                bad_epochs += 1
        else:
            # Held-out labels are never consulted for fitting or epoch selection.
            improved = epoch == epochs
            if improved:
                best_epoch = epoch
        history.append(entry)
        if improved:
            torch.save({"model": name, "config": config, "seed": seed,
                        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        "scaler": scale, "epoch": best_epoch,
                        "training_loss": "per-minute scaled log-return MSE"}, checkpoint)
        print(f"{directory.name} epoch={epoch} {entry}", flush=True)
        if fixed_epochs is None and bad_epochs >= config["patience"]:
            break
    sync(device)
    training_seconds = time.perf_counter() - started
    write_json(directory / "history.json", history)
    # Use the saved best weights, rather than the last optimizer state.
    model, payload = load_checkpoint(checkpoint, device)
    expected = predict(model, x_eval, batch_size).cpu().numpy()[..., 0]
    # Independent construction/load must reproduce every prediction.
    reloaded, _ = load_checkpoint(checkpoint, device)
    reproduced = predict(reloaded, x_eval, batch_size).cpu().numpy()[..., 0]
    if not np.array_equal(expected, reproduced):
        raise AssertionError("Reloaded checkpoint does not reproduce predictions exactly")
    del reloaded
    # Warm up and time inference only, excluding CPU transfer and label processing.
    benchmark = x_eval[:min(len(x_eval), batch_size)]
    for _ in range(3):
        predict(model, benchmark, batch_size)
    sync(device)
    start = time.perf_counter()
    for _ in range(20):
        predict(model, benchmark, batch_size)
    sync(device)
    latency = (time.perf_counter() - start) / (20 * len(benchmark)) * 1000
    pred = expected.astype(np.float64) * scale["std"]
    peak_memory = torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda" else None
    result = {"best_epoch": payload["epoch"], "epochs_run": len(history),
              "parameters": sum(p.numel() for p in model.parameters()),
              "peak_cuda_allocated_mib": peak_memory,
              "inference_ms_per_sample": latency, "inference_batch_size": len(benchmark),
              "training_seconds": training_seconds, "checkpoint_exact_reproduction": True}
    del model, optimizer, x_train, y_train, x_eval
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pred, result


def run_one(name, seed, config, data, split, output, fixed_epochs=None):
    directory = output / split["name"] / f"{name}_seed{seed}"
    if (directory / "result.json").exists():
        if not (directory / "predictions.npz").exists():
            raise RuntimeError(f"Missing predictions in {directory}")
        print(f"Resume: {directory}", flush=True)
        return read_json(directory / "result.json")
    directory.mkdir(parents=True, exist_ok=True)
    run_started_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    train_origins, eval_origins = (data.samples(split[key]) for key in ("train_dates", "eval_dates"))
    scale = data.scale(split["train_dates"])
    train = data.arrays(train_origins, scale["std"])
    evaluate = data.arrays(eval_origins, scale["std"])
    # Labels for scoring are recovered directly from unscaled returns, not float32 round-trips.
    truth = data.returns[eval_origins[:, None] + np.arange(1, data.horizon + 1)]
    write_json(directory / "scaler.json", scale)
    write_json(directory / "split.json", split)
    print(f"START {split['name']} {name} seed={seed} train={len(train_origins)} eval={len(eval_origins)}", flush=True)
    if name in NEURAL_MODELS:
        pred, result = fit_neural(name, config, seed, train, evaluate, scale, directory, fixed_epochs)
    else:
        start = time.perf_counter()
        if name == "zero":
            pred = np.zeros_like(truth)
            parameters = 0
        else:
            ridge = Ridge(alpha=config["ridge_alpha"])
            ridge.fit(train[0][..., 0].astype(np.float64), train[1][..., 0].astype(np.float64))
            training_seconds = time.perf_counter() - start
            x_eval = evaluate[0][..., 0].astype(np.float64)
            predicted_scaled = ridge.predict(x_eval)
            np.savez_compressed(directory / "checkpoint.npz", coef=ridge.coef_, intercept=ridge.intercept_)
            with np.load(directory / "checkpoint.npz") as state:
                reproduced = x_eval @ state["coef"].T + state["intercept"]
            if not np.array_equal(reproduced, predicted_scaled):
                np.testing.assert_allclose(reproduced, predicted_scaled, rtol=1e-13, atol=1e-13)
            pred = predicted_scaled * scale["std"]
            parameters = ridge.coef_.size + ridge.intercept_.size
        if name == "zero":
            training_seconds = time.perf_counter() - start
        start = time.perf_counter()
        for _ in range(20):
            if name == "zero":
                np.zeros_like(truth[:config["batch_size"]])
            else:
                ridge.predict(x_eval[:config["batch_size"]])
        timed_samples = min(len(truth), config["batch_size"])
        result = {"parameters": parameters, "peak_cuda_allocated_mib": None,
                  "inference_ms_per_sample": (time.perf_counter() - start) / (20 * timed_samples) * 1000,
                  "inference_batch_size": timed_samples, "training_seconds": training_seconds,
                  "checkpoint_exact_reproduction": True}
    save_predictions(directory / "predictions.npz", data, eval_origins, truth, pred)
    result.update({"model": name, "seed": seed, "split": split["name"], "samples": len(truth),
                   "started_at_utc": run_started_at_utc,
                   "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "metrics": forecast_metrics(truth, pred)})
    if split["name"] == "test":
        result["frozen_protocol_sha256"] = hashlib.sha256((output / "frozen.json").read_bytes()).hexdigest()
    write_json(directory / "result.json", result)
    return result


def validation_results(output, config):
    results = []
    for fold in range(1, config["folds"] + 1):
        for name in config["models"]:
            for seed in (config["seeds"] if name in NEURAL_MODELS else [0]):
                path = output / f"fold_{fold}" / f"{name}_seed{seed}" / "result.json"
                if not path.exists():
                    raise RuntimeError(f"Validation incomplete: {path}")
                results.append(read_json(path))
    return results


def freeze(output, config, signature):
    results = validation_results(output, config)
    scores, epochs = {}, {}
    for name in config["models"]:
        group = [r for r in results if r["model"] == name]
        scores[name] = float(np.average([r["metrics"]["12"]["mse"] for r in group],
                                        weights=[r["samples"] for r in group]))
        if name in NEURAL_MODELS:
            epochs[name] = max(1, int(np.floor(np.median([r["best_epoch"] for r in group]) + 0.5)))
    originals = [n for n in ("xpatch48", "xpatch240") if n in scores]
    protocol = {"signature": signature, "validation_cumulative12_mse": scores,
                "final_epochs": epochs,
                "selected_original": min(originals, key=scores.get) if originals else None,
                "primary_candidate": "ls", "test_selection": "none",
                "epoch_rule": "rounded median best epoch across development folds and seeds",
                "claim_rule": "LS beats zero and ridge; paired day-block CI upper bound < 0 against frozen original; all seeds improve",
                "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if (output / "frozen.json").exists():
        existing = read_json(output / "frozen.json")
        for key in ("signature", "validation_cumulative12_mse", "final_epochs", "selected_original"):
            if existing[key] != protocol[key]:
                raise RuntimeError("Frozen validation protocol changed; use a fresh output directory")
        return existing
    write_json(output / "frozen.json", protocol)
    return protocol


def validate_config(config):
    if not config["models"] or len(set(config["models"])) != len(config["models"]) or set(config["models"]) - ALL_MODELS:
        raise ValueError("Invalid or duplicate model names")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Need unique seeds")
    if config["horizon"] != 12:
        raise ValueError("This protocol fixes a 12-minute horizon")
    for key in ("lookback", "short_len", "horizon", "pool", "patch_len", "stride", "folds",
                "batch_size", "max_epochs", "patience", "bootstrap_repeats", "threads"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not (0 < config["alpha"] <= 1 and config["learning_rate"] > 0 and config["ridge_alpha"] > 0):
        raise ValueError("Invalid EMA, learning rate, or ridge settings")
    if config["lookback"] < config["short_len"] or config["short_len"] < config["patch_len"]:
        raise ValueError("Incompatible history and patch lengths")
    if config["lookback"] % config["pool"]:
        raise ValueError("Lookback must be divisible by pool")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/finance_ls.json")
    parser.add_argument("--output", default="artifacts/finance_ls")
    parser.add_argument("--stage", choices=["audit", "validate", "freeze", "test", "report", "all"], default="all")
    args = parser.parse_args()
    config = read_json(args.config)
    validate_config(config)
    if config["device"] == "auto":
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(config["threads"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    path = Path(config["data"])
    if not path.is_absolute():
        path = ROOT / path
    data = load_market_data(path, config["lookback"], config["horizon"])
    splits = chronological_splits(data, config["test_fraction"], config["folds"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    identity = {"config": config, "data_sha256": data.audit["sha256"], "sources": source_hashes()}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path)["signature"] != signature:
        raise RuntimeError("Config, source, or data changed; use a new output directory")
    if not manifest_path.exists():
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        except subprocess.CalledProcessError:
            commit = "unknown"
        write_json(manifest_path, {"signature": signature, **identity, "upstream_commit": commit,
                                  "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                                                  "numpy": np.__version__, "device": config["device"],
                                                  "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None},
                                  "scope": "single AAPL data source; no transaction-cost or profitability claim"})
        import sys
        lock = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        (output / "environment.txt").write_text(lock)
    write_json(output / "audit.json", data.audit)
    write_json(output / "splits.json", splits)
    print(f"Data: {data.audit['rows']} prices, {len(data.origins)} eligible origins; {config['device']}", flush=True)
    if args.stage == "audit":
        return
    if args.stage in ("validate", "all"):
        if (output / "frozen.json").exists():
            validation_results(output, config)
        else:
            for split in splits[:-1]:
                for name in config["models"]:
                    for seed in (config["seeds"] if name in NEURAL_MODELS else [0]):
                        run_one(name, seed, config, data, split, output)
    if args.stage in ("freeze", "all"):
        freeze(output, config, signature)
    if args.stage in ("test", "all"):
        if not (output / "frozen.json").exists():
            raise RuntimeError("Run validation and freeze before opening the test set")
        protocol = read_json(output / "frozen.json")
        if protocol["signature"] != signature:
            raise RuntimeError("Frozen signature mismatch")
        for name in config["models"]:
            for seed in (config["seeds"] if name in NEURAL_MODELS else [0]):
                run_one(name, seed, config, data, splits[-1], output,
                        fixed_epochs=protocol["final_epochs"].get(name))
    if args.stage in ("report", "all"):
        from .report import generate_report
        generate_report(output)


if __name__ == "__main__":
    main()
