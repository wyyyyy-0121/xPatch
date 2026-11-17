"""Bounded local engineering check; never scores a forecast or starts a matrix."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from .bars import Bars, file_digest, targets
from .baselines import predict as baseline
from .features import CloseWindows, fit_scaler
from .models import make_model
from .ridge import fit as fit_ridge
from .spec import OUT, PROFILES, TARGETS
from .splits import rolling_folds


def step(model, optimizer, x, anchor, y):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = model(x, anchor)
    if prediction.shape != y.shape:
        raise ValueError("Prediction and target shapes differ")
    loss = (prediction - y).square().mean()
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite training loss")
    loss.backward()
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError("Missing or nonfinite gradient")
    optimizer.step()
    return prediction


def compare_nested(left, right):
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.shape == right.shape and torch.allclose(left.cpu(), right.cpu(), rtol=1e-6, atol=1e-7)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(compare_nested(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(compare_nested(x, y) for x, y in zip(left, right))
    return left == right


def restore_check(checkpoint_path: Path, reference_path: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    model = make_model(checkpoint["model_name"], checkpoint["lookback"], checkpoint["horizon"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=checkpoint["learning_rate"])
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["rng_cpu"].cpu())
    if device == "cuda":
        torch.cuda.set_rng_state(checkpoint["rng_cuda"].cpu())
    step(model, optimizer, checkpoint["x"].to(device), checkpoint["anchor"].to(device), checkpoint["y"].to(device))
    if device == "cuda":
        torch.cuda.synchronize()
    if not compare_nested(model.state_dict(), reference["model"]):
        raise ValueError("Restored model state differs from the next uninterrupted step")
    if not compare_nested(optimizer.state_dict(), reference["optimizer"]):
        raise ValueError("Restored optimizer state differs from the next uninterrupted step")
    print(json.dumps({"status": "passed", "comparison": "model and optimizer after one resumed step",
                      "pid": __import__("os").getpid()}))


def read_bars(directory: Path, manifest: dict, frequency: str):
    name = f"bars_{frequency}.npz"
    path = directory / name
    if file_digest(path) != manifest["files"][name]:
        raise ValueError("Bars differ from admission receipt")
    with np.load(path, allow_pickle=False) as value:
        start, end, dates = value["start_ns"], value["end_ns"], value["session_date"]
        if frequency in {"day", "month"}:
            # Until academic usage is verified, test model shapes with synthetic
            # prices on the public exchange calendar, not licensed long-run prices.
            pos = np.arange(len(start), dtype=np.float64)
            close = 100.0 + pos * 0.01 + 2.0 * np.sin(pos / 19.0)
            observed = np.ones(len(start), dtype=bool)
        else:
            close, observed = value["close"], value["observed"]
        return Bars(frequency, start, end, close, observed, dates)


def run(admission_dir: Path, device: str):
    directory = admission_dir.resolve()
    if not directory.is_relative_to((OUT / "admission").resolve()):
        raise ValueError("Admission path outside the isolated multiscale directory")
    with (directory / "admission.json").open(encoding="utf-8") as stream:
        receipt = json.load(stream)
    if receipt.get("Q3_values_read") is not False or receipt.get("experiment_scored") is not False:
        raise ValueError("Unexpected admission receipt state")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Local CUDA is unavailable")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    run_dir = OUT / "smoke" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    all_bars = {freq: read_bars(directory, receipt, freq) for freq in ("minute", "hour", "day", "month")}
    actions = receipt.get("split_actions") or receipt.get("split_actions_agree_between_provider_responses")
    if not actions:
        raise ValueError("Known split-action evidence is missing")
    periods = {"minute": 390, "hour": 6, "day": 252, "month": 12}
    records, recovery = [], None
    for profile in PROFILES:
        bars = all_bars[profile.frequency]
        windows = CloseWindows(bars, max(profile.lookbacks), actions)
        for horizon in profile.horizons:
            origin_file = f"origins_{profile.name}_h{horizon}.npy"
            origin_path = directory / origin_file
            if file_digest(origin_path) != receipt["files"][origin_file]:
                raise ValueError("Origins differ from admission receipt")
            origins = np.load(origin_path, allow_pickle=False)
            folds = rolling_folds(bars, origins, horizon)
            eligible = [fold for fold in folds if fold["inner"] is not None and len(fold["inner"]["fit"]) >= 8]
            if len(folds) != 5 or not eligible:
                raise ValueError("No development fold has eight fit origins for the engineering check")
            fit = eligible[0]["inner"]["fit"][:64]
            batch = fit[:8]
            for mode in TARGETS:
                scaler = fit_scaler(windows, fit, horizon, mode)
                for name in ("last_close", "drift", "seasonal"):
                    pred = baseline(name, windows, batch, horizon, mode,
                                    period=periods[profile.frequency] if name == "seasonal" else None)
                    if pred.shape != (len(batch), horizon) or not np.isfinite(pred).all():
                        raise ValueError("Baseline output invalid")
                ridge = fit_ridge(windows, scaler, fit, horizon, mode, alpha=1.0)
                pred = ridge.predict(windows, scaler, batch)
                if pred.shape != (len(batch), horizon) or not np.isfinite(pred).all():
                    raise ValueError("Ridge output invalid")
                x, anchor = windows.arrays(batch)
                x, anchor = scaler.transform_x(x, anchor)
                y = scaler.transform_y(targets(bars, batch, horizon, mode))
                x = torch.as_tensor(x[:, :, None], device=device)
                anchor = torch.as_tensor(anchor, device=device)
                y = torch.as_tensor(y, device=device)
                for name in ("mlp", "gru", "lstm", "xpatch"):
                    torch.manual_seed(2026)
                    if device == "cuda":
                        torch.cuda.manual_seed_all(2026)
                    model = make_model(name, windows.lookback, horizon).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                    before = [p.detach().clone() for p in model.parameters()]
                    step(model, optimizer, x, anchor, y)
                    if not any(not torch.equal(a, b.detach()) for a, b in zip(before, model.parameters())):
                        raise ValueError("No trainable tensor changed")
                    records.append({"profile": profile.name, "horizon": horizon, "target": mode,
                                    "model": name, "lookback": windows.lookback, "batch": len(batch),
                                    "engineering_fold": eligible[0]["fold"],
                                    "parameter_count": sum(p.numel() for p in model.parameters()),
                                    "native_horizon": getattr(model, "native_horizon", horizon),
                                    "one_update_finite": True})
                    if profile.name == "minute" and horizon == 60 and mode == "future_log_return" and name == "xpatch":
                        checkpoint_path, reference_path = run_dir / "checkpoint.pt", run_dir / "reference.pt"
                        torch.save({"model_name": name, "lookback": windows.lookback, "horizon": horizon,
                                    "learning_rate": 1e-3, "model": model.state_dict(),
                                    "optimizer": optimizer.state_dict(), "x": x.detach().cpu(),
                                    "anchor": anchor.detach().cpu(), "y": y.detach().cpu(),
                                    "rng_cpu": torch.get_rng_state(),
                                    "rng_cuda": torch.cuda.get_rng_state() if device == "cuda" else None}, checkpoint_path)
                        step(model, optimizer, x, anchor, y)
                        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, reference_path)
                        child = subprocess.run([sys.executable, "-m", "multiscale.smoke", "--device", device,
                                                "--restore-check", str(checkpoint_path), "--reference", str(reference_path)],
                                               capture_output=True, text=True, timeout=50, check=False)
                        if child.returncode != 0:
                            raise ValueError("Independent checkpoint recovery failed: " + child.stderr[-1200:])
                        recovery = json.loads(child.stdout.strip().splitlines()[-1])
                    del model, optimizer, before
    if device == "cuda":
        torch.cuda.synchronize()
    result = {"status": "passed", "started_from_admission": file_digest(directory / "admission.json"),
              "purpose": "small engineering batches only; no forecast scoring or model selection",
              "input_policy": "previously admitted SIP source for minute/hour; new horizon gates provisional; synthetic day/month prices on exchange-calendar timestamps",
              "device": device, "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
              "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
              "elapsed_seconds": time.monotonic() - started, "task_count": 20,
              "baseline_names": ["last_close", "drift", "seasonal", "ridge"],
              "neural_updates": len(records), "records": records,
              "restore": recovery,
              "restore_limit": "one repeated mini-batch; does not verify a real multi-epoch sampler or full deployment package",
              "Q3_values_read": False}
    if len(records) != 80 or recovery is None:
        raise ValueError("Not all planned engineering cases or recovery were exercised")
    (run_dir / "smoke.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"directory": str(run_dir), "status": result["status"], "tasks": 20,
                      "neural_updates": len(records), "device": device,
                      "gpu_name": result["gpu_name"], "peak_gpu_allocated_bytes": result["peak_gpu_allocated_bytes"],
                      "elapsed_seconds": result["elapsed_seconds"], "restore": recovery}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--restore-check", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if args.restore_check:
        if args.reference is None:
            parser.error("--restore-check requires --reference")
        restore_check(args.restore_check, args.reference, args.device)
    elif args.admission:
        run(args.admission, args.device)
    else:
        parser.error("Provide --admission for a smoke check")


if __name__ == "__main__":
    main()
