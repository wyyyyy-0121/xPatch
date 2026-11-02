"""Independently audit completed artifacts, target alignment and model reloads."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .data import chronological_splits, load_market_data
from .metrics import forecast_metrics
from .models import NEURAL_MODELS
from .run import ROOT, load_checkpoint, predict, read_json, source_hashes, write_json


def verify(output):
    output = Path(output)
    manifest = read_json(output / "manifest.json")
    config = manifest["config"]
    if manifest["sources"] != source_hashes():
        raise AssertionError("Current source differs from experiment source")
    identity = {k: manifest[k] for k in ("config", "data_sha256", "sources")}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    assert manifest["signature"] == signature
    frozen = read_json(output / "frozen.json")
    assert frozen["signature"] == signature
    data_path = Path(config["data"])
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    data = load_market_data(data_path, config["lookback"], config["horizon"])
    assert data.audit["sha256"] == manifest["data_sha256"]
    splits = chronological_splits(data, config["test_fraction"], config["folds"])
    assert read_json(output / "splits.json") == splits
    torch.set_num_threads(config["threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    checks = []
    device = torch.device(config["device"])
    for split in splits:
        origins = data.samples(split["eval_dates"])
        train = data.samples(split["train_dates"])
        assert data.timestamps[train[-1] + data.horizon] < data.timestamps[origins[0] + 1]
        for name in config["models"]:
            for seed in (config["seeds"] if name in NEURAL_MODELS else [0]):
                directory = output / split["name"] / f"{name}_seed{seed}"
                scaler = read_json(directory / "scaler.json")
                assert scaler == data.scale(split["train_dates"])
                assert read_json(directory / "split.json") == split
                with np.load(directory / "predictions.npz") as saved:
                    np.testing.assert_array_equal(saved["origins"], origins)
                    np.testing.assert_array_equal(saved["dates"], data.dates[origins].astype(str))
                    np.testing.assert_array_equal(saved["timestamps"], data.timestamps[origins].astype(str))
                    y = data.returns[origins[:, None] + np.arange(1, data.horizon + 1)]
                    np.testing.assert_array_equal(saved["y"], y)
                    forecast = saved["pred"].copy()
                result = read_json(directory / "result.json")
                if split["name"] == "test":
                    assert result["started_at_utc"] >= frozen["frozen_at_utc"]
                    frozen_hash = hashlib.sha256((output / "frozen.json").read_bytes()).hexdigest()
                    assert result["frozen_protocol_sha256"] == frozen_hash
                assert result["metrics"] == forecast_metrics(y, forecast)
                record = {"run": str(directory.relative_to(output)), "samples": len(origins),
                          "targets_and_metrics": True, "train_only_scaler": True,
                          "checkpoint_reproduced": True, "cpu_gpu_checked": False}
                if name in NEURAL_MODELS:
                    model, state = load_checkpoint(directory / "checkpoint.pt", device)
                    assert state["config"] == config and state["scaler"] == scaler
                    assert state["seed"] == seed and state["model"] == name
                    if split["name"] == "test":
                        assert state["epoch"] == frozen["final_epochs"][name]
                    x, _ = data.arrays(origins, scaler["std"])
                    reproduced = predict(model, torch.from_numpy(x).to(device), config["batch_size"])
                    reproduced = reproduced.cpu().numpy()[..., 0].astype(np.float64) * scaler["std"]
                    np.testing.assert_array_equal(reproduced, forecast)
                    assert sum(p.numel() for p in model.parameters()) == result["parameters"]
                    if torch.cuda.is_available():
                        cpu_model, _ = load_checkpoint(directory / "checkpoint.pt", "cpu")
                        gpu_model, _ = load_checkpoint(directory / "checkpoint.pt", "cuda")
                        sample = torch.from_numpy(x[:64])
                        with torch.no_grad():
                            cpu_pred = cpu_model(sample)
                            gpu_pred = gpu_model(sample.cuda()).cpu()
                        torch.testing.assert_close(cpu_pred, gpu_pred, rtol=2e-4, atol=2e-5)
                        record["cpu_gpu_checked"] = True
                        record["max_cpu_gpu_abs_difference_scaled"] = float((cpu_pred - gpu_pred).abs().max())
                        del cpu_model, gpu_model
                    del model
                elif name == "ridge":
                    x, _ = data.arrays(origins, scaler["std"])
                    with np.load(directory / "checkpoint.npz") as state:
                        reproduced = (x[..., 0].astype(np.float64) @ state["coef"].T + state["intercept"]) * scaler["std"]
                    np.testing.assert_allclose(reproduced, forecast, rtol=1e-12, atol=1e-15)
                else:
                    np.testing.assert_array_equal(forecast, np.zeros_like(y))
                checks.append(record)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    # Report generation independently recalculates all metrics and aligned comparisons.
    from .report import generate_report
    summary = generate_report(output)
    acceptance = {"passed": True, "signature": signature, "runs_checked": len(checks),
                  "continuous_history_and_future": True, "target": "close-derived minute log returns",
                  "split_labels_disjoint": True, "train_only_scaling": True,
                  "frozen_before_test_protocol": frozen["frozen_at_utc"],
                  "all_saved_forecasts_reproduced": True,
                  "cpu_gpu_checks_available": torch.cuda.is_available(), "runs": checks,
                  "research_claim_supported": summary["long_short_effectiveness_supported"]}
    write_json(output / "acceptance.json", acceptance)
    print(f"Independent acceptance passed: {len(checks)} runs", flush=True)
    return acceptance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/finance_ls")
    verify(parser.parse_args().output)
