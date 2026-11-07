"""Check current fold recovery on CUDA using synthetic bars in separate processes.

No market data are opened and the resulting scores are only test artifacts. This
bounded check is not a benchmark, data admission, or a formal experiment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from .bars import Bars, MINUTE_NS, eligible_origins, file_digest
from .development import FoldConfig
from .spec import MODELS, OUT
from .splits import rolling_folds
from .verify_development import check_fold


def synthetic_fold():
    """Use deterministic artificial prices on a small, pre-cutoff session grid."""
    dates = pd.bdate_range("2024-01-02", periods=18).strftime("%Y-%m-%d").tolist()
    stamps, labels = [], []
    for date in dates:
        start = pd.Timestamp(date + " 09:30", tz="America/New_York")
        grid = pd.date_range(start, periods=30, freq="min").tz_convert("UTC").as_unit("ns").asi8
        stamps.extend(grid)
        labels.extend([date] * len(grid))
    starts = np.asarray(stamps, dtype=np.int64)
    position = np.arange(len(starts), dtype=np.float64)
    close = 100.0 + 0.01 * position + np.sin(position / 11.0)
    bars = Bars("minute", starts, starts + MINUTE_NS, close,
                np.ones(len(starts), dtype=bool), np.asarray(labels))
    origins = eligible_origins(bars, 6, 3, same_session=True)
    folds = rolling_folds(bars, origins, 3)
    if len(folds) != 5:
        raise ValueError("Synthetic fold could not be constructed")
    cfg = FoldConfig("minute", 3, "future_log_return", 6, 6, (0.1, 1.0),
                     2, 2, 8, 1e-3, 2026, "synthetic-gpu-recovery-v1")
    return bars, folds[0], cfg


def worker(directory: Path, mode: str):
    """Deliberately exit during xPatch, then resume or run an intact reference."""
    import multiscale.development as development

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.set_num_threads(2)
    bars, fold, cfg = synthetic_fold()
    if mode == "interrupt":
        original = development.run_fit

        def stop_xpatch(windows, fit, stop, config, model_dir, identity, **kwargs):
            if config.model == "xpatch":
                result = original(windows, fit, stop, config, model_dir, identity,
                                  stop_after_steps=3, **kwargs)
                if result.get("status") != "interrupted_for_test":
                    raise RuntimeError("Synthetic interruption was not reached")
                os._exit(73)
            return original(windows, fit, stop, config, model_dir, identity, **kwargs)

        development.run_fit = stop_xpatch
    development.run_development_fold(bars, fold, [], cfg, directory, "synthetic-gpu-data-v1",
                                     device="cuda", resume=(mode == "resume"))


def same_value(left, right):
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(same_value(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(same_value(a, b) for a, b in zip(left, right))
    return left == right


def check(output: Path):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    output = output.resolve()
    allowed = (OUT / "gpu_recovery").resolve()
    if not output.is_relative_to(allowed) or output == allowed:
        raise ValueError("Output must be a new GPU recovery directory")
    output.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    runs = []
    for mode, folder, expected in (("interrupt", "interrupted", 73),
                                   ("resume", "interrupted", 0),
                                   ("fresh", "reference", 0)):
        cmd = [sys.executable, "-m", "multiscale.gpu_recovery_check", "--worker", mode,
               "--directory", str(output / folder)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=environment)
        runs.append({"mode": mode, "returncode": result.returncode,
                     "stdout": result.stdout[-4000:], "stderr": result.stderr[-4000:]})
        if result.returncode != expected:
            (output / "failure.json").write_text(json.dumps({"runs": runs}, indent=2), encoding="utf8")
            raise RuntimeError(f"Synthetic GPU {mode} process failed; see failure.json")
    left_dir, right_dir = output / "interrupted", output / "reference"
    left_receipt = json.loads((left_dir / "fold_receipt.json").read_text(encoding="utf8"))
    right_receipt = json.loads((right_dir / "fold_receipt.json").read_text(encoding="utf8"))
    if left_receipt["resume_count"] != 1 or left_receipt["fold_identity"] != right_receipt["fold_identity"]:
        raise ValueError("GPU resumed fold identity or resume count differs")
    left_verification, right_verification = check_fold(left_dir), check_fold(right_dir)
    for name in MODELS:
        with np.load(left_dir / f"prediction_{name}.npz", allow_pickle=False) as left:
            with np.load(right_dir / f"prediction_{name}.npz", allow_pickle=False) as right:
                if set(left.files) != set(right.files) or not all(
                        np.array_equal(left[key], right[key]) for key in left.files):
                    raise ValueError(f"Resumed GPU prediction differs: {name}")
        if left_receipt["models"][name]["metrics"] != right_receipt["models"][name]["metrics"]:
            raise ValueError(f"Resumed GPU metrics differ: {name}")
    for name in ("mlp", "gru", "lstm", "xpatch"):
        left = torch.load(left_dir / name / "latest.pt", map_location="cpu", weights_only=True)
        right = torch.load(right_dir / name / "latest.pt", map_location="cpu", weights_only=True)
        if not same_value(left, right):
            raise ValueError(f"Resumed GPU checkpoint state differs: {name}")
    receipt = {"status": "passed", "scope": "synthetic CUDA fold interruption and separate-process recovery",
               "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__),
               "cuda_runtime": torch.version.cuda, "models": list(MODELS),
               "interrupted_model": "xpatch", "interruption_after_updates": 3,
               "fold_resume_count": left_receipt["resume_count"],
               "checks": ["eight prediction arrays and metrics identical to uninterrupted reference",
                          "all four neural latest checkpoints including optimizer and RNG identical",
                          "independent saved-fold verification passed for both processes"],
               "verification_status": [left_verification["status"], right_verification["status"]],
               "fold_receipt_sha256": {"resumed": file_digest(left_dir / "fold_receipt.json"),
                                       "reference": file_digest(right_dir / "fold_receipt.json")},
               "market_data_read": False, "formal_experiment": False,
               "limitations": ["Small synthetic input only; not a real-data admission or training benchmark.",
                               "Exact recovery verified only on the recorded local hardware and software."],
               "processes": runs}
    target = output / "gpu_recovery_receipt.json"
    target.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf8")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("interrupt", "resume", "fresh"), help=argparse.SUPPRESS)
    parser.add_argument("--directory", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.directory is None or not args.directory.resolve().is_relative_to((OUT / "gpu_recovery").resolve()):
            raise ValueError("Invalid worker directory")
        worker(args.directory, args.worker)
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = check(OUT / "gpu_recovery" / stamp)
    print(json.dumps({"status": "passed", "receipt": str(target), "market_data_read": False}))


if __name__ == "__main__":
    main()
