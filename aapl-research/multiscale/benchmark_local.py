"""Small, bounded local SIP training-shape and recovery benchmark.

This uses only inner development examples and is not a model comparison. It does
not open sealed data, contact the cloud, choose hyperparameters, or train a matrix.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
import torch

from .bars import file_digest
from .features import CloseWindows
from .smoke import compare_nested, read_bars
from .spec import OUT
from .splits import rolling_folds
from .training import TrainConfig, run_fit


def inputs(admission: Path, fit_count: int):
    directory = admission.resolve()
    if not directory.is_relative_to((OUT / "admission").resolve()):
        raise ValueError("Admission path is outside the multiscale workspace")
    with (directory / "admission.json").open(encoding="utf-8") as stream:
        receipt = json.load(stream)
    if receipt.get("Q3_values_read") is not False or receipt.get("experiment_scored") is not False:
        raise ValueError("Unexpected source receipt")
    bars = read_bars(directory, receipt, "minute")
    name = "origins_minute_h60.npy"
    if file_digest(directory / name) != receipt["files"][name]:
        raise ValueError("Minute origin identity changed")
    origins = np.load(directory / name, allow_pickle=False)
    folds = rolling_folds(bars, origins, 60)
    if len(folds) != 5 or folds[0]["status"] != "ok":
        raise ValueError("Minute inner development fold is unavailable")
    inner = folds[0]["inner"]
    fit, stop = inner["fit"][:fit_count], inner["stop"][:128]
    if len(fit) != fit_count or len(stop) != 128:
        raise ValueError("Not enough inner examples for the bounded benchmark")
    windows = CloseWindows(bars, 480, receipt.get("split_actions") or
                           receipt.get("split_actions_agree_between_provider_responses") or [])
    ident = hashlib.sha256((file_digest(directory / "admission.json") + receipt["files"]["bars_minute.npz"] +
                            receipt["files"][name]).encode()).hexdigest()
    return windows, fit, stop, ident


def config(name):
    return TrainConfig(name, 480, 60, "future_log_return", max_epochs=1, patience=1,
                       batch_size=128, learning_rate=1e-3, seed=2026, checkpoint_batches=16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--resume-check", type=Path)
    parser.add_argument("--fit-count", type=int, default=32768)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise ValueError("Local CUDA is unavailable")
    if not 4096 <= args.fit_count <= 65536 or args.fit_count % 128:
        raise ValueError("fit-count must be a multiple of 128 between 4096 and 65536")
    torch.set_num_threads(2)
    windows, fit, stop, data_id = inputs(args.admission, args.fit_count)
    if args.resume_check:
        restored = run_fit(windows, fit, stop, config("xpatch"), args.resume_check, data_id,
                           device="cuda", resume=True)
        print(json.dumps({"status": restored["status"], "identity": restored["identity"],
                          "steps": restored["steps"], "epochs": restored["epochs"]}))
        return
    run_dir = OUT / "local_benchmark" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for name in ("mlp", "gru", "lstm", "xpatch"):
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        trained = run_fit(windows, fit, stop, config(name), run_dir / name, data_id, device="cuda")
        torch.cuda.synchronize()
        seconds = time.monotonic() - start
        model_files = list((run_dir / name).glob("*.pt"))
        records.append({"model": name, "lookback": 480, "horizon": 60, "batch": 128,
                        "training_origins": len(fit), "stopping_origins": len(stop),
                        "steps": trained["steps"], "epochs": trained["epochs"],
                        "wall_seconds_including_scaler_stop_and_checkpoints": seconds,
                        "steps_per_second_end_to_end": trained["steps"] / seconds,
                        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
                        "checkpoint_file_bytes": sum(p.stat().st_size for p in model_files)})
    partial_dir = run_dir / "xpatch_interrupted"
    partial = run_fit(windows, fit, stop, config("xpatch"), partial_dir, data_id,
                      device="cuda", stop_after_steps=3)
    if partial["status"] != "interrupted_for_test":
        raise ValueError("Training did not stop at the intended recovery boundary")
    child = subprocess.run([sys.executable, "-m", "multiscale.benchmark_local", "--admission",
                            str(args.admission.resolve()), "--fit-count", str(args.fit_count),
                            "--resume-check", str(partial_dir)],
                           capture_output=True, text=True, timeout=50, check=False)
    if child.returncode:
        raise ValueError("Independent multi-epoch recovery process failed: " + child.stderr[-1200:])
    restored = json.loads(child.stdout.strip().splitlines()[-1])
    reference = torch.load(run_dir / "xpatch/latest.pt", map_location="cpu", weights_only=True)
    recovered = torch.load(partial_dir / "latest.pt", map_location="cpu", weights_only=True)
    same_model = compare_nested(reference["model"], recovered["model"])
    same_optimizer = compare_nested(reference["optimizer"], recovered["optimizer"])
    same_history = reference["state"]["history"] == recovered["state"]["history"]
    if not (same_model and same_optimizer and same_history):
        raise ValueError("Interrupted and uninterrupted benchmark fits differ")
    result = {"status": "passed", "purpose": "bounded engineering throughput only; not a scored model comparison",
              "source": "previously admitted SIP minute source; new H60 origins are provisional",
              "device": torch.cuda.get_device_name(0), "records": records,
              "cpu_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "recovery": {"status": "passed", "interruption_after_steps": 3,
                           "resumed_process": restored, "model_equal": same_model,
                           "optimizer_equal": same_optimizer, "inner_history_equal": same_history},
              "limitations": ["One epoch on a bounded inner-fit subset is not adequate model training.",
                              "Throughput on this laptop does not predict a cloud 5090's throughput.",
                              "The first bounded fit origins and 128 inner-stop origins are engineering samples, not a final selection set.",
                              "Independent final confirmation, publication data rights and full protocol remain pending."],
              "Q3_values_read": False}
    (run_dir / "benchmark.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"directory": str(run_dir), "status": result["status"], "device": result["device"],
                      "records": records, "recovery": result["recovery"]}))


if __name__ == "__main__":
    main()
