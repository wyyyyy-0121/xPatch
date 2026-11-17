"""Measure representative multiscale shapes on entirely synthetic prices.

Runs a bounded local CUDA engineering workload, with no market-data reads or
formal scoring. Results do not establish convergence, data rights, cloud speed,
or admission of a training budget.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import socket
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import torch

from .bars import Bars, MINUTE_NS, eligible_origins, file_digest
from .features import CloseWindows
from .spec import DEV_END, NY, OUT, ROOT
from .training import TrainConfig, atomic_json_save, predict_best, run_fit


NEURAL = ("mlp", "gru", "lstm", "xpatch")
# Rounded synthetic sizes approximate the longest candidate shapes without
# embedding any provider's original observations.
SHAPES = (
    ("minute", "minute", 480, 60, 320_000, 56_000, 75_000),
    ("hourly", "hour", 96, 8, 5_700, 1_000, 1_300),
    ("monthly", "day", 252, 21, 1_600, 280, 390),
    ("quarterly", "month", 36, 3, 50, 10, 10),
    ("annual", "month", 60, 12, 24, 4, 6),
)
BENCHMARK_SOURCE_FILES = (
    "multiscale/benchmark_matrix_synthetic.py", "multiscale/training.py", "multiscale/bars.py",
    "multiscale/features.py", "multiscale/models.py", "multiscale/mechanism.py",
    "aapl/models.py", "aapl/vendor/xpatch_network.py",
)


def synthetic_bars(frequency: str):
    """Generate synthetic bar grids before the protected cutoff."""
    if frequency in ("minute", "hour"):
        dates = pd.bdate_range("2019-01-02", periods=1600)
        count = 390 if frequency == "minute" else 6
        spacing = MINUTE_NS if frequency == "minute" else 60 * MINUTE_NS
        duration = spacing
        local_start = (dates + pd.Timedelta(hours=9, minutes=30)).tz_localize(NY)
        beginnings = local_start.tz_convert("UTC").as_unit("ns").asi8
        starts = np.repeat(beginnings, count) + np.tile(np.arange(count, dtype=np.int64), len(dates)) * spacing
        labels = np.repeat(dates.strftime("%Y-%m-%d").to_numpy(dtype="U10"), count)
    elif frequency == "day":
        dates = pd.bdate_range("2014-01-02", periods=3000)
        local_start = dates.tz_localize(NY)
        starts = local_start.tz_convert("UTC").as_unit("ns").asi8
        labels = dates.strftime("%Y-%m-%d").to_numpy(dtype="U10")
        duration = 60 * MINUTE_NS
    elif frequency == "month":
        dates = pd.date_range("2000-01-31", periods=300, freq="ME")
        local_start = dates.tz_localize(NY)
        starts = local_start.tz_convert("UTC").as_unit("ns").asi8
        labels = dates.strftime("%Y-%m-%d").to_numpy(dtype="U10")
        duration = 60 * MINUTE_NS
    else:
        raise ValueError("Unknown synthetic frequency")
    if labels[-1] > DEV_END:
        raise ValueError("Synthetic grid crosses the protected cutoff")
    position = np.arange(len(starts), dtype=np.float64)
    close = 100.0 + 0.00002 * position + 1.2 * np.sin(position / 37.0) + 0.3 * np.cos(position / 5.0)
    return Bars(frequency, starts, starts + duration, close, np.ones(len(starts), dtype=bool), labels)


def partitions(bars: Bars, lookback: int, horizon: int, counts):
    origins = eligible_origins(bars, lookback, horizon, same_session=(bars.frequency == "minute"))
    fit_n, stop_n, validation_n = counts
    if min(counts) < 2 or len(origins) < sum(counts):
        raise ValueError("Synthetic grid cannot supply the requested partitions")
    fit = origins[:fit_n]
    stop_first = np.searchsorted(bars.end_ns[origins], bars.end_ns[fit[-1] + horizon], side="right")
    stop = origins[stop_first:stop_first + stop_n]
    if len(stop) != stop_n:
        raise ValueError("Synthetic stopping partition is incomplete")
    validation_first = np.searchsorted(bars.end_ns[origins], bars.end_ns[stop[-1] + horizon], side="right")
    validation = origins[validation_first:validation_first + validation_n]
    if len(validation) != validation_n:
        raise ValueError("Synthetic validation partition is incomplete")
    return fit, stop, validation


def write_prediction(prediction, origins, destination: Path, *, retain: bool):
    """Measure compression, durable write and hashing; keep only requested samples."""
    target = destination / "synthetic_prediction.npz" if retain else None
    stream = target.open("x+b") if target is not None else tempfile.TemporaryFile(mode="w+b", dir=destination)
    with stream:
        np.savez_compressed(stream, prediction=prediction, origins=origins)
        stream.flush()
        os.fsync(stream.fileno())
        size = os.fstat(stream.fileno()).st_size
        stream.seek(0)
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"synthetic_prediction_sha256": digest.hexdigest(), "prediction_file_bytes": size,
            "prediction_retained": retain, "prediction_path": str(target) if target is not None else None}


def measure(bars, spec, model, destination: Path, *, small=False, retain_prediction=True):
    profile, _, lookback, horizon, fit_n, stop_n, validation_n = spec
    if small:
        fit_n, stop_n, validation_n = 64, 16, 32
    fit, stop, validation = partitions(bars, lookback, horizon, (fit_n, stop_n, validation_n))
    windows = CloseWindows(bars, lookback, ())
    config = TrainConfig(model, lookback, horizon, "future_log_return", max_epochs=1,
                         patience=1, batch_size=128, learning_rate=0.001, seed=2026,
                         checkpoint_batches=100)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    interrupted = model == "xpatch" and profile == "minute" and fit_n > 2 * config.batch_size
    if interrupted:
        stopped = run_fit(windows, fit, stop, config, destination, "synthetic-benchmark-" + profile,
                          device="cuda", stop_after_steps=3)
        if stopped["status"] != "interrupted_for_test":
            raise ValueError("Synthetic recovery interruption was not reached")
        trained = run_fit(windows, fit, stop, config, destination, "synthetic-benchmark-" + profile,
                          device="cuda", resume=True)
    else:
        trained = run_fit(windows, fit, stop, config, destination, "synthetic-benchmark-" + profile,
                          device="cuda")
    torch.cuda.synchronize()
    training_seconds = time.monotonic() - started
    prediction_started = time.monotonic()
    prediction = predict_best(destination / "best.pt", windows, validation, trained["identity"], device="cuda")
    torch.cuda.synchronize()
    prediction_seconds = time.monotonic() - prediction_started
    save_started = time.monotonic()
    saved = write_prediction(prediction, validation, destination, retain=retain_prediction)
    storage_seconds = time.monotonic() - save_started
    return {"profile": profile, "frequency": bars.frequency, "horizon": horizon, "lookback": lookback,
            "model": model, "target": "synthetic_future_log_return", "fit_origins": fit_n,
            "stop_origins": stop_n, "validation_origins": validation_n, "epochs": trained["epochs"],
            "steps": trained["steps"], "batch_size": config.batch_size,
            "training_seconds_including_scaler_stop_and_checkpoints": training_seconds,
            "prediction_seconds": prediction_seconds, "compressed_storage_and_hash_seconds": storage_seconds,
            "total_seconds": training_seconds + prediction_seconds + storage_seconds,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
            "prediction_shape": list(prediction.shape), **saved, "recovery_included": interrupted,
            "output_directory": str(destination)}


def run(duration_seconds: int, *, small: bool):
    if not 1 <= duration_seconds <= 1800:
        raise ValueError("Duration must be between 1 and 1800 seconds")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.set_num_threads(2)
    start = time.monotonic()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = OUT / "synthetic_matrix_benchmark" / stamp
    directory.mkdir(parents=True, exist_ok=False)
    source_hashes = {name: file_digest(ROOT / name) for name in BENCHMARK_SOURCE_FILES}
    free_gpu, total_gpu = torch.cuda.mem_get_info()
    report = {"status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "purpose": "synthetic local engineering measurement only",
              "device": torch.cuda.get_device_name(0), "torch": str(torch.__version__),
              "cuda_runtime": torch.version.cuda, "target_duration_seconds": duration_seconds,
              "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                          "platform": platform.platform(), "hostname": socket.gethostname(),
                          "cpu_affinity": sorted(os.sched_getaffinity(0)), "torch_cpu_threads": torch.get_num_threads(),
                          "gpu_free_bytes_at_start": free_gpu, "gpu_total_bytes": total_gpu,
                          "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")},
              "small_smoke_mode": small, "market_data_read": False, "formal_experiment": False,
              "prediction_storage_policy": "retain the first sample of each shape and model; repeat predictions use temporary files",
              "source_sha256_at_start": source_hashes, "records": [],
              "limitations": ["Synthetic data and one epoch do not establish convergence or real data quality.",
                  "A local laptop measurement does not predict cloud performance.",
                  "Synthetic compression ratios may differ from real predictions; no firm storage peak is implied.",
                  "One return target is timed per shape; budget and data rights remain unapproved.",
                  "GPU reserved-memory peaks can include allocations cached by earlier shapes in this process.",
                  "Repeat prediction files are temporary; their recorded hashes can be compared with the retained representative samples.",
                  "The stop time is checked between jobs, so the last job can extend beyond the target duration."]}
    report_path = directory / "benchmark.json"
    atomic_json_save(report, report_path)
    try:
        bars_by_freq = {frequency: synthetic_bars(frequency) for frequency in ("minute", "hour", "day", "month")}
        agenda = [(shape, model) for shape in SHAPES for model in NEURAL]
        index = 0
        references = {}
        while index < len(agenda) or (not small and time.monotonic() - start < duration_seconds):
            if index >= len(agenda):
                # Once every shape has been attempted, repeat only the costly minute cases.
                shape, model = SHAPES[0], NEURAL[(index - len(agenda)) % len(NEURAL)]
            else:
                shape, model = agenda[index]
            if index and time.monotonic() - start >= duration_seconds:
                break
            destination = directory / f"{index + 1:03d}_{shape[0]}_{model}"
            report["active_job"] = {"profile": shape[0], "model": model, "directory": str(destination)}
            atomic_json_save(report, report_path)
            retain_prediction = index < len(agenda)
            result = measure(bars_by_freq[shape[1]], shape, model, destination, small=small,
                             retain_prediction=retain_prediction)
            key = (shape[0], model)
            if retain_prediction:
                references[key] = result["synthetic_prediction_sha256"]
                result["prediction_matches_retained_reference"] = None
            else:
                result["prediction_matches_retained_reference"] = (
                    result["synthetic_prediction_sha256"] == references[key])
            report["records"].append(result)
            report.pop("active_job", None)
            report["elapsed_seconds"] = time.monotonic() - start
            atomic_json_save(report, report_path)
            index += 1
        seen = {(record["profile"], record["model"]) for record in report["records"]}
        expected = {(shape[0], model) for shape in SHAPES for model in NEURAL}
        end_hashes = {name: file_digest(ROOT / name) for name in BENCHMARK_SOURCE_FILES}
        if end_hashes != source_hashes:
            raise ValueError("Benchmark source code changed during the measurement")
        report.update(status="completed_engineering_only", elapsed_seconds=time.monotonic() - start,
                      all_shapes_measured=seen == expected,
                      shapes_measured=len(seen), shapes_expected=len(expected),
                      source_sha256_at_end=end_hashes,
                      cpu_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic_json_save(report, report_path)
        return report_path, report
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.monotonic() - start)
        atomic_json_save(report, report_path)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-seconds", type=int, default=900)
    parser.add_argument("--small-smoke", action="store_true")
    args = parser.parse_args()
    path, result = run(args.duration_seconds, small=args.small_smoke)
    print(json.dumps({"status": result["status"], "report": str(path), "records": len(result["records"]),
                      "all_shapes_measured": result["all_shapes_measured"],
                      "elapsed_seconds": result["elapsed_seconds"], "market_data_read": False}))


if __name__ == "__main__":
    main()
