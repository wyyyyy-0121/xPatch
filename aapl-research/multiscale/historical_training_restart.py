"""Bounded fresh-process restart check on immutable real-training checkpoints.

Two updates from the same archived state are compared with one update, process
exit, and one resumed update. Only copies are changed; no validation is scored.
This is an engineering check, not an additional forecasting experiment.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .historical_coverage import OUT, ROOT, digest, in_tree, read_json, require


NIGHT = OUT / "private_low_load/20261107T153438623158Z"
ADMISSION = OUT / "admission/alpaca_20261107T052228573242Z"
CASES = ((787612, "gru", 4), (787613, "xpatch", 5))


def write_new(path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def prepare_case(pid, model, fold_number):
    import numpy as np
    import torch
    from .development import bars_identity, validate_split
    from .features import CloseWindows
    from .matrix_executor import read_bars
    from .splits import rolling_folds
    from .training import TrainConfig, digest_indices

    require((pid, model, fold_number) in CASES, "Unregistered recovery case")
    require(1 <= len(os.sched_getaffinity(0)) <= 2 and
            os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "Use at most two CPUs with GPU hidden")
    torch.set_num_threads(min(2, len(os.sched_getaffinity(0))))
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    snapshot = read_json(NIGHT / "CHECKPOINT_AND_BUDGET_SNAPSHOT.json")
    worker = next(row for row in snapshot["workers"] if row["pid"] == pid)
    folder = NIGHT / f"checkpoint_snapshot/worker_{pid}"
    checkpoint, state_path = folder / model / "latest.pt", folder / "fold_state.json"
    for path in (checkpoint, state_path):
        record = next(row for row in worker["checkpoint_files"] if Path(row["copy"]).resolve() == path.resolve())
        require(digest(path) == record["sha256"], "Immutable paused snapshot changed")
    state = read_json(state_path)
    require(state["Q3_values_read"] is False and state["independent_test"] is False,
            "Archived fold is not historical development")
    for name, expected in state["source_sha256"].items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT) and digest(path) == expected, "Original training source changed")
    admission_path = ADMISSION / "admission.json"
    admission = read_json(admission_path)
    require(admission["Q3_values_read"] is False and
            state["data_identity"].split(":", 1)[1] == digest(admission_path), "Admission differs")
    bars = read_bars(ADMISSION, admission, "minute")
    require(bars_identity(bars) == state["identity_fields"]["bars_sha256"], "Original input bars differ")
    origin_path = ADMISSION / "origins_minute_h60.npy"
    require(digest(origin_path) == admission["files"][origin_path.name], "Original origins changed")
    origins = np.load(origin_path, allow_pickle=False)
    split = rolling_folds(bars, origins, 60)[fold_number - 1]
    _, fit, stop, _ = validate_split(bars, split, 60)
    require(digest_indices(fit) == state["identity_fields"]["fit_indices_sha256"] and
            digest_indices(stop) == state["identity_fields"]["stop_indices_sha256"], "Original inner split differs")
    cfg = state["config"]
    config = TrainConfig(model=model, **{field.name: cfg[field.name] for field in fields(TrainConfig)
                                       if field.name in cfg})
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(saved["state"]["stale_epochs"] < config.patience and
            saved["state"]["epoch"] < config.max_epochs and saved["state"]["permutation"] is not None and
            saved["state"]["cursor"] + 2 * config.batch_size + 1 < len(fit),
            "Recovery check must stay inside an unfinished epoch for two updates")
    windows = CloseWindows(bars, config.lookback, state["identity_fields"]["actions"])
    training_data_identity = (state["data_identity"] + ":" + cfg["protocol_identity"] + ":" +
                              state["fold_identity"])
    from .training import identity as training_identity
    require(training_identity(config, training_data_identity, fit, stop, "cpu") == saved["identity"],
            "Reconstructed training identity differs from the archived checkpoint")
    return windows, fit, stop, config, training_data_identity, saved, checkpoint, state_path


def worker(directory, pid, model, fold, mode):
    from .training import run_fit

    directory = in_tree(directory, OUT / "historical_training_recovery")
    windows, fit, stop, config, identity, saved, _, _ = prepare_case(pid, model, fold)
    steps = 2 if mode == "reference" else 1
    result = run_fit(windows, fit, stop, config, directory, identity,
                     device="cpu", resume=True, stop_after_steps=steps)
    expected = saved["state"]["steps"] + (2 if mode == "resume" else steps)
    require(result["status"] == "interrupted_for_test" and result["steps"] == expected,
            "Bounded update count or restart status differs")
    if mode == "interrupt":
        os._exit(73)


def check(output):
    import torch
    from .gpu_recovery_check import same_value

    output = in_tree(output, OUT / "historical_training_recovery")
    output.mkdir(parents=True, exist_ok=False)
    require(1 <= len(os.sched_getaffinity(0)) <= 2 and
            os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "Use at most two CPUs with GPU hidden")
    started = time.monotonic()
    checks = []
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2",
                       OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    for pid, model, fold in CASES:
        snapshot = read_json(NIGHT / "CHECKPOINT_AND_BUDGET_SNAPSHOT.json")
        saved_worker = next(row for row in snapshot["workers"] if row["pid"] == pid)
        checkpoint = NIGHT / f"checkpoint_snapshot/worker_{pid}/{model}/latest.pt"
        expected = next(row["sha256"] for row in saved_worker["checkpoint_files"]
                        if Path(row["copy"]).resolve() == checkpoint.resolve())
        require(digest(checkpoint) == expected, "Source checkpoint changed before copy")
        case = output / f"{model}_fold{fold}"
        case.mkdir()
        for name in ("reference", "restarted"):
            (case / name).mkdir()
            shutil.copyfile(checkpoint, case / name / "latest.pt")
        initial = torch.load(checkpoint, map_location="cpu", weights_only=True)
        runs = []
        for mode, directory, code in (("reference", "reference", 0),
                                       ("interrupt", "restarted", 73), ("resume", "restarted", 0)):
            require(time.monotonic() - started < 180, "Engineering audit exceeded its 180-second admission")
            command = [sys.executable, "-m", "multiscale.historical_training_restart", "--worker", mode,
                       "--output", str(case / directory), "--pid", str(pid), "--model", model, "--fold", str(fold)]
            child = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True,
                                   text=True, timeout=min(60, max(1, 180 - (time.monotonic() - started))))
            runs.append({"mode": mode, "exit_code": child.returncode,
                         "stdout": child.stdout[-2000:], "stderr": child.stderr[-3000:]})
            write_new(case / f"{mode}_process.json", runs[-1])
            require(child.returncode == code, f"Fresh process failed for {model} / {mode}: {child.stderr[-2000:]}")
        reference = torch.load(case / "reference/latest.pt", map_location="cpu", weights_only=True)
        restarted = torch.load(case / "restarted/latest.pt", map_location="cpu", weights_only=True)
        require(same_value(reference, restarted), "Restarted optimizer, parameters, cursor or RNG differ")
        require(reference["state"]["steps"] == initial["state"]["steps"] + 2,
                "Comparison did not advance exactly two optimizer updates")
        require(digest(checkpoint) == expected, "Engineering audit changed the source snapshot")
        checks.append({"model": model, "fold": fold, "source_checkpoint": str(checkpoint),
                       "source_checkpoint_sha256": expected, "training_identity": initial["identity"],
                       "initial_steps": initial["state"]["steps"], "final_steps": reference["state"]["steps"],
                       "initial_epoch": initial["state"]["epoch"], "initial_cursor": initial["state"]["cursor"],
                       "all_checkpoint_fields_exactly_equal": True, "source_unchanged": True, "processes": runs})
        print(json.dumps({"stage": "real_checkpoint_restart_verified", "model": model, "fold": fold}), flush=True)
    result = {"status": "bounded_real_training_fresh_process_restart_verified",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "snapshot_sha256": digest(NIGHT / "CHECKPOINT_AND_BUDGET_SNAPSHOT.json"),
              "script_sha256": digest(Path(__file__)), "checks": checks,
              "fresh_processes": 3 * len(checks), "total_optimizer_updates_in_audit": 4 * len(checks),
              "logical_cpus": sorted(os.sched_getaffinity(0)), "device": "cpu",
              "validation_scoring_performed": False, "Q3_values_read": False,
              "original_workers_or_queues_modified": False, "comparison_results_modified": False,
              "maximum_audit_seconds": 180, "elapsed_seconds": time.monotonic() - started,
              "limitations": ["Two updates per compared trajectory from each of two real archived mid-epoch states.",
                              "This checks CPU process restart, not a complete resumed real-data fold or GPU restart.",
                              "It creates no additional forecasting result and changes no experiment deadline."]}
    write_new(output / "RESTART_AUDIT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker", choices=("reference", "interrupt", "resume"), help=argparse.SUPPRESS)
    parser.add_argument("--pid", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument("--fold", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.output, args.pid, args.model, args.fold, args.worker)
    else:
        result = check(args.output)
        print(json.dumps({key: result[key] for key in ("status", "fresh_processes", "elapsed_seconds")} |
                         {"audit": str((args.output / "RESTART_AUDIT.json").resolve())}))


if __name__ == "__main__":
    main()
