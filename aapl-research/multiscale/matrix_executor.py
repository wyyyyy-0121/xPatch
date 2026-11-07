"""Freeze and execute a reviewed, bounded historical-development matrix.

This program has no override for missing rights, source, exposure, or compute
evidence. The current candidate does not pass those checks. Review attestations
are not an automated legal decision, and no cloud resource is provisioned here.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import date, datetime, timezone
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import socket
import time

import numpy as np
import pandas as pd

from .bars import Bars, eligible_origins, file_digest
from .development import FoldConfig, atomic_json, run_development_fold, validate_split
from .matrix_gate import (REQUIRED_SOURCE_FILES, _reviewer, locked_protocol_issues,
                          profile_selection, review_issues, task_inventory)
from .spec import DEV_END, MODELS, NY, OUT, PROFILES, ROOT, TARGETS
from .splits import rolling_folds, warmup_folds
from .training import digest_indices
from .verify_development import check_fold


PERIODS = {"minute": 390, "hour": 6, "day": 252, "month": 12}


def _json_inside(path: Path, root: Path):
    path = path.resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Required JSON is outside its allowed workspace")
    return json.loads(path.read_text(encoding="utf8")), file_digest(path), path


def load_reviewed(admission_dir: Path, review_path: Path, *, selected_profiles=None):
    admission, admission_hash, admission_path = _json_inside(
        admission_dir.resolve() / "admission.json", OUT / "admission")
    review, review_hash, review_path = _json_inside(review_path, OUT)
    note = OUT / "POST_CUTOFF_EXPOSURE.md"
    note_hash = file_digest(note) if note.is_file() else None
    issues = review_issues(admission, admission_hash, review, OUT, note_hash, selected_profiles=selected_profiles)
    if issues:
        raise ValueError("Matrix admission blocked: " + ", ".join(issues))
    return admission, admission_hash, admission_path.parent, review, review_hash, review_path


def read_bars(directory: Path, admission: dict, frequency: str):
    """Check saved metadata and hashes before opening any admitted prices."""
    name = f"bars_{frequency}.npz"
    path = (directory / name).resolve()
    expected = admission.get("files", {}).get(name)
    if not path.is_relative_to(directory.resolve()) or not expected or file_digest(path) != expected:
        raise ValueError("Candidate bar artifact identity changed")
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"start_ns", "end_ns", "close", "observed", "session_date"}:
            raise ValueError("Unexpected candidate bar fields")
        dates, start, end, observed = (data[key] for key in ("session_date", "start_ns", "end_ns", "observed"))
        if dates.dtype.kind != "U" or not len(dates) or (dates > DEV_END).any():
            raise ValueError("Candidate bar dates cross the protected cutoff")
        cutoff = pd.Timestamp("2026-07-01", tz=NY).tz_convert("UTC").as_unit("ns").value
        if observed.dtype != np.bool_ or len(observed) != len(end) or (end[observed] >= cutoff).any():
            raise ValueError("An observed candidate bar completes after the cutoff")
        close = data["close"]
    return Bars(frequency, start, end, close, observed, dates)


def _check_history(bars: Bars, profile):
    observed_dates = bars.session_date[bars.observed]
    if not len(observed_dates):
        raise ValueError("The candidate has no observed bars")
    span = (date.fromisoformat(str(observed_dates[-1])) - date.fromisoformat(str(observed_dates[0]))).days / 365.2425
    if span < profile.minimum_history_years:
        raise ValueError("Observed source history does not meet the task minimum")


def materialize(directory: Path, admission: dict, job: dict):
    profile = next((p for p in PROFILES if p.name == job["profile"]), None)
    if profile is None or job["frequency"] != profile.frequency or job["horizon"] not in profile.horizons:
        raise ValueError("Job is outside the requested task scope")
    bars = read_bars(directory, admission, profile.frequency)
    _check_history(bars, profile)
    name = f"origins_{profile.name}_h{job['horizon']}.npy"
    path = (directory / name).resolve()
    if (not path.is_relative_to(directory.resolve()) or
            file_digest(path) != admission.get("files", {}).get(name)):
        raise ValueError("Candidate origin artifact identity changed")
    origins = np.load(path, allow_pickle=False)
    expected = eligible_origins(bars, max(profile.lookbacks), job["horizon"],
                                same_session=(profile.frequency == "minute"))
    if not np.array_equal(origins, expected):
        raise ValueError("Saved origins differ from the complete valid origin set")
    cutoff = pd.Timestamp("2026-07-01", tz=NY).tz_convert("UTC").as_unit("ns").value
    if (bars.end_ns[origins + job["horizon"]] >= cutoff).any():
        raise ValueError("A candidate target crosses the protected cutoff")
    strategy = job["fold_strategy"]
    if strategy == "rolling":
        folds = rolling_folds(bars, origins, job["horizon"])
    elif strategy == "warmup":
        rule = job["warmup"]
        folds = warmup_folds(bars, origins, job["horizon"], min_inner_fit=rule["min_inner_fit"],
                             min_inner_stop=rule["min_inner_stop"],
                             min_validation_groups=rule["min_validation_groups"])
    else:
        raise ValueError("Unknown frozen fold strategy")
    if len(folds) != 5:
        raise ValueError("The frozen strategy cannot form five folds")
    fold = folds[job["fold"] - 1]
    train, fit, stop, validation = validate_split(bars, fold, job["horizon"])
    expected_hashes = {"train_indices_sha256": digest_indices(train), "fit_indices_sha256": digest_indices(fit),
                       "stop_indices_sha256": digest_indices(stop),
                       "validation_indices_sha256": digest_indices(validation)}
    if any(job[key] != value for key, value in expected_hashes.items()):
        raise ValueError("Actual chronological splits differ from the frozen identities")
    if [len(fit), len(stop), len(validation)] != [job["fit_origins"], job["stop_origins"], job["validation_origins"]]:
        raise ValueError("Actual split sizes differ from the frozen protocol")
    return bars, fold


def _training_config(job, training, protocol_hash):
    return FoldConfig(job["profile"], job["horizon"], job["target"], job["lookback"], job["seasonal_period"],
                      tuple(training["ridge_alphas"]), training["max_epochs"], training["patience"],
                      training["batch_size"], training["learning_rate"], training["seed"], protocol_hash)


def freeze(admission_dir: Path, review_path: Path, recipe_path: Path):
    """Bind predeclared complete scales and five folds before any scores."""
    recipe, recipe_hash, _ = _json_inside(recipe_path, OUT)
    selected = profile_selection(recipe.get("selected_profiles"))
    admission, admission_hash, source, review, review_hash, _ = load_reviewed(
        admission_dir, review_path, selected_profiles=selected)
    kind = admission.get("data_kind", "market")
    if kind not in ("market", "synthetic_test"):
        raise ValueError("Unknown data kind")
    required_recipe_status = "reviewed_synthetic_test" if kind == "synthetic_test" else "reviewed_and_approved"
    if (recipe.get("status") != required_recipe_status or recipe.get("data_kind", "market") != kind or
            recipe.get("admission_sha256") != admission_hash or not _reviewer(recipe)):
        raise ValueError("Recipe has not been reviewed and bound to the candidate data")
    strategies = recipe.get("fold_strategies")
    if not isinstance(strategies, dict) or set(strategies) != {p.name for p in PROFILES}:
        raise ValueError("Recipe must specify one fold strategy for each profile")
    omitted = recipe.get("omitted_profiles")
    expected_omitted = {p.name for p in PROFILES if p.name not in selected}
    if (not isinstance(omitted, dict) or set(omitted) != expected_omitted or
            any(not isinstance(note, str) or len(note.strip()) < 20 for note in omitted.values())):
        raise ValueError("Every omitted scale needs a predeclared reason")
    training = recipe.get("training")
    if not isinstance(training, dict):
        raise ValueError("A complete fixed training configuration is required")
    warmup = recipe.get("warmup", {})
    if set(warmup) != {"min_inner_fit", "min_inner_stop", "min_validation_groups"} or any(
            type(value) is not int or value < 1 for value in warmup.values()):
        raise ValueError("Invalid predeclared warm-up settings")
    rows = {(row["profile"], row["horizon"], row["target"]): row for row in admission["tasks"]}
    jobs = []
    for profile in PROFILES:
        if profile.name not in selected:
            continue
        strategy = strategies[profile.name]
        if strategy not in ("rolling", "warmup"):
            raise ValueError("Unknown fold strategy")
        bars = read_bars(source, admission, profile.frequency)
        _check_history(bars, profile)
        for horizon in profile.horizons:
            name = f"origins_{profile.name}_h{horizon}.npy"
            path = (source / name).resolve()
            if (not path.is_relative_to(source.resolve()) or
                    file_digest(path) != admission.get("files", {}).get(name)):
                raise ValueError("Candidate origin identity changed")
            origins = np.load(path, allow_pickle=False)
            expected = eligible_origins(bars, max(profile.lookbacks), horizon,
                                        same_session=(profile.frequency == "minute"))
            if not np.array_equal(origins, expected):
                raise ValueError("Saved origins do not match the valid origins")
            folds = (rolling_folds(bars, origins, horizon) if strategy == "rolling" else
                     warmup_folds(bars, origins, horizon, **warmup))
            if len(folds) != 5:
                raise ValueError("Recipe cannot produce all five chronological folds")
            for target in TARGETS:
                if rows[(profile.name, horizon, target)].get("minimum_history_met") is not True:
                    raise ValueError("A task does not meet the original minimum history")
                for fold in folds:
                    train, fit, stop, validation = validate_split(bars, fold, horizon)
                    job = {"profile": profile.name, "frequency": profile.frequency, "horizon": horizon,
                           "target": target, "fold": fold["fold"], "lookback": max(profile.lookbacks),
                           "origin_lookback": max(profile.lookbacks), "seasonal_period": PERIODS[profile.frequency],
                           "fold_strategy": strategy, "warmup": warmup,
                           "fit_origins": len(fit), "stop_origins": len(stop), "validation_origins": len(validation),
                           "train_indices_sha256": digest_indices(train), "fit_indices_sha256": digest_indices(fit),
                           "stop_indices_sha256": digest_indices(stop),
                           "validation_indices_sha256": digest_indices(validation)}
                    _training_config(job, training, "pre-freeze-shape-check").validate()
                    jobs.append(job)
    protocol = {"schema_version": 1,
                "status": "frozen_for_synthetic_test" if kind == "synthetic_test" else "frozen_for_historical_development",
                "data_kind": kind, "frozen": True,
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "admission_sha256": admission_hash,
                "review_sha256": review_hash, "recipe_sha256": recipe_hash,
                "development_cutoff_new_york": DEV_END, "Q3_values_read": False, "independent_test": False,
                "matrix_scope": "full" if len(selected) == len(PROFILES) else "partial",
                "selected_profiles": selected, "omitted_profiles": omitted,
                "task_inventory": task_inventory(selected, omitted),
                "models": list(MODELS), "training": training, "analysis": recipe.get("analysis"), "jobs": jobs,
                "source_code_sha256": {name: file_digest(ROOT / name) for name in REQUIRED_SOURCE_FILES}}
    issues = locked_protocol_issues(admission, admission_hash, protocol, review_hash)
    if issues:
        raise ValueError("Cannot freeze matrix: " + ", ".join(issues))
    directory = (OUT / "synthetic_matrix_validation" / "locked_protocols" if kind == "synthetic_test"
                 else OUT / "locked_protocols")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    with target.open("x", encoding="utf8") as stream:
        json.dump(protocol, stream, indent=2, allow_nan=False)
    return target


def _parent_death_guard(parent_pid):
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0 or os.getppid() != parent_pid:
        raise RuntimeError("Cannot establish the parent-death guard")


def _worker(admission_dir, review_path, protocol_path, job, folder, resume, parent_pid, queue):
    try:
        _parent_death_guard(parent_pid)
        protocol, protocol_hash, _ = _json_inside(Path(protocol_path), OUT)
        selected = profile_selection(protocol.get("selected_profiles"))
        admission, admission_hash, source, review, review_hash, _ = load_reviewed(
            Path(admission_dir), Path(review_path), selected_profiles=selected)
        issues = locked_protocol_issues(admission, admission_hash, protocol, review_hash)
        if issues or job not in protocol["jobs"]:
            raise ValueError("Worker protocol is no longer valid")
        bars, fold = materialize(source, admission, job)
        config = _training_config(job, protocol["training"], protocol_hash)
        actions = admission.get("split_actions")
        if not isinstance(actions, list):
            raise ValueError("Reviewed split actions are missing")
        data_identity = protocol["data_kind"] + ":" + admission_hash
        run_development_fold(bars, fold, actions, config, Path(folder), data_identity,
                             device=review["compute"]["device"], resume=resume)
        verification = check_fold(Path(folder))
        if not all(verification["current_source_matches_recorded"].values()):
            raise ValueError("Saved fold source identities no longer match")
        queue.put({"status": "completed", "receipt_sha256": verification["fold_receipt_sha256"]})
    except Exception as exc:
        queue.put({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)[:1000]})


def _boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()


def _job_key(job):
    return f"{job['profile']}/h{job['horizon']}/{job['target']}/fold{job['fold']:02d}"


def _recovered_elapsed(active, current_boot, now_monotonic, now_wall):
    if active.get("boot_id") == current_boot and now_monotonic >= active["started_monotonic"]:
        return now_monotonic - active["started_monotonic"]
    elapsed = now_wall - active["started_wall"]
    if elapsed < 0:
        raise ValueError("Cannot conservatively recover elapsed matrix time after restart")
    return elapsed


def execute(admission_dir: Path, review_path: Path, protocol_path: Path, *, max_jobs=None):
    """Run a single-device queue with identity, locking and wall-time checks."""
    if max_jobs is not None and (type(max_jobs) is not int or max_jobs < 1):
        raise ValueError("The optional job limit must be a positive integer")
    protocol, protocol_hash, protocol_path = _json_inside(protocol_path, OUT)
    selected = profile_selection(protocol.get("selected_profiles"))
    admission, admission_hash, _, review, review_hash, _ = load_reviewed(
        admission_dir, review_path, selected_profiles=selected)
    issues = locked_protocol_issues(admission, admission_hash, protocol, review_hash)
    if issues:
        raise ValueError("Frozen matrix blocked: " + ", ".join(issues))
    compute = review["compute"]
    if compute["host"] != socket.gethostname():
        raise ValueError("This host differs from the reviewed compute host")
    if compute["device"] == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise ValueError("Reviewed CUDA device is unavailable")
        if torch.cuda.get_device_name(0) != compute["device_name"]:
            raise ValueError("CUDA device differs from the reviewed budget")
    root = OUT / "matrix_runs"
    root.mkdir(parents=True, exist_ok=True)
    run_root = OUT / "synthetic_matrix_validation" / "runs" if protocol["data_kind"] == "synthetic_test" else root
    run_root.mkdir(parents=True, exist_ok=True)
    run_dir = run_root / protocol_hash[:24]
    run_dir.mkdir(exist_ok=True)
    if run_dir.is_symlink() or not run_dir.resolve().is_relative_to(run_root.resolve()):
        raise ValueError("Matrix run directory is not an isolated local directory")
    state_path = run_dir / "matrix_state.json"
    with (root / (compute["device"] + ".lock")).open("a+") as device_lock, (run_dir / "matrix.lock").open("a+") as run_lock:
        try:
            fcntl.flock(device_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("The device or matrix queue is already active") from None
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf8"))
            if (state.get("protocol_sha256") != protocol_hash or state.get("review_sha256") != review_hash or
                    state.get("matrix_scope") != protocol["matrix_scope"] or
                    state.get("selected_profiles") != protocol["selected_profiles"]):
                raise ValueError("Existing matrix state identity differs")
        else:
            state = {"schema_version": 1, "protocol_sha256": protocol_hash, "review_sha256": review_hash,
                     "data_kind": protocol["data_kind"], "status": "running",
                     "matrix_scope": protocol["matrix_scope"], "selected_profiles": protocol["selected_profiles"],
                     "omitted_profiles": protocol["omitted_profiles"],
                     "consumed_wall_seconds": 0.0, "active": None, "jobs": {},
                     "time_policy": "wall time, including interruptions; unexpected downtime may be conservatively counted"}
            atomic_json(state_path, state)
        active = state.get("active")
        if active:
            elapsed = _recovered_elapsed(active, _boot_id(), time.monotonic(), time.time())
            state["consumed_wall_seconds"] += elapsed
            state["active"] = None
            state["status"] = "interrupted"
            atomic_json(state_path, state)
        maximum = float(compute["maximum_wall_seconds"])
        ctx = mp.get_context("spawn")
        completed_this_call = 0
        for job in protocol["jobs"]:
            key = _job_key(job)
            existing = state["jobs"].get(key, {})
            if existing.get("status") == "completed":
                folder = (run_dir / existing["directory"]).resolve()
                if not folder.is_relative_to(run_dir.resolve()):
                    raise ValueError("Completed fold path escapes the matrix directory")
                verified = check_fold(folder)
                if (verified["fold_receipt_sha256"] != existing["receipt_sha256"] or
                        verified["protocol_identity"] != protocol_hash or
                        verified["data_identity"] != protocol["data_kind"] + ":" + admission_hash or
                        not all(verified["current_source_matches_recorded"].values())):
                    raise ValueError("A completed fold no longer matches its recorded identity")
                continue
            if max_jobs is not None and completed_this_call >= max_jobs:
                state["status"] = "paused_after_job_limit"
                atomic_json(state_path, state)
                return state
            remaining = maximum - state["consumed_wall_seconds"]
            if remaining <= 0:
                state["status"] = "budget_exhausted"
                atomic_json(state_path, state)
                return state
            base = run_dir / key
            folder = base
            if folder.exists() and not (folder / "fold_state.json").is_file():
                for number in range(1, 1000):
                    folder = base.with_name(base.name + f"-retry-{number:03d}")
                    if not folder.exists() or (folder / "fold_state.json").is_file():
                        break
                else:
                    raise ValueError("Too many incomplete fold attempts")
            resume = (folder / "fold_state.json").is_file()
            if not folder.resolve().is_relative_to(run_dir.resolve()):
                raise ValueError("Fold path escapes the matrix directory")
            relative = str(folder.relative_to(run_dir))
            state["jobs"][key] = {"status": "running", "directory": relative}
            active = {"job": key, "directory": relative, "boot_id": _boot_id(),
                      "started_monotonic": time.monotonic(), "started_wall": time.time()}
            state.update(active=active, status="running")
            atomic_json(state_path, state)
            queue = ctx.Queue(maxsize=1)
            process = ctx.Process(target=_worker, args=(str(admission_dir), str(review_path), str(protocol_path),
                                                       job, str(folder), resume, os.getpid(), queue))
            process.start()
            deadline = active["started_monotonic"] + remaining
            while process.is_alive() and time.monotonic() < deadline:
                process.join(min(10.0, max(0.01, deadline - time.monotonic())))
            timed_out = process.is_alive()
            if timed_out:
                process.terminate()
                process.join(10)
                if process.is_alive():
                    process.kill()
                    process.join()
            elapsed = time.monotonic() - active["started_monotonic"]
            state["consumed_wall_seconds"] += elapsed
            state["active"] = None
            if timed_out:
                state["jobs"][key]["status"] = "interrupted_at_budget"
                state["status"] = "budget_exhausted"
                atomic_json(state_path, state)
                return state
            try:
                outcome = queue.get(timeout=2)
            except Exception:
                outcome = {"status": "failed", "error_type": "WorkerExit", "error": str(process.exitcode)}
            queue.close()
            if process.exitcode != 0 or outcome.get("status") != "completed":
                state["jobs"][key].update(outcome)
                state["status"] = "failed"
                atomic_json(state_path, state)
                return state
            state["jobs"][key].update(status="completed", receipt_sha256=outcome["receipt_sha256"])
            completed_this_call += 1
            atomic_json(state_path, state)
        state["status"] = ("completed_synthetic_test_only" if protocol["data_kind"] == "synthetic_test" else
                           "completed_development_only" if protocol["matrix_scope"] == "full" else
                           "completed_partial_development_only")
        atomic_json(state_path, state)
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "execute"))
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--max-jobs", type=int, help="Stop after this many completed folds; resume using the same protocol")
    args = parser.parse_args()
    if args.action == "freeze":
        if args.recipe is None or args.protocol is not None or args.max_jobs is not None:
            raise ValueError("Freeze requires a reviewed recipe, not a protocol")
        path = freeze(args.admission, args.review, args.recipe)
        print(json.dumps({"status": json.loads(path.read_text(encoding="utf8"))["status"], "protocol": str(path)}))
    else:
        if args.protocol is None or args.recipe is not None:
            raise ValueError("Execute requires a frozen protocol")
        state = execute(args.admission, args.review, args.protocol, max_jobs=args.max_jobs)
        print(json.dumps({"status": state["status"], "completed_folds":
                          sum(v.get("status") == "completed" for v in state["jobs"].values()),
                          "consumed_wall_seconds": state["consumed_wall_seconds"]}))


if __name__ == "__main__":
    main()
