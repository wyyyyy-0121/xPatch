"""Run the separately reviewed shared-horizon comparison as a bounded queue.

No market work is admitted without the core review, completed full matrix,
mechanism-specific review, projection and separate user authorization. The
synthetic path is isolated and does not claim that the core study completed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import socket
import time

from .development import atomic_json
from .matrix_executor import _boot_id, _parent_death_guard, _recovered_elapsed, materialize
from .shared_development import SharedConfig, _MARKET_GATE_TOKEN, run_shared_fold
from .shared_gate import HORIZONS, freeze, load_locked
from .spec import OUT
from .verify_shared_development import SPLIT_KEYS, check_shared_fold


def _job_key(job):
    return f"seed{job['seed']}/fold{job['fold']:02d}"


def _config(job, protocol, protocol_hash):
    training = protocol["training"]
    return SharedConfig(job["base_job"]["lookback"], training["max_epochs"], training["patience"],
                        training["batch_size"], training["learning_rate"], job["seed"], protocol_hash)


def _verify_job(folder, job, protocol, protocol_hash, admission_hash):
    data_identity = protocol["data_kind"] + ":" + admission_hash
    checked = check_shared_fold(folder, expected_data_identity=data_identity,
                                expected_protocol_identity=protocol_hash,
                                expected_split_hashes={key: job["base_job"][key] for key in SPLIT_KEYS})
    receipt = json.loads((Path(folder) / "fold_receipt.json").read_text(encoding="utf8"))
    expected_config = asdict(_config(job, protocol, protocol_hash))
    expected_config["horizons"] = list(HORIZONS)
    if (receipt.get("config") != expected_config or
            any(receipt.get(key) != job["base_job"].get(key)
                for key in ("fit_origins", "stop_origins", "validation_origins"))):
        raise ValueError("Shared fold differs from its frozen configuration or sample counts")
    return checked


def _worker(admission_dir, base_review, base_protocol, mechanism_review, mechanism_protocol,
            base_run, job, expected_protocol_hash, folder, parent_pid, queue):
    try:
        _parent_death_guard(parent_pid)
        context, protocol, protocol_hash, _ = load_locked(
            Path(admission_dir), Path(base_review), Path(base_protocol), Path(mechanism_review),
            Path(mechanism_protocol), Path(base_run) if base_run else None)
        if protocol_hash != expected_protocol_hash or job not in protocol["jobs"]:
            raise ValueError("Shared worker protocol or job changed")
        bars, fold = materialize(context["source"], context["admission"], job["base_job"])
        cfg = _config(job, protocol, protocol_hash)
        identity = protocol["data_kind"] + ":" + context["admission_hash"]
        run_shared_fold(bars, fold, context["admission"]["split_actions"], cfg, Path(folder), identity,
                        device=context["mechanism_review"]["compute"]["device"],
                        _market_gate_token=_MARKET_GATE_TOKEN if protocol["data_kind"] == "market" else None)
        _, after_protocol, after_hash, _ = load_locked(
            Path(admission_dir), Path(base_review), Path(base_protocol), Path(mechanism_review),
            Path(mechanism_protocol), Path(base_run) if base_run else None)
        if after_hash != protocol_hash or job not in after_protocol["jobs"]:
            raise ValueError("Shared protocol or evidence changed during the fold")
        checked = _verify_job(Path(folder), job, protocol, protocol_hash, context["admission_hash"])
        queue.put({"status": "completed", "receipt_sha256": checked["fold_receipt_sha256"]})
    except Exception as exc:
        queue.put({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)[:1000]})


def execute(admission_dir: Path, base_review: Path, base_protocol: Path, mechanism_review: Path,
            mechanism_protocol: Path, base_run: Path | None = None, *, max_jobs=None, retry_incomplete=False):
    if max_jobs is not None and (type(max_jobs) is not int or max_jobs < 1):
        raise ValueError("The optional job limit must be a positive integer")
    if type(retry_incomplete) is not bool:
        raise ValueError("Retry selection must be boolean")
    # Recheck the saved core artifacts before each market queue invocation;
    # receipt hashes alone would not detect changed prediction/checkpoint files.
    # This read-only core preflight precedes the mechanism device lock.
    context, protocol, protocol_hash, _ = load_locked(admission_dir, base_review, base_protocol,
                                                       mechanism_review, mechanism_protocol, base_run,
                                                       deep_core=True)
    compute = context["mechanism_review"]["compute"]
    if compute["host"] != socket.gethostname():
        raise ValueError("This host differs from the reviewed mechanism budget")
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != compute["device_name"]:
        raise ValueError("The reviewed CUDA device is unavailable or differs")
    device_root = OUT / "matrix_runs"
    device_root.mkdir(parents=True, exist_ok=True)
    run_root = (OUT / "synthetic_mechanism_validation" / "runs" if protocol["data_kind"] == "synthetic_test"
                else OUT / "mechanism_runs")
    run_root.mkdir(parents=True, exist_ok=True)
    run_dir = run_root / protocol["budget_identity"][:24]
    run_dir.mkdir(exist_ok=True)
    if run_dir.is_symlink() or not run_dir.resolve().is_relative_to(run_root.resolve()):
        raise ValueError("Mechanism run directory is not isolated")
    state_path = run_dir / "mechanism_state.json"
    with (device_root / "cuda.lock").open("a+") as device_lock, (run_dir / "mechanism.lock").open("a+") as run_lock:
        try:
            fcntl.flock(device_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("The device or mechanism queue is already active") from None
        started, started_wall = time.monotonic(), time.time()
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf8"))
            if (state.get("schema_version") != 1 or state.get("protocol_sha256") != protocol_hash or
                    state.get("budget_identity") != protocol["budget_identity"] or
                    state.get("mechanism_review_sha256") != context["mechanism_review_hash"] or
                    state.get("data_kind") != protocol["data_kind"] or not isinstance(state.get("jobs"), dict)):
                raise ValueError("Existing mechanism state differs; a budget cannot be reset by refreezing")
        else:
            state = {"schema_version": 1, "protocol_sha256": protocol_hash,
                     "budget_identity": protocol["budget_identity"],
                     "mechanism_review_sha256": context["mechanism_review_hash"],
                     "data_kind": protocol["data_kind"], "independent_test": False, "status": "new",
                     "consumed_wall_seconds": 0.0, "active": None, "jobs": {},
                     "time_policy": "Queue wall time after device lock; interrupted downtime is conservatively counted."}
            atomic_json(state_path, state)
        consumed = state.get("consumed_wall_seconds")
        if isinstance(consumed, bool) or not isinstance(consumed, (int, float)) or not math.isfinite(consumed) or consumed < 0:
            raise ValueError("Invalid saved mechanism budget usage")
        keys = {_job_key(job) for job in protocol["jobs"]}
        if not set(state["jobs"]).issubset(keys):
            raise ValueError("Mechanism state contains an unexpected job")
        active = state.get("active")
        if active:
            state["consumed_wall_seconds"] += _recovered_elapsed(active, _boot_id(), started, started_wall)
            key = active.get("job")
            if key in state["jobs"] and state["jobs"][key].get("status") == "running":
                state["jobs"][key]["status"] = "interrupted"
                attempts = state["jobs"][key].get("attempts", [])
                if attempts:
                    attempts[-1]["status"] = "interrupted"
            state.update(active=None, status="interrupted")
            atomic_json(state_path, state)
        if any(record.get("status") == "running" for record in state["jobs"].values()):
            raise ValueError("A running mechanism job has no matching active budget record")
        maximum = float(compute["maximum_wall_seconds"])
        if state["consumed_wall_seconds"] >= maximum:
            state["consumed_wall_seconds"] += max(0.0, time.monotonic() - started)
            state["status"] = "budget_exhausted"
            atomic_json(state_path, state)
            return state
        initial = float(state["consumed_wall_seconds"])
        deadline = started + maximum - initial
        state.update(status="running", active={"boot_id": _boot_id(), "started_monotonic": started,
                                               "started_wall": started_wall, "job": None})
        atomic_json(state_path, state)
        process = None
        ctx = mp.get_context("spawn")
        completed_this_call = 0
        try:
            for job in protocol["jobs"]:
                key = _job_key(job)
                existing = state["jobs"].get(key, {})
                if existing.get("status") == "completed":
                    folder = (run_dir / existing["directory"]).resolve()
                    if not folder.is_relative_to(run_dir.resolve()):
                        raise ValueError("Completed mechanism fold escapes its run")
                    checked = _verify_job(folder, job, protocol, protocol_hash, context["admission_hash"])
                    if checked["fold_receipt_sha256"] != existing.get("receipt_sha256"):
                        raise ValueError("Completed mechanism fold identity changed")
                    if time.monotonic() >= deadline:
                        state["status"] = "budget_exhausted"
                        return state
                    continue
                if max_jobs is not None and completed_this_call >= max_jobs:
                    state["status"] = "paused_after_job_limit"
                    return state
                if existing.get("status") in ("failed", "interrupted", "interrupted_at_budget") and not retry_incomplete:
                    state["status"] = "needs_explicit_retry"
                    return state
                if time.monotonic() >= deadline:
                    state["status"] = "budget_exhausted"
                    return state
                attempts = list(existing.get("attempts", []))
                parent = run_dir / f"seed{job['seed']}"
                parent.mkdir(exist_ok=True)
                number = len(attempts) + 1
                while number < 1000 and (parent / f"fold{job['fold']:02d}-attempt-{number:03d}").exists():
                    number += 1
                if number >= 1000:
                    raise ValueError("Too many incomplete mechanism attempts")
                folder = parent / f"fold{job['fold']:02d}-attempt-{number:03d}"
                if not folder.resolve().is_relative_to(run_dir.resolve()):
                    raise ValueError("Mechanism fold path escapes its run")
                relative = str(folder.relative_to(run_dir))
                attempts.append({"directory": relative, "status": "running"})
                state["jobs"][key] = {"status": "running", "directory": relative, "attempts": attempts}
                state["active"]["job"] = key
                atomic_json(state_path, state)
                queue = ctx.Queue(maxsize=1)
                process = ctx.Process(target=_worker, args=(str(admission_dir), str(base_review), str(base_protocol),
                    str(mechanism_review), str(mechanism_protocol), str(base_run) if base_run else None,
                    job, protocol_hash, str(folder), os.getpid(), queue))
                process.start()
                while process.is_alive() and time.monotonic() < deadline:
                    process.join(min(5.0, max(0.01, deadline - time.monotonic())))
                timed_out = process.is_alive()
                if timed_out:
                    process.terminate()
                    process.join(10)
                    if process.is_alive():
                        process.kill()
                        process.join()
                try:
                    outcome = queue.get(timeout=max(0.01, min(2.0, deadline - time.monotonic()))) if not timed_out else {}
                except Exception:
                    outcome = {"status": "failed", "error_type": "WorkerExit", "error": str(process.exitcode)}
                exit_code = process.exitcode
                queue.close()
                process = None
                if timed_out:
                    state["jobs"][key]["status"] = attempts[-1]["status"] = "interrupted_at_budget"
                    state["status"] = "budget_exhausted"
                    return state
                if exit_code != 0 or outcome.get("status") != "completed":
                    if outcome.get("status") == "completed":
                        outcome = {"status": "failed", "error_type": "WorkerExit", "error": str(exit_code)}
                    state["jobs"][key].update(outcome)
                    attempts[-1].update(outcome)
                    state["status"] = "failed"
                    return state
                checked = _verify_job(folder, job, protocol, protocol_hash, context["admission_hash"])
                if checked["fold_receipt_sha256"] != outcome.get("receipt_sha256"):
                    raise ValueError("Mechanism worker receipt changed")
                state["jobs"][key].update(status="completed", receipt_sha256=outcome["receipt_sha256"])
                attempts[-1].update(status="completed", receipt_sha256=outcome["receipt_sha256"])
                state["active"]["job"] = None
                completed_this_call += 1
                atomic_json(state_path, state)
                if time.monotonic() >= deadline:
                    state["status"] = "budget_exhausted"
                    return state
            state["status"] = ("completed_synthetic_test_only" if protocol["data_kind"] == "synthetic_test"
                               else "completed_historical_development_only")
            return state
        except BaseException as exc:
            if process is not None and process.is_alive():
                process.terminate()
                process.join(10)
                if process.is_alive():
                    process.kill()
                    process.join()
            key = state["active"].get("job") if state.get("active") else None
            if key in state["jobs"] and state["jobs"][key].get("status") == "running":
                failure = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)[:1000]}
                state["jobs"][key].update(failure)
                attempts = state["jobs"][key].get("attempts", [])
                if attempts:
                    attempts[-1].update(failure)
            state["status"] = "failed"
            state["error"] = f"{type(exc).__name__}: {str(exc)[:1000]}"
            raise
        finally:
            state["consumed_wall_seconds"] = initial + max(0.0, time.monotonic() - started)
            state["active"] = None
            atomic_json(state_path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "execute"))
    for option in ("admission", "base-review", "base-protocol", "mechanism-review"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--base-run", type=Path)
    parser.add_argument("--mechanism-protocol", type=Path)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--retry-incomplete", action="store_true")
    args = parser.parse_args()
    if args.action == "freeze":
        if args.mechanism_protocol is not None or args.max_jobs is not None or args.retry_incomplete:
            raise ValueError("Freeze does not execute or retry jobs")
        path = freeze(args.admission, args.base_review, args.base_protocol, args.mechanism_review, args.base_run)
        print(json.dumps({"status": json.loads(path.read_text(encoding="utf8"))["status"], "protocol": str(path)}))
    else:
        if args.mechanism_protocol is None:
            raise ValueError("Execute requires the frozen mechanism protocol")
        state = execute(args.admission, args.base_review, args.base_protocol, args.mechanism_review,
                        args.mechanism_protocol, args.base_run, max_jobs=args.max_jobs,
                        retry_incomplete=args.retry_incomplete)
        print(json.dumps({"status": state["status"],
                          "completed_folds": sum(item.get("status") == "completed" for item in state["jobs"].values()),
                          "consumed_wall_seconds": state["consumed_wall_seconds"],
                          "data_kind": state["data_kind"]}))


if __name__ == "__main__":
    main()
