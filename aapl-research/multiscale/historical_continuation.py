"""Prepare and, only after separate user approval, finish original minute folds.

This module adds resource control around the frozen protocols. It never changes
their training code, identities, seeds, budgets, or queue-parent processes.
Preparation and preflight do not train or signal the original workers.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "outputs/multiscale"
TOOLS = OUT / "private_exploration_tools"
NIGHT = OUT / "private_low_load/20261107T153438623158Z"
MATRIX = OUT / "private_minute_matrix_development/20261107T151850900588Z"
PARALLEL = OUT / "private_minute_parallel_folds/20261107T143936100555Z"
ENDPOINT = OUT / "budget_endpoint_audits/20261108_original_cpu_budget/ORIGINAL_BUDGET_ENDPOINT_AUDIT.json"
ADMISSION = OUT / "admission/alpaca_20261107T052228573242Z"
SLOTS = ((0, 2), (4, 6), (8, 10), (12, 14), (16, 18), (20, 22))
CONTROL_CORES = (24, 26)
MAX_WALL_SECONDS = 36000
HEARTBEAT_SECONDS = 20
DEAD = {"Z", "X", "x"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".continuation-",
                                     encoding="utf8", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


@contextmanager
def locked(run, timeout=None):
    with (Path(run) / "STATE.lock").open("a+") as stream:
        if timeout is None:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        else:
            until = boot_time() + timeout
            while True:
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if boot_time() >= until:
                        raise TimeoutError("Controller holds the registry lock") from None
                    time.sleep(0.02)
        yield


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def boot_time():
    return time.clock_gettime(time.CLOCK_BOOTTIME)


def process(pid):
    if not isinstance(pid, int) or pid <= 1:
        return None
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        fields = raw[raw.rfind(")") + 2:].split()
        argv = Path(f"/proc/{pid}/cmdline").read_bytes()
        return {"pid": pid, "start_ticks": int(fields[19]), "boot_id": boot_id(),
                "argv_sha256": hashlib.sha256(argv).hexdigest(), "state": fields[0]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def matched(record):
    if not record:
        return None
    current = process(record["pid"])
    keys = ("pid", "start_ticks", "boot_id", "argv_sha256")
    return current if current and all(current[k] == record[k] for k in keys) else None


def pidfd_open(pid):
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    libc = ctypes.CDLL(None, use_errno=True)
    require(hasattr(libc, "pidfd_open"), "No pidfd API is available; no worker will be signalled")
    function = libc.pidfd_open
    function.argtypes, function.restype = [ctypes.c_int, ctypes.c_uint], ctypes.c_int
    result = function(pid, 0)
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def pidfd_signal(fd, signum):
    if hasattr(signal, "pidfd_send_signal"):
        return signal.pidfd_send_signal(fd, signum)
    libc = ctypes.CDLL(None, use_errno=True)
    require(hasattr(libc, "pidfd_send_signal"), "No pidfd signal API is available")
    function = libc.pidfd_send_signal
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    if function(fd, signum, None, 0) < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def safe_signal(record, signum):
    """Pin the process identity with a pidfd; never signal a process group."""
    if not matched(record):
        return False
    try:
        fd = pidfd_open(record["pid"])
    except ProcessLookupError:
        return False
    try:
        if not matched(record):
            return False
        pidfd_signal(fd, signum)
        return True
    except ProcessLookupError:
        return False
    finally:
        os.close(fd)


def check_affinity(record, slot):
    require(matched(record), "Worker process identity changed")
    for thread in Path(f"/proc/{record['pid']}/task").iterdir():
        require(os.sched_getaffinity(int(thread.name)).issubset(set(slot)),
                "A worker thread exceeds its two-core slot")


def config_for(protocol, job, protocol_hash):
    return {"profile": "minute", "horizon": job["horizon"], "target": job["target"],
            **{k: protocol[k] for k in ("lookback", "seasonal_period", "ridge_alphas", "max_epochs",
                                       "patience", "batch_size", "learning_rate", "seed")},
            "protocol_identity": protocol_hash}


def prepare(destination):
    """Inventory and bind the remaining original work, without opening arrays."""
    from .historical_coverage import snapshot
    destination = Path(destination).resolve()
    require(destination.is_relative_to(OUT / "remaining_compute_proposals"), "Invalid proposal location")
    require(not destination.exists(), "Proposal directory already exists")
    coverage = snapshot()
    endpoint = read(ENDPOINT)
    original_boot = read(NIGHT / "CHECKPOINT_AND_BUDGET_SNAPSHOT.json")["boot_id"]
    immutable = {str(ENDPOINT): digest(ENDPOINT)}
    for item in endpoint["files"]:
        require(digest(item["source"]) == digest(item["copy"]) == item["sha256"],
                "Original preserved checkpoint changed")
    rows = {(r["horizon"], r["target"], r["fold"]): r for r in coverage["folds"]
            if r["status"] != "completed"}
    require(rows and all(r["profile"] == "minute" for r in rows.values()), "Unexpected remaining scope")
    paused = {r["job"].split(":", 1)[1]: r for r in endpoint["workers"]}
    jobs = []
    for folder, kind in ((PARALLEL, "parallel"), (MATRIX, "matrix")):
        pp = folder / "PRIVATE_PROTOCOL.json"
        protocol = read(pp)
        ph = digest(pp)
        script = TOOLS / ("minute_parallel_folds_45.py" if kind == "parallel" else "minute_matrix_queue.py")
        require(digest(script) == protocol["script_sha256"], "Original worker code changed")
        immutable.update({str(pp): ph, str(script): digest(script)})
        immutable.update({str(PROJECT / name): value for name, value in protocol["source_sha256"].items()})
        immutable[str(ADMISSION / "admission.json")] = protocol["admission_sha256"]
        for original in protocol["jobs"]:
            job = dict(original)
            if kind == "parallel":
                job.update(horizon=60, target="future_log_return")
            task = (job["horizon"], job["target"], job["fold"])
            if task not in rows:
                continue
            key = f"h{task[0]}_{task[1]}_f{task[2]}"
            qkey = str(job["fold"]) if kind == "parallel" else job["key"]
            fd = folder / f"fold_{job['fold']:02d}" if kind == "parallel" else folder / "jobs" / key
            vp = (folder / f"fold_{job['fold']:02d}_verification.json" if kind == "parallel"
                  else folder / "verifications" / (key + ".json"))
            cfg = config_for(protocol, job, ph)
            require(cfg["seed"] == 2026 and cfg["max_epochs"] == 20 and cfg["patience"] == 3,
                    "Original training settings differ")
            saved = read(fd / "fold_state.json") if (fd / "fold_state.json").exists() else None
            if saved:
                require(saved["config"] == cfg and saved["fold_identity"] == canonical(saved["identity_fields"])
                        and saved["Q3_values_read"] is False, "Saved fold identity differs")
            record = paused.get(qkey)
            live, slot = None, None
            if record:
                slot = record["affinity"]
                live = process(record["pid"])
                if live and live["boot_id"] == original_boot and live["start_ticks"] == record["start_ticks"]:
                    argv = Path(f"/proc/{live['pid']}/cmdline").read_bytes().split(b"\0")
                    require(str(script).encode() in argv and str(pp).encode() in argv and b"--worker" in argv,
                            "Original PID no longer belongs to this worker")
                    require(live["state"] in ("T", "t"), "Original worker is already active")
                    check_affinity(live, slot)
                else:
                    live = None
            require(not record or saved, "Paused worker lacks resumable state")
            jobs.append({"key": key, "original_job": original, "kind": kind, "protocol": str(pp),
                         "protocol_sha256": ph, "queue": str(folder / "PRIVATE_QUEUE_STATE.json"),
                         "queue_collection": "workers" if kind == "parallel" else "jobs", "queue_key": qkey,
                         "folder": str(fd), "verification": str(vp), "config": cfg,
                         "data_identity": ("private-minute-parallel:" if kind == "parallel" else
                                           "private-minute-matrix:") + protocol["admission_sha256"],
                         "initial_fold_state_sha256": digest(fd / "fold_state.json") if saved else None,
                         "initial_fold_identity": saved["fold_identity"] if saved else None,
                         "original_process": live, "preferred_slot": slot,
                         "origins_sha256": protocol["origins_sha256"] if kind == "parallel"
                         else job["origins_sha256"]})
    require(len(jobs) == len(rows) and len({j["key"] for j in jobs}) == len(rows),
            "Remaining folds are not represented exactly once")
    manifest = {"status": "prepared_not_authorized_or_started", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "code_sha256": digest(__file__), "completed_folds_at_preparation": coverage["completed_folds"],
                "planned_folds": 100, "maximum_wall_seconds": MAX_WALL_SECONDS,
                "maximum_workers": 6, "threads_per_worker": 2, "slots": [list(s) for s in SLOTS],
                "control_cores": list(CONTROL_CORES), "gpu_training": False, "new_seeds": 0,
                "new_configurations": 0, "Q3_values_read": False, "immutable_evidence": immutable,
                "jobs": jobs, "original_budget_records_modified": False,
                "limitations": ["An OS shutdown can lose updates newer than the last periodic checkpoint.",
                                "The additional budget requires a new direct user approval; it is not granted here."]}
    validate_manifest(manifest)
    destination.mkdir(parents=True)
    save(destination / "COVERAGE.json", coverage)
    save(destination / "MANIFEST.json", manifest)
    return manifest


def validate_manifest(manifest):
    require(manifest["status"] == "prepared_not_authorized_or_started" and
            manifest["code_sha256"] == digest(__file__), "Continuation code or scope changed; prepare again")
    require(manifest["maximum_wall_seconds"] == MAX_WALL_SECONDS and manifest["maximum_workers"] == 6 and
            manifest["threads_per_worker"] == 2 and manifest["slots"] == [list(s) for s in SLOTS] and
            manifest["gpu_training"] is False and manifest["Q3_values_read"] is False and
            manifest["new_seeds"] == manifest["new_configurations"] == 0, "Continuation limits differ")
    for path, expected in manifest["immutable_evidence"].items():
        require(digest(path) == expected, "Original protocol, source or admission changed: " + path)
    for job in manifest["jobs"]:
        require(job["kind"] in ("parallel", "matrix"), "Unexpected original worker kind")
        protocol = read(job["protocol"])
        require(job["protocol_sha256"] == digest(job["protocol"]) and
                protocol["Q3_values_read"] is False and protocol["independent_test"] is False,
                "Original protocol scope differs")
        original = dict(job["original_job"])
        if job["kind"] == "parallel":
            original.update(horizon=60, target="future_log_return")
        require(job["original_job"] in protocol["jobs"] and
                job["config"] == config_for(protocol, original, job["protocol_sha256"]),
                "Job configuration differs from its original protocol")
        folder = PARALLEL if job["kind"] == "parallel" else MATRIX
        number = original["fold"]
        key = f"h{original['horizon']}_{original['target']}_f{number}"
        expected_folder = folder / f"fold_{number:02d}" if job["kind"] == "parallel" else folder / "jobs" / key
        expected_verification = (folder / f"fold_{number:02d}_verification.json" if job["kind"] == "parallel"
                                 else folder / "verifications" / (key + ".json"))
        require(Path(job["protocol"]) == folder / "PRIVATE_PROTOCOL.json" and job["key"] == key and
                Path(job["folder"]) == expected_folder and Path(job["verification"]) == expected_verification and
                Path(job["queue"]) == folder / "PRIVATE_QUEUE_STATE.json" and
                job["queue_collection"] == ("workers" if job["kind"] == "parallel" else "jobs") and
                job["queue_key"] == (str(number) if job["kind"] == "parallel" else key),
                "Job or mutable queue location differs from its original protocol")
        require(job["preferred_slot"] is None or tuple(job["preferred_slot"]) in SLOTS, "Unrecognized preferred slot")
        expected_data = ("private-minute-parallel:" if job["kind"] == "parallel" else "private-minute-matrix:")
        require(job["data_identity"] == expected_data + protocol["admission_sha256"] and
                job["origins_sha256"] == (protocol["origins_sha256"] if job["kind"] == "parallel"
                                          else original["origins_sha256"]), "Job data or origin identity differs")
    require(len({j["key"] for j in manifest["jobs"]}) == len(manifest["jobs"]), "Duplicated continuation job")


def approval(path, manifest_path):
    require(path and Path(path).is_file(), "No additional compute approval: training remains disabled")
    value = read(path)
    seconds = value.get("maximum_wall_seconds")
    workers = value.get("maximum_workers")
    require(value.get("status") == "approved_by_user" and
            value.get("scope") == "remaining_original_core_folds_only" and
            value.get("manifest_sha256") == digest(manifest_path) and
            value.get("evidence", {}).get("kind") == "direct_user_reply" and
            bool(value.get("evidence", {}).get("text", "").strip()) and
            type(seconds) is int and 5 <= seconds <= MAX_WALL_SECONDS and
            type(workers) is int and 1 <= workers <= 6 and value.get("threads_per_worker") == 2 and
            value.get("gpu_training") is False, "Invalid or excessive additional compute approval")
    return value


def completed(job, extra_verification=None):
    from .historical_coverage import verify_fold
    folder = Path(job["folder"])
    if not (folder / "fold_receipt.json").is_file():
        return None
    cfg = job["config"]
    candidates = [Path(job["verification"])]
    candidates += [candidates[0].with_name(candidates[0].stem + "_night_resumed.json")]
    queue = read(job["queue"])
    prior = queue[job["queue_collection"]][job["queue_key"]]
    if prior.get("verification"):
        candidates.append(Path(prior["verification"]))
    if extra_verification:
        candidates.append(Path(extra_verification))
    errors = []
    for candidate in candidates:
        if candidate.is_file():
            entry = {"status": "completed", "exit_code": 0, "verification": str(candidate)}
            try:
                return verify_fold(entry, ("minute", cfg["horizon"], cfg["target"], job["original_job"]["fold"]),
                                   job["original_job"], candidate, folder, protocol_hash=job["protocol_sha256"])
            except ValueError as exc:
                errors.append(str(exc))
    require(not errors, "Completion evidence differs: " + "; ".join(errors))
    return None


def preflight(manifest):
    validate_manifest(manifest)
    pause = read(NIGHT / "PAUSE_MANIFEST.json")
    if pause["boot_id"] == boot_id():
        for record in pause["processes"]:
            if record["pid"] not in pause["parents"]:
                continue
            current = process(record["pid"])
            if current and current["start_ticks"] == record["start_ticks"]:
                require(current["state"] in DEAD | {"T", "t"}, "An original queue parent is active")
    require(read(NIGHT / "NIGHT_RESUME_STATE.json")["status"] == "finished_with_paused_jobs",
            "The original recovery controller has not finished its budget accounting")
    fd = pidfd_open(os.getpid())
    try:
        pidfd_signal(fd, 0)  # Capability probe only; no process state changes.
    finally:
        os.close(fd)
    report = []
    for job in manifest["jobs"]:
        if completed(job):
            mode = "skip_verified_completion"
        else:
            state = Path(job["folder"]) / "fold_state.json"
            require((digest(state) if state.exists() else None) == job["initial_fold_state_sha256"],
                    "Incomplete fold changed since preparation; prepare a new inventory")
            live = matched(job["original_process"])
            if live and live["state"] not in DEAD:
                require(live["state"] in ("T", "t"), "Original worker is no longer frozen")
                check_affinity(live, job["preferred_slot"])
                mode = "live_memory"
            else:
                mode = "checkpoint_fallback" if state.exists() else "original_unstarted_fold"
        report.append({"key": job["key"], "mode": mode})
    return report


def choose_jobs(manifest, state, maximum_workers):
    active = [v for v in state["jobs"].values() if v["phase"] == "running"]
    used = {tuple(v["slot"]) for v in active}
    require(len(used) == len(active) and len(active) <= maximum_workers, "Active CPU slots overlap")
    available = [s for s in SLOTS if s not in used]
    selected = []
    for job in manifest["jobs"]:
        if state["jobs"][job["key"]]["phase"] not in ("pending", "frozen"):
            continue
        preferred = tuple(job["preferred_slot"]) if job["preferred_slot"] else None
        slot = preferred if preferred in available else (available[0] if available and preferred is None else None)
        if slot is None or len(active) + len(selected) >= maximum_workers:
            continue
        selected.append((job, slot))
        available.remove(slot)
    return selected


def deadline_reason(state):
    if state["boot_id"] != boot_id():
        return "boot_changed"
    if time.time() >= state["deadline_epoch"] or boot_time() >= state["deadline_boottime"]:
        return "additional_budget_exhausted"
    if not matched(state["controller"]):
        return "controller_exited"
    if boot_time() - state["heartbeat_boottime"] > HEARTBEAT_SECONDS:
        return "controller_heartbeat_lost"
    return None


def guard_ready(state):
    require(state["phase"] in ("armed", "running") and not deadline_reason(state),
            "Continuation is stopped or its approved budget expired")
    guard = matched(state.get("watchdog"))
    require(guard and guard["state"] not in DEAD | {"T", "t"}, "Independent watchdog is not running")


def copy_stable(folder, destination):
    files = []
    folder, destination = Path(folder), Path(destination)
    if not folder.exists():
        return files
    for source in sorted(folder.rglob("*")):
        require(not source.is_symlink(), "Refuse a symlink in the checkpoint tree")
        if not source.is_file() or source.name.startswith(".") or source.suffix not in (".json", ".pt", ".npz", ".npy"):
            continue
        target = destination / source.relative_to(folder)
        expected = digest(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            require(digest(target) == expected, "Existing preservation copy differs")
        else:
            with tempfile.NamedTemporaryFile("w+b", dir=target.parent, prefix=".preserve-", delete=False) as dst:
                temporary = Path(dst.name)
                try:
                    with source.open("rb") as src:
                        for block in iter(lambda: src.read(1024 * 1024), b""):
                            dst.write(block)
                    dst.flush()
                    os.fsync(dst.fileno())
                    require(digest(temporary) == digest(source) == expected, "Checkpoint changed during copy")
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
        require(digest(source) == digest(target) == expected, "Checkpoint changed while preserving")
        files.append({"source": str(source), "copy": str(target), "sha256": expected, "bytes": target.stat().st_size})
    return files


def freeze_and_preserve(run, manifest, state, reason):
    """Caller either holds the lock or has already frozen its controller."""
    run = Path(run)
    if state["phase"] in ("completed", "paused_preserved"):
        return state
    state.update(phase="stopping", stop_reason=state.get("stop_reason", reason))
    save(run / "STATE.json", state)
    records = [v.get("process") for v in state["jobs"].values() if v.get("process")]
    for record in records:
        safe_signal(record, signal.SIGSTOP)
    until = boot_time() + 5
    while any((current := matched(r)) and current["state"] not in DEAD | {"T", "t"} for r in records):
        require(boot_time() < until, "A matching worker has not frozen; no inconsistent copy made")
        time.sleep(0.05)
    files = []
    for job in manifest["jobs"]:
        if state["jobs"][job["key"]]["phase"] != "completed":
            files.extend(copy_stable(job["folder"], run / "preserved" / job["key"]))
            state["jobs"][job["key"]]["phase"] = "paused"
    receipt = {"status": "matching_workers_frozen_and_files_preserved", "reason": state["stop_reason"],
               "at_utc": datetime.now(timezone.utc).isoformat(), "files": files,
               "total_bytes": sum(f["bytes"] for f in files),
               "processes": [{"record": r, "current": matched(r)} for r in records],
               "original_budget_reset": False, "Q3_values_read": False}
    save(run / "STOP_RECEIPT.json", receipt)
    state.update(phase="paused_preserved", stopped_at_utc=receipt["at_utc"])
    save(run / "STATE.json", state)
    return state


def stop_run(run, manifest, reason):
    """Controller path: serialize preservation with its own worker release."""
    with locked(run):
        return freeze_and_preserve(run, manifest, read(Path(run) / "STATE.json"), reason)


def independent_stop(run, manifest, reason):
    """The guard must not wait for a lock held by a stalled controller."""
    run = Path(run)
    state = read(run / "STATE.json")  # Atomic JSON publication makes this read coherent.
    if state["phase"] in ("completed", "paused_preserved"):
        return state
    controller = state["controller"]
    require(controller["pid"] != os.getpid(), "Watchdog cannot be its own controller")
    safe_signal(controller, signal.SIGSTOP)
    until = boot_time() + 5
    while (current := matched(controller)) and current["state"] not in DEAD | {"T", "t"}:
        # Stop any published workers promptly even if the controller is in I/O.
        for item in read(run / "STATE.json")["jobs"].values():
            if item.get("process"):
                safe_signal(item["process"], signal.SIGSTOP)
        require(boot_time() < until, "Controller has not frozen; no unsafe registry takeover")
        time.sleep(0.02)
    # The controller can neither rewrite the registry nor release another job.
    state = freeze_and_preserve(run, manifest, read(run / "STATE.json"), reason)
    # Only this new controller is retired, after verified preservation. Its
    # checked KillMode=process service keeps all frozen worker RAM intact.
    safe_signal(controller, signal.SIGKILL)
    return state


def watchdog_loop(run, manifest):
    run = Path(run)
    try:
        with locked(run, timeout=1):
            state = read(run / "STATE.json")
            state["watchdog"] = process(os.getpid())
            save(run / "STATE.json", state)
            save(run / "WATCHDOG_READY.json", {"process": state["watchdog"],
                                               "manifest_sha256": state["manifest_sha256"]})
    except TimeoutError:
        independent_stop(run, manifest, "controller_lock_stalled_at_guard_start")
        return
    while True:
        state = read(run / "STATE.json")
        if state["phase"] in ("completed", "paused_preserved"):
            return
        reason = ("user_cancelled" if (run / "CANCEL").exists() else deadline_reason(state))
        if reason:
            independent_stop(run, manifest, reason)
            return
        time.sleep(0.2)


def bounded_environment(cores):
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="-1", PYTHONPATH=str(PROJECT), PYTHONDONTWRITEBYTECODE="1",
               TOKENIZERS_PARALLELISM="false", AAPL_CPU_IDS=",".join(map(str, cores)))
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "POLARS_MAX_THREADS", "RAYON_NUM_THREADS", "ARROW_NUM_THREADS"):
        env[key] = "2"
    return env


def install_watchdog(run):
    unit = "aapl-continuation-" + run.name.lower()
    args = ["systemd-run", "--user", "--unit=" + unit, "--property=Restart=on-failure",
            "--property=RestartSec=1", "--property=WorkingDirectory=" + str(PROJECT),
            "/usr/bin/taskset", "-c", ",".join(map(str, CONTROL_CORES)), str(PROJECT / ".venv/bin/python"),
            "-m", "multiscale.historical_continuation", "watchdog", "--run", str(run)]
    result = subprocess.run(args, capture_output=True, text=True, timeout=20)
    require(result.returncode == 0, "Independent watchdog registration failed: " + result.stderr[-300:])
    deadline = boot_time() + 10
    while not (run / "WATCHDOG_READY.json").exists():
        require(boot_time() < deadline, "Independent watchdog did not become ready")
        time.sleep(0.1)
    active = subprocess.run(["systemctl", "--user", "is-active", unit + ".service"],
                            capture_output=True, text=True, timeout=5)
    require(active.returncode == 0 and active.stdout.strip() == "active", "Watchdog service is not active")
    with locked(run):
        state = read(run / "STATE.json")
        guard_ready(state)
        state["watchdog_unit"] = unit
        save(run / "STATE.json", state)


def update_completed(job, evidence, run):
    """Only mutable completion entries change; old budget fields stay intact."""
    queue = read(job["queue"])
    entry = queue[job["queue_collection"]][job["queue_key"]]
    entry.update(status="completed", exit_code=0, receipt_sha256=evidence["receipt_sha256"],
                 verification=evidence["verification"], verification_sha256=evidence["verification_sha256"],
                 completed_in_continuation=str(run))
    if job["queue_collection"] == "jobs":
        queue["completed"] = sum(v["status"] == "completed" for v in queue["jobs"].values())
    save(job["queue"], queue)


def memory_available():
    return next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:"))


def release_job(run, job, slot):
    """A new worker stops before importing numerical code or opening data."""
    with locked(run):
        state = read(run / "STATE.json")
        guard_ready(state)
        entry = state["jobs"][job["key"]]
        record = entry.get("process")
        current = matched(record)
        if not current or current["state"] in DEAD:
            log = (run / (job["key"] + ".log")).open("x", encoding="utf8")
            try:
                child = subprocess.Popen([str(PROJECT / ".venv/bin/python"), "-m", "multiscale.historical_continuation",
                                          "worker", "--run", str(run), "--job", job["key"],
                                          "--cores", ",".join(map(str, slot))], cwd=PROJECT,
                                         env=bounded_environment(slot), stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
            finally:
                log.close()
            until = boot_time() + 5
            while True:
                record = process(child.pid)
                if record and record["state"] in ("T", "t"):
                    break
                require(child.poll() is None and boot_time() < until, "New worker did not stop at its launch gate")
                time.sleep(0.02)
            entry["mode"] = "checkpoint_fallback" if (Path(job["folder"]) / "fold_state.json").exists() else "fresh_original_fold"
        else:
            require(current["state"] in ("T", "t"), "Matching worker is already active")
            entry["mode"] = "live_memory"
        check_affinity(record, slot)
        entry.update(phase="frozen", process=record, slot=list(slot))
        save(run / "STATE.json", state)
        guard_ready(state)
        entry.update(phase="running", process=record, slot=list(slot))
        state["phase"] = "running"
        save(run / "STATE.json", state)
        require(safe_signal(record, signal.SIGCONT), "Worker disappeared before release")


def check_controller_service():
    unit = os.environ.get("AAPL_CONTINUATION_UNIT")
    require(unit and unit.startswith("aapl-continuation-controller-"),
            "Use launch after approval: the controller must run independently of the tool session")
    result = subprocess.run(["systemctl", "--user", "show", unit + ".service",
                             "--property=MainPID", "--property=KillMode"],
                            capture_output=True, text=True, timeout=5)
    properties = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    require(result.returncode == 0 and properties.get("MainPID") == str(os.getpid()) and
            properties.get("KillMode") == "process", "Controller service identity or preservation policy differs")
    return unit


def launch(manifest_path, approval_path):
    """Register the approved controller outside any tool/terminal session."""
    manifest_path, approval_path = Path(manifest_path).resolve(), Path(approval_path).resolve() if approval_path else None
    preflight(read(manifest_path))
    approval(approval_path, manifest_path)
    unit = "aapl-continuation-controller-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ").lower()
    args = ["systemd-run", "--user", "--unit=" + unit, "--property=Restart=no",
            "--property=KillMode=process", "--property=WorkingDirectory=" + str(PROJECT),
            "--setenv=AAPL_CONTINUATION_UNIT=" + unit]
    env = bounded_environment(CONTROL_CORES)
    for key in ("CUDA_VISIBLE_DEVICES", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS",
                "RAYON_NUM_THREADS", "ARROW_NUM_THREADS", "TOKENIZERS_PARALLELISM"):
        args.append("--setenv=" + key + "=" + env[key])
    args.extend(["/usr/bin/taskset", "-c", ",".join(map(str, CONTROL_CORES)), str(PROJECT / ".venv/bin/python"),
                 "-m", "multiscale.historical_continuation", "start", "--manifest", str(manifest_path),
                 "--approval", str(approval_path)])
    result = subprocess.run(args, capture_output=True, text=True, timeout=20)
    require(result.returncode == 0, "Controller service registration failed: " + result.stderr[-300:])
    return {"controller_unit": unit, "status": "approved_controller_service_registered",
            "note": "Inspect ACTIVE.json and WATCHDOG_READY.json to confirm worker release."}


def start(manifest_path, approval_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = read(manifest_path)
    modes = preflight(manifest)
    allowed = approval(approval_path, manifest_path)
    controller_unit = check_controller_service()
    parent = OUT / "remaining_continuations"
    parent.mkdir(parents=True, exist_ok=True)
    with (parent / "active.lock").open("a+") as global_lock:
        fcntl.flock(global_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (parent / "ACTIVE.json").exists():
            prior = read(parent / "ACTIVE.json")
            require(read(Path(prior["run"]) / "STATE.json")["phase"] in ("completed", "paused_preserved"),
                    "A previous continuation is still active or lacks verified preservation")
        used = parent / ("APPROVAL_USED_" + canonical(allowed) + ".json")
        require(not used.exists(), "This approval has already been consumed; no budget reset")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run = parent / stamp
        run.mkdir()
        (run / "verifications").mkdir()
        save(run / "APPROVAL.json", allowed)
        # One second of margin also covers watcher polling; suspend time counts.
        seconds = allowed["maximum_wall_seconds"] - 1
        state = {"phase": "armed", "boot_id": boot_id(), "controller": process(os.getpid()),
                 "controller_unit": controller_unit,
                 "manifest": str(manifest_path), "manifest_sha256": digest(manifest_path),
                 "approval_sha256": digest(run / "APPROVAL.json"), "heartbeat_boottime": boot_time(),
                 "deadline_epoch": time.time() + seconds, "deadline_boottime": boot_time() + seconds,
                 "approved_maximum_wall_seconds": allowed["maximum_wall_seconds"],
                 "maximum_workers": allowed["maximum_workers"], "watchdog": None,
                 "jobs": {j["key"]: {"phase": "completed" if modes[i]["mode"] == "skip_verified_completion" else
                                      ("frozen" if j["original_process"] else "pending"),
                                      "process": j["original_process"], "slot": j["preferred_slot"]}
                          for i, j in enumerate(manifest["jobs"])}}
        save(run / "STATE.json", state)
        save(used, {"run": str(run), "manifest_sha256": digest(manifest_path)})
        save(parent / "ACTIVE.json", {"run": str(run)})
        cancelled = False

        def cancel(signum, frame):
            nonlocal cancelled
            cancelled = True

        signal.signal(signal.SIGTERM, cancel)
        signal.signal(signal.SIGINT, cancel)
        try:
            install_watchdog(run)  # Must be live and verified before any release.
            while True:
                with locked(run):
                    state = read(run / "STATE.json")
                    if state["phase"] == "paused_preserved":
                        return run
                    guard_ready(state)
                    state["heartbeat_boottime"] = boot_time()
                    for job in manifest["jobs"]:
                        entry = state["jobs"][job["key"]]
                        current = matched(entry.get("process"))
                        if entry["phase"] == "running" and (not current or current["state"] in DEAD):
                            evidence = completed(job, run / "verifications" / (job["key"] + ".json"))
                            require(evidence, "Worker exited without verified completion: " + job["key"])
                            update_completed(job, evidence, run)
                            entry.update(phase="completed", evidence=evidence)
                    if all(v["phase"] == "completed" for v in state["jobs"].values()):
                        state["phase"] = "completed"
                        save(run / "STATE.json", state)
                        return run
                    save(run / "STATE.json", state)
                if cancelled or (run / "CANCEL").exists() or memory_available() < 2 * 1024 ** 3:
                    stop_run(run, manifest, "user_cancelled" if cancelled or (run / "CANCEL").exists() else "low_available_memory")
                    return run
                for job, slot in choose_jobs(manifest, state, allowed["maximum_workers"]):
                    if memory_available() < 4 * 1024 ** 3:
                        break
                    release_job(run, job, slot)
                time.sleep(0.5)
        except BaseException:
            reason = deadline_reason(read(run / "STATE.json")) or "controller_error"
            stop_run(run, manifest, reason)
            if reason == "additional_budget_exhausted":
                return run
            raise


def worker_context(job):
    """Rebuild the original identity, including checkpoint optimizer/scaler state."""
    import numpy as np
    import pandas as pd
    import torch
    from .development import FoldConfig, bars_identity, validate_split
    from .features import CloseWindows, fit_scaler
    from .matrix_executor import read_bars
    from .splits import rolling_folds
    from .training import TrainConfig, digest_indices, identity
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg_dict = dict(job["config"])
    cfg_dict["ridge_alphas"] = tuple(cfg_dict["ridge_alphas"])
    cfg = FoldConfig(**cfg_dict)
    source = read(ADMISSION / "admission.json")
    require(source["Q3_values_read"] is False, "Protected source scope changed")
    op = ADMISSION / f"origins_minute_h{cfg.horizon}.npy"
    require(digest(op) == job["origins_sha256"], "Original origins changed")
    bars = read_bars(ADMISSION, source, "minute")
    origins = np.load(op, allow_pickle=False)
    fold = rolling_folds(bars, origins, cfg.horizon)[job["original_job"]["fold"] - 1]
    train, fit, stop, val = validate_split(bars, fold, cfg.horizon)
    for key, part in (("train", train), ("fit", fit), ("stop", stop), ("validation", val)):
        require(digest_indices(part) == job["original_job"]["split_sha256"][key], "Original split changed")
    path = Path(job["folder"]) / "fold_state.json"
    details = []
    if path.exists():
        state = read(path)
        fields = state["identity_fields"]
        runtime = {"device": "cpu", "python": sys.version.split()[0], "numpy": np.__version__,
                   "pandas": pd.__version__, "torch": str(torch.__version__), "cuda": torch.version.cuda, "gpu": None}
        require(state["config"] == job["config"] and state["data_identity"] == job["data_identity"] and
                state["fold_identity"] == canonical(fields) and fields["runtime"] == runtime and
                fields["bars_sha256"] == bars_identity(bars) and fields["actions"] == source["split_actions"],
                "Saved fold data, configuration, actions or runtime differs")
        for name, part in (("train", train), ("fit", fit), ("stop", stop), ("validation", val)):
            require(fields[name + "_indices_sha256"] == digest_indices(part), "Saved split differs")
        windows = CloseWindows(bars, cfg.lookback, fields["actions"])
        scaler = None
        for name in ("mlp", "gru", "lstm", "xpatch"):
            latest = path.parent / name / "latest.pt"
            if name in state["models"] or not latest.exists():
                continue
            if scaler is None:
                scaler = asdict(fit_scaler(windows, fit, cfg.horizon, cfg.target))
            model_cfg = TrainConfig(name, cfg.lookback, cfg.horizon, cfg.target, cfg.max_epochs,
                                    cfg.patience, cfg.batch_size, cfg.learning_rate, cfg.seed)
            expected = identity(model_cfg, job["data_identity"] + ":" + cfg.protocol_identity + ":" +
                                state["fold_identity"], fit, stop, "cpu")
            checkpoint = torch.load(latest, map_location="cpu", weights_only=True)
            require(checkpoint["identity"] == expected and checkpoint["scaler"] == scaler,
                    "Checkpoint optimizer identity or scaler differs")
            details.append({"model": name, "epoch": checkpoint["state"]["epoch"],
                            "steps": checkpoint["state"]["steps"], "checkpoint_sha256": digest(latest)})
    return bars, fold, source["split_actions"], cfg, details


def run_worker(run, key, cores):
    os.sched_setaffinity(0, set(cores))
    os.kill(os.getpid(), signal.SIGSTOP)  # No data or numerical library is loaded before this gate.
    run = Path(run)
    state = read(run / "STATE.json")
    manifest_path = Path(state["manifest"])
    require(digest(manifest_path) == state["manifest_sha256"], "Active manifest changed")
    manifest = read(manifest_path)
    validate_manifest(manifest)
    allowed = approval(run / "APPROVAL.json", manifest_path)
    require(digest(run / "APPROVAL.json") == state["approval_sha256"] and allowed["gpu_training"] is False,
            "Active approval changed")
    guard_ready(state)
    entry = state["jobs"][key]
    require(entry["phase"] == "running" and matched(entry["process"]) and
            entry["process"]["pid"] == os.getpid() and entry["slot"] == list(cores), "Worker gate was not registered")
    job = next(j for j in manifest["jobs"] if j["key"] == key)
    bars, fold, actions, cfg, _ = worker_context(job)
    from .development import run_development_fold
    from .verify_development import check_fold
    folder = Path(job["folder"])
    run_development_fold(bars, fold, actions, cfg, folder, job["data_identity"], device="cpu",
                         resume=(folder / "fold_state.json").is_file())
    verification = check_fold(folder)
    require(all(verification["current_source_matches_recorded"].values()), "Source changed during continuation")
    save(run / "verifications" / (key + ".json"), verification)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "preflight", "checkpoint-check", "launch", "start", "watchdog", "worker"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--job")
    parser.add_argument("--cores", default="24,26")
    args = parser.parse_args()
    if args.mode == "worker":
        cores = tuple(map(int, args.cores.split(",")))
        require(cores in SLOTS, "Unrecognized worker CPU slot")
        run_worker(args.run, args.job, cores)
        return
    os.sched_setaffinity(0, set(CONTROL_CORES))
    if args.mode == "prepare":
        manifest = prepare(args.output)
        print(json.dumps({"status": manifest["status"], "jobs": len(manifest["jobs"]), "training_started": False}))
    elif args.mode in ("preflight", "checkpoint-check"):
        manifest = read(args.manifest)
        modes = preflight(manifest)
        details = None
        if args.mode == "checkpoint-check":
            job = next(j for j in manifest["jobs"] if j["key"] == args.job)
            details = worker_context(job)[-1]
        result = {"status": "continuation_preflight_passed", "jobs": modes, "checkpoint_details": details,
                  "training_started": False, "processes_resumed": False, "Q3_values_read": False,
                  "manifest_sha256": digest(args.manifest)}
        if args.output:
            require(not args.output.exists(), "Verification output already exists")
            save(args.output, result)
        print(json.dumps(result))
    elif args.mode == "launch":
        print(json.dumps(launch(args.manifest, args.approval)))
    elif args.mode == "start":
        require(args.manifest is not None, "Manifest is required")
        print(json.dumps({"run": str(start(args.manifest, args.approval))}))
    else:
        state = read(args.run / "STATE.json")
        require(digest(state["manifest"]) == state["manifest_sha256"], "Active manifest changed")
        manifest = read(state["manifest"])
        # The watchdog must still freeze/archive if source or approval files later change.
        watchdog_loop(args.run, manifest)


if __name__ == "__main__":
    main()
