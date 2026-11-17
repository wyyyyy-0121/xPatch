"""Amend an existing continuation to eight workers without extending its budget.

The original manifest, approval, consumed-approval marker, training sources,
worker identities remain intact. A fresh registry and watchdog inherit the
exact original epoch and boottime deadlines. No approval budget is reset.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from multiscale import historical_continuation as hc

EXTRA_SLOTS = ((1, 3), (5, 7))
SLOTS = hc.SLOTS + EXTRA_SLOTS


def context(amendment_path):
    amendment_path = Path(amendment_path).resolve()
    a = hc.read(amendment_path)
    hc.require(a["status"] == "approved_by_user_concurrency_only" and
               a["evidence"]["kind"] == "direct_user_reply" and
               a["evidence"]["text"] == "8个吧降低一些" and
               a["maximum_workers"] == 8 and a["threads_per_worker"] == 2 and
               a["slots"] == [list(s) for s in SLOTS] and a["gpu_training"] is False and
               a["additional_wall_seconds_granted"] == 0 and
               a["script_sha256"] == hc.digest(__file__), "Concurrency amendment is missing or differs")
    run = Path(a["run"])
    state = hc.read(run / "STATE.json")
    hc.require(state["boot_id"] == a["boot_id"] == hc.boot_id() and
               state["deadline_epoch"] == a["deadline_epoch"] and
               state["deadline_boottime"] == a["deadline_boottime"] and
               state["manifest_sha256"] == a["original_manifest_sha256"] and
               state["approval_sha256"] == a["original_approval_sha256"], "Existing budget or binding changed")
    manifest = hc.read(state["manifest"])
    hc.require(hc.digest(state["manifest"]) == state["manifest_sha256"] and
               hc.digest(run / "APPROVAL.json") == state["approval_sha256"], "Original approval or manifest changed")
    hc.validate_manifest(manifest)
    hc.approval(run / "APPROVAL.json", state["manifest"])
    hc.require(hc.digest(a["original_approval_used_marker"]) == a["original_approval_used_marker_sha256"] and
               hc.read(a["original_approval_used_marker"])["run"] == a["budget_origin_run"],
               "Consumed original approval marker changed")
    hc.require(hc.digest(Path(a["budget_origin_run"])/"STOP_RECEIPT.json") == a["source_pause_receipt_sha256"],
               "Verified preservation lineage changed")
    return a, run, manifest


def validate_active(state, amendment_path):
    hc.require(state.get("resource_amendment") == str(Path(amendment_path).resolve()) and
               state.get("resource_amendment_sha256") == hc.digest(amendment_path) and
               state["maximum_workers"] == 8 and state["effective_slots"] == [list(s) for s in SLOTS],
               "Effective concurrency is not the user's eight-worker amendment")
    hc.guard_ready(state)


def selections(manifest, state):
    active = [v for v in state["jobs"].values() if v["phase"] == "running"]
    used = {tuple(v["slot"]) for v in active}
    hc.require(len(used) == len(active) and len(active) <= 8 and used.issubset(set(SLOTS)),
               "Active workers overlap or exceed the amended limit")
    available = [slot for slot in SLOTS if slot not in used]
    selected = []
    for job in manifest["jobs"]:
        entry = state["jobs"][job["key"]]
        if entry["phase"] not in ("pending", "frozen") or len(active)+len(selected) >= 8:
            continue
        preferred = tuple(entry["slot"]) if entry.get("process") and entry.get("slot") else None
        slot = preferred if preferred in available else (available[0] if available and preferred is None else None)
        if slot is not None:
            selected.append((job, slot))
            available.remove(slot)
    return selected


def release(run, job, slot, amendment_path):
    with hc.locked(run):
        state = hc.read(run / "STATE.json")
        validate_active(state, amendment_path)
        entry = state["jobs"][job["key"]]
        record = entry.get("process")
        current = hc.matched(record)
        if not current or current["state"] in hc.DEAD:
            log = (run / (job["key"] + "_parallel8.log")).open("x", encoding="utf8")
            try:
                child = subprocess.Popen([str(hc.PROJECT / ".venv/bin/python"), str(Path(__file__).resolve()),
                    "worker", "--amendment", str(amendment_path), "--job", job["key"],
                    "--cores", ",".join(map(str, slot))], cwd=hc.PROJECT,
                    env=hc.bounded_environment(slot), stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
            finally:
                log.close()
            until = hc.boot_time()+5
            while True:
                record = hc.process(child.pid)
                if record and record["state"] in ("T", "t"):
                    break
                hc.require(child.poll() is None and hc.boot_time() < until,
                           "New worker failed before its stopped launch gate")
                time.sleep(0.02)
            entry["mode"] = "checkpoint_fallback" if (Path(job["folder"])/"fold_state.json").exists() else "fresh_original_fold"
        else:
            hc.require(current["state"] in ("T", "t"), "Matching worker is already active")
            entry["mode"] = "live_memory"
        hc.check_affinity(record, slot)
        entry.update(phase="running", process=record, slot=list(slot))
        state.update(phase="running", heartbeat_boottime=hc.boot_time())
        hc.save(run / "STATE.json", state)
        validate_active(state, amendment_path)
        hc.require(hc.safe_signal(record, signal.SIGCONT), "Worker exited before its registered release")


def worker(amendment_path, key, slot):
    hc.require(slot in SLOTS, "Worker slot exceeds the eight-worker allocation")
    os.sched_setaffinity(0, set(slot))
    os.kill(os.getpid(), signal.SIGSTOP)
    _, run, manifest = context(amendment_path)
    state = hc.read(run / "STATE.json")
    validate_active(state, amendment_path)
    entry = state["jobs"][key]
    hc.require(entry["phase"] == "running" and hc.matched(entry["process"]) and
               entry["process"]["pid"] == os.getpid() and entry["slot"] == list(slot),
               "Worker launch gate was not registered")
    job = next(j for j in manifest["jobs"] if j["key"] == key)
    bars, fold, actions, cfg, _ = hc.worker_context(job)
    from multiscale.development import run_development_fold
    from multiscale.verify_development import check_fold
    folder = Path(job["folder"])
    run_development_fold(bars, fold, actions, cfg, folder, job["data_identity"], device="cpu",
                         resume=(folder / "fold_state.json").is_file())
    verification = check_fold(folder)
    hc.require(all(verification["current_source_matches_recorded"].values()), "Training source changed")
    hc.save(run / "verifications" / (key+".json"), verification)


def initialize(amendment_path, amendment, run):
    unit = hc.check_controller_service()
    with hc.locked(run):
        state = hc.read(run / "STATE.json")
        hc.require(state["phase"] == "prepared_budget_continuation" and state["controller"] is None,
                   "Recovery run is already owned or no longer prepared")
        hc.require(time.time() < state["deadline_epoch"] and hc.boot_time() < state["deadline_boottime"],
                   "Original approval deadline has passed")
        for entry in state["jobs"].values():
            current = hc.matched(entry.get("process"))
            if entry["phase"] == "frozen":
                hc.require(current and current["state"] in ("T", "t"), "Original worker is not safely frozen")
                hc.check_affinity(entry["process"], entry["slot"])
        state.update(phase="armed", controller=hc.process(os.getpid()), controller_unit=unit,
                     heartbeat_boottime=hc.boot_time(), watchdog=None)
        hc.save(run / "STATE.json", state)
    return state


def controller(amendment_path):
    amendment, run, manifest = context(amendment_path)
    hc.require(os.sched_getaffinity(0) == {24, 26}, "Controller must use the analysis/control CPUs")
    owned = False
    cancelled = False
    def cancel(signum, frame):
        nonlocal cancelled
        cancelled = True
    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    try:
        # Claim the global lease before changing ownership. The failed attempt's
        # controller has already exited and all six original workers are frozen.
        with (run.parent / "active.lock").open("a+") as global_lock:
            fcntl.flock(global_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            initialize(amendment_path, amendment, run)
            owned = True
            hc.install_watchdog(run)
            hc.save(amendment_path.parent / "RECOVERY_HANDOFF_RECEIPT.json", {
                "status": "eight_worker_recovery_controller_and_independent_watchdog_ready",
                "at_utc": datetime.now(timezone.utc).isoformat(), "run": str(run),
                "budget_origin_run": amendment["budget_origin_run"],
                "original_deadline_epoch": amendment["deadline_epoch"],
                "original_deadline_boottime": amendment["deadline_boottime"],
                "additional_wall_seconds": 0, "original_approval_reconsumed": False,
                "source_pause_receipt_sha256": amendment["source_pause_receipt_sha256"],
                "Q3_values_read": False})
            while True:
                with hc.locked(run):
                    state = hc.read(run / "STATE.json")
                    if state["phase"] == "paused_preserved":
                        return
                    validate_active(state, amendment_path)
                    state["heartbeat_boottime"] = hc.boot_time()
                    for job in manifest["jobs"]:
                        entry = state["jobs"][job["key"]]
                        current = hc.matched(entry.get("process"))
                        if entry["phase"] == "running" and (not current or current["state"] in hc.DEAD):
                            evidence = hc.completed(job, run / "verifications" / (job["key"]+".json"))
                            hc.require(evidence, "Worker exited without verified completion: "+job["key"])
                            hc.update_completed(job, evidence, run)
                            entry.update(phase="completed", evidence=evidence)
                    if all(v["phase"] == "completed" for v in state["jobs"].values()):
                        state["phase"] = "completed"
                        hc.save(run / "STATE.json", state)
                        return
                    hc.save(run / "STATE.json", state)
                if cancelled or (run / "CANCEL").exists() or hc.memory_available() < 2*1024**3:
                    hc.stop_run(run, manifest, "user_cancelled" if cancelled or (run/"CANCEL").exists() else "low_available_memory")
                    return
                for job, slot in selections(manifest, state):
                    if hc.memory_available() < 4*1024**3:
                        break
                    release(run, job, slot, amendment_path)
                time.sleep(0.5)
    except BaseException:
        if owned:
            hc.stop_run(run, manifest, hc.deadline_reason(hc.read(run/"STATE.json")) or "amended_controller_error")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("controller", "worker", "preflight"))
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--job")
    parser.add_argument("--cores")
    args = parser.parse_args()
    path = args.amendment.resolve()
    if args.mode == "worker":
        worker(path, args.job, tuple(map(int, args.cores.split(","))))
    elif args.mode == "controller":
        controller(path)
    else:
        amendment, run, manifest = context(path)
        state = hc.read(run / "STATE.json")
        frozen = [v for v in state["jobs"].values() if v["phase"] == "frozen"]
        hc.require(state["phase"] == "prepared_budget_continuation" and state["controller"] is None and
                   len(frozen) == 6 and len(set(c for s in SLOTS for c in s)) == 16,
                   "Recovery preflight ownership or slots differ")
        for entry in frozen:
            current = hc.matched(entry["process"])
            hc.require(current and current["state"] in ("T", "t"), "Original live memory is not frozen")
            hc.check_affinity(entry["process"], entry["slot"])
        with (run.parent / "active.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = frozen
        print(json.dumps({"status": "eight_worker_amendment_preflight_passed", "current_workers": len(active),
                          "new_worker_capacity": 8, "logical_cpu_capacity": 16,
                          "available_memory_bytes": hc.memory_available(),
                          "deadline_epoch": state["deadline_epoch"], "new_training_started": False,
                          "signals_sent": False}))


if __name__ == "__main__":
    main()
