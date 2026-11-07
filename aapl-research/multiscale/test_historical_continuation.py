"""Exercise resource guards with fictional jobs and idle temporary processes."""
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from . import historical_continuation as control


def wait_for(predicate, seconds=5):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        result = predicate()
        if result:
            return result
        time.sleep(0.02)
    raise AssertionError("Temporary process did not reach expected state")


@pytest.fixture
def idle_child():
    children = []

    def launch(stopped=False):
        source = "import os,signal,time; "
        if stopped:
            source += "os.kill(os.getpid(),signal.SIGSTOP); "
        source += "time.sleep(60)"
        child = subprocess.Popen([sys.executable, "-c", source], start_new_session=True)
        children.append(child)
        wait_for(lambda: (p := control.process(child.pid)) and p["state"] == ("T" if stopped else "S"))
        return child, control.process(child.pid)

    yield launch
    for child in children:
        child.kill()
        child.wait(timeout=5)


def fake_run(tmp_path, record, *, seconds=30, controller=None):
    run = tmp_path / "run"
    run.mkdir()
    fold = tmp_path / "fictional_fold"
    fold.mkdir()
    control.save(fold / "fold_state.json", {"fictional": True, "steps": 123})
    (fold / "mlp").mkdir()
    (fold / "mlp/latest.pt").write_bytes(b"fictional checkpoint, not a model")
    manifest = {"jobs": [{"key": "fictional", "folder": str(fold)}]}
    state = {"phase": "armed", "boot_id": control.boot_id(),
             "controller": controller or control.process(os.getpid()), "manifest_sha256": "fictional",
             "heartbeat_boottime": control.boot_time(), "deadline_epoch": time.time() + seconds,
             "deadline_boottime": control.boot_time() + seconds, "watchdog": None,
             "jobs": {"fictional": {"phase": "running", "process": record, "slot": [24, 26]}}}
    control.save(run / "STATE.json", state)
    return run, manifest, state


def test_preparation_has_no_implicit_compute_approval(tmp_path):
    manifest = tmp_path / "manifest.json"
    control.save(manifest, {"fictional": True})
    with pytest.raises(ValueError, match="No additional compute approval"):
        control.approval(None, manifest)
    assert list(tmp_path.iterdir()) == [manifest]


def test_start_refuses_missing_approval_before_creating_run(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    control.save(manifest, {"fictional": True})
    monkeypatch.setattr(control, "preflight", lambda value: [])
    monkeypatch.setattr(control, "OUT", tmp_path)
    with pytest.raises(ValueError, match="No additional compute approval"):
        control.start(manifest, None)
    assert not (tmp_path / "remaining_continuations").exists()


def test_launch_requires_detached_service_with_process_only_termination(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    approval = tmp_path / "approval.json"
    control.save(manifest, {"fictional": True})
    control.save(approval, {"fictional": True})
    monkeypatch.setattr(control, "preflight", lambda value: [])
    monkeypatch.setattr(control, "approval", lambda *args: {})
    calls = []

    def capture(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(control.subprocess, "run", capture)
    result = control.launch(manifest, approval)
    assert result["controller_unit"].startswith("aapl-continuation-controller-")
    assert len(calls) == 1
    assert "--property=KillMode=process" in calls[0]
    assert "--property=Restart=no" in calls[0]
    assert "--setenv=CUDA_VISIBLE_DEVICES=-1" in calls[0]
    assert "--setenv=OMP_NUM_THREADS=2" in calls[0]


def test_plain_tool_session_cannot_start_approved_controller(monkeypatch):
    monkeypatch.delenv("AAPL_CONTINUATION_UNIT", raising=False)
    with pytest.raises(ValueError, match="independently of the tool session"):
        control.check_controller_service()


def test_worker_stops_before_reading_manifest_or_loading_training_code(tmp_path):
    absent_run = tmp_path / "not_created"
    source = ("from multiscale.historical_continuation import run_worker; "
              f"run_worker({str(absent_run)!r}, 'fictional', (24,26))")
    child = subprocess.Popen([sys.executable, "-c", source], start_new_session=True)
    try:
        wait_for(lambda: (p := control.process(child.pid)) and p["state"] == "T")
        assert child.poll() is None
        assert not absent_run.exists()
        assert os.sched_getaffinity(child.pid) == {24, 26}
    finally:
        child.kill()
        child.wait(timeout=5)


@pytest.mark.parametrize("changed", [{"maximum_wall_seconds": 36001}, {"maximum_workers": 7},
                                      {"threads_per_worker": 3}, {"gpu_training": True},
                                      {"manifest_sha256": "different"}, {"evidence": {"kind": "goal_continuation", "text": "continue"}}])
def test_approval_rejects_scope_or_budget_expansion(tmp_path, changed):
    manifest = tmp_path / "manifest.json"
    control.save(manifest, {"fictional": True})
    value = {"status": "approved_by_user", "scope": "remaining_original_core_folds_only",
             "manifest_sha256": control.digest(manifest), "maximum_wall_seconds": 36000,
             "maximum_workers": 6, "threads_per_worker": 2, "gpu_training": False,
             "evidence": {"kind": "direct_user_reply", "text": "fictional test approval"}}
    value.update(changed)
    path = tmp_path / "approval.json"
    control.save(path, value)
    with pytest.raises(ValueError, match="Invalid or excessive"):
        control.approval(path, manifest)


def test_seven_frozen_workers_share_one_slot_without_overlapping():
    slots = [control.SLOTS[2], *control.SLOTS, None]
    jobs = [{"key": str(i), "preferred_slot": list(s) if s else None} for i, s in enumerate(slots)]
    state = {"jobs": {j["key"]: {"phase": "frozen"} for j in jobs}}
    selected = control.choose_jobs({"jobs": jobs}, state, 6)
    assert len(selected) == 6
    assert len({slot for _, slot in selected}) == 6
    assert jobs[3] not in [j for j, _ in selected]  # Second occupant of 8,10 waits.
    for job, slot in selected:
        state["jobs"][job["key"]].update(phase="running", slot=list(slot))
    assert control.choose_jobs({"jobs": jobs}, state, 6) == []
    state["jobs"]["0"]["phase"] = "completed"
    assert control.choose_jobs({"jobs": jobs}, state, 6) == [(jobs[3], control.SLOTS[2])]


def test_completed_jobs_never_reenter_the_queue():
    job = {"key": "done", "preferred_slot": None}
    assert control.choose_jobs({"jobs": [job]}, {"jobs": {"done": {"phase": "completed"}}}, 6) == []


def test_pid_identity_mismatch_never_signals_reused_pid(idle_child):
    child, record = idle_child(stopped=True)
    for key, wrong in (("start_ticks", record["start_ticks"] + 1), ("boot_id", "other-boot"),
                       ("argv_sha256", "other-command")):
        replaced = dict(record, **{key: wrong})
        assert control.safe_signal(replaced, signal.SIGCONT) is False
        assert control.process(child.pid)["state"] == "T"


def test_guard_refuses_release_without_live_watchdog(tmp_path, idle_child):
    _, record = idle_child(stopped=True)
    _, _, state = fake_run(tmp_path, record)
    with pytest.raises(ValueError, match="Independent watchdog"):
        control.guard_ready(state)
    assert control.matched(record)["state"] == "T"


def test_stable_archive_detects_differing_existing_copy(tmp_path):
    source, dest = tmp_path / "source", tmp_path / "copy"
    source.mkdir()
    dest.mkdir()
    (source / "latest.pt").write_bytes(b"checkpoint")
    (dest / "latest.pt").write_bytes(b"different")
    with pytest.raises(ValueError, match="Existing preservation copy differs"):
        control.copy_stable(source, dest)


@pytest.mark.parametrize("trigger", ["budget", "controller_exit", "heartbeat", "cancel"])
def test_independent_watchdog_freezes_and_preserves(tmp_path, idle_child, trigger):
    child, record = idle_child()
    _, unrelated = idle_child(stopped=True)
    controller_child, controller_record = idle_child()
    run, manifest, state = fake_run(tmp_path, record, controller=controller_record)
    budget_file = tmp_path / "original_budget.json"
    control.save(budget_file, {"maximum_wall_seconds": 7200, "elapsed_seconds": 7200})
    before = control.digest(budget_file)
    # Fork only this small stdlib guard; both it and idle children inherit the
    # test runner's two-core affinity. No training code or market arrays open.
    guard = multiprocessing.get_context("fork").Process(target=control.watchdog_loop, args=(run, manifest))
    guard.start()
    try:
        wait_for(lambda: (run / "WATCHDOG_READY.json").exists())
        if trigger == "controller_exit":
            controller_child.kill()
            controller_child.wait(timeout=5)
        elif trigger == "cancel":
            (run / "CANCEL").touch()
        else:
            with control.locked(run):
                current = control.read(run / "STATE.json")
                if trigger == "budget":
                    current["deadline_boottime"] = control.boot_time() - 1
                else:
                    current["heartbeat_boottime"] = control.boot_time() - control.HEARTBEAT_SECONDS - 1
                control.save(run / "STATE.json", current)
        wait_for(lambda: (run / "STOP_RECEIPT.json").exists())
        guard.join(timeout=5)
        assert guard.exitcode == 0
        assert control.process(child.pid)["state"] == "T"
        assert control.matched(unrelated)["state"] == "T"
        receipt = control.read(run / "STOP_RECEIPT.json")
        expected = {"budget": "additional_budget_exhausted", "controller_exit": "controller_exited",
                    "heartbeat": "controller_heartbeat_lost", "cancel": "user_cancelled"}[trigger]
        assert receipt["reason"] == expected
        assert len(receipt["files"]) == 2
        assert all(control.digest(f["source"]) == control.digest(f["copy"]) == f["sha256"] for f in receipt["files"])
        assert control.digest(budget_file) == before
        assert control.read(run / "STATE.json")["phase"] == "paused_preserved"
    finally:
        if guard.is_alive():
            guard.kill()
            guard.join(timeout=5)


def test_completion_updates_preserve_original_budget_and_other_entries(tmp_path):
    queue = tmp_path / "queue.json"
    original = {"status": "paused_budget_exhausted", "maximum_wall_seconds": 7200, "elapsed_seconds": 7200,
                "night_pause": {"remaining_active_seconds_at_resume": 6246}, "completed": 1,
                "jobs": {"done": {"status": "completed", "receipt_sha256": "earlier"},
                         "new": {"status": "paused_budget_exhausted", "pid": 123}}}
    control.save(queue, original)
    job = {"queue": str(queue), "queue_collection": "jobs", "queue_key": "new"}
    evidence = {"receipt_sha256": "new-receipt", "verification": "/fictional/check.json",
                "verification_sha256": "new-check"}
    control.update_completed(job, evidence, tmp_path / "run")
    result = control.read(queue)
    assert result["completed"] == 2
    assert result["jobs"]["done"] == original["jobs"]["done"]
    assert result["jobs"]["new"]["receipt_sha256"] == "new-receipt"
    assert all(result[k] == original[k] for k in ("status", "maximum_wall_seconds", "elapsed_seconds", "night_pause"))


def test_guard_does_not_block_behind_a_stalled_controller_lock(tmp_path, idle_child):
    child, record = idle_child()
    controller_child, controller = idle_child()
    run, manifest, _ = fake_run(tmp_path, record, controller=controller)
    # An unrelated process holding the same flock simulates a controller
    # stuck inside its critical section. The guard cannot acquire this lock.
    source = ("import fcntl,time; "
              f"f=open({str(run / 'STATE.lock')!r},'a+'); "
              "fcntl.flock(f.fileno(),fcntl.LOCK_EX); print('locked',flush=True); time.sleep(60)")
    holder = subprocess.Popen([sys.executable, "-c", source], stdout=subprocess.PIPE, text=True)
    assert holder.stdout.readline().strip() == "locked"
    guard = multiprocessing.get_context("fork").Process(target=control.watchdog_loop, args=(run, manifest))
    guard.start()
    try:
        wait_for(lambda: (run / "STOP_RECEIPT.json").exists())
        guard.join(timeout=5)
        assert guard.exitcode == 0
        assert control.matched(record)["state"] == "T"
        assert control.read(run / "STOP_RECEIPT.json")["reason"] == "controller_lock_stalled_at_guard_start"
        assert control.read(run / "STATE.json")["phase"] == "paused_preserved"
        controller_child.wait(timeout=5)
        assert controller_child.returncode == -signal.SIGKILL
        assert holder.poll() is None  # Lock holder cannot prevent preservation.
    finally:
        holder.kill()
        holder.wait(timeout=5)
        if guard.is_alive():
            guard.kill()
            guard.join(timeout=5)
