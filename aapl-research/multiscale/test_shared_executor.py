"""Exercise failure and budget handling without market data or GPU work."""
import json
import socket

import pytest
import torch

from . import shared_executor


class _Queue:
    def get(self, timeout):
        return {"status": "completed", "receipt_sha256": "e" * 64}

    def close(self):
        pass


class _Process:
    def __init__(self, owner):
        self.owner = owner
        self.exitcode = 0

    def start(self):
        self.owner.started += 1

    def is_alive(self):
        return False

    def join(self, timeout=None):
        pass


class _Context:
    def __init__(self):
        self.started = 0

    def Queue(self, maxsize):
        return _Queue()

    def Process(self, **kwargs):
        return _Process(self)


@pytest.fixture
def queue_fixture(tmp_path, monkeypatch):
    worker = _Context()
    protocol = {"data_kind": "synthetic_test", "budget_identity": "b" * 64,
                "jobs": [{"seed": 2026, "fold": 1}], "training": {}}
    context = {"mechanism_review_hash": "c" * 64, "admission_hash": "a" * 64,
               "mechanism_review": {"compute": {"host": socket.gethostname(), "device_name": "fictional GPU",
                                                "maximum_wall_seconds": 180}}}
    monkeypatch.setattr(shared_executor, "OUT", tmp_path)
    monkeypatch.setattr(shared_executor, "load_locked", lambda *args, **kwargs: (context, protocol, "d" * 64, None))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "fictional GPU")
    monkeypatch.setattr(shared_executor.mp, "get_context", lambda name: worker)
    monkeypatch.setattr(shared_executor, "_verify_job", lambda *args: (_ for _ in ()).throw(
        ValueError("simulated parent-side artifact verification failure")))
    state_path = tmp_path / "synthetic_mechanism_validation" / "runs" / ("b" * 24) / "mechanism_state.json"
    return worker, protocol, context, state_path


def _execute():
    return shared_executor.execute("fictional admission", "fictional base review", "fictional base protocol",
                                   "fictional mechanism review", "fictional mechanism protocol", max_jobs=1)


def test_parent_verification_failure_is_recorded_and_requires_explicit_retry(queue_fixture):
    worker, _, _, state_path = queue_fixture
    with pytest.raises(ValueError, match="parent-side artifact verification"):
        _execute()
    state = json.loads(state_path.read_text(encoding="utf8"))
    record = state["jobs"]["seed2026/fold01"]
    assert state["status"] == "failed" and state["active"] is None
    assert record["status"] == record["attempts"][-1]["status"] == "failed"
    assert record["error_type"] == "ValueError" and "artifact verification" in record["error"]
    assert _execute()["status"] == "needs_explicit_retry"
    assert worker.started == 1


def test_explicit_retry_keeps_the_failed_attempt_and_uses_a_new_directory(queue_fixture, monkeypatch):
    worker, _, _, _ = queue_fixture
    with pytest.raises(ValueError, match="parent-side artifact verification"):
        _execute()
    monkeypatch.setattr(shared_executor, "_verify_job", lambda *args: {"fold_receipt_sha256": "e" * 64})
    state = shared_executor.execute("fictional admission", "fictional base review", "fictional base protocol",
                                    "fictional mechanism review", "fictional mechanism protocol",
                                    max_jobs=1, retry_incomplete=True)
    record = state["jobs"]["seed2026/fold01"]
    assert state["status"] == "completed_synthetic_test_only" and record["status"] == "completed"
    assert [attempt["status"] for attempt in record["attempts"]] == ["failed", "completed"]
    assert record["attempts"][0]["directory"] != record["attempts"][1]["directory"]
    assert worker.started == 2


def test_interrupted_downtime_is_charged_before_new_work(queue_fixture, monkeypatch):
    worker, _, _, state_path = queue_fixture
    with pytest.raises(ValueError, match="parent-side artifact verification"):
        _execute()
    state = json.loads(state_path.read_text(encoding="utf8"))
    state["consumed_wall_seconds"] = 12.0
    state["jobs"]["seed2026/fold01"]["status"] = "running"
    state["jobs"]["seed2026/fold01"]["attempts"][-1]["status"] = "running"
    state["active"] = {"job": "seed2026/fold01", "boot_id": "fictional", "started_monotonic": 1,
                       "started_wall": 1}
    state_path.write_text(json.dumps(state), encoding="utf8")
    monkeypatch.setattr(shared_executor, "_recovered_elapsed", lambda *args: 200.0)
    recovered = _execute()
    assert recovered["status"] == "budget_exhausted" and recovered["consumed_wall_seconds"] >= 212.0
    assert recovered["jobs"]["seed2026/fold01"]["status"] == "interrupted"
    assert recovered["active"] is None and worker.started == 1


def test_refreezing_cannot_reset_the_same_synthetic_budget(queue_fixture, monkeypatch):
    worker, protocol, context, _ = queue_fixture
    with pytest.raises(ValueError, match="parent-side artifact verification"):
        _execute()
    monkeypatch.setattr(shared_executor, "load_locked", lambda *args, **kwargs: (context, protocol, "f" * 64, None))
    with pytest.raises(ValueError, match="budget cannot be reset"):
        _execute()
    assert worker.started == 1


def test_orphaned_running_job_fails_closed(queue_fixture):
    worker, _, _, state_path = queue_fixture
    with pytest.raises(ValueError, match="parent-side artifact verification"):
        _execute()
    state = json.loads(state_path.read_text(encoding="utf8"))
    state["jobs"]["seed2026/fold01"]["status"] = "running"
    state_path.write_text(json.dumps(state), encoding="utf8")
    with pytest.raises(ValueError, match="no matching active budget record"):
        _execute()
    assert worker.started == 1
