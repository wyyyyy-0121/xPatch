"""Hard timeout, failure preservation and shared CPU allocation integration."""
import time

from finance import scheduler as module
from finance.io import ROOT, read_json


def slow_worker(config, cache_directory, cache_identity, cpu_ids, gpu, connection):
    from finance.runtime import configure, snapshot
    configure(len(cpu_ids), cpu_ids)
    connection.send({"kind": "ready", "resource": snapshot()})
    while True:
        request = connection.recv()
        if request is None:
            break
        time.sleep(2)


def test_hard_timeout_saves_failure_and_resume_reuses_it(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "worker_loop", slow_worker)
    config = read_json(ROOT / "configs/rv30.json")
    config["runtime"].update(cpu_parallel_tasks=1, cpu_threads_per_task=1, gpu_task_cpu_threads=1,
        task_timeout_seconds=.1, minimum_available_memory_gib=0)
    directory = ROOT / "artifacts/rv30/test_scheduler" / tmp_path.parent.name / tmp_path.name
    request = {"job_id": "deliberate_timeout", "directory": str((directory / "job").relative_to(ROOT)),
        "signature": "test_only", "config_id": "fixed", "model": "last_rv", "panel": "aapl", "params": {},
        "seed": 2026, "phase": "smoke", "partition": {"name": "timeout"}}
    with module.Scheduler(config, "unused", "unused", directory) as pool:
        results = pool.map([request])
        allocations = [set(worker["cpu_ids"]) for worker in pool.workers]
        assert not allocations[0] & allocations[1]
        assert sum(map(len, allocations)) == 2
    assert len(results) == 1 and results[0]["status"] == "failed"
    assert "timeout" in results[0]["error"]
    with module.Scheduler(config, "unused", "unused", directory) as pool:
        reused = pool.map([request])
    assert reused[0]["reused"] and reused[0]["request_hash"] == results[0]["request_hash"]
