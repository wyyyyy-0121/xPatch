"""Check actual subprocess affinity and shared allocation, not configuration alone."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
import os
import subprocess
import sys
import threading

import pytest

from finance.runtime import ResourcePool, exclusive_run


def test_worker_and_grandchild_inherit_limit():
    cpus = sorted(os.sched_getaffinity(0))[:2]
    code = """
import json, os, subprocess, sys
from finance.runtime import configure
evidence = configure(2, json.loads(sys.argv[1]))
child = subprocess.check_output([sys.executable, '-c', 'import json,os;print(json.dumps(sorted(os.sched_getaffinity(0))))'], text=True)
evidence['grandchild_cpu_ids'] = json.loads(child)
print(json.dumps(evidence))
"""
    result = subprocess.run([sys.executable, "-c", code, json.dumps(cpus)], capture_output=True, text=True, check=True)
    evidence = json.loads(result.stdout)
    assert evidence["cpu_ids"] == evidence["grandchild_cpu_ids"] == cpus
    assert evidence["worker_threads"] == 2
    assert set(evidence["thread_environment"].values()) == {"1"}
    assert evidence["nice"] >= 10
    assert evidence["peak_rss_bytes"] > 0


def test_cpu_leases_are_disjoint_and_block_oversubscription():
    pool = ResourcePool(maximum=4, minimum_available_gib=0)
    started = threading.Event()
    def waiting_job():
        started.set()
        with pool.lease(1, "waiting", timeout_seconds=3) as ids:
            return ids
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pool.lease(2, "first") as first:
            with pool.lease(2, "second") as second:
                assert not set(first).intersection(second)
                future = executor.submit(waiting_job)
                assert started.wait(1)
                assert not future.done()
                assert pool.peak_allocated_cpus == 4
        assert len(future.result(timeout=2)) == 1
    assert len(pool.free) == 4


def test_leases_release_on_failure_and_reject_invalid_request():
    pool = ResourcePool(maximum=2, minimum_available_gib=0)
    with pytest.raises(RuntimeError):
        with pool.lease(2, "failing"):
            raise RuntimeError("worker failure")
    assert len(pool.free) == 2
    with pytest.raises(ValueError):
        with pool.lease(3, "too-large"):
            pass
    with pytest.raises(ValueError):
        ResourcePool(maximum=23)


def test_memory_pressure_wait_times_out_without_launching(monkeypatch):
    monkeypatch.setattr("finance.runtime.available_memory_bytes", lambda: 0)
    pool = ResourcePool(maximum=2, minimum_available_gib=3)
    with pytest.raises(TimeoutError):
        with pool.lease(1, "memory-blocked", timeout_seconds=0):
            pytest.fail("Job launched despite memory pressure")
    assert not pool.active


def test_os_run_lock_blocks_another_process_and_recovers(tmp_path):
    path = tmp_path / "run.lock"
    code = "from finance.runtime import exclusive_run;import sys\nwith exclusive_run(sys.argv[1]): print('acquired')"
    with exclusive_run(path):
        failed = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True, text=True)
        assert failed.returncode != 0
        assert "Another driver" in failed.stderr
    recovered = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True, text=True, check=True)
    assert recovered.stdout.strip() == "acquired"


def test_torch_uses_declared_native_pool_in_fresh_worker():
    cpus = sorted(os.sched_getaffinity(0))[:2]
    code = """
import json, sys
from finance.runtime import configure, configure_torch
configure(2, json.loads(sys.argv[1]))
evidence = configure_torch()
import torch
assert torch.isfinite(torch.ones(64, 64) @ torch.ones(64, 64)).all()
print(json.dumps(evidence))
"""
    result = subprocess.run([sys.executable, "-c", code, json.dumps(cpus)], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == {"torch_threads": 2, "torch_interop_threads": 1}


def test_full_22_cpu_mix_reaches_workers_without_overlap():
    if len(os.sched_getaffinity(0)) < 22:
        pytest.skip("Full user resource profile requires 22 available logical CPUs")
    pool = ResourcePool(maximum=22, minimum_available_gib=0)
    code = "import json,sys;from finance.runtime import configure;print(json.dumps(configure(int(sys.argv[1]),json.loads(sys.argv[2]))))"
    def launch(ids):
        output = subprocess.check_output([sys.executable, "-c", code, str(len(ids)), json.dumps(ids)], text=True)
        return json.loads(output)
    with ExitStack() as stack:
        allocations = [stack.enter_context(pool.lease(count, f"mixed-{i}")) for i, count in enumerate((4, 6, 6, 6))]
        with ThreadPoolExecutor(max_workers=4) as executor:
            evidence = list(executor.map(launch, allocations))
        actual = [cpu for item in evidence for cpu in item["cpu_ids"]]
        assert len(actual) == len(set(actual)) == 22
        assert sorted(item["worker_threads"] for item in evidence) == [4, 6, 6, 6]
        assert pool.peak_allocated_cpus == 22
    assert len(pool.free) == 22
