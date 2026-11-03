"""Enforce a shared 22-logical-CPU ceiling before loading numerical libraries."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import resource
import threading
import time


MAX_CPUS = 22
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


def available_memory_bytes():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Cannot establish available memory")


def configure(threads, cpu_ids=None, nice_increment=10):
    """Narrow affinity and thread pools; never enlarge a parent's CPU allocation."""
    allowed = sorted(os.sched_getaffinity(0))
    if not isinstance(threads, int) or not 1 <= threads <= MAX_CPUS:
        raise ValueError("Worker thread budget must be between 1 and 22")
    chosen = list(cpu_ids) if cpu_ids is not None else allowed[:MAX_CPUS]
    if not chosen or len(chosen) != len(set(chosen)) or len(chosen) > MAX_CPUS:
        raise ValueError("Invalid CPU allocation; global maximum is 22")
    if not set(chosen).issubset(allowed) or threads > len(chosen):
        raise ValueError("Worker cannot exceed its inherited CPU allocation")
    if not 0 <= nice_increment <= 19:
        raise ValueError("Only nonnegative priority adjustments are allowed")
    os.sched_setaffinity(0, chosen)
    for name in THREAD_ENV:
        os.environ[name] = "1"
    os.environ["XPATCH_WORKER_THREADS"] = str(threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
    os.environ["MALLOC_ARENA_MAX"] = "2"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    current_nice = os.getpriority(os.PRIO_PROCESS, 0)
    if nice_increment and current_nice < nice_increment:
        os.nice(nice_increment - current_nice)
    return snapshot()


def configure_torch():
    """Call in a fresh worker after configure(), before any tensor computation."""
    import torch
    threads = int(os.environ["XPATCH_WORKER_THREADS"])
    if threads > len(os.sched_getaffinity(0)):
        raise RuntimeError("Torch allocation differs from inherited CPU allocation")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    return {"torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads()}


def snapshot():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {"pid": os.getpid(), "cpu_ids": sorted(os.sched_getaffinity(0)),
        "worker_threads": int(os.environ.get("XPATCH_WORKER_THREADS", "1")),
        "thread_environment": {key: os.environ.get(key) for key in THREAD_ENV},
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "available_memory_bytes": available_memory_bytes(),
        "peak_rss_bytes": usage.ru_maxrss * 1024,
        "cpu_seconds": usage.ru_utime + usage.ru_stime}


class ResourcePool:
    """A driver allocates disjoint CPU IDs to all of its numerical subprocesses."""
    def __init__(self, maximum=MAX_CPUS, minimum_available_gib=3):
        if not 1 <= maximum <= MAX_CPUS:
            raise ValueError("Global CPU budget must not exceed 22")
        self.cpu_ids = tuple(sorted(os.sched_getaffinity(0))[:maximum])
        self.free = set(self.cpu_ids)
        self.minimum_available_bytes = int(minimum_available_gib * 1024 ** 3)
        self.condition = threading.Condition()
        self.active = {}
        self.pending = set()
        self.peak_allocated_cpus = 0

    @contextmanager
    def lease(self, count, job_id, timeout_seconds=3600):
        if not 1 <= count <= len(self.cpu_ids):
            raise ValueError("Requested job exceeds global CPU allocation")
        deadline = time.monotonic() + timeout_seconds
        with self.condition:
            if job_id in self.active or job_id in self.pending:
                raise ValueError("Duplicate active job identity")
            self.pending.add(job_id)
            try:
                while len(self.free) < count or available_memory_bytes() < self.minimum_available_bytes:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Resource wait exceeded its declared budget")
                    self.condition.wait(timeout=min(remaining, 1.0))
            finally:
                self.pending.discard(job_id)
            chosen = tuple(sorted(self.free)[:count])
            self.free.difference_update(chosen)
            self.active[job_id] = chosen
            self.peak_allocated_cpus = max(self.peak_allocated_cpus, len(self.cpu_ids) - len(self.free))
        try:
            yield chosen
        finally:
            with self.condition:
                self.free.update(self.active.pop(job_id))
                self.condition.notify_all()


@contextmanager
def exclusive_run(path):
    """An OS lock prevents a second driver, including when an old PID file exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another driver holds the run lock; do not start a duplicate") from None
        stream.seek(0)
        stream.truncate()
        stream.write(json.dumps({"pid": os.getpid(), "kernel_start_ticks": Path("/proc/self/stat").read_text().split()[21]}) + "\n")
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
