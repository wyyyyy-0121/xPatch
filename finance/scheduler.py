"""Four persistent workers, disjoint CPU allocations and hard per-job deadlines."""
from collections import deque
from datetime import datetime, timezone
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import time
import traceback

from .io import ROOT, read_json, sha, stable_hash, write_json
from .jobs import receipt
from .runtime import available_memory_bytes, snapshot


GPU_MODELS = {"xpatch", "chronos2"}


def worker_loop(config, cache_directory, cache_identity, cpu_ids, gpu, connection):
    from .jobs import initialize, run_job
    try:
        # A killed driver must not leave a silent numerical process running.
        import ctypes
        import signal
        if ctypes.CDLL(None).prctl(1, signal.SIGTERM) != 0 or os.getppid() == 1:
            raise RuntimeError("Could not establish worker lifetime ownership")
        initialize(config, cache_directory, cache_identity, cpu_ids, gpu)
        connection.send({"kind": "ready", "resource": snapshot()})
        while True:
            request = connection.recv()
            if request is None:
                break
            connection.send({"kind": "result", "result": run_job(request)})
    except BaseException as error:
        try:
            connection.send({"kind": "worker_error", "error": str(error), "traceback": traceback.format_exc()})
        except (BrokenPipeError, EOFError):
            pass
    finally:
        connection.close()


def failed_job(request, reason, elapsed=0., worker=None):
    directory = ROOT / request["directory"]
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "request.json", request)
    result = {"job_id": request["job_id"], "request_hash": stable_hash(request), "signature": request["signature"],
        "directory": request["directory"], "config_id": request["config_id"], "model": request["model"],
        "panel": request["panel"], "params": request["params"], "seed": request["seed"],
        "phase": request["phase"], "fold": request["partition"]["name"], "stage_epochs": request.get("max_epochs"),
        "status": "failed", "error_type": "WorkerStopped", "error": reason, "wall_seconds": elapsed,
        "allocated_cpu_seconds": elapsed * (worker["threads"] if worker else 0),
        "finished_utc": datetime.now(timezone.utc).isoformat()}
    result["files"] = {p.name: sha(p) for p in sorted(directory.iterdir()) if p.is_file() and p.name != "result.json"}
    write_json(directory / "result.json", result)
    return result


class Scheduler:
    def __init__(self, config, cache_directory, cache_identity, evidence_directory):
        self.config, self.cache_directory, self.cache_identity = config, str(cache_directory), cache_identity
        self.directory = Path(evidence_directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        settings = config["runtime"]
        ids = sorted(os.sched_getaffinity(0))[:settings["max_cpu_threads_total"]]
        cpu_count, gpu_count = settings["cpu_parallel_tasks"], settings["gpu_tasks"]
        required = cpu_count * settings["cpu_threads_per_task"] + gpu_count * settings["gpu_task_cpu_threads"]
        if required > len(ids) or required > 22 or gpu_count != 1:
            raise ValueError("Worker allocations exceed the shared 22-CPU budget")
        self.context = mp.get_context("spawn")
        self.workers = []
        offset = 0
        for index in range(cpu_count + gpu_count):
            gpu = index >= cpu_count
            threads = settings["gpu_task_cpu_threads"] if gpu else settings["cpu_threads_per_task"]
            worker = {"name": f"{'gpu' if gpu else 'cpu'}{index}", "gpu": gpu,
                "cpu_ids": ids[offset:offset + threads], "threads": threads, "active": None}
            offset += threads
            self.workers.append(worker)
        self.pending = {False: deque(), True: deque()}
        self.completed = []
        self.known = set()
        self.closed = False
        self.started = time.monotonic()
        self.previous_allocated_seconds = 0.
        self.previous_gpu_seconds = 0.
        self.charged = set()
        self.artifact_bytes = 0
        self.peak_rss_bytes = 0
        relative = self.directory.relative_to(ROOT / "artifacts/rv30")
        protocol_component = relative.parts[0]
        self.budget_root = ROOT / "artifacts/rv30" / protocol_component if len(protocol_component) == 64 else None
        if self.budget_root is not None:
            existing_jobs = set()
            for path in self.budget_root.rglob("result.json"):
                result = read_json(path)
                if result.get("signature") != protocol_component:
                    raise ValueError("Another signature appeared in this budget scope")
                self.charged.add(result["directory"])
                existing_jobs.add(result["job_id"])
                self.previous_allocated_seconds += result.get("allocated_cpu_seconds", 0.)
                if result["model"] in GPU_MODELS:
                    self.previous_gpu_seconds += result.get("wall_seconds", 0.)
            # Charge unfinished work from an interrupted prior driver exactly once.
            charges_path = self.budget_root / "interrupted_resource_charges.json"
            charges = read_json(charges_path) if charges_path.exists() else {}
            old_status = self.directory / "scheduler.json"
            if old_status.exists():
                previous = read_json(old_status)
                for worker in previous["workers"]:
                    if worker["job_id"] and worker["job_id"] not in existing_jobs:
                        key = stable_hash([previous["pid"], previous["updated_utc"], worker["job_id"]])
                        elapsed = worker.get("job_elapsed_seconds") or 0.
                        charges.setdefault(key, {"allocated_cpu_seconds": elapsed * len(worker["cpu_ids"]),
                            "gpu_wall_seconds": elapsed if worker["gpu"] else 0., "job_id": worker["job_id"]})
            write_json(charges_path, charges)
            self.previous_allocated_seconds += sum(r["allocated_cpu_seconds"] for r in charges.values())
            self.previous_gpu_seconds += sum(r["gpu_wall_seconds"] for r in charges.values())
            self.artifact_bytes = sum(p.stat().st_size for p in self.budget_root.rglob("*") if p.is_file())
        self.budget_exhausted = False
        self.peak_allocated_cpus = 0
        for worker in self.workers:
            self._start(worker)
        self._save_status()

    def _start(self, worker):
        parent, child = self.context.Pipe()
        process = self.context.Process(target=worker_loop, args=(self.config, self.cache_directory,
            self.cache_identity, worker["cpu_ids"], worker["gpu"], child), daemon=False)
        process.start()
        child.close()
        worker.update(process=process, connection=parent, ready=False, startup=time.monotonic(), active=None)

    def submit(self, requests):
        for request in requests:
            identity = request["job_id"]
            if identity in self.known:
                raise ValueError(f"Duplicate scheduler job: {identity}")
            self.known.add(identity)
            previous = receipt(ROOT / request["directory"], stable_hash(request))
            if previous is not None:
                self.completed.append({**previous, "reused": True})
                if previous["directory"] not in self.charged:
                    self.previous_allocated_seconds += previous.get("allocated_cpu_seconds", 0.)
                    if request["model"] in GPU_MODELS:
                        self.previous_gpu_seconds += previous.get("wall_seconds", 0.)
                    self.charged.add(previous["directory"])
            else:
                self.pending[request["model"] in GPU_MODELS].append(request)

    def _usage(self):
        active = [worker for worker in self.workers if worker["active"]]
        return (self.previous_allocated_seconds + sum((time.monotonic() - w["job_start"]) * w["threads"] for w in active),
            self.previous_gpu_seconds + sum(time.monotonic() - w["job_start"] for w in active if w["gpu"]))

    def _save_status(self):
        allocated, gpu = self._usage()
        write_json(self.directory / "scheduler.json", {"pid": os.getpid(), "updated_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds_this_session": time.monotonic() - self.started, "allocated_cpu_core_hours": allocated / 3600,
            "gpu_wall_hours": gpu / 3600, "budget_exhausted": self.budget_exhausted,
            "pending_cpu": len(self.pending[False]), "pending_gpu": len(self.pending[True]),
            "jobs_known": len(self.known), "peak_allocated_cpus": self.peak_allocated_cpus,
            "peak_worker_rss_bytes": self.peak_rss_bytes, "trial_artifact_bytes": self.artifact_bytes,
            "workers": [{"name": w["name"], "pid": w["process"].pid, "gpu": w["gpu"], "cpu_ids": w["cpu_ids"],
                "ready": w["ready"], "job_id": w["active"]["job_id"] if w["active"] else None,
                "job_elapsed_seconds": time.monotonic() - w["job_start"] if w["active"] else None}
                for w in self.workers]})

    def _stop(self, worker):
        process = worker["process"]
        if process.is_alive():
            process.terminate()
            process.join(3)
        if process.is_alive():
            process.kill()
            process.join(3)
        worker["connection"].close()

    def _finish(self, worker, result):
        elapsed = time.monotonic() - worker["job_start"]
        result["allocated_cpu_seconds"] = elapsed * worker["threads"]
        # Charge wall time, including idle Python and GPU waits, conservatively.
        self.previous_allocated_seconds += result["allocated_cpu_seconds"]
        if worker["gpu"]:
            self.previous_gpu_seconds += elapsed
        write_json(ROOT / result["directory"] / "result.json", result)
        self.completed.append(result)
        self.charged.add(result["directory"])
        self.artifact_bytes += sum(p.stat().st_size for p in (ROOT / result["directory"]).iterdir() if p.is_file())
        worker["active"] = None

    def poll(self, seconds=1):
        """Local process polling consumes no assistant turns or model tokens."""
        if self.closed:
            raise RuntimeError("Scheduler is closed")
        settings = self.config["runtime"]
        allocated, gpu = self._usage()
        rss_bytes = 0
        rss_by_pid = {}
        for pid in [os.getpid(), *(w["process"].pid for w in self.workers)]:
            try:
                for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        size = int(line.split()[1]) * 1024
                        rss_bytes += size
                        rss_by_pid[pid] = size
            except FileNotFoundError:
                pass
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss_bytes)
        if (rss_bytes >= settings["memory_budget_gib"] * 1024 ** 3 and
                not any(w["active"] for w in self.workers) and any(self.pending.values())):
            # Reclaim idle numerical caches instead of waiting forever for memory.
            idle = max((w for w in self.workers if w["ready"]),
                key=lambda w: rss_by_pid.get(w["process"].pid, 0), default=None)
            if idle is not None:
                self._stop(idle)
                self._start(idle)
        if ((settings["total_cpu_core_hours"] is not None and allocated >= settings["total_cpu_core_hours"] * 3600) or
                (settings["total_gpu_hours"] is not None and gpu >= settings["total_gpu_hours"] * 3600) or
                self.artifact_bytes >= settings["max_artifact_gib"] * 1024 ** 3):
            self.budget_exhausted = True
        connections = [w["connection"] for w in self.workers if w["process"].is_alive()]
        ready = wait(connections, timeout=seconds) if connections else []
        for worker in self.workers:
            request = worker["active"]
            if request is None and worker["ready"] and not worker["process"].is_alive():
                self._stop(worker)
                if not self.budget_exhausted:
                    self._start(worker)
            if worker["connection"] in ready:
                try:
                    message = worker["connection"].recv()
                except (EOFError, OSError):
                    message = {"kind": "worker_error", "error": "Worker pipe closed"}
                if message["kind"] == "ready":
                    worker["ready"] = True
                    write_json(self.directory / f"{worker['name']}_resources.json", message["resource"])
                elif message["kind"] == "result":
                    if request is None or message["result"]["job_id"] != request["job_id"]:
                        raise RuntimeError("Worker response has another job identity")
                    self._finish(worker, message["result"])
                else:
                    if request is None:
                        self.close()
                        raise RuntimeError(f"Numerical worker failed at startup: {message.get('error')}")
                    result = failed_job(request, message.get("error", "Worker failed"), time.monotonic() - worker["job_start"], worker)
                    self._finish(worker, result)
                    self._stop(worker)
                    self._start(worker)
            request = worker["active"]
            if request is not None:
                timeout = settings["task_timeout_seconds"]
                timeout = timeout[request["model"]] if isinstance(timeout, dict) else timeout
                reason = "Total predeclared compute budget exhausted" if self.budget_exhausted else None
                if timeout is not None and time.monotonic() - worker["job_start"] > timeout:
                    reason = "Predeclared job timeout"
                if not worker["process"].is_alive():
                    reason = "Numerical worker exited"
                if reason:
                    self._stop(worker)
                    result = failed_job(request, reason, time.monotonic() - worker["job_start"], worker)
                    self._finish(worker, result)
                    if not self.budget_exhausted:
                        self._start(worker)
            elif not worker["ready"] and time.monotonic() - worker["startup"] > 180:
                self.close()
                raise RuntimeError("Worker initialization exceeded 180 seconds")
            if worker["ready"] and worker["active"] is None and not self.budget_exhausted and self.pending[worker["gpu"]]:
                if (available_memory_bytes() >= settings["minimum_available_memory_gib"] * 1024 ** 3 and
                        rss_bytes < settings["memory_budget_gib"] * 1024 ** 3):
                    request = self.pending[worker["gpu"]].popleft()
                    worker["connection"].send(request)
                    worker.update(active=request, job_start=time.monotonic())
        active_cpus = sum(w["threads"] for w in self.workers if w["active"])
        self.peak_allocated_cpus = max(self.peak_allocated_cpus, active_cpus)
        if self.budget_exhausted:
            for queue in self.pending.values():
                while queue:
                    self.completed.append(failed_job(queue.popleft(), "Not started: predeclared compute budget exhausted"))
        self._save_status()
        results, self.completed = self.completed, []
        return results

    def map(self, requests):
        requests = list(requests)
        self.submit(requests)
        results = []
        while len(results) < len(requests):
            results.extend(self.poll())
        return results

    def close(self):
        if self.closed:
            return
        self.closed = True
        for worker in self.workers:
            if worker["process"].is_alive() and worker["active"] is None:
                try:
                    worker["connection"].send(None)
                    worker["process"].join(3)
                except (BrokenPipeError, OSError):
                    pass
            self._stop(worker)
        self._save_status()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
