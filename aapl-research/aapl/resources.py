"""Process-tree resource telemetry and admission limits; no global settings."""
import os
import subprocess
import threading
import time

import psutil

ACTIVE_MONITOR = None


def check_resources():
    if ACTIVE_MONITOR is not None:
        ACTIVE_MONITOR.check()


class ResourceMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.stop_event = threading.Event()
        self.rows = []
        self.violation = None

    def sample(self):
        root = psutil.Process()
        processes = [root, *root.children(recursive=True)]
        rss, largest_process_rss, threads, cpu_seconds = 0, 0, 0, 0.0
        affinities = []
        for p in processes:
            try:
                process_rss = p.memory_info().rss
                rss += process_rss
                largest_process_rss = max(largest_process_rss, process_rss)
                threads += p.num_threads()
                ct = p.cpu_times()
                cpu_seconds += ct.user + ct.system
                affinities.append(len(p.cpu_affinity()))
            except psutil.Error:
                pass
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5)
        total, used = map(int, gpu.stdout.strip().split(","))
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        row = {"time": time.time(), "rss_gib": rss / 2**30, "available_ram_gib": mem.available / 2**30,
               'largest_process_rss_gib': largest_process_rss / 2**30,
               "swap_used_mib": swap.used / 2**20, "gpu_total_mib": total, "gpu_used_mib": used,
               "threads_including_idle": threads, "max_process_affinity_count": max(affinities),
               "cpu_seconds": cpu_seconds}
        if len(self.rows):
            previous = self.rows[-1]
            row["equivalent_active_cpu_cores"] = max(0.0, (cpu_seconds - previous["cpu_seconds"]) / (row["time"] - previous["time"]))
        self.rows.append(row)
        if largest_process_rss > 8 * 2**30 or mem.available < 4 * 2**30 or used / total > 0.85 or total - used < 1536:
            self.violation = row
        if max(affinities) > 18:
            self.violation = row

    def run(self):
        while not self.stop_event.is_set():
            try:
                self.sample()
            except Exception as exc:
                self.violation = {"telemetry_error": repr(exc)}
            self.stop_event.wait(self.interval)

    def __enter__(self):
        global ACTIVE_MONITOR
        if ACTIVE_MONITOR is not None:
            raise RuntimeError("Nested resource monitors are not supported")
        ACTIVE_MONITOR = self
        self.sample()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def check(self):
        if self.violation:
            raise RuntimeError(f"Resource admission limit exceeded: {self.violation}")

    def __exit__(self, *args):
        global ACTIVE_MONITOR
        self.stop_event.set()
        self.thread.join(timeout=6)
        self.sample()
        ACTIVE_MONITOR = None

    def summary(self):
        return {"sample_count": len(self.rows), "passed": self.violation is None,
                "violation": self.violation,
                "peak_rss_gib": max(r["rss_gib"] for r in self.rows),
                "min_available_ram_gib": min(r["available_ram_gib"] for r in self.rows),
                "peak_gpu_used_mib": max(r["gpu_used_mib"] for r in self.rows),
                "peak_threads_including_idle": max(r["threads_including_idle"] for r in self.rows),
                "max_affinity_cpus": max(r["max_process_affinity_count"] for r in self.rows),
                "peak_active_cpu_cores_interval": max(r.get("equivalent_active_cpu_cores", 0) for r in self.rows),
                "swap_growth_mib": self.rows[-1]["swap_used_mib"] - self.rows[0]["swap_used_mib"]}
