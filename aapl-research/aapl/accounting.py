"""Physical single-GPU busy time, bounded wall time, and local artifact accounting."""
from contextlib import contextmanager
from datetime import datetime, timezone
import os
import signal
import threading
import time

import psutil

from .common import ROOT, read_json, utc_now, write_json

FORMAL = ROOT / 'outputs/formal'


@contextmanager
def charge(directory, model):
    path = directory / 'execution_charge.json'
    # Every invocation is retained, including a resumed or failed invocation.
    old = read_json(path) if path.exists() else {'attempts': []}
    attempt = {'start_epoch': time.time(), 'start_at_utc': utc_now(), 'pid': os.getpid(),
               'gpu': model in ('xpatch', 'patchtst', 'chronos2_frozen', 'chronos2_lora'), 'active': True}
    old['attempts'].append(attempt)
    write_json(path, old)
    try:
        yield
        attempt['passed'] = True
    except BaseException as exc:
        attempt.update(passed=False, error_type=type(exc).__name__)
        raise
    finally:
        attempt.update(active=False, end_epoch=time.time(), end_at_utc=utc_now())
        write_json(path, old)


def usage():
    now = time.time()
    intervals, job_seconds = [], 0.0
    for path in FORMAL.rglob('execution_charge.json'):
        for row in read_json(path)['attempts']:
            end = row.get('end_epoch', now)
            if row['gpu']:
                intervals.append((row['start_epoch'], end))
                job_seconds += end - row['start_epoch']
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    gpu_seconds = sum(b - a for a, b in merged)
    state = read_json(FORMAL / 'run.json')
    wall_seconds = now - state['started_epoch']
    size = sum(p.stat().st_size for p in (ROOT / 'outputs').rglob('*') if p.is_file())
    return {'physical_gpu_busy_hours': gpu_seconds / 3600, 'sum_gpu_job_hours': job_seconds / 3600,
            'wall_hours': wall_seconds / 3600, 'output_artifact_gib': size / 2**30,
            'accounting': 'Union of job intervals on the single shared device; sum of overlapping job time reported separately'}


class BudgetGuard:
    def __enter__(self):
        self.stop = threading.Event()
        self.violation = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def _run(self):
        while not self.stop.wait(10):
            try:
                row = usage()
                write_json(FORMAL / 'usage.json', row)
                if row['physical_gpu_busy_hours'] > 24 or row['wall_hours'] > 48 or row['output_artifact_gib'] > 20:
                    self.violation = row
                    write_json(FORMAL / 'budget_failure.json', {'failed_at_utc': utc_now(), 'usage': row})
                    # Only subprocesses owned by this executor; the legacy observer is unrelated.
                    for process in psutil.Process().children(recursive=True):
                        try:
                            process.terminate()
                        except psutil.Error:
                            pass
                    return
            except Exception as exc:
                self.violation = {'telemetry_error': repr(exc)}
                write_json(FORMAL / 'budget_failure.json', self.violation)
                for process in psutil.Process().children(recursive=True):
                    try:
                        process.terminate()
                    except psutil.Error:
                        pass
                return

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=12)
        write_json(FORMAL / 'usage.json', usage())
        if self.violation and not args[0]:
            raise RuntimeError('Formal resource budget exceeded; children stopped and evidence preserved')
