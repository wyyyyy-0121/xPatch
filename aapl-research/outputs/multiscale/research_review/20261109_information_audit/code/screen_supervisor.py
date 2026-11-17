"""One CPU worker under the screen's frozen absolute deadline."""
from pathlib import Path
import argparse
import ctypes
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parents[1]
WORKER = BASE/'code/volume_screen.py'
CHILD = None
DEADLINE = None
MONOTONIC_DEADLINE = None


def remaining():
    return min(DEADLINE-time.time(), MONOTONIC_DEADLINE-time.monotonic())


def kill_group(sig):
    if CHILD is not None:
        try:
            os.killpg(CHILD.pid, sig)
        except ProcessLookupError:
            pass


def alarm(*_):
    kill_group(signal.SIGKILL)
    raise TimeoutError('Frozen absolute volume screen deadline')


def external_stop(*_):
    kill_group(signal.SIGUSR1)
    raise InterruptedError('External stop; retain sufficient statistics')


def fence(expected):
    lib = ctypes.CDLL(None, use_errno=True)
    if lib.prctl(1, int(signal.SIGKILL), 0, 0, 0) != 0 or os.getppid() != expected:
        os._exit(125)


def main():
    global CHILD, DEADLINE, MONOTONIC_DEADLINE
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); output = args.output.resolve()
    os.sched_setaffinity(0, {24, 26})
    rules = json.loads((output/'RULES.json').read_text())
    if (rules['maximum_processes'] != 1 or rules['maximum_logical_CPUs'] != 2
            or rules['GPU'] is not False or rules['output_directory'] != str(output)):
        raise ValueError('Wrong frozen resource/output identity')
    DEADLINE = float(rules['deadline_epoch'])
    if not 0 < DEADLINE-time.time() <= 7200:
        raise ValueError('Frozen screen window is expired or reset')
    MONOTONIC_DEADLINE = time.monotonic()+DEADLINE-time.time()
    signal.signal(signal.SIGALRM, alarm)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, external_stop)
    signal.setitimer(signal.ITIMER_REAL, remaining())
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES='-1', OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2',
               OMP_THREAD_LIMIT='2', MKL_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2',
               AAPL_VOLUME_SUPERVISOR_PID=str(os.getpid()))
    parent = os.getpid()
    try:
        with (output/'WORKER.log').open('ab', buffering=0) as log:
            CHILD = subprocess.Popen([sys.executable, str(WORKER), '--output', str(output)],
                cwd=BASE, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, preexec_fn=lambda: fence(parent))
            while CHILD.poll() is None:
                if remaining() <= 3:
                    kill_group(signal.SIGUSR1)
                time.sleep(min(0.1, max(0.001, remaining())))
            if CHILD.returncode:
                raise RuntimeError('Volume screen worker exited: '+str(CHILD.returncode))
            print(json.dumps({'status':'worker_completed','returncode':CHILD.returncode}))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if CHILD is not None:
            kill_group(signal.SIGUSR1)
            end = time.monotonic()+min(3, max(0, remaining()))
            while CHILD.poll() is None and time.monotonic() < end:
                time.sleep(0.02)
            kill_group(signal.SIGKILL)
            CHILD.wait(timeout=1)


if __name__ == '__main__':
    main()
