"""Standard-library deadline supervisor; preparation never starts a worker.

A real, new human-approval receipt is required for every execution window.
Synthetic acceptance and market diagnostics run serially in separate process
groups. A CPU worker is never allowed a grace period beyond its hard boundary.
This file is source preparation, not evidence of any executed acceptance.
"""
from pathlib import Path
import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time

BASE = Path(__file__).resolve().parent
SELF = BASE / 'local_probe_supervisor.py'
RUNTIME = BASE / 'local_probe_runtime.py'
ROOT = BASE.parents[3]
CPUS = {24, 26}
OLD_DEADLINE = 1794216709.0
MAXIMUM_RECEIPT_BYTES = 2 * 1024 * 1024
GRACE_SECONDS = 0.5
POLL_SECONDS = 0.1
ACTIVE_GROUP = None
HARD_EPOCH = None
HARD_MONOTONIC = None
ARM_REASON = None
STOP_REASON = None
LIBC = ctypes.CDLL(None, use_errno=True)
LIBC.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                      ctypes.c_ulong, ctypes.c_ulong]
LIBC.prctl.restype = ctypes.c_int


class BoundaryReached(RuntimeError):
    pass


def small_json(path):
    """Only bounded receipts are polled; never deserialize market artifacts."""
    with Path(path).open('rb') as handle:
        raw = handle.read(MAXIMUM_RECEIPT_BYTES + 1)
    if len(raw) > MAXIMUM_RECEIPT_BYTES:
        raise ValueError('Receipt exceeds the declared small-file limit: ' + str(path))
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as handle:
            temporary = handle.name
            json.dump(content, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def validate_basic_authorization(auth, output):
    """Check direct-human provenance fields before importing even the runtime."""
    if not (auth.get('authorized') is True
            and auth.get('direct_human_approval') is True
            and auth.get('new_window_explicitly_approved') is True
            and auth.get('thread_id') == '01a1b438-df6f-7f00-9f94-f7d72e9d59f3'
            and auth.get('scope') == '18_fit_only_local_direction_scale_units'
            and isinstance(auth.get('human_reply_text'), str)
            and auth['human_reply_text'].strip()):
        raise ValueError('A direct human approval for the new window is required')
    received = float(auth['received_epoch'])
    deadline = float(auth['deadline_epoch'])
    if not (math.isfinite(received) and math.isfinite(deadline)
            and received >= OLD_DEADLINE and deadline != OLD_DEADLINE
            and 0 < deadline - received <= 1800
            and received <= time.time() < deadline):
        raise ValueError('No unexpired, explicitly new window of at most 30 minutes')
    if not (auth.get('GPU') is False and auth.get('maximum_processes') == 1
            and auth.get('maximum_logical_CPUs') == 2
            and auth.get('paid_resources') is False):
        raise ValueError('Only one CPU worker on two logical CPUs is admitted')
    if Path(auth['output_directory']).resolve() != output.resolve():
        raise ValueError('Output differs from the approved immutable receipt')
    manifest = auth.get('source_sha256')
    required = {'local_probe_supervisor.py', 'local_probe_runtime.py',
                'torch_local_probe.py', 'local_update_probe.py'}
    if not isinstance(manifest, dict) or not required.issubset(manifest):
        raise ValueError('The full source manifest is absent')
    return deadline


def global_remaining():
    # A backwards wall-clock adjustment can never extend the original window.
    return min(HARD_EPOCH - time.time(), HARD_MONOTONIC - time.monotonic())


def send_group(group, sig):
    if group is None:
        return
    try:
        os.killpg(group, sig)
    except ProcessLookupError:
        pass


def alarm_handler(_signum, _frame):
    # This also works if a native operation has delayed the ordinary poll loop.
    send_group(ACTIVE_GROUP, signal.SIGKILL)
    raise BoundaryReached(ARM_REASON or 'absolute_authorized_deadline')


def stop_handler(signum, _frame):
    global STOP_REASON
    STOP_REASON = 'external_signal_' + str(signum)
    send_group(ACTIVE_GROUP, signal.SIGTERM)
    if ACTIVE_GROUP is None:
        raise BoundaryReached(STOP_REASON)


def arm(first_deadline=None):
    global ARM_REASON
    remaining = global_remaining()
    ARM_REASON = 'absolute_authorized_deadline'
    if first_deadline is not None:
        initial_remaining = first_deadline - time.monotonic()
        if initial_remaining < remaining:
            remaining = initial_remaining
            ARM_REASON = 'initial_market_120_second_boundary'
    if remaining <= 0:
        send_group(ACTIVE_GROUP, signal.SIGKILL)
        raise BoundaryReached(ARM_REASON)
    signal.setitimer(signal.ITIMER_REAL, remaining)


def runtime_module():
    """The reviewed runtime has only standard-library imports at module level."""
    spec = importlib.util.spec_from_file_location('authorized_local_probe_runtime', RUNTIME)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bound_receipt(path, auth_hash, manifest, accepted_status):
    receipt, _ = small_json(path)
    if (receipt.get('authorization_sha256') != auth_hash
            or receipt.get('source_sha256') != manifest):
        raise ValueError('Receipt belongs to a different approval or source: ' + str(path))
    if receipt.get('status') != accepted_status:
        raise ValueError('Required receipt has not passed: ' + str(path))
    return receipt


def admission(output, auth_hash, manifest):
    path = output / 'ADMISSION.json'
    if not path.exists():
        return False
    bound_receipt(path, auth_hash, manifest, 'admitted')
    first = output / 'units' / 'f1_s2026_q25.json'
    unit, _ = small_json(first)
    if (unit.get('id') != 'f1_s2026_q25'
            or unit.get('authorization_sha256') != auth_hash
            or unit.get('source_sha256') != manifest):
        raise ValueError('Admission lacks its same-identity completed first unit')
    return True


def stop_process_group(process, first_deadline=None):
    """TERM grace is consumed inside, never added after, the hard boundary."""
    send_group(process.pid, signal.SIGTERM)
    remaining = max(0.0, global_remaining())
    if first_deadline is not None:
        remaining = min(remaining, max(0.0, first_deadline - time.monotonic()))
    until = time.monotonic() + min(GRACE_SECONDS, remaining)
    while process.poll() is None and time.monotonic() < until:
        time.sleep(min(0.02, max(0.0, until - time.monotonic())))
    # Kill the entire original group even if its leader exited during grace.
    send_group(process.pid, signal.SIGKILL)
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        raise RuntimeError('Worker group did not reap after SIGKILL')


def worker_environment():
    environment = dict(os.environ)
    environment.update(CUDA_VISIBLE_DEVICES='-1', OPENBLAS_NUM_THREADS='1',
        OMP_NUM_THREADS='2', OMP_THREAD_LIMIT='2', MKL_NUM_THREADS='2',
        NUMEXPR_NUM_THREADS='2', AAPL_LOCAL_PROBE_SUPERVISOR_PID=str(os.getpid()))
    return environment


def parent_death_fence(expected_parent):
    """Linux direct-worker insurance if its supervisor exits without cleanup."""
    # PR_SET_PDEATHSIG=1. Configure before exec, then close the fork/parent-death
    # race by checking that the original supervising parent still exists.
    if LIBC.prctl(1, int(signal.SIGKILL), 0, 0, 0) != 0:
        os._exit(125)
    if os.getppid() != expected_parent:
        os._exit(125)


def run_worker(kind, auth_path, output, auth_hash, manifest, progress):
    global ACTIVE_GROUP
    if STOP_REASON:
        raise BoundaryReached(STOP_REASON)
    first_deadline = None
    if kind == 'market' and not admission(output, auth_hash, manifest):
        # Set before Popen: imports, source checks, data and checkpoint setup count.
        first_deadline = time.monotonic() + 120.0
    arm(first_deadline)
    command = [sys.executable, str(RUNTIME), '--worker', kind,
               '--authorization', str(auth_path), '--output', str(output)]
    process = None
    expected_parent = os.getpid()
    started = time.monotonic()
    try:
        with (output / ('supervisor_' + kind + '.log')).open('ab', buffering=0) as log:
            process = subprocess.Popen(command, cwd=ROOT, env=worker_environment(),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
                preexec_fn=lambda: parent_death_fence(expected_parent))
            ACTIVE_GROUP = process.pid
            progress('worker_started', worker=kind, worker_pid=process.pid,
                     initial_market_watchdog_active=first_deadline is not None)
            termination_reason = None
            while True:
                if STOP_REASON:
                    raise BoundaryReached(STOP_REASON)
                arm(first_deadline)
                if kind == 'market' and first_deadline is not None:
                    if admission(output, auth_hash, manifest):
                        first_deadline = None
                        arm()
                        progress('market_admission_passed', worker=kind,
                                 initial_market_watchdog_active=False)
                remaining = global_remaining()
                if first_deadline is not None:
                    remaining = min(remaining, first_deadline - time.monotonic())
                if remaining <= GRACE_SECONDS and termination_reason is None:
                    termination_reason = ARM_REASON
                    send_group(process.pid, signal.SIGTERM)
                code = process.poll()
                if code is not None:
                    if termination_reason is not None:
                        raise BoundaryReached(termination_reason)
                    if code != 0:
                        raise RuntimeError(kind + ' worker exited with code ' + str(code))
                    if kind == 'market' and first_deadline is not None:
                        # A zero exit does not transform a denied admission into success.
                        if not admission(output, auth_hash, manifest):
                            raise RuntimeError('Market worker exited without passed admission')
                    return time.monotonic() - started
                time.sleep(min(POLL_SECONDS, max(0.001, remaining - GRACE_SECONDS)))
    finally:
        # Disable the just-finished first timer while cleaning up; a kill never
        # receives extra compute time because cleanup uses its original boundary.
        signal.setitimer(signal.ITIMER_REAL, 0)
        if process is not None:
            stop_process_group(process, first_deadline)
        ACTIVE_GROUP = None
        if global_remaining() > 0:
            arm()


def run(auth_path, output):
    global HARD_EPOCH, HARD_MONOTONIC
    auth_path, output = auth_path.resolve(), output.resolve()
    auth, auth_hash = small_json(auth_path)
    HARD_EPOCH = validate_basic_authorization(auth, output)
    HARD_MONOTONIC = time.monotonic() + HARD_EPOCH - time.time()
    argv = Path('/proc/self/cmdline').read_bytes().split(b'\0')
    if str(SELF).encode() not in argv:
        raise ValueError('Invoke this supervisor using its absolute script path')
    os.sched_setaffinity(0, CPUS)
    signal.signal(signal.SIGALRM, alarm_handler)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, stop_handler)
    arm()
    output_ready = False
    try:
        # Check the runtime's own bytes before importing its stdlib-only module.
        for path in (SELF, RUNTIME):
            if file_sha(path) != auth['source_sha256'][path.name]:
                raise ValueError('Entry point differs from the newly frozen source')
        runtime = runtime_module()
        checked = runtime.authorization(auth_path, output)
        current, current_hash = small_json(auth_path)
        if checked != auth or current != auth or current_hash != auth_hash:
            raise ValueError('Authorization changed during complete source verification')
        # runtime.authorization verified the complete provenance manifest once.
        # The idle parent below polls small receipts, never large checkpoints.
        if STOP_REASON:
            raise BoundaryReached(STOP_REASON)
        arm()
        output.mkdir(parents=True, exist_ok=True)
        output_ready = True

        def progress(status, **extra):
            atomic(output / 'SUPERVISOR_STATE.json', {
                'status': status, 'authorization_sha256': auth_hash,
                'source_sha256': auth['source_sha256'],
                'deadline_epoch': HARD_EPOCH, 'supervisor_pid': os.getpid(),
                'CPU_affinity': sorted(os.sched_getaffinity(0)), 'GPU': False,
                'maximum_concurrent_workers': 1, 'completed_unit_files_preserved': True,
                'research_goal_achieved': False, 'independent_confirmation': False,
                'updated_epoch': time.time(), **extra})

        manifest = auth['source_sha256']
        prior_units = list((output / 'units').glob('*.json'))
        has_admission = admission(output, auth_hash, manifest)
        if prior_units and not has_admission:
            raise ValueError('Preserved units cannot bypass missing/failed complete-matrix admission')
        state_path = output / 'STATE.json'
        if state_path.exists():
            state, _ = small_json(state_path)
            if (state.get('status') == 'completed_all18_local_diagnostic_units'
                    and state.get('authorization_sha256') == auth_hash
                    and state.get('source_sha256') == manifest):
                progress('already_completed_no_worker_started')
                return
        progress('authorized_sources_verified_no_worker_started')
        synthetic_seconds = run_worker('synthetic', auth_path, output,
            auth_hash, manifest, progress)
        acceptance = bound_receipt(output / 'TORCH_SYNTHETIC_ACCEPTANCE.json',
                                   auth_hash, manifest, 'passed')
        if acceptance.get('synthetic_only') is not True:
            raise ValueError('Synthetic acceptance receipt is not synthetic-only')
        progress('synthetic_passed_market_not_started', synthetic_seconds=synthetic_seconds)
        market_seconds = run_worker('market', auth_path, output, auth_hash, manifest, progress)
        state = bound_receipt(output / 'STATE.json', auth_hash, manifest,
                              'completed_all18_local_diagnostic_units')
        if state.get('completed_units') != 18:
            raise ValueError('Market worker did not preserve all 18 completed units')
        progress('completed_all18_local_diagnostic_units',
                 synthetic_seconds=synthetic_seconds, market_seconds=market_seconds)
    except BaseException as exc:
        if output_ready:
            atomic(output / 'SUPERVISOR_STATE.json', {
                'status': 'stopped_existing_completed_units_preserved',
                'reason': type(exc).__name__ + ': ' + str(exc),
                'authorization_sha256': auth_hash, 'source_sha256': auth['source_sha256'],
                'deadline_epoch': HARD_EPOCH, 'GPU': False,
                'completed_unit_files_preserved': True, 'research_goal_achieved': False,
                'independent_confirmation': False, 'updated_epoch': time.time()})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--authorization', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.prepare:
        # Standard-library metadata and raw hashes only. No acceptance is run.
        print(json.dumps(runtime_module().prepare(), indent=2))
        return
    if args.authorization is None or args.output is None:
        parser.error('No workers can start without a real new approval receipt and output')
    run(args.authorization, args.output)


if __name__ == '__main__':
    main()
