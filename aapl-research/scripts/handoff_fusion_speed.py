"""Transfer a completed cloud stage to the verified fast entry, under one lock."""
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time

import psutil

from fusion.next.common import OUT, ROOT, digest, identity, read_json, source_identity, utc_now, write_json


REQUEST = OUT/'cloud/speed_handoff_request.json'
STATE = OUT/'cloud/speed_handoff_state.json'


def alive(row):
    try:
        p = psutil.Process(row['pid'])
        return (abs(p.create_time()-row['created_unix']) < .01
                and p.is_running() and p.status() != psutil.STATUS_ZOMBIE)
    except psutil.NoSuchProcess:
        return False


def check_request(request):
    if request['identity'] != identity({k:v for k,v in request.items() if k != 'identity'}):
        raise ValueError('Handoff request changed')
    if request['stage_receipt'] != 'F4_complete.json':
        raise ValueError('Only the explicitly requested F4 boundary is admitted')
    if request['source_identity'] != source_identity():
        raise ValueError('Handoff base engine changed')
    if digest(ROOT/'scripts/handoff_fusion_speed.py') != request['controller_sha256']:
        raise ValueError('Handoff controller changed')
    limit = read_json(OUT/'cloud/hard_time_limit.json')
    if limit != request['limit'] or limit['total_wall_hours'] != 100:
        raise ValueError('Handoff hard limit changed')
    if (datetime.fromisoformat(limit['deadline_utc'])-datetime.now(timezone.utc)).total_seconds() <= 600:
        raise TimeoutError('No handoff in the last checkpoint grace interval')
    from scripts.future_speed_runtime import checked_policy
    if checked_policy()['identity'] != request['speed_policy_identity']:
        raise ValueError('Handoff speed policy changed')
    current = read_json(OUT/'cloud/bounded_run_dispatch.json')
    if current != request['old_dispatch']:
        raise ValueError('Handoff original execution identity changed')
    if not alive(request['limiter']):
        raise RuntimeError('Original hard deadline limiter is not alive')
    return limit


def complete_boundary(request):
    path = OUT/request['stage_receipt']
    if not path.exists():
        return False
    row = read_json(path)
    expected = {'source_identity': request['source_identity'],
                'scope_identity': request['old_dispatch']['scope_identity'],
                'numerical_policy_identity': request['numerical_policy_identity'],
                'evaluation_precision': 'float32', 'original_complete_scope_claimed': False}
    if any(row.get(k) != v for k,v in expected.items()):
        raise ValueError('Completed stage does not belong to the bound experiment')
    if row['scope_amendment_sha256'] != digest(OUT/'performance/scope_100h.json'):
        raise ValueError('Completed stage scope receipt differs')
    if not (OUT/'selections/fusion.json').exists():
        raise ValueError('F4 completion lacks a committed fusion selection')
    return True


def stop_exact(row):
    """SIGINT uses the trainer's existing atomic checkpoints, never SIGKILL."""
    if not alive(row):
        raise RuntimeError('Old runner disappeared before the planned stop')
    p = psutil.Process(row['pid'])
    if Path(p.cwd()).resolve() != ROOT.resolve() or 'scripts/run_fusion_100h.py' not in p.cmdline():
        raise ValueError('Refuse to signal an unrelated process')
    p.send_signal(signal.SIGINT)


def run():
    if os.environ.get('FUSION_EXECUTION_PROFILE') != 'autodl':
        raise RuntimeError('Handoff controller is cloud-only')
    request = read_json(REQUEST)
    check_request(request)
    lock = (OUT/'cloud/speed_handoff.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('A handoff controller is already active') from None
    p = psutil.Process()
    def state(phase, **fields):
        write_json(STATE, {'phase': phase, 'request_identity': request['identity'],
                   'controller_pid': p.pid, 'controller_created_unix': p.create_time(),
                   'at_utc': utc_now(), **fields})
    if STATE.exists() and read_json(STATE).get('phase') == 'complete':
        raise RuntimeError('Handoff already completed; do not repeat it')
    state('waiting_for_stage')
    try:
        while not complete_boundary(request):
            check_request(request)
            if not alive(request['old_dispatch']):
                raise RuntimeError('Old runner exited before the requested boundary')
            state('waiting_for_stage')
            time.sleep(1)
        # Publish intent before signaling so the local observer accepts only this
        # explicit bounded transition, instead of treating arbitrary exits as OK.
        boundary_sha = digest(OUT/request['stage_receipt'])
        state('stopping_old_runner', boundary_sha256=boundary_sha)
        stop_exact(request['old_dispatch'])
        deadline = time.monotonic()+60
        while alive(request['old_dispatch']) and time.monotonic() < deadline:
            time.sleep(.2)
        if alive(request['old_dispatch']):
            raise RuntimeError('Old runner did not exit; no second training process started')
        check_request(request)
        if not complete_boundary(request) or digest(OUT/request['stage_receipt']) != boundary_sha:
            raise ValueError('Completed boundary changed during handoff')
        # Independently prove that the original execution lock is available.
        gate = (OUT/'cloud/bounded_run.lock').open('a')
        try:
            fcntl.flock(gate, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Old execution lock is still held; do not launch') from None
        fcntl.flock(gate, fcntl.LOCK_UN)
        gate.close()
        state('starting_fast_runner', boundary_sha256=boundary_sha)
        command = ['taskset','-c',request['cpu_ids'],'bash','cloud/run.sh',
                   'scripts/start_fast_fusion.py','scripts/run_fusion_100h.py']
        log = OUT/'cloud/logs/bounded_run_fast_handoff.log'
        with log.open('ab') as output:
            child = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                     start_new_session=True,
                                     env={**os.environ,'AAPL_CPU_THREADS':str(request['cpu_threads'])})
        created = psutil.Process(child.pid).create_time()
        dispatch = {**request['old_dispatch'], 'pid':child.pid,'created_unix':created,
                    'command':command, 'started_at_utc':utc_now(),
                    'previous_dispatch':request['old_dispatch'],
                    'speed_policy_identity':request['speed_policy_identity'],
                    'handoff_request_identity':request['identity']}
        write_json(OUT/'cloud/bounded_run_dispatch.json', dispatch)
        # The limiter discovers the unchanged runner filename/cwd. Require a
        # real running stage and activation receipt before declaring success.
        deadline = time.monotonic()+90
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError('Fast runner failed during startup; inspect its preserved log')
            status = read_json(OUT/'cloud/bounded_run_state.json')
            activated = list((OUT/'performance/speed_activations').rglob('*.json'))
            if status['stage'] in ('F5_running','system_freeze','F6_running','F6_complete_return_ready') and activated:
                state('complete', new_dispatch=dispatch, boundary_sha256=boundary_sha,
                      speed_activation_sha256=digest(activated[0]),
                      old_runner_confirmed_exited=True, hard_deadline_unchanged=True)
                return
            time.sleep(.5)
        raise TimeoutError('Fast startup was not verified; do not start another process')
    except BaseException as error:
        state('failed', error_type=type(error).__name__, error=str(error))
        raise


if __name__ == '__main__':
    run()
