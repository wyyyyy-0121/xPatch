"""Low-level independent process-group guard for a future admitted experiment.

This is not a training launcher or an admission check. A future launcher must
first verify the date, original completion, fixed recipes, benchmark and locks.
The detached guard survives loss of its calling scheduler, stops that caller's
worker group, and keeps all signal grace inside the original cutoff/reserve.
Only inherited process groups are supported; workers must not daemonize.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import fcntl
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

from fusion.next.common import ROOT, digest, identity, read_json, utc_now, write_json


RUNS = ROOT/'outputs/fusion_followup/runs'
DEADLINE = '2026-11-09T09:34:22.509412+00:00'
RESERVE_SECONDS = 8*3600
MAX_STAGE_SECONDS = 6*3600


def process_identity(pid=None):
    process = psutil.Process(pid)
    return {'pid':process.pid, 'created_unix':process.create_time()}


def exact_alive(row):
    try:
        process = psutil.Process(row['pid'])
        return (abs(process.create_time()-row['created_unix']) < .01
                and process.is_running() and process.status() != psutil.STATUS_ZOMBIE)
    except psutil.NoSuchProcess:
        return False


def checked_directory(directory):
    directory = Path(directory).resolve()
    if not directory.is_relative_to(RUNS) or directory == RUNS:
        raise ValueError('Guard evidence must use a separate follow-up run directory')
    return directory


def start(directory, command, limit, stage_seconds, *, checkpoint_grace=60., termination_grace=5.):
    """Start the guard, not an admitted GPU experiment. Never pass shell text.

    The caller must stay alive until the job finishes. Its disappearance is a
    stop request, not permission for an unattended new run to continue.
    """
    directory = checked_directory(directory)
    if limit.get('total_wall_hours') != 100 or limit.get('deadline_utc') != DEADLINE:
        raise ValueError('The original absolute deadline must not change')
    if (not math.isfinite(stage_seconds) or not 0 < stage_seconds <= MAX_STAGE_SECONDS
            or not 0 <= termination_grace <= checkpoint_grace < stage_seconds):
        raise ValueError('Use a finite stage cap <=6 hours with signal grace inside it')
    if (not isinstance(command, (list, tuple)) or not command
            or not all(isinstance(arg, str) and '\x00' not in arg for arg in command)):
        raise ValueError('Use an argument array without a shell')
    started = time.time()
    stop_by = min(started+stage_seconds, datetime.fromisoformat(DEADLINE).timestamp()-RESERVE_SECONDS)
    if stop_by <= started:
        raise TimeoutError('Original deadline and delivery reserve leave no exploration time')
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'guard_launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = directory/'guard_request.json'
        if path.exists():
            raise RuntimeError('Preserve the previous guard request; use a new stage directory')
        request = {'directory':str(directory), 'command':list(command), 'cwd':str(ROOT),
                   'owner':process_identity(), 'limit':limit, 'stage_seconds':stage_seconds,
                   'requested_at_unix':started, 'stop_by_unix':stop_by,
                   'delivery_reserve_seconds':RESERVE_SECONDS,
                   'checkpoint_grace_seconds':checkpoint_grace,
                   'termination_grace_seconds':termination_grace,
                   'guard_sha256':digest(Path(__file__)),
                   'scientific_and_date_admission_included':False}
        request['identity'] = identity(request)
        write_json(path,request)
        with (directory/'guard.log').open('ab') as log:
            guard = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--request',str(path)],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True)
        receipt = {'request_identity':request['identity'], 'guard':process_identity(guard.pid),
                   'caller':request['owner'], 'stop_by_unix':stop_by,
                   'worker_has_not_yet_been_verified':True}
        write_json(directory/'guard_dispatch.json',receipt)
    return guard,receipt


def group_members(leader):
    """Require the original session, group and creation boundary before signals."""
    rows = []
    for process in psutil.process_iter(['pid','create_time','status']):
        try:
            if (process.info['create_time'] >= leader['created_unix']-.01
                    and process.info['status'] != psutil.STATUS_ZOMBIE
                    and os.getpgid(process.pid) == leader['pid']
                    and os.getsid(process.pid) == leader['pid']):
                rows.append({'pid':process.pid,'created_unix':process.info['create_time']})
        except (psutil.NoSuchProcess, ProcessLookupError):
            continue
    return rows


def reap_orphans():
    # Call only after Popen.poll() has collected the direct worker's return code.
    while True:
        try:
            pid,_ = os.waitpid(-1,os.WNOHANG)
            if not pid:
                return
        except ChildProcessError:
            return


def supervise(path):
    path = Path(path).resolve()
    directory = checked_directory(path.parent)
    if path != directory/'guard_request.json':
        raise ValueError('Unexpected guard request path')
    request = read_json(path)
    if (request['identity'] != identity({k:v for k,v in request.items() if k != 'identity'})
            or request['guard_sha256'] != digest(Path(__file__))
            or request['directory'] != str(directory) or request['cwd'] != str(ROOT)
            or request['limit']['total_wall_hours'] != 100 or request['limit']['deadline_utc'] != DEADLINE
            or request['delivery_reserve_seconds'] != RESERVE_SECONDS
            or request['stop_by_unix'] != min(request['requested_at_unix']+request['stage_seconds'],
                    datetime.fromisoformat(DEADLINE).timestamp()-RESERVE_SECONDS)
            or not 0 < request['stage_seconds'] <= MAX_STAGE_SECONDS
            or not 0 <= request['termination_grace_seconds'] <= request['checkpoint_grace_seconds'] < request['stage_seconds']):
        raise ValueError('Guard request identity or cutoff differs')
    state = {'request_identity':request['identity'], 'guard':process_identity(),
             'owner':request['owner'], 'worker':None, 'stop_by_unix':request['stop_by_unix'],
             'absolute_deadline_utc':DEADLINE, 'delivery_reserve_seconds':RESERVE_SECONDS,
             'signal_events':[], 'stage':'starting', 'worker_started':False}
    def save(stage,**details):
        state.update(stage=stage,updated_at_utc=utc_now(),**details)
        write_json(directory/'guard_state.json',state)
    resource = (RUNS.parent/'execution_guard.lock').open('a')
    try:
        fcntl.flock(resource,fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        save('refused',reason='another_followup_guard_is_active')
        resource.close()
        return 1
    worker = None
    interrupted = []
    handlers = {}
    for number in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
        handlers[number] = signal.signal(number,lambda signum,frame:interrupted.append(signum))
    try:
        # Adopt and reap worker grandchildren if their intermediate parent exits.
        # This is Linux-only, matching the research execution environment.
        libc = ctypes.CDLL(None,use_errno=True)
        if libc.prctl(36,1,0,0,0):  # PR_SET_CHILD_SUBREAPER
            raise OSError(ctypes.get_errno(),'Cannot enable descendant reaping')
        cutoff = time.monotonic()+max(0,request['stop_by_unix']-time.time())
        if not exact_alive(request['owner']):
            save('refused',reason='calling_scheduler_is_not_alive')
            return 1
        if min(cutoff-time.monotonic(),request['stop_by_unix']-time.time()) <= 0 or interrupted:
            save('refused',reason='cutoff_or_stop_request_before_start')
            return 1
        with (directory/'worker.log').open('ab') as log:
            worker = subprocess.Popen(request['command'],cwd=ROOT,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state.update(worker=process_identity(worker.pid),worker_started=True)
        save('running')
        reason,fast_stop = None,None
        last_saved = time.monotonic()
        sent = set()
        def send(number):
            if number in sent:
                return
            sent.add(number)
            members = group_members(state['worker'])
            delivered = False
            if members:
                try:
                    os.killpg(state['worker']['pid'],number)
                    delivered = True
                except ProcessLookupError:
                    pass
            state['signal_events'].append({'signal':int(number),'sent_at_unix':time.time(),
                                           'sent':delivered,'members':members})
            save('stopping',reason=reason)
        while True:
            now = time.monotonic()
            # A wall-clock jump backward cannot extend the monotonic deadline;
            # a forward jump shortens it to preserve the absolute cutoff.
            remaining = min(cutoff-now,request['stop_by_unix']-time.time())
            returncode = worker.poll()
            if returncode is not None:
                reap_orphans()
                if not group_members(state['worker']):
                    save('completed' if reason is None and returncode == 0 else 'stopped',
                         reason=reason or ('worker_failed' if returncode else 'worker_completed'),
                         worker_returncode=returncode,live_group_members=[])
                    return 0 if state['stage'] == 'completed' else 1
                if reason is None:
                    reason = 'leader_exited_with_live_descendants'
                    fast_stop = now+request['termination_grace_seconds']
            if reason is None and (interrupted or not exact_alive(request['owner'])):
                reason = 'guard_received_stop_signal' if interrupted else 'calling_scheduler_exited'
                fast_stop = now+request['termination_grace_seconds']
            if reason is None and remaining <= request['checkpoint_grace_seconds']:
                reason = 'stage_time_limit'
            if fast_stop is not None:
                send(signal.SIGTERM)
                if now >= fast_stop or remaining <= 0:
                    send(signal.SIGKILL)
            elif remaining <= 0:
                send(signal.SIGKILL)
            elif remaining <= request['termination_grace_seconds']:
                send(signal.SIGTERM)
            elif reason is not None:
                send(signal.SIGINT)
            if now-last_saved >= 10:
                save(state['stage'],reason=reason)
                last_saved = now
            time.sleep(.05 if reason is not None else min(.25,max(.01,remaining-request['checkpoint_grace_seconds'])))
    except BaseException as error:
        save('failed',reason=type(error).__name__,error=str(error))
        raise
    finally:
        if worker is not None:
            if state['worker'] is None:
                state['worker'] = process_identity(worker.pid)
            if group_members(state['worker']):
                try:
                    os.killpg(worker.pid,signal.SIGKILL)
                except ProcessLookupError:
                    pass
            worker.wait()
            reap_orphans()
        for number,handler in handlers.items():
            signal.signal(number,handler)
        fcntl.flock(resource,fcntl.LOCK_UN)
        resource.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Internal group guard; not a training admission launcher')
    parser.add_argument('--request',type=Path,required=True)
    sys.exit(supervise(parser.parse_args().request))
