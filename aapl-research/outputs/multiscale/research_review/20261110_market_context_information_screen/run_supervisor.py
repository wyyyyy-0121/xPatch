"""External pause and original-deadline supervisor for one fixed SPY screen."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(65536), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--phase', choices=('first', 'resume', 'review'), required=True)
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    body = args.protocol.read_bytes(); protocol = json.loads(body)
    identity = hashlib.sha256(body).hexdigest(); now = time.time()
    start, end = protocol['started_epoch'], protocol['deadline_epoch']
    if not (protocol.get('market_arrays_admitted') is True
            and type(start) in (int, float) and type(end) in (int, float)
            and math.isfinite(start) and math.isfinite(end) and end-start == 900
            and start <= now < end-2 and protocol['supervisor_sha256'] == digest(Path(__file__))
            and protocol.get('GPU') is False and protocol.get('Q3_allowed') is False):
        raise RuntimeError('Original screen protocol/window identity invalid')
    root = base/'run'; root.mkdir(exist_ok=True)
    receipt = root/(args.phase.upper()+'_SUPERVISOR.json')
    if receipt.exists():
        raise RuntimeError('Do not overwrite existing execution receipt')
    pause_receipt = root/'EXTERNAL_PAUSE_RECEIPT.json'
    if (args.phase == 'first' and pause_receipt.exists()
            or args.phase == 'resume' and not pause_receipt.exists()):
        raise RuntimeError('Actual pause/resume ordering invalid')
    role = 'review' if args.phase == 'review' else 'main'
    command = protocol[role+'_command']
    if not isinstance(command, list) or not command or any(not isinstance(x, str) for x in command):
        raise RuntimeError('Frozen command invalid')
    os.sched_setaffinity(0, {26 if role == 'review' else 24})
    began = time.time(); marker = None; pause_sent = False
    with (root/(args.phase.upper()+'_PROCESS.log')).open('x') as output:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        state = {'protocol_sha256': identity, 'phase': args.phase, 'PID': child.pid,
                 'supervisor_sha256': protocol['supervisor_sha256'], 'started_epoch': began,
                 'deadline_epoch': end, 'status': 'running', 'forced_kill': False,
                 'original_deadline_reset': False, 'CPU_affinity': [26 if role == 'review' else 24]}
        atomic(receipt, state)
        def send(signum):
            try:
                os.killpg(child.pid, signum)
                return True
            except ProcessLookupError:
                return False
        signal.signal(signal.SIGTERM, lambda *_: send(signal.SIGTERM))
        signal.signal(signal.SIGINT, lambda *_: send(signal.SIGTERM))
        term_sent = False
        while child.poll() is None:
            now = time.time()
            if args.phase == 'first' and not pause_sent and 'pause_error' not in state:
                ready = root/'READY_FOR_EXTERNAL_PAUSE.json'
                if ready.exists():
                    try:
                        marker = json.loads(ready.read_bytes())
                        path = Path(marker['checkpoint_path']).resolve()
                        if (marker.get('protocol_sha256') != identity or marker.get('pid') != child.pid
                                or marker.get('cursor') != 2048 or not path.is_relative_to(root.resolve())
                                or digest(path) != marker['checkpoint_sha256']):
                            raise ValueError('Invalid pause marker identity')
                    except Exception:
                        send(signal.SIGTERM)
                        state['pause_error'] = 'ready_marker_or_checkpoint_identity_invalid'
                    else:
                        pause_sent = send(signal.SIGUSR1)
                        state['external_pause_sent_epoch'] = time.time()
                        state['checkpoint_before_sha256'] = marker['checkpoint_sha256']
            if now >= end:
                state['forced_kill'] = send(signal.SIGKILL); child.wait(); break
            if now >= end-1 and not term_sent:
                send(signal.SIGTERM); term_sent = True
            time.sleep(min(0.05, max(0.001, end-now)))
        state.update(status='terminal', child_terminal=True, exit_code=child.returncode,
                     ended_epoch=time.time(), seconds=time.time()-began,
                     external_pause_signal_sent=pause_sent)
        if args.phase == 'first' and pause_sent and child.returncode == 75 and 'pause_error' not in state:
            try:
                after = digest(marker['checkpoint_path'])
                state['checkpoint_after_sha256'] = after
                if after != marker['checkpoint_sha256']:
                    raise ValueError('Checkpoint changed during actual pause')
            except Exception:
                state['pause_error'] = 'checkpoint_changed_during_actual_pause'
            else:
                atomic(pause_receipt, {'kind': 'screen_external_pause_v1', 'protocol_sha256': identity,
                    'pid': child.pid, 'signal': 'SIGUSR1', 'exit_code': 75, 'cursor': 2048,
                    'checkpoint_path': marker['checkpoint_path'],
                    'checkpoint_sha256_before': marker['checkpoint_sha256'],
                    'checkpoint_sha256_after': after,
                    'supervisor_source_sha256': protocol['supervisor_sha256'],
                    'signal_sent_epoch': state['external_pause_sent_epoch'], 'ended_epoch': time.time()})
        elif args.phase == 'first':
            state['pause_error'] = state.get('pause_error', 'required_actual_pause_not_demonstrated')
        atomic(receipt, state)
    print(json.dumps(state))
    okay = child.returncode == 0 if args.phase != 'first' else child.returncode == 75 and 'pause_error' not in state
    raise SystemExit(0 if okay else 75)


if __name__ == '__main__':
    main()
