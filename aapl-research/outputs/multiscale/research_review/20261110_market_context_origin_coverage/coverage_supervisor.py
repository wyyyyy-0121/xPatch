"""One fixed clock-coverage command under its original absolute180s deadline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--role', choices=('main', 'review'), required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(); root = Path(__file__).resolve().parent
    body = args.protocol.read_bytes(); p = json.loads(body)
    now = time.time()
    if not (p.get('execution_admitted') is True and p['deadline_epoch']-p['started_epoch'] == 180
            and p['started_epoch'] <= now < p['deadline_epoch'] and args.command
            and p['supervisor_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            and args.command == p[args.role+'_command']):
        raise RuntimeError('Coverage original-window identity invalid')
    receipt = root/(args.role.upper()+'_SUPERVISOR.json')
    if receipt.exists():
        raise RuntimeError('Do not overwrite execution receipts')
    os.sched_setaffinity(0, {24 if args.role == 'main' else 26})
    started = time.time()
    with (root/(args.role.upper()+'_PROCESS.log')).open('x') as stream:
        child = subprocess.Popen(args.command, stdin=subprocess.DEVNULL, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        state = {'protocol_sha256': hashlib.sha256(body).hexdigest(), 'PID': child.pid,
                 'started_epoch': started, 'deadline_epoch': p['deadline_epoch'],
                 'role': args.role, 'status': 'running', 'forced_kill': False,
                 'original_deadline_reset': False}
        atomic(receipt, state)
        def stop(signum):
            try:
                os.killpg(child.pid, signum)
                return True
            except ProcessLookupError:
                return False
        signal.signal(signal.SIGTERM, lambda *_: stop(signal.SIGTERM))
        signal.signal(signal.SIGINT, lambda *_: stop(signal.SIGTERM))
        terminated = False
        while child.poll() is None:
            now = time.time()
            if now >= p['deadline_epoch']:
                state['forced_kill'] = stop(signal.SIGKILL); child.wait(); break
            if now >= p['deadline_epoch']-1 and not terminated:
                stop(signal.SIGTERM); terminated = True
            time.sleep(min(0.1, max(0.001, p['deadline_epoch']-now)))
        state.update(status='terminal', child_terminal=True, exit_code=child.returncode,
                     ended_epoch=time.time(), seconds=time.time()-started)
        atomic(receipt, state)
    print(json.dumps(state))
    raise SystemExit(child.returncode)


if __name__ == '__main__':
    main()
