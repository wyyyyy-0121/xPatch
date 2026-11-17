"""Run one fixed acquisition scope beneath the protocol's absolute deadline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def atomic(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--scope', choices=('calibration', 'remaining'), required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    receipt = root/(args.scope.upper()+'_SUPERVISOR.json')
    if receipt.exists():
        raise RuntimeError('Do not overwrite existing execution evidence')
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    now = time.time()
    if not (protocol.get('execution_admitted') is True
            and protocol['started_epoch'] <= now < protocol['deadline_epoch']-3
            and protocol['deadline_epoch']-protocol['started_epoch'] == 3600
            and hashlib.sha256((root/'download_compressed_spy.py').read_bytes()).hexdigest()
                == protocol['source_sha256']):
        raise RuntimeError('Protocol identity or original absolute deadline invalid')
    os.sched_setaffinity(0, {24})
    started = time.time()
    log = root/(args.scope.upper()+'_PROCESS.log')
    with log.open('x') as stream:
        process = subprocess.Popen([sys.executable, '-B', str(root/'download_compressed_spy.py'),
            '--protocol', str(args.protocol.resolve()), '--scope', args.scope],
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True)
        state = {'protocol_sha256': hashlib.sha256(protocol_bytes).hexdigest(),
                 'supervisor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 'scope': args.scope, 'PID': process.pid, 'started_epoch': started,
                 'deadline_epoch': protocol['deadline_epoch'], 'CPU_affinity': [24],
                 'status': 'running', 'terminal_signal_sent': False,
                 'forced_kill': False, 'deadline_reset': False}
        atomic(receipt, state)
        def safe_signal(signum):
            try:
                os.killpg(process.pid, signum)
                return True
            except ProcessLookupError:
                return False
        def forward_stop(_signum, _frame):
            if process.poll() is None:
                safe_signal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, forward_stop)
        signal.signal(signal.SIGINT, forward_stop)
        while process.poll() is None:
            now = time.time()
            if now >= protocol['deadline_epoch']:
                state['forced_kill'] = safe_signal(signal.SIGKILL)
                process.wait()
                break
            if now >= protocol['deadline_epoch']-1 and not state['terminal_signal_sent']:
                state['terminal_signal_sent'] = safe_signal(signal.SIGTERM)
            time.sleep(min(0.2, max(0.001, protocol['deadline_epoch']-now)))
        state.update(status='terminal', exit_code=process.returncode, ended_epoch=time.time(),
                     child_terminal=True, seconds=time.time()-started)
        atomic(receipt, state)
    print(json.dumps(state))
    raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
