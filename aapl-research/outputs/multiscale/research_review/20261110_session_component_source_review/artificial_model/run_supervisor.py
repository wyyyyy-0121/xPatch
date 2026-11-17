"""Run one artificial-only acceptance and replay within one absolute window."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    body = args.protocol.read_bytes()
    protocol = json.loads(body)
    identity = hashlib.sha256(body).hexdigest()
    start, end = protocol['started_epoch'], protocol['deadline_epoch']
    if not (type(start) in (int, float) and type(end) in (int, float)
            and math.isfinite(start) and math.isfinite(end) and end-start == 180
            and start <= time.time() < end-2
            and protocol.get('market_arrays_admitted') is False
            and protocol.get('GPU') is False and protocol.get('Q3_allowed') is False
            and protocol['supervisor_sha256'] == sha(Path(__file__))):
        raise ValueError('Artificial scope/window/source identity invalid')
    for role in ('main', 'review'):
        command = protocol.get(role+'_command')
        if not (isinstance(command, list) and command
                and all(isinstance(x, str) for x in command)):
            raise ValueError('Both frozen commands must be lists of strings')
    output = Path(__file__).resolve().parent/'artificial_run'
    output.mkdir(exist_ok=True)
    receipt_path = output/'SUPERVISOR.json'
    if receipt_path.exists():
        raise ValueError('Refuse to overwrite a prior run')
    state = {'protocol_sha256': identity, 'started_epoch': time.time(),
             'deadline_epoch': end, 'status': 'running', 'phases': [],
             'original_deadline_reset': False, 'artificial_only': True}
    failed = False
    child = None
    phase = None
    try:
        atomic(receipt_path, state)
        for role, cpu in [('main', 24), ('review', 26)]:
            command = protocol[role+'_command']
            if time.time() >= end-1:
                failed = True
                state['reason'] = 'original_deadline_before_'+role
                break
            os.sched_setaffinity(0, {cpu})
            phase = {'role': role, 'CPU_affinity': [cpu], 'started_epoch': time.time(),
                     'forced_kill': False}
            child = None
            state['phases'].append(phase)
            with (output/(role.upper()+'_PROCESS.log')).open('x') as stream:
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                         stdout=stream, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                phase['PID'] = child.pid
                atomic(receipt_path, state)

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
                    if now >= end:
                        phase['forced_kill'] = send(signal.SIGKILL)
                        child.wait()
                        break
                    if now >= end-1 and not term_sent:
                        send(signal.SIGTERM)
                        term_sent = True
                    time.sleep(min(.025, max(.001, end-now)))
                phase.update(exit_code=child.returncode, ended_epoch=time.time(),
                             terminal=True)
            atomic(receipt_path, state)
            if child.returncode != 0:
                failed = True
                break
    except BaseException as error:
        failed = True
        state['supervision_error'] = {'type': type(error).__name__, 'reason': str(error)}
    finally:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
            phase.update(forced_kill=True, terminal=True, exit_code=child.returncode,
                         ended_epoch=time.time())
        state.update(status='failed' if failed else 'complete', ended_epoch=time.time(),
                     new_market_execution=False, research_goal_achieved=False)
        try:
            atomic(receipt_path, state)
        except Exception as error:
            failed = True
            state.update(status='failed', terminal_receipt_error=type(error).__name__)
    print(json.dumps(state))
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
