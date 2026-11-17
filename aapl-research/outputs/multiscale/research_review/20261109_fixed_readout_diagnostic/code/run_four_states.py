"""Run only the four frozen fit diagnostics, sharing one absolute deadline."""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time

BASE = Path(__file__).resolve().parents[1]
STOP = False
CHILD = None


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def atomic(path, value):
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)


def stop(*_):
    global STOP
    STOP = True
    if CHILD is not None and CHILD.poll() is None:
        CHILD.send_signal(signal.SIGUSR1)


def main():
    global CHILD
    assert set(os.sched_getaffinity(0)) == {24}
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGUSR1]:
        signal.signal(sig, stop)
    admission = load(BASE/'MARKET_ADMISSION.json')
    assert admission['status'] == 'admitted_four_fit_only_states'
    assert admission['controller_sha256'] == sha(Path(__file__))
    assert admission['code_freeze_sha256'] == sha(BASE/'CODE_FREEZE.json')
    assert admission['input_identities_sha256'] == sha(BASE/'INPUT_IDENTITIES.json')
    states = load(BASE/'INPUT_IDENTITIES.json')['states']
    assert len(states) == 4 and [s['id'] for s in states] == admission['state_ids']
    deadline = load(BASE/'STAGE_SCOPE.json')['deadline_epoch']
    rows = []
    for state in states:
        if STOP or time.time() >= deadline:
            break
        directory = BASE/'runs'/state['id']
        directory.mkdir(parents=True, exist_ok=True)
        log = directory/'WORKER_LOG.txt'
        command = [sys.executable, str(BASE/'code/readout_worker.py'), '--mode', 'market',
                   '--state-id', state['id'], '--run-dir', str(directory)]
        started = time.time()
        with log.open('a') as output:
            CHILD = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
            atomic(BASE/'QUEUE_STATE.json', {'status': 'running', 'controller_pid': os.getpid(),
                'worker_pid': CHILD.pid, 'current_state': state['id'], 'completed': rows,
                'deadline_epoch': deadline, 'GPU': False, 'Q3_values_read': False})
            code = CHILD.wait()
        rows.append({'state_id': state['id'], 'exit_code': code,
                     'wall_seconds': time.time()-started,
                     'result_exists': (directory/'RESULT.json').exists(),
                     'checkpoint_exists': (directory/'CHECKPOINT.json').exists()})
        atomic(BASE/'QUEUE_STATE.json', {'status': 'running', 'completed': rows,
                'deadline_epoch': deadline, 'GPU': False, 'Q3_values_read': False})
        CHILD = None
        if code != 0:
            break
    complete = len(rows) == 4 and all(row['exit_code'] == 0 and row['result_exists'] for row in rows)
    report = {'status': 'completed_all_four_fit_diagnostics' if complete else 'partial_or_paused_preserved',
              'completed': rows, 'deadline_epoch': deadline, 'finished_epoch': time.time(),
              'GPU': False, 'Q3_values_read': False, 'stop_outer_scored': False,
              'new_neural_training': False, 'new_forecasting_candidate_admitted': False,
              'research_goal_achieved': False}
    atomic(BASE/'QUEUE_STATE.json', report)
    print(json.dumps(report, indent=2))
    return 0 if complete else 2


if __name__ == '__main__':
    raise SystemExit(main())
