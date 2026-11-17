"""Execute only the frozen synthetic admission sequence and real process resumes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True, type=Path)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    stage = protocol_path.parent
    assert sha(Path(__file__)) == protocol['supervisor_sha256']
    assert set(os.sched_getaffinity(0)) <= {24}
    deadline = protocol['deadline_epoch']
    assert time.time() < deadline
    runs = stage / 'runs'
    runs.mkdir(exist_ok=True)
    events = []
    receipt = stage / 'EXECUTION_RECEIPT.json'

    def terminate_and_reap(child):
        if child.poll() is not None:
            return
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)

    def checkpoint(status, error=None):
        atomic_json(receipt, {'status': status, 'at_UTC': datetime.now(timezone.utc).isoformat(),
                             'protocol_sha256': sha(protocol_path), 'events': events,
                             'error': error, 'deadline_epoch': deadline,
                             'deadline_reset': False, 'market_data_opened': False,
                             'Q3_values_read': False, 'full_goal_achieved': False})

    def run(mode, name, resume=False, pause=None):
        assert time.time() + 30 < deadline
        out = runs / name
        out.mkdir(exist_ok=True)
        command = [sys.executable, str(stage / 'code/guarded_run.py'),
                   '--protocol', str(protocol_path), '--mode', mode, '--output', str(out)]
        if resume:
            command.append('--resume')
        if pause:
            command.extend(['--pause-kind', pause])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                   PYTHONDONTWRITEBYTECODE='1')
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1, env=env)
        event = {'mode': mode, 'name': name, 'pid': child.pid, 'resume': resume,
                 'pause_kind': pause, 'command': command, 'signals_sent': [],
                 'started_epoch': time.time()}
        events.append(event)
        checkpoint('running')
        selector = selectors.DefaultSelector()
        selector.register(child.stdout, selectors.EVENT_READ)
        lines = []
        try:
            while True:
                if time.time() >= deadline - 15 and child.poll() is None:
                    child.send_signal(signal.SIGUSR1)
                    event['signals_sent'].append({'signal': 'SIGUSR1', 'reason': 'global_deadline',
                                                   'at_epoch': time.time()})
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        terminate_and_reap(child)
                    raise TimeoutError('Global deadline reached; do not restart this stage')
                for key, _ in selector.select(timeout=0.25):
                    line = key.fileobj.readline()
                    if line:
                        lines.append(line)
                        if pause == 'signal' and 'READY_FOR_SIGNAL' in line:
                            assert not event['signals_sent'], 'Repeated signal-ready event'
                            os.kill(child.pid, signal.SIGUSR1)
                            event['signals_sent'].append({'signal': 'SIGUSR1',
                                                           'reason': 'actual_update4_fixture',
                                                           'at_epoch': time.time()})
                if child.poll() is not None:
                    lines.append(child.stdout.read())
                    break
        finally:
            selector.close()
            if child.poll() is None:
                terminate_and_reap(child)
            log = out / ('resume.log' if resume else 'initial.log')
            log.write_text(''.join(lines))
            event.update(exit_code=child.returncode, finished_epoch=time.time(), log=str(log),
                         log_sha256=sha(log))
            checkpoint('running')
        expected = 75 if pause and not resume else 0
        assert child.returncode == expected, 'Synthetic worker failed: ' + str(log)
        if pause == 'signal' and not resume:
            assert len(event['signals_sent']) == 1
        print(json.dumps({'completed_mode': mode, 'name': name, 'resume': resume,
                          'exit_code': child.returncode, 'pid': child.pid}), flush=True)

    try:
        run('identities', 'identities')
        run('cost', 'cost')
        run('trajectory', 'continuous')
        run('trajectory', 'signal', pause='signal')
        run('trajectory', 'signal', resume=True)
        run('trajectory', 'deadline', pause='deadline')
        run('trajectory', 'deadline', resume=True)
        checkpoint('completed_worker_sequence_pending_independent_result_acceptance')
    except BaseException as exc:
        checkpoint('stopped_failed_or_deadline_checkpoint_preserved',
                   {'type': type(exc).__name__, 'message': str(exc)})
        raise


if __name__ == '__main__':
    main()
