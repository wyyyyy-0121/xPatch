"""One bounded artificial-only screen/controls acceptance. No market source opens."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    tmp.replace(path)


def main():
    protocol_path = (Path(sys.argv[1]).resolve() if len(sys.argv) == 2
                     else BASE / 'SCREEN_ARTIFICIAL_EXECUTION_PROTOCOL.json')
    body = protocol_path.read_bytes(); p = json.loads(body)
    began = time.time()
    assert p['kind'] == 'two_state_screen_artificial_acceptance_v1'
    assert p['deadline_epoch'] - p['started_epoch'] == 180
    assert p['started_epoch'] <= began < p['deadline_epoch']
    assert p['market_arrays_allowed'] is False and p['GPU'] is False
    assert p['supervisor_sha256'] == sha(Path(__file__))
    for row in p['locks']:
        assert sha(Path(row['path'])) == row['sha256'], row['path']
    output_name = p.get('output_subdirectory', 'screen_artificial_run')
    assert isinstance(output_name, str) and Path(output_name).name == output_name
    out = BASE / output_name; out.mkdir(exist_ok=False)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1')

    def restrict():
        os.sched_setaffinity(0, {24})
        resource.setrlimit(resource.RLIMIT_AS, (6*1024**3, 6*1024**3))

    command = [sys.executable, '-B', str(BASE/'artificial_screen_tests.py')]
    with (out/'stdout.json').open('w') as stdout, (out/'stderr.txt').open('w') as stderr:
        proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env,
                                start_new_session=True, preexec_fn=restrict)
        reason = None; maximum_rss = 0
        while proc.poll() is None:
            try:
                status = Path(f'/proc/{proc.pid}/status').read_text()
                for line in status.splitlines():
                    if line.startswith('VmRSS:'):
                        maximum_rss = max(maximum_rss, int(line.split()[1])*1024)
            except FileNotFoundError:
                pass
            if time.time() >= p['deadline_epoch'] or maximum_rss > 3*1024**3:
                reason = 'absolute_deadline' if time.time() >= p['deadline_epoch'] else 'RSS_budget'
                os.killpg(proc.pid, signal.SIGKILL); break
            time.sleep(.02)
        code = proc.wait()
    finish = time.time()
    result = {'kind': p['kind'], 'protocol_sha256': hashlib.sha256(body).hexdigest(),
              'pid': proc.pid, 'exit_code': code, 'termination_reason': reason,
              'command': command, 'started_epoch': p['started_epoch'],
              'deadline_epoch': p['deadline_epoch'], 'supervisor_started_epoch': began,
              'finished_epoch': finish, 'window_elapsed_seconds': finish-p['started_epoch'],
              'maximum_observed_RSS_bytes': maximum_rss,
              'child_ru_maxrss_bytes': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,
              'stdout_sha256': sha(out/'stdout.json'), 'stderr_sha256': sha(out/'stderr.txt'),
              'market_source_opened_by_supervisor': False, 'GPU': False,
              'research_goal_achieved': False}
    if code == 0:
        results = json.loads((out/'stdout.json').read_text())
        result['artificial_status'] = results['status']
        result['check_groups'] = len(results['checks'])
    atomic(out/'SUPERVISOR.json', result)
    print(json.dumps(result, allow_nan=False))
    return 0 if code == 0 and reason is None and finish <= p['deadline_epoch'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
