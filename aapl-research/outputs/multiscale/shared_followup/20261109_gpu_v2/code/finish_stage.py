"""One-shot local review and capsule pipeline after complete market analysis.

This process never starts training or opens market arrays. The existing queue
owns training/evaluation. Each child enforces its own completion/identity gate.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time

BASE = Path(__file__).resolve().parents[1]
DEADLINE = 1794216709.0
STOP = False


def interrupted(*_):
    global STOP
    STOP = True


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as target:
        json.dump(value, target, indent=2, ensure_ascii=False)
        target.write('\n')
        target.flush()
        os.fsync(target.fileno())
        temporary = target.name
    os.replace(temporary, path)


def state(status, **fields):
    atomic(BASE / 'audit/FOLLOWUP_STATE.json', dict(
        status=status, pid=os.getpid(), deadline_epoch=DEADLINE,
        at_utc=now(), Q3_values_read=False, independent_confirmation=False,
        research_goal_achieved=False, **fields))


def ready():
    rules = read(BASE / 'protocol/RULES.json')
    paths = [BASE / 'runs' / j['id'] / 'DONE.json' for j in rules['jobs']]
    paths += [BASE / 'audit/COMPLETE.json', BASE / 'analysis/ANALYSIS_COMPLETE.json']
    return len(rules['jobs']) == 30 and all(p.is_file() for p in paths)


def controller_live(pid):
    try:
        command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        return str(BASE / 'code/controller.py').encode() in command
    except FileNotFoundError:
        return False


def child(stage, args, expected_hashes):
    assert not STOP and time.time() < DEADLINE
    for relative, digest in expected_hashes.items():
        assert sha(BASE / relative) == digest, f'Prepared stage source changed: {relative}'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2',
        MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    with (BASE / 'logs' / f'followup_{stage}.log').open('a') as log:
        process = subprocess.Popen([sys.executable, *args], cwd=BASE,
            env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
        state('running_' + stage, child_pid=process.pid, stage=stage)
        signaled_at = None
        while process.poll() is None:
            if STOP or time.time() >= DEADLINE:
                if signaled_at is None:
                    # A stage may start inference grandchildren. Stop its
                    # isolated process group, never the training queue group.
                    os.killpg(process.pid, signal.SIGTERM)
                    signaled_at = time.monotonic()
                elif time.monotonic() - signaled_at >= 10:
                    os.killpg(process.pid, signal.SIGKILL)
            time.sleep(0.5)
        if signaled_at is not None:
            # The wrapper can exit before a cooperatively stopping grandchild.
            # Bound its isolated group without touching retained train states.
            while time.monotonic() - signaled_at < 10:
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.2)
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    if process.returncode != 0 or STOP or time.time() >= DEADLINE:
        state('paused_or_failed', stage=stage, child_exit_code=process.returncode,
            existing_training_and_analysis_preserved=True)
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    assert set(os.sched_getaffinity(0)).issubset({24, 26})
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    if args.check:
        print(json.dumps({'complete_analysis_ready': ready(),
            'before_deadline': time.time() < DEADLINE, 'market_arrays_read': False}))
        return
    for sig in (signal.SIGUSR1, signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    config = read(BASE / 'audit/FOLLOWUP_LAUNCH.json')
    assert config['deadline_epoch'] == DEADLINE
    assert config['rules_sha256'] == sha(BASE / 'protocol/RULES.json')
    assert config['source_hashes']['code/finish_stage.py'] == sha(Path(__file__))
    with (BASE / 'audit/FOLLOWUP_LOCK').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state('waiting_for_complete_analysis', controller_pid=config['controller_pid'])
        while not ready():
            if STOP or time.time() >= DEADLINE:
                state('paused_at_authorized_boundary', existing_progress_preserved=True)
                return
            paused = BASE / 'audit/PAUSED.json'
            pipeline = BASE / 'audit/PIPELINE_STATE.json'
            if paused.exists() or (pipeline.exists() and read(pipeline)['status'] != 'completed'):
                state('upstream_incomplete', existing_progress_preserved=True)
                return
            if not controller_live(config['controller_pid']):
                # Never launch a replacement queue from a missing observation.
                state('upstream_process_missing_needs_review', existing_progress_preserved=True)
                return
            time.sleep(5)
        analysis = read(BASE / 'analysis/ANALYSIS_COMPLETE.json')
        assert analysis['status'] == 'completed' and analysis['complete_bundles'] == 30
        assert analysis['Q3_values_read'] is False
        assert analysis['independent_confirmation'] is False
        stages = (
            ('independent_statistics', [str(BASE / 'review/independent_statistics.py')]),
            ('market_capsule', [str(BASE / 'code/stage_release.py'), '--mode', 'market',
                '--name', config['capsule_name']]),
        )
        for stage, command in stages:
            if not child(stage, command, config['source_hashes']):
                return
            if stage == 'independent_statistics':
                receipt = read(BASE / 'review/INDEPENDENT_STATISTICS_REVIEW.json')
                assert receipt['status'] == 'passed'
                assert receipt['analysis_complete_sha256'] == sha(BASE / 'analysis/ANALYSIS_COMPLETE.json')
                assert receipt['script_sha256'] == config['source_hashes']['review/independent_statistics.py']
        release = BASE / 'release' / (config['capsule_name'] + '.RELEASE.json')
        receipt = read(release)
        assert receipt['status'] == 'passed' and receipt['mode'] == 'market'
        manuscript = config.get('manuscript_command')
        if manuscript is not None:
            assert manuscript == [str(BASE / 'code/build_manuscript.py'), '--render']
            if not child('manuscript', manuscript, config['source_hashes']):
                return
            build = read(BASE / 'manuscript/new_version/DOCUMENT_BUILD.json')
            assert build['status'] == 'rendered_requires_visual_review'
            assert build['visual_review_complete'] is False
            state('rendered_pending_visual_review', release_receipt=str(release),
                manuscript_directory=str(BASE / 'manuscript/new_version'),
                independent_statistics_receipt=str(BASE / 'review/INDEPENDENT_STATISTICS_REVIEW.json'),
                formal_visual_review_complete=False)
            return
        state('statistics_and_capsule_complete', release_receipt=str(release),
            independent_statistics_receipt=str(BASE / 'review/INDEPENDENT_STATISTICS_REVIEW.json'),
            manuscript_integration_pending=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if not isinstance(exc, (SystemExit, KeyboardInterrupt)):
            state('failed_needs_review', error=repr(exc), existing_progress_preserved=True)
        raise
