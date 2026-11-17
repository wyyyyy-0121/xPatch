"""Run the frozen synthetic worker with a repository-local access boundary.

Python audit hooks are process-level checks, not an operating-system sandbox.
Only the new synthetic directory, one recorded model source, and the installed
Python runtime may be opened inside this repository. No network is admitted.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--mode', choices=('identities', 'cost', 'trajectory'), required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--pause-kind', choices=('signal', 'deadline'))
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    stage = protocol_path.parent.resolve()
    repository = next(p for p in stage.parents if (p / 'MULTISCALE_CURRENT_PLAN_ZH.md').exists())
    rules = json.loads(Path(protocol['rules_path']).read_text())
    worker = stage / 'code/function_centering.py'
    assert digest(protocol['rules_path']) == protocol['rules_sha256']
    assert digest(worker) == protocol['worker_sha256']
    assert digest(__file__) == protocol['guard_sha256']
    assert digest(rules['source_review_decision']) == rules['source_review_decision_sha256']
    assert protocol['deadline_epoch'] == rules['deadline_epoch']
    assert time.time() < protocol['deadline_epoch']
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    assert set(os.sched_getaffinity(0)) <= {24}
    args.output.resolve().relative_to(Path(protocol['output_directory']).resolve())
    allowed_source = Path(rules['original_model_source']).resolve()
    source_decision = Path(rules['source_review_decision']).resolve()
    runtime = (repository / '.venv').resolve()

    def audit(event, values):
        if event in ('socket.connect', 'socket.bind', 'socket.getaddrinfo'):
            raise PermissionError('Network is not admitted by synthetic protocol')
        if event != 'open' or not values:
            return
        raw = values[0]
        if isinstance(raw, int):
            return
        if not isinstance(raw, (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(raw)).resolve()
        try:
            path.relative_to(repository)
        except ValueError:
            return  # Installed runtime and OS files; not a market-data boundary.
        if path in (allowed_source, source_decision) or path.is_relative_to(stage) or path.is_relative_to(runtime):
            return
        raise PermissionError('Repository input outside synthetic admission: ' + str(path))

    sys.addaudithook(audit)
    sys.dont_write_bytecode = True
    worker_args = [str(worker), '--protocol', str(protocol_path), '--mode', args.mode,
                   '--output', str(args.output.resolve())]
    if args.resume:
        worker_args.append('--resume')
    if args.pause_kind:
        worker_args.extend(['--pause-kind', args.pause_kind])
    sys.argv = worker_args
    runpy.run_path(str(worker), run_name='__main__')


if __name__ == '__main__':
    main()
