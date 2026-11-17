"""One fixed state/seed Q/S pair, including the first ten-batch cost package.

This module has no scheduler and cannot admit the remaining queue itself.
External supervision owns process/CPU/RSS limits and the original hard deadline.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE/'training_core'))
import torch
from common_adapter import access
from market_reader import MarketInputs
from train_pair import train_pair, SCIENCE_RULES_SHA256


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    tmp = path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with tmp.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    tmp.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--cost-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); startup = time.time()
    body = args.protocol.read_bytes(); p = json.loads(body)
    protocol_sha = hashlib.sha256(body).hexdigest()
    if not p['started_epoch'] <= time.time() < p['deadline_epoch']:
        raise ValueError('Original deadline before worker initialization')
    locks = {str(Path(r['path']).resolve()): r['sha256'] for r in p['locks']}
    for path, digest in locks.items():
        if sha(path) != digest:
            raise ValueError('Frozen source drift: '+path)
    if locks.get(str(Path(__file__).resolve())) != sha(__file__):
        raise ValueError('Worker source not locked')
    jobs = p['training_jobs']; matching = [r for r in jobs if r['job_id'] == args.job_id]
    expected = {(task, fold, seed) for task in
                ('minute_h60_future_log_return', 'hourly_h8_future_log_return')
                for fold in (1, 2) for seed in (2026, 2027, 2028)}
    if len(jobs) != 12 or len({r['job_id'] for r in jobs}) != 12 or len(matching) != 1 \
            or {(r['task'], r['fold'], r['seed']) for r in jobs} != expected:
        raise ValueError('Exactly twelve fixed state/seed pairs required')
    job = matching[0]
    if Path(args.job_id).name != args.job_id or not args.job_id:
        raise ValueError('Invalid job directory')
    if sorted(os.sched_getaffinity(0)) != job['CPU_affinity'] or job['threads'] != 2:
        raise ValueError('Fixed job CPU/thread binding')
    if len(job['CPU_affinity']) != 2 or set(job['CPU_affinity']) & {24, 26}:
        raise ValueError('Training/analysis CPU separation')
    if any(os.environ.get(k) != '2' for k in
           ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')):
        raise ValueError('Training thread environment')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
        raise ValueError('GPU must be hidden before import')
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    output = Path(p['output_directory'])
    if not output.is_absolute():
        raise ValueError('Absolute protocol output directory required')
    if args.cost_only:
        if args.resume or job['job_id'] != p['first_cost_job_id'] or job['seed'] != 2026 \
                or job['task'] != 'minute_h60_future_log_return' or job['fold'] != 2:
            raise ValueError('First work package is fixed minute fold2 seed2026 ten common batches')
    else:
        admission = json.loads((output/'REMAINING_QUEUE_ADMISSION.json').read_text())
        if admission['protocol_sha256'] != protocol_sha or admission['accepted'] is not True:
            raise ValueError('Remaining queue lacks same-window cost admission')
    inputs = MarketInputs(args.protocol)
    began_io = time.perf_counter(); a = inputs.adapter(job['task'], job['fold'])
    if a.state.state_uid != job['state_uid'] or (args.cost_only and len(a.roles['Q']) <= 1280):
        raise ValueError('Fixed state identity/first-ten-batch domain')
    io_seconds = time.perf_counter()-began_io
    began_scaler = time.perf_counter(); a.fit_scaler(access(a.state, 'Q', 'fit_scaler'))
    scaler_seconds = time.perf_counter()-began_scaler; inputs.guard()
    prototype = Path(p['prototype_path']).resolve()
    if locks.get(str(prototype)) != sha(prototype):
        raise ValueError('Prototype not frozen')
    spec = importlib.util.spec_from_file_location('frozen_two_state_market_prototype', prototype)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    identity = {'source': {'prototype': sha(prototype), 'adapter': sha(BASE/'common_adapter.py')},
                'code': {'worker': sha(__file__), 'kernel': sha(BASE/'training_core/train_pair.py'),
                         'reader': sha(BASE/'market_reader.py')},
                'data': {'protocol_sha256': protocol_sha, 'state_uid': a.state.state_uid,
                         'input_reference_sha256': p['input_reference_sha256']},
                'spec': SCIENCE_RULES_SHA256}
    run_dir = output/'pairs'/job['job_id']; run_dir.parent.mkdir(parents=True, exist_ok=True)
    batch_times = []; previous = time.perf_counter()

    def observe(event, summary, token):
        nonlocal previous
        now = time.perf_counter()
        if event == 'COMMON_BATCH_COMMITTED':
            if args.cost_only:
                batch_times.append(now-previous)
            previous = now
            if args.cost_only and summary['common_steps'] == 10:
                token.request('fixed_first10_cost_package_complete')
        print('PROGRESS '+json.dumps({'event': event, **summary}), flush=True)

    result = train_pair(a, module.paired_models, seed=job['seed'], absolute_deadline=p['deadline_epoch'],
                        outputdir=run_dir, identity=identity, resume=args.resume, progress_hook=observe)
    receipt = {'job_id': job['job_id'], 'protocol_sha256': protocol_sha, 'PID': os.getpid(),
               'worker_started_epoch': startup, 'finished_epoch': time.time(),
               'absolute_deadline': p['deadline_epoch'], 'deadline_reset': False,
               'cost_only': args.cost_only, 'resume': args.resume, 'bar_adapter_io_seconds': io_seconds,
               'Q_scaler_seconds': scaler_seconds, 'first_common_batch_seconds_including_initialization':
                    batch_times[0] if batch_times else None,
               'subsequent_common_batch_seconds': batch_times[1:], 'input_decode_audit': inputs.audit,
               'result': result, 'C_cal_T_role_features_labels_constructed': False,
               'all_preQ3_close_columns_physically_decoded': True,
               'GPU': False, 'research_goal_achieved': False}
    name = 'COST_PACKAGE.json' if args.cost_only else 'PAIR_WORKER_RESULT.json'
    atomic(output/(job['job_id']+'_'+name), receipt)
    print('RESULT '+json.dumps(receipt, allow_nan=False), flush=True)
    return result['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
