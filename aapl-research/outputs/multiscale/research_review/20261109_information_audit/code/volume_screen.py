"""Frozen deterministic fit-internal price/volume information screen.

No market access or execution on import. Uses only admitted pre-July bars and
the existing fit partition. New outer/stop/Q3 scores are never constructed.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import os
import signal
import sys
import tempfile
import time
import numpy as np

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]
PRIOR = ROOT/'outputs/multiscale/controlled_followup/20261108_v2'
REFERENCE = PRIOR/'code/reference_source'
CONTEXT = ROOT/'outputs/multiscale/mechanism_repair/20261109_v3b/code/predictor.py'
AUTH = ROOT/'outputs/multiscale/research_review/20261109_post_shared_mechanism/CONTINUING_RESEARCH_AUTHORIZATION_20261109.json'
TASKS = (('minute', 60, 480), ('hourly', 8, 96))
CELLS = {'A': 17, 'B': 18, 'C': 19}
STOP = False


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, data):
    path = Path(path); path.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno()); name = f.name
    os.replace(name, path)


def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.NamedTemporaryFile('wb', dir=path.parent, delete=False) as f:
        np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno()); name = f.name
    os.replace(name, path)


def dependencies():
    files = [BASE/'INFORMATION_MECHANISM_DESIGN_ZH.md',
             BASE/'SOURCE_INFORMATION_AVAILABILITY.json', AUTH, CONTEXT,
             PRIOR/'protocol/DATA.json', PRIOR/'protocol/CODE_FREEZE.json']
    files += sorted((BASE/'code').glob('*.py'))
    frozen = read(PRIOR/'protocol/CODE_FREEZE.json')['hashes']
    for relative, expected in frozen.items():
        path = PRIOR/'code'/relative
        if sha(path) != expected:
            raise ValueError('An original source dependency changed: '+str(path))
        files.append(path)
    protocol = read(PRIOR/'protocol/DATA.json')
    source = Path(protocol['admission'])
    if sha(source/'admission.json') != protocol['admission_sha256']:
        raise ValueError('Original admission changed')
    files.append(source/'admission.json')
    for profile, h, _ in TASKS:
        task = f'{profile}_h{h}_future_log_return'; path = PRIOR/'cache'/task/'TASK.json'
        if sha(path) != protocol['task_files'][task]:
            raise ValueError('Original task metadata changed')
        files.append(path); meta = read(path)
        for fold in (1, 2):
            files.append(Path(meta['folds'][str(fold)]['split_file']))
    return {str(path): sha(path) for path in files}


def freeze(output):
    authorization = read(AUTH)
    if not (authorization['direct_human_approval'] is True
            and authorization['subsequent_local_research_can_continue_autonomously'] is True
            and authorization['GPU'] is False):
        raise ValueError('Continuing local research authorization not found')
    now = time.time()
    content = {'study': 'fit_internal_price_volume_interaction',
        'sources': dependencies(), 'authorization_sha256': sha(AUTH),
        'received_epoch': now, 'deadline_epoch': now+7200,
        'maximum_processes': 1, 'maximum_logical_CPUs': 2, 'GPU': False,
        'output_directory': str(output.resolve()), 'ridge_penalty': 0.1,
        'folds': [1, 2], 'task_count': 2, 'planned_fits': 12,
        'cells': CELLS, 'unique_continuation_form': 'C_raw',
        'future_stop_outer_Q3_scoring': False}
    path = output/'RULES.json'
    if path.exists():
        old = read(path)
        if old['sources'] != content['sources'] or old['output_directory'] != content['output_directory']:
            raise ValueError('Refuse to resume changed source/output identity')
        return old
    atomic(path, content)
    return content


def empty_stats(horizon):
    return {'n': 0, 'sx': np.zeros(19), 'sy': np.zeros(horizon),
            'xx': np.zeros((19, 19)), 'xy': np.zeros((19, horizon))}


def update(stats, x, y):
    stats['n'] += len(x); stats['sx'] += x.sum(0); stats['sy'] += y.sum(0)
    stats['xx'] += x.T@x; stats['xy'] += x.T@y


def solve(stats, columns):
    n = int(stats['n']); mx = stats['sx']/n; my = stats['sy']/n
    sd = np.sqrt(np.maximum(np.diag(stats['xx'])/n-mx*mx, 0))
    sd = np.where(sd > 1e-12, sd, 1.0)
    g = (stats['xx']/n-np.outer(mx, mx))/sd[:, None]/sd[None, :]
    b = (stats['xy']/n-np.outer(mx, my))/sd[:, None]
    coef = np.linalg.solve(g[:columns, :columns]+0.1*np.eye(columns), b[:columns])
    return {'coef': coef, 'x_mean': mx[:columns], 'x_std': sd[:columns], 'y_mean': my}


def predict(model, x):
    return model['y_mean']+(x[:, :len(model['x_mean'])]-model['x_mean'])/model['x_std']@model['coef']


def partition_fit(fit, end_ns, horizon, available):
    boundary = int(np.floor(0.8*len(fit)))
    first = int(fit[boundary]); train, diag = fit[:boundary], fit[boundary:]
    train = train[end_ns[train+horizon] < end_ns[first]]
    return boundary, first, train[available(train)], diag[available(diag)]


def synthetic_acceptance():
    """Compare streaming/resumed sufficient statistics to a direct centered solve."""
    from volume_source import synthetic_acceptance as source_acceptance
    source_result = source_acceptance()
    rng = np.random.default_rng(2026); x = rng.normal(size=(300, 19)); y = rng.normal(size=(300, 8))
    stats = empty_stats(8); update(stats, x[:113], y[:113])
    # An actual serialized sufficient-stat snapshot, then resumed accumulation.
    with tempfile.TemporaryDirectory(prefix='aapl-volume-synthetic-') as d:
        path = Path(d)/'checkpoint.npz'; atomic_npz(path, **stats)
        with np.load(path, allow_pickle=False) as f:
            restored = {k: f[k].copy() for k in f.files}
        restored['n'] = int(restored['n']); update(restored, x[113:], y[113:])
    diffs = []
    for k in CELLS.values():
        m = solve(restored, k); mx = x[:, :k].mean(0); sd = x[:, :k].std(0)
        z = (x[:, :k]-mx)/sd; my = y.mean(0)
        direct = np.linalg.solve(z.T@z/len(x)+0.1*np.eye(k), z.T@(y-my)/len(x))
        np.testing.assert_allclose(m['coef'], direct, rtol=1e-11, atol=1e-12)
        diffs.append(float(np.max(abs(m['coef']-direct))))
    origins = np.arange(20, 120); stamps = np.arange(150)*60_000_000_000
    boundary, first, train, diag = partition_fit(origins, stamps, 8, lambda o: o % 3 != 0)
    assert boundary == 80 and first == 100
    assert np.all(stamps[train+8] < stamps[100]) and np.all(diag >= 100)
    assert np.all(train % 3 != 0) and np.all(diag % 3 != 0)
    # H volume observations and H+1 closes describe the same completed move.
    volume = np.arange(1., 25.); cumulative = np.r_[0., np.cumsum(volume)]
    origin, horizon = 15, 8
    mean = (cumulative[origin+1]-cumulative[origin-horizon+1])/horizon
    assert mean == np.mean(volume[8:16])
    log_close = np.arange(24.)**2
    assert log_close[origin]-log_close[origin-horizon] == 176
    return {'status': 'passed', 'source_feature_checks': source_result,
            'streaming_resumed_vs_direct_max_coefficient_error': max(diffs),
            'partition_original_80percent_boundary_then_mask_and_strict_label_purge': True,
            'H_volume_Hplus1_close_alignment': True,
            'market_data_used': False}


def run(output, rules):
    global STOP
    identity = sha(output/'RULES.json'); deadline = rules['deadline_epoch']
    if rules['sources'] != dependencies():
        raise ValueError('Source changed after freeze')
    def guard():
        if STOP or time.time() >= deadline:
            raise InterruptedError('Frozen volume screen deadline/stop signal')
    def stop(*_):
        global STOP
        STOP = True
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, stop)
    sys.path.insert(0, str(REFERENCE))
    from multiscale.features import CloseWindows
    from multiscale.bars import Bars, targets
    from volume_source import load_volume
    module_spec = importlib.util.spec_from_file_location('original_context', CONTEXT)
    context = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(context)
    acceptance = synthetic_acceptance(); acceptance['rules_sha256'] = identity
    atomic(output/'SYNTHETIC_ACCEPTANCE.json', acceptance); guard()
    cost_path = output/'COST_ADMISSION.json'
    if cost_path.exists():
        old = read(cost_path)
        if old.get('status') != 'admitted' or old.get('rules_sha256') != identity:
            raise ValueError('Prior cost admission failed or belongs to different rules')
    elif any(output.glob('*/fold_*/stats.npz')):
        raise ValueError('Resumed sufficient statistics require the same passed cost admission')
    source = load_volume(output/'cache', guard)
    admission = read(Path(read(PRIOR/'protocol/DATA.json')['admission'])/'admission.json')
    results = []
    for profile, h, lookback in TASKS:
        data = source['minute' if profile == 'minute' else 'hour']
        bars = Bars('minute' if profile == 'minute' else 'hour',
                    **{k: data[k] for k in ('start_ns', 'end_ns', 'close', 'observed', 'session_date')})
        windows = CloseWindows(bars, lookback, admission['split_actions'])
        task = f'{profile}_h{h}_future_log_return'; meta = read(PRIOR/'cache'/task/'TASK.json')
        abnormal = data['abnormal']; valid = np.isfinite(abnormal)
        prefix = np.concatenate(([0.], np.cumsum(np.where(valid, abnormal, 0.))))
        bad = np.concatenate(([0], np.cumsum(~valid)))
        lag_h_col = context.feature_names(h, lookback).index(f'return_lag_{h}')
        def batch(origins):
            guard(); history, _ = windows.arrays(origins)
            old = context.features(history, bars.end_ns[origins], h)
            u = (prefix[origins+1]-prefix[origins-h+1])/h
            x = np.column_stack((old, u, old[:, lag_h_col]*u))
            y = targets(bars, origins, h, 'future_log_return')
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError('Nonfinite information features or targets')
            return x, y
        for fold in (1, 2):
            guard(); folder = output/task/f'fold_{fold}'; folder.mkdir(parents=True, exist_ok=True)
            with np.load(meta['folds'][str(fold)]['split_file'], allow_pickle=False) as f:
                fit = f['fit'].copy()
            original_split = meta['folds'][str(fold)]
            if fit.dtype != np.dtype('int64') or fit.ndim != 1 or (
                    len(fit) != original_split['counts']['fit'] or (np.diff(fit) <= 0).any()
                    or fit[0] < lookback-1 or fit[-1]+h >= len(bars.close)
                    or hashlib.sha256(fit.astype('<i8').tobytes()).hexdigest() !=
                    original_split['split_sha256']['fit']):
                raise ValueError('Actual fit origins differ from the original frozen partition')
            # Boundary is fixed BEFORE the source-only common availability mask.
            def available(o):
                return bad[o+1]-bad[o-h+1] == 0
            boundary, diagnostic_first, train, diag = partition_fit(fit, bars.end_ns, h, available)
            if len(train) < 100 or len(diag) < 20:
                raise ValueError('Insufficient common fit-only information origins')
            atomic(folder/'PARTITION.json', {'rules_sha256': identity,
                'original_fit_count': len(fit), 'original_80percent_index': boundary,
                'diagnostic_first_origin_before_common_mask': diagnostic_first,
                'train_count': len(train), 'diagnostic_count': len(diag),
                'train_sha256': hashlib.sha256(train.astype('<i8').tobytes()).hexdigest(),
                'diagnostic_sha256': hashlib.sha256(diag.astype('<i8').tobytes()).hexdigest(),
                'train_last_label_end_ns': int(bars.end_ns[train[-1]+h]),
                'diagnostic_first_decision_ns': int(bars.end_ns[diag[0]]),
                'all_arms_share_origins': True})
            checkpoint = folder/'stats.npz'; state_path = folder/'FIT_STATE.json'; cursor = 0
            stats = empty_stats(h)
            if checkpoint.exists():
                with np.load(checkpoint, allow_pickle=False) as f:
                    stats = {k: f[k].copy() for k in ('n','sx','sy','xx','xy')}
                    cursor = int(f['cursor']); saved_identity = str(f['rules_sha256'])
                stats['n'] = int(stats['n'])
                if saved_identity != identity or stats['n'] != cursor or not 0 <= cursor <= len(train):
                    raise ValueError('Sufficient-stat checkpoint identity mismatch')
            start = time.monotonic()
            for i in range(cursor, len(train), 2048):
                guard(); x, y = batch(train[i:i+2048]); update(stats, x, y)
                cursor = i+len(x)
                atomic_npz(checkpoint, **stats, cursor=cursor, rules_sha256=identity)
                atomic(state_path, {'cursor': cursor, 'total': len(train), 'rules_sha256': identity})
                if profile == 'minute' and fold == 1 and not cost_path.exists():
                    elapsed = time.monotonic()-start
                    # Bound all four fits as the first full-size fit plus margin.
                    projection = elapsed*len(train)/len(x)*4*1.25+600
                    admitted = elapsed < 120 and projection <= deadline-time.time()
                    atomic(cost_path, {'status':'admitted' if admitted else 'not_admitted',
                        'rules_sha256':identity, 'first_max_shape_batch_seconds':elapsed,
                        'projected_compute_seconds_including_600second_reserve':projection})
                    if not admitted:
                        raise InterruptedError('Cost gate failed; do not shrink the fixed matrix')
            models = {name: solve(stats, n) for name, n in CELLS.items()}
            for name, model in models.items():
                atomic_npz(folder/(name+'.npz'), **model, rules_sha256=identity)
            metrics = {name: {'endpoint_sse':0., 'path_sse':0.} for name in CELLS}
            zero = {'endpoint_sse':0., 'path_sse':0.}; seen = 0
            for i in range(0, len(diag), 2048):
                x, y = batch(diag[i:i+2048]); seen += len(x)
                zero['endpoint_sse'] += float((y[:, -1]**2).sum())
                zero['path_sse'] += float((y**2).sum())
                for name, model in models.items():
                    error = predict(model, x)-y
                    metrics[name]['endpoint_sse'] += float((error[:, -1]**2).sum())
                    metrics[name]['path_sse'] += float((error**2).sum())
            for row in list(metrics.values())+[zero]:
                row['endpoint_mse'] = row['endpoint_sse']/seen
                row['path_mse'] = row['path_sse']/(seen*h)
            interaction = float(models['C']['coef'][-1, -1]/models['C']['x_std'][-1])
            checks = {f'C_endpoint_below_{name}':metrics['C']['endpoint_mse'] < metrics[name]['endpoint_mse']
                      for name in ('A','B')}
            checks.update({f'C_path_not_above_{name}':metrics['C']['path_mse'] <= metrics[name]['path_mse']
                           for name in ('A','B')})
            checks['C_endpoint_not_above_zero'] = metrics['C']['endpoint_mse'] <= zero['endpoint_mse']
            row = {'task':task,'fold':fold,'diagnostic_origins':seen,'metrics':metrics,'zero':zero,
                'interaction_original_unit_endpoint_coefficient':interaction,'checks':checks,
                'rules_sha256':identity,'passed_all_local_checks':all(checks.values()),
                'stop_or_outer_scored':False,'Q3_values_read':False,'independent_confirmation':False}
            atomic(folder/'RESULT.json', row); results.append(row)
    signs = {}
    for profile, h, _ in TASKS:
        rows = [r for r in results if r['task'].startswith(profile+'_')]
        c = [r['interaction_original_unit_endpoint_coefficient'] for r in rows]
        signs[profile] = bool(c[0]*c[1] > 0)
    gate = all(r['passed_all_local_checks'] for r in results) and all(signs.values())
    atomic(output/'RESULTS.json', {'status':'completed_12_fit_internal_cells','states':results,
        'same_task_two_fold_interaction_sign':signs,'continuation_gate_passed':gate,
        'rules_sha256':identity,'new_candidate_development_admitted_by_scientific_gate':gate,
        'method_success':False,'independent_confirmation':False,'Q3_values_read':False,
        'stop_or_outer_scored':False,'research_goal_achieved':False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1' or not set(os.sched_getaffinity(0)).issubset({24,26}):
        raise ValueError('CPU24/26 and CUDA_VISIBLE_DEVICES=-1 required')
    if not args.freeze:
        parent = os.getppid()
        if os.environ.get('AAPL_VOLUME_SUPERVISOR_PID') != str(parent) or (
                str(BASE/'code/screen_supervisor.py').encode() not in
                Path(f'/proc/{parent}/cmdline').read_bytes().split(b'\0')):
            raise ValueError('Use the frozen external supervisor')
    rules = freeze(args.output)
    if args.freeze:
        print(json.dumps({'deadline_epoch':rules['deadline_epoch'],'rules':str(args.output/'RULES.json')})); return
    try:
        run(args.output, rules)
    except BaseException as exc:
        atomic(args.output/'STOPPED.json', {'reason':repr(exc),'partial_artifacts_retained':True,
            'rules_sha256':sha(args.output/'RULES.json'),'research_goal_achieved':False})
        raise


if __name__ == '__main__':
    main()
