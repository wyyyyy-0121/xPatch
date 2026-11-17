"""Frozen four-cell inner-only context experiment with resumable sufficient statistics."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import sys
import time
import numpy as np

BASE = Path(__file__).resolve().parents[1]
V3 = BASE.parent / '20261109_v3'
V3A = BASE.parent / '20261109_v3a'
sys.path.insert(0, str(V3 / 'code'))
from study import read, sha, atomic, npz, now, make_job, source_hashes, PRIOR, SOURCE
from engine import Data
from multiscale.bars import targets
sys.path.insert(0, str(BASE / 'code'))
from predictor import features, feature_names, Predictor

DEADLINE = 1794205909.6228158
STOP = False
CELLS = {'A_mean': np.arange(0, dtype=np.int64), 'B_context': np.arange(9, 17),
         'C_history': np.arange(9), 'D_joint': np.arange(17)}


def interrupted(*_):
    global STOP
    STOP = True


def frozen_sources():
    return {name: sha(BASE / 'code' / name) for name in ('predictor.py', 'experiment.py')}


def freeze():
    assert set(os.sched_getaffinity(0)).issubset({24, 26})
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    assert read(V3 / 'protocol/BUDGET.json')['deadline_epoch'] == DEADLINE
    assert source_hashes() == read(V3 / 'protocol/CODE_FREEZE.json')['hashes']
    data_protocol = read(PRIOR / 'protocol/DATA.json'); partitions = {}
    for task in ('minute_h60_future_log_return', 'hourly_h8_future_log_return'):
        taskfile = PRIOR / 'cache' / task / 'TASK.json'
        assert sha(taskfile) == data_protocol['task_files'][task]
        meta = read(taskfile); partitions[task] = {}
        for fold in range(1, 6):
            rec = meta['folds'][str(fold)]
            # TASK records hashes of index arrays, not one NPZ byte hash.
            with np.load(rec['split_file'], allow_pickle=False) as split:
                assert set(split.files) == set(rec['split_sha256'])
                for name, digest in rec['split_sha256'].items():
                    value = np.asarray(split[name], dtype='<i8')
                    assert hashlib.sha256(value.tobytes()).hexdigest() == digest
            assert sha(rec['scaler_file']) == rec['scaler_sha256']
            partitions[task][str(fold)] = {'split_sha256': rec['split_sha256'],
                'split_npz_file_sha256': sha(rec['split_file']),
                'scaler_sha256': rec['scaler_sha256'], 'outer_targets_registered_sha256': rec['targets_sha256']}
    path = BASE / 'protocol/RULES.json'
    rule = {'study': 'Post-observation deterministic conditional-context screening',
        'plan_sha256': sha(BASE / 'PLAN_ZH.md'), 'sources': frozen_sources(),
        'preceding_training_sources': source_hashes(), 'original_budget_sha256': sha(V3 / 'protocol/BUDGET.json'),
        'original_data_protocol_sha256': sha(PRIOR / 'protocol/DATA.json'),
        'source_admission_sha256': sha(SOURCE / 'admission.json'),
        'original_purged_partitions': partitions,
        'fixed_controls': ['zero', 'A_mean_raw', 'A_mean_shrink', 'B_context_raw', 'B_context_shrink',
            'C_history_raw', 'C_history_shrink', 'archived_default_raw', 'archived_default_shrink',
            'default_three_seed_raw', 'default_three_seed_shrink', 'repair_three_seed_raw', 'repair_three_seed_shrink',
            'default_each_seed_2026_2027_2028_raw_and_shrink', 'repair_each_seed_2026_2027_2028_raw_and_shrink',
            'ridge_history', 'ridge_history_anchor', 'ridge_history_selected', 'ridge_history_anchor_selected',
            'fit_mean_raw', 'fit_mean_shrink'],
        'raw_diagnostic_contrasts': ['D_joint_raw_vs_C_history_raw', 'D_joint_raw_vs_B_context_raw',
            'B_context_raw_vs_A_mean_raw'],
        'deadline_epoch': DEADLINE, 'tasks': ['minute_h60_future_log_return', 'hourly_h8_future_log_return'],
        'screen_folds': [1, 2], 'candidate': 'D_joint_shrink',
        'feature_names': {p: feature_names(h, l) for p, h, l in [('minute', 60, 480), ('hourly', 8, 96)]},
        'cells': {k: v.tolist() for k, v in CELLS.items()}, 'include_nominal_price_anchor': False,
        'objective': 'Original full cumulative-log-return path, deterministic fit-only normalized Ridge',
        'regularization': 0.1, 'intercept': 'Unpenalized fit-only target mean', 'batch_size': 2048,
        'screen_gate': 'D_joint raw inner-stop endpoint MSE strictly below zero and C_history on BOTH first folds; independentlyreported, requiredforindependentconfirmation',
        'full_admission': 'All40fixedcellsadmittedfromfit-onlycostbenchmark beforeanynewstop/outer scores; nooutputselection; originaldeadlineunchanged',
        'shrinkage': 'Each ofA/B/C/D has its original clipped[0,1] endpoint scalar from inner-stop only; applies to allpath steps',
        'seeds': 'Not applicable to deterministic solver; no triple identical pseudo-replication',
        'deterministic_performance_rule': '>=1% overzero,>=4/5positivefolds,both frozen block lengths against everyprelisted control and each of3 stochasticbenchmarkseeds',
        'original_neural_seed_gate_unchanged': True, 'independent_confirmation': False, 'Q3_values_read': False,
        'at_utc': now(), 'at_epoch': time.time()}
    if path.exists():
        old = read(path)
        assert {k: v for k, v in old.items() if k not in ('at_utc', 'at_epoch')} == \
               {k: v for k, v in rule.items() if k not in ('at_utc', 'at_epoch')}
    else:
        atomic(path, rule)
    return read(path)


def solve(stats, columns, ridge=0.1):
    n = int(stats['count']); sx = stats['sx']; sy = stats['sy']
    mx, my = sx/n, sy/n
    variance = np.maximum(np.diag(stats['xx'])/n - mx*mx, 0.)
    sd = np.sqrt(variance); sd = np.where(sd > 1e-12, sd, 1.)
    gram = (stats['xx']/n - np.outer(mx, mx)) / sd[:, None] / sd[None, :]
    rhs = (stats['xy']/n - np.outer(mx, my)) / sd[:, None]
    cols = np.asarray(columns, dtype=np.int64)
    coef = np.linalg.solve(gram[np.ix_(cols, cols)] + ridge*np.eye(len(cols)), rhs[cols]) if len(cols) else np.empty((0, len(my)))
    return {'coef': coef, 'x_mean': mx[cols], 'x_std': sd[cols], 'y_mean': my, 'columns': cols}


def fit(profile, horizon, fold, *, stop_after_batches=None):
    rules = freeze()
    cfg = make_job(profile, horizon, 'future_log_return', 'xpatch', 'anchor_baseline', fold)
    task = f'{profile}_h{horizon}_future_log_return'
    folder = BASE / 'fits' / task / f'fold_{fold}'; folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'INNER_DONE.json').exists():
        receipt = read(folder / 'INNER_DONE.json')
        assert receipt['rules_sha256'] == sha(BASE / 'protocol/RULES.json')
        assert all(sha(folder / name) == value for name, value in receipt['model_hashes'].items())
        return receipt
    data = Data(cfg); idx = data.splits['fit']; stop = data.splits['stop']; h = horizon
    assert data.bars.end_ns[idx[-1]+h] < data.bars.end_ns[stop[0]]
    assert data.bars.end_ns[stop[-1]+h] < data.bars.end_ns[data.splits['validation'][0]]
    identity = hashlib.sha256(json.dumps({'rules': sha(BASE / 'protocol/RULES.json'),
        'cfg': cfg, 'split_sha256': sha(data.rec['split_file'])}, sort_keys=True).encode()).hexdigest()
    partial = folder / 'FIT_ACCUM.npz'
    start = time.time(); batches = 0
    if partial.exists():
        with np.load(partial, allow_pickle=False) as a:
            assert str(a['identity']) == identity
            state = {k: a[k].copy() for k in ('cursor', 'count', 'sx', 'sy', 'xx', 'xy')}
    else:
        state = {'cursor': np.asarray(0), 'count': np.asarray(0), 'sx': np.zeros(17),
                 'sy': np.zeros(h), 'xx': np.zeros((17, 17)), 'xy': np.zeros((17, h))}
    cursor = int(state['cursor']); before = cursor
    while cursor < len(idx):
        if STOP or time.time() >= DEADLINE or (stop_after_batches is not None and batches >= stop_after_batches):
            npz(partial, identity=np.asarray(identity), **state)
            receipt = {'status': 'paused', 'identity': identity, 'cursor': cursor, 'fit_count': len(idx),
                'checkpoint_sha256': sha(partial), 'source_hashes': frozen_sources(), 'at_utc': now(),
                'deadline_epoch': DEADLINE, 'outer_scoring_started': False}
            atomic(folder / 'STATE.json', receipt)
            return receipt
        use = idx[cursor:cursor+rules['batch_size']]
        x, _ = data.windows.arrays(use)
        f = features(x, data.bars.end_ns[use], h)
        y = targets(data.bars, use, h, 'future_log_return')
        state['sx'] += f.sum(0); state['sy'] += y.sum(0)
        state['xx'] += f.T @ f; state['xy'] += f.T @ y
        cursor += len(use); batches += 1
        state['cursor'] = np.asarray(cursor); state['count'] = np.asarray(cursor)
        npz(partial, identity=np.asarray(identity), **state)
    assert int(state['count']) == len(idx)
    atomic(folder / 'FIT_DONE.json', {'status': 'completed', 'identity': identity,
        'fit_count': len(idx), 'resumed_from_cursor': before, 'fit_elapsed_seconds_this_invocation': time.time()-start,
        'sufficient_statistics_sha256': sha(partial), 'at_utc': now(), 'outer_labels_constructed': False})
    models = {}
    for name, columns in CELLS.items():
        model = solve(state, columns)
        npz(folder / f'{name}.npz', **model, horizon=np.asarray(h), alpha=np.asarray(1.))
        models[name] = Predictor(folder / f'{name}.npz')
    sums = {name: {'p2': 0., 'py': 0., 'raw_error': 0., 'path_p2': 0., 'path_py': 0., 'path_y2': 0.}
            for name in CELLS}
    zero_sum = 0.; example = None
    for cursor in range(0, len(stop), 2048):
        if STOP or time.time() >= DEADLINE:
            atomic(folder / 'STATE.json', {'status': 'fit_complete_inner_scoring_paused', 'identity': identity,
                'deadline_epoch': DEADLINE, 'at_utc': now()})
            return {'status': 'paused'}
        use = stop[cursor:cursor+2048]
        x, _ = data.windows.arrays(use); ns = data.bars.end_ns[use]
        y = targets(data.bars, use, h, 'future_log_return'); zero_sum += float(np.sum(y[:, -1]**2))
        for name, predictor in models.items():
            prediction = predictor.predict(x, ns)
            a = sums[name]
            a['p2'] += float(np.sum(prediction[:, -1]**2)); a['py'] += float(np.sum(prediction[:, -1]*y[:, -1]))
            a['raw_error'] += float(np.sum((prediction[:, -1]-y[:, -1])**2))
            a['path_p2'] += float(np.sum(prediction**2)); a['path_py'] += float(np.sum(prediction*y))
            a['path_y2'] += float(np.sum(y**2))
        if example is None:
            example = {'history': x[:128], 'completion_ns': ns[:128], 'origins': use[:128]}
    inner = {}; hashes = {}
    for name, model in models.items():
        a = sums[name]
        alpha = float(np.clip(a['py']/a['p2'], 0., 1.)) if a['p2'] > 0 else 0.
        inner[name] = {'raw_endpoint_mse': a['raw_error']/len(stop),
            'shrink_endpoint_mse': (alpha*alpha*a['p2']-2*alpha*a['py']+zero_sum)/len(stop),
            'raw_improvement_over_zero_pct': 100*(1-a['raw_error']/zero_sum), 'alpha': alpha,
            'raw_path_mse': (a['path_p2']-2*a['path_py']+a['path_y2'])/(len(stop)*h)}
        contents = {k: getattr(model, k) for k in ('coef', 'x_mean', 'x_std', 'y_mean', 'columns')}
        npz(folder / f'{name}.npz', **contents, horizon=np.asarray(h), alpha=np.asarray(alpha))
        hashes[f'{name}.npz'] = sha(folder / f'{name}.npz')
        expected = model.predict(example['history'], example['completion_ns'])
        npz(folder / f'{name}_EXAMPLE.npz', **example, expected_raw=expected, expected_shrink=alpha*expected)
    receipt = {'status': 'completed', 'identity': identity, 'task': task, 'fold': fold,
        'fit_count': len(idx), 'stop_count': len(stop), 'zero_inner_endpoint_mse': zero_sum/len(stop),
        'inner_metrics': inner, 'model_hashes': hashes, 'sufficient_statistics_sha256': sha(partial),
        'rules_sha256': sha(BASE / 'protocol/RULES.json'), 'source_hashes': frozen_sources(),
        'fit_labels_last_end_ns': int(data.bars.end_ns[idx[-1]+h]), 'stop_first_origin_end_ns': int(data.bars.end_ns[stop[0]]),
        'stop_labels_last_end_ns': int(data.bars.end_ns[stop[-1]+h]),
        'outer_first_origin_end_ns': int(data.bars.end_ns[data.splits['validation'][0]]),
        'outer_labels_constructed': False, 'seed': None, 'solver_deterministic': True,
        'fit_and_inner_wall_seconds_this_invocation': time.time()-start,
        'deadline_epoch': DEADLINE, 'at_utc': now(), 'at_epoch': time.time()}
    atomic(folder / 'INNER_DONE.json', receipt)
    print(json.dumps({'task': task, 'fold': fold, 'inner': inner, 'outer_scored': False}), flush=True)
    return receipt


def screen(*, recovery_probe=False):
    rules = freeze()
    receipts = []
    for profile, h in [('minute', 60), ('hourly', 8)]:
        for fold in (1, 2):
            result = fit(profile, h, fold, stop_after_batches=2 if recovery_probe else None)
            if result['status'] != 'completed':
                return result
            receipts.append(result)
    path = BASE / 'protocol/GATE_AND_ADMISSION.json'
    if path.exists():
        return read(path)
    tasks = {}; full = list(rules['tasks'])
    estimate = 0.
    for task in rules['tasks']:
        local = [r for r in receipts if r['task'] == task]
        directions = [r['inner_metrics']['D_joint']['raw_endpoint_mse'] < r['zero_inner_endpoint_mse'] and
                      r['inner_metrics']['D_joint']['raw_endpoint_mse'] < r['inner_metrics']['C_history']['raw_endpoint_mse'] for r in local]
        # The full existing task's total fit origin count is known from split
        # files; no future prices or labels are read for this projection.
        profile, hrest = task.split('_h'); h = int(hrest.split('_')[0])
        speed = max(r['fit_and_inner_wall_seconds_this_invocation']/r['fit_count'] for r in local)
        total = 0
        for fold in (3, 4, 5):
            data = Data(make_job(profile, h, 'future_log_return', 'xpatch', 'anchor_baseline', fold))
            total += len(data.splits['fit']) + len(data.splits['stop']) + len(data.splits['validation'])
        projected = speed * total * 3.
        tasks[task] = {'inner_gate_passed': all(directions), 'fold_directions': directions,
                       'projected_remaining_seconds_with_factor3': projected}
        estimate += projected
    time_ok = time.time() + estimate + 300 < DEADLINE
    receipt = {'status': 'frozen_before_any_new_candidate_outer_scoring', 'tasks': tasks,
        'inner_direction_supported_tasks': [task for task, r in tasks.items() if r['inner_gate_passed']],
        'full_tasks': full if time_ok else [],
        'compute_admitted': time_ok, 'estimated_remaining_seconds': estimate, 'reserve_seconds': 300,
        'deadline_epoch': DEADLINE, 'rules_sha256': sha(BASE / 'protocol/RULES.json'),
        'screen_receipt_hashes': {f"{r['task']}_f{r['fold']}": sha(BASE / 'fits' / r['task'] / f"fold_{r['fold']}" / 'INNER_DONE.json') for r in receipts},
        'outer_labels_used_for_admission': False, 'Q3_values_read': False, 'independent_confirmation': False,
        'at_utc': now(), 'at_epoch': time.time()}
    atomic(path, receipt)
    print(json.dumps(receipt), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['freeze', 'screen', 'expand'], default='screen')
    parser.add_argument('--recovery-probe', action='store_true')
    args = parser.parse_args()
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    freeze()
    if args.stage == 'freeze':
        print('Protocol frozen before new inner-stop scoring.', flush=True)
    elif args.stage == 'screen':
        screen(recovery_probe=args.recovery_probe)
    else:
        gate = read(BASE / 'protocol/GATE_AND_ADMISSION.json')
        assert gate['rules_sha256'] == sha(BASE / 'protocol/RULES.json')
        for task in gate['full_tasks']:
            profile, rest = task.split('_h'); h = int(rest.split('_')[0])
            for fold in (3, 4, 5):
                if fit(profile, h, fold)['status'] != 'completed':
                    return


if __name__ == '__main__':
    main()
