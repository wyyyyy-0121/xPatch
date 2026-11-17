"""Frozen outer evaluation of all four cells, with every prelisted control."""
from pathlib import Path
import argparse
import os
import signal
import sys
import time
import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'code'))
from experiment import freeze, CELLS, DEADLINE, V3, V3A, PRIOR, make_job
from predictor import Predictor
from study import read, sha, atomic, npz, now, metrics
from engine import Data
from portable_stats import day_groups, paired_intervals
from multiscale.bars import targets

STOP = False
CANDIDATE = 'D_joint_shrink'
DIAGNOSTIC_PAIRS = [('D_joint_raw', 'C_history_raw'), ('D_joint_raw', 'B_context_raw'),
                    ('B_context_raw', 'A_mean_raw')]


def interrupted(*_):
    global STOP
    STOP = True


def control_paths(task, fold):
    """Enumerate frozen inputs without computing any outer scores."""
    rec = read(PRIOR / 'cache' / task / 'TASK.json')['folds'][str(fold)]
    paths = [Path(rec['targets']), Path(rec['split_file']),
        V3 / 'analysis/candidate_paired_losses' / task / f'fold_{fold}.npz',
        V3 / 'analysis/paired_losses' / task / 'xpatch' / f'fold_{fold}.npz',
        V3A / 'mean_controls/predictions' / task / f'fold_{fold}.npz']
    for variant in ('anchor_random', 'anchor_baseline'):
        for seed in (2026, 2027, 2028):
            job = f'{task}_xpatch_{variant}_f{fold}_s{seed}'
            paths += [V3 / 'runs' / job / name for name in ('DONE.json', 'predictions.npz')]
            paths += [V3 / 'calibration' / job / name for name in ('DONE.json', 'predictions.npz')]
    local = BASE / 'fits' / task / f'fold_{fold}'
    paths += [local / 'INNER_DONE.json'] + [local / f'{cell}.npz' for cell in CELLS]
    return paths


def freeze_evaluation():
    rules = freeze()
    gate = read(BASE / 'protocol/GATE_AND_ADMISSION.json')
    assert gate['compute_admitted'] and gate['full_tasks'] == rules['tasks']
    inputs = {}
    for task in rules['tasks']:
        for fold in range(1, 6):
            folder = BASE / 'fits' / task / f'fold_{fold}'
            done = read(folder / 'INNER_DONE.json')
            assert done['status'] == 'completed' and done['rules_sha256'] == sha(BASE / 'protocol/RULES.json')
            assert all(sha(folder / name) == digest for name, digest in done['model_hashes'].items())
            for path in control_paths(task, fold):
                inputs[str(path)] = sha(path)
    value = {'status': 'frozen_before_new_outer_scoring',
        'rules_sha256': sha(BASE / 'protocol/RULES.json'),
        'gate_sha256': sha(BASE / 'protocol/GATE_AND_ADMISSION.json'),
        'evaluator_sha256': sha(__file__), 'portable_stats_sha256': sha(V3 / 'code/portable_stats.py'),
        'input_hashes': inputs, 'candidate': CANDIDATE, 'diagnostic_pairs': DIAGNOSTIC_PAIRS,
        'blocks': {'minute': [1, 5], 'hourly': [2, 5]}, 'repetitions': 5000, 'statistics_seed': 2026,
        'scope': 'Exposed historical development; no multiplicity correction or independent confirmation',
        'deadline_epoch': DEADLINE, 'Q3_values_read': False, 'at_utc': now(), 'at_epoch': time.time()}
    path = BASE / 'protocol/EVALUATION_FREEZE.json'
    # JSON canonicalizes tuples to lists; normalize before comparison.
    import json
    value = json.loads(json.dumps(value))
    if path.exists():
        old = read(path)
        assert {k: v for k, v in old.items() if k not in ('at_utc', 'at_epoch')} == \
               {k: v for k, v in value.items() if k not in ('at_utc', 'at_epoch')}
    else:
        atomic(path, value)
    return read(path)


def load_arrays(path, origins, expected_sha=None):
    if expected_sha is not None:
        assert sha(path) == expected_sha
    with np.load(path, allow_pickle=False) as a:
        assert np.array_equal(a['origins'], origins), str(path)
        return {key: a[key].copy() for key in a.files if key != 'origins'}


def load_controls(task, fold, origins, y):
    candidate = load_arrays(V3 / 'analysis/candidate_paired_losses' / task / f'fold_{fold}.npz', origins)
    factorial = load_arrays(V3 / 'analysis/paired_losses' / task / 'xpatch' / f'fold_{fold}.npz', origins)
    means = load_arrays(V3A / 'mean_controls/predictions' / task / f'fold_{fold}.npz', origins)
    zero = y[:, -1] ** 2
    assert np.array_equal(zero, candidate['loss_zero']) and np.array_equal(zero, factorial['loss_zero'])
    assert np.array_equal(zero, means['loss_baseline'])
    losses = {'zero': zero,
        'archived_default_raw': candidate['loss_original'],
        'archived_default_shrink': candidate['loss_original_shrink'],
        'fit_mean_raw': means['loss_raw'], 'fit_mean_shrink': means['loss_shrink']}
    for name in ('history', 'history_anchor', 'history_selected', 'history_anchor_selected'):
        losses['ridge_' + name] = candidate['loss_ridge_' + name]
        assert np.array_equal(losses['ridge_' + name], factorial['loss_ridge_' + name])
    for prefix, variant in [('default', 'anchor_random'), ('repair', 'anchor_baseline')]:
        seed_losses = {'raw': [], 'shrink': []}
        for seed in (2026, 2027, 2028):
            job = f'{task}_xpatch_{variant}_f{fold}_s{seed}'
            folder = V3 / 'runs' / job; done = read(folder / 'DONE.json')
            raw = load_arrays(folder / 'predictions.npz', origins, done['prediction_sha256'])['trained']
            caldir = V3 / 'calibration' / job; cal = read(caldir / 'DONE.json')
            assert cal['source_checkpoint_sha256'] == done['checkpoint_sha256']
            shrunk = load_arrays(caldir / 'predictions.npz', origins, cal['prediction_sha256'])['prediction']
            # Return targets use an exact zero anchor. Preserve original saved
            # calibration arithmetic rather than refitting on outer labels.
            assert np.allclose(shrunk, raw * cal['alpha'], rtol=0, atol=1e-15)
            for mode, pred in [('raw', raw), ('shrink', shrunk)]:
                error = np.square(pred[:, -1] - y[:, -1])
                losses[f'{prefix}_seed_{seed}_{mode}'] = error
                seed_losses[mode].append(error)
        for mode, key in [('raw', 'trained'), ('shrink', 'shrink')]:
            pooled = np.mean(seed_losses[mode], axis=0)
            assert np.allclose(pooled, factorial[f'loss_{variant}__{key}'], rtol=1e-13, atol=1e-18)
            losses[f'{prefix}_three_seed_{mode}'] = pooled
    for name, value in losses.items():
        assert value.shape == (len(origins),) and np.isfinite(value).all() and (value >= 0).all(), name
    return losses


def evaluate_fold(task, fold):
    rules = read(BASE / 'protocol/RULES.json')
    profile, rest = task.split('_h'); horizon = int(rest.split('_')[0])
    data = Data(make_job(profile, horizon, 'future_log_return', 'xpatch', 'anchor_baseline', fold))
    vi = data.splits['validation']; rec = data.rec
    assert sha(rec['targets']) == rec['targets_sha256']
    with np.load(rec['targets'], allow_pickle=False) as a:
        assert np.array_equal(a['origins'], vi)
        y = a['actual']; anchor = a['anchor']
    assert np.array_equal(y, targets(data.bars, vi, horizon, 'future_log_return'))
    losses = load_controls(task, fold, vi, y)
    folder = BASE / 'fits' / task / f'fold_{fold}'
    models = {cell: Predictor(folder / f'{cell}.npz') for cell in CELLS}
    predicted = {cell: [] for cell in CELLS}; example = None
    for start in range(0, len(vi), 2048):
        if STOP or time.time() >= DEADLINE:
            raise InterruptedError('Preserve complete preceding folds; replay this bounded fold on resume')
        idx = vi[start:start+2048]
        x, _ = data.windows.arrays(idx); stamps = data.bars.end_ns[idx]
        for cell, model in models.items():
            predicted[cell].append(model.predict(x, stamps))
        if example is None:
            example = {'history': x[:128], 'completion_ns': stamps[:128], 'origins': idx[:128]}
    rows = []; path_losses = {}
    for cell, model in models.items():
        raw = np.concatenate(predicted[cell])
        for mode, pred in [('raw', raw), ('shrink', model.alpha * raw)]:
            name = cell + '_' + mode
            losses[name] = np.square(pred[:, -1] - y[:, -1])
            path_losses[name] = np.mean(np.square(pred - y), axis=1)
            rows.append({'task': task, 'fold': fold, 'cell': cell, 'method': name,
                'mode': mode, **metrics(pred, y, anchor, 'future_log_return'),
                'alpha': model.alpha, 'training_seed': None, 'deterministic_solver': True})
        npz(BASE / 'delivery' / task / f'fold_{fold}' / f'{cell}.npz', **example,
            expected_raw=raw[:len(example['origins'])],
            expected_shrink=model.alpha * raw[:len(example['origins'])])
    out = BASE / 'analysis/paired_losses' / task / f'fold_{fold}.npz'
    dates = data.bars.session_date[vi]
    day_grid = np.unique(data.bars.session_date[(data.bars.session_date >= dates[0]) &
                                               (data.bars.session_date <= dates[-1])])
    npz(out, origins=vi, session_date=dates, day_grid=day_grid, **losses)
    path_out = BASE / 'analysis/path_losses' / task / f'fold_{fold}.npz'
    npz(path_out, origins=vi, **path_losses)
    atomic(BASE / 'analysis/folds' / task / f'fold_{fold}.json', {'status': 'completed',
        'task': task, 'fold': fold, 'metrics': rows,
        'paired_loss_sha256': sha(out), 'evaluation_freeze_sha256': sha(BASE / 'protocol/EVALUATION_FREEZE.json'),
        'path_loss_sha256': sha(path_out),
        'model_hashes': read(folder / 'INNER_DONE.json')['model_hashes'],
        'origins': len(vi), 'loss_names': list(losses), 'Q3_values_read': False, 'at_utc': now()})
    print(f'Outer matrix complete: {task} fold {fold}', flush=True)


def statistics():
    rules = read(BASE / 'protocol/RULES.json'); evalrule = read(BASE / 'protocol/EVALUATION_FREEZE.json')
    gate = read(BASE / 'protocol/GATE_AND_ADMISSION.json')
    result = {'status': 'complete_frozen_development_matrix', 'candidate': CANDIDATE,
        'rules_sha256': sha(BASE / 'protocol/RULES.json'),
        'evaluation_freeze_sha256': sha(BASE / 'protocol/EVALUATION_FREEZE.json'),
        'tasks': {}, 'Q3_values_read': False, 'independent_confirmation': False,
        'research_goal_achieved': False, 'at_utc': now()}
    all_rows = []
    for task in rules['tasks']:
        profile, rest = task.split('_h'); horizon = int(rest.split('_')[0])
        groups = []; totals = {}; byfold = {}; n = 0
        for fold in range(1, 6):
            path = BASE / 'analysis/paired_losses' / task / f'fold_{fold}.npz'
            receipt = read(BASE / 'analysis/folds' / task / f'fold_{fold}.json')
            assert sha(path) == receipt['paired_loss_sha256']
            with np.load(path, allow_pickle=False) as a:
                origins = a['origins']; losses = {k: a[k] for k in a.files if k not in ('origins', 'session_date', 'day_grid')}
            data = Data(make_job(profile, horizon, 'future_log_return', 'xpatch', 'anchor_baseline', fold))
            groups.append(day_groups(data.bars, origins, losses)); n += len(origins)
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.) + float(value.sum())
                byfold.setdefault(name, []).append(float(value.mean()))
            all_rows.extend(receipt['metrics'])
        summary = {name: {'origins': n, 'endpoint_mse': total / n,
            'improvement_pct': 100 * (1 - total / totals['zero']),
            'positive_folds': sum(x < z for x, z in zip(byfold[name], byfold['zero'])),
            'fold_mse': byfold[name]} for name, total in totals.items()}
        controls = [name for name in totals if name not in (CANDIDATE, 'D_joint_raw')]
        assert len(controls) == 31
        pairs = [(CANDIDATE, control) for control in controls] + DIAGNOSTIC_PAIRS
        intervals = []
        for block in evalrule['blocks'][profile]:
            if STOP or time.time() >= DEADLINE:
                raise InterruptedError('Statistical summaries can resume from saved fold losses')
            intervals.extend(paired_intervals(groups, pairs, block, task))
        comparisons = {control: [r for r in intervals if r['primary'] == CANDIDATE and r['reference'] == control]
                       for control in controls}
        check = {'minimum_one_percent': summary[CANDIDATE]['improvement_pct'] >= 1.,
            'four_positive_folds': summary[CANDIDATE]['positive_folds'] >= 4,
            'all_control_intervals_both_blocks': all(len(rows) == 2 and all(r['supports_improvement'] for r in rows)
                                                     for rows in comparisons.values())}
        perf = all(check.values()); inner = gate['tasks'][task]['inner_gate_passed']
        result['tasks'][task] = {'summary': summary, 'comparisons': comparisons,
            'diagnostic_contrasts': [r for r in intervals if (r['primary'], r['reference']) in DIAGNOSTIC_PAIRS],
            'checks': check, 'performance_gate_passed': perf,
            'inner_direction_gate_passed': inner, 'independent_confirmation_eligible': perf and inner,
            'training_seed_count': None, 'deterministic_solver': True}
        print(task, summary[CANDIDATE], check, flush=True)
    result['both_primary_performance_gates_passed'] = all(x['performance_gate_passed'] for x in result['tasks'].values())
    result['independent_confirmation_eligible'] = all(x['independent_confirmation_eligible'] for x in result['tasks'].values())
    atomic(BASE / 'analysis/STATISTICS.json', result)
    atomic(BASE / 'analysis/OUTER_RESULTS.json', {'status': 'completed', 'matrix_cells': 40,
        'task_count': 2, 'folds_per_task': 5, 'metrics': all_rows, 'statistics_sha256': sha(BASE / 'analysis/STATISTICS.json'),
        'Q3_values_read': False, 'independent_confirmation': False, 'at_utc': now()})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--stage', choices=['freeze', 'run'], default='run')
    args = parser.parse_args()
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    rules = freeze_evaluation()
    if args.stage == 'freeze':
        print('Outer evaluator and complete source controls frozen before scoring.', flush=True); return
    try:
        for task in read(BASE / 'protocol/RULES.json')['tasks']:
            for fold in range(1, 6):
                receipt_path = BASE / 'analysis/folds' / task / f'fold_{fold}.json'
                if receipt_path.exists():
                    receipt = read(receipt_path)
                    assert receipt['evaluation_freeze_sha256'] == sha(BASE / 'protocol/EVALUATION_FREEZE.json')
                    assert sha(BASE / 'analysis/paired_losses' / task / f'fold_{fold}.npz') == receipt['paired_loss_sha256']
                    continue
                evaluate_fold(task, fold)
        statistics()
    except InterruptedError as exc:
        atomic(BASE / 'audit/OUTER_PAUSED.json', {'reason': str(exc), 'at_utc': now(),
            'deadline_epoch': DEADLINE, 'completed_folds_retained': True, 'safe_to_resume': True})


if __name__ == '__main__':
    main()
