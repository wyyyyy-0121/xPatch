"""Portable saved-loss/forecast arithmetic for v3a; no workspace or network imports."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from portable_stats import day_groups, paired_intervals


def read(p):
    return json.loads(Path(p).read_text())


def close(a, b):
    assert np.allclose(a, b, rtol=1e-11, atol=1e-12), (a, b)


def compare_interval(a, b):
    for k in ('status', 'supports_improvement', 'task', 'primary', 'reference', 'block'):
        assert a[k] == b[k], (k, a, b)
    for k in ('mse_reference_minus_primary', 'relative_improvement_pct', 'difference_ci95', 'relative_ci95_pct'):
        if k in b:
            if b[k] is None:
                assert a[k] is None
            else:
                close(a[k], b[k])


def join(groups):
    return [{'names': groups[0]['names'], 'counts': np.concatenate([g['counts'] for g in groups]),
             'sums': np.concatenate([g['sums'] for g in groups])}]


def intervals(groups, expected, task):
    n = 0
    for block in sorted({r['block'] for r in expected}):
        rows = [r for r in expected if r['block'] == block]
        computed = paired_intervals(groups, [(r['primary'], r['reference']) for r in rows], block, task)
        for a, b in zip(computed, rows):
            compare_interval(a, b)
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    dates = {}
    for freq in ('month', 'day', 'hour', 'minute'):
        with np.load(root / 'data' / f'bars_{freq}.npz', allow_pickle=False) as a:
            dates[freq] = SimpleNamespace(session_date=a['session_date'].copy())
        assert dates[freq].session_date[-1] <= '2026-06-30'
    frequency = {'annual': 'month', 'quarterly': 'month', 'monthly': 'day', 'hourly': 'hour', 'minute': 'minute'}
    long = read(root / 'longscale_replication/analysis/RESULTS.json')
    mean = read(root / 'mean_controls/audit/RESULTS.json')
    hist = read(root / 'history_sensitivity/analysis/RESULTS.json')
    count = {'longscale_pooled': 0, 'longscale_seed_rows': 0, 'longscale_intervals': 0,
             'longscale_sensitivity': 0, 'mean_pooled': 0, 'mean_intervals': 0,
             'history_pooled': 0, 'history_rows': 0, 'history_intervals': 0}
    for task in dict.fromkeys(r['task'] for r in long['pooled']):
        with np.load(root / 'longscale_replication/analysis/paired_losses' / f'{task}.npz', allow_pickle=False) as a:
            origins = a['origins'].copy(); folds = a['fold'].copy()
            losses = {k[5:]: a[k].copy() for k in a.files if k.startswith('loss_')}
            assert np.array_equal(a['session_date'], dates[frequency[task.split('_h')[0]]].session_date[origins])
            origin_dates = a['session_date'].copy(); endpoint_dates = a['endpoint_date'].copy()
        base = 'zero' if task.endswith('future_log_return') else 'persistence'
        denom = np.mean(losses[base])
        for r in [r for r in long['pooled'] if r['task'] == task]:
            assert r['origins'] == len(origins)
            close(r['endpoint_mse'], np.mean(losses[r['method']]))
            close(r['improvement_pct'], 100 * (1 - np.mean(losses[r['method']]) / denom))
            assert r['positive_folds'] == sum(np.mean(losses[r['method']][folds == f]) <
                np.mean(losses[base][folds == f]) for f in range(1, 6))
            count['longscale_pooled'] += 1
        groups = [day_groups(dates[frequency[task.split('_h')[0]]], origins[folds == f],
                             {k: v[folds == f] for k, v in losses.items()}) for f in range(1, 6)]
        count['longscale_intervals'] += intervals(join(groups),
            [r for r in long['paired_intervals'] if r['task'] == task], task)
        horizon = int(task.split('_h')[1].split('_')[0])
        greedy = np.zeros(len(origins), dtype=bool); last = -10**10
        for i, o in enumerate(origins):
            if o >= last + horizon:
                greedy[i] = True; last = int(o)
        for r in [r for r in long['sensitivity'] if r['task'] == task]:
            rule = r['rule']
            if rule == 'original_global_earliest_greedy':
                mask = greedy
            elif rule == 'absolute_bar_phase':
                mask = origins % horizon == r['offset']
            elif rule == 'leave_origin_year_out':
                mask = np.asarray([str(d)[:4] != r['year'] for d in origin_dates])
            elif rule in ('crosses_split', 'no_split'):
                mask = np.zeros(len(origins), dtype=bool)
                for date in r['split_dates']:
                    mask |= (origin_dates < date) & (endpoint_dates >= date)
                if rule == 'no_split':
                    mask = ~mask
            else:
                raise AssertionError(rule)
            assert r['origins'] == int(mask.sum())
            if mask.any():
                d = np.mean(losses[base][mask])
                for name, value in r['methods'].items():
                    close(value['endpoint_mse'], np.mean(losses[name][mask]))
                    close(value['improvement_pct'], 100 * (1 - np.mean(losses[name][mask]) / d))
            else:
                assert r['status'] == 'not_estimable'
            count['longscale_sensitivity'] += 1

    # Per-seed forecasts remain separate from the mean-seed loss tables.
    manifest = read(root / 'longscale_replication/protocol/JOBS.json')
    locations = read(root / 'PORTABLE_LOCATIONS.json')
    for r in long['rows']:
        task, fold, seed = r['task'], r['fold'], r['seed']
        kind, mode = r['method'].split('_')
        cfg = next(c for c in manifest['jobs'] if
            f"{c['profile']}_h{c['horizon']}_{c['target']}" == task and c['fold'] == fold and c['seed'] == seed and
            c['initialization'] == ('baseline' if kind == 'repair' else 'random'))
        with np.load(root / 'history_sensitivity/analysis' / task / f'fold_{fold}.npz', allow_pickle=False) as a:
            truth = a['actual']; baseline = a['baseline']
        file = (root / locations[cfg['id']] / 'predictions.npz') if mode == 'trained' else \
            (root / 'longscale_replication/calibration' / cfg['id'] / 'predictions.npz')
        with np.load(file, allow_pickle=False) as a:
            prediction = a['trained' if mode == 'trained' else 'prediction']
        close(r['endpoint_mse'], np.mean((prediction[:, -1] - truth[:, -1])**2))
        close(r['path_mse'], np.mean((prediction - truth)**2))
        close(r['endpoint_mae'], np.mean(np.abs(prediction[:, -1] - truth[:, -1])))
        close(r['baseline_mse'], np.mean((baseline[:, -1] - truth[:, -1])**2))
        count['longscale_seed_rows'] += 1

    for task in dict.fromkeys(r['task'] for r in mean['pooled']):
        groups = []; byfold = []; arrays = []
        rows = [r for r in mean['pooled'] if r['task'] == task]
        names = [r['method'] for r in rows]
        bars = dates[frequency[task.split('_h')[0]]]
        for f in range(1, 6):
            with np.load(root / 'mean_controls/predictions' / task / f'fold_{f}.npz', allow_pickle=False) as a:
                losses = {name: a['loss_' + name].copy() for name in names}
                assert np.array_equal(a['session_date'], bars.session_date[a['origins']])
                groups.append(day_groups(bars, a['origins'], losses))
                byfold.append({name: float(v.mean()) for name, v in losses.items()})
                arrays.append({k: a[k].copy() for k in a.files if k.startswith(('loss_', 'path_loss_', 'absolute_'))})
        totals = {name: np.concatenate([a['loss_' + name] for a in arrays]) for name in names}
        d = np.mean(totals['baseline'])
        for r in rows:
            name = r['method']
            close(r['endpoint_mse'], np.mean(totals[name]))
            close(r['improvement_pct'], 100 * (1 - np.mean(totals[name]) / d))
            assert r['origins'] == len(totals[name])
            assert r['positive_folds'] == sum(a[name] < a['baseline'] for a in byfold)
            for field, prefix in [('path_mse', 'path_loss_'), ('endpoint_mae', 'absolute_')]:
                if r[field] is not None:
                    if all(prefix + name in a for a in arrays):
                        close(r[field], np.mean(np.concatenate([a[prefix + name] for a in arrays])))
                    else:
                        # Prior v3 methods retain endpoint losses here; their
                        # auxiliary path/MAE values have weighted fold receipts.
                        # Their independent per-origin replay belongs to v3.
                        assert name.startswith('v3_')
                        receipts = [a for a in mean['fold_results'] if a['task'] == task]
                        close(r[field], sum(a['metrics'][name][field] * a['origins'] for a in receipts) /
                              sum(a['origins'] for a in receipts))
            count['mean_pooled'] += 1
        if task.split('_h')[0] in ('annual', 'quarterly', 'monthly'):
            groups = join(groups)
        count['mean_intervals'] += intervals(groups, [r for r in mean['intervals'] if r['task'] == task], task)

    for task in dict.fromkeys(r['task'] for r in hist['pooled']):
        bars = dates[frequency[task.split('_h')[0]]]
        groups = []; totals = {}
        for fold in range(1, 6):
            with np.load(root / 'history_sensitivity/analysis' / task / f'fold_{fold}.npz', allow_pickle=False) as a:
                origins = a['origins']; truth = a['actual']; baseline = a['baseline']
                losses = {}
                for mode in ('actual', 'fit_mean_history', 'flat_history'):
                    seedloss = []
                    for seed in (2026, 2027, 2028):
                        pred = a[f'{mode}_seed_{seed}']
                        loss = (pred[:, -1] - truth[:, -1])**2
                        seedloss.append(loss)
                        r = next(r for r in hist['rows'] if r['task'] == task and r['fold'] == fold and r['seed'] == seed)
                        close(r['metrics'][mode]['endpoint_mse'], np.mean(loss))
                        close(r['metrics'][mode]['improvement_pct'],
                            100 * (1 - np.mean(loss) / np.mean((baseline[:, -1] - truth[:, -1])**2)))
                        if mode == 'actual':
                            count['history_rows'] += 1
                    losses[mode] = np.mean(seedloss, axis=0)
                losses['baseline'] = (baseline[:, -1] - truth[:, -1])**2
                groups.append(day_groups(bars, origins, losses))
                for name, v in losses.items():
                    totals.setdefault(name, []).append(v)
        totals = {name: np.concatenate(v) for name, v in totals.items()}
        for r in [r for r in hist['pooled'] if r['task'] == task]:
            close(r['endpoint_mse'], np.mean(totals[r['method']]))
            close(r['improvement_over_baseline_pct'],
                100 * (1 - np.mean(totals[r['method']]) / np.mean(totals['baseline'])))
            count['history_pooled'] += 1
        count['history_intervals'] += intervals(join(groups),
            [r for r in hist['paired_intervals'] if r['task'] == task], task)
    assert count == {'longscale_pooled': 66, 'longscale_seed_rows': 360, 'longscale_intervals': 120,
                     'longscale_sensitivity': 130, 'mean_pooled': 280, 'mean_intervals': 280,
                     'history_pooled': 24, 'history_rows': 90, 'history_intervals': 24}, count
    print(json.dumps({'status': 'passed', **count, 'Q3_values_read': False,
                      'preceding_aux_path_and_mae_source': 'weighted fold receipts; original per-origin reproduction in v3',
                      'scope': 'saved supplemental losses/forecasts, not independent market validation'}), flush=True)


if __name__ == '__main__':
    main()
