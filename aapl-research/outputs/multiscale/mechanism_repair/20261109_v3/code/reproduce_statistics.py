"""Recompute mechanism, cap and twenty-task summaries from portable saved losses."""
from pathlib import Path
import argparse, json, sys, hashlib
import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE/'reference_source'))
from multiscale.bars import Bars
from portable_stats import day_groups, paired_intervals, interaction


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def equal(row, expected):
    assert row['status'] == expected['status']
    for field in ('mse_reference_minus_primary', 'relative_improvement_pct', 'mean_loss_interaction'):
        if field in row:
            assert np.isclose(row[field], expected[field], rtol=1e-12, atol=1e-14), (field, row, expected)
    for field in ('difference_ci95', 'relative_ci95_pct'):
        if field in row and row[field] is not None:
            assert np.allclose(row[field], expected[field], rtol=1e-12, atol=1e-12), (field, row, expected)
    for field in ('supports_improvement', 'joint_gain_beyond_additivity_supported'):
        if field in row:
            assert row[field] == expected[field]


def losses_for(path):
    with np.load(path, allow_pickle=False) as a:
        return a['origins'], a['session_date'], {k[5:]: a[k] for k in a.files if k.startswith('loss_')}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, default=CODE.parent)
    args = parser.parse_args(); root = args.root.resolve()
    rep = read(root/'analysis/REPLICATION_RESULTS.json')
    candidate = read(root/'analysis/CANDIDATE_RESULTS.json'); cap = read(root/'analysis/CAP_RESULTS.json')
    data_manifest = read(root/'data/SOURCE_HASHES.json')['files']
    for name, digest in data_manifest.items():
        assert hashlib.sha256((root/'data'/name).read_bytes()).hexdigest() == digest
    bars = {}
    for frequency in ('month', 'day', 'hour', 'minute'):
        with np.load(root/'data'/f'bars_{frequency}.npz', allow_pickle=False) as a:
            bars[frequency] = Bars(frequency, **{k: a[k] for k in a.files})
        assert bars[frequency].session_date[-1] <= '2026-06-30'
    row_count = interval_count = interaction_count = 0

    def collect(folder, frequency):
        groups = []; totals = {}; n = 0
        for fold in range(1, 6):
            origins, dates, losses = losses_for(folder/f'fold_{fold}.npz')
            assert np.array_equal(dates, bars[frequency].session_date[origins])
            groups.append(day_groups(bars[frequency], origins, losses)); n += len(origins)
            for name, values in losses.items():
                totals[name] = totals.get(name, 0.)+float(values.sum())
        return groups, totals, n

    for group in rep['admission_groups']:
        if not group['admitted']:
            continue
        task, model = group['task'], group['model']; frequency = 'minute' if task.startswith('minute_') else 'hour'
        groups, totals, n = collect(root/'analysis/paired_losses'/task/model, frequency)
        baseline_name = 'zero' if task.endswith('future_log_return') else 'persistence'
        for row in rep['pooled']:
            if row['task'] == task and row['model'] == model:
                assert row['origins'] == n and np.isclose(totals[row['method']]/n, row['endpoint_mse'], rtol=1e-12)
                assert np.isclose(row['improvement_pct'], 100*(1-totals[row['method']]/totals[baseline_name]), rtol=1e-12, atol=1e-12)
                positive = sum(g['sums'][:, g['names'].index(row['method'])].sum() <
                               g['sums'][:, g['names'].index(baseline_name)].sum() for g in groups)
                assert row['positive_folds'] == positive
                row_count += 1
        expected = [r for r in rep['intervals'] if r['task'] == task and r['model'] == model]
        for block in ([1, 5] if frequency == 'minute' else [2, 5]):
            target_rows = [r for r in expected if r['block'] == block]
            pairs = [(r['primary'], r['reference']) for r in target_rows]
            actual = paired_intervals(groups, pairs, block, task)
            for a, e in zip(actual, target_rows):
                equal(a, e); interval_count += 1
            for e in [r for r in rep['interactions'] if r['task'] == task and r['model'] == model and r['block'] == block]:
                equal(interaction(groups, e['weights'], block, task, e['mode']), e); interaction_count += 1
    for task in sorted({r['task'] for r in candidate['pooled']}):
        profile = task.split('_h')[0]
        frequency = {'annual': 'month', 'quarterly': 'month', 'monthly': 'day', 'hourly': 'hour', 'minute': 'minute'}[profile]
        groups, totals, n = collect(root/'analysis/candidate_paired_losses'/task, frequency)
        baseline_name = 'zero' if task.endswith('future_log_return') else 'persistence'
        for row in candidate['pooled']:
            if row['task'] == task:
                assert row['origins'] == n and np.isclose(totals[row['method']]/n, row['endpoint_mse'], rtol=1e-12)
                assert np.isclose(row['improvement_pct'], 100*(1-totals[row['method']]/totals[baseline_name]), rtol=1e-12, atol=1e-12)
                positive = sum(g['sums'][:, g['names'].index(row['method'])].sum() <
                               g['sums'][:, g['names'].index(baseline_name)].sum() for g in groups)
                assert row['positive_folds'] == positive
                row_count += 1
        if profile in ('annual', 'quarterly', 'monthly'):
            groups = [{'names': groups[0]['names'], 'counts': np.concatenate([g['counts'] for g in groups]),
                       'sums': np.concatenate([g['sums'] for g in groups])}]
        for block in sorted({r['block'] for r in candidate['intervals'] if r['task'] == task}):
            expected = [r for r in candidate['intervals'] if r['task'] == task and r['block'] == block]
            pairs = [(r['primary'], r['reference']) for r in expected]
            for a, e in zip(paired_intervals(groups, pairs, block, task), expected):
                equal(a, e); interval_count += 1
    for model in ('lstm', 'xpatch'):
        groups, totals, n = collect(root/'analysis/cap_paired_losses'/model, 'hour')
        for row in cap['pooled']:
            if row['model'] == model:
                assert row['origins'] == n and np.isclose(totals[row['method']]/n, row['endpoint_mse'], rtol=1e-12)
                assert np.isclose(row['improvement_over_persistence_pct'],
                                  100*(1-totals[row['method']]/totals['persistence']), rtol=1e-12, atol=1e-12)
                row_count += 1
        for block in (2, 5):
            expected = [r for r in cap['intervals'] if r['model'] == model and r['block'] == block]
            pairs = [(r['primary'], r['reference']) for r in expected]
            for a, e in zip(paired_intervals(groups, pairs, block, 'hourly_h1_future_close'), expected):
                equal(a, e); interval_count += 1
    print(json.dumps({'status': 'passed', 'pooled_rows': row_count, 'paired_intervals': interval_count,
                      'factorial_interactions': interaction_count, 'Q3_values_read': False,
                      'scope': 'all registered mechanism, cap, fixed-candidate saved-loss statistics'}), flush=True)


if __name__ == '__main__':
    main()
