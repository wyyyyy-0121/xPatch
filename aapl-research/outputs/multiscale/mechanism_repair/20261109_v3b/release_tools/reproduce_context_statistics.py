"""Portable reconstruction of the complete frozen context-study statistics."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_statistics import close, grouped_days, paired_intervals, compare_intervals


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(); root = args.root.resolve()
    rules = read(root/'protocol/RULES.json')
    evaluation = read(root/'protocol/EVALUATION_FREEZE.json')
    gate = read(root/'protocol/GATE_AND_ADMISSION.json')
    stats = read(root/'analysis/STATISTICS.json')
    outer = read(root/'analysis/OUTER_RESULTS.json')
    rule_hash = sha(root/'protocol/RULES.json')
    eval_hash = sha(root/'protocol/EVALUATION_FREEZE.json')
    assert stats['rules_sha256'] == evaluation['rules_sha256'] == rule_hash
    assert stats['evaluation_freeze_sha256'] == eval_hash
    assert evaluation['gate_sha256'] == sha(root/'protocol/GATE_AND_ADMISSION.json')
    assert outer['statistics_sha256'] == sha(root/'analysis/STATISTICS.json')
    assert sha(root/'PLAN_ZH.md') == rules['plan_sha256']
    assert sha(root/'code/evaluate_context.py') == evaluation['evaluator_sha256']
    assert sha(root/'code/portable_stats_registered.py') == evaluation['portable_stats_sha256']
    assert rules['candidate'] == stats['candidate'] == evaluation['candidate'] == 'D_joint_shrink'
    assert rules['tasks'] == list(stats['tasks'])
    assert evaluation['repetitions'] == 5000 and evaluation['statistics_seed'] == 2026
    assert all(item['Q3_values_read'] is False for item in (rules, evaluation, stats, outer))
    assert rules['independent_confirmation'] is stats['independent_confirmation'] is outer['independent_confirmation'] is False
    for name, expected in rules['sources'].items():
        assert sha(root/'code'/name) == expected
    candidate = rules['candidate']
    counts = {'pooled_rows': 0, 'endpoint_metric_rows': 0, 'mae_metric_rows': 0,
              'path_metric_rows': 0, 'paired_intervals': 0, 'folds': 0}
    performance = []; eligible = []
    for task in rules['tasks']:
        desired = stats['tasks'][task]
        groups = []; losses_by_fold = []; metric_by_fold = []; all_losses = {}
        inner_passes = []
        for fold in range(1, 6):
            receipt = read(root/'analysis/folds'/task/f'fold_{fold}.json')
            source = root/'analysis/paired_losses'/task/f'fold_{fold}.npz'
            assert sha(source) == receipt['paired_loss_sha256']
            assert receipt['evaluation_freeze_sha256'] == eval_hash
            with np.load(source, allow_pickle=False) as a:
                losses = {key: a[key].copy() for key in a.files if key not in ('origins', 'session_date', 'day_grid')}
                origins = a['origins'].copy()
                assert len(origins) == receipt['origins'] and np.all(np.diff(origins) > 0)
                assert str(a['session_date'][-1]) <= '2026-06-30'
                assert list(losses) == receipt['loss_names']
                groups.append(grouped_days(a['session_date'], a['day_grid'], losses))
            losses_by_fold.append(losses)
            for name, value in losses.items():
                assert (value >= 0).all()
                all_losses.setdefault(name, []).append(value)
            model_done = read(root/'fits'/task/f'fold_{fold}'/'INNER_DONE.json')
            assert receipt['model_hashes'] == model_done['model_hashes']
            if fold <= 2:
                im = model_done['inner_metrics']
                inner_passes.append(im['D_joint']['raw_endpoint_mse'] < model_done['zero_inner_endpoint_mse'] and
                    im['D_joint']['raw_endpoint_mse'] < im['C_history']['raw_endpoint_mse'])
            metrics = [row for row in outer['metrics'] if row['task'] == task and row['fold'] == fold]
            assert len(metrics) == len(receipt['metrics']) == 8
            assert metrics == receipt['metrics']
            path_file = root/'analysis/path_losses'/task/f'fold_{fold}.npz'
            path_losses = {}
            if path_file.exists():
                if 'path_loss_sha256' in receipt:
                    assert sha(path_file) == receipt['path_loss_sha256']
                with np.load(path_file, allow_pickle=False) as p:
                    assert np.array_equal(p['origins'], origins)
                    path_losses = {key: p[key].copy() for key in p.files if key != 'origins'}
            for row in metrics:
                name = row['method']; loss = losses[name]
                assert row['origins'] == len(origins)
                close(row['endpoint_mse'], np.mean(loss), 'endpoint MSE')
                close(row['baseline_mse'], np.mean(losses['zero']), 'baseline MSE')
                close(row['improvement_pct'], 100*(1-np.mean(loss)/np.mean(losses['zero'])), 'fold improvement')
                close(row['endpoint_mae'], np.mean(np.sqrt(loss)), 'endpoint MAE')
                if name in path_losses:
                    close(row['path_mse'], np.mean(path_losses[name]), 'complete-path MSE')
                    counts['path_metric_rows'] += 1
                counts['endpoint_metric_rows'] += 1
                counts['mae_metric_rows'] += 1
            counts['folds'] += 1
        concatenated = {name: np.concatenate(parts) for name, parts in all_losses.items()}
        assert set(concatenated) == set(desired['summary'])
        for name, losses in concatenated.items():
            row = desired['summary'][name]
            byfold = [float(f[name].mean()) for f in losses_by_fold]
            assert row['origins'] == len(losses)
            close(row['endpoint_mse'], losses.mean(), 'pooled endpoint MSE')
            close(row['improvement_pct'], 100*(1-losses.mean()/concatenated['zero'].mean()), 'pooled improvement')
            close(row['fold_mse'], byfold, 'per-fold MSE')
            assert row['positive_folds'] == sum(a < b['zero'].mean() for a, b in zip(byfold, losses_by_fold))
            counts['pooled_rows'] += 1
        expected = [row for rows in desired['comparisons'].values() for row in rows] + desired['diagnostic_contrasts']
        for block in evaluation['blocks'][task.split('_h')[0]]:
            selected = [row for row in expected if row['block'] == block]
            pairs = [(row['primary'], row['reference']) for row in selected]
            actual = paired_intervals(groups, pairs, block, task)
            compare_intervals(actual, selected)
            counts['paired_intervals'] += len(actual)
        checks = {'minimum_one_percent': desired['summary'][candidate]['improvement_pct'] >= 1.,
            'four_positive_folds': desired['summary'][candidate]['positive_folds'] >= 4,
            'all_control_intervals_both_blocks': all(len(rows) == 2 and all(row['supports_improvement'] for row in rows)
                for rows in desired['comparisons'].values())}
        assert checks == desired['checks']
        assert all(inner_passes) == gate['tasks'][task]['inner_gate_passed'] == desired['inner_direction_gate_passed']
        passed = all(checks.values())
        assert passed == desired['performance_gate_passed']
        assert (passed and all(inner_passes)) == desired['independent_confirmation_eligible']
        performance.append(passed); eligible.append(passed and all(inner_passes))
    assert counts['pooled_rows'] == 66 and counts['paired_intervals'] == 136
    assert counts['endpoint_metric_rows'] == counts['mae_metric_rows'] == 80
    assert counts['folds'] == 10 and outer['matrix_cells'] == 40
    assert stats['both_primary_performance_gates_passed'] == all(performance)
    assert stats['independent_confirmation_eligible'] == all(eligible)
    print(json.dumps({'status': 'passed', **counts,
        'complete_path_statistics_recomputed': counts['path_metric_rows'] == 80,
        'path_mse_limitation': None if counts['path_metric_rows'] == 80 else 'Per-origin path losses absent; original-path independent review is separate',
        'source_hashes_verified': True, 'saved_loss_intervals_recomputed': True,
        'new_model_fitting_or_target_construction': False, 'Q3_values_read': False,
        'independent_confirmation': False, 'scope': 'Exposed-development saved-loss arithmetic, not new market validation'}), flush=True)


if __name__ == '__main__':
    main()
