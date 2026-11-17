"""Recompute saved scores and fixed continuation rules without model helpers."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    stage = Path(__file__).resolve().parents[1]
    args.results.resolve().relative_to(stage)
    args.output.resolve().relative_to(stage)
    assert time.time() < args.deadline
    import numpy as np

    results = json.loads(args.results.read_text())
    state_ids = [row['state'] for row in results['results']]
    expected_states = ({'minute_f1', 'minute_f2', 'hourly_f1', 'hourly_f2'}
                       if results.get('market_read') is True else {'synthetic_native'})
    assert set(state_ids) == expected_states and len(state_ids) == len(expected_states)
    predictors = ['D', 'Z', 'C', 'zero', 'prefix_update_mean', 'same_information_Ridge']
    checks, recalculated, inputs = [], [], []
    raw_fixture_label_checks = 0
    for row in results['results']:
        directory = args.results.parent
        if not (directory / 'PREDICTIONS.npz').exists():
            directory = directory / row['state']
        file = directory / 'PREDICTIONS.npz'
        file.resolve().relative_to(stage)
        with np.load(file, allow_pickle=False) as archive:
            assert set(archive.files) == set(predictors + ['labels', 'origins'])
            arrays = {key: archive[key] for key in archive.files}
        y = arrays['labels']
        assert y.shape == (row['count'], row['horizon']) and np.isfinite(y).all()
        origins = arrays['origins']
        assert origins.dtype == np.dtype('int64') and origins.shape == (len(y),)
        assert np.all(np.diff(origins) > 0) and np.count_nonzero(arrays['zero']) == 0
        fixture = directory / 'NATIVE_FIXTURE.npz'
        if fixture.exists():
            fixture.resolve().relative_to(stage)
            with np.load(fixture, allow_pickle=False) as raw:
                close, tail = raw['close'], raw['tail']
            assert np.array_equal(origins, tail)
            direct_labels = np.log(close[origins[:, None] + np.arange(1, row['horizon'] + 1)] /
                                   close[origins, None])
            assert np.array_equal(y, direct_labels)
            raw_fixture_label_checks += 1
        assert np.array_equal(arrays['prefix_update_mean'],
                              np.broadcast_to(arrays['prefix_update_mean'][0], y.shape))
        rebuilt = {'state': row['state'], 'predictors': {}}
        for key in predictors:
            pred = arrays[key]
            assert pred.shape == y.shape and np.isfinite(pred).all()
            error = pred - y
            values = {'endpoint_MSE': float(np.mean(error[:, -1]**2, dtype=np.float64)),
                      'path_MSE': float(np.mean(error**2, dtype=np.float64)),
                      'bias': float(np.mean(error, dtype=np.float64)),
                      'prediction_variance': float(np.var(pred, dtype=np.float64)),
                      'endpoint_sse': float(np.sum(error[:, -1]**2, dtype=np.float64)),
                      'path_sse': float(np.sum(error**2, dtype=np.float64)),
                      'sum_error': float(np.sum(error, dtype=np.float64)),
                      'sum_pred': float(np.sum(pred, dtype=np.float64)),
                      'sum_pred_square': float(np.sum(pred**2, dtype=np.float64))}
            for metric, actual in values.items():
                expected = row['predictors'][key][metric]
                allowance = 1e-18 + 1e-12 * max(abs(actual), abs(expected))
                passed = abs(actual - expected) <= allowance
                checks.append({'state': row['state'], 'predictor': key, 'metric': metric,
                               'passed': passed, 'max_abs': abs(actual - expected),
                               'allowance': allowance})
                assert passed, (row['state'], key, metric, actual, expected)
            rebuilt['predictors'][key] = values
        recalculated.append(rebuilt)
        inputs.append({'path': str(file.resolve()),
                       'sha256': hashlib.sha256(file.read_bytes()).hexdigest()})
    independent_states = []
    for row in recalculated:
        tests = []
        for control in ['D', 'Z', 'zero', 'prefix_update_mean', 'same_information_Ridge']:
            for metric in ['endpoint_MSE', 'path_MSE']:
                c, ref = row['predictors']['C'][metric], row['predictors'][control][metric]
                tests.append(bool(np.isfinite(ref) and ref > 0 and c <= 0.999 * ref))
        independent_states.append({'state': row['state'], 'passed': all(tests)})
    tasks = []
    for name in ['minute', 'hourly']:
        rows = [row for row in recalculated if row['state'].startswith(name + '_')]
        if rows:
            assert len(rows) == 2
            improvement = sum(1 - row['predictors']['C']['endpoint_MSE'] /
                              row['predictors']['zero']['endpoint_MSE'] for row in rows) / 2
            tasks.append({'task': name, 'equal_fold_improvement': improvement,
                          'passed': improvement >= 0.01})
    continuation = (len(recalculated) == 4 and len(tasks) == 2 and
                    all(row['passed'] for row in independent_states + tasks))
    if results.get('market_read') is True:
        decision = results['decision']
        assert continuation == decision['continue_to_formal_development_review']
        for row in independent_states:
            original = next(s for s in decision['per_state'] if s['state'] == row['state'])
            assert row['passed'] == original['passed']
        assert {row['task'] for row in decision['per_task']} == {'minute', 'hourly'}
        assert len(decision['per_task']) == 2
        for row in tasks:
            original = next(s for s in decision['per_task'] if s['task'] == row['task'])
            assert row['passed'] == original['passed']
            assert abs(row['equal_fold_improvement'] -
                       original['equal_fold_zero_endpoint_improvement']) <= 1e-12
    assert time.time() < args.deadline
    report = {'status': 'passed_independent_saved_prediction_score_reconstruction',
              'checks_passed': len(checks), 'checks': checks, 'inputs': inputs,
              'recomputed': recalculated, 'independent_state_rules': independent_states,
              'independent_task_rules': tasks,
              'continue_to_formal_development_review': continuation,
              'numeric_comparison_atol': 1e-18, 'numeric_comparison_rtol': 1e-12,
              'derived_relative_gain_comparison_atol': 1e-12,
              'raw_synthetic_fixture_label_checks': raw_fixture_label_checks,
              'worker_or_model_helpers_imported': False, 'Torch_imported': False,
              'only_this_stage_own_prediction_arrays_read': True,
              'Q3_read': False, 'forecast_success': False}
    temporary = args.output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(args.output)
    print(json.dumps({'status': report['status'], 'checks_passed': len(checks)}))


if __name__ == '__main__':
    main()
