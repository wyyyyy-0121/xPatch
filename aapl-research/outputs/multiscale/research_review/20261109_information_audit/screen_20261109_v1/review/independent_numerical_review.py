"""Independent fit-only review: no production feature additions or solver imports.

The previously audited 17 context columns are reused. New volume means use
explicit sliding windows; the interaction uses physical log-close differences.
All Ridge systems use materialized, directly centered training matrices.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
import signal
import time
import traceback
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = OUT.parents[4]
PRIOR = ROOT / 'outputs/multiscale/controlled_followup/20261108_v2'
CONTEXT = ROOT / 'outputs/multiscale/mechanism_repair/20261109_v3b/code/predictor.py'
EXPECTED_RULES = '3276502a5c13f659e6e4fad8b42d40a063f131a1a6951a6eb8cdc7535dd56ad9'
FIELDS = ('start_ns', 'end_ns', 'close', 'observed', 'session_date',
          'volume', 'abnormal', 'valid', 'slot')
START = time.monotonic()
REPORT = {'status': 'running', 'rules_sha256': EXPECTED_RULES,
          'maximum_runtime_seconds': 300, 'GPU': False,
          'CPU_affinity': sorted(os.sched_getaffinity(0)),
          'independent_solver': 'direct centered full-matrix standardized Ridge',
          'independent_new_features': 'sliding-window mean and physical return interaction',
          'stop_or_outer_targets_or_predictions_read': False,
          'Q3_values_read': False, 'scientific_gate_tolerance': 0,
          'numerical_check_tolerance': {'rtol': 2e-8, 'atol': 2e-11},
          'states': [], 'normalization': {}}


def write_report():
    REPORT['elapsed_seconds'] = time.monotonic() - START
    REPORT['review_script_sha256'] = sha(Path(__file__))
    temp = HERE / '.INDEPENDENT_NUMERICAL_REVIEW.json.tmp'
    temp.write_text(json.dumps(REPORT, indent=2, allow_nan=False) + '\n')
    os.replace(temp, HERE / 'INDEPENDENT_NUMERICAL_REVIEW.json')


def alarm(*_):
    raise TimeoutError('Independent review reached its 300-second bound')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def ihash(origins):
    return hashlib.sha256(origins.astype('<i8').tobytes()).hexdigest()


def equal_array(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-11)
    return float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))


def normalization_review(data, profile, session_order):
    """Sorted Python values from exactly the preceding 20 GLOBAL sessions."""
    dates = data['session_date'].astype(str)
    slots = data['slot'].astype(int)
    session_number = {date: s for s, date in enumerate(session_order)}
    row_session = np.array([session_number[d] for d in dates], dtype=np.int64)
    width = 390 if profile == 'minute' else 6
    grid = np.full((len(session_order), width), np.nan)
    grid[row_session, slots] = data['volume']
    assert len(set(zip(row_session.tolist(), slots.tolist()))) == len(dates)
    counts = np.bincount(row_session, minlength=len(session_order))
    half = np.flatnonzero((counts > 0) & (counts < width))
    sessions = {0, 1, 19, 20, 21, len(session_order)-1}
    for s in half:
        sessions.update(int(s)+j for j in (0, 1, 19, 20, 21))
    sessions = {s for s in sessions if 0 <= s < len(session_order)}
    relevant_slots = (0, 179, 180, 389) if profile == 'minute' else (0, 2, 3, 5)
    indices = np.flatnonzero(np.isin(row_session, list(sessions)) & np.isin(slots, relevant_slots))
    maxdiff = 0.
    absent_count = 0
    finite_count = 0
    samples = []
    for i in indices:
        s, k = int(row_session[i]), int(slots[i])
        history = grid[s-20:s, k] if s >= 20 else np.array([])
        valid = bool(len(history) == 20 and np.isfinite(history).all() and np.isfinite(grid[s, k]))
        actual = float(data['abnormal'][i])
        assert bool(np.isfinite(actual)) == valid, (profile, s, k, 'availability')
        if valid:
            logs = sorted(math.log1p(float(v)) for v in history)
            expected = math.log1p(float(grid[s, k])) - (logs[9]+logs[10])/2
            maxdiff = max(maxdiff, abs(actual-expected))
            assert abs(actual-expected) <= 1e-12, (profile, s, k, actual, expected)
            finite_count += 1
        else:
            absent_count += 1
        if len(samples) < 24 or s in (20, 21):
            samples.append({'session_number': s, 'date': str(session_order[s]),
                            'slot': k, 'valid': valid})
    assert np.array_equal(data['valid'].astype(bool), np.isfinite(data['abnormal']))
    return {'status': 'passed', 'sampled_present_rows': len(indices),
            'finite_baselines_checked': finite_count, 'missing_baselines_checked': absent_count,
            'half_day_sessions': [str(session_order[s]) for s in half],
            'exact_previous_20_global_sessions': True, 'current_session_excluded': True,
            'sorted_scalar_log1p_median_max_absolute_difference': maxdiff,
            'cache_valid_mask_equal_finite_abnormal': True, 'samples': samples}


def matrices(data, origins, horizon, lookback, context, rolling):
    X = np.empty((len(origins), 19), dtype=np.float64)
    Y = np.empty((len(origins), horizon), dtype=np.float64)
    log_close = np.log(data['close'])
    lag_col = context.feature_names(horizon, lookback).index(f'return_lag_{horizon}')
    feature_return_diff = 0.
    for start in range(0, len(origins), 2048):
        chunk = origins[start:start+2048]
        hist_indices = chunk[:, None] + np.arange(1-lookback, 1)
        assert data['observed'][hist_indices].all()
        history = log_close[hist_indices] - log_close[chunk, None]
        old = context.features(history, data['end_ns'][chunk], horizon)
        volume_mean = rolling[chunk-horizon+1]
        physical_return = log_close[chunk] - log_close[chunk-horizon]
        feature_return_diff = max(feature_return_diff, float(np.max(abs(physical_return-old[:, lag_col]))))
        X[start:start+len(chunk)] = np.column_stack((old, volume_mean, physical_return*volume_mean))
        future = chunk[:, None] + np.arange(1, horizon+1)
        assert data['observed'][future].all()
        if horizon == 60:
            assert np.all(data['session_date'][chunk] == data['session_date'][chunk+horizon])
        Y[start:start+len(chunk)] = np.log(data['close'][future]/data['close'][chunk, None])
    assert np.isfinite(X).all() and np.isfinite(Y).all()
    assert feature_return_diff <= 1e-12
    return X, Y, feature_return_diff


def metrics(prediction, target):
    error = prediction-target
    return {'endpoint_sse': float(np.sum(error[:, -1]**2)),
            'path_sse': float(np.sum(error**2)),
            'endpoint_mse': float(np.mean(error[:, -1]**2)),
            'path_mse': float(np.mean(error**2))}


def gates(rows, zero):
    return {'C_endpoint_below_A': rows['C']['endpoint_mse'] < rows['A']['endpoint_mse'],
            'C_endpoint_below_B': rows['C']['endpoint_mse'] < rows['B']['endpoint_mse'],
            'C_path_not_above_A': rows['C']['path_mse'] <= rows['A']['path_mse'],
            'C_path_not_above_B': rows['C']['path_mse'] <= rows['B']['path_mse'],
            'C_endpoint_not_above_zero': rows['C']['endpoint_mse'] <= zero['endpoint_mse']}


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    assert set(os.sched_getaffinity(0)).issubset({24, 26})
    assert sha(OUT/'RULES.json') == EXPECTED_RULES
    rules = load_json(OUT/'RULES.json')
    # Only metadata hashes are read; no target or prediction paths are followed.
    for name, expected in rules['sources'].items():
        assert sha(name) == expected, ('frozen_source_hash', name)
    assert sha(CONTEXT) == 'cf574d0087e74d4004207166c95283bc2a627877a07933cd307a71206652d4e6'
    spec = importlib.util.spec_from_file_location('audited_original_context', CONTEXT)
    context = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(context)
    receipt = load_json(OUT/'cache/VOLUME_SOURCE_RECEIPT.json')
    source = {}
    for profile in ('minute', 'hour'):
        path = OUT/'cache'/(profile+'_volume.npz')
        assert sha(path) == receipt['cache_sha256'][profile]
        with np.load(path, allow_pickle=False) as f:
            source[profile] = {k: f[k].copy() for k in FIELDS}
        assert np.all(source[profile]['session_date'].astype(str) < '2026-07-01')
    sessions = np.unique(source['minute']['session_date'].astype(str))
    for profile in source:
        REPORT['normalization'][profile] = normalization_review(source[profile], profile, sessions)
    coefficients = {'minute': [], 'hourly': []}
    for profile, cache_profile, h, lookback in (('minute', 'minute', 60, 480), ('hourly', 'hour', 8, 96)):
        data = source[cache_profile]
        # Explicit sliding-window oracle differs from production cumulative sums.
        abnormal_windows = np.lib.stride_tricks.sliding_window_view(data['abnormal'], h)
        rolling = abnormal_windows.mean(axis=1)
        available = np.isfinite(abnormal_windows).all(axis=1)
        task = f'{profile}_h{h}_future_log_return'
        task_path = PRIOR/'cache'/task/'TASK.json'
        meta = load_json(task_path)
        for fold in (1, 2):
            folder = OUT/task/f'fold_{fold}'
            part = load_json(folder/'PARTITION.json')
            expected = load_json(folder/'RESULT.json')
            fm = meta['folds'][str(fold)]
            with np.load(fm['split_file'], allow_pickle=False) as f:
                fit = f['fit'].copy()  # Only this ZIP member is decompressed.
            assert fit.dtype == np.dtype('int64') and fit.ndim == 1
            assert (np.diff(fit) > 0).all() and len(fit) == fm['counts']['fit']
            assert ihash(fit) == fm['split_sha256']['fit']
            boundary = int(len(fit)*0.8)
            first = int(fit[boundary])
            train = fit[:boundary]
            train = train[data['end_ns'][train+h] < data['end_ns'][first]]
            diag = fit[boundary:]
            train = train[available[train-h+1]]
            diag = diag[available[diag-h+1]]
            rebuilt = {'original_fit_count': len(fit), 'original_80percent_index': boundary,
                       'diagnostic_first_origin_before_common_mask': first,
                       'train_count': len(train), 'diagnostic_count': len(diag),
                       'train_sha256': ihash(train), 'diagnostic_sha256': ihash(diag),
                       'train_last_label_end_ns': int(data['end_ns'][train[-1]+h]),
                       'diagnostic_first_decision_ns': int(data['end_ns'][diag[0]])}
            for key, value in rebuilt.items():
                assert part[key] == value, (task, fold, 'partition', key)
            assert part['all_arms_share_origins'] is True
            assert np.all(data['end_ns'][train+h] < data['end_ns'][first])
            X, Y, return_diff = matrices(data, train, h, lookback, context, rolling)
            D, T, diag_return_diff = matrices(data, diag, h, lookback, context, rolling)
            row = {'task': task, 'fold': fold, 'partition': rebuilt,
                   'strict_purge_and_common_mask_verified': True,
                   'physical_return_vs_audited_context_maxdiff': max(return_diff, diag_return_diff),
                   'cells': {}}
            computed = {}
            for name, k in (('A', 17), ('B', 18), ('C', 19)):
                train_x = X[:, :k]
                mean = train_x.mean(axis=0)
                std = train_x.std(axis=0)
                std = np.where(std > 1e-12, std, 1.)
                y_mean = Y.mean(axis=0)
                Z = (train_x-mean)/std
                coef = np.linalg.solve(Z.T@Z/len(Z)+0.1*np.eye(k), Z.T@(Y-y_mean)/len(Z))
                pred = y_mean + ((D[:, :k]-mean)/std)@coef
                with np.load(folder/(name+'.npz'), allow_pickle=False) as f:
                    saved = {key: f[key].copy() for key in ('coef', 'x_mean', 'x_std', 'y_mean')}
                    assert str(f['rules_sha256']) == EXPECTED_RULES
                differences = {key: equal_array(saved[key], value)
                               for key, value in (('coef', coef), ('x_mean', mean), ('x_std', std), ('y_mean', y_mean))}
                saved_prediction = saved['y_mean'] + ((D[:, :k]-saved['x_mean'])/saved['x_std'])@saved['coef']
                differences['all_diagnostic_predictions'] = equal_array(saved_prediction, pred)
                computed[name] = metrics(pred, T)
                differences['metric_max_absolute_difference'] = max(equal_array(computed[name][key], expected['metrics'][name][key]) for key in computed[name])
                row['cells'][name] = {'status': 'passed', 'max_absolute_differences': differences,
                                      'independent_metrics': computed[name]}
                if name == 'C':
                    interaction = float(coef[-1, -1]/std[-1])
            zero = metrics(np.zeros_like(T), T)
            for key in zero:
                equal_array(zero[key], expected['zero'][key])
            local_checks = gates(computed, zero)
            assert local_checks == expected['checks'], (task, fold, 'scientific_gates')
            assert all(local_checks.values()) == expected['passed_all_local_checks']
            interaction_diff = equal_array(interaction, expected['interaction_original_unit_endpoint_coefficient'])
            coefficients[profile].append(interaction)
            row.update({'independent_zero': zero, 'checks': local_checks,
                        'interaction_original_unit_endpoint_coefficient': interaction,
                        'interaction_coefficient_maxdiff': interaction_diff,
                        'passed_all_local_checks': all(local_checks.values()), 'status': 'passed'})
            REPORT['states'].append(row)
            write_report()
            print(json.dumps({'task': task, 'fold': fold, 'review': 'passed'}), flush=True)
            del X, Y, D, T, Z
    signs = {key: bool(value[0]*value[1] > 0) for key, value in coefficients.items()}
    gate = all(s['passed_all_local_checks'] for s in REPORT['states']) and all(signs.values())
    global_saved = load_json(OUT/'RESULTS.json')
    assert signs == global_saved['same_task_two_fold_interaction_sign']
    assert gate == global_saved['continuation_gate_passed']
    assert gate == global_saved['new_candidate_development_admitted_by_scientific_gate']
    REPORT.update({'status': 'passed', 'fits_independently_rebuilt': 12,
                   'same_task_two_fold_interaction_sign': signs,
                   'continuation_gate_passed': gate,
                   'scientific_continuation_gate_reproduced_exactly': True,
                   'source_and_rules_unchanged': True,
                   'predictive_success_claimed': False,
                   'research_goal_achieved': False})
    assert sha(OUT/'RULES.json') == EXPECTED_RULES


signal.signal(signal.SIGALRM, alarm)
signal.alarm(300)
try:
    main()
except BaseException as exc:
    REPORT.update({'status': 'failed', 'priority': 'P1', 'error': repr(exc),
                   'traceback': traceback.format_exc()})
    write_report()
    raise
else:
    signal.alarm(0)
    write_report()
    print(json.dumps({'status': REPORT['status'], 'continuation_gate_passed': REPORT['continuation_gate_passed'],
                      'elapsed_seconds': REPORT['elapsed_seconds']}), flush=True)
