"""Small generated observations test independent formulas, not market outcomes."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

import independent_result_review as review


def main():
    started = time.time(); checks = []
    def check(name, value):
        if not bool(value):
            raise AssertionError(name)
        checks.append(name)
    rng = np.random.default_rng(2711); n = 600
    close = np.exp(5+np.cumsum(rng.normal(0, 0.0005, n)))
    spy = np.exp(6+np.cumsum(rng.normal(0, 0.0003, n))); spy[220] = np.nan
    abnormal = rng.normal(size=n); abnormal[145] = np.nan
    start = pd.date_range('2021-01-04T14:30:00Z', periods=n, freq='min').asi8
    grid = {'start_ns': start, 'end_ns': start+60_000_000_000, 'close': close,
            'observed': np.ones(n, dtype=bool), 'session_date': np.full(n, '2021-01-04')}
    source = review.FeatureSource(grid, abnormal, spy, 96, 8, [])
    check('current_SPY_missing_excluded', source.mS[220])
    check('previous_SPY_missing_included', not source.mS[221])
    check('negative_SPY_prefix_missing', not source.mS[95] and source.mS[96])
    check('volume_all_H_required', not source.mV[152] and source.mV[153])
    o = np.array([180, 200], dtype=np.int64)
    x = source.x(o, 'A')
    expected_history = np.array([[np.log(close[j])-np.log(close[i]) for j in range(i-95, i+1)] for i in o])
    np.testing.assert_array_equal(x[:, :96], expected_history)
    check('original_nominal_anchor_and_shape', x.shape == (2, 114) and np.array_equal(x[:, 96], close[o]))
    b = source.x(o, 'B')
    expected_u = np.array([np.mean(abnormal[i-7:i+1]) for i in o])
    np.testing.assert_array_equal(b[:, -2], expected_u)
    np.testing.assert_array_equal(b[:, -1], x[:, 96+1+3]*expected_u)
    check('old_u_and_rH_u_order', b.shape == (2, 116))
    truth = np.array([[np.log(close[i+h]/close[i]) for h in range(1, 9)] for i in o])
    np.testing.assert_array_equal(source.y(o), truth)
    checks.append('raw_path_target_exact')
    p = np.arange(103, 350, dtype=np.int64); q = p[source.mV[p]]
    mean = source.raw_spy(q[source.mS[q]]).mean(axis=0)
    plus = source.x(q, 'Plus', mean)
    np.testing.assert_array_equal(plus[~source.mS[q], 116:], np.tile(mean, ((~source.mS[q]).sum(), 1)))
    checks.append('missing_SPY_neutral_raw_mean')
    model = review.direct_centered_fit(source, q, 'Plus', mean, lambda: None)
    y = source.y(q); mx, my = plus.mean(axis=0), y.mean(axis=0)
    sd = plus.std(axis=0); sd = np.where(sd > 1e-12, sd, 1.)
    z = (plus-mx)/sd
    expected = np.linalg.solve(z.T@z/len(q)+0.1*np.eye(211), z.T@(y-my)/len(q))
    np.testing.assert_allclose(model['coef'], expected, rtol=1e-9, atol=1e-11)
    check('direct_centered_objective_lambda_point1', model['coef'].shape == (211, 8))
    error = review.score_sums({'plus': np.zeros_like(y)}, y)
    check('endpoint_path_denominators', error['plus']['endpoint_mse'] == np.mean(y[:, -1]**2)
          and error['plus']['path_mse'] == np.mean(y*y))
    rows = []
    for name in ('minute_h60_future_log_return', 'hourly_h8_future_log_return'):
        for fold in (1, 2):
            metrics = {k: {'endpoint_mse': value, 'path_mse': value}
                       for k, value in [('A_full', 1.1), ('base', 1.0), ('zero', 1.2),
                                        ('prefix_mean', 1.15), ('plus', 0.9)]}
            rows.append({'task': name, 'fold': fold, 'metrics': metrics})
    result = review.science_decisions(rows)
    check('four_state_gate_and_equalfold_ratio', result['overall_pass'] and len(result['states']) == 4
          and all(np.isclose(x['gain_overzero'], 0.25) for x in result['primary'].values()))
    rows[0]['metrics']['plus']['endpoint_mse'] = 1.0
    check('strict_equal_endpoint_fails', not review.science_decisions(rows)['overall_pass'])
    rows[0]['metrics']['plus']['endpoint_mse'] = float('nan')
    try:
        review.science_decisions(rows)
    except ValueError:
        checks.append('NaN_metric_rejected')
    else:
        raise AssertionError('NaN_metric_rejected')
    root = Path(__file__).resolve().parent
    report = {'passed': True, 'checks_count': len(checks), 'checks': checks,
              'reviewer_sha256': review.sha(root/'independent_result_review.py'),
              'test_sha256': review.sha(Path(__file__)), 'seconds': time.time()-started,
              'actual_market_data_read': False, 'HTTP': False, 'Q3_read': False,
              'primary_worker_or_core_imported': False}
    (root/'INDEPENDENT_FORMULA_CHECKS.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'passed': True, 'checks_count': len(checks), 'seconds': report['seconds']}))


if __name__ == '__main__':
    main()
