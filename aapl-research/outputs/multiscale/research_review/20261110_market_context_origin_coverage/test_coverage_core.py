"""Synthetic clock tests; no project market files are opened."""
from itertools import product
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from coverage_core import (delayed_history_available, hour_source_available,
                           original_hour_groups, partition_original_fit)


def main():
    started = time.time()
    checks = []
    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append(name)

    # A separate explicit list oracle catches both sides of the delayed window.
    cases = 0
    for n in range(1, 10):
        origins = np.arange(n, dtype=np.int64)
        for pattern in product((False, True), repeat=n):
            observed = np.asarray(pattern, dtype=bool)
            for length in range(1, n+2):
                expected = [i >= length and all(pattern[j] for j in range(i-length, i))
                            for i in origins]
                actual = delayed_history_available(observed, origins, length)
                if actual.tolist() != expected:
                    raise AssertionError(('enumerated_history', n, pattern, length))
                cases += 1
    check('exhaustive_clock_oracle', cases > 8000)
    observed = np.ones(12, dtype=bool)
    origins = np.arange(12, dtype=np.int64)
    baseline = delayed_history_available(observed, origins, 4)
    check('original_first_i_L_minus_1_missing', not baseline[3])
    check('first_delayed_i_L_available', baseline[4])
    changed = observed.copy(); changed[8:] = False
    check('current_and_future_excluded', delayed_history_available(changed, origins, 4)[8])
    check('previous_missing_included', not delayed_history_available(changed, origins, 4)[9])
    check('empty_origins', delayed_history_available(observed, origins[:0], 4).shape == (0,))
    permutation = np.array([8, 4, 3, 8], dtype=np.int64)
    check('original_origin_order_and_duplicates_preserved',
          delayed_history_available(observed, permutation, 4).tolist() == [True, True, False, True])

    # Irregular real-clock intervals must be used without reselecting the split.
    ends = np.array([1, 2, 3, 4, 10, 11, 12, 90, 91, 92, 93, 94, 95, 96], dtype=np.int64)
    fit = np.arange(10, dtype=np.int64)
    fit_before = fit.copy()
    cut, prefix, tail = partition_original_fit(fit, ends, 2)
    check('fixed_original_80_percent_boundary', cut == 8)
    check('tail_unchanged', np.array_equal(tail, [8, 9]))
    check('equality_label_clock_purged', np.array_equal(prefix, np.arange(6)))
    check('fit_not_mutated', np.array_equal(fit, fit_before))
    sparse = np.array([0, 2, 3, 5, 7, 9], dtype=np.int64)
    cut, prefix, tail = partition_original_fit(sparse, ends, 2)
    expected = [int(i) for i in sparse[:cut] if int(ends[i+2]) < int(ends[sparse[cut]])]
    check('nonconsecutive_original_fit_uses_actual_clock', prefix.tolist() == expected)
    check('sparse_original_tail_retained', tail.tolist() == sparse[cut:].tolist())

    minute = 60_000_000_000
    start = np.r_[np.arange(390)*minute,
                  (1440+np.arange(210))*minute].astype(np.int64)
    end = start+minute
    session = np.asarray(['2021-01-04']*390+['2021-01-05']*210)
    a, b = original_hour_groups(start, end, session)
    check('six_full_hours_normal_three_halfday', len(a) == 9)
    check('sessions_restart_hour_anchor', a.tolist() == [0, 60, 120, 180, 240, 300, 390, 450, 510])
    check('residuals_excluded', b.tolist() == [60, 120, 180, 240, 300, 360, 450, 510, 570])
    source = np.ones(len(start), dtype=bool)
    source[389] = False
    check('missing_residual_not_hour_missing', hour_source_available(source, a, b).all())
    source[450] = False
    hourly = hour_source_available(source, a, b)
    check('one_missing_constituent_invalidates_only_its_hour',
          np.flatnonzero(~hourly).tolist() == [7])
    delayed_hourly = delayed_history_available(hourly, np.arange(9, dtype=np.int64), 2)
    check('prior_session_history_retained', delayed_hourly[6])
    check('current_hour_source_not_used', delayed_hourly[7])
    check('previous_hour_source_missing_used', not delayed_hourly[8])

    bad_start = start.copy(); bad_start[10:390] += minute
    negative = [
        ('reject_within_session_gap', lambda: original_hour_groups(bad_start, bad_start+minute, session)),
        ('reject_subminute_duration', lambda: original_hour_groups(start, end+1, session)),
        ('reject_negative_origin', lambda: delayed_history_available(source, np.array([-1], dtype=np.int64), 2)),
        ('reject_outside_origin', lambda: delayed_history_available(source, np.array([600], dtype=np.int64), 2)),
        ('reject_float_origin', lambda: delayed_history_available(source, np.array([1.0]), 2)),
        ('reject_nonbool_source', lambda: delayed_history_available(source.astype(int), origins, 2)),
        ('reject_zero_lookback', lambda: delayed_history_available(source, origins, 0)),
        ('reject_unsorted_fit', lambda: partition_original_fit(np.array([2, 1], dtype=np.int64), ends, 2)),
        ('reject_no_label_clock', lambda: partition_original_fit(fit, ends, 5)),
        ('reject_non60_hour', lambda: hour_source_available(source, a, b-1)),
        ('reject_out_of_array_hour', lambda: hour_source_available(source, np.array([590], dtype=np.int64), np.array([650], dtype=np.int64))),
    ]
    for name, operation in negative:
        try:
            operation()
        except ValueError:
            checks.append(name)
        else:
            raise AssertionError(name)
    root = Path(__file__).resolve().parent
    report = {'kind': 'synthetic_clock_only_coverage_checks_v1',
              'passed': True, 'checks': checks, 'checks_count': len(checks),
              'exhaustive_oracle_cases': cases, 'market_files_opened': False,
              'AAPL_prices_labels_read': False, 'SPY_prices_read': False,
              'fit_models': 0, 'Q3_read': False, 'GPU': False,
              'core_sha256': hashlib.sha256((root/'coverage_core.py').read_bytes()).hexdigest(),
              'test_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'seconds': time.time()-started}
    (root/'SYNTHETIC_CLOCK_CHECKS.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'passed': True, 'checks_count': len(checks),
                      'exhaustive_oracle_cases': cases, 'seconds': report['seconds']}))


if __name__ == '__main__':
    main()
