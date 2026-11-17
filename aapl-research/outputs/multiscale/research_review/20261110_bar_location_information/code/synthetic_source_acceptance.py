"""Executable artificial source fixtures; no real source, targets or training.

Run only after independent source review and the parent protocol admits this
synthetic stage. Importing this file does not run fixtures or load market data.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import tempfile

import ohlc_source as source


def _reject(label, function, checks):
    try:
        function()
    except (ValueError, RuntimeError, FloatingPointError, FileNotFoundError):
        checks.append(label)
    else:
        raise AssertionError('Synthetic source fixture admitted ' + label)


def _grid(day, count):
    np = source._numpy()
    old = source.legacy_volume()
    first = old._timestamp(day + 'T09:30:00-05:00')[0]
    starts = first + old.MINUTE_NS * np.arange(count, dtype=np.int64)
    return {'start_ns': starts, 'end_ns': starts + old.MINUTE_NS,
            'close': np.full(count, 100.0, dtype=np.float64),
            'observed': np.ones(count, dtype=bool),
            'session_date': np.repeat(day, count)}


def _hours(minute):
    np = source._numpy()
    dates = minute['session_date']
    boundaries = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1]])
    ends = np.r_[boundaries[1:], len(dates)]
    first = np.asarray([i for a, b in zip(boundaries, ends)
                        for i in range(int(a), int(b) - 59, 60)], dtype=np.int64)
    last = first + 59
    return {'start_ns': minute['start_ns'][first], 'end_ns': minute['end_ns'][last],
            'close': minute['close'][last], 'observed': np.ones(len(first), dtype=bool),
            'session_date': dates[first]}


def _write_rows(path, rows):
    old = source.legacy_volume()
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=old.CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _row(timestamp, *, open_price=100.0, high=101.0, low=99.0, close=100.0, volume=2.0):
    # These two fields are retained CSV text, intentionally not CLV inputs.
    return {'timestamp': timestamp, 'open': str(open_price), 'high': str(high),
            'low': str(low), 'close': str(close), 'volume': str(volume),
            'trade_count': 'not_used_by_this_adapter', 'vwap': 'not_used_by_this_adapter'}


def run_synthetic_acceptance():
    np = source._numpy()
    old = source.legacy_volume()
    checks = []
    # Calls only the previous module's own freshly created synthetic fixtures;
    # verifies the unchanged volume slot/20-session behavior being reused.
    inherited = old.synthetic_acceptance()
    if inherited['status'] != 'passed' or inherited['market_data_read'] is not False:
        raise AssertionError('Frozen legacy synthetic volume fixtures failed')
    checks.append('unchanged_legacy_slot_baseline_synthetic_fixtures')

    op = np.asarray([100.0, 100.0, 100.0, 100.0, 100.0])
    hi = np.asarray([102.0, 101.0, 100.0, 101.0, 100.0])
    lo = np.asarray([99.0, 98.0, 100.0, 100.0, 99.0])
    cl = np.full(5, 100.0, dtype=np.float64)
    clv = source.close_location(op, hi, lo, cl)
    assert np.array_equal(clv, np.asarray([-1.0/3.0, 1.0/3.0, 0.0, -1.0, 1.0]))
    checks += ['same_close_same_range_opposite_location_witness', 'legal_flat_bar_clv_zero',
               'exact_close_at_high_low_clv_endpoints']
    for field in source.PRICE_FIELDS:
        for label, value in [('nan', float('nan')), ('infinity', float('inf')),
                             ('zero', 0.0), ('negative', -1.0)]:
            prices = dict(open=op.copy(), high=hi.copy(), low=lo.copy(), close=cl.copy())
            prices[field][0] = value
            _reject('reject_' + field + '_' + label,
                    lambda prices=prices: source.close_location(*(prices[k] for k in source.PRICE_FIELDS)), checks)
    for label, prices in {
            'high_below_low': ([100.0], [99.0], [101.0], [100.0]),
            'open_outside_range': ([102.0], [101.0], [99.0], [100.0]),
            'close_outside_range': ([100.0], [101.0], [99.0], [102.0]),
            'invalid_flat_open': ([99.0], [100.0], [100.0], [100.0]),
            'mismatched_shapes': ([100.0, 100.0], [101.0], [99.0], [100.0])}.items():
        _reject('reject_' + label, lambda prices=prices: source.close_location(*prices), checks)

    normal = _grid('2021-01-04', 390)
    half = _grid('2021-01-05', 210)
    minute = {key: np.concatenate((normal[key], half[key])) for key in normal}
    hour = _hours(minute)
    prices = {'open': minute['close'].copy(), 'high': minute['close'] + 1.0,
              'low': minute['close'] - 1.0, 'close': minute['close'].copy(),
              'volume': np.arange(len(minute['close']), dtype=np.float64)}
    prices['high'][0] = 104.0
    prices['clv'] = source.close_location(*(prices[k] for k in source.PRICE_FIELDS))
    aggregated, grouping = source.aggregate_hour_ohlc(minute, hour, prices)
    assert grouping['full_hours'] == 9 and grouping['sessions'] == 2
    assert grouping['residual_minutes_excluded'] == 60
    assert aggregated['open'][0] == 100.0 and aggregated['high'][0] == 104.0
    assert aggregated['low'][0] == 99.0 and aggregated['close'][0] == 100.0
    assert aggregated['clv'][0] == -0.6
    assert aggregated['clv'][0] != np.mean(prices['clv'][:60])
    assert aggregated['volume'][0] == 1770.0
    checks += ['complete60minute_ohlc_grouping', 'normal_and_halfday_residual_exclusion',
               'hour_clv_not_mean_of_minute_clv', 'hour_volume_uses_unchanged_exact_sum']
    for field, delta in [('start_ns', old.MINUTE_NS), ('end_ns', old.MINUTE_NS)]:
        wrong = {key: value.copy() for key, value in hour.items()}
        wrong[field][0] += delta
        _reject('reject_hour_' + field, lambda wrong=wrong: source.aggregate_hour_ohlc(minute, wrong, prices), checks)
    wrong = {key: value.copy() for key, value in hour.items()}
    wrong['close'][0] = np.nextafter(100.0, 101.0)
    _reject('reject_hour_nonexact_close', lambda: source.aggregate_hour_ohlc(minute, wrong, prices), checks)
    wrong_day = {key: value.copy() for key, value in hour.items()}
    wrong_day['session_date'][0] = '2021-01-05'
    _reject('reject_hour_session', lambda: source.aggregate_hour_ohlc(minute, wrong_day, prices), checks)
    gap = {key: value.copy() for key, value in minute.items()}
    gap['start_ns'][1] += old.MINUTE_NS
    _reject('reject_nonconsecutive_minute_grid', lambda: source.aggregate_hour_ohlc(gap, hour, prices), checks)

    small = {key: value[:3].copy() for key, value in normal.items()}
    rows = [_row('2021-01-04T09:29:00-05:00', volume=0.0),
            _row('2021-01-04T09:30:00-05:00', high=102.0, low=99.0),
            _row('2021-01-04T09:31:00-05:00', high=101.0, low=98.0),
            _row('2021-01-04T09:32:00-05:00', high=100.0, low=100.0)]
    expected_counts = {'all': 4, 'regular': 3, 'outside_regular': 1}
    with tempfile.TemporaryDirectory(prefix='_synthetic_source_', dir=source.STAGE_DIR) as temporary:
        root = Path(temporary)
        path = root / 'synthetic_source.csv'
        _write_rows(path, rows)
        sha = source.file_sha256(path)
        values, metadata = source.align_csv_ohlc(path, small, expected_sha256=sha,
                                                expected_counts=expected_counts)
        assert np.array_equal(values['clv'], clv[:3])
        assert metadata['counts'] == expected_counts
        assert metadata['first_source_start_ns'] == old._timestamp(rows[0]['timestamp'])[0]
        assert small['start_ns'][0] == old._timestamp(rows[1]['timestamp'])[0]
        checks += ['streaming_left_boundary_timestamp_and_outside_count', 'exact_source_close_and_flat_bar']
        _reject('reject_wrong_csv_hash', lambda: source.align_csv_ohlc(
            path, small, expected_sha256='0' * 64, expected_counts=expected_counts), checks)

        variants = {
            'missing_minute': rows[:2] + rows[3:],
            'duplicate_timestamp': rows[:2] + [dict(rows[1])] + rows[2:],
            'right_boundary_timestamp_shift': [dict(rows[1], timestamp='2021-01-04T09:31:00-05:00')],
            'timezone_missing': [dict(rows[0], timestamp='2021-01-04T09:29:00')] + rows[1:],
            'subminute_timestamp': [dict(rows[0], timestamp='2021-01-04T09:29:01-05:00')] + rows[1:],
            'post_cutoff_date': [dict(rows[0], timestamp='2026-07-01T09:29:00-04:00')] + rows[1:],
            'nonexact_close': [dict(rows[0]), dict(rows[1], close=str(np.nextafter(100.0, 101.0)))] + rows[2:],
            'bad_outside_ohlc': [dict(rows[0], high='NaN')] + rows[1:],
            'flat_invalid_close': rows[:3] + [dict(rows[3], close='99.0')],
            'negative_volume': [dict(rows[0], volume='-1.0')] + rows[1:],
            'malformed_price': [dict(rows[0], open='not_a_number')] + rows[1:],
        }
        for field in source.PRICE_FIELDS:
            for label, value in [('nan', 'NaN'), ('nonpositive', '0.0')]:
                changed = copy.deepcopy(rows)
                changed[1][field] = value
                variants[field + '_' + label] = changed
        for label, changed_rows in variants.items():
            _write_rows(path, changed_rows)
            digest = source.file_sha256(path)
            _reject('stream_reject_' + label, lambda digest=digest: source.align_csv_ohlc(
                path, small, expected_sha256=digest, expected_counts=expected_counts), checks)

        result, cache_grouping = source.assemble_sources(minute, hour, prices)
        identity = source.adapter_identity({'synthetic_only': True, 'fixture': 'two_sessions_390_210'})
        cache = root / 'cache'
        source.write_cache(cache, result, identity, {'synthetic_only': True}, cache_grouping)
        restored = source.read_cache(cache, identity)
        for frequency in ('minute', 'hour'):
            for field, original in result[frequency].items():
                if original.dtype.kind == 'f':
                    assert np.array_equal(original, restored[frequency][field], equal_nan=True)
                else:
                    assert np.array_equal(original, restored[frequency][field])
        checks.append('cache_roundtrip_exact_values_dtypes_and_nan_validity')
        bad_identity = copy.deepcopy(identity)
        bad_identity['source']['fixture'] = 'different_source'
        _reject('reject_wrong_cache_source_identity', lambda: source.read_cache(cache, bad_identity), checks)
        bad_code = copy.deepcopy(identity)
        bad_code['adapter_sha256'] = '0' * 64
        _reject('reject_wrong_cache_code_identity', lambda: source.read_cache(cache, bad_code), checks)
        _reject('reject_overwrite_completed_cache', lambda: source.write_cache(
            cache, result, identity, {'synthetic_only': True}, cache_grouping), checks)
        partial = root / 'partial'
        partial.mkdir()
        (partial / 'minute_ohlc.npz').write_bytes((cache / 'minute_ohlc.npz').read_bytes())
        _reject('reject_partial_cache_without_commit_receipt', lambda: source.read_cache(partial, identity), checks)
        with (cache / 'minute_ohlc.npz').open('ab') as handle:
            handle.write(b'corrupt')
        _reject('reject_changed_cache_bytes', lambda: source.read_cache(cache, identity), checks)
        receipt_path = cache / 'OHLC_SOURCE_RECEIPT.json'
        receipt = json.loads(receipt_path.read_text())
        receipt['status'] = 'incomplete'
        receipt_path.write_text(json.dumps(receipt))
        _reject('reject_incomplete_cache_receipt', lambda: source.read_cache(cache, identity), checks)

    _reject('source_call_requires_guard_before_any_real_access', lambda: source.load_ohlc(
        '/never_read_by_this_fixture', None), checks)
    return {'status': 'passed', 'synthetic_only': True,
            'real_source_or_market_arrays_read': False, 'targets_models_weights_or_Q3_read': False,
            'training_or_forecast_scoring': False, 'checks': checks,
            'legacy_synthetic_checks': inherited['checks'],
            'adapter_identity': source.adapter_identity({'synthetic_only': True})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic-only', action='store_true', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if hasattr(os, 'sched_getaffinity') and set(os.sched_getaffinity(0)) != {26}:
        raise RuntimeError('Synthetic source acceptance must be launched on CPU26 only')
    # Environment must be set by the invoking frozen command, before NumPy loads.
    report = run_synthetic_acceptance()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source.legacy_volume()._atomic_json(args.output, report)


if __name__ == '__main__':
    main()
