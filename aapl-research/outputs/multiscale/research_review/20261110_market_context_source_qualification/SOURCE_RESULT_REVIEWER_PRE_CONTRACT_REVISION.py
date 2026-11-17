"""Independent stdlib-only fixed-SPY-session source/clock reviewer.

No network, credentials, project imports, AAPL arrays, returns, or model work.
Actual response decoding requires an explicit CLI flag and the original deadline.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

UTC = timezone.utc
OPEN = datetime(2021, 1, 4, 14, 30, tzinfo=UTC)
CLOSE = datetime(2021, 1, 4, 21, 0, tzinfo=UTC)
MINUTE = timedelta(minutes=1)
STARTS = tuple(OPEN + i * MINUTE for i in range(390))
ENDS = tuple(t + MINUTE for t in STARTS)
HOUR_STARTS = tuple(OPEN + timedelta(hours=i) for i in range(6))
HOUR_ENDS = tuple(t + timedelta(hours=1) for t in HOUR_STARTS)
REQUIRED = frozenset(('t', 'o', 'h', 'l', 'c', 'v', 'n', 'vw'))
MAX_BYTES = 1024 * 1024
STAMP = re.compile(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.(\d{1,9}))?(?:Z|[+-]\d{2}:\d{2})\Z')


class Rejected(ValueError):
    pass


def iso(value):
    return value.isoformat().replace('+00:00', 'Z') if value is not None else None


def stamp(value):
    if not isinstance(value, str):
        raise Rejected('timestamp_not_string')
    match = STAMP.fullmatch(value)
    if match is None:
        raise Rejected('timestamp_not_aware_RFC3339')
    if any(c != '0' for c in (match.group(1) or '')):
        raise Rejected('off_grid_fractional_timestamp')
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone(UTC)
    except ValueError as exc:
        raise Rejected('invalid_timestamp') from exc
    if parsed not in STARTS:
        raise Rejected('off_grid_timestamp')
    return parsed


def number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise Rejected('invalid_numeric_type_' + field)
    if isinstance(value, float) and not math.isfinite(value):
        raise Rejected('nonfinite_' + field)
    result = Decimal(str(value))
    if not result.is_finite():
        raise Rejected('nonfinite_' + field)
    return result


def check_fields(row):
    if not isinstance(row, dict) or set(row) != REQUIRED:
        raise Rejected('bar_field_set_mismatch')
    prices = {k: number(row[k], k) for k in ('o', 'h', 'l', 'c', 'vw')}
    if any(x <= 0 for x in prices.values()):
        raise Rejected('nonpositive_price_field')
    if prices['h'] < max(prices['o'], prices['l'], prices['c']):
        raise Rejected('high_inconsistent')
    if prices['l'] > min(prices['o'], prices['h'], prices['c']):
        raise Rejected('low_inconsistent')
    volume = number(row['v'], 'v')
    count = number(row['n'], 'n')
    if volume < 0 or count < 0:
        raise Rejected('negative_count_field')
    if volume != volume.to_integral_value() or count != count.to_integral_value():
        raise Rejected('noninteger_count_field')


def lag_check(profile, decision, source_end):
    decisions = ENDS if profile == 'minute' else HOUR_ENDS if profile == 'hour' else ()
    if decision not in decisions:
        raise Rejected('decision_not_on_profile_grid')
    index = decisions.index(decision)
    boundary = (OPEN if index == 0 else decisions[index - 1])
    if source_end is not None:
        if source_end > decision:
            raise Rejected('future_source_forbidden')
        if source_end == decision:
            raise Rejected('current_source_forbidden')
        if source_end not in decisions:
            raise Rejected('source_not_on_profile_grid')
        if source_end > boundary:
            raise Rejected('one_grid_lag_not_met')
    return boundary


def trace(profile, source_ends):
    decisions = ENDS if profile == 'minute' else HOUR_ENDS
    records = []
    for decision in decisions:
        boundary = lag_check(profile, decision, None)
        index = bisect_right(source_ends, boundary) - 1
        selected = source_ends[index] if index >= 0 else None
        lag_check(profile, decision, selected)
        records.append((decision, boundary, selected))
    return records


def review(payload, check_deadline=lambda: None):
    check_deadline()
    if not isinstance(payload, dict) or set(payload) != {'bars', 'next_page_token'}:
        raise Rejected('top_level_field_set_mismatch')
    if payload['next_page_token'] not in (None, ''):
        raise Rejected('pagination_incomplete')
    grouped = payload['bars']
    if not isinstance(grouped, dict) or set(grouped) != {'SPY'}:
        raise Rejected('only_multi_symbol_SPY_wrapper_allowed')
    rows = grouped['SPY']
    if not isinstance(rows, list):
        raise Rejected('bars_not_list')
    starts = []
    seen = set()
    for row in rows:
        check_deadline()
        check_fields(row)
        point = stamp(row['t'])
        if point in seen:
            raise Rejected('duplicate_timestamp')
        if starts and point <= starts[-1]:
            raise Rejected('timestamps_not_strictly_ascending')
        seen.add(point)
        starts.append(point)
    missing = tuple(t for t in STARTS if t not in seen)
    source_ends = tuple(t + MINUTE for t in starts)
    complete_hours = tuple(end for start, end in zip(HOUR_STARTS, HOUR_ENDS)
                           if all(start + i * MINUTE in seen for i in range(60)))
    minute_trace = trace('minute', source_ends)
    hour_trace = trace('hour', complete_hours)
    residual = tuple(t for t in starts if t >= HOUR_ENDS[-1])
    minute_available = sum(row[2] is not None for row in minute_trace)
    hour_available = sum(row[2] is not None for row in hour_trace)
    passed = (len(starts) == 390 and not missing and len(complete_hours) == 6
              and len(residual) == 30 and minute_available == 389 and hour_available == 5)
    check_deadline()
    def example(row):
        return {'decision': iso(row[0]), 'previous_grid_boundary': iso(row[1]),
                'selected_source_end': iso(row[2])}
    def digest(records):
        encoded = [[iso(x) for x in row] for row in records]
        return hashlib.sha256(json.dumps(encoded, separators=(',', ':')).encode()).hexdigest()
    return {
        'accepted': passed,
        'status': 'fixed_one_day_source_clock_qualified' if passed else 'fixed_one_day_support_incomplete',
        'session_UTC': [iso(OPEN), iso(CLOSE)],
        'query_bar_start_inclusive': [iso(STARTS[0]), iso(STARTS[-1])],
        'observed_minute_rows': len(starts), 'expected_minute_rows': 390,
        'missing_minute_rows': len(missing), 'missing_starts_UTC': [iso(x) for x in missing],
        'complete_60minute_source_blocks': len(complete_hours),
        'expected_60minute_blocks': 6, 'observed_residual_minutes': len(residual),
        'residual_interval_UTC': [iso(HOUR_ENDS[-1]), iso(CLOSE)],
        'residual_is_seventh_hour': False,
        'minute_decisions_retained': len(minute_trace),
        'minute_lag_source_available': minute_available,
        'minute_no_source_in_single_day': len(minute_trace) - minute_available,
        'hour_decisions_retained': len(hour_trace),
        'hour_lag_source_available': hour_available,
        'hour_no_source_in_single_day': len(hour_trace) - hour_available,
        'minute_trace_examples': [example(minute_trace[i]) for i in (0, 1, -1)],
        'hour_trace_examples': [example(hour_trace[i]) for i in (0, 1, -1)],
        'minute_trace_sha256': digest(minute_trace), 'hour_trace_sha256': digest(hour_trace),
        'current_future_offgrid_sources_selected': 0,
        'fields_checked': sorted(REQUIRED), 'price_statistics_returns_labels_computed': False,
        'predictive_increment_tested': False, 'market_candidate_admitted': False,
        'historical_received_time_proved': False, 'all_history_coverage_proved': False,
        'research_goal_achieved': False,
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise Rejected('duplicate_JSON_key')
        result[key] = value
    return result


def bad_constant(_):
    raise Rejected('nonfinite_JSON_constant')


def decode(raw):
    try:
        return json.loads(raw.decode('utf-8'), parse_float=Decimal,
                          object_pairs_hook=unique_object, parse_constant=bad_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Rejected('invalid_UTF8_JSON') from exc


def artificial():
    return {'bars': {'SPY': [
        {'t': iso(t), 'o': 100, 'h': 101, 'l': 99, 'c': 100,
         'v': 1, 'n': 1, 'vw': 100} for t in STARTS]}, 'next_page_token': None}


def fixtures():
    import copy
    checks = []
    def good(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
    def rejects(name, expected, mutate=None, payload=None, direct=None):
        specimen = artificial() if payload is None else payload
        if mutate is not None:
            mutate(specimen)
        try:
            direct() if direct is not None else review(specimen)
        except Rejected as exc:
            good(name, str(exc) == expected)
        else:
            raise AssertionError('accepted forbidden fixture: ' + name)
    base = artificial()
    answer = review(base)
    good('full_grid_390', answer['accepted'] and answer['observed_minute_rows'] == 390)
    good('six_full_hours_residual30', answer['complete_60minute_source_blocks'] == 6 and answer['observed_residual_minutes'] == 30)
    good('minute_all390_retained_389_available', answer['minute_decisions_retained'] == 390 and answer['minute_lag_source_available'] == 389)
    good('hour_all6_retained_5_available', answer['hour_decisions_retained'] == 6 and answer['hour_lag_source_available'] == 5)
    good('first_decisions_not_forward_filled', answer['minute_trace_examples'][0]['selected_source_end'] is None and answer['hour_trace_examples'][0]['selected_source_end'] is None)
    good('last_minute_uses_previous_end', answer['minute_trace_examples'][-1]['selected_source_end'] == '2021-01-04T20:59:00Z')
    good('last_hour_uses_previous_end', answer['hour_trace_examples'][-1]['selected_source_end'] == '2021-01-04T19:30:00Z')
    good('residual_not_hour7', not answer['residual_is_seventh_hour'])
    missing = copy.deepcopy(base); del missing['bars']['SPY'][5]
    partial = review(missing)
    good('missing_rejects_complete_qualification', not partial['accepted'] and partial['missing_minute_rows'] == 1)
    good('missing_never_drops_original_decisions', partial['minute_decisions_retained'] == 390 and partial['hour_decisions_retained'] == 6)
    good('gap_invalidates_60minute_block', partial['complete_60minute_source_blocks'] == 5)
    rejects('single_symbol_wrapper_rejected', 'top_level_field_set_mismatch', payload={'bars': [], 'symbol': 'SPY', 'next_page_token': None})
    rejects('wrong_symbol', 'only_multi_symbol_SPY_wrapper_allowed', mutate=lambda d: d['bars'].update({'AAPL': []}))
    rejects('missing_token_key', 'top_level_field_set_mismatch', mutate=lambda d: d.pop('next_page_token'))
    rejects('pagination_not_chased', 'pagination_incomplete', mutate=lambda d: d.update(next_page_token='more'))
    rejects('duplicate_bar', 'duplicate_timestamp', mutate=lambda d: d['bars']['SPY'].insert(1, copy.deepcopy(d['bars']['SPY'][0])))
    rejects('out_of_order', 'timestamps_not_strictly_ascending', mutate=lambda d: d['bars']['SPY'].reverse())
    rejects('after_session_start', 'off_grid_timestamp', mutate=lambda d: d['bars']['SPY'][-1].update(t='2021-01-04T21:00:00Z'))
    rejects('before_session_start', 'off_grid_timestamp', mutate=lambda d: d['bars']['SPY'][0].update(t='2021-01-04T14:29:00Z'))
    rejects('second_offgrid', 'off_grid_timestamp', mutate=lambda d: d['bars']['SPY'][0].update(t='2021-01-04T14:30:01Z'))
    rejects('nanosecond_offgrid', 'off_grid_fractional_timestamp', mutate=lambda d: d['bars']['SPY'][0].update(t='2021-01-04T14:30:00.000000001Z'))
    rejects('naive_timestamp', 'timestamp_not_aware_RFC3339', mutate=lambda d: d['bars']['SPY'][0].update(t='2021-01-04T14:30:00'))
    zero_frac = copy.deepcopy(base); zero_frac['bars']['SPY'][0]['t'] = '2021-01-04T14:30:00.000000000Z'
    good('zero_fraction_allowed', review(zero_frac)['accepted'])
    same_time = copy.deepcopy(base); same_time['bars']['SPY'][0]['t'] = '2021-01-04T09:30:00-05:00'
    good('equivalent_timezone_same_instant', review(same_time)['accepted'])
    rejects('field_missing', 'bar_field_set_mismatch', mutate=lambda d: d['bars']['SPY'][0].pop('vw'))
    rejects('boolean_price', 'invalid_numeric_type_c', mutate=lambda d: d['bars']['SPY'][0].update(c=True))
    rejects('nonfinite_price', 'nonfinite_c', mutate=lambda d: d['bars']['SPY'][0].update(c=float('nan')))
    rejects('negative_volume', 'negative_count_field', mutate=lambda d: d['bars']['SPY'][0].update(v=-1))
    rejects('fractional_trade_count', 'noninteger_count_field', mutate=lambda d: d['bars']['SPY'][0].update(n=Decimal('1.5')))
    rejects('high_inconsistent', 'high_inconsistent', mutate=lambda d: d['bars']['SPY'][0].update(h=99))
    rejects('low_inconsistent', 'low_inconsistent', mutate=lambda d: d['bars']['SPY'][0].update(l=101))
    rejects('nonpositive_close', 'nonpositive_price_field', mutate=lambda d: d['bars']['SPY'][0].update(c=0))
    rejects('current_minute_rejected', 'current_source_forbidden', direct=lambda: lag_check('minute', ENDS[3], ENDS[3]))
    rejects('future_minute_rejected', 'future_source_forbidden', direct=lambda: lag_check('minute', ENDS[3], ENDS[4]))
    rejects('offgrid_minute_source_rejected', 'source_not_on_profile_grid', direct=lambda: lag_check('minute', ENDS[3], ENDS[1] + timedelta(seconds=1)))
    rejects('current_hour_rejected', 'current_source_forbidden', direct=lambda: lag_check('hour', HOUR_ENDS[3], HOUR_ENDS[3]))
    rejects('future_hour_rejected', 'future_source_forbidden', direct=lambda: lag_check('hour', HOUR_ENDS[3], HOUR_ENDS[4]))
    rejects('residual_is_not_hour_source', 'source_not_on_profile_grid', direct=lambda: lag_check('hour', HOUR_ENDS[-1], OPEN + timedelta(minutes=330)))
    rejects('invalid_decision_rejected', 'decision_not_on_profile_grid', direct=lambda: lag_check('hour', CLOSE, HOUR_ENDS[-1]))
    good('independent_JSON_decode', review(decode(json.dumps(base).encode()))['accepted'])
    rejects('duplicate_JSON_keys_rejected', 'duplicate_JSON_key', direct=lambda: decode(b'{"bars":{},"bars":{},"next_page_token":null}'))
    rejects('JSON_NaN_rejected', 'nonfinite_JSON_constant', direct=lambda: decode(b'{"bars":NaN,"next_page_token":null}'))
    rejects('bad_JSON_rejected', 'invalid_UTF8_JSON', direct=lambda: decode(b'{'))
    return {'status': 'independent_artificial_fixtures_passed', 'checks': len(checks),
            'check_names': checks, 'market_response_decoded': False,
            'project_validator_called': False, 'CPU_affinity': sorted(os.sched_getaffinity(0)),
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def atomic_json(path, value):
    if path.exists():
        raise FileExistsError('Review output exists; do not overwrite evidence')
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix='.' + path.name + '.', suffix='.partial', delete=False) as out:
        temporary = Path(out.name)
        json.dump(value, out, sort_keys=True, separators=(',', ':'))
        out.write('\n'); out.flush(); os.fsync(out.fileno())
    os.link(temporary, path)  # Exclusive, atomic publication; existing results survive.
    temporary.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--real-decode-authorized', action='store_true')
    parser.add_argument('--raw', type=Path)
    parser.add_argument('--expected-sha256')
    parser.add_argument('--deadline-epoch', type=float)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if set(os.sched_getaffinity(0)) != {26}:
        raise SystemExit('CPU26 affinity required')
    if args.self_test:
        if args.real_decode_authorized or args.raw or args.expected_sha256 or args.deadline_epoch:
            raise SystemExit('Self-test cannot decode a real response')
        report = fixtures()
    else:
        if not (args.real_decode_authorized and args.raw and args.expected_sha256
                and re.fullmatch('[0-9a-f]{64}', args.expected_sha256)
                and args.deadline_epoch is not None and math.isfinite(args.deadline_epoch)):
            raise SystemExit('Real review requires explicit admission, raw identity and original deadline')
        def deadline():
            if time.time() >= args.deadline_epoch:
                raise Rejected('original_absolute_deadline_reached')
        try:
            deadline()
            if args.raw.stat().st_size > MAX_BYTES:
                raise Rejected('response_exceeds_frozen_1MiB_limit')
            with args.raw.open('rb') as stream:
                raw = stream.read(MAX_BYTES + 1)
            if len(raw) > MAX_BYTES:
                raise Rejected('response_exceeds_frozen_1MiB_limit')
            actual_sha = hashlib.sha256(raw).hexdigest()
            if actual_sha != args.expected_sha256:
                raise Rejected('raw_response_identity_mismatch')
            deadline()
            report = review(decode(raw), deadline)
            report.update(raw_sha256=actual_sha, raw_bytes=len(raw),
                          source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          CPU_affinity=[26], original_deadline_epoch=args.deadline_epoch,
                          project_validator_called=False, HTTP_requests=0, AAPL_arrays_read=False)
        except Rejected as exc:
            report = {'accepted': False, 'status': 'independent_source_review_rejected',
                      'reason': str(exc), 'CPU_affinity': [26],
                      'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      'original_deadline_epoch': args.deadline_epoch,
                      'project_validator_called': False, 'HTTP_requests': 0, 'AAPL_arrays_read': False}
    atomic_json(args.output, report)
    print(json.dumps({'status': report['status'], 'accepted': report.get('accepted'),
                      'checks': report.get('checks'), 'output': str(args.output)}, separators=(',', ':')))
    return 0 if report.get('accepted', args.self_test) else 1


if __name__ == '__main__':
    raise SystemExit(main())
