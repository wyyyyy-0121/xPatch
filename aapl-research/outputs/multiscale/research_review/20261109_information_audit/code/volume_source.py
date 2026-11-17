"""Frozen pre-Q3 volume alignment and strictly past-session normalization.

Importing this module reads no files and starts no data/model operation. Only
timestamp, close (identity control), and volume are converted from CSV values;
the other retained source fields are not numerically inspected. Callers own
the execution authorization, global deadline, CPU allocation and output review.
"""
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile
import os
import re
from zoneinfo import ZoneInfo

import numpy as np

INFO_DIR = Path(__file__).resolve().parents[1]
SOURCE_INFO = INFO_DIR / 'SOURCE_INFORMATION_AVAILABILITY.json'
SOURCE_INFO_SHA = '54a2afdeddcdd19fec63a75d2aa29023105f8f2b2e76c3747020318176df4665'
ADMISSION_SHA = 'f6fc738205532cf5016d74b2baa4691692cb03674c3db9b6a8ccf8ce8126ae86'
RAW_SHA = 'dafb2d2ac05ae7652fea0334caea209457db6418368efdcad8b7d648cdfb5830'
BAR_SHA = {
    'minute': 'c81d57cd1e190456e1fdbf59d7493c915704551fa2e6c5870b5bd66f4cd94332',
    'hour': 'c41851d735d179c3243dd3356204d17845b55289c56c59f9e170c0b5f432881c',
}
EXPECTED_COUNTS = {'all': 1061536, 'regular': 535620, 'outside_regular': 525916,
                   'hour': 8238, 'sessions': 1378, 'residual': 41340}
CSV_FIELDS = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']
BAR_FIELDS = {'start_ns', 'end_ns', 'close', 'observed', 'session_date'}
CACHE_FIELDS = BAR_FIELDS | {'volume', 'abnormal', 'valid', 'slot'}
NY = ZoneInfo('America/New_York')
MINUTE_NS = 60_000_000_000
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
CUTOFF_DATE = date(2026, 6, 30)
FIRST_DATE = date(2021, 1, 4)
RAW_FIRST = '2021-01-04T09:00:00+00:00'
RAW_LAST = '2026-06-30T23:59:00+00:00'
WINDOW_SESSIONS = 20
STAMP_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:00(?:\.0+)?(?:Z|[+-]\d{2}:\d{2})$')


def _guard(callback):
    if callback is not None:
        callback()


def _sha(path, guard=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            _guard(guard)
            digest.update(block)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _atomic_json(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as handle:
            temporary = handle.name
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_npz(path, arrays):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('wb', dir=path.parent, delete=False) as handle:
            temporary = handle.name
            np.savez_compressed(handle, **arrays)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def _timestamp(value):
    """Require an explicit timezone and an exact minute; no float timestamp."""
    if not value or STAMP_PATTERN.fullmatch(value) is None:
        raise ValueError('CSV timestamp must be an exact minute with explicit timezone')
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if stamp.tzinfo is None or stamp.utcoffset() is None or stamp.second or stamp.microsecond:
        raise ValueError('CSV timestamp is not an exact timezone-aware minute')
    utc = stamp.astimezone(timezone.utc)
    elapsed = utc - EPOCH
    ns = (elapsed.days * 86400 + elapsed.seconds) * 1_000_000_000
    return ns, utc.astimezone(NY)


def _local_clock(starts, guard=None):
    dates, minutes = [], np.empty(len(starts), dtype=np.int64)
    for i, value in enumerate(starts):
        if i % 4096 == 0:
            _guard(guard)
        seconds, remainder = divmod(int(value), 1_000_000_000)
        if remainder or seconds % 60:
            raise ValueError('Bar start is not an exact minute')
        local = (EPOCH + timedelta(seconds=seconds)).astimezone(NY)
        dates.append(local.date().isoformat())
        minutes[i] = local.hour * 60 + local.minute
    return np.asarray(dates, dtype='U10'), minutes


def _validate_bars(bars, frequency, guard=None):
    if set(bars) != BAR_FIELDS:
        raise ValueError('Unexpected original close-only bar fields')
    n = len(bars['start_ns'])
    if not n or any(value.ndim != 1 or len(value) != n for value in bars.values()):
        raise ValueError('Original bars need matching nonempty flat arrays')
    if bars['start_ns'].dtype != np.dtype('int64') or bars['end_ns'].dtype != np.dtype('int64'):
        raise ValueError('Original bar nanoseconds must be int64')
    if bars['observed'].dtype != np.dtype('bool') or not bars['observed'].all():
        raise ValueError('Frozen volume study requires every original bar observed; no filling')
    if bars['close'].dtype != np.dtype('float64') or not np.isfinite(bars['close']).all() or (
            bars['close'] <= 0).any():
        raise ValueError('Original close must be finite positive float64')
    if bars['session_date'].dtype.kind != 'U':
        raise ValueError('Original session dates must be fixed Unicode')
    if (np.diff(bars['start_ns']) <= 0).any() or (np.diff(bars['end_ns']) <= 0).any():
        raise ValueError('Duplicate or unordered bar timestamps')
    duration = MINUTE_NS * (1 if frequency == 'minute' else 60)
    if not np.array_equal(bars['end_ns'] - bars['start_ns'], np.full(n, duration, dtype=np.int64)):
        raise ValueError('Bar durations differ from frozen completed-bar definitions')
    local_dates, local_minutes = _local_clock(bars['start_ns'], guard)
    if not np.array_equal(local_dates, bars['session_date']):
        raise ValueError('New York session labels differ from bar starts')
    if (local_dates < FIRST_DATE.isoformat()).any() or (local_dates > CUTOFF_DATE.isoformat()).any():
        raise ValueError('A bar lies outside the explicit pre-Q3 source period')
    if ((local_minutes < 570) | (local_minutes + (1 if frequency == 'minute' else 60) > 960)).any():
        raise ValueError('A bar is outside the frozen regular-session clock')
    if (bars['start_ns'][1:] < bars['end_ns'][:-1]).any():
        raise ValueError('Original bar intervals overlap')
    slots = local_minutes - 570
    if frequency == 'hour':
        if (slots % 60).any():
            raise ValueError('Hours must start at09:30 plus complete60-minute blocks')
        slots //= 60
    _guard(guard)
    return slots


def _load_original(path, frequency, guard):
    _guard(guard)
    with np.load(path, allow_pickle=False) as saved:
        bars = {name: saved[name].copy() for name in saved.files}
    if _sha(path, guard) != BAR_SHA[frequency]:
        raise ValueError('Original bar bytes changed during deserialization')
    slots = _validate_bars(bars, frequency, guard)
    return bars, slots


def _identity(guard):
    """Check the explicitly reviewed lineage before market deserialization."""
    if _sha(SOURCE_INFO, guard) != SOURCE_INFO_SHA:
        raise ValueError('Reviewed source-information receipt changed')
    info = _read(SOURCE_INFO)
    source = info['minute_source']
    if (source['recorded_source_sha256'] != RAW_SHA or source['recorded_first_UTC'] != RAW_FIRST
            or source['recorded_last_UTC'] != RAW_LAST or source['recorded_ranges_end'] != '2026-06-30'):
        raise ValueError('The explicitly pre-Q3 minute source identity changed')
    raw = Path(source['path']).resolve()
    if 'holdout_sealed' in raw.parts:
        raise ValueError('Sealed sources are never allowed')
    directory = Path(info['current_multiscale_caches']['directory']).resolve()
    admission_path = directory / 'admission.json'
    if _sha(admission_path, guard) != ADMISSION_SHA:
        raise ValueError('Original close-only admission receipt changed')
    admission = _read(admission_path)
    if admission['development_cutoff_new_york'] != '2026-06-30' or admission['Q3_values_read'] is not False:
        raise ValueError('Original dataset is not the approved pre-Q3 source')
    identity = {'source_information_sha256': SOURCE_INFO_SHA, 'admission_sha256': ADMISSION_SHA,
                'source_csv_path': str(raw), 'source_csv_sha256': _sha(raw, guard),
                'code_sha256': _sha(Path(__file__), guard), 'bars_sha256': {},
                'formula': 'log1pV minus same-slot median of the immediately preceding20global sessions; require all20slots present',
                'window_sessions': WINDOW_SESSIONS, 'close_comparison': 'exact_float64_equality_no_tolerance'}
    if identity['source_csv_sha256'] != RAW_SHA:
        raise ValueError('Original pre-Q3 CSV bytes changed')
    for frequency, expected in BAR_SHA.items():
        if admission['files']['bars_' + frequency + '.npz'] != expected:
            raise ValueError('Original bar identity differs from frozen source review')
        actual = _sha(directory / ('bars_' + frequency + '.npz'), guard)
        if actual != expected:
            raise ValueError('Original bar bytes changed: ' + frequency)
        identity['bars_sha256'][frequency] = actual
    return identity, raw, directory


def align_csv_volume(path, minute_bars, guard=None, *, expected_counts=None):
    """Strict streaming projection onto the original regular-minute grid.

    Used by load_volume after source hash checks. The optional counts argument
    also permits tiny independent synthetic CSV fixtures; dates remain pre-Q3.
    Other source columns are consumed as CSV text but never converted or used.
    """
    starts = minute_bars['start_ns']
    volume = np.full(len(starts), np.nan, dtype=np.float64)
    cursor, total, outside, previous = 0, 0, 0, None
    first = last = None
    with Path(path).open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_FIELDS:
            raise ValueError('Original minute CSV schema changed')
        for row in reader:
            if total % 2048 == 0:
                _guard(guard)
            if set(row) != set(CSV_FIELDS) or any(row[key] is None for key in CSV_FIELDS):
                raise ValueError('Malformed or truncated CSV row')
            ns, local = _timestamp(row['timestamp'])
            if not FIRST_DATE <= local.date() <= CUTOFF_DATE:
                raise ValueError('CSV row crosses the explicit New York pre-Q3 date bound')
            if previous is not None and ns <= previous:
                raise ValueError('CSV has duplicate or unordered timestamps')
            previous, last = ns, ns
            if first is None:
                first = ns
            close, observed_volume = float(row['close']), float(row['volume'])
            if not math.isfinite(close) or close <= 0 or not math.isfinite(observed_volume) or observed_volume < 0:
                raise ValueError('Nonfinite/nonpositive close or nonfinite/negative volume')
            if cursor < len(starts) and ns > int(starts[cursor]):
                raise ValueError('The CSV is missing an expected regular-session minute')
            if cursor < len(starts) and ns == int(starts[cursor]):
                if close != float(minute_bars['close'][cursor]):
                    raise ValueError('CSV close differs exactly from original bars at index' + str(cursor))
                volume[cursor] = observed_volume
                cursor += 1
            else:
                outside += 1
            total += 1
    _guard(guard)
    if cursor != len(starts) or not np.isfinite(volume).all():
        raise ValueError('Not every original minute has exactly one finite source volume')
    counts = {'all': total, 'regular': cursor, 'outside_regular': outside}
    if expected_counts is not None and counts != expected_counts:
        raise ValueError('CSV source/RTH/outside counts differ: ' + repr(counts))
    if expected_counts == {key: EXPECTED_COUNTS[key] for key in ('all', 'regular', 'outside_regular')}:
        if first != _timestamp(RAW_FIRST)[0] or last != _timestamp(RAW_LAST)[0]:
            raise ValueError('CSV first/last timestamp differs from frozen receipt')
    return volume, counts


def aggregate_hour_volume(minute_bars, hour_bars, volume, guard=None):
    """Sum precisely the old session-open-anchored60-minute groups; no residuals."""
    if np.asarray(volume).shape != minute_bars['start_ns'].shape or not np.isfinite(volume).all() or (volume < 0).any():
        raise ValueError('Need nonnegative finite volume for every original minute')
    dates = minute_bars['session_date']
    first = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1]])
    last = np.r_[first[1:], len(dates)]
    expected_groups = []
    residual = 0
    for a, b in zip(first, last):
        _guard(guard)
        if not np.array_equal(minute_bars['start_ns'][a:b],
                minute_bars['start_ns'][a] + MINUTE_NS * np.arange(b-a, dtype=np.int64)):
            raise ValueError('Nonconsecutive minutes within a regular session')
        residual += (int(b) - int(a)) % 60
        expected_groups.extend(range(int(a), int(b) - 59, 60))
    indices = np.asarray(expected_groups, dtype=np.int64)
    if len(indices) != len(hour_bars['start_ns']):
        raise ValueError('Original full-hour count differs from exact minute grouping')
    ends = indices + 59
    for key, minute_values, hour_values in (
            ('start_ns', minute_bars['start_ns'][indices], hour_bars['start_ns']),
            ('end_ns', minute_bars['end_ns'][ends], hour_bars['end_ns']),
            ('close', minute_bars['close'][ends], hour_bars['close']),
            ('session_date', minute_bars['session_date'][indices], hour_bars['session_date'])):
        if not np.array_equal(minute_values, hour_values):
            raise ValueError('Original hour ' + key + ' differs from exact60-minute grouping')
    hour_volume = np.empty(len(indices), dtype=np.float64)
    for i, a in enumerate(indices):
        if i % 512 == 0:
            _guard(guard)
        hour_volume[i] = math.fsum(float(x) for x in volume[a:a+60])
    if not np.isfinite(hour_volume).all():
        raise ValueError('Nonfinite exact-group volume sum')
    return hour_volume, {'full_hours': len(indices), 'sessions': len(first),
                         'residual_minutes_excluded': residual,
                         'sum_rule': 'math.fsum of exactly60aligned source volumes',
                         'start_end_close_session_exact': True}


def abnormal_by_session(volume, session_date, slots, session_order, guard=None):
    """Use last20GLOBAL completed sessions, requiring all20same-slot values.

    Half-day absent slots remain missing in the session-by-slot matrix. They
    invalidate a baseline until that session leaves the20-session window; no
    older sessions are substituted. Today's entire session is excluded.
    """
    volume = np.asarray(volume, dtype=np.float64)
    dates, slots, sessions = map(np.asarray, (session_date, slots, session_order))
    if volume.ndim != 1 or dates.shape != volume.shape or slots.shape != volume.shape or not len(volume):
        raise ValueError('Need matching nonempty volume/date/slot vectors')
    if not np.isfinite(volume).all() or (volume < 0).any() or slots.dtype.kind not in 'iu' or (slots < 0).any():
        raise ValueError('Volume must be finite nonnegative; slots nonnegative integers')
    if sessions.ndim != 1 or len(sessions) == 0 or len(np.unique(sessions)) != len(sessions) or (
            sessions[1:] <= sessions[:-1]).any():
        raise ValueError('Global trading sessions must be unique and ordered')
    session_index = np.searchsorted(sessions, dates)
    if (session_index >= len(sessions)).any() or not np.array_equal(sessions[session_index], dates):
        raise ValueError('A source session is absent from the original global session grid')
    width = int(slots.max()) + 1
    matrix = np.full((len(sessions), width), np.nan, dtype=np.float64)
    assigned = np.zeros(matrix.shape, dtype=bool)
    log_volume = np.log1p(volume)
    for i, (s, slot) in enumerate(zip(session_index, slots)):
        if i % 4096 == 0:
            _guard(guard)
        if assigned[s, slot]:
            raise ValueError('Duplicate session/slot observation')
        matrix[s, slot] = log_volume[i]
        assigned[s, slot] = True
    abnormal_matrix = np.full(matrix.shape, np.nan, dtype=np.float64)
    for s in range(WINDOW_SESSIONS, len(sessions)):
        _guard(guard)
        history = matrix[s-WINDOW_SESSIONS:s]
        valid_slots = np.isfinite(history).all(axis=0) & np.isfinite(matrix[s])
        abnormal_matrix[s, valid_slots] = (matrix[s, valid_slots]
                - np.median(history[:, valid_slots], axis=0))
    result = abnormal_matrix[session_index, slots]
    valid = np.isfinite(result)
    _guard(guard)
    return result, valid


def _check_cached(arrays, frequency, guard):
    if set(arrays) != CACHE_FIELDS:
        raise ValueError('Unexpected volume cache fields')
    slots = _validate_bars({key: arrays[key] for key in BAR_FIELDS}, frequency, guard)
    if not np.array_equal(slots, arrays['slot']):
        raise ValueError('Cached New York slots changed')
    n = len(slots)
    for name in ('volume', 'abnormal', 'valid'):
        if arrays[name].ndim != 1 or len(arrays[name]) != n:
            raise ValueError('Cached derived arrays do not align with original bars')
    if not np.isfinite(arrays['volume']).all() or (arrays['volume'] < 0).any() or np.isinf(arrays['abnormal']).any() or (
            arrays['valid'].dtype != np.dtype('bool')) or not np.array_equal(
            arrays['valid'], np.isfinite(arrays['abnormal'])):
        raise ValueError('Invalid cached volume/abnormal validity')


def load_volume(cache_dir, guard):
    """Return minute/hour dictionaries aligned to the unchanged original bars.

    Keys per frequency: volume, abnormal, valid, start_ns, end_ns, close,
    observed, session_date, slot. `_receipt` contains provenance and counts.
    The guard is called during hashing, parsing, grouping and normalization.
    Completed caches are atomic and source/code-bound; no model/target is read.
    """
    if not callable(guard):
        raise ValueError('A deadline/resource guard callback is required')
    guard()
    cache_dir = Path(cache_dir).resolve()
    if not cache_dir.is_relative_to(INFO_DIR) or cache_dir in (INFO_DIR, INFO_DIR/'code'):
        raise ValueError('Use a separate cache directory inside this information audit')
    identity, raw, directory = _identity(guard)
    receipt_path = cache_dir / 'VOLUME_SOURCE_RECEIPT.json'
    if receipt_path.exists():
        receipt = _read(receipt_path)
        if receipt.get('status') != 'complete' or receipt.get('identity') != identity:
            raise ValueError('Existing volume cache belongs to different source/code identity')
        result = {}
        for frequency in ('minute', 'hour'):
            path = cache_dir / (frequency + '_volume.npz')
            if _sha(path, guard) != receipt['cache_sha256'][frequency]:
                raise ValueError('Completed volume cache bytes changed')
            with np.load(path, allow_pickle=False) as saved:
                arrays = {name: saved[name].copy() for name in saved.files}
            _check_cached(arrays, frequency, guard)
            result[frequency] = arrays
        result['_receipt'] = receipt
        guard()
        return result
    minute, minute_slots = _load_original(directory/'bars_minute.npz', 'minute', guard)
    hour, hour_slots = _load_original(directory/'bars_hour.npz', 'hour', guard)
    counts_expected = {key: EXPECTED_COUNTS[key] for key in ('all', 'regular', 'outside_regular')}
    minute_volume, counts = align_csv_volume(raw, minute, guard, expected_counts=counts_expected)
    if _sha(raw, guard) != RAW_SHA:
        raise ValueError('Original CSV changed during streaming alignment')
    hour_volume, grouping = aggregate_hour_volume(minute, hour, minute_volume, guard)
    if grouping['full_hours'] != EXPECTED_COUNTS['hour'] or grouping['sessions'] != EXPECTED_COUNTS['sessions'] or (
            grouping['residual_minutes_excluded'] != EXPECTED_COUNTS['residual']):
        raise ValueError('Full-hour/residual/session identity differs from original admission')
    sessions = np.unique(minute['session_date'])
    result = {}
    for frequency, bars, volume, slots in (
            ('minute', minute, minute_volume, minute_slots), ('hour', hour, hour_volume, hour_slots)):
        abnormal, valid = abnormal_by_session(volume, bars['session_date'], slots, sessions, guard)
        result[frequency] = dict(bars, volume=volume, abnormal=abnormal, valid=valid, slot=slots)
    guard()
    cache_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for frequency in ('minute', 'hour'):
        path = cache_dir / (frequency + '_volume.npz')
        _atomic_npz(path, result[frequency])
        hashes[frequency] = _sha(path, guard)
    receipt = {'status': 'complete', 'identity': identity, 'source_counts': counts,
        'hour_aggregation': grouping, 'cache_sha256': hashes,
        'normalization': {'previous_global_sessions': WINDOW_SESSIONS,
            'require_all20_same_slot': True, 'current_session_excluded': True,
            'half_day_missing_slots_filled': False, 'zero_volume_is_real_zero': True,
            'valid_bars': {name: int(result[name]['valid'].sum()) for name in ('minute', 'hour')}},
        'fields_numerically_used': ['timestamp', 'close_identity_only', 'volume'],
        'close_equality': 'exact_float64_equal_no_tolerance',
        'missing_volume_filled': False, 'Q3_values_read': False,
        'target_or_model_loaded': False, 'point_in_time_availability_established': False,
        'asof_limit': 'Historical final bars; earliest assumed availability is bar completion, revision/receipt latency unresolved',
        'research_goal_achieved': False}
    guard()
    _atomic_json(receipt_path, receipt)
    result['_receipt'] = receipt
    return result


def synthetic_acceptance():
    """Synthetic-only fixtures, callable later; never run at module import."""
    dates = np.asarray([(date(2021, 1, 4) + timedelta(days=i)).isoformat() for i in range(44)], dtype='U10')
    row_dates, slots, volumes = [], [], []
    for s, day in enumerate(dates):
        for slot in (0, 1):
            if s == 5 and slot == 1:
                continue  # Half-day-style missing position, not volume zero.
            row_dates.append(day); slots.append(slot)
            volumes.append(0.0 if slot == 0 else float(s+1))
    row_dates, slots, volumes = np.asarray(row_dates), np.asarray(slots), np.asarray(volumes)
    abnormal, valid = abnormal_by_session(volumes, row_dates, slots, dates)
    session_index = np.searchsorted(dates, row_dates)
    assert not valid[session_index < 20].any()
    assert valid[(session_index >= 20) & (slots == 0)].all()
    assert np.array_equal(abnormal[(session_index >= 20) & (slots == 0)],
                          np.zeros(np.sum((session_index >= 20) & (slots == 0))))
    assert not valid[(session_index >= 20) & (session_index <= 25) & (slots == 1)].any()
    assert valid[(session_index >= 26) & (slots == 1)].all()
    changed = volumes.copy(); changed[session_index >= 30] += 9999
    changed_abnormal, changed_valid = abnormal_by_session(changed, row_dates, slots, dates)
    assert np.array_equal(abnormal[session_index < 30], changed_abnormal[session_index < 30], equal_nan=True)
    assert np.array_equal(valid[session_index < 30], changed_valid[session_index < 30])

    # Independent scalar/sorted oracle and positive response: a single changed
    # historical session can move a later median, but leaves the window after20.
    ramp = 10.0 + np.arange(len(dates), dtype=np.float64)
    ramp_slots = np.zeros(len(dates), dtype=np.int64)
    original_ramp, original_ramp_valid = abnormal_by_session(ramp, dates, ramp_slots, dates)
    changed_ramp = ramp.copy(); changed_ramp[21] += 10000.0
    response, response_valid = abnormal_by_session(changed_ramp, dates, ramp_slots, dates)
    assert np.array_equal(original_ramp_valid, response_valid)
    for s in range(20, len(dates)):
        reference = sorted(math.log1p(float(v)) for v in changed_ramp[s-20:s])
        expected = math.log1p(float(changed_ramp[s])) - (reference[9] + reference[10])/2.0
        assert math.isclose(float(response[s]), expected, rel_tol=0.0, abs_tol=2e-15)
    assert response[21] > original_ramp[21]  # Current abnormal must actually react.
    assert response[32] < original_ramp[32]  # The changed past value moves the median.
    assert np.array_equal(response[:21], original_ramp[:21], equal_nan=True)
    assert np.array_equal(response[42:], original_ramp[42:], equal_nan=True)
    affected = np.flatnonzero(response_valid & (response != original_ramp))
    assert len(affected) > 1 and affected.min() == 21 and affected.max() <= 41

    # Future sessions can be added and shortened/lengthened without changing
    # earlier features. This changes matrix shape AND future slot availability.
    prefix = session_index < 30
    extended_dates = np.asarray([(date(2021, 1, 4)+timedelta(days=i)).isoformat()
                                 for i in range(46)], dtype='U10')
    future_dates, future_slots, future_volumes = [], [], []
    for s in range(30, len(extended_dates)):
        for slot in range(1 if s % 2 == 0 else 3):
            future_dates.append(extended_dates[s]); future_slots.append(slot)
            future_volumes.append(float(10000+s+slot))
    varied_dates = np.concatenate([row_dates[prefix], np.asarray(future_dates)])
    varied_slots = np.concatenate([slots[prefix], np.asarray(future_slots, dtype=np.int64)])
    varied_volumes = np.concatenate([volumes[prefix], np.asarray(future_volumes)])
    varied, varied_valid = abnormal_by_session(varied_volumes, varied_dates, varied_slots, extended_dates)
    prefix_count = int(prefix.sum())
    assert np.array_equal(varied[:prefix_count], abnormal[prefix], equal_nan=True)
    assert np.array_equal(varied_valid[:prefix_count], valid[prefix])

    # One complete session with150minutes: two full hours and30excluded residuals.
    first_ns = _timestamp('2021-01-04T09:30:00-05:00')[0]
    starts = first_ns + MINUTE_NS*np.arange(150, dtype=np.int64)
    close = np.arange(150, dtype=np.float64)+100
    minute = {'start_ns': starts, 'end_ns': starts+MINUTE_NS, 'close': close,
        'observed': np.ones(150, dtype=bool), 'session_date': np.repeat('2021-01-04', 150)}
    idx = np.asarray([0, 60]); end = idx+59
    hour = {'start_ns': starts[idx], 'end_ns': (starts+MINUTE_NS)[end], 'close': close[end],
        'observed': np.ones(2, dtype=bool), 'session_date': np.repeat('2021-01-04', 2)}
    volume = np.arange(150, dtype=np.float64)
    summed, metadata = aggregate_hour_volume(minute, hour, volume)
    assert np.array_equal(summed, [1770., 5370.]) and metadata['residual_minutes_excluded'] == 30
    corrupt = dict(hour, end_ns=hour['end_ns'].copy()); corrupt['end_ns'][0] += MINUTE_NS
    try:
        aggregate_hour_volume(minute, corrupt, volume)
    except ValueError:
        pass
    else:
        raise AssertionError('A changed hour completion boundary was admitted')

    # Real streaming parser path, but only a freshly generated tiny fake CSV.
    # Unused fields intentionally contain nonnumeric text to demonstrate that
    # the volume path does not convert OHLC/trade_count/VWAP values.
    small = {key: value[:3].copy() for key, value in minute.items()}
    fixture_rows = []
    for clock, price, amount in [('09:29', 99.0, 3.0), ('09:30', 100.0, 0.0),
                                 ('09:31', 101.0, 2.0), ('09:32', 102.0, 5.0)]:
        row = {key: 'unused_not_converted' for key in CSV_FIELDS}
        row.update(timestamp='2021-01-04T'+clock+':00-05:00',
                   close=repr(price), volume=repr(amount))
        fixture_rows.append(row)
    expected_counts = {'all': 4, 'regular': 3, 'outside_regular': 1}
    with tempfile.TemporaryDirectory(prefix='aapl_volume_synthetic_') as temporary:
        path = Path(temporary)/'synthetic_source.csv'

        def write_rows(rows):
            with path.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                writer.writeheader(); writer.writerows(rows)

        write_rows(fixture_rows)
        aligned, counts = align_csv_volume(path, small, expected_counts=expected_counts)
        assert np.array_equal(aligned, [0.0, 2.0, 5.0]) and counts == expected_counts
        bad_negative = [dict(row) for row in fixture_rows]
        bad_negative[2]['volume'] = '-1'
        bad_close = [dict(row) for row in fixture_rows]
        bad_close[2]['close'] = '101.00000000000001'
        rejection_cases = {
            'missing_regular_minute': fixture_rows[:2]+fixture_rows[3:],
            'duplicate_timestamp': fixture_rows[:1]+fixture_rows,
            'negative_volume': bad_negative,
            'nonexact_close': bad_close,
        }
        for label, rows in rejection_cases.items():
            write_rows(rows)
            try:
                align_csv_volume(path, small, expected_counts=expected_counts)
            except ValueError:
                pass
            else:
                raise AssertionError('Streaming fixture admitted '+label)
    return {'status': 'passed', 'synthetic_only': True, 'market_data_read': False,
        'checks': ['at_least20_global_sessions', 'missing_slot_not_backfilled',
                   'zero_volume_is_zero', 'future_changes_do_not_affect_past',
                   'historical_positive_response_matches_independent_sorted_oracle',
                   'single_session_effect_expires_after20session_window',
                   'future_session_addition_and_length_do_not_affect_past',
                   'exact60minute_hour_sum_and_residual_exclusion', 'reject_changed_hour_end',
                   'streaming_CSV_exact_alignment_and_zero', 'reject_missing_minute',
                   'reject_duplicate_timestamp', 'reject_negative_volume', 'reject_nonexact_close']}
