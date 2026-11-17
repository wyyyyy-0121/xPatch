"""Strict OHLC/CLV source adapter; importing starts no data operation.

Only an explicit ``load_ohlc(cache_dir, guard)`` can enter the frozen real-source
path. Lower-level functions also accept newly created synthetic fixtures. This
module never constructs targets, reads a model, or decides market admission.
Callers own the frozen protocol, resource/deadline guard, and execution approval.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys


STAGE_DIR = Path(__file__).resolve().parents[1]
LEGACY_PATH = STAGE_DIR.parent / '20261109_information_audit/code/volume_source.py'
LEGACY_SHA256 = '026cdf3ba8dfa1fb815d47e58409be5ee9028a746d94e3eeba59955b2d375fb0'
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')
PRICE_FIELDS = ('open', 'high', 'low', 'close')
CLV_FORMULA = '((close-low)-(high-close))/(high-low); valid high==low -> 0; no clip/epsilon'
SCHEMA_VERSION = 'ohlc_clv_source_v1'
_LEGACY = None


def _numpy():
    if any(os.environ.get(name) != '1' for name in THREAD_VARIABLES):
        raise RuntimeError('Set every declared numeric thread environment variable to 1 before Python starts')
    import numpy as np
    return np


def _guard(callback):
    if callback is not None:
        callback()


def file_sha256(path, guard=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            _guard(guard)
            digest.update(block)
    return digest.hexdigest()


def require_sha256(path, expected, guard=None):
    actual = file_sha256(path, guard)
    if actual != expected:
        raise ValueError('Source/cache hash mismatch: ' + str(path))
    return actual


def legacy_volume():
    """Load only the hash-bound old implementation, never its real loader."""
    global _LEGACY
    _numpy()
    require_sha256(LEGACY_PATH, LEGACY_SHA256)
    if _LEGACY is None:
        name = '_clv_frozen_volume_source'
        spec = importlib.util.spec_from_file_location(name, LEGACY_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError('Cannot load frozen volume-source implementation')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        _LEGACY = module
    return _LEGACY


def validate_ohlc(open_price, high, low, close):
    """Reject all invalid prices; a zero range is valid only with O=C=H=L."""
    np = _numpy()
    values = [np.asarray(value, dtype=np.float64) for value in (open_price, high, low, close)]
    if not values or any(value.ndim != 1 or value.shape != values[0].shape for value in values):
        raise ValueError('OHLC requires equal flat arrays')
    if not len(values[0]) or any(not np.isfinite(value).all() or (value <= 0).any() for value in values):
        raise ValueError('Every source OHLC value must be finite and positive')
    op, hi, lo, cl = values
    if ((hi < lo) | (op < lo) | (op > hi) | (cl < lo) | (cl > hi)).any():
        raise ValueError('Source OHLC violates low <= open,close <= high')
    return op, hi, lo, cl


def close_location(open_price, high, low, close):
    np = _numpy()
    _, hi, lo, cl = validate_ohlc(open_price, high, low, close)
    width = hi - lo
    result = np.zeros(cl.shape, dtype=np.float64)
    nonzero = width != 0.0
    # No epsilon, clipping, relaxed boundary or 2*C overflow-prone numerator.
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        result[nonzero] = ((cl[nonzero] - lo[nonzero]) -
                           (hi[nonzero] - cl[nonzero])) / width[nonzero]
    if not np.isfinite(result).all() or ((result < -1.0) | (result > 1.0)).any():
        raise ValueError('Computed CLV is outside the exact finite [-1,1] contract')
    return result


def align_csv_ohlc(path, minute_bars, guard=None, *, expected_sha256, expected_counts=None):
    """Validate every CSV row and project OHLC/volume onto the old minute grid.

    Extending the old parser does not inherit its intentionally unconverted
    OHLC values. Even rows outside RTH must have valid OHLC and volume. No bad
    source row is removed according to labels or a prospective performance mask.
    ``timestamp`` is the left boundary of the original completed-minute bar.
    """
    np = _numpy()
    old = legacy_volume()
    old._validate_bars(minute_bars, 'minute', guard)
    require_sha256(path, expected_sha256, guard)
    starts = minute_bars['start_ns']
    fields = {name: np.empty(len(starts), dtype=np.float64) for name in PRICE_FIELDS}
    volume = np.empty(len(starts), dtype=np.float64)
    cursor = total = outside = 0
    previous = first = last = None
    with Path(path).open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != old.CSV_FIELDS:
            raise ValueError('Original eight-column CSV schema changed')
        for row in reader:
            if total % 2048 == 0:
                _guard(guard)
            if set(row) != set(old.CSV_FIELDS) or any(row[key] is None for key in old.CSV_FIELDS):
                raise ValueError('Malformed or truncated source CSV row')
            ns, local = old._timestamp(row['timestamp'])
            if not old.FIRST_DATE <= local.date() <= old.CUTOFF_DATE:
                raise ValueError('Source row crosses the explicit pre-Q3 date boundary')
            if previous is not None and ns <= previous:
                raise ValueError('Duplicate or unordered source timestamps')
            previous = last = ns
            if first is None:
                first = ns
            op, hi, lo, cl = (float(row[name]) for name in PRICE_FIELDS)
            if any(not math.isfinite(value) or value <= 0 for value in (op, hi, lo, cl)):
                raise ValueError('Every source OHLC value must be finite and positive')
            if hi < lo or not lo <= op <= hi or not lo <= cl <= hi:
                raise ValueError('Source row violates the OHLC range contract')
            amount = float(row['volume'])
            if not math.isfinite(amount) or amount < 0:
                raise ValueError('Source volume must be finite and nonnegative')
            if cursor < len(starts) and ns > int(starts[cursor]):
                raise ValueError('Source is missing an expected original regular minute')
            if cursor < len(starts) and ns == int(starts[cursor]):
                if cl != float(minute_bars['close'][cursor]):
                    raise ValueError('Source close differs exactly from original minute close')
                for name, value in zip(PRICE_FIELDS, (op, hi, lo, cl)):
                    fields[name][cursor] = value
                volume[cursor] = amount
                cursor += 1
            else:
                outside += 1
            total += 1
    _guard(guard)
    if cursor != len(starts):
        raise ValueError('Source did not supply every original regular minute')
    counts = {'all': total, 'regular': cursor, 'outside_regular': outside}
    if expected_counts is not None and counts != expected_counts:
        raise ValueError('Source row counts differ from the frozen contract')
    frozen_counts = {name: old.EXPECTED_COUNTS[name] for name in counts}
    if expected_counts == frozen_counts and (
            first != old._timestamp(old.RAW_FIRST)[0] or last != old._timestamp(old.RAW_LAST)[0]):
        raise ValueError('Original first/last source timestamps changed')
    require_sha256(path, expected_sha256, guard)
    fields['volume'] = volume
    fields['clv'] = close_location(*(fields[name] for name in PRICE_FIELDS))
    return fields, {'counts': counts,
                    'first_source_start_ns': first, 'last_source_start_ns': last,
                    'timestamp_label': 'bar_start', 'close_comparison': 'exact_float64_equal'}


def aggregate_hour_ohlc(minute_bars, hour_bars, minute_ohlc, guard=None):
    """Reuse old exact hour-group validation; aggregate prices, never mean CLV."""
    np = _numpy()
    old = legacy_volume()
    old._validate_bars(minute_bars, 'minute', guard)
    old._validate_bars(hour_bars, 'hour', guard)
    validate_ohlc(*(minute_ohlc[name] for name in PRICE_FIELDS))
    if any(np.asarray(minute_ohlc[name]).shape != minute_bars['close'].shape
           for name in PRICE_FIELDS):
        raise ValueError('Minute OHLC does not match the original grid')
    if not np.array_equal(minute_ohlc['close'], minute_bars['close']):
        raise ValueError('Minute OHLC close identity differs')
    hour_volume, grouping = old.aggregate_hour_volume(
        minute_bars, hour_bars, minute_ohlc['volume'], guard)
    # The legacy function just certified precisely these session-open blocks,
    # including start/end/close/session identity and residual exclusion.
    starts = np.searchsorted(minute_bars['start_ns'], hour_bars['start_ns'])
    out = {name: np.empty(len(starts), dtype=np.float64) for name in PRICE_FIELDS}
    for i, start in enumerate(starts):
        _guard(guard)
        stop = int(start) + 60
        out['open'][i] = minute_ohlc['open'][start]
        out['high'][i] = np.max(minute_ohlc['high'][start:stop])
        out['low'][i] = np.min(minute_ohlc['low'][start:stop])
        out['close'][i] = minute_ohlc['close'][stop - 1]
    if not np.array_equal(out['close'], hour_bars['close']):
        raise ValueError('Aggregated hourly last close changed')
    out['volume'] = hour_volume
    out['clv'] = close_location(*(out[name] for name in PRICE_FIELDS))
    return out, dict(grouping, price_rule='first open,max high,min low,last close of original 60-minute group',
                     clv_rule='compute after OHLC aggregation; never average minute CLV')


def assemble_sources(minute_bars, hour_bars, minute_ohlc, guard=None):
    """Return only original bar fields plus OHLC/CLV, not a volume experiment.

    The enclosing screen uses its separately qualified old volume cache; this
    adapter does not write another volume/normalization cache implicitly.
    """
    np = _numpy()
    old = legacy_volume()
    old._validate_bars(minute_bars, 'minute', guard)
    old._validate_bars(hour_bars, 'hour', guard)
    minute_clv = close_location(*(minute_ohlc[name] for name in PRICE_FIELDS))
    if not np.array_equal(minute_ohlc['close'], minute_bars['close']):
        raise ValueError('Minute source close changed')
    hour_ohlc, grouping = aggregate_hour_ohlc(minute_bars, hour_bars, minute_ohlc, guard)
    result = {}
    for frequency, bars, prices in (
            ('minute', minute_bars, dict(minute_ohlc, clv=minute_clv)),
            ('hour', hour_bars, hour_ohlc)):
        result[frequency] = dict(bars, open=prices['open'], high=prices['high'], low=prices['low'],
                                 clv=prices['clv'])
        validate_cached_source(result[frequency], frequency, guard)
    return result, grouping


def validate_cached_source(arrays, frequency, guard=None):
    np = _numpy()
    old = legacy_volume()
    expected = old.BAR_FIELDS | {'open', 'high', 'low', 'clv'}
    if set(arrays) != expected:
        raise ValueError('Unexpected cached OHLC/volume schema')
    old._validate_bars({name: arrays[name] for name in old.BAR_FIELDS}, frequency, guard)
    for name in ('open', 'high', 'low', 'clv'):
        if (arrays[name].dtype != np.dtype('float64') or arrays[name].ndim != 1 or
                arrays[name].shape != arrays['close'].shape):
            raise ValueError('OHLC/CLV cached dtype or shape changed')
    rebuilt = close_location(*(arrays[name] for name in PRICE_FIELDS))
    if not np.array_equal(rebuilt, arrays['clv']):
        raise ValueError('Cached CLV does not match its exact OHLC formula')


def adapter_identity(source_identity):
    """Bind the full caller-supplied source identity, formula and both sources."""
    # JSON round-trip rejects unsupported identity objects/NaN before writes.
    clean = json.loads(json.dumps(source_identity, sort_keys=True, allow_nan=False))
    require_sha256(LEGACY_PATH, LEGACY_SHA256)
    return {'schema': SCHEMA_VERSION, 'source': clean,
            'adapter_sha256': file_sha256(Path(__file__)),
            'legacy_volume_sha256': LEGACY_SHA256, 'clv_formula': CLV_FORMULA,
            'price_fields': list(PRICE_FIELDS), 'hour_grouping': 'unchanged_original_complete60minute_groups',
            'volume_controls': 'separately qualified old volume cache; not generated or persisted here',
            'source_price_invalidity': 'fail_whole_source_no_label_filter'}


def _cache_directory(path):
    directory = Path(path).resolve()
    code = STAGE_DIR / 'code'
    if (not directory.is_relative_to(STAGE_DIR) or directory == STAGE_DIR or
            directory == code or code in directory.parents):
        raise ValueError('Use a source-cache subdirectory inside this stage, outside code')
    return directory


def write_cache(cache_dir, result, identity, source_metadata, grouping, guard=None):
    """Atomic commit: absent receipt means incomplete/rebuild, never a resume."""
    cache_dir = _cache_directory(cache_dir)
    old = legacy_volume()
    if set(result) != {'minute', 'hour'}:
        raise ValueError('Need precisely minute/hour source artifacts')
    receipt_path = cache_dir / 'OHLC_SOURCE_RECEIPT.json'
    if receipt_path.exists():
        raise ValueError('Refuse to overwrite an already committed source cache')
    cache_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for frequency in ('minute', 'hour'):
        _guard(guard)
        validate_cached_source(result[frequency], frequency, guard)
        path = cache_dir / (frequency + '_ohlc.npz')
        old._atomic_npz(path, result[frequency])
        hashes[frequency] = file_sha256(path, guard)
    receipt = {'status': 'complete', 'identity': identity, 'source_metadata': source_metadata,
               'hour_aggregation': grouping, 'cache_sha256': hashes,
               'target_or_model_loaded': False, 'Q3_values_read': False,
               'point_in_time_availability_established': False,
               'asof_limit': 'Historical final bars; completion boundary assumption, actual receipt/revisions unresolved'}
    _guard(guard)
    old._atomic_json(receipt_path, receipt)
    return receipt


def read_cache(cache_dir, identity, guard=None):
    """Reject wrong source/code, incomplete commits and any changed bytes."""
    np = _numpy()
    cache_dir = _cache_directory(cache_dir)
    receipt_path = cache_dir / 'OHLC_SOURCE_RECEIPT.json'
    if not receipt_path.is_file():
        raise ValueError('No complete source cache receipt; partial artifacts cannot resume')
    receipt_hash = file_sha256(receipt_path, guard)
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('status') != 'complete' or receipt.get('identity') != identity:
        raise ValueError('Source cache has a different identity or incomplete status')
    if set(receipt.get('cache_sha256', {})) != {'minute', 'hour'}:
        raise ValueError('Source cache lacks both frequency hashes')
    result = {}
    for frequency in ('minute', 'hour'):
        path = cache_dir / (frequency + '_ohlc.npz')
        expected = receipt['cache_sha256'][frequency]
        require_sha256(path, expected, guard)
        with np.load(path, allow_pickle=False) as saved:
            if len(saved.files) != len(set(saved.files)):
                raise ValueError('Duplicate cache array members')
            arrays = {name: saved[name].copy() for name in saved.files}
        require_sha256(path, expected, guard)
        validate_cached_source(arrays, frequency, guard)
        result[frequency] = arrays
    require_sha256(receipt_path, receipt_hash, guard)
    result['_receipt'] = receipt
    _guard(guard)
    return result


def load_ohlc(cache_dir, guard):
    """Explicit real-source boundary. No caller receives labels or target masks.

    This explicit function call is not scientific/data admission. The enclosing
    separately frozen protocol must authorize it; no screen calls it implicitly.
    """
    if not callable(guard):
        raise ValueError('A frozen deadline/resource guard callback is required')
    guard()
    old = legacy_volume()
    cache_dir = _cache_directory(cache_dir)
    old_identity, raw, directory = old._identity(guard)
    identity = adapter_identity(old_identity)
    if (cache_dir / 'OHLC_SOURCE_RECEIPT.json').exists():
        return read_cache(cache_dir, identity, guard)
    minute, _ = old._load_original(directory / 'bars_minute.npz', 'minute', guard)
    hour, _ = old._load_original(directory / 'bars_hour.npz', 'hour', guard)
    counts = {key: old.EXPECTED_COUNTS[key] for key in ('all', 'regular', 'outside_regular')}
    prices, metadata = align_csv_ohlc(raw, minute, guard, expected_sha256=old.RAW_SHA,
                                    expected_counts=counts)
    result, grouping = assemble_sources(minute, hour, prices, guard)
    if (grouping['full_hours'] != old.EXPECTED_COUNTS['hour'] or
            grouping['sessions'] != old.EXPECTED_COUNTS['sessions'] or
            grouping['residual_minutes_excluded'] != old.EXPECTED_COUNTS['residual']):
        raise ValueError('Original hour/session/residual counts changed')
    # Bind source bytes again before committing a newly built derivative.
    current_identity, _, _ = old._identity(guard)
    if current_identity != old_identity:
        raise ValueError('Original source identity changed while building fields')
    receipt = write_cache(cache_dir, result, identity, metadata, grouping, guard)
    result['_receipt'] = receipt
    return result

