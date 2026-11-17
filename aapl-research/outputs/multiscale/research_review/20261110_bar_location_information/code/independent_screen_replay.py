"""Independent, explicit replay of the frozen historical CLV screen.

No screen_core/screen_worker imports, streaming normal-equation reconstruction,
source loader calls, model training, or stop/outer/Q3 member reads. The only CLI
operation is --run after matching protocol, source qualification, completed main
run, and a separately frozen replay-artifact admission have been verified.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]
OLD = BASE.parent / '20261109_information_audit'
PRIOR = ROOT / 'outputs/multiscale/controlled_followup/20261108_v2'
OUT = BASE / 'screen'
ATOL = 1e-11
RTOL = 1e-9
PENALTY = 0.1
STD_FLOOR = 1e-12
BATCH = 2048
TASKS = (('minute', 60, 480), ('hourly', 8, 96))
BAR_FIELDS = {'start_ns', 'end_ns', 'close', 'observed', 'session_date'}
OHLC_FIELDS = BAR_FIELDS | {'open', 'high', 'low', 'clv'}
VOLUME_FIELDS = BAR_FIELDS | {'volume', 'abnormal', 'valid', 'slot'}
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')
MINUTE_NS = 60_000_000_000
STOP = False


def digest(path, guard=None):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            if guard is not None:
                guard()
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as handle:
            temporary = handle.name
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def _signal(*_):
    global STOP
    STOP = True


class Audit:
    """Numerical tolerances never change science comparisons or thresholds."""
    def __init__(self, np):
        self.np = np
        self.comparisons = 0
        self.maximum_absolute_difference = 0.0
        self.failures = []
        self.scientific_mismatches = []

    def exact(self, label, actual, expected):
        self.comparisons += 1
        if actual != expected:
            self.failures.append(label)
            raise ValueError('Identity/decision mismatch: ' + label)

    def near(self, label, actual, expected):
        np = self.np
        a, b = np.asarray(actual), np.asarray(expected)
        self.comparisons += 1
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            self.failures.append(label)
            raise ValueError('Nonfinite or mismatched numerical arrays: ' + label)
        difference = float(np.max(np.abs(a - b))) if a.size else 0.0
        self.maximum_absolute_difference = max(self.maximum_absolute_difference, difference)
        if not np.allclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=False):
            self.failures.append(label)
            raise ValueError('Frozen replay tolerance failed: ' + label)

    def science(self, label, actual, expected):
        self.comparisons += 1
        if actual != expected:
            self.scientific_mismatches.append({'comparison': label,
                                              'recomputed': actual, 'main': expected})


def check_runtime():
    if not hasattr(os, 'sched_getaffinity') or set(os.sched_getaffinity(0)) != {26}:
        raise RuntimeError('Independent replay must be launched on logical CPU26 only')
    if any(os.environ.get(name) != '1' for name in THREAD_VARIABLES):
        raise RuntimeError('Every numeric thread environment variable must equal 1 before Python starts')
    sys.dont_write_bytecode = True


def verify_admission(guard):
    """Standard-library identities before importing NumPy or reading arrays."""
    protocol_path = BASE / 'MARKET_PROTOCOL.json'
    protocol = read_json(protocol_path)
    identity = digest(protocol_path, guard)
    if protocol.get('GPU') is not False or protocol.get('planned_fits') != 12:
        raise ValueError('Unexpected frozen market-screen design')
    dependencies = protocol.get('dependencies')
    if not isinstance(dependencies, dict) or not dependencies:
        raise ValueError('The market protocol must bind all source/code inputs')
    dependencies = {str(Path(path).resolve()): value for path, value in dependencies.items()}
    if str(Path(__file__).resolve()) not in dependencies:
        raise ValueError('This independent replay source is not a frozen dependency')
    for path, expected in dependencies.items():
        guard()
        if digest(path, guard) != expected:
            raise ValueError('Frozen dependency changed: ' + path)

    def bound_json(path):
        path = Path(path).resolve()
        if str(path) not in dependencies or digest(path, guard) != dependencies[str(path)]:
            raise ValueError('Metadata is not bound by the market protocol: ' + str(path))
        return read_json(path)

    market_admission = read_json(BASE / 'MARKET_ADMISSION.json')
    if market_admission.get('protocol_sha256') != identity:
        raise ValueError('No matching original market admission')
    replay_admission = read_json(OUT / 'REPLAY_ADMISSION.json')
    manifest_path = OUT / 'FROZEN_REPLAY_ARTIFACTS.json'
    if (replay_admission.get('admitted') is not True or
            replay_admission.get('protocol_sha256') != identity or
            replay_admission.get('deadline_epoch') != protocol['deadline_epoch'] or
            replay_admission.get('artifact_manifest_sha256') != digest(manifest_path, guard)):
        raise ValueError('No matching frozen replay-artifact admission')
    manifest = read_json(manifest_path)
    if manifest.get('protocol_sha256') != identity or not isinstance(manifest.get('artifacts'), dict):
        raise ValueError('Invalid replay artifact manifest')
    artifacts = {str(Path(path).resolve()): value for path, value in manifest['artifacts'].items()}
    for path, expected in artifacts.items():
        guard()
        if (Path(path).name in ('INDEPENDENT_REPLAY.json', 'REPLAY_ADMISSION.json',
                               'FROZEN_REPLAY_ARTIFACTS.json') or
                not Path(path).is_relative_to(OUT) or digest(path, guard) != expected):
            raise ValueError('Frozen main-run artifact changed or escaped output directory')

    qualification_path = Path(protocol['source_qualification_file']).resolve()
    if qualification_path != BASE / 'SOURCE_QUALIFICATION.json':
        raise ValueError('Unexpected source qualification path')
    qualification = bound_json(qualification_path)
    source_protocol = BASE / 'SOURCE_PROTOCOL.json'
    source_receipt_path = BASE / 'source_cache/OHLC_SOURCE_RECEIPT.json'
    if (qualification.get('status') != 'qualified_historical_final_bar_OHLC' or
            qualification.get('protocol_sha256') != dependencies.get(str(source_protocol)) or
            qualification.get('receipt_sha256') != dependencies.get(str(source_receipt_path))):
        raise ValueError('Cache completion is not matching historical OHLC source qualification')
    bound_json(source_protocol)
    ohlc_receipt = bound_json(source_receipt_path)
    volume_receipt_path = OLD / 'screen_20261109_v1/cache/VOLUME_SOURCE_RECEIPT.json'
    volume_receipt = bound_json(volume_receipt_path)
    for receipt in (ohlc_receipt, volume_receipt):
        if receipt.get('status') != 'complete' or receipt.get('Q3_values_read') is not False:
            raise ValueError('Source cache receipt is incomplete or has wrong exposure identity')
    return protocol, identity, dependencies, artifacts, ohlc_receipt, volume_receipt


def load_bound_npz(path, hashes, np, guard):
    path = Path(path).resolve()
    expected = hashes.get(str(path))
    if expected is None or digest(path, guard) != expected:
        raise ValueError('Array artifact lacks its frozen identity: ' + str(path))
    with np.load(path, allow_pickle=False) as saved:
        if len(saved.files) != len(set(saved.files)):
            raise ValueError('Duplicate NPZ member')
        arrays = {key: saved[key].copy() for key in saved.files}
    if digest(path, guard) != expected:
        raise ValueError('Array artifact changed during replay read')
    return arrays


def original_module(path, name, dependencies, guard):
    path = Path(path).resolve()
    if dependencies.get(str(path)) != digest(path, guard):
        raise ValueError('Original source module is not a matching dependency')
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError('Cannot import original context source')
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def independent_clv(arrays, np):
    op, hi, lo, cl = (arrays[name] for name in ('open', 'high', 'low', 'close'))
    if any(value.dtype != np.dtype('float64') or value.ndim != 1 or value.shape != cl.shape
           or not np.isfinite(value).all() or (value <= 0).any() for value in (op, hi, lo, cl)):
        raise ValueError('Replay source prices are not valid finite positive vectors')
    if ((hi < lo) | (op < lo) | (op > hi) | (cl < lo) | (cl > hi)).any():
        raise ValueError('Replay OHLC logic failed')
    answer = np.zeros(len(cl), dtype=np.float64)
    use = hi != lo
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        answer[use] = ((cl[use] - lo[use]) - (hi[use] - cl[use])) / (hi[use] - lo[use])
    if not np.isfinite(answer).all() or ((answer < -1) | (answer > 1)).any():
        raise ValueError('Replay CLV is outside its fixed range; no clipping allowed')
    return answer


def reconstruct_sources(dependencies, ohlc_receipt, volume_receipt, np, audit, guard):
    result = {}
    for frequency in ('minute', 'hour'):
        ohlc_path = BASE / 'source_cache' / (frequency + '_ohlc.npz')
        volume_path = OLD / 'screen_20261109_v1/cache' / (frequency + '_volume.npz')
        audit.exact('ohlc_cache_hash_' + frequency, dependencies.get(str(ohlc_path)),
                    ohlc_receipt['cache_sha256'][frequency])
        audit.exact('volume_cache_hash_' + frequency, dependencies.get(str(volume_path)),
                    volume_receipt['cache_sha256'][frequency])
        ohlc = load_bound_npz(ohlc_path, dependencies, np, guard)
        volume = load_bound_npz(volume_path, dependencies, np, guard)
        if set(ohlc) != OHLC_FIELDS or set(volume) != VOLUME_FIELDS:
            raise ValueError('Qualified source-cache schema differs')
        for key in BAR_FIELDS:
            audit.exact(f'{frequency}_common_original_{key}',
                        bool(np.array_equal(ohlc[key], volume[key])), True)
        dates = ohlc['session_date']
        if (dates < '2021-01-04').any() or (dates > '2026-06-30').any():
            raise ValueError('Replay attempted a source outside the admitted pre-Q3 period')
        duration = MINUTE_NS * (1 if frequency == 'minute' else 60)
        if (ohlc['start_ns'].dtype != np.dtype('int64') or
                ohlc['end_ns'].dtype != np.dtype('int64') or
                not ohlc['observed'].all() or
                not np.all(ohlc['end_ns'] - ohlc['start_ns'] == duration) or
                not np.all(np.diff(ohlc['start_ns']) > 0)):
            raise ValueError('Replay original bar clock or observed identity failed')
        clv = independent_clv(ohlc, np)
        audit.exact(f'{frequency}_clv_rebuilt_exact', bool(np.array_equal(clv, ohlc['clv'])), True)
        if (not np.isfinite(volume['volume']).all() or (volume['volume'] < 0).any() or
                not np.array_equal(np.isfinite(volume['abnormal']), volume['valid'])):
            raise ValueError('Replay old-volume validity failed')
        result[frequency] = {'ohlc': ohlc, 'volume': volume, 'clv': clv}

    minute, hour = result['minute']['ohlc'], result['hour']['ohlc']
    boundaries = np.flatnonzero(np.r_[True, minute['session_date'][1:] != minute['session_date'][:-1]])
    stops = np.r_[boundaries[1:], len(minute['close'])]
    groups = []
    residual = 0
    for first, stop in zip(boundaries, stops):
        guard()
        length = int(stop - first)
        expected = minute['start_ns'][first] + MINUTE_NS * np.arange(length, dtype=np.int64)
        if not np.array_equal(minute['start_ns'][first:stop], expected):
            raise ValueError('Original minute session has a gap')
        groups.extend(range(int(first), int(stop) - 59, 60))
        residual += length % 60
    if len(groups) != len(hour['close']):
        raise ValueError('Independent complete-hour count mismatch')
    first = np.asarray(groups, dtype=np.int64)
    last = first + 59
    for key, values in (('start_ns', minute['start_ns'][first]),
                        ('end_ns', minute['end_ns'][last]),
                        ('session_date', minute['session_date'][first]),
                        ('open', minute['open'][first]), ('close', minute['close'][last])):
        audit.exact('hour_aggregation_' + key, bool(np.array_equal(values, hour[key])), True)
    for i, start in enumerate(groups):
        guard()
        audit.exact('hour_high_' + str(i), float(max(minute['high'][start:start+60])), float(hour['high'][i]))
        audit.exact('hour_low_' + str(i), float(min(minute['low'][start:start+60])), float(hour['low'][i]))
    grouping = ohlc_receipt['hour_aggregation']
    audit.exact('hour_group_count', len(groups), grouping['full_hours'])
    audit.exact('hour_session_count', len(boundaries), grouping['sessions'])
    audit.exact('hour_residual_exclusion', residual, grouping['residual_minutes_excluded'])
    return result


def fit_member_only(path, original, dependencies, np, guard):
    path = Path(path).resolve()
    expected = dependencies.get(str(path))
    if expected is None or digest(path, guard) != expected:
        raise ValueError('Original split artifact is not frozen')
    with np.load(path, allow_pickle=False) as saved:
        fit = saved['fit'].copy()  # No stop/outer array or label construction.
    if digest(path, guard) != expected:
        raise ValueError('Original split changed while reading its fit member')
    if (fit.dtype != np.dtype('int64') or fit.ndim != 1 or not len(fit) or
            len(fit) != original['counts']['fit'] or not np.all(np.diff(fit) > 0) or
            hashlib.sha256(fit.astype('<i8').tobytes()).hexdigest() != original['split_sha256']['fit']):
        raise ValueError('Original fit-member identity failed')
    return fit


def partition_origins(fit, end_ns, horizon, abnormal, np):
    """Independently cut before masking, purge endpoint >= original boundary."""
    cut = (4 * len(fit)) // 5
    first = int(fit[cut])
    left = fit[:cut]
    left = left[end_ns[left + horizon] < end_ns[first]]
    right = fit[cut:]
    # Direct H-window finiteness, not the main cumulative invalid-count mask.
    left_mask = np.asarray([np.isfinite(abnormal[o-horizon+1:o+1]).all() for o in left])
    right_mask = np.asarray([np.isfinite(abnormal[o-horizon+1:o+1]).all() for o in right])
    return cut, first, left[left_mask], right[right_mask]


def raw_labels(bars, origins, horizon, np):
    future = origins[:, None] + np.arange(1, horizon + 1, dtype=np.int64)
    if (not bars.observed[origins].all() or not bars.observed[future].all() or
            (bars.frequency == 'minute' and not np.all(
                bars.session_date[origins] == bars.session_date[origins+horizon]))):
        raise ValueError('Independent labels violate the original observed/session contract')
    return np.log(bars.close[future] / bars.close[origins, None])


def feature_batch(windows, bars, context, origins, horizon, abnormal, clv, np):
    history, anchor = windows.arrays(origins)
    old = context.features(history, bars.end_ns[origins], horizon)
    # Direct original past-H mean; no main cumulative-sum implementation.
    u = np.asarray([np.mean(abnormal[o-horizon+1:o+1], dtype=np.float64) for o in origins])
    lag = context.feature_names(horizon, windows.lookback).index('return_lag_' + str(horizon))
    x = np.column_stack((history, anchor, old, u, old[:, lag] * u, clv[origins]))
    if x.shape != (len(origins), windows.lookback + 21) or not np.isfinite(x).all():
        raise ValueError('Independent fixed feature dictionary failed')
    return x


def qualification_from_centered(z, variance, np, guard):
    """Independent direct-matrix reconstruction of the fixed prefix gate."""
    n, columns = z.shape
    if not np.isfinite(variance[-1]) or variance[-1] <= STD_FLOOR ** 2:
        return {'passed': False, 'reason': 'constant_or_nonfinite_CLV',
                'clv_variance': float(variance[-1])}
    base = z[:, :-1]
    gram = (base.T @ base) / n
    gram = (gram + gram.T) * 0.5
    eigen, vectors = np.linalg.eigh(gram)
    maximum, minimum = float(eigen[-1]), float(eigen[0])
    if maximum <= 0 or minimum < -1e-8 * maximum:
        raise ValueError('Independent prefix PSD qualification failed')
    keep = eigen > 1e-10 * maximum
    cross = (base.T @ z[:, -1]) / n
    coefficients = vectors[:, keep] @ ((vectors[:, keep].T @ cross) / eigen[keep])
    rss = denominator = 0.0
    for first in range(0, n, BATCH):
        guard()
        part = z[first:first+BATCH]
        error = part[:, -1] - part[:, :-1] @ coefficients
        rss += float(error @ error)
        denominator += float(part[:, -1] @ part[:, -1])
    if denominator <= 0:
        raise ValueError('Independent CLV denominator is not positive')
    ratio = rss / denominator
    return {'passed': bool(np.isfinite(ratio) and ratio > 1e-8),
            'clv_variance': float(variance[-1]), 'residual_ratio': ratio,
            'retained_rank': int(np.count_nonzero(keep)), 'baseline_columns': columns-1,
            'minimum_B_eigenvalue': minimum, 'maximum_B_eigenvalue': maximum,
            'eigen_relative_cutoff': 1e-10, 'negative_eigen_relative_limit': 1e-8,
            'minimum_residual_ratio': 1e-8}


def centered_models(x, y, columns, np, guard):
    """Center the one materialized N-by-p matrix in place; no main stats."""
    n = len(x)
    mx = x.mean(axis=0)
    my = y.mean(axis=0)
    x -= mx
    variance = np.einsum('ij,ij->j', x, x) / n
    scale = np.sqrt(variance)
    scale = np.where(scale > STD_FLOOR, scale, 1.0)
    x /= scale
    y -= my
    qualification = qualification_from_centered(x, variance, np, guard)
    models = {}
    for arm, p in columns.items():
        guard()
        z = x[:, :p]
        matrix = (z.T @ z) / n + PENALTY * np.eye(p)
        right = (z.T @ y) / n
        coefficient = np.linalg.solve(matrix, right)
        norm = np.linalg.norm(matrix, np.inf) * np.linalg.norm(coefficient, np.inf) + np.linalg.norm(right, np.inf)
        stationarity = float(np.linalg.norm(matrix @ coefficient - right, np.inf)) / max(float(norm), np.finfo(float).tiny)
        if not np.isfinite(coefficient).all() or not np.isfinite(stationarity) or stationarity > 1e-10:
            raise ValueError('Independent direct Ridge stationarity qualification failed')
        models[arm] = {'coef': coefficient, 'x_mean': mx[:p].copy(),
                       'x_std': scale[:p].copy(), 'y_mean': my.copy()}
    return models, qualification, my


def decisions(metrics):
    c = metrics['C']
    answer = {}
    for name in ('A', 'B', 'zero', 'prefix_mean'):
        answer['C_endpoint_strictly_below_' + name] = bool(c['endpoint_mse'] < metrics[name]['endpoint_mse'])
    for name in ('A', 'B'):
        answer['C_path_not_above_' + name] = bool(c['path_mse'] <= metrics[name]['path_mse'])
    return answer


def replay(guard, report):
    check_runtime()
    protocol, identity, dependencies, artifacts, ohlc_receipt, volume_receipt = verify_admission(guard)
    report['protocol_sha256'] = identity
    if time.time() >= protocol['deadline_epoch']:
        raise InterruptedError('Original market/replay absolute deadline expired')
    import numpy as np
    audit = Audit(np)
    main_path = OUT / 'RESULTS.json'
    if artifacts.get(str(main_path)) != digest(main_path, guard):
        raise ValueError('Completed screen result is not in the replay freeze')
    main = read_json(main_path)
    audit.exact('main_status', main.get('status'), 'completed')
    audit.exact('main_protocol', main.get('protocol_sha256'), identity)
    audit.exact('main_fitted_models', main.get('fitted_models'), 12)
    sources = reconstruct_sources(dependencies, ohlc_receipt, volume_receipt, np, audit, guard)
    context_path = ROOT / 'outputs/multiscale/mechanism_repair/20261109_v3b/code/predictor.py'
    context = original_module(context_path, '_clv_replay_original_context', dependencies, guard)
    reference = PRIOR / 'code/reference_source'
    for relative in ('multiscale/features.py', 'multiscale/bars.py'):
        path = reference / relative
        if dependencies.get(str(path)) != digest(path, guard):
            raise ValueError('Original close/target source is not bound')
    if any(name in sys.modules for name in ('multiscale.features', 'multiscale.bars')):
        raise ValueError('Launch a fresh replay process; original modules were already loaded')
    sys.path.insert(0, str(reference))
    from multiscale.features import CloseWindows
    from multiscale.bars import Bars
    for name, relative in (('multiscale.features', 'multiscale/features.py'),
                           ('multiscale.bars', 'multiscale/bars.py')):
        if Path(sys.modules[name].__file__).resolve() != reference / relative:
            raise ValueError('Unexpected original-module import path')
    data_protocol_path = PRIOR / 'protocol/DATA.json'
    if dependencies.get(str(data_protocol_path)) != digest(data_protocol_path, guard):
        raise ValueError('Original data metadata is not bound')
    data_protocol = read_json(data_protocol_path)
    admission_path = Path(data_protocol['admission']) / 'admission.json'
    if dependencies.get(str(admission_path)) != digest(admission_path, guard):
        raise ValueError('Original bar admission is not bound')
    admission = read_json(admission_path)
    rows = []
    for profile, horizon, lookback in TASKS:
        frequency = 'minute' if profile == 'minute' else 'hour'
        volume = sources[frequency]['volume']
        bars = Bars(frequency, **{key: volume[key] for key in BAR_FIELDS})
        windows = CloseWindows(bars, lookback, admission['split_actions'])
        task = f'{profile}_h{horizon}_future_log_return'
        task_path = PRIOR / 'cache' / task / 'TASK.json'
        if dependencies.get(str(task_path)) != digest(task_path, guard):
            raise ValueError('Original task metadata is not bound')
        metadata = read_json(task_path)
        for fold in (1, 2):
            guard()
            folder = OUT / task / f'fold_{fold}'
            for name in ('PARTITION.json', 'FEATURE_QUALIFICATION.json', 'RESULT.json', 'A.npz', 'B.npz', 'C.npz'):
                path = folder / name
                if artifacts.get(str(path)) != digest(path, guard):
                    raise ValueError('State artifact missing from frozen replay manifest: ' + str(path))
            original = metadata['folds'][str(fold)]
            fit = fit_member_only(original['split_file'], original, dependencies, np, guard)
            if fit[0] < lookback-1 or fit[-1]+horizon >= len(bars.close):
                raise ValueError('Original fit origins do not have their complete context/labels')
            cut, first, prefix, tail = partition_origins(fit, bars.end_ns, horizon, volume['abnormal'], np)
            if len(prefix) < 100 or len(tail) < 20:
                raise ValueError('Independent reconstruction lacks the fixed common origins')
            columns = {'A': lookback+18, 'B': lookback+20, 'C': lookback+21}
            partition = read_json(folder / 'PARTITION.json')
            reconstructed = {'protocol_sha256': identity, 'original_fit_count': len(fit),
                'boundary_before_mask': cut, 'first_tail_origin_before_mask': first,
                'prefix_count': len(prefix), 'tail_count': len(tail),
                'prefix_sha256': hashlib.sha256(prefix.astype('<i8').tobytes()).hexdigest(),
                'tail_sha256': hashlib.sha256(tail.astype('<i8').tobytes()).hexdigest(),
                'prefix_last_label_end_ns': int(bars.end_ns[prefix[-1]+horizon]),
                'original_tail_first_decision_ns': int(bars.end_ns[first]),
                'all_arms_share_origins': True, 'columns': columns}
            audit.exact(task + f'_fold{fold}_partition', reconstructed, partition)
            x = np.empty((len(prefix), columns['C']), dtype=np.float64)
            y = np.empty((len(prefix), horizon), dtype=np.float64)
            for start in range(0, len(prefix), BATCH):
                guard()
                origins = prefix[start:start+BATCH]
                x[start:start+len(origins)] = feature_batch(windows, bars, context, origins, horizon,
                    volume['abnormal'], sources[frequency]['clv'], np)
                y[start:start+len(origins)] = raw_labels(bars, origins, horizon, np)
            models, qualification, prefix_mean = centered_models(x, y, columns, np, guard)
            del x, y
            gc.collect()
            saved_qualification = read_json(folder / 'FEATURE_QUALIFICATION.json')
            audit.exact(task + f'_fold{fold}_qualification_protocol', saved_qualification['protocol_sha256'], identity)
            for key, value in qualification.items():
                if isinstance(value, (bool, int, str)):
                    audit.exact(task + f'_fold{fold}_qualification_' + key, value, saved_qualification[key])
                else:
                    audit.near(task + f'_fold{fold}_qualification_' + key, value, saved_qualification[key])
            if not qualification['passed']:
                raise ValueError('A completed main run requires all four independent prefix gates')
            for arm, model in models.items():
                saved = load_bound_npz(folder / (arm + '.npz'), artifacts, np, guard)
                if set(saved) != {'coef', 'x_mean', 'x_std', 'y_mean', 'protocol_sha256'}:
                    raise ValueError('Unexpected saved Ridge model schema')
                audit.exact(task + f'_fold{fold}_{arm}_protocol', str(saved['protocol_sha256']), identity)
                for key, values in model.items():
                    audit.near(task + f'_fold{fold}_{arm}_' + key, values, saved[key])
            totals = {name: {'endpoint_sse': 0.0, 'path_sse': 0.0}
                      for name in ('A', 'B', 'C', 'zero', 'prefix_mean')}
            for start in range(0, len(tail), BATCH):
                guard()
                origins = tail[start:start+BATCH]
                features = feature_batch(windows, bars, context, origins, horizon,
                    volume['abnormal'], sources[frequency]['clv'], np)
                truth = raw_labels(bars, origins, horizon, np)
                for arm in totals:
                    if arm in models:
                        model = models[arm]
                        width = len(model['x_mean'])
                        prediction = ((features[:, :width] - model['x_mean']) / model['x_std']) @ model['coef'] + model['y_mean']
                    else:
                        prediction = 0.0 if arm == 'zero' else prefix_mean
                    error = prediction - truth
                    totals[arm]['endpoint_sse'] += float(np.dot(error[:, -1], error[:, -1]))
                    totals[arm]['path_sse'] += float(np.einsum('ij,ij->', error, error))
            metrics = {arm: dict(sums, endpoint_mse=sums['endpoint_sse']/len(tail),
                                 path_mse=sums['path_sse']/(len(tail)*horizon))
                       for arm, sums in totals.items()}
            original_result = read_json(folder / 'RESULT.json')
            audit.exact(task + f'_fold{fold}_result_protocol', original_result['protocol_sha256'], identity)
            audit.exact(task + f'_fold{fold}_tail_count', original_result['tail_origins'], len(tail))
            for arm, values in metrics.items():
                for key, value in values.items():
                    audit.near(task + f'_fold{fold}_{arm}_' + key, value, original_result['metrics'][arm][key])
            checks = decisions(metrics)
            audit.science(task + f'_fold{fold}_scientific_checks', checks, original_result['checks'])
            passed = all(checks.values())
            audit.science(task + f'_fold{fold}_scientific_passed', passed, original_result['passed'])
            main_rows = [row for row in main['rows'] if row['task'] == task and row['fold'] == fold]
            if len(main_rows) != 1 or main_rows[0] != original_result:
                raise ValueError('Aggregate main result does not preserve exactly one copy of each state')
            rows.append({'task': task, 'fold': fold, 'tail_origins': len(tail),
                         'metrics': metrics, 'checks': checks, 'passed': passed,
                         'prefix_sha256': reconstructed['prefix_sha256'],
                         'tail_sha256': reconstructed['tail_sha256']})
            report['states_replayed'] = len(rows)
            del models, saved, features, truth, error, prediction, prefix, tail, fit
            gc.collect()
    audit.exact('exact_four_main_rows', len(main['rows']), 4)
    task_gates = {}
    for profile, horizon, _ in TASKS:
        task = f'{profile}_h{horizon}_future_log_return'
        states = sorted((row for row in rows if row['task'] == task), key=lambda row: row['fold'])
        audit.exact(task + '_exact_folds', [row['fold'] for row in states], [1, 2])
        # Ratio of equal-fold MSEs, not mean percentage gains or pooled origin SSE.
        numerator = sum(row['metrics']['C']['endpoint_mse'] for row in states)
        denominator = sum(row['metrics']['zero']['endpoint_mse'] for row in states)
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError('Nonpositive original-task zero-risk denominator')
        gain = 1.0 - numerator / denominator
        task_gates[task] = {'equal_fold_MSE_gain_over_zero': gain,
                            'at_least_one_percent': bool(gain >= 0.01)}
        audit.near(task + '_equal_fold_ratio_gain', gain, main['task_gates'][task]['equal_fold_MSE_gain_over_zero'])
        audit.science(task + '_one_percent_gate', gain >= 0.01, main['task_gates'][task]['at_least_one_percent'])
    continued = all(row['passed'] for row in rows) and all(row['at_least_one_percent'] for row in task_gates.values())
    audit.science('scientific_continuation_gate', continued, main['continued'])
    audit.science('passed_state_count', sum(row['passed'] for row in rows), main['passed_states'])
    # Check bound metadata and main outputs again before accepting the replay.
    frozen_files = dict(dependencies)
    frozen_files.update(artifacts)
    for path, expected in frozen_files.items():
        guard()
        if digest(path, guard) != expected:
            raise ValueError('A frozen input/output changed during replay')
    matched = not audit.scientific_mismatches
    report.update(status='completed' if matched else 'scientific_decision_mismatch',
        numerical_qualification_passed=True, replay_accepted=matched,
        scientific_decisions_match_main=matched, scientific_continuation_gate_passed=continued,
        scientific_decision_mismatches=audit.scientific_mismatches,
        comparison_count=audit.comparisons, maximum_absolute_difference=audit.maximum_absolute_difference,
        rows=rows, task_gates=task_gates, fitted_main_models_reconstructed=12,
        exposed_historical_development_only=True, independent_market_confirmation=False,
        predictive_improvement_goal_achieved=False, source_point_in_time_established=False,
        market_source_loader_called=False, source_csv_read=False,
        split_members_read=['fit'], stop_outer_Q3_values_read=False, GPU=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args()
    started = time.monotonic()
    report = {'status': 'starting', 'numerical_qualification_passed': False,
              'scientific_decisions_match_main': False, 'states_replayed': 0,
              'rtol': RTOL, 'atol': ATOL, 'same_original_deadline': True,
              'implementation': 'direct centered matrix Ridge; independent mask, OHLC/CLV, metrics and gates'}
    # Only metadata is read before the guarded replay. No deadline reset occurs.
    protocol = read_json(BASE / 'MARKET_PROTOCOL.json')
    deadline = protocol['deadline_epoch']
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, _signal)

    def guard():
        if STOP or time.time() >= deadline:
            raise InterruptedError('Frozen original replay deadline/stop signal; no continuation authorized here')

    try:
        replay(guard, report)
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error, InterruptedError) else 'failed',
                      error_type=type(error).__name__, reason=str(error))
        report['elapsed_seconds'] = time.monotonic() - started
        atomic_json(OUT / 'INDEPENDENT_REPLAY.json', report)
        raise
    report['elapsed_seconds'] = time.monotonic() - started
    atomic_json(OUT / 'INDEPENDENT_REPLAY.json', report)
    if report.get('replay_accepted') is not True:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
