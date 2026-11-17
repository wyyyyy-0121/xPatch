"""Qualified-source timestamps and original fit clocks only; never AAPL prices."""
from __future__ import annotations
import argparse
import calendar
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import types
import zipfile

import numpy as np

import coverage_core as core

BASE = Path(__file__).resolve().parent
SOURCE = BASE.parent/'20261110_market_context_coverage/download_spy_coverage.py'
SOURCE_SHA = '9cc0d1e2af800d4dfe33ac10db009d483f66f5e0912f2a4ca3f3953863c9b32a'
CUTOFF_NS = calendar.timegm((2026, 7, 1, 4, 0, 0))*1_000_000_000
BAR_MEMBERS = ('start_ns', 'end_ns', 'observed', 'session_date')


def sha(body):
    return hashlib.sha256(body).hexdigest()


def file_sha(path, guard=lambda: None):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            guard(); block = stream.read(65536)
            if not block:
                return h.hexdigest()
            h.update(block)


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    tmp.replace(path)


def lock_table(refs):
    table = {}
    for ref in refs:
        path = str(Path(ref['path']).resolve())
        if path in table or not isinstance(ref['sha256'], str) or len(ref['sha256']) != 64:
            raise ValueError('Duplicate or malformed frozen lock')
        table[path] = ref['sha256']
    return table


def require_locked(path, table, guard=lambda: None):
    key = str(Path(path).resolve())
    if key not in table or file_sha(path, guard) != table[key]:
        raise ValueError('Input missing from frozen locks or identity mismatch')


def frozen_json(path, table, guard=lambda: None):
    guard(); body = Path(path).read_bytes(); key = str(Path(path).resolve())
    if key not in table or sha(body) != table[key]:
        raise ValueError('Parsed JSON bytes differ from frozen identity')
    return json.loads(body)


def read_selected_members(path, names, audit):
    """Zip metadata may be listed; only the named NPY members are decoded."""
    allowed = set(BAR_MEMBERS) if set(names) == set(BAR_MEMBERS) else {'fit'}
    if set(names) != allowed:
        raise ValueError('Member selection outside clock/fit allowlist')
    result = {}
    with zipfile.ZipFile(path) as archive:
        listing = archive.namelist()
        for name in names:
            key = name+'.npy'
            if listing.count(key) != 1:
                raise ValueError('Missing or duplicate selected member')
            with archive.open(key) as stream:
                result[name] = np.lib.format.read_array(stream, allow_pickle=False)
            audit.append({'path': str(Path(path).resolve()), 'member': key})
    return result


def check_bar_clock(bars, expected_count, duration_ns):
    n = len(bars['start_ns'])
    if n != expected_count or any(value.ndim != 1 or len(value) != n for value in bars.values()):
        raise ValueError('Original clock shape mismatch')
    if (bars['start_ns'].dtype != np.dtype('int64') or bars['end_ns'].dtype != np.dtype('int64')
            or bars['observed'].dtype != np.bool_ or bars['session_date'].dtype.kind != 'U'
            or np.any(np.diff(bars['start_ns']) <= 0) or np.any(np.diff(bars['end_ns']) <= 0)
            or np.any(bars['end_ns']-bars['start_ns'] != duration_ns)
            or np.any(bars['end_ns'] > CUTOFF_NS)):
        raise ValueError('Original development clock/dtype invalid')


def check_original_hour_clock(minute, hour):
    a, b = core.original_hour_groups(minute['start_ns'], minute['end_ns'], minute['session_date'])
    for name, expected in [('start_ns', minute['start_ns'][a]),
                           ('end_ns', minute['end_ns'][b-1]),
                           ('session_date', minute['session_date'][a])]:
        if not np.array_equal(hour[name], expected):
            raise ValueError('Minute-derived hour clock differs from original hourly cache')
    return a, b


def map_exact_source(minute_start, source_starts):
    """No nearest, carry-forward, grid compression, or session expansion."""
    if (source_starts.dtype != np.dtype('int64') or source_starts.ndim != 1
            or (np.diff(source_starts) <= 0).any()):
        raise ValueError('Source timestamp order invalid')
    positions = np.searchsorted(minute_start, source_starts)
    in_bounds = positions < len(minute_start)
    positions = positions[in_bounds]; stamps = source_starts[in_bounds]
    matched = positions[minute_start[positions] == stamps]
    available = np.zeros(len(minute_start), dtype=bool)
    available[matched] = True
    return available, len(source_starts)-len(matched)


def qualified_source_starts(index, protocol, guard):
    body = SOURCE.read_bytes()
    if sha(body) != SOURCE_SHA:
        raise ValueError('Source validator identity changed')
    source = types.ModuleType('locked_clock_source'); source.__file__ = str(SOURCE)
    exec(compile(body, str(SOURCE), 'exec'), source.__dict__)
    if (index['kind'] != 'SPY_qualified_canonical_index_v1'
            or index['all_67_ranges_complete'] is not True
            or index['ranges'] != source.RANGES or index['Q3_allowed'] is not False
            or index['source_only_not_RTH_or_context_admission'] is not True
            or index['protocol_sha256'] != protocol['source_protocol_sha256']
            or index['inheritance_manifest_sha256'] != protocol['inheritance_manifest_sha256']):
        raise ValueError('Qualified source index identity invalid')
    pages = index['pages']
    if (not 67 <= len(pages) <= 298
            or any(type(p['range']) is not int or not 0 <= p['range'] < 67 for p in pages)):
        raise ValueError('Canonical page range/count invalid')
    if [(p['range'], p['page']) for p in pages] != sorted((p['range'], p['page']) for p in pages):
        raise ValueError('Canonical pages not in original range/page order')
    stamps = []
    for r in range(67):
        group = [p for p in pages if p['range'] == r]
        if not group:
            raise ValueError('Missing source range')
        previous, token = None, None
        for number, page in enumerate(group):
            guard(); ref = page['canonical_ref']; path = Path(ref['path'])
            if (page['page'] != number or page['request_page_token'] != token
                    or (number and token is None) or not path.is_absolute()
                    or not 0 < ref['bytes'] <= 4*1024*1024):
                raise ValueError('Canonical page ref/token/hash invalid')
            payload = path.read_bytes()
            if len(payload) != ref['bytes'] or sha(payload) != ref['sha256']:
                raise ValueError('Parsed source bytes differ from canonical identity')
            metadata = source.validate_page(payload, r, previous, guard)
            if (metadata['rows'] != page['rows'] or metadata['first_UTC'] != page['first_UTC']
                    or metadata['last_UTC'] != page['last_UTC']
                    or metadata['next_token'] != page['next_page_token']):
                raise ValueError('Canonical page metadata differs')
            # Generic JSON decoding instantiates all SPY fields; only t is retained.
            parsed = json.loads(payload)
            for i, bar in enumerate(parsed['bars']['SPY']):
                if i % 256 == 0:
                    guard()
                stamp = datetime.fromisoformat(bar['t'].replace('Z', '+00:00'))
                stamps.append(calendar.timegm(stamp.utctimetuple())*1_000_000_000)
            previous, token = metadata['last_UTC'], metadata['next_token']
        if token is not None:
            raise ValueError('Source range has unfinished pagination')
    result = np.asarray(stamps, dtype=np.int64)
    if not len(result) or (result >= CUTOFF_NS).any() or (np.diff(result) <= 0).any():
        raise ValueError('Source final timestamp order/cutoff invalid')
    return result


def run(protocol_path):
    body = protocol_path.read_bytes(); protocol = json.loads(body)
    started = time.time()
    if (protocol.get('kind') != 'SPY_original_origin_clock_coverage_v1'
            or protocol.get('execution_admitted') is not True
            or protocol['deadline_epoch']-protocol['started_epoch'] != 180
            or not protocol['started_epoch'] <= started < protocol['deadline_epoch']
            or protocol['worker_sha256'] != file_sha(Path(__file__))
            or protocol['core_sha256'] != file_sha(BASE/'coverage_core.py')
            or protocol.get('Q3_allowed') is not False or protocol.get('fitting_allowed') is not False):
        raise ValueError('Coverage protocol identity/window invalid')
    def guard():
        if time.time() >= protocol['deadline_epoch']:
            raise TimeoutError('Original coverage deadline reached')
    def expired(_sig, _frame):
        raise TimeoutError('Original coverage deadline reached')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, protocol['deadline_epoch']-time.time())
    os.sched_setaffinity(0, {24})
    out = BASE/'run'
    out.mkdir(exist_ok=True)
    if (out/'RESULTS.json').exists():
        raise ValueError('Do not overwrite completed coverage')
    audit = []
    table = lock_table(protocol['locks'])
    for lock in protocol['locks']:
        guard()
        if file_sha(lock['path'], guard) != lock['sha256']:
            raise ValueError('Frozen input identity mismatch')
    index = frozen_json(protocol['source_index_path'], table, guard)
    stamps = qualified_source_starts(index, protocol, guard)
    grids = {}
    for frequency, duration in [('minute', 60_000_000_000), ('hour', 3_600_000_000_000)]:
        ref = protocol['original_bars'][frequency]
        require_locked(ref['path'], table, guard)
        grids[frequency] = read_selected_members(ref['path'], BAR_MEMBERS, audit)
        check_bar_clock(grids[frequency], ref['rows'], duration)
    groups = check_original_hour_clock(grids['minute'], grids['hour'])
    minute_available, outside = map_exact_source(grids['minute']['start_ns'], stamps)
    source_available = {'minute': minute_available,
        'hour': core.hour_source_available(minute_available, *groups)}
    rows = []
    expected_tasks = [('minute_h60_future_log_return', 'minute', 480, 60),
                      ('hourly_h8_future_log_return', 'hour', 96, 8)]
    if len(protocol['tasks']) != 2:
        raise ValueError('Exactly the original two tasks are required')
    for task_ref, expected in zip(protocol['tasks'], expected_tasks):
        guard(); task = frozen_json(task_ref['path'], table, guard)
        name, frequency, lookback, horizon = expected
        if ((task['task'], task['frequency'], task['lookback'], task['horizon']) != expected
                or task_ref['folds'] != [1, 2]):
            raise ValueError('Task/lookback/horizon/fold identity changed')
        for fold in task_ref['folds']:
            guard(); meta = task['folds'][str(fold)]
            require_locked(meta['split_file'], table, guard)
            fit = read_selected_members(meta['split_file'], ('fit',), audit)['fit']
            if (fit.dtype != np.dtype('int64') or len(fit) != meta['counts']['fit']
                    or sha(fit.astype('<i8').tobytes()) != meta['split_sha256']['fit']
                    or np.any(fit < lookback-1)):
                raise ValueError('Original fit identity invalid')
            ends = grids[frequency]['end_ns']
            cut, prefix, tail = core.partition_original_fit(fit, ends, horizon)
            arrays, partitions = {}, {}
            for partition, origins in [('full_fit', fit), ('purged_prefix', prefix), ('unchanged_tail', tail)]:
                available = core.delayed_history_available(source_available[frequency], origins, lookback)
                arrays[partition+'_origins'] = origins
                arrays[partition+'_complete_history'] = available
                partitions[partition] = {'total': len(origins), 'complete': int(available.sum()),
                    'incomplete': int((~available).sum()),
                    'origin_array_sha256': sha(origins.astype('<i8').tobytes()),
                    'mask_sha256': sha(available.tobytes())}
            path = out/(name+'_fold_'+str(fold)+'_clock_masks.npz')
            tmp = path.with_suffix('.partial')
            with tmp.open('wb') as stream:
                np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
            guard(); tmp.replace(path)
            row = {'task': name, 'fold': fold, 'lookback': lookback, 'horizon': horizon,
                   'original_cut': cut, 'purged_count': cut-len(prefix), 'partitions': partitions,
                   'first_tail_decision_ns': int(ends[tail[0]]),
                   'last_prefix_label_end_ns': int(ends[prefix[-1]+horizon]) if len(prefix) else None,
                   'mask_file': str(path), 'mask_file_sha256': file_sha(path, guard)}
            rows.append(row)
            atomic_json(out/'CHECKPOINT.json', {'protocol_sha256': sha(body), 'completed': rows,
                       'deadline_epoch': protocol['deadline_epoch'], 'decoded_AAPL_members': audit})
    guard()
    result = {'status': 'source_clock_coverage_only_complete', 'protocol_sha256': sha(body),
        'started_epoch': started, 'ended_epoch': time.time(), 'deadline_epoch': protocol['deadline_epoch'],
        'seconds': time.time()-started, 'source_rows': len(stamps), 'source_rows_outside_original_RTH': outside,
        'source_observed_minutes': int(minute_available.sum()),
        'source_complete_hours': int(source_available['hour'].sum()),
        'original_hour_clock_exactly_matched': True, 'rows': rows,
        'decoded_AAPL_members': audit, 'AAPL_close_labels_stop_outer_decoded': False,
        'SPY_prices_instantiated_for_schema_validation_only': True,
        'features_models_scores_built': False, 'Q3_read': False, 'GPU': False,
        'predictive_screen_automatically_admitted': False, 'research_goal_achieved': False}
    atomic_json(out/'RESULTS.json', result)
    signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps({'status': result['status'], 'seconds': result['seconds'], 'rows': len(rows)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    run(parser.parse_args().protocol)
