"""Historical routing support only: four bar clock members and original fit origins."""
from __future__ import annotations
import argparse
import calendar
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import signal
import time
import zipfile
np = None

BASE = Path(__file__).resolve().parent
BAR_MEMBERS = ('start_ns', 'end_ns', 'observed', 'session_date')
CUTOFF = calendar.timegm((2026, 7, 1, 4, 0, 0))*1_000_000_000


def numeric_runtime():
    global np
    required = {'CUDA_VISIBLE_DEVICES': '-1', 'OMP_NUM_THREADS': '1',
                'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}
    if any(os.environ.get(key) != value for key, value in required.items()):
        raise ValueError('thread_or_device_environment_not_frozen')
    os.sched_setaffinity(0, {24})
    resource.setrlimit(resource.RLIMIT_AS, (2*1024**3, 2*1024**3))
    import numpy as numeric
    np = numeric


def sha(body):
    return hashlib.sha256(body).hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    tmp.replace(path)


def route_metadata(session, origins, length, horizon):
    """Window includes origin. Boundary at window first slot is explicitly ignored."""
    if (session.ndim != 1 or session.dtype.kind != 'U' or not len(session)
            or (session[1:] < session[:-1]).any()
            or origins.dtype != np.dtype('int64') or origins.ndim != 1
            or type(length) is not int or length < 1 or type(horizon) is not int or horizon < 1
            or (origins < length-1).any() or (origins+horizon >= len(session)).any()
            or (np.diff(origins) <= 0).any()):
        raise ValueError('invalid_clock_routing_inputs')
    changed = np.r_[False, session[1:] != session[:-1]]
    cumulative = np.r_[0, np.cumsum(changed, dtype=np.int64)]
    last_start = np.maximum.accumulate(np.where(changed, np.arange(len(session)), 0))
    start = origins-length+1
    return {
        'history_boundary_count': cumulative[origins+1]-cumulative[start+1],
        'last_session_suffix_len': np.minimum(length, origins-last_start[origins]+1),
        'future_boundary_count': cumulative[origins+horizon+1]-cumulative[origins+1],
    }


def synthetic():
    numeric_runtime()
    cases = 0
    for pattern in range(128):
        groups = np.r_[0, np.cumsum([(pattern >> j) & 1 for j in range(7)])]
        session = np.asarray(['2020-01-'+str(int(v)+1).zfill(2) for v in groups])
        for length in range(1, 6):
            for horizon in (1, 2):
                origins = np.arange(length-1, len(session)-horizon, dtype=np.int64)
                got = route_metadata(session, origins, length, horizon)
                for q, origin in enumerate(origins):
                    dates = session[origin-length+1:origin+1]
                    flags = np.r_[False, dates[1:] != dates[:-1]]
                    positions = np.flatnonzero(flags)
                    suffix = length-int(positions[-1]) if len(positions) else length
                    future = int(np.count_nonzero(session[origin+1:origin+horizon+1] != session[origin:origin+horizon]))
                    assert got['history_boundary_count'][q] == flags.sum()
                    assert got['last_session_suffix_len'][q] == suffix
                    assert got['future_boundary_count'][q] == future
                    cases += 1
    return {'status': 'passed', 'exact_window_oracle_cases': cases,
            'market_arrays_read': False, 'GPU': False}


def run(path):
    protocol_bytes = path.read_bytes(); p = json.loads(protocol_bytes); started = time.time()
    if (not all(type(p[k]) in (int, float) and math.isfinite(p[k]) for k in ('started_epoch', 'deadline_epoch'))
            or p['kind'] != 'matched_two_state_original_fit_clock_support_v1'
            or p['deadline_epoch']-p['started_epoch'] != 180
            or not p['started_epoch'] <= started < p['deadline_epoch']
            or p['worker_sha256'] != sha(Path(__file__).read_bytes())
            or p['market_prices_labels_allowed'] is not False or p['Q3_allowed'] is not False
            or p['model_fitting_allowed'] is not False):
        raise ValueError('invalid_protocol_scope_or_identity')
    def guard():
        if time.time() >= p['deadline_epoch']:
            raise TimeoutError('clock_support_deadline')
    def expire(_signal, _frame):
        raise TimeoutError('clock_support_deadline')
    for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, expire)
    signal.setitimer(signal.ITIMER_REAL, p['deadline_epoch']-time.time())
    numeric_runtime()
    table = {r['path']: r['sha256'] for r in p['locks']}
    if len(table) != len(p['locks']):
        raise ValueError('duplicate_lock')
    audit = []
    def body(ref):
        guard(); filename = str(Path(ref).resolve())
        if filename not in table:
            raise ValueError('input_path_not_frozen')
        raw = Path(filename).read_bytes()
        if sha(raw) != table[filename]:
            raise ValueError('unfrozen_or_changed_input')
        return raw
    def selected(ref, members):
        if members not in (BAR_MEMBERS, ('fit',)):
            raise ValueError('clock_fit_allowlist_only')
        result = {}
        with zipfile.ZipFile(io.BytesIO(body(ref))) as archive:
            names = archive.namelist()
            for name in members:
                guard(); member = name+'.npy'
                if names.count(member) != 1:
                    raise ValueError('ambiguous_npz_member')
                with archive.open(member) as stream:
                    result[name] = np.lib.format.read_array(stream, allow_pickle=False)
                audit.append({'path': str(Path(ref).resolve()), 'member': member})
        return result
    for lock in p['locks']:
        body(lock['path'])
    out = BASE/'run'
    out.mkdir(exist_ok=False)
    rows = []
    for frequency, length, horizon, task_name in (
            ('minute', 480, 60, 'minute_h60_future_log_return'),
            ('hour', 96, 8, 'hourly_h8_future_log_return')):
        bars = selected(p['original_bars'][frequency]['path'], BAR_MEMBERS)
        starts, ends, observed, session = (bars[name] for name in BAR_MEMBERS)
        n = p['original_bars'][frequency]['rows']
        duration = 60_000_000_000 if frequency == 'minute' else 3_600_000_000_000
        if (any(a.ndim != 1 or len(a) != n for a in bars.values())
                or starts.dtype != np.dtype('int64') or ends.dtype != np.dtype('int64')
                or observed.dtype != np.bool_ or session.dtype.kind != 'U'
                or (np.diff(starts) <= 0).any() or (np.diff(ends) <= 0).any()
                or (ends-starts != duration).any() or (ends > CUTOFF).any()
                or (session[1:] < session[:-1]).any()):
            raise ValueError('original_development_clock_invalid')
        task = json.loads(body(p['tasks'][frequency]))
        if ((task['task'], task['frequency'], task['lookback'], task['horizon'])
                != (task_name, frequency, length, horizon)
                or sorted(task['folds']) != ['1','2','3','4','5']):
            raise ValueError('original_primary_or_folds_changed')
        missing = np.r_[0, np.cumsum(~observed, dtype=np.int64)]
        for fold in range(1, 6):
            guard(); meta = task['folds'][str(fold)]
            fit = selected(meta['split_file'], ('fit',))['fit']
            if (fit.dtype != np.dtype('int64') or fit.ndim != 1
                    or len(fit) != meta['counts']['fit'] or (np.diff(fit) <= 0).any()
                    or sha(fit.astype('<i8').tobytes()) != meta['split_sha256']['fit']):
                raise ValueError('original_fit_identity_changed')
            routing = route_metadata(session, fit, length, horizon)
            history_ok = missing[fit+1] == missing[fit-length+1]
            target_ok = missing[fit+horizon+1] == missing[fit+1]
            if not history_ok.all() or not target_ok.all():
                raise ValueError('original_fit_has_unobserved_history_or_target_clock')
            if frequency == 'minute' and routing['future_boundary_count'].any():
                raise ValueError('original_minute_target_crosses_session')
            cut = 4*len(fit)//5
            tail_mask = np.arange(len(fit)) >= cut
            prefix_mask = (np.arange(len(fit)) < cut) & (ends[fit+horizon] < ends[fit[cut]])
            if not prefix_mask.any() or not tail_mask.any():
                raise ValueError('empty_original_fit_tail_partition')
            saved = {'fit_origins': fit, 'prefix_mask': prefix_mask, 'tail_mask': tail_mask, **routing}
            dst = out/(task_name+'_fold_'+str(fold)+'_support.npz')
            tmp = dst.with_suffix('.tmp')
            with tmp.open('wb') as stream:
                np.savez_compressed(stream, **saved); stream.flush(); os.fsync(stream.fileno())
            tmp.replace(dst)
            parts = {}
            for name, mask in [('full_fit', np.ones(len(fit), bool)), ('purged_prefix', prefix_mask), ('unchanged_tail', tail_mask)]:
                origins = fit[mask]
                parts[name] = {'origins': len(origins), 'origin_sha256': sha(origins.astype('<i8').tobytes())}
                for key, values in routing.items():
                    unique, counts = np.unique(values[mask], return_counts=True)
                    parts[name][key+'_histogram'] = {str(int(v)): int(c) for v,c in zip(unique, counts)}
            row = {'task': task_name, 'frequency': frequency, 'fold': fold, 'L': length, 'H': horizon,
                   'cut': cut, 'purged': int(cut-prefix_mask.sum()), 'all_original_fit_origins_retained': True,
                   'all_history_and_future_clock_observed': True, 'first_slot_boundary_is_false': True,
                   'last_prefix_label_end_ns': int(ends[fit[prefix_mask][-1]+horizon]),
                   'first_tail_decision_end_ns': int(ends[fit[tail_mask][0]]), 'partitions': parts,
                   'support_file': str(dst), 'support_file_sha256': sha(dst.read_bytes())}
            rows.append(row)
            atomic_json(out/'CHECKPOINT.json', {'protocol_sha256': sha(protocol_bytes), 'rows': rows,
                'deadline_epoch': p['deadline_epoch'], 'decoded_npz_members': audit})
    guard()
    result = {'status': 'original_five_fold_fit_clock_support_complete', 'protocol_sha256': sha(protocol_bytes),
              'started_epoch': started, 'ended_epoch': time.time(), 'deadline_epoch': p['deadline_epoch'],
              'seconds': time.time()-started, 'CPU_affinity': sorted(os.sched_getaffinity(0)),
              'rows': rows, 'decoded_npz_members': audit, 'model_fits': 0,
              'RLIMIT_AS_bytes': resource.getrlimit(resource.RLIMIT_AS)[0],
              'maximum_RSS_bytes_ru_maxrss': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
              'whole_npz_compressed_file_bytes_read_for_identity': True,
              'price_or_label_members_decoded': False, 'old_stop_outer_members_decoded': False,
              'Q3_read': False, 'GPU': False, 'market_training_automatically_admitted': False,
              'research_goal_achieved': False}
    atomic_json(out/'RESULTS.json', result)
    return {'status': result['status'], 'states': len(rows), 'seconds': result['seconds']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--protocol', type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(synthetic() if args.synthetic else run(args.protocol)))
    except BaseException as exc:
        if args.protocol:
            atomic_json(BASE/('FAILED_'+str(os.getpid())+'.json'), {'error_type': type(exc).__name__,
                'message': str(exc), 'ended_epoch': time.time(), 'research_goal_achieved': False})
        raise
