"""Independent all-origin clock-only support verification; never imports the worker."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import signal
import tempfile
import time
import zipfile

BASE = Path(__file__).resolve().parent
ROOT = Path('/home/ayaka/wslcode/aapl-research')
AS_LIMIT = 2*1024**3
CONTRACT_SHA = '697b1e125b2643373011f9cbfd304d2449e411043b8632cacde7f63957ae95d7'
CUTOFF_NS = 1782878400000000000  # 2026-07-01 04:00:00 UTC
CLOCK_NAMES = ('start_ns', 'end_ns', 'observed', 'session_date')
ROUTING_NAMES = ('history_boundary_count', 'last_session_suffix_len', 'future_boundary_count')
SUPPORT_NAMES = ('fit_origins', 'prefix_mask', 'tail_mask') + ROUTING_NAMES
ORIGINAL_BARS = {
    'minute': 'c81d57cd1e190456e1fdbf59d7493c915704551fa2e6c5870b5bd66f4cd94332',
    'hour': 'c41851d735d179c3243dd3356204d17845b55289c56c59f9e170c0b5f432881c',
}
ORIGINAL_TASKS = {
    'minute': '597d5a0d94d82611392882c3c4128cda0ba3acc37243034ed1d9b64d4c8d50a7',
    'hour': '791101167b41ef39b4f0ab09050521c1d436f6b7ee87734827ddf5607eb28b9e',
}
PARTITION_SOURCE = ROOT/'outputs/multiscale/research_review/20261110_market_context_origin_coverage/coverage_core.py'
PARTITION_SHA = '44e331dc4d94cf0da16b1d9225e83450066c9bb31f6446896457197316fff5b6'


def digest(body):
    return hashlib.sha256(body).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(65536), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atomic_new(path, value):
    """Expose a complete new receipt without replacing an existing artifact."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as stream:
            temporary = stream.name
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
        temporary = None
        fd = os.open(path.parent, os.O_DIRECTORY | os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temporary is not None:
            os.unlink(temporary)


def run(protocol_path):
    began = time.time()
    require(protocol_path.is_absolute(), 'Absolute protocol path required')
    protocol_bytes = protocol_path.read_bytes()
    p = json.loads(protocol_bytes)
    identity = digest(protocol_bytes)
    start, end = p['started_epoch'], p['deadline_epoch']
    require(type(start) in (int, float) and type(end) in (int, float)
            and math.isfinite(start) and math.isfinite(end) and end-start == 180
            and start <= began < end-1, 'Original finite180s window required')
    require(p['kind'] == 'matched_two_state_original_fit_clock_support_v1'
            and p['market_prices_labels_allowed'] is False
            and p['model_fitting_allowed'] is False and p['Q3_allowed'] is False
            and p['GPU'] is False, 'Clock-only scope required')
    require(os.sched_getaffinity(0) == {26}, 'Independent reviewer must use CPU26 only')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        require(os.environ.get(key) == '1', 'Thread environment must precede NumPy import')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '-1', 'GPU visibility must be disabled')
    require(p['contract_sha256'] == CONTRACT_SHA
            and file_hash(BASE/'CLOCK_SUPPORT_CONTRACT.json') == CONTRACT_SHA,
            'Original clock-support contract changed')
    require(p['reviewer_sha256'] == file_hash(Path(__file__)), 'Independent reviewer hash changed')
    require(p['worker_sha256'] == file_hash(BASE/'session_clock_support.py'), 'Frozen worker changed')
    require(p['supervisor_sha256'] == file_hash(BASE/'run_clock_supervisor.py'), 'Frozen supervisor changed')
    require(file_hash(PARTITION_SOURCE) == PARTITION_SHA, 'Original80/20 strict purge source changed')
    resource.setrlimit(resource.RLIMIT_AS, (AS_LIMIT, AS_LIMIT))
    output = BASE/'run'/'INDEPENDENT_CLOCK_REVIEW.json'
    require(not output.exists(), 'Independent receipt already exists')

    def guard():
        require(time.time() < end-1, 'Original clock-support deadline reached')
        require(os.sched_getaffinity(0) == {26}, 'Independent CPU affinity drift')

    def expire(*_):
        raise TimeoutError('Original clock-support window or external termination')

    for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, expire)
    signal.setitimer(signal.ITIMER_REAL, max(.001, end-1-time.time()))
    try:
        import numpy as np
        guard()
        locks = {}
        for row in p['locks']:
            path = Path(row['path'])
            require(path.is_absolute() and str(path.resolve()) == row['path']
                    and row['path'] not in locks, 'Unique canonical absolute locks required')
            locks[row['path']] = row['sha256']
        require(set(p['original_bars']) == set(ORIGINAL_BARS)
                and set(p['tasks']) == set(ORIGINAL_TASKS), 'Exactly two original tasks required')
        input_audit = []
        derived_audit = []
        original_hashes = {}
        derived_hashes = {}

        def locked_bytes(ref):
            guard()
            path = str(Path(ref).resolve())
            require(path in locks, 'Reject unlocked path before reading')
            raw = Path(path).read_bytes()
            require(digest(raw) == locks[path], 'Frozen input hash changed')
            original_hashes[path] = locks[path]
            return raw

        def selected(raw, path, names, audit, derived=False):
            require(names in (CLOCK_NAMES, ('fit',), SUPPORT_NAMES), 'Decode allowlist violation')
            result = {}
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                entries = archive.namelist()
                if derived:
                    require(set(entries) == {name+'.npy' for name in SUPPORT_NAMES}
                            and len(entries) == len(SUPPORT_NAMES), 'Unexpected derived support member')
                for name in names:
                    guard()
                    require(entries.count(name+'.npy') == 1, 'Ambiguous NPZ member')
                    with archive.open(name+'.npy') as stream:
                        value = np.lib.format.read_array(stream, allow_pickle=False)
                    require(not value.dtype.hasobject, 'Object arrays are not admitted')
                    result[name] = value
                    audit.append({'path': str(Path(path).resolve()), 'member': name+'.npy'})
            return result

        for path in locks:
            locked_bytes(path)
        result_path = BASE/'run'/'RESULTS.json'
        checkpoint_path = BASE/'run'/'CHECKPOINT.json'
        result_raw, checkpoint_raw = result_path.read_bytes(), checkpoint_path.read_bytes()
        result, checkpoint = json.loads(result_raw), json.loads(checkpoint_raw)
        require(result['status'] == 'original_five_fold_fit_clock_support_complete'
                and result['protocol_sha256'] == checkpoint['protocol_sha256'] == identity
                and result['deadline_epoch'] == checkpoint['deadline_epoch'] == end
                and start <= result['started_epoch'] <= result['ended_epoch'] < end
                and result['CPU_affinity'] == [24]
                and result['model_fits'] == 0
                and result['RLIMIT_AS_bytes'] == AS_LIMIT
                and result['whole_npz_compressed_file_bytes_read_for_identity'] is True
                and result['price_or_label_members_decoded'] is False
                and result['old_stop_outer_members_decoded'] is False
                and result['Q3_read'] is False and result['GPU'] is False
                and result['market_training_automatically_admitted'] is False
                and result['research_goal_achieved'] is False,
                'Main clock support receipt scope/time mismatch')
        require(result['rows'] == checkpoint['rows'] and len(result['rows']) == 10,
                'Final checkpoint must contain all10 exact states')
        reviewed = []
        total_origins = total_scalar_metadata = 0
        expected_main_audit = []
        state_index = 0
        for frequency, length, horizon, task_name in (
                ('minute', 480, 60, 'minute_h60_future_log_return'),
                ('hour', 96, 8, 'hourly_h8_future_log_return')):
            guard()
            bar_info = p['original_bars'][frequency]
            bar_path = str(Path(bar_info['path']).resolve())
            require(locks.get(bar_path) == bar_info['sha256'] == ORIGINAL_BARS[frequency],
                    'Archive is not the original pre-July bar identity')
            task_path = str(Path(p['tasks'][frequency]).resolve())
            require(locks.get(task_path) == ORIGINAL_TASKS[frequency], 'Original task identity changed')
            bars = selected(locked_bytes(bar_path), bar_path, CLOCK_NAMES, input_audit)
            expected_main_audit.extend({'path': bar_path, 'member': name+'.npy'} for name in CLOCK_NAMES)
            starts, ends, observed, dates = (bars[name] for name in CLOCK_NAMES)
            count = bar_info['rows']
            duration = 60_000_000_000 if frequency == 'minute' else 3_600_000_000_000
            require(type(count) is int and count > 0
                    and all(value.ndim == 1 and value.size == count for value in bars.values())
                    and starts.dtype == ends.dtype == np.dtype('int64')
                    and observed.dtype == np.dtype('bool') and dates.dtype.kind == 'U'
                    and np.all(starts[1:] > starts[:-1]) and np.all(ends[1:] > ends[:-1])
                    and np.all(ends-starts == duration) and np.all(ends <= CUTOFF_NS)
                    and np.all(dates <= '2026-06-30') and np.all(dates[1:] >= dates[:-1]),
                    'Original clock cutoff/shape/dtype/order mismatch')
            # Independent construction: sorted starts of constant-session groups.
            # Never uses cumulative boundary counts or the worker maximum-accumulate routine.
            group_starts = np.concatenate((np.array([0], dtype=np.int64),
                                            np.flatnonzero(dates[1:] != dates[:-1])+1))
            unobserved = np.flatnonzero(~observed)
            task = json.loads(locked_bytes(task_path))
            require((task['task'], task['frequency'], task['lookback'], task['horizon'])
                    == (task_name, frequency, length, horizon)
                    and sorted(task['folds']) == ['1', '2', '3', '4', '5'],
                    'Original task configuration mismatch')
            for fold in range(1, 6):
                guard()
                meta = task['folds'][str(fold)]
                split_path = str(Path(meta['split_file']).resolve())
                fit = selected(locked_bytes(split_path), split_path, ('fit',), input_audit)['fit']
                expected_main_audit.append({'path': split_path, 'member': 'fit.npy'})
                require(fit.dtype == np.dtype('int64') and fit.ndim == 1 and len(fit) >= 2
                        and len(fit) == meta['counts']['fit'] and np.all(fit[1:] > fit[:-1])
                        and int(fit[0]) >= length-1 and int(fit[-1])+horizon < count
                        and digest(fit.astype('<i8').tobytes()) == meta['split_sha256']['fit'],
                        'Original fit shape/order/bounds/identity mismatch')
                group_at_origin = np.searchsorted(group_starts, fit, side='right')
                group_at_first = np.searchsorted(group_starts, fit-length+1, side='right')
                group_at_future = np.searchsorted(group_starts, fit+horizon, side='right')
                expected = {
                    'history_boundary_count': (group_at_origin-group_at_first).astype(np.int64),
                    'last_session_suffix_len': np.minimum(length, fit-group_starts[group_at_origin-1]+1),
                    'future_boundary_count': (group_at_future-group_at_origin).astype(np.int64),
                }
                require(np.all(np.searchsorted(unobserved, fit, side='right')
                               == np.searchsorted(unobserved, fit-length+1, side='left'))
                        and np.all(np.searchsorted(unobserved, fit+horizon, side='right')
                                   == np.searchsorted(unobserved, fit+1, side='left')),
                        'Fit interval has unobserved history/future clock')
                if frequency == 'minute':
                    require(not expected['future_boundary_count'].any(), 'Minute future crossed session')
                # Reconstruct original80/20 cut and strict endpoint-time purge independently.
                cut = len(fit)*80//100
                first_tail_clock = int(ends[int(fit[cut])])
                prefix = np.fromiter((j < cut and int(ends[int(origin)+horizon]) < first_tail_clock
                                      for j, origin in enumerate(fit)), dtype=bool, count=len(fit))
                tail = np.fromiter((j >= cut for j in range(len(fit))), dtype=bool, count=len(fit))
                require(prefix.any() and tail.any() and not (prefix & tail).any(), 'Invalid purged partitions')
                destination = BASE/'run'/(task_name+'_fold_'+str(fold)+'_support.npz')
                row = result['rows'][state_index]
                state_index += 1
                require(row['support_file'] == str(destination), 'Derived path changed')
                payload = destination.read_bytes()
                support_hash = digest(payload)
                require(support_hash == row['support_file_sha256'], 'Saved support hash mismatch')
                saved = selected(payload, destination, SUPPORT_NAMES, derived_audit, derived=True)
                derived_hashes[str(destination)] = support_hash
                require(saved['fit_origins'].dtype == np.dtype('int64')
                        and np.array_equal(saved['fit_origins'], fit), 'Saved all-fit origins differ')
                for name, mask in (('prefix_mask', prefix), ('tail_mask', tail)):
                    require(saved[name].dtype == np.dtype('bool') and saved[name].shape == fit.shape
                            and np.array_equal(saved[name], mask), 'Saved partition mask mismatch: '+name)
                for name, values in expected.items():
                    require(saved[name].dtype == np.dtype('int64') and saved[name].shape == fit.shape
                            and np.array_equal(saved[name], values), 'All-origin routing mismatch: '+name)
                parts = {}
                for name, mask in (('full_fit', np.ones(len(fit), dtype=bool)),
                                   ('purged_prefix', prefix), ('unchanged_tail', tail)):
                    selected_fit = fit[mask]
                    part = {'origins': int(selected_fit.size),
                            'origin_sha256': digest(selected_fit.astype('<i8').tobytes())}
                    for field, values in expected.items():
                        part[field+'_histogram'] = {str(int(k)): int(v)
                            for k, v in sorted(Counter(values[mask].tolist()).items())}
                    parts[name] = part
                expected_row = {
                    'task': task_name, 'frequency': frequency, 'fold': fold, 'L': length, 'H': horizon,
                    'cut': cut, 'purged': int(cut-int(prefix.sum())),
                    'all_original_fit_origins_retained': True,
                    'all_history_and_future_clock_observed': True,
                    'first_slot_boundary_is_false': True,
                    'last_prefix_label_end_ns': int(ends[int(fit[prefix][-1])+horizon]),
                    'first_tail_decision_end_ns': first_tail_clock, 'partitions': parts,
                    'support_file': str(destination), 'support_file_sha256': support_hash,
                }
                require(row == expected_row, 'Complete row/partition/histogram mismatch')
                require(expected_row['last_prefix_label_end_ns'] < first_tail_clock, 'Strict purge failed')
                total_origins += len(fit)
                total_scalar_metadata += 3*len(fit)
                reviewed.append({'task': task_name, 'fold': fold, 'all_fit_origins_checked': len(fit),
                                 'all_three_metadata_checked': True, 'both_masks_checked': True,
                                 'all_three_partition_histograms_checked': True,
                                 'origin_sha256': meta['split_sha256']['fit'],
                                 'support_file_sha256': support_hash, 'cut': cut,
                                 'purged_prefix_count': int(prefix.sum()), 'unchanged_tail_count': int(tail.sum())})
        require(state_index == 10 and len(expected_main_audit) == 18
                and input_audit == expected_main_audit
                and result['decoded_npz_members'] == expected_main_audit
                and checkpoint['decoded_npz_members'] == expected_main_audit,
                'Complete original-only18-member decode audit mismatch')
        require(len(derived_audit) == 60, 'All10 six-member support archives must be checked')
        guard()
        require(protocol_path.read_bytes() == protocol_bytes
                and result_path.read_bytes() == result_raw and checkpoint_path.read_bytes() == checkpoint_raw,
                'Protocol/results/checkpoint changed during review')
        for path, expected_hash in original_hashes.items():
            guard()
            require(file_hash(path) == expected_hash, 'Original source changed during review')
        for path, expected_hash in derived_hashes.items():
            guard()
            require(file_hash(path) == expected_hash, 'Derived support changed during review')
        require(file_hash(BASE/'session_clock_support.py') == p['worker_sha256']
                and file_hash(BASE/'run_clock_supervisor.py') == p['supervisor_sha256']
                and file_hash(Path(__file__)) == p['reviewer_sha256'], 'Frozen code changed during review')
        result_review = {
            'kind': 'independent_group_start_searchsorted_all_origin_clock_support_review_v1',
            'status': 'all10_states_complete_origin_metadata_masks_histograms_hashes_audits_passed',
            'protocol_sha256': identity, 'contract_sha256': CONTRACT_SHA,
            'worker_sha256': p['worker_sha256'],
            'reviewer_sha256': p['reviewer_sha256'], 'supervisor_sha256': p['supervisor_sha256'],
            'partition_source_sha256': PARTITION_SHA, 'result_sha256': digest(result_raw),
            'checkpoint_sha256': digest(checkpoint_raw), 'states': reviewed,
            'total_fit_origin_occurrences_checked': total_origins,
            'total_routing_scalar_values_checked': total_scalar_metadata,
            'original_input_npz_member_audit': input_audit, 'derived_support_member_audit': derived_audit,
            'input_hashes_checked': original_hashes, 'support_hashes_checked': derived_hashes,
            'started_epoch': began, 'ended_epoch': time.time(), 'deadline_epoch': end,
            'seconds': time.time()-began, 'deadline_reset': False,
            'CPU_affinity': sorted(os.sched_getaffinity(0)), 'environment_threads': 1,
            'RLIMIT_AS_bytes': AS_LIMIT, 'getrusage_ru_maxrss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'worker_source_imported': False, 'worker_routing_function_called': False,
            'independent_algorithm': 'constant-session-group starts with searchsorted interval counts; direct strict purge and Counter histograms',
            'whole_npz_compressed_file_bytes_read_for_identity': True,
            'prices_or_label_members_decoded': False, 'old_stop_outer_members_decoded': False,
            'Q3_read': False, 'GPU': False, 'model_fits': 0, 'HTTP_calls': 0,
            'market_training_automatically_admitted': False,
            'predictive_success_claimed': False, 'research_goal_achieved': False,
            'limitations': ['Historical original-fit clock support only; origin occurrences across expanding folds are not independent market samples.',
                            'Future session and observed-clock checks describe archived eligibility and cannot be treated as point-in-time future availability features.',
                            'No checkpoint process-restart, forecasting, pricing, target or predictive mechanism test is performed.']}
        guard()
        atomic_new(output, result_review)
        signal.setitimer(signal.ITIMER_REAL, 0)
        return {key: result_review[key] for key in ('status', 'seconds', 'total_fit_origin_occurrences_checked', 'total_routing_scalar_values_checked')}
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        atomic_new(BASE/('INDEPENDENT_FAILED_'+str(os.getpid())+'.json'),
                   {'protocol_sha256': identity, 'status': 'failed_preserve_artifacts',
                    'error_type': type(error).__name__, 'reason': str(error),
                    'ended_epoch': time.time(), 'deadline_epoch': end,
                    'market_training_automatically_admitted': False, 'research_goal_achieved': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.protocol)))
