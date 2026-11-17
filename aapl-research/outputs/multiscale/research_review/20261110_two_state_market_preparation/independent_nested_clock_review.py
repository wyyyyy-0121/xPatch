"""Independent original-clock nested role and crossstate decision-clock review.

NumPy is imported only after frozen protocol/resource/scope admission.
This file never imports nested_partitions.py or any primary worker/model module.
"""
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
AS_LIMIT = 2147483648
CONTRACT_SHA = '6f1dec0856c94a9c3df090156a247a27d79ff893dd88398a90406b6c725ace11'
PRIOR_PROTOCOL_SHA = '57966fd8856fe2b4b44309bf3e8d65560d470fc15398f2ea1671699c55360598'
PRIOR_RESULTS_SHA = '6c26511af0f3dd6b27ded47a4a913d9803dc74674ded41e3d3090ee8099c026d'
PRIOR_ACCEPTANCE_SHA = 'a27fdafaceddc61ec9bfd070a903f1035a9a5050f207471c6229c0c22847d66c'
NESTED_SHA = '726700c715b2d4ce50552ee3d9190a6c18583d43973a54da9785f0ef4e7ee4e8'
CUTOFF_NS = 1782878400000000000
ROLE_ORDER = ('Q', 'S', 'C_cal', 'gap_Q_to_S', 'gap_B_to_C', 'gap_P_to_T', 'T')

PARTITIONS = ('F', 'P', 'B', 'Q', 'S', 'C_cal', 'T',
              'gap_P_to_T', 'gap_B_to_C', 'gap_Q_to_S')
PARENTS = {'F': None, 'P': 'F', 'B': 'P', 'Q': 'B', 'S': 'B',
           'C_cal': 'P', 'T': 'F', 'gap_P_to_T': 'F',
           'gap_B_to_C': 'P', 'gap_Q_to_S': 'B'}
TERMINAL_SOURCES = (('Q_fit', 'Q'), ('S_early_stop', 'S'),
                    ('C_calibration', 'C_cal'), ('gap_Q_to_S', 'gap_Q_to_S'),
                    ('gap_B_to_C', 'gap_B_to_C'), ('gap_P_to_T', 'gap_P_to_T'),
                    ('T_assessment', 'T'))


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def int64_hash(origins):
    import numpy as np
    return hashlib.sha256(np.asarray(origins, dtype='<i8').tobytes(order='C')).hexdigest()


def build_reference(clock, fit, accepted_prefix_mask, accepted_tail_mask, *, horizon, lookback):
    """Independently use ordered session lists and scalar strict-clock predicates."""
    import numpy as np
    require(type(horizon) is int and horizon >= 1 and type(lookback) is int and lookback >= 1,
            'Invalid fixed L/H')
    require(isinstance(clock, dict) and set(clock) == {'end_ns', 'session_date'}, 'Clock-only fields')
    ends, sessions = clock['end_ns'], clock['session_date']
    require(isinstance(ends, np.ndarray) and ends.ndim == 1 and ends.dtype == np.dtype('int64')
            and isinstance(sessions, np.ndarray) and sessions.dtype.kind == 'U'
            and sessions.shape == ends.shape and np.all(ends[1:] > ends[:-1])
            and np.all(sessions[1:] >= sessions[:-1]), 'Ordered original clock arrays required')
    require(isinstance(fit, np.ndarray) and fit.ndim == 1 and fit.dtype == np.dtype('int64')
            and len(fit) >= 2 and np.all(fit[1:] > fit[:-1])
            and int(fit[0]) >= lookback-1 and int(fit[-1])+horizon < len(ends), 'Original fit bounds')
    for mask in (accepted_prefix_mask, accepted_tail_mask):
        require(isinstance(mask, np.ndarray) and mask.dtype == np.dtype('bool')
                and mask.shape == fit.shape, 'Original fit-position boolean masks required')
    cut = len(fit)*80//100
    original_first_tail = int(ends[int(fit[cut])])
    expected_prefix = np.fromiter((position < cut and int(ends[int(origin)+horizon]) < original_first_tail
                                   for position, origin in enumerate(fit)), dtype=bool, count=len(fit))
    expected_tail = np.fromiter((position >= cut for position in range(len(fit))),
                                dtype=bool, count=len(fit))
    require(np.array_equal(accepted_prefix_mask, expected_prefix)
            and np.array_equal(accepted_tail_mask, expected_tail), 'Accepted P/T relation changed')
    F = fit.copy()
    P, T = F[expected_prefix], F[expected_tail]
    require(len(P) > 0 and len(T) > 0, 'Original P/T nonempty')
    gap_PT = F[~expected_prefix & ~expected_tail]

    def inner(train, stage):
        # Independent ordered groups, rather than np.unique/np.isin worker routine.
        labels = list(dict.fromkeys(str(sessions[int(origin)]) for origin in train))
        tail_sessions = max(1, math.ceil(len(labels)*.15))
        require(len(labels) > tail_sessions, stage+':insufficient_groups')
        selected_sessions = set(labels[len(labels)-tail_sessions:])
        stopped = [int(origin) for origin in train if str(sessions[int(origin)]) in selected_sessions]
        require(bool(stopped), stage+':empty_stop')
        first_stop_clock = int(ends[stopped[0]])
        earlier = [int(origin) for origin in train if int(ends[int(origin)+horizon]) < first_stop_clock]
        require(bool(earlier), stage+':insufficient_fit')
        chosen = set(earlier) | set(stopped)
        gap = [int(origin) for origin in train if int(origin) not in chosen]
        return (np.asarray(earlier, dtype=np.int64), np.asarray(stopped, dtype=np.int64),
                np.asarray(gap, dtype=np.int64),
                {'group_count': len(labels), 'stop_group_count': tail_sessions,
                 'stop_fraction': .15, 'first_stop_decision_ns': first_stop_clock})

    B, cal, gap_BC, meta_BC = inner(P, 'P_to_B_Ccal')
    Q, S, gap_QS, meta_QS = inner(B, 'B_to_Q_S')
    require(len(Q) >= 2 and len(S) > 0 and len(cal) > 0, 'Nested minimum counts')
    for left, right in ((Q, S), (S, cal), (cal, T)):
        require(max(int(ends[int(o)+horizon]) for o in left)
                < min(int(ends[int(o)]) for o in right), 'Strict adjacent role label purge')
    arrays = {'F': F, 'P': P, 'B': B, 'Q': Q, 'S': S, 'C_cal': cal, 'T': T,
              'gap_P_to_T': gap_PT, 'gap_B_to_C': gap_BC, 'gap_Q_to_S': gap_QS}
    fit_lookup = {int(origin): position for position, origin in enumerate(F)}
    partitions = {}
    for name in PARTITIONS:
        origins = arrays[name]
        positions = np.asarray([fit_lookup[int(o)] for o in origins], dtype=np.int64)
        parent = PARENTS[name]
        if parent is None:
            parent_positions = positions.copy()
        else:
            parent_lookup = {int(o): position for position, o in enumerate(arrays[parent])}
            parent_positions = np.asarray([parent_lookup[int(o)] for o in origins], dtype=np.int64)
        partitions[name] = {'origins': origins.copy(), 'fit_positions': positions,
                            'parent': parent, 'parent_positions': parent_positions}
    terminal = ['']*len(F)
    for role, name in TERMINAL_SOURCES:
        for origin in arrays[name]:
            position = fit_lookup[int(origin)]
            require(terminal[position] == '', 'Overlapping terminal roles')
            terminal[position] = role
    require(all(terminal), 'Unassigned original fit origin')
    terminal_array = np.asarray(terminal, dtype='<U24')
    group_starts = np.concatenate((np.array([0], dtype=np.int64),
                                   np.flatnonzero(sessions[1:] != sessions[:-1])+1))
    at_origin = np.searchsorted(group_starts, F, side='right')
    at_window = np.searchsorted(group_starts, F-lookback+1, side='right')
    suffix = np.minimum(lookback, F-group_starts[at_origin-1]+1).astype(np.int64)
    return {'partitions': partitions,
            'ledger': {'origins': F.copy(), 'fit_positions': np.arange(len(F), dtype=np.int64),
                       'terminal_role': terminal_array},
            'history_support': {'origins': F.copy(), 'fit_positions': np.arange(len(F), dtype=np.int64),
                                'suffix_length': suffix,
                                'last_reset_local_position': (lookback-suffix).astype(np.int64),
                                'boundary_count': (at_origin-at_window).astype(np.int64)},
            'reference_metadata': {'original_cut': cut, 'P_to_B_Ccal': meta_BC, 'B_to_Q_S': meta_QS,
                                   'partition_origin_hashes': {name: int64_hash(values)
                                                               for name, values in arrays.items()}}}


def ordered_intersection(left, right):
    """Sorted join independent of np.intersect1d; preserve canonical origin order."""
    import numpy as np
    if not len(left) or not len(right):
        return np.empty(0, dtype=np.int64)
    position = np.searchsorted(right, left)
    valid = position < len(right)
    selection = np.zeros(len(left), dtype=bool)
    selection[valid] = right[position[valid]] == left[valid]
    return left[selection].astype(np.int64, copy=True)


def intersection_reference(state_a, state_b):
    """All49 terminal-role pairs; decision-clock join also handles crossfrequency."""
    require(state_a['stateUID'] != state_b['stateUID'], 'Distinct state pair required')
    same = state_a['frequency'] == state_b['frequency']
    left_clock, right_clock = state_a['clock']['end_ns'], state_b['clock']['end_ns']
    rows = []
    for role_a in ROLE_ORDER:
        for role_b in ROLE_ORDER:
            left = state_a['reference']['partitions'][role_a]['origins']
            right = state_b['reference']['partitions'][role_b]['origins']
            values = ordered_intersection(left_clock[left], right_clock[right])
            origins = ordered_intersection(left, right) if same else None
            if same:
                import numpy as np
                require(np.array_equal(left_clock[origins], values), 'Same-grid origin/clock join mismatch')
            rows.append({'left_stateUID': state_a['stateUID'], 'right_stateUID': state_b['stateUID'],
                         'left_role': role_a, 'right_role': role_b, 'same_frequency': same,
                         'decision_clock_intersection_count': len(values),
                         'ordered_decision_clock_sha256': int64_hash(values),
                         'ordered_origin_sha256': int64_hash(origins) if same else None,
                         'first_decision_ns': int(values[0]) if len(values) else None,
                         'last_decision_ns': int(values[-1]) if len(values) else None})
    return rows


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(65536), b''):
            h.update(block)
    return h.hexdigest()


def byte_hash(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def atomic_new_json(path, value):
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
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temporary is not None:
            os.unlink(temporary)


def run(protocol_path):
    began = time.time()
    require(protocol_path.is_absolute(), 'Absolute protocol required')
    protocol_raw = protocol_path.read_bytes()
    p = json.loads(protocol_raw)
    identity = byte_hash(protocol_raw)
    start, end = p['started_epoch'], p['deadline_epoch']
    require(type(start) in (int, float) and type(end) in (int, float)
            and math.isfinite(start) and math.isfinite(end) and end-start == 180
            and start <= began < end-1, 'Original finite180s window')
    require(p['kind'] == 'two_state_nested_clock_roles_v1'
            and all(p[key] is False for key in ('market_prices_labels_allowed', 'model_fitting_allowed',
                                               'Q3_allowed', 'GPU', 'paid_resources')),
            'Clock-only no-price/no-label scope')
    require(os.sched_getaffinity(0) == {26}, 'Reviewer requires CPU26')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '-1', 'GPU disabled before imports')
    require(all(os.environ.get(key) == '1' for key in
                ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')),
            'Single-thread numerical environment before import')
    code_paths = {'worker': BASE/'nested_clock_worker.py', 'reviewer': Path(__file__).resolve(),
                  'supervisor': BASE/'run_nested_supervisor.py', 'nested_module': Path(p['nested_module_path'])}
    for role, path in code_paths.items():
        require(file_hash(path) == p[role+'_sha256'], 'Frozen code identity changed: '+role)
    require(p['nested_module_sha256'] == NESTED_SHA and p['contract_sha256'] == CONTRACT_SHA
            and file_hash(BASE/'NESTED_CLOCK_CONTRACT.json') == CONTRACT_SHA, 'Contract/nested identity')
    resource.setrlimit(resource.RLIMIT_AS, (AS_LIMIT, AS_LIMIT))
    result_path = BASE/'nested_run'/'RESULTS.json'
    checkpoint_path = BASE/'nested_run'/'CHECKPOINT.json'
    output_path = BASE/'nested_run'/'INDEPENDENT_NESTED_CLOCK_REVIEW.json'
    require(not output_path.exists(), 'Refuse existing independent result')

    def guard():
        require(time.time() < end-1, 'Original180s deadline reached')
        require(os.sched_getaffinity(0) == {26}, 'CPU affinity drift')

    def expire(*_):
        raise TimeoutError('Original nested-clock deadline or termination')

    for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, expire)
    signal.setitimer(signal.ITIMER_REAL, max(.001, end-1-time.time()))
    try:
        import numpy as np
        guard()
        locks = {}
        for lock in p['locks']:
            path = str(Path(lock['path']).resolve())
            require(Path(lock['path']).is_absolute() and path == lock['path'] and path not in locks,
                    'Unique canonical input locks')
            locks[path] = lock['sha256']
        checked_inputs, decoded_inputs, decoded_derived, checked_outputs = {}, [], [], {}

        def body(ref):
            guard()
            path = str(Path(ref).resolve())
            require(path in locks, 'Reject unlocked source before reading')
            raw = Path(path).read_bytes()
            require(byte_hash(raw) == locks[path], 'Frozen input changed')
            checked_inputs[path] = locks[path]
            return raw

        input_bars = ('end_ns', 'session_date')
        input_support = ('fit_origins', 'prefix_mask', 'tail_mask')
        derived_keys = tuple(name+'_'+field for name in PARTITIONS
                             for field in ('origins', 'fit_positions', 'parent_positions'))
        derived_keys += ('terminal_role', 'history_suffix_length', 'history_boundary_count')

        def read_members(raw, path, names, audit, derived=False):
            require(names in (input_bars, input_support, derived_keys), 'NPZ member allowlist')
            values = {}
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                listing = archive.namelist()
                if derived:
                    require(len(listing) == 33 and set(listing) == {name+'.npy' for name in names},
                            'Exactly33 derived members required')
                for name in names:
                    guard()
                    require(listing.count(name+'.npy') == 1, 'Ambiguous member')
                    with archive.open(name+'.npy') as stream:
                        value = np.lib.format.read_array(stream, allow_pickle=False)
                    require(not value.dtype.hasobject, 'Object array forbidden')
                    values[name] = value
                    audit.append({'path': str(Path(path).resolve()), 'member': name+'.npy'})
            return values

        for path in locks:
            body(path)
        prior_raw = body(p['prior_results_path'])
        acceptance_raw = body(p['prior_acceptance_path'])
        prior_protocol_raw = body(p['prior_protocol_path'])
        require(byte_hash(prior_raw) == PRIOR_RESULTS_SHA
                and byte_hash(acceptance_raw) == PRIOR_ACCEPTANCE_SHA
                and byte_hash(prior_protocol_raw) == PRIOR_PROTOCOL_SHA, 'Original accepted evidence identity')
        prior, acceptance, prior_protocol = map(json.loads, (prior_raw, acceptance_raw, prior_protocol_raw))
        require(prior['status'] == 'original_five_fold_fit_clock_support_complete'
                and acceptance['status'] == 'accepted_all_original_fit_clock_support_metadata_only'
                and acceptance['task_fold_states'] == 10 and len(prior['rows']) == 10
                and acceptance['evidence_sha256']['EXECUTION_PROTOCOL.json'] == PRIOR_PROTOCOL_SHA
                and acceptance['evidence_sha256']['run/RESULTS.json'] == PRIOR_RESULTS_SHA,
                'Original acceptance/results binding')
        require(p['original_bars'] == prior_protocol['original_bars'], 'Original bars config changed')
        clocks = {}
        expected_audit = []
        for frequency in ('minute', 'hour'):
            info = p['original_bars'][frequency]
            require(locks[info['path']] == info['sha256'], 'Original bar lock mismatch')
            clock = read_members(body(info['path']), info['path'], input_bars, decoded_inputs)
            ends, dates = clock['end_ns'], clock['session_date']
            require(ends.dtype == np.dtype('int64') and ends.ndim == 1 and len(ends) == info['rows']
                    and dates.dtype.kind == 'U' and dates.shape == ends.shape
                    and np.all(ends[1:] > ends[:-1]) and np.all(ends <= CUTOFF_NS)
                    and np.all(dates[1:] >= dates[:-1]) and np.all(dates <= '2026-06-30'),
                    'Original pre-July clock invalid')
            clocks[frequency] = clock
            expected_audit.extend({'path': info['path'], 'member': name+'.npy'} for name in input_bars)
        result_raw, checkpoint_raw = result_path.read_bytes(), checkpoint_path.read_bytes()
        result, checkpoint = json.loads(result_raw), json.loads(checkpoint_raw)
        require(result['kind'] == 'matched_two_state_original_clock_nested_roles_v1'
                and result['status'] == 'all10_nested_roles_and2205_clock_overlap_pairs_complete'
                and result['protocol_sha256'] == checkpoint['protocol_sha256'] == identity
                and result['deadline_epoch'] == checkpoint['deadline_epoch'] == end
                and start <= result['started_epoch'] <= result['ended_epoch'] < end
                and result['CPU_affinity'] == [24] and result['RLIMIT_AS_bytes'] == AS_LIMIT,
                'Actual main status/scope/window')
        require(result['rows'] == checkpoint['completed_rows'] and len(result['rows']) == 10
                and result['role_order'] == list(ROLE_ORDER), 'All10 complete row/role ledger')
        require(result['whole_compressed_npz_bytes_read_for_identity'] is True
                and result['model_fits'] == 0
                and all(result[key] is False for key in ('price_or_label_members_decoded',
                    'old_stop_outer_members_decoded', 'Q3_read', 'GPU', 'full_g_tensor_verified',
                    'market_training_admitted', 'research_goal_achieved')), 'Main scope remains closed')
        references, state_receipts = [], []
        total_origins = 0
        for old, actual in zip(prior['rows'], result['rows']):
            guard()
            frequency, task, fold = old['frequency'], old['task'], old['fold']
            require((frequency, task, old['L'], old['H']) in
                    (('minute', 'minute_h60_future_log_return', 480, 60),
                     ('hour', 'hourly_h8_future_log_return', 96, 8)) and fold in range(1, 6),
                    'Original task/fold identity')
            path = old['support_file']
            require(locks.get(path) == old['support_file_sha256'], 'Accepted support archive changed')
            accepted = read_members(body(path), path, input_support, decoded_inputs)
            expected_audit.extend({'path': path, 'member': name+'.npy'} for name in input_support)
            reference = build_reference(clocks[frequency], accepted['fit_origins'],
                                        accepted['prefix_mask'], accepted['tail_mask'],
                                        horizon=old['H'], lookback=old['L'])
            part_meta = {}
            expected_arrays = {}
            for name in PARTITIONS:
                part = reference['partitions'][name]
                origins = part['origins']
                for field in ('origins', 'fit_positions', 'parent_positions'):
                    expected_arrays[name+'_'+field] = part[field]
                part_meta[name] = {
                    'count': len(origins), 'origin_sha256': int64_hash(origins),
                    'fit_positions_sha256': int64_hash(part['fit_positions']), 'parent': part['parent'],
                    'parent_positions_sha256': int64_hash(part['parent_positions']),
                    'session_count': len(set(str(clocks[frequency]['session_date'][int(o)]) for o in origins)),
                    'first_decision_ns': int(clocks[frequency]['end_ns'][int(origins[0])]) if len(origins) else None,
                    'last_decision_ns': int(clocks[frequency]['end_ns'][int(origins[-1])]) if len(origins) else None,
                    'last_label_end_ns': int(clocks[frequency]['end_ns'][int(origins[-1])+old['H']]) if len(origins) else None}
            require(part_meta['F']['origin_sha256'] == old['partitions']['full_fit']['origin_sha256']
                    and part_meta['P']['origin_sha256'] == old['partitions']['purged_prefix']['origin_sha256']
                    and part_meta['T']['origin_sha256'] == old['partitions']['unchanged_tail']['origin_sha256'],
                    'Accepted original F/P/T changed')
            uid_input = {'task': task, 'fold': fold, 'L': old['L'], 'H': old['H'],
                         'clock_end_sha256': int64_hash(clocks[frequency]['end_ns']),
                         'session_date_sha256': byte_hash(clocks[frequency]['session_date'].tobytes()),
                         'original_fit_sha256': int64_hash(accepted['fit_origins']),
                         'nested_module_sha256': p['nested_module_sha256'],
                         'role_origin_hashes': {name: part_meta[name]['origin_sha256'] for name in PARTITIONS}}
            uid = byte_hash(canonical(uid_input))
            expected_arrays['terminal_role'] = reference['ledger']['terminal_role']
            expected_arrays['history_suffix_length'] = reference['history_support']['suffix_length']
            expected_arrays['history_boundary_count'] = reference['history_support']['boundary_count']
            destination = BASE/'nested_run'/(task+'_fold_'+str(fold)+'_nested.npz')
            raw = destination.read_bytes()
            output_hash = byte_hash(raw)
            saved = read_members(raw, destination, derived_keys, decoded_derived, derived=True)
            for name, expected in expected_arrays.items():
                require(saved[name].dtype == expected.dtype and saved[name].shape == expected.shape
                        and np.array_equal(saved[name], expected), 'Exact all-origin derived array mismatch: '+name)
            checked_outputs[str(destination)] = output_hash
            expected_row = {'task': task, 'frequency': frequency, 'fold': fold, 'L': old['L'], 'H': old['H'],
                            'stateUID': uid, 'state_UID_input': uid_input, 'partitions': part_meta,
                            'terminal_role_counts': dict(sorted(Counter(expected_arrays['terminal_role'].tolist()).items())),
                            'terminal_role_sha256': byte_hash(expected_arrays['terminal_role'].tobytes()),
                            'history_suffix_length_sha256': int64_hash(expected_arrays['history_suffix_length']),
                            'history_boundary_count_sha256': int64_hash(expected_arrays['history_boundary_count']),
                            'nested_file': str(destination), 'nested_file_sha256': output_hash}
            require(actual == expected_row, 'Complete stateUID/partition/hash/role row mismatch')
            total_origins += part_meta['F']['count']
            state_receipts.append({'stateUID': uid, 'task': task, 'fold': fold,
                                   'all_F_occurrences_checked': part_meta['F']['count'],
                                   'all33_arrays_exact': True, 'nested_file_sha256': output_hash,
                                   'all_partition_metadata_UID_and_terminal_roles_exact': True})
            references.append({'stateUID': uid, 'frequency': frequency, 'horizon': old['H'],
                               'clock': clocks[frequency], 'reference': reference})
        require(len({state['stateUID'] for state in references}) == 10, 'Unique10 stateUIDs')
        overlap_rows = []
        for index, left in enumerate(references):
            for right in references[index+1:]:
                guard()
                overlap_rows.extend(intersection_reference(left, right))
        require(len(overlap_rows) == 2205 and result['crossstate_role_clock_overlaps'] == overlap_rows,
                'All45x49 exact decision-clock intersections differ')
        require(len(expected_audit) == 34 and decoded_inputs == expected_audit
                and result['decoded_members'] == checkpoint['decoded_members'] == expected_audit,
                'Exact34 inputmember access audit')
        require(len(decoded_derived) == 330, 'All10x33 derived member audit')
        guard()
        require(protocol_path.read_bytes() == protocol_raw and result_path.read_bytes() == result_raw
                and checkpoint_path.read_bytes() == checkpoint_raw, 'Frozen result/protocol bytes changed')
        for path, expected_hash in checked_inputs.items():
            guard()
            require(file_hash(path) == expected_hash, 'Original input changed during review')
        for path, expected_hash in checked_outputs.items():
            guard()
            require(file_hash(path) == expected_hash, 'Derived output changed during review')
        for role, path in code_paths.items():
            require(file_hash(path) == p[role+'_sha256'], 'Frozen code changed during review')
        report = {
            'kind': 'independent_original_clock_nested_roles_and_crossstate_intersections_review_v1',
            'status': 'all10_states_33arrays_UIDs_partitions_and2205_decision_clock_pairs_exact',
            'protocol_sha256': identity, 'contract_sha256': CONTRACT_SHA,
            'worker_sha256': p['worker_sha256'], 'reviewer_sha256': p['reviewer_sha256'],
            'supervisor_sha256': p['supervisor_sha256'], 'nested_module_sha256': p['nested_module_sha256'],
            'main_result_sha256': byte_hash(result_raw), 'main_checkpoint_sha256': byte_hash(checkpoint_raw),
            'states': state_receipts, 'fit_origin_occurrences_checked': total_origins,
            'partition_arrays_per_state': 33, 'state_pairs': 45, 'terminal_role_pairs_per_state_pair': 49,
            'decision_clock_overlap_rows_checked': len(overlap_rows),
            'overlap_rows_canonical_sha256': byte_hash(canonical(overlap_rows)),
            'original_member_audit': decoded_inputs, 'derived_member_audit': decoded_derived,
            'input_hashes_checked': checked_inputs, 'nested_output_hashes_checked': checked_outputs,
            'started_epoch': began, 'ended_epoch': time.time(), 'deadline_epoch': end,
            'seconds': time.time()-began, 'original_deadline_reset': False,
            'CPU_affinity': sorted(os.sched_getaffinity(0)), 'threads': 1, 'RLIMIT_AS_bytes': AS_LIMIT,
            'RSS_ru_maxrss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'primary_worker_imported': False, 'primary_nested_module_imported': False,
            'independent_algorithm': 'ordered Python sessiongroups, scalar strictpurge and originalposition lookups; searchsorted joins forclock intersections',
            'whole_compressed_NPZ_bytes_read_for_identity': True,
            'price_or_label_members_decoded': False, 'old_stop_outer_members_decoded': False,
            'Q3_read': False, 'GPU': False, 'model_fits': 0, 'HTTP_calls': 0,
            'full_g_tensor_verified': False, 'market_training_admitted': False,
            'predictive_success_claimed': False, 'research_goal_achieved': False,
            'limitations': ['Origin occurrences across folds are repeated historical supports, not independent market samples.',
                            'Crossstate intersections cover decision timestamps only, not every raw-history/target-path payload reuse.',
                            'Existing same-time reuse across states prevents globaltimestampblindness; phase authorization remains state/role/operation scoped.',
                            'History suffix/count checks do not certify a complete g tensor or any price/scaler/model adapter.']}
        guard()
        atomic_new_json(output_path, report)
        signal.setitimer(signal.ITIMER_REAL, 0)
        return {key: report[key] for key in ('status', 'seconds', 'fit_origin_occurrences_checked', 'decision_clock_overlap_rows_checked')}
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        atomic_new_json(BASE/('INDEPENDENT_NESTED_FAILED_'+str(os.getpid())+'.json'),
                        {'protocol_sha256': identity, 'status': 'failed_preserve_artifacts',
                         'error_type': type(error).__name__, 'reason': str(error),
                         'ended_epoch': time.time(), 'deadline_epoch': end,
                         'market_training_admitted': False, 'research_goal_achieved': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    print(json.dumps(run(parser.parse_args().protocol)))
