"""Independent source/JSON/log review only; never opens PT, NPZ or market arrays."""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

BASE = Path(__file__).resolve().parents[1]
RUN = BASE / 'runs_io_r1'
inventory = {}
checks = []

def read(path):
    path = Path(path).resolve()
    if path.suffix not in {'.json', '.py', '.md', '.log', '.pth'}:
        raise RuntimeError('Non-source/non-small-receipt input denied: ' + str(path))
    value = path.read_bytes()
    inventory[str(path)] = {'sha256': hashlib.sha256(value).hexdigest(), 'bytes': len(value)}
    return value

def obj(path):
    return json.loads(read(path))

def check(name, condition, detail=None):
    checks.append({'name': name, 'passed': bool(condition), 'detail': detail})
    if not condition:
        raise RuntimeError('Independent result review failed: ' + name)

def sha(path):
    read(path)
    return inventory[str(Path(path).resolve())]['sha256']

def runtime_check(runtime):
    return runtime == {'torch': '2.14.1+cu130', 'mkldnn_enabled': True,
                       'CPU': [24], 'GPU_hidden': True, 'threads': 1, 'interop': 1}

def main():
    protocol_path = BASE / 'SYNTHETIC_PROTOCOL_IO_R1.json'
    protocol = obj(protocol_path)
    protocol_sha = sha(protocol_path)
    check('frozen_protocol_sha', protocol_sha == '9d3b76c65659b9cf4af7afa123bb7cde6f20cdb7fad50c14e428e8ac5bc54a7c')
    rules = obj(protocol['rules_path'])
    check('frozen_rules_sha', sha(protocol['rules_path']) == protocol['rules_sha256'] == '2ae72258bb5108680df7297e4850cd72098014f50a50e170a6793ff0915b313e')
    acceptance = obj(protocol['source_acceptance_path'])
    check('frozen_source_acceptance_sha', sha(protocol['source_acceptance_path']) == protocol['source_acceptance_sha256'] == '89c45bbc4661bfc586f731d65927644cee184b7f91aa391bdf312162284626fe')
    for group in ['code_files', 'source_files']:
        for path, expected in protocol[group].items():
            check('unchanged_' + group + ':' + str(Path(path).name), sha(path) == expected)
    check('eight_source_bindings', len(protocol['code_files']) == 8 and all(protocol['code_files'][p] == v['sha256'] for p, v in acceptance['source_bindings'].items()))
    for key in ['design', 'independent_design_review', 'binding_clarifications', 'synthetic_acceptance']:
        bound = rules[key]
        check('preparation_evidence_binding:' + key, sha(bound['path']) == bound['sha256'])
    check('metadata_only_binding', sha(protocol['input_metadata_path']) == protocol['input_metadata_sha256'])
    check('previous_protocol_preserved', sha(protocol['previous_protocol_path']) == protocol['previous_protocol_sha256'])
    io = acceptance['IO_repair_review']
    io_class = obj(io['classification_path'])
    check('IO_classification_binding', sha(io['classification_path']) == io['classification_sha256'] == 'b6be3276e9119081009484f2a87aea375481695d97f93ea9dc7eacc37d563737')
    check('IO_only_no_numeric_rescue', io['numeric_gate_failure_rescue'] is False and io['other_seven_source_bindings_unchanged'] is True and io['scientific_fixture_threshold_backend_model_seed_optimizer_and_deadline_unchanged'] is True)
    deadline = rules['deadline_epoch']
    check('single_absolute_deadline', deadline == 1794257947.4375608 == protocol['deadline_epoch'] and not protocol['deadline_may_reset'] and not protocol['deadline_reset'])
    check('synthetic_only_admission', protocol['market_admission'] is False and protocol['market_files'] == {} and not protocol['Q3_allowed'] and not protocol['GPU'] and not protocol['paid_resources'])
    receipt = obj(RUN / 'EXECUTION_RECEIPT.json')
    check('terminal_receipt', receipt['status'] == 'completed_worker_sequence_pending_independent_acceptance' and receipt['error'] is None and receipt['protocol_sha256'] == protocol_sha)
    events = receipt['events']
    expected = [('qualify', 'qualify', False, None, 0), ('cost', 'cost', False, None, 0),
                ('trajectory', 'continuous', False, None, 0), ('trajectory', 'signal', False, 'signal', 75),
                ('trajectory', 'signal', True, None, 0), ('trajectory', 'deadline', False, 'deadline', 75),
                ('trajectory', 'deadline', True, None, 0), ('compare', 'comparison', False, None, 0)]
    actual = [(e['mode'], e['name'], e['resume'], e['pause_kind'], e['exit_code']) for e in events]
    check('eight_child_sequence_and_codes', actual == expected)
    check('eight_distinct_pids', len({e['pid'] for e in events}) == 8)
    check('strict_sequence_before_fixed_deadline', all(e['started_epoch'] <= e['finished_epoch'] < deadline for e in events) and all(a['finished_epoch'] <= b['started_epoch'] for a, b in zip(events, events[1:])))
    check('source_accepted_before_actual_launch', datetime.fromisoformat(acceptance['at_UTC']).timestamp() < events[0]['started_epoch'])
    for e in events:
        cmd = ['/home/ayaka/wslcode/aapl-research/.venv/bin/python', str(BASE / 'code/guarded_run.py'), '--protocol', str(protocol_path), '--mode', e['mode'], '--output', str(RUN / e['name'])]
        if e['pause_kind']:
            cmd += ['--pause-kind', e['pause_kind']]
        if e['resume']:
            cmd += ['--resume']
        check('child_command:' + str(e['pid']), e['command'] == cmd)
        log = read(e['log'])
        check('log_commitment:' + str(e['pid']), inventory[str(Path(e['log']).resolve())]['sha256'] == e['log_sha256'] and b'Traceback' not in log)
    signal_event = events[3]['signals_sent']
    check('actual_external_SIGUSR1_receipt', len(signal_event) == 1 and signal_event[0]['signal'] == 'SIGUSR1' and signal_event[0]['reason'] == 'actual_commonbatch4' and events[3]['started_epoch'] < signal_event[0]['epoch'] < events[3]['finished_epoch'])
    ready = obj(RUN / 'signal/READY_FOR_SIGNAL.json')
    check('signal_ready_actual_pid_batch4', ready == {'pid': events[3]['pid'], 'steps': 4})
    for name, reason, first, second in [('signal', 'signal', 3, 4), ('deadline', 'expired_deadline', 5, 6)]:
        pause = obj(RUN / name / 'PAUSED.json')
        check('pause4_newPIDresume:' + name, pause == {'reason': reason, 'steps': 4, 'deadline_epoch': deadline, 'deadline_reset': False} and events[first]['pid'] != events[second]['pid'])
    for key in ['market_read', 'market_access_attempted', 'whole_preQ3_decode_receipt_available', 'Q3_read', 'full_goal_achieved', 'deadline_reset']:
        check('execution_scope_flag:' + key, receipt[key] is False)
    check('execution_receipt_fixed_deadline', receipt['deadline_epoch'] == deadline)

    scalers, identities, results = [], [], []
    for name in ['continuous', 'signal', 'deadline']:
        summary = obj(RUN / name / 'SUMMARY.json')
        checkpoint = obj(RUN / name / 'CHECKPOINT.json')
        freeze = obj(RUN / name / 'GLOBAL_FREEZE.json')
        started = obj(RUN / name / 'STARTED.json')
        scalers.append(obj(RUN / name / 'SCALER.json'))
        result = obj(RUN / name / 'RESULTS.json'); results.append(result)
        identities.append(checkpoint['identity'])
        check('complete20common60updates2epochs:' + name, summary['status'] == 'complete' and summary['steps'] == 20 and summary['step_unit'] == 'common_lockstep_batch' and summary['actual_optimizer_updates'] == 60 and summary['epochs'] == 2)
        check('summary_committed_manifest:' + name, summary['manifest'] == checkpoint and checkpoint['steps'] == 20 and checkpoint['epoch'] == 2)
        check('started_identity:' + name, started == {'state': 'synthetic_native', 'protocol_sha256': protocol_sha})
        check('summary_synthetic_scope:' + name, summary['synthetic'] and not summary['market_read'] and not summary['Q3_read'] and runtime_check(summary['runtime']))
        check('globalfreeze_before_tail:' + name, freeze['all_models_controls_frozen'] and freeze['states'] == ['synthetic_native'] and freeze['learner_count'] == 3 and freeze['control_state_count'] == 1 and freeze['prediction_path_reloaded_from_committed_payloads'] and freeze['single_global_postfreeze_tail_pass'] and freeze['not_globally_blind'] and freeze['protocol_sha256'] == protocol_sha)
        final = freeze['final_fits']
        check('finalfit_prefix_counter:' + name, len(final) == 1 and final[0]['dedicated_tail_rows_before_freeze'] == 0 and final[0]['phase'] == 'prefix' and final[0]['common_batches'] == 20 and final[0]['actual_optimizer_updates'] == 60 and final[0]['checkpoint']['identity'] == checkpoint['identity'])
        check('final_result_scope:' + name, result['status'] == 'completed_all_fixed_states' and result['decision'] == {'synthetic_only': True, 'forecast_success': False} and all(result[k] is False for k in ['market_read', 'whole_preQ3_containers_decoded', 'Q3_read', 'full_goal_achieved']) and runtime_check(result['runtime']))
    check('identical_identity_scaler_and_scores_across3', identities[0] == identities[1] == identities[2] and scalers[0] == scalers[1] == scalers[2] and results[0] == results[1] == results[2])
    identity = identities[0]
    check('identity_code_source_backend', identity['protocol_sha256'] == protocol_sha and identity['code_files'] == protocol['code_files'] and identity['source_files'] == protocol['source_files'] and identity['epochs'] == 2 and identity['batch'] == 128 and identity['seed'] == 2026 and identity['backend_identity'] == {'torch': '2.14.1+cu130', 'mkldnn_enabled': True, 'mkldnn_available': True, 'deterministic_algorithms': True, 'threads': 1, 'interop': 1, 'CPU': [24], 'GPU_hidden': True})
    check('fixed_partition', identity['partition'] == {'id': 'synthetic_native', 'original_fit_count': 1675, 'boundary': 1340, 'prefix_count': 1280, 'tail_count': 335, 'first_tail_origin': 1819, 'prefix_sha256': 'c08f71ee8affbdd1ac5a59e8fdfc33f2de2e78a3495ae39496e1c9675d63a52b', 'tail_sha256': 'ec4d20835f5b347ca144803cceadeb1fb69157cd59146a6ad39ed899bab0d16c'})
    qualification = obj(RUN / 'qualify/RESULTS.json')
    initial = qualification['native_initial']
    scaler = scalers[0]['scaler']
    bound = 32 * 2**-23 * max(1., max(map(abs, scaler['y_mean'])), max(map(abs, scaler['y_std'])))
    qchecks = {x['check']: x for x in initial['checks']}
    check('physicalzero_independent_bound', all(qchecks[n]['bound'] == bound and 0 <= qchecks[n]['max_abs'] <= bound for n in ['Z_physical_zero', 'C_physical_zero']), {'FP32_eps': 2**-23, 'bound': bound})
    check('all3_native_parameterVJP', all(qchecks['VJP_' + n]['max_abs'] == 0. for n in ['ones', 'sin', 'cos']))
    cbone = initial['C_firstbone_inf']; cbound = 10 * (1e-7 + 1e-5 * cbone)
    check('frozen_Cbone_discrimination_and_Zzero', initial['Z_firstbone_inf'] == 0. and cbone > cbound, {'Cbone': cbone, 'frozen_threshold': cbound})
    check('immutable_reference_no_forecast_claim', initial['status'] == 'passed_native_initial_identities' and initial['reference_immutable'] and not initial['forecast_success'])
    check('qualification_native_target_gate', qualification['status'] == 'passed_new_native_pipeline_qualification' and qualification['tail_target_denied_before_freeze'] and not qualification['market_read'] and not qualification['Q3_read'] and runtime_check(qualification['runtime']))
    check('Ridge_3_frozen_cases', [x['case'] for x in qualification['Ridge']] == ['fullrank', 'constant_last_column', 'singular_duplicate_columns'] and all(0 <= x['max_pred_difference'] <= 1e-10 and 0 <= x['backward_error'] <= 1e-10 for x in qualification['Ridge']))
    for item in qualification['source_ast_receipt']:
        tree = ast.parse(read(item['source']).decode())
        nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == item['name']]
        check('original_AST:' + item['name'], len(nodes) == 1 and hashlib.sha256(ast.dump(nodes[0], include_attributes=False).encode()).hexdigest() == item['ast_sha256'])
    for name in ['continuous', 'signal', 'deadline']:
        check('per_state_same_initial_gate:' + name, obj(RUN / name / 'INITIAL_QUALIFICATION.json') == initial)

    cost = obj(RUN / 'cost/RESULTS.json')
    check('max_shape_cost_identity', cost['shape'] == {'B': 128, 'L': 480, 'H': 60, 'dtype': 'float32'} and runtime_check(cost['runtime']) and not cost['market_read'] and not cost['Q3_read'] and not cost['forecast_success'])
    check('all9_trainstep_finite_below30seconds', [x['arm'] for x in cost['arms']] == ['D', 'Z', 'C'] and all(len(x['trainstep_seconds']) == 3 and all(math.isfinite(t) and 0 < t <= 30 for t in x['trainstep_seconds']) for x in cost['arms']))
    check('all9_inference_costs_finite', all(len(x['inference_seconds']) == 3 and all(math.isfinite(t) and t > 0 for t in x['inference_seconds']) for x in cost['arms']))
    check('native_data_Ridge_costs_finite', all(math.isfinite(cost[k]) and cost[k] > 0 for k in ['batch_prepare_seconds', 'native_prefix1280_Ridge_seconds']) and 0 <= cost['Ridge_backward_error'] <= 1e-10)
    worker = read(BASE / 'code/matched_worker.py').decode()
    maximum_cost = next(n for n in ast.parse(worker).body if isinstance(n, ast.FunctionDef) and n.name == 'maximum_cost')
    maximum_source = ast.get_source_segment(worker, maximum_cost)
    check('cost_finite_guards_present_in_pinned_source', all(s in maximum_source for s in ['Maximumshape nonfinite training prediction', 'Maximumshape nonfinite training loss', 'Maximumshape missing/nonfinite gradients', 'Maximumshape nonfinite updated parameters', 'Maximumshape nonfinite inference output']))

    comparison = obj(RUN / 'comparison/PAYLOAD_COMPARISON.json')
    topkeys = ['schema', 'identity', 'state', 'scaler', 'learners', 'reference', 'reference_initial', 'baseline', 'optimizers', 'modes', 'numpy_rng', 'python_rng', 'torch_rng', 'controls', 'evaluation', 'fixture']
    check('full_payload_comparison_receipt', comparison['status'] == 'passed_independent_full_typed_payload_equality' and comparison['included_top_level_keys'] == topkeys and comparison['canonical_payload_sha256'] == '0ac9d28d2fb08ccd943a668890af24498fa4d40270c530ab8a4440ceff6ee58b')
    check('comparator_scope', all(comparison[k] is False for k in ['comparison_algorithm_imported_worker', 'market_read', 'Q3_read', 'forecast_success']))
    recoveries = comparison['recoveries']
    check('both_recovery_reports_present', [r['recovery'] for r in recoveries] == ['signal', 'deadline'])
    dtype_sizes = {'torch.float32': 4, 'torch.float64': 8, 'torch.int64': 8, 'torch.uint8': 1, 'torch.bool': 1}
    for recovery in recoveries:
        leaves = recovery['checks']; paths = [x['path'] for x in leaves]
        check('3240_unique_leaves:' + recovery['recovery'], recovery['passed'] and recovery['all_leaf_checks'] == len(leaves) == len(set(paths)) == 3240)
        check('all_leaf_kinds:' + recovery['recovery'], dict(Counter(x['kind'] for x in leaves)) == {'int': 675, 'str': 2236, 'bool': 23, 'NoneType': 8, 'float': 201, 'tensor': 97})
        check('every_top_payload_branch:' + recovery['recovery'], {p.split('/')[1] for p in paths} == set(topkeys))
        for arm in ['D', 'Z', 'C']:
            check('all6learner18Adam_tensors:' + recovery['recovery'] + ':' + arm, sum(x['kind'] == 'tensor' and x['path'].startswith('payload/learners/' + arm + '/') for x in leaves) == 6 and sum(x['kind'] == 'tensor' and x['path'].startswith('payload/optimizers/' + arm + '/') for x in leaves) == 18)
        tensors = [x for x in leaves if x['kind'] == 'tensor']
        check('tensor_shape_dtype_byte_metadata:' + recovery['recovery'], all(x['dtype'] in dtype_sizes and x['bytes'] == math.prod(x['shape']) * dtype_sizes[x['dtype']] for x in tensors))
    check('identical_recovery_leaf_paths_metadata', recoveries[0]['checks'] == recoveries[1]['checks'])
    comparator_source = read(BASE / 'code/compare_payloads.py').decode()
    imports = [n for n in ast.walk(ast.parse(comparator_source)) if isinstance(n, (ast.Import, ast.ImportFrom))]
    modules = [n.module if isinstance(n, ast.ImportFrom) else alias.name for n in imports for alias in ([None] if isinstance(n, ast.ImportFrom) else n.names)]
    check('comparison_algorithm_does_not_import_worker', not any('worker' in str(n) or 'native_core' in str(n) for n in modules))

    result = results[0]['results'][0]
    for predictor, score in result['predictors'].items():
        n, h = result['count'], result['horizon']
        recomputed = {'endpoint_MSE': score['endpoint_sse'] / n, 'path_MSE': score['path_sse'] / (n*h), 'bias': score['sum_error'] / (n*h), 'prediction_variance': score['sum_pred_square'] / (n*h) - (score['sum_pred'] / (n*h))**2}
        check('score_scalar_consistency:' + predictor, all(math.isfinite(score[k]) and math.isclose(v, score[k], rel_tol=1e-12, abs_tol=1e-18) for k, v in recomputed.items()))
    replay = obj(BASE / 'review/INDEPENDENT_NATIVE_REPLAY.json')
    check('root_native_replay_receipt_bound', replay['status'] == 'passed_independent_raw_synthetic_native_scaler_and_direct_Ridge_reconstruction' and replay['verifier_sha256'] == protocol['code_files'][str(BASE / 'review/independent_native_replay.py')] and replay['checks_passed'] == 11 and all(c['passed'] for c in replay['checks']) and replay['normal_equation_backward_error'] <= 1e-10 and not replay['worker_or_feature_helpers_imported'] and not replay['Torch_imported'] and not replay['market_arrays_or_archived_weights_read'] and not replay['Q3_read'] and not replay['predictive_success_claimed'])
    check('review_within_unchanged_deadline', time.time() < deadline)
    check('review_CPU26_stdlib_only', sorted(os.sched_getaffinity(0)) == [26])
    report = {
        'status': 'passed_independent_source_small_receipt_and_log_review_of_native_synthetic_execution',
        'reviewer': '/root/shared_gate_audit', 'at_UTC': datetime.now(timezone.utc).isoformat(),
        'affinity': sorted(os.sched_getaffinity(0)), 'protocol_sha256': protocol_sha,
        'rules_sha256': protocol['rules_sha256'], 'source_acceptance_sha256': protocol['source_acceptance_sha256'],
        'deadline_epoch': deadline, 'deadline_reset': False, 'checks_passed': len(checks), 'checks': checks,
        'input_inventory': inventory,
        'execution_chain': [{k: e[k] for k in ['mode', 'name', 'resume', 'pause_kind', 'pid', 'exit_code', 'started_epoch', 'finished_epoch']} for e in events],
        'payload_comparison': {'canonical_payload_sha256': comparison['canonical_payload_sha256'], 'leaves_per_recovery': 3240, 'tensors_per_recovery': 97, 'all16_top_keys_present': True, 'both_recovery_leaf_metadata_identical': True},
        'scope_limits': [
            'Reviewer used only Python standard library, source text, small JSON and logs on CPU26. No Torch/NumPy import, numerical model execution, PT/NPZ/NPY read or hash occurred in this review.',
            'Tensor equality is supported by pinned independently implemented comparator source and its completed JSON receipt; this reviewer did not re-read checkpoint tensor bytes. Comparator algorithm has no worker/native-core imports, but its parent dispatcher imports worker/native numeric modules.',
            'Deadline recovery injected an already-expired local deadline after common batch4, then exercised the real wall-clock save/exit75 branch; it did not wait for natural expiry of the future preparation deadline.',
            'Finite model/loss/gradient/parameter gates are source-verified and supported by cost child exit0; no arrays were independently recomputed by this reviewer.',
            'Root native scaler/direct-Ridge replay is linked as separately authored numerical receipt. This review independently checked only its SHA/source binding and small JSON results, plus scalar score arithmetic.',
            'Process audit hook and reported access fields are source/receipt evidence, not an operating-system sandbox or external proof of arbitrary process behavior.',
            'Synthetic qualification and checkpoint recovery establish engineering readiness only; no market method success, formal candidate admission, independent market validation or full Goal completion is claimed.',
            'C versus Z remains a total recipe intervention, not pure first-step causality. Market arrays remain unadmitted; a separate market source/input/admission/deadline freeze remains required.'
        ],
        'market_admission': False, 'market_arrays_read_by_reviewer': False, 'archived_weights_read_by_reviewer': False,
        'Q3_read': False, 'forecast_success': False, 'full_goal_achieved': False,
        'unresolved_P1': []
    }
    output = BASE / 'review/INDEPENDENT_RESULT_REVIEW.json'
    if output.exists():
        raise RuntimeError('Refusing to overwrite existing review')
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'path': str(output), 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(), 'checks_passed': len(checks), 'status': report['status']}, indent=2))

if __name__ == '__main__':
    main()
