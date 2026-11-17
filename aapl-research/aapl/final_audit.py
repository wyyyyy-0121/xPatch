"""Audit admission or completed execution against the user's revised implementation plan."""
import argparse
import json

import numpy as np

from .common import ROOT, digest, read_json, runtime, utc_now, write_json
from .runner import require_lock
from .lock_protocol import MODELS
from .search_space import generate


def audit(completion=False):
    runtime()
    protocol = require_lock()
    data = read_json(ROOT / 'outputs/data_admission.json')
    model = read_json(ROOT / 'outputs/model_admission.json')
    perf = read_json(ROOT / 'outputs/performance_admission.json')
    manifest = read_json(ROOT / 'holdout_manifest.json')
    checks = {}
    checks['current_user_plan_identity'] = digest(ROOT / 'IMPLEMENTATION_PLAN.md') == protocol['authoritative_plan']['sha256']
    checks['development_source_identity'] = digest(protocol['source_csv_read_only']) == data['source_sha256'] == protocol['source_csv_expected_sha256']
    checks['raw_receipts_all_fields_verified'] = data['raw_receipt_verification']['pages_verified'] == 134 and data['raw_receipt_verification']['rows_verified'] == data['all_rows']
    checks['development_calendar_and_labels'] = data['regular_rows'] == 535620 and all(data['checks'].values())
    with np.load(ROOT / 'outputs/common_origins.npz') as origin:
        checks['exact12minute_target_endpoint'] = len(origin['row']) == data['eligible_origins'] and bool(np.all(origin['label_end_ns'] - origin['bar_start_ns'] == 720_000_000_000))
    checks['development_splits_identity'] = digest(ROOT / 'outputs/development_splits.json') == data['development_splits_sha256']
    split = read_json(ROOT / 'outputs/development_splits.json')
    checks['planned_three_validation_blocks'] = [i['quarter'] for i in split['inner']] == ['2024Q4', '2025Q2', '2025Q4']
    checks['final_fit_and_june_calibration'] = max(split['final_fit_dates']) == '2026-05-29' and len(split['calibration_dates']) == 20
    checks['sealed_manifest_identity'] = digest(ROOT / 'holdout_manifest.json') == protocol['holdout_manifest_sha256']
    sealed_path = ROOT / 'holdout_sealed' / manifest['csv']
    checks['sealed_data_hash_and_readonly_mode'] = digest(sealed_path) == manifest['csv_sha256'] and sealed_path.stat().st_mode & 0o222 == 0
    checks['sealed_calendar_verified'] = manifest['calendar_sessions'] == manifest['effective_sessions'] == 64 and manifest['eligible_origins'] == 8832
    checks['legacy_access_audit_passed'] = read_json(ROOT / 'outputs/holdout_exposure_audit.json')['passed']
    checks['native_ema_values_and_gradients'] = model['checks']['ema']['cases'] == 24 and model['checks']['ema']['gradients_checked']
    for name in ('xpatch', 'patchtst', 'chronos2_lora'):
        row = model['checks'][name]
        checks[name + '_recovery'] = row['mid_epoch_resume_identical_losses_weights_predictions'] and row['fresh_model_optimizer_reconstruction_verified']
        checks[name + '_identity_rejection'] = row['corrupted_bytes_rejected'] and row['wrong_signature_rejected']
    lora = model['checks']['chronos2_lora']
    checks['real_lora_with_frozen_base'] = lora['changed_trainable_tensor_count'] > 0 and lora['all_frozen_base_tensors_unchanged'] and lora['weight_identity']['trainable_parameters'] < lora['weight_identity']['all_parameters'] / 10
    checks['chronos_causality_and_isolation'] = lora['causal_future_row_perturbation'] and lora['label_changes_cannot_change_same_forward_predictions'] and lora['origin_order_and_isolated_origin_invariance']
    for name in MODELS:
        for mode in ('small', 'maximum'):
            entry = perf['profiles'][name][mode]
            path = ROOT / entry['path']
            row = read_json(path)
            checks[f'{name}_{mode}_passed_with_identity'] = digest(path) == entry['sha256'] and row['passed'] and row['resources']['passed']
            resource = row['resources']
            checks[f'{name}_{mode}_resources'] = resource['peak_rss_gib'] <= 8 and resource['min_available_ram_gib'] >= 4 and resource['max_affinity_cpus'] <= 18 and resource['swap_growth_mib'] <= 8
        row = read_json(ROOT / perf['profiles'][name]['maximum']['path'])
        checks[name + '_current_maximum_N'] = row['training_samples'] == row['max_planned_training_samples'] == 185886
        if name in ('xpatch', 'patchtst'):
            checks[name + '_sustained10minutes'] = row['training_wall_seconds'] >= 600
        if name == 'chronos2_lora':
            checks['lora900updates_and_checkpoint'] = row['updates'] == 900 and row['saved_model_prediction_verified']
    checks['all_shape_timings_identity'] = digest(ROOT / 'outputs/shape_profiles.json') == perf['all_supervised_shapes_sha256']
    trials = read_json(ROOT / protocol['trial_list_file'])
    generated = generate()
    checks['search_lists_predeclared_subset'] = all(all(t in generated[name] for t in rows) for name, rows in trials.items())
    checks['supervised_losses_balanced'] = all(sum(t['loss'] == 'step_mse' for t in trials[name]) * 2 == len(trials[name]) for name in ('xpatch', 'patchtst'))
    checks['local24h_budget_with50pct_margin'] = protocol['measured_budget']['safety_factor'] == 1.5 and protocol['measured_budget']['gpu_hours_with_margin'] <= 24 and protocol['measured_budget']['wall_hours_with_margin_and_overhead'] <= 48
    checks['storage_budget'] = perf['storage_projection']['gib'] <= 20
    checks['optimization_and_parallel_admission'] = read_json(ROOT / 'outputs/optimization_admission.json')['passed'] and read_json(ROOT / 'outputs/parallel_admission.json')['passed']
    checks['bounded_compile_test_finished'] = read_json(ROOT / 'outputs/compile_admission.json')['completed']
    checks['executor_smoke_passed'] = read_json(ROOT / 'outputs/executor_smoke.json')['passed']
    checks['pooled_executor_promotion_pruning_resume'] = read_json(ROOT / 'outputs/parallel_executor_smoke.json')['passed']
    checks['pipeline_prediction_smoke_passed'] = read_json(ROOT / 'outputs/pipeline_smoke.json')['passed']
    tests = (ROOT / 'outputs/core_tests.log').read_text()
    checks['core_tests_passed'] = 'passed' in tests and 'FAILED' not in tests and 'ERROR' not in tests
    checks['old_inspected_files_unchanged'] = all(digest(r['path']) == r['sha256'] for r in read_json(ROOT / 'sources/evidence_manifest.json')['legacy_read_only_files'])
    if not completion:
        checks['locked_before_formal_scores'] = not (ROOT / 'outputs/formal').exists()
        accesses = [read_json(p) for p in (ROOT / 'outputs/holdout_access').glob('*.json')]
        checks['holdout_not_opened_to_models'] = all(r['kind'] != 'authorized_read' for r in accesses)
    else:
        summary = read_json(ROOT / 'outputs/formal/final_summary.json')
        checks['completed_fixed_research_path'] = summary['status'] in ('completed_development_negative', 'completed_sealed_test')
        checks['final_summary_protocol_identity'] = summary['protocol_sha256'] == digest(ROOT / 'protocol_locked.json')
        checks['independent_metric_verification'] = read_json(ROOT / 'outputs/formal/independent_verification.json')['passed']
        checks['research_conclusion_delivered'] = (ROOT / 'RESULTS.md').exists()
        usage = read_json(ROOT / 'outputs/formal/usage.json')
        checks['actual_resource_budgets_respected'] = usage['physical_gpu_busy_hours'] <= 24 and usage['wall_hours'] <= 48 and usage['output_artifact_gib'] <= 20
        resources = [read_json(p) for p in (ROOT / 'outputs/formal').rglob('*resources.json')]
        checks['formal_resource_telemetry_passed'] = bool(resources) and all(r.get('passed', False) for r in resources)
        checks['all_prediction_receipt_resources_passed'] = all(read_json(p)['resources']['passed'] for p in (ROOT / 'outputs/formal').glob('*/*/seed*/receipt.json'))
    report = {'passed': all(checks.values()), 'completed_at_utc': utc_now(), 'checks': checks,
              'scope': 'completed_plan' if completion else 'P0_P5_admission',
              'protocol_sha256': digest(ROOT / 'protocol_locked.json')}
    write_json(ROOT / ('outputs/completion_audit.json' if completion else 'outputs/admission_audit.json'), report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report['passed']:
        raise SystemExit('Implementation-plan audit failed')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--completion', action='store_true')
    args = parser.parse_args()
    audit(args.completion)


if __name__ == '__main__':
    main()
