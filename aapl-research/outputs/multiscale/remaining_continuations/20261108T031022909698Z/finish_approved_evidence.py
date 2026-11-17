"""Wait for the approved original folds, then verify the full 100-fold study.

This process performs only scoring, metadata checks, predictor restoration and
package export. It has no training or process-signalling entry point.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from multiscale.historical_continuation import digest, matched, read, save
from multiscale.historical_coverage import snapshot
from multiscale.historical_restore_check import fold_worker_command

ROOT = Path('/home/ayaka/wslcode/aapl-research')
OUT = ROOT / 'outputs/multiscale'
RUN = Path(__file__).resolve().parent
DEST = OUT / 'completed_core_evidence/20261108_approved_continuation'
PRIOR = OUT / 'private_model_restoration/20261108_original_budget_endpoint/COMBINED_ENDPOINT_RESTORATION.json'
ADMISSION = OUT / 'admission/alpaca_20261107T052228573242Z'


def command(arguments, label, timeout=1200, environment=None):
    log = DEST / (label + '.log')
    save(DEST / 'PIPELINE_STATE.json', {'phase': label, 'at_utc': datetime.now(timezone.utc).isoformat(),
                                      'training_started': False, 'command': arguments})
    result = subprocess.run(arguments, cwd=ROOT, capture_output=True, text=True,
                            env=environment or os.environ, timeout=timeout)
    log.write_text(result.stdout + result.stderr, encoding='utf8')
    if result.returncode:
        raise RuntimeError(label + ' failed; see ' + str(log))


def key(item):
    return tuple(item[k] for k in ('profile', 'horizon', 'target', 'fold'))


def restore_union(coverage_path):
    coverage = read(coverage_path)
    prior = read(PRIOR)
    old_coverage = read(prior['coverage'])
    assert digest(prior['coverage']) == prior['coverage_sha256']
    old_rows = {key(row): row for row in old_coverage['folds'] if row['status'] == 'completed'}
    old_checks = {key(row): row for row in prior['checks']}
    assert set(old_checks) == set(old_rows) and len(old_checks) == 85
    folder = DEST / 'restoration'
    folder.mkdir()
    (folder / 'fold_checks').mkdir()
    checks = []
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for row in coverage['folds']:
        identity = key(row)
        if identity in old_checks:
            assert old_rows[identity]['evidence'] == row['evidence']
            item = old_checks[identity]
            path = Path(item['check'])
            assert digest(path) == item['sha256']
            reused = True
        else:
            path = folder / 'fold_checks' / ('_'.join(map(str, identity)) + '.json')
            command(fold_worker_command(row, path, ADMISSION), 'restore_' + path.stem, timeout=240, environment=env)
            reused = False
        evidence = read(path)
        assert evidence['Q3_values_read'] is False and evidence['training_started'] is False
        assert evidence.get('saved_target_arrays_opened') is not True and len(evidence['models']) == 4
        assert all(m['exact_array_equal'] and m['max_absolute_difference'] == 0 for m in evidence['models'])
        checks.append({**{k: row[k] for k in ('profile', 'horizon', 'target', 'fold')},
                       'check': str(path), 'sha256': digest(path), 'reused_prior_exact_check': reused,
                       'origin_model_paths': sum(m['origins_checked'] for m in evidence['models'])})
    assert len(checks) == 100
    result = {'status': 'all100_original_folds_saved_neural_predictors_exactly_reproduced',
              'at_utc': datetime.now(timezone.utc).isoformat(), 'coverage': str(coverage_path),
              'coverage_sha256': digest(coverage_path), 'prior_union': str(PRIOR), 'prior_union_sha256': digest(PRIOR),
              'completed_folds': 100, 'neural_models': 400, 'all_arrays_exact': True,
              'max_absolute_difference': 0.0, 'origin_model_paths': sum(c['origin_model_paths'] for c in checks),
              'reused_exact_fold_checks': 85, 'new_fresh_fold_processes': 15, 'checks': checks,
              'saved_target_arrays_opened': False, 'new_training_started': False, 'Q3_values_read': False,
              'target_access_evidence': 'Inspected original restore functions and prior hash-bound union; standard per-fold schema omits this flag.'}
    save(folder / 'ALL100_RESTORATION.json', result)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1' and os.sched_getaffinity(0) == {24, 26}
    assert not DEST.exists(), 'Use existing pipeline state instead of repeating work'
    DEST.mkdir(parents=True)
    state = read(RUN / 'STATE.json')
    save(DEST / 'PIPELINE_STATE.json', {'phase': 'waiting_for_approved_training', 'run': str(RUN),
                                      'at_utc': datetime.now(timezone.utc).isoformat(), 'training_started': False})
    while state['phase'] not in ('completed', 'paused_preserved'):
        if not matched(state['controller']):
            time.sleep(2)
            state = read(RUN / 'STATE.json')
            if state['phase'] not in ('completed', 'paused_preserved'):
                raise RuntimeError('Training controller disappeared; inspect the independent watchdog')
            break
        time.sleep(10)
        state = read(RUN / 'STATE.json')
    if state['phase'] != 'completed':
        save(DEST / 'PIPELINE_STATE.json', {'phase': 'training_paused_no_full_study_claim',
                                          'training_state': str(RUN / 'STATE.json'),
                                          'at_utc': datetime.now(timezone.utc).isoformat(), 'training_started': False})
        return
    coverage = snapshot()
    assert coverage['completed_folds'] == 100 and coverage['five_fold_tasks'] == 20
    assert all(r['status'] == 'completed' for r in coverage['folds'])
    coverage_path = DEST / 'COVERAGE.json'
    save(coverage_path, coverage)
    python = sys.executable
    command([python, '-m', 'multiscale.historical_results', '--coverage', str(coverage_path),
             '--output', str(DEST / 'results')], 'compile_all100_results')
    command([python, '-m', 'multiscale.historical_design_audit', '--coverage', str(coverage_path),
             '--destination', str(DEST / 'design')], 'audit_all100_design')
    restore_union(coverage_path)
    package = DEST / 'inference_package'
    command([python, '-m', 'multiscale.study_package', 'export', '--coverage', str(coverage_path),
             '--permission', str(OUT / 'permission_confirmations/20261108_ALPACA_KIBOT_USER_CONFIRMED.json'),
             '--source-manifest', str(OUT / 'data_provenance/20261108_CURRENT_DEVELOPMENT_SOURCE_MANIFEST.json'),
             '--admission', str(ADMISSION), '--output', str(package)], 'export_full_package')
    command([python, '-m', 'multiscale.verify_study_package', '--package', str(package),
             '--registry-sha256', digest(package / 'REGISTRY.json'), '--admission', str(ADMISSION),
             '--output', str(DEST / 'package_audit')], 'audit_full_package')
    results = read(DEST / 'results/RESULTS.json')
    design = read(DEST / 'design/DESIGN_TIMING_AUDIT.json')
    restoration = read(DEST / 'restoration/ALL100_RESTORATION.json')
    registry = read(package / 'REGISTRY.json')
    audit = read(DEST / 'package_audit/PACKAGE_AUDIT.json')
    assert results['status'] == 'complete_historical_development_results' and results['completed_folds'] == 100
    assert results['complete_tasks'] == 20 and design['completed_saved_fold_metadata_verified'] == 100
    assert registry['core_comparison_complete'] and registry['five_fold_tasks'] == 20
    assert sum(len(t['completed_folds']) for t in registry['tasks']) == 100
    assert audit['core_comparison_complete'] and audit['tasks_verified'] == 20 and audit['models_verified'] == 160
    receipt = {'status': 'all100_evidence_ready_for_final_manuscript', 'at_utc': datetime.now(timezone.utc).isoformat(),
               'completed_folds': 100, 'complete_tasks': 20, 'restored_neural_predictors': 400,
               'restored_origin_model_paths': restoration['origin_model_paths'],
               'run': str(RUN), 'training_approval_sha256': digest(RUN / 'APPROVAL.json'),
               'artifacts': {relative: digest(DEST / relative) for relative in
                    ('COVERAGE.json', 'results/RESULTS.json', 'design/DESIGN_TIMING_AUDIT.json',
                     'restoration/ALL100_RESTORATION.json', 'inference_package/REGISTRY.json', 'package_audit/PACKAGE_AUDIT.json')},
               'new_training_started_by_pipeline': False, 'Q3_values_read': False,
               'manuscript_complete': False, 'script_sha256': digest(__file__)}
    save(DEST / 'ALL100_EVIDENCE_RECEIPT.json', receipt)
    save(DEST / 'PIPELINE_STATE.json', {'phase': receipt['status'], 'receipt': str(DEST / 'ALL100_EVIDENCE_RECEIPT.json'),
                                      'at_utc': receipt['at_utc'], 'training_started': False})
    (OUT / 'LATEST_COMPLETED_CORE_EVIDENCE.txt').write_text(str(DEST) + '\n')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if DEST.exists():
            save(DEST / 'PIPELINE_ERROR.json', {'error': type(exc).__name__ + ': ' + str(exc),
                'at_utc': datetime.now(timezone.utc).isoformat(), 'training_started': False})
        raise
