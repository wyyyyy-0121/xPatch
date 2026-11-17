"""Reuse hash-matched fold checks and restore only newly completed folds."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from multiscale.historical_coverage import digest, read_json, require, snapshot
from multiscale.historical_results import ADMISSION, checked_coverage
from multiscale.historical_restore_check import fold_worker_command


ROOT = Path('/home/ayaka/wslcode/aapl-research')
OUT = ROOT / 'outputs/multiscale'
PRIOR = OUT / 'private_model_restoration/20261108_additional_completed_minute_folds/COMBINED_78_FOLD_RESTORATION.json'
COVERAGE = OUT / 'private_coverage_verified/20261108_timing_design_snapshot.json'
DESTINATION = OUT / 'private_model_restoration/20261108_extend_81_fold_snapshot_verified'


def key(row): return tuple(row[k] for k in ('profile', 'horizon', 'target', 'fold'))
def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    start = time.monotonic()
    coverage, prior = read_json(COVERAGE), read_json(PRIOR)
    rows = [r for r in checked_coverage(coverage, snapshot(), allow_partial=True) if r['status'] == 'completed']
    previous_coverage = Path(prior['coverage'])
    require(digest(previous_coverage) == prior['coverage_sha256'], 'Prior coverage changed')
    old_rows = {key(r): r for r in read_json(previous_coverage)['folds'] if r['status'] == 'completed'}
    old_checks = {key(r): r for r in prior['checks']}
    require(set(old_rows) == set(old_checks) and len(old_checks) == 78 and
            prior['saved_target_arrays_opened'] is False, 'Prior union is incomplete')
    DESTINATION.mkdir(parents=True, exist_ok=False)
    (DESTINATION / 'fold_checks').mkdir()
    write(DESTINATION / 'COVERAGE.json', coverage)
    checks, fresh = [], 0
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONPATH=str(ROOT))
    for row in rows:
        identity = key(row)
        if identity in old_checks:
            require(old_rows[identity]['evidence'] == row['evidence'], 'Prior fold evidence changed')
            item = old_checks[identity]
            path = Path(item['check'])
            require(digest(path) == item['sha256'], 'Prior fold check changed')
            reused = True
        else:
            path = DESTINATION / 'fold_checks' / ('_'.join(map(str, identity)) + '.json')
            result = subprocess.run(fold_worker_command(row, path, ADMISSION), cwd=ROOT,
                                    env=environment, capture_output=True, text=True, timeout=180)
            require(result.returncode == 0, 'New restoration failed: ' + result.stderr[-3000:])
            fresh += 1
            reused = False
            print(json.dumps({'stage': 'new_completed_fold_restored', 'identity': identity,
                              'fresh_fold_processes': fresh}), flush=True)
        evidence = read_json(path)
        require(evidence.get('Q3_values_read') is False and evidence.get('training_started') is False and
                evidence.get('saved_target_arrays_opened') is not True and len(evidence['models']) == 4 and
                evidence['status'] in ('saved_neural_restore_checked', 'first_minute_fold_saved_neural_restore_checked'),
                'Fold audit scope differs')
        require(all(m['exact_array_equal'] and m['max_absolute_difference'] == 0 for m in evidence['models']),
                'Fold predictions did not reproduce exactly')
        checks.append({**dict(zip(('profile', 'horizon', 'target', 'fold'), identity)),
                       'check': str(path), 'sha256': digest(path), 'reused_prior_exact_check': reused,
                       'origin_model_paths': sum(m['origins_checked'] for m in evidence['models'])})
    require(len(checks) == coverage['completed_folds'] == 81, 'Union misses a completed fold')
    report = {'status': 'all_completed_folds_at_81_fold_snapshot_saved_predictors_exactly_reproduced',
              'created_at_utc': datetime.now(timezone.utc).isoformat(), 'coverage': str(COVERAGE),
              'coverage_sha256': digest(COVERAGE), 'prior_union': str(PRIOR), 'prior_union_sha256': digest(PRIOR),
              'completed_folds': len(checks), 'neural_models': len(checks) * 4,
              'origin_model_paths': sum(r['origin_model_paths'] for r in checks),
              'all_arrays_exact': True, 'max_absolute_difference': 0.0, 'checks': checks,
              'new_fresh_fold_processes': fresh, 'reused_exact_fold_checks': len(checks) - fresh,
              'new_training_started': False, 'saved_target_arrays_opened': False, 'Q3_values_read': False,
              'target_access_evidence': 'Original union and inspected restore function; standard per-fold schema omits this flag.',
              'elapsed_seconds': time.monotonic() - start, 'script_sha256': digest(Path(__file__)),
              'limitations': ['Covers only the completed folds at the fixed 81-fold snapshot.',
                              'Does not complete unfinished folds or establish independent forecast advantage.']}
    write(DESTINATION / 'COMBINED_81_FOLD_RESTORATION.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'checks'}), flush=True)


if __name__ == '__main__': main()
