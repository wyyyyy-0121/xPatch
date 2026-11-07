"""Check saved neural forecasts in fresh CPU processes, using past-only inputs.

This audits the completed tasks in a saved historical coverage inventory. Run
with no visible GPU and at most two logical CPUs. It reads each predictor's
saved validation origins and past input windows, without training or opening
saved target arrays. The original prediction batch size is retained.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/multiscale'
SOURCE = OUT / 'admission/alpaca_20261107T052228573242Z'
NEURAL = ('mlp', 'gru', 'lstm', 'xpatch')
RTOL, ATOL = 1e-5, 1e-8


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def checked_source(relative, expected):
    path = (ROOT / relative).resolve()
    require(path.is_relative_to(ROOT.resolve()) and path.is_file(),
            'Recorded training source is outside the project or missing: ' + relative)
    require(digest(path) == expected, 'Recorded training source changed: ' + relative)


def check_fold(folder, expected_receipt, source=SOURCE):
    import numpy as np
    import torch
    from multiscale.bars import Bars
    from multiscale.development import bars_identity
    from multiscale.features import CloseWindows
    from multiscale.matrix_executor import read_bars
    from multiscale.spec import PROFILES
    from multiscale.training import digest_indices, predict_best

    torch.set_num_threads(min(2, len(os.sched_getaffinity(0))))
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    require(1 <= len(os.sched_getaffinity(0)) <= 2, 'Audit requires at most two logical CPUs')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '-1', 'GPU visibility must be disabled')
    folder = Path(folder).resolve()
    require(folder.is_relative_to(OUT), 'Fold is outside the research output tree')
    receipt_path = folder / 'fold_receipt.json'
    require(digest(receipt_path) == expected_receipt, 'Receipt changed')
    receipt = load(receipt_path)
    require(receipt.get('status') == 'completed' and receipt.get('Q3_values_read') is False and
            receipt.get('independent_test') is False, 'Not a completed historical development fold')
    fields, cfg = receipt['identity_fields'], receipt['config']
    require(hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest() == receipt['fold_identity'],
            'Fold identity is invalid')
    for relative, sha in receipt['source_sha256'].items():
        checked_source(relative, sha)
    source = Path(source).resolve()
    require(source.is_relative_to(OUT), 'Admission is outside the research output tree')
    admission = load(source / 'admission.json')
    require(admission['Q3_values_read'] is False, 'Source cutoff record is invalid')
    profile = next(p for p in PROFILES if p.name == cfg['profile'])
    bars = read_bars(source, admission, profile.frequency)
    require(bars_identity(bars) == fields['bars_sha256'], 'Source bars differ from the trained fold')
    model_results = []
    common_origins = None
    for name in NEURAL:
        record = receipt['models'][name]
        checkpoint = folder / record['checkpoint_dir'] / 'best.pt'
        prediction_path = folder / ('prediction_' + name + '.npz')
        require(digest(checkpoint) == record['best_checkpoint_sha256'], 'Best checkpoint changed')
        require(digest(prediction_path) == record['prediction_sha256'], 'Saved predictions changed')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        require(saved['identity'] == record['training_identity'], 'Training identity differs')
        for key in ('lookback', 'horizon', 'target', 'max_epochs', 'patience', 'batch_size',
                    'learning_rate', 'seed'):
            require(saved['config'][key] == cfg[key], 'Checkpoint configuration differs: ' + key)
        require(saved['config']['model'] == name, 'Checkpoint architecture differs')
        with np.load(prediction_path, allow_pickle=False) as stored:
            require(set(stored.files) == {'origins', 'decision_ns', 'prediction'},
                    'Unexpected saved-prediction fields')
            origins, decisions = stored['origins'], stored['decision_ns']
            expected = np.asarray(stored['prediction'], dtype=np.float64)
        require(len(origins) == receipt['validation_origins'] and
                digest_indices(origins) == fields['validation_indices_sha256'] and
                np.array_equal(decisions, bars.end_ns[origins]), 'Saved origins or decisions differ')
        if common_origins is None:
            common_origins = origins
        require(np.array_equal(origins, common_origins), 'Neural models use different origins')
        # Preserve the original prediction batch shape. Each input row still
        # ends at its own decision; no target array is opened by this audit.
        end = int(origins[-1]) + 1
        historical = Bars(bars.frequency, bars.start_ns[:end], bars.end_ns[:end],
                          bars.close[:end], bars.observed[:end], bars.session_date[:end])
        require(int(historical.end_ns[-1]) == int(decisions[-1]),
                'Historical input extends beyond the final validation decision')
        window_indices = origins[:, None] + np.arange(1 - cfg['lookback'], 1)
        require(np.all(historical.end_ns[window_indices] <= decisions[:, None]),
                'An input window contains a post-decision bar')
        windows = CloseWindows(historical, cfg['lookback'], fields['actions'])
        actual = predict_best(checkpoint, windows, origins, record['training_identity'],
                              device='cpu', batch_size=512)
        require(actual.shape == expected.shape and np.isfinite(actual).all() and
                np.allclose(actual, expected, rtol=RTOL, atol=ATOL),
                f'Restored {name} forecasts differ beyond the fixed float32 tolerance')
        require(digest(checkpoint) == record['best_checkpoint_sha256'], 'Audit changed a checkpoint')
        model_results.append({'model': name, 'checkpoint_sha256': digest(checkpoint),
                              'training_identity': record['training_identity'],
                              'origins_checked': len(origins), 'horizon': cfg['horizon'],
                              'max_absolute_difference': float(np.max(np.abs(actual - expected))),
                              'exact_array_equal': bool(np.array_equal(actual, expected)),
                              'same_saved_forecasts_within_tolerance': True,
                              'post_decision_values_used_in_input_windows': False})
    require(digest(receipt_path) == expected_receipt, 'Receipt changed during the audit')
    return {'status': 'saved_neural_restore_checked', 'fold_directory': str(folder),
            'receipt_sha256': expected_receipt, 'profile': cfg['profile'],
            'horizon': cfg['horizon'], 'target': cfg['target'],
            'validation_origins_checked': len(common_origins), 'models': model_results,
            'rtol': RTOL, 'atol': ATOL, 'training_started': False,
            'Q3_values_read': False, 'device': 'cpu', 'logical_cpu_set': sorted(os.sched_getaffinity(0)),
            'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__, 'torch': str(torch.__version__)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coverage', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fold', type=Path)
    parser.add_argument('--receipt-sha256')
    parser.add_argument('--admission', type=Path, default=SOURCE,
                        help='Original cutoff-limited admission directory')
    args = parser.parse_args()
    if args.fold:
        write_new(args.output, check_fold(args.fold, args.receipt_sha256, args.admission))
        return
    require(args.coverage is not None, 'A coverage snapshot is required')
    coverage = load(args.coverage)
    require(coverage.get('status') == 'private_verified_coverage_snapshot',
            'Expected a verified historical coverage snapshot')
    keys = [(r['profile'], r['horizon'], r['target'], r['fold']) for r in coverage['folds']]
    from .historical_coverage import TASKS
    require(len(keys) == len(set(keys)) and
            set(keys) == {(*task, fold) for task in TASKS for fold in range(1, 6)},
            'Coverage task-fold inventory differs')
    counts = Counter((r['profile'], r['horizon'], r['target']) for r in coverage['folds']
                     if r['status'] == 'completed')
    rows = [r for r in coverage['folds'] if r['status'] == 'completed' and
            counts[(r['profile'], r['horizon'], r['target'])] == 5]
    require(len(rows) == 5 * coverage['five_fold_tasks'], 'Complete-task coverage differs')
    args.output.mkdir(parents=True, exist_ok=False)
    checks = args.output / 'fold_checks'
    checks.mkdir()
    start = time.monotonic()
    results = []
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONPATH=str(ROOT))
    for row in rows:
        evidence = row['evidence']
        require('fold_directory' in evidence, 'A completed task lacks a schema-2 fold')
        name = f"{row['profile']}_h{row['horizon']}_{row['target']}_f{row['fold']}.json"
        path = checks / name
        command = [sys.executable, '-m', 'multiscale.historical_restore_check', '--fold', evidence['fold_directory'],
                   '--receipt-sha256', evidence['receipt_sha256'], '--output', str(path),
                   '--admission', str(args.admission.resolve())]
        child = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
        require(child.returncode == 0, 'Fresh-process restore failed: ' + name + '\n' + child.stderr[-3000:])
        results.append({'fold': row['fold'], 'check': str(path.resolve()),
                        'check_sha256': digest(path), 'result': load(path)})
        print(json.dumps({'stage': 'saved_fold_restored', 'folds_checked': len(results),
                          'planned': len(rows), 'task': name}), flush=True)
    payload = {'status': 'completed_task_saved_neural_restoration_verified',
               'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'coverage': str(args.coverage.resolve()), 'coverage_sha256': digest(args.coverage),
               'script_sha256': digest(Path(__file__)), 'complete_tasks_checked': coverage['five_fold_tasks'],
               'folds_checked': len(results), 'fresh_processes': len(results),
               'model_restorations_checked': len(results) * len(NEURAL),
               'origin_model_paths_checked': sum(m['origins_checked'] for r in results for m in r['result']['models']),
               'selection': 'all validation origins in every fold of each completed task; original batch size 512',
               'all_saved_predictions_exactly_reproduced': all(m['exact_array_equal']
                   for r in results for m in r['result']['models']),
               'rtol': RTOL, 'atol': ATOL, 'results': results, 'training_started': False,
               'Q3_values_read': False, 'wall_seconds': time.monotonic() - start,
               'limitations': ['Only tasks with all five completed folds are included.',
                               'The production inference packaging gate and interrupted-training recovery are separate checks.']}
    path = args.output / 'RESTORATION_AUDIT.json'
    write_new(path, payload)
    print(json.dumps({k: payload[k] for k in ('status', 'complete_tasks_checked', 'folds_checked',
                                             'model_restorations_checked', 'origin_model_paths_checked', 'wall_seconds')}
                     | {'path': str(path.resolve())}), flush=True)


if __name__ == '__main__':
    main()
