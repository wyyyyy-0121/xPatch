"""Recompute detailed private development metrics from already-saved predictions.

No model is fitted and no protected-period dataset is read. Results are internal
historical development evidence, not independent confirmation or publication admission.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path('/home/ayaka/wslcode/aapl-research')
OUT = ROOT / 'outputs/multiscale'
PARENT = OUT / 'private_working_manuscripts/20261107T162317831116Z'
THREE = OUT / 'private_three_scale_development/20261107T134128870773Z'
ANNUAL = OUT / 'private_annual_short_development/20261107T143734746737Z'
MODELS = ('last_close', 'drift', 'seasonal', 'ridge', 'mlp', 'gru', 'lstm', 'xpatch')
LABELS = {'last_close': 'Persistence / zero', 'drift': 'Drift', 'seasonal': 'Seasonal',
          'ridge': 'Ridge', 'mlp': 'MLP', 'gru': 'GRU', 'lstm': 'LSTM', 'xpatch': 'xPatch'}
NY = ZoneInfo('America/New_York')
CUTOFF_NS = int(datetime(2026, 7, 1, tzinfo=NY).timestamp()) * 1_000_000_000


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def near(a, b):
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=1e-16)


def local_time(ns):
    return datetime.fromtimestamp(int(ns) / 1_000_000_000, timezone.utc).astimezone(NY).isoformat()


def source_rows(path):
    with Path(path).open(encoding='utf8', newline='') as f:
        return list(csv.DictReader(f))


parent_audit = load(PARENT / 'PRIVATE_MANUSCRIPT_EVIDENCE_AUDIT.json')
parent_md = PARENT / 'PRIVATE_WORKING_MANUSCRIPT_EN.md'
require(digest(parent_md) == parent_audit['manuscript_sha256'], 'Parent manuscript hash differs')
require(parent_audit['checks']['completed_task_folds'] == 68 and parent_audit['independent_test'] is False,
        'Unexpected parent evidence scope')

existing = source_rows(THREE / 'PRIVATE_THREE_SCALE_RESULTS.csv') + source_rows(ANNUAL / 'PRIVATE_ANNUAL_RESULTS.csv')
existing_by_key = {(x['profile'], int(x['horizon']), x['target'], x['model']): x for x in existing}
require(len(existing_by_key) == 96, 'Unexpected pooled result inventory')
fold_rows = []
saved_artifact_count = 0
folds_seen = set()

for root in (THREE, ANNUAL):
    queue = load(root / 'PRIVATE_QUEUE_STATE.json')
    for job_name, job in sorted(queue['jobs'].items()):
        require(job['status'] == 'completed', 'Queue contains unfinished fold')
        folder = Path(job['fold_directory'])
        receipt_path = folder / 'fold_receipt.json'
        require(digest(receipt_path) == job['receipt_sha256'], 'Receipt mismatch: ' + job_name)
        rec = load(receipt_path)
        cfg = rec['config']
        match = re.search(r'_f(\d+)$', job_name)
        require(match is not None, 'Invalid fold key')
        fold = int(match.group(1))
        key = (cfg['profile'], int(cfg['horizon']), cfg['target'], fold)
        require(key not in folds_seen, 'Duplicate fold')
        folds_seen.add(key)
        require(rec['status'] == 'completed' and rec['phase'] == 'development' and
                rec['independent_test'] is False and rec['Q3_values_read'] is False, 'Invalid fold scope')
        require(set(rec['models']) == set(MODELS), 'Incomplete model set')
        target_path = folder / 'targets.npz'
        require(digest(target_path) == rec['targets_sha256'], 'Target artifact mismatch')
        saved_artifact_count += 1
        with np.load(target_path, allow_pickle=False) as z:
            require(set(z.files) == {'origins', 'actual', 'anchor', 'decision_ns', 'target_completion_ns'},
                    'Unexpected target archive')
            decisions = np.asarray(z['decision_ns'], dtype=np.int64)
            completions = np.asarray(z['target_completion_ns'], dtype=np.int64)
            # Validate the protected boundary before loading market values from this archive.
            require(len(decisions) > 0 and int(completions.max()) < CUTOFF_NS and
                    np.all(completions > decisions), 'A saved label crosses the cutoff')
            origin = np.asarray(z['origins'])
            actual = np.asarray(z['actual'], dtype=np.float64)
            anchor = np.asarray(z['anchor'], dtype=np.float64)
        n, h = actual.shape
        require(n == rec['validation_origins'] and h == cfg['horizon'] and
                all(x.shape == (n,) for x in (decisions, completions, origin, anchor)),
                'Saved array dimensions differ')
        require(np.all(np.diff(origin) > 0) and np.all(np.diff(decisions) > 0) and
                np.all(np.isfinite(actual)) and np.all(np.isfinite(anchor)) and np.all(anchor > 0),
                'Invalid saved target values')
        task_models = []
        for model in MODELS:
            record = rec['models'][model]
            prediction_path = folder / f'prediction_{model}.npz'
            require(digest(prediction_path) == record['prediction_sha256'], 'Prediction identity mismatch')
            saved_artifact_count += 1
            with np.load(prediction_path, allow_pickle=False) as z:
                require(set(z.files) == {'prediction', 'origins', 'decision_ns'}, 'Unexpected prediction archive')
                require(np.array_equal(z['origins'], origin) and np.array_equal(z['decision_ns'], decisions),
                        'Model origins do not match the task')
                prediction = np.asarray(z['prediction'], dtype=np.float64)
            require(prediction.shape == actual.shape and np.isfinite(prediction).all(), 'Invalid prediction values')
            error = prediction - actual
            square = np.square(error)
            absolute = np.abs(error)
            endpoint_mse = float(square[:, -1].mean())
            endpoint_mae = float(absolute[:, -1].mean())
            path_mse = float(square.mean())
            path_mae = float(absolute.mean())
            if cfg['target'] == 'future_close':
                predicted_move, actual_move = prediction[:, -1] - anchor, actual[:, -1] - anchor
            elif cfg['target'] == 'future_log_return':
                predicted_move, actual_move = prediction[:, -1], actual[:, -1]
            else:
                raise ValueError('Unknown target')
            active = predicted_move != 0
            active_n = int(active.sum())
            correct_n = int((np.sign(predicted_move[active]) == np.sign(actual_move[active])).sum())
            coverage = active_n / n
            accuracy = correct_n / active_n if active_n else None
            computed = {'endpoint_mse': endpoint_mse, 'endpoint_rmse': math.sqrt(endpoint_mse),
                        'endpoint_mae': endpoint_mae, 'path_mse': path_mse, 'path_mae': path_mae,
                        'direction_coverage': coverage, 'direction_accuracy_when_active': accuracy}
            saved = record['metrics']
            for name, value in computed.items():
                require(near(value, saved[name]), f'Fold metric mismatch {job_name} {model} {name}')
            require(np.allclose(square.mean(axis=0), saved['per_step_mse'], rtol=1e-11, atol=1e-16) and
                    np.allclose(absolute.mean(axis=0), saved['per_step_mae'], rtol=1e-11, atol=1e-16),
                    'Per-step metrics mismatch')
            task_models.append({
                'profile': cfg['profile'], 'horizon': int(cfg['horizon']), 'target': cfg['target'],
                'fold': fold, 'model': model, 'origins': n,
                'decision_first_ny': local_time(decisions[0]), 'decision_last_ny': local_time(decisions[-1]),
                'target_last_completion_ny': local_time(completions.max()),
                'endpoint_mse': endpoint_mse, 'endpoint_rmse': math.sqrt(endpoint_mse),
                'endpoint_mae': endpoint_mae, 'path_mse': path_mse, 'path_mae': path_mae,
                'active_count': active_n, 'correct_active_count': correct_n,
                'direction_coverage': coverage, 'direction_accuracy_when_active': accuracy,
                'fold_receipt_sha256': job['receipt_sha256'],
                'targets_sha256': rec['targets_sha256'], 'prediction_sha256': record['prediction_sha256'],
            })
        baseline = next(x['endpoint_mse'] for x in task_models if x['model'] == 'last_close')
        require(baseline > 0, 'A baseline has zero MSE')
        for row in task_models:
            row['relative_improvement'] = 1 - row['endpoint_mse'] / baseline
            saved = rec['models'][row['model']]['against_last_close_or_zero']['relative_improvement']
            require(near(row['relative_improvement'], saved), 'Fold baseline comparison mismatch')
            fold_rows.append(row)
        if len(folds_seen) % 10 == 0:
            print(json.dumps({'stage': 'verified_saved_folds', 'folds': len(folds_seen)}, ensure_ascii=False), flush=True)

require(len(folds_seen) == 60 and len(fold_rows) == 480 and saved_artifact_count == 540,
        'Unexpected number of verified fold artifacts')
pooled = []
for task_model in sorted(existing_by_key):
    rows = sorted([x for x in fold_rows if (x['profile'], x['horizon'], x['target'], x['model']) == task_model],
                  key=lambda x: x['fold'])
    require(len(rows) == 5 and [x['fold'] for x in rows] == [1, 2, 3, 4, 5], 'Missing development folds')
    require(all(rows[i]['decision_last_ny'] < rows[i + 1]['decision_first_ny'] for i in range(4)),
            'Outer decision windows overlap or are unordered')
    n = sum(x['origins'] for x in rows)
    def weighted(metric):
        return sum(x['origins'] * x[metric] for x in rows) / n
    active_n = sum(x['active_count'] for x in rows)
    correct_n = sum(x['correct_active_count'] for x in rows)
    item = {'profile': task_model[0], 'horizon': task_model[1], 'target': task_model[2], 'model': task_model[3],
            'folds': 5, 'origins': n, 'decision_first_ny': rows[0]['decision_first_ny'],
            'decision_last_ny': rows[-1]['decision_last_ny'],
            'target_last_completion_ny': max(x['target_last_completion_ny'] for x in rows),
            'endpoint_mse': weighted('endpoint_mse'), 'endpoint_mae': weighted('endpoint_mae'),
            'path_mse': weighted('path_mse'), 'path_mae': weighted('path_mae'),
            'direction_coverage': active_n / n,
            'direction_accuracy_when_active': correct_n / active_n if active_n else None,
            'active_count': active_n, 'correct_active_count': correct_n}
    item['endpoint_rmse'] = math.sqrt(item['endpoint_mse'])
    prior = existing_by_key[task_model]
    require(n == int(prior['validation_origins']) and near(item['endpoint_mse'], prior['endpoint_mse']),
            'Pooled endpoint MSE does not match previous independently checked aggregate')
    pooled.append(item)
by_pooled = {(x['profile'], x['horizon'], x['target'], x['model']): x for x in pooled}
for item in pooled:
    base = by_pooled[(item['profile'], item['horizon'], item['target'], 'last_close')]['endpoint_mse']
    item['relative_improvement'] = 1 - item['endpoint_mse'] / base
    prior = existing_by_key[(item['profile'], item['horizon'], item['target'], item['model'])]
    require(near(item['relative_improvement'], prior['relative_improvement']), 'Pooled relative improvement differs')

def csv_text(rows):
    fields = list(rows[0])
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()

scale_order = [('annual', 12, 'Annual H12 (short history)'), ('quarterly', 3, 'Quarterly H3'),
               ('monthly', 21, 'Monthly H21'), ('hourly', 1, 'Hourly H1'),
               ('hourly', 4, 'Hourly H4'), ('hourly', 8, 'Hourly H8')]

def value(x):
    return f'{x:.6g}'

appendix = [
    '', '## Appendix A. Detailed historical development results', '',
    '**Private, already-exposed development evidence; not independent confirmation or publication-admitted data.** '
    'This appendix covers the 12 tasks with five completed folds; the eight completed minute folds remain '
    'summarized in Section 5 and are not included in the tables below.', '',
    'Endpoint MSE, RMSE and MAE are pooled across origins, as are path MSE and path MAE. '
    'Price MSE is measured in (USD per share)² and price RMSE/MAE in USD per share; log-return errors are '
    'dimensionless, with MSE in squared log-return units. Relative improvement is 100 × (1 − model endpoint '
    'MSE / matched persistence endpoint MSE); it is not a return from trading. Direction accuracy is computed '
    'only for nonzero predicted endpoint changes; an all-abstaining model has no accuracy value. '
    'The underlying per-fold CSV also records path MAE and all counts.', '',
    'Dates are New York timestamps from the saved development artifacts, not a guarantee that a vendor value '
    'was available without later revision. The annual variant has a twelve-month lookback and uses the shorter '
    'history accepted for exploration. Overlapping targets, splits, prior exposure, and unverified publication '
    'rights constrain interpretation.', ''
]
for profile, horizon, label in scale_order:
    for target, title in [('future_close', 'nominal close'), ('future_log_return', 'cumulative log return')]:
        rows = [by_pooled[(profile, horizon, target, model)] for model in MODELS]
        first = rows[0]
        appendix += [f'### {label}: {title}', '',
                     f"Five folds; {first['origins']:,} origins. First and last decision: "
                     f"{first['decision_first_ny']} to {first['decision_last_ny']}; "
                     f"last target completion: {first['target_last_completion_ny']}.", '',
                     '| Model | End MSE | End RMSE | End MAE | Path MSE | Improvement | Direction coverage | Active accuracy |',
                     '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for r in rows:
            acc = '—' if r['direction_accuracy_when_active'] is None else f"{100*r['direction_accuracy_when_active']:.2f}%"
            appendix.append(f"| {LABELS[r['model']]} | {value(r['endpoint_mse'])} | {value(r['endpoint_rmse'])} | "
                            f"{value(r['endpoint_mae'])} | {value(r['path_mse'])} | "
                            f"{100*r['relative_improvement']:.2f}% | {100*r['direction_coverage']:.2f}% | {acc} |")
        appendix.append('')

text = parent_md.read_text(encoding='utf8') + '\n'.join(appendix) + '\n'
pooled.sort(key=lambda x: (x['profile'], x['horizon'], x['target'], MODELS.index(x['model'])))
fold_rows.sort(key=lambda x: (x['profile'], x['horizon'], x['target'], x['fold'], MODELS.index(x['model'])))
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
directory = OUT / 'private_working_manuscripts' / stamp
directory.mkdir(parents=True, exist_ok=False)
md = directory / 'PRIVATE_WORKING_MANUSCRIPT_EN.md'
pooled_path = directory / 'PRIVATE_METRICS_POOLED.csv'
fold_path = directory / 'PRIVATE_METRICS_BY_FOLD.csv'
history_path = directory / 'TRAINING_HISTORY_AUDIT.csv'
md.write_text(text, encoding='utf8')
pooled_path.write_text(csv_text(pooled), encoding='utf8', newline='')
fold_path.write_text(csv_text(fold_rows), encoding='utf8', newline='')
history_path.write_bytes((PARENT / 'TRAINING_HISTORY_AUDIT.csv').read_bytes())
audit = {
    'status': 'private_detailed_metrics_independently_recalculated_not_for_submission',
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'parent_manuscript': str(parent_md), 'parent_manuscript_sha256': digest(parent_md),
    'parent_evidence_audit': str(PARENT / 'PRIVATE_MANUSCRIPT_EVIDENCE_AUDIT.json'),
    'parent_evidence_audit_sha256': digest(PARENT / 'PRIVATE_MANUSCRIPT_EVIDENCE_AUDIT.json'),
    'script_sha256': digest(Path(__file__).resolve()), 'manuscript_sha256': digest(md),
    'pooled_csv_sha256': digest(pooled_path), 'fold_csv_sha256': digest(fold_path),
    'training_history_csv_sha256': digest(history_path),
    'checks': {
        'completed_tasks_in_appendix': 12, 'saved_folds_recomputed': len(folds_seen),
        'saved_prediction_and_target_hashes_verified': saved_artifact_count,
        'model_fold_rows': len(fold_rows), 'pooled_model_rows': len(pooled),
        'saved_fold_metrics_and_per_step_metrics_match': True,
        'pooled_endpoint_mse_and_improvement_match_existing_aggregate': True,
        'all_saved_target_completions_before_2026q3': True,
        'all_model_origins_matched_within_each_fold': True,
    },
    'numpy_version': np.__version__, 'training_started': False, 'Q3_values_read': False,
    'independent_test': False, 'publication_rights': 'not confirmed',
    'limitations': [
        'Arithmetic recomputation of exposed development predictions does not establish independent confirmation.',
        'Minute folds are not included in the detailed appendix.',
        'Source rights, point-in-time data quality, convergence, and final protocol remain unresolved.',
    ],
}
ap = directory / 'PRIVATE_DETAILED_RESULTS_AUDIT.json'
ap.write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf8')
print(json.dumps({'status': audit['status'], 'directory': str(directory), 'manuscript': str(md),
                  'pooled': str(pooled_path), 'folds': str(fold_path), 'audit': str(ap),
                  'checks': audit['checks']}, ensure_ascii=False), flush=True)
