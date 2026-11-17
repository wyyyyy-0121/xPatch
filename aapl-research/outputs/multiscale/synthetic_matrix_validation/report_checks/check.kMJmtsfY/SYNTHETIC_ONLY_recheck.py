"""Independently recheck two five-fold synthetic tasks; never read market data."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from multiscale.matrix_executor import _job_key

root = Path('outputs/multiscale/synthetic_matrix_validation').resolve()
protocol, run, output = (Path(arg).resolve() for arg in sys.argv[1:4])
if not all(path.is_relative_to(root) for path in (protocol, run, output)):
    raise ValueError('All paths must be within the synthetic validation root')
proto = json.loads(protocol.read_text())
state = json.loads((run / 'matrix_state.json').read_text())
report = json.loads((output / 'SYNTHETIC_ONLY_report.json').read_text())
manifest = json.loads((output / 'SYNTHETIC_ONLY_manifest.json').read_text())
if any(obj.get('data_kind') != 'synthetic_test' for obj in (proto, state, report, manifest)):
    raise ValueError('Only explicitly synthetic inputs are allowed')
if report['independent_test'] is not False or report['complete_twenty_task_matrix'] is not False:
    raise ValueError('Synthetic report misstates the scope')
if report['task_status_counts'] != {'complete': 2, 'not_started': 18}:
    raise ValueError('Unexpected task coverage')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
if any(sha(output / name) != digest for name, digest in manifest['files_sha256'].items()):
    raise ValueError('Report output hash mismatch')
with (output / 'SYNTHETIC_ONLY_model_rows.csv').open(newline='') as stream:
    rows = list(csv.DictReader(stream))
with (output / 'SYNTHETIC_ONLY_training_rows.csv').open(newline='') as stream:
    training = list(csv.DictReader(stream))
if len(rows) != 160 or len(training) != 80:
    raise ValueError('Unexpected row counts')

for target in ('future_close', 'future_log_return'):
    task = next(t for t in report['tasks'] if t['profile'] == 'annual' and t['horizon'] == 12 and t['target'] == target)
    if task['completed_folds'] != [1, 2, 3, 4, 5] or task['missing_folds']:
        raise ValueError('Five-fold task coverage is incorrect')
    jobs = sorted((j for j in proto['jobs'] if j['profile'] == 'annual' and
                   j['horizon'] == 12 and j['target'] == target), key=lambda j: j['fold'])
    folders = [run / state['jobs'][_job_key(job)]['directory'] for job in jobs]
    actuals = []
    for folder in folders:
        with np.load(folder / 'targets.npz', allow_pickle=False) as data:
            actuals.append(data['actual'])
    actual = np.concatenate(actuals)
    calculated = {}
    for name, summary in task['summary']['models'].items():
        predictions = []
        for folder in folders:
            with np.load(folder / ('prediction_' + name + '.npz'), allow_pickle=False) as data:
                predictions.append(data['prediction'])
        error = np.concatenate(predictions) - actual
        step_mse, step_mae = np.mean(error ** 2, axis=0), np.mean(np.abs(error), axis=0)
        expected = {'endpoint_mse': step_mse[-1], 'endpoint_mae': step_mae[-1],
                    'path_mse': step_mse.mean(), 'path_mae': step_mae.mean()}
        if (not all(np.isclose(summary[k], value, rtol=1e-10, atol=1e-14) for k, value in expected.items()) or
                not np.allclose(summary['per_step_mse'], step_mse, rtol=1e-10, atol=1e-14) or
                not np.allclose(summary['per_step_mae'], step_mae, rtol=1e-10, atol=1e-14)):
            raise ValueError('Independent synthetic metric recomputation differs')
        calculated[name] = float(step_mse[-1])
        row = next(r for r in rows if r['profile'] == 'annual' and r['horizon'] == '12' and
                   r['target'] == target and r['model'] == name)
        unit = 'USD^2/share^2' if target == 'future_close' else 'log_return^2'
        if row['squared_error_unit'] != unit or row['completed_folds'] != '1,2,3,4,5' or row['missing_folds']:
            raise ValueError('Units or fold labels differ')
    baseline = calculated['last_close']
    for name, summary in task['summary']['models'].items():
        expected = 1 - calculated[name] / baseline if baseline > 0 else None
        actual_improvement = summary['relative_improvement_vs_persistence']
        if (expected is None and actual_improvement is not None) or (expected is not None and
                not np.isclose(actual_improvement, expected, rtol=1e-10, atol=1e-14)):
            raise ValueError('Baseline-relative error differs')
if any(row['endpoint_mse'] for row in rows if row['task_status'] == 'not_started'):
    raise ValueError('An unstarted task has a score')
if any(row['max_epochs'] != '1' or row['budget_cap_reached'] != 'True' for row in training
       if row['model'] in ('mlp', 'gru', 'lstm', 'xpatch')):
    raise ValueError('Synthetic training cap is not disclosed')
receipt = {'status': 'passed_synthetic_test_only', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
           'market_data_read': False, 'formal_experiment': False, 'five_fold_tasks_verified': 2,
           'models_per_task': 8, 'coverage_rows': len(rows), 'training_rows': len(training),
           'checks': ['independent per-step, endpoint, and path errors', 'baseline-relative errors',
                      'price and return units', 'fold coverage and missing values',
                      'training epoch caps', 'output file hashes'],
           'protocol_sha256': sha(protocol), 'matrix_state_sha256': sha(run / 'matrix_state.json'),
           'report_manifest_sha256': sha(output / 'SYNTHETIC_ONLY_manifest.json'),
           'verification_source_sha256': sha(Path(__file__)),
           'limitations': ['Fictional data only; no market result or independent confirmation.',
                           'Bootstrap intervals and real-data source permissions are not established by this recheck.']}
path = Path(__file__).parent / 'SYNTHETIC_ONLY_receipt.json'
path.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf8')
print(json.dumps({'status': receipt['status'], 'receipt': str(path), 'tasks': 2, 'models': 8}))
