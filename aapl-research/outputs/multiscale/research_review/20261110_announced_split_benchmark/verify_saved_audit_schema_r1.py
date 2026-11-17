"""Independent, read-only replay of the frozen announced-split audit.

This module never imports the main audit worker. The only market containers it
opens are explicitly listed in the frozen protocol, after checking all hashes.
The replay uses scalar calendar conditions and math.fsum loss aggregation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
from zoneinfo import ZoneInfo


HERE = Path(__file__).resolve().parent
ROOT = Path('/home/ayaka/wslcode/aapl-research')
NAMES = ('last_close', 'drift', 'seasonal', 'ridge', 'mlp', 'gru', 'lstm', 'xpatch')
TINY = 'tiny_long_aligned'
SUMS = ('endpoint_sse', 'endpoint_absolute_sum', 'path_sse')
MEANS = ('endpoint_mse', 'endpoint_mae', 'path_mse')
NY = ZoneInfo('America/New_York')
RTOL, ATOL = 1e-11, 1e-14
GAIN_ATOL_PERCENTAGE_POINTS = 1e-10
STOP = False


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        require(not path.exists(), 'Never replace an earlier independent receipt: ' + str(path))
        os.rename(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def stop_requested(_signum, _frame):
    global STOP
    STOP = True


def check_time(protocol):
    require(not STOP, 'Independent replay interrupted; original audit files remain intact')
    require(time.time() < protocol['deadline_epoch'], 'Frozen absolute deadline reached; no reset allowed')


def ny_string(ns):
    whole, fraction = divmod(int(ns), 1000000000)
    return datetime.fromtimestamp(whole, timezone.utc).astimezone(NY).isoformat() + f'; subsecond_ns={fraction}'


class Comparison:
    def __init__(self):
        self.count = 0
        self.numeric_count = 0
        self.max_absolute_error = 0.0
        self.max_scaled_error = 0.0

    def compare(self, expected, observed, path='root'):
        if isinstance(expected, dict):
            require(isinstance(observed, dict), path + ': expected object')
            for key, value in expected.items():
                require(key in observed, path + ': missing ' + key)
                self.compare(value, observed[key], path + '/' + key)
            return
        if isinstance(expected, list):
            require(isinstance(observed, list) and len(expected) == len(observed), path + ': list coverage differs')
            for i, (a, b) in enumerate(zip(expected, observed)):
                self.compare(a, b, path + '/' + str(i))
            return
        self.count += 1
        if isinstance(expected, float):
            require(isinstance(observed, (int, float)) and not isinstance(observed, bool), path + ': expected number')
            require(math.isfinite(expected) and math.isfinite(observed), path + ': nonfinite value')
            error = abs(expected - observed)
            absolute_tolerance = GAIN_ATOL_PERCENTAGE_POINTS if '_gain_pct_vs_' in path else ATOL
            tolerance = absolute_tolerance + RTOL * abs(expected)
            self.numeric_count += 1
            self.max_absolute_error = max(self.max_absolute_error, error)
            self.max_scaled_error = max(self.max_scaled_error, error / tolerance)
            require(error <= tolerance, path + f': mismatch {expected!r} != {observed!r}')
        else:
            require(type(expected) is type(observed) or
                    (isinstance(expected, int) and not isinstance(expected, bool) and isinstance(observed, int)),
                    path + ': scalar type differs')
            require(expected == observed, path + f': mismatch {expected!r} != {observed!r}')


def load(np, path, required, optional=()):
    with np.load(path, allow_pickle=False) as archive:
        require(set(required).issubset(archive.files), 'Required NPZ fields missing: ' + str(path))
        result = {key: archive[key] for key in (*required, *optional) if key in archive.files}
    require(all(a.dtype.kind != 'O' for a in result.values()), 'Object array forbidden')
    return result


def losses(np, prediction, actual):
    difference = prediction.astype('float64') - actual
    require(np.isfinite(difference).all(), 'Nonfinite prediction error')
    endpoint = difference[:, -1]
    return {'endpoint_sse': math.fsum(float(x) ** 2 for x in endpoint),
            'endpoint_absolute_sum': math.fsum(abs(float(x)) for x in endpoint),
            'path_sse': math.fsum(float(x) ** 2 for x in difference.flat)}


def means(sums, count, horizon):
    return dict(sums, endpoint_mse=sums['endpoint_sse'] / count,
                endpoint_mae=sums['endpoint_absolute_sum'] / count,
                path_mse=sums['path_sse'] / (count * horizon))


def gains(metrics, old, announced):
    answer = dict(metrics)
    for name, baseline in (('old_benchmark', old), ('announced_benchmark', announced)):
        for metric in MEANS:
            answer[metric + '_gain_pct_vs_' + name] = (100.0 * (1.0 - metrics[metric] / baseline[metric])
                                                        if baseline[metric] > 0 else None)
    return answer


def replay_fold(np, protocol, identity, task, fold, bars):
    cached = load(np, fold['targets'], ('origins', 'actual', 'anchor', 'decision_ns', 'target_completion_ns'))
    origins = cached['origins']
    require(origins.ndim == 1 and origins.dtype.kind in 'iu' and len(origins) > 0,
            'Invalid original origins')
    h, n = task['horizon'], len(origins)
    require(all(0 <= int(i) < len(bars['close']) - h for i in origins), 'Origin outside admitted bars')
    require(all(int(a) < int(b) for a, b in zip(origins[:-1], origins[1:])), 'Origin order changed')
    anchor = cached['anchor'].reshape(-1).astype('float64')
    decisions = cached['decision_ns']
    completion = cached['target_completion_ns']
    future = np.array([[int(i) + step for step in range(1, h + 1)] for i in origins], dtype='int64')
    require(np.array_equal(anchor, bars['close'][origins]), 'Nominal origin price changed')
    require(np.array_equal(decisions, bars['end_ns'][origins]), 'Origin completion time changed')
    require(np.array_equal(completion, bars['end_ns'][origins + h]), 'Endpoint completion time changed')
    nominal = bars['close'][future]
    if task['target'] == 'future_log_return':
        nominal = np.log(nominal / anchor[:, None])
    actual = cached['actual'].astype('float64')
    require(actual.shape == (n, h) and np.array_equal(nominal, actual), 'Saved nominal labels do not reconstruct')
    old_prediction = np.empty((n, h), dtype='float64')
    new_prediction = np.empty((n, h), dtype='float64')
    cross = np.zeros((n, h), dtype=bool)
    changed = np.zeros((n, h), dtype=bool)
    events = []
    sessions = bars['session_date'][origins]
    effective = protocol['effective_date']
    for row, origin in enumerate(origins):
        origin_date = str(sessions[row])
        is_known = int(decisions[row]) >= protocol['known_ns']
        for column, target_index in enumerate(future[row]):
            date = str(bars['session_date'][target_index])
            crosses_event = origin_date < effective <= date
            uses_announcement = is_known and crosses_event
            cross[row, column], changed[row, column] = crosses_event, uses_announcement
            if task['target'] == 'future_close':
                old_prediction[row, column] = anchor[row]
                new_prediction[row, column] = anchor[row] / (protocol['ratio'] if uses_announcement else 1.0)
            else:
                old_prediction[row, column] = 0.0
                new_prediction[row, column] = -math.log(protocol['ratio']) if uses_announcement else 0.0
        if cross[row].any():
            steps = [j + 1 for j in range(h) if cross[row, j]]
            events.append({'origin_index': int(origin), 'origin_session': origin_date,
                           'decision_ns': int(decisions[row]), 'decision_new_york': ny_string(decisions[row]),
                           'known_at_origin': is_known,
                           'endpoint_session': str(bars['session_date'][future[row, -1]]),
                           'cross_event_path_cells': len(steps), 'known_event_path_cells': int(changed[row].sum()),
                           'first_cross_event_step': steps[0], 'last_cross_event_step': steps[-1]})
    require(np.array_equal(old_prediction[~changed], new_prediction[~changed]), 'Factor-one cells changed')
    old = means(losses(np, old_prediction, actual), n, h)
    new = means(losses(np, new_prediction, actual), n, h)
    model_rows, seeds, tiny_losses = {}, [], []
    for model in sorted(fold['models'], key=lambda m: (m['name'], m['seed'])):
        check_time(protocol)
        pred = load(np, model['prediction'], ('prediction', 'origins'), ('decision_ns',))
        require(np.array_equal(origins, pred['origins']), 'Prediction origins differ')
        if 'decision_ns' in pred:
            require(np.array_equal(decisions, pred['decision_ns']), 'Prediction decisions differ')
        else:
            require(model['name'] == TINY, 'Unexpected missing original decision timestamps')
        require(pred['prediction'].shape == actual.shape and np.isfinite(pred['prediction']).all(),
                'Prediction shape or finiteness differs')
        raw_losses = losses(np, pred['prediction'], actual)
        metrics = gains(means(raw_losses, n, h), old, new)
        seeds.append({'name': model['name'], 'seed': model['seed'], 'prediction': model['prediction'],
                      'prediction_sha256': protocol['inputs'][model['prediction']],
                      'decision_ns_present_and_verified': 'decision_ns' in pred, **metrics})
        if model['name'] == TINY:
            tiny_losses.append(raw_losses)
        else:
            model_rows[model['name']] = {'seeds': [2026], 'seed_count': 1, **metrics}
    require(len(tiny_losses) == 3, 'All frozen tiny seeds are required')
    # Arithmetic of loss sums commutes with the equally weighted per-origin
    # seed mean. Predictions are never averaged in this independent replay.
    tiny = {key: math.fsum(v[key] for v in tiny_losses) / 3 for key in SUMS}
    model_rows[TINY] = {'seeds': [2026, 2027, 2028], 'seed_count': 3,
                        'aggregation': 'per_origin_mean_seed_loss_not_mean_prediction',
                        **gains(means(tiny, n, h), old, new)}
    return {'schema': 'announced_split_fold_v1', 'protocol_sha256': identity,
            **{k: task[k] for k in ('task', 'profile', 'frequency', 'horizon', 'target')},
            'fold': fold['fold'], 'origins': n,
            'origins_sha256': hashlib.sha256(origins.astype('<i8').tobytes()).hexdigest(),
            'first_origin_session': str(sessions[0]), 'last_origin_session': str(sessions[-1]),
            'first_decision_ns': int(decisions[0]), 'last_decision_ns': int(decisions[-1]),
            'last_target_completion_ns': int(completion[-1]),
            'cross_event_origins': int(cross.any(axis=1).sum()), 'known_event_origins': int(changed.any(axis=1).sum()),
            'cross_event_path_cells': int(cross.sum()), 'known_event_path_cells': int(changed.sum()),
            'adjusted_endpoint_origins': int(changed[:, -1].sum()), 'event_origins': events,
            'old_benchmark': old, 'announced_benchmark': new, 'models': model_rows, 'seed_metrics': seeds,
            'scope': 'recomputed_preQ3_exposed_validation_saved_nominal_labels_and_predictions',
            'new_training': False, 'new_inference': False, 'independent_confirmation': False, 'Q3_read': False}


def replay_pooled(task, rows):
    n = sum(r['origins'] for r in rows)
    metrics = {}
    for name in (*NAMES, TINY, 'old_benchmark', 'announced_benchmark'):
        sources = [r[name] if name.endswith('benchmark') else r['models'][name] for r in rows]
        sums = {key: math.fsum(v[key] for v in sources) for key in SUMS}
        metrics[name] = means(sums, n, task['horizon'])
    pooled = []
    for name in (*NAMES, TINY, 'old_benchmark', 'announced_benchmark'):
        pooled.append({**{key: task[key] for key in ('task', 'profile', 'horizon', 'target')},
                       'name': name, 'origins': n, 'completed_folds': 5,
                       'seed_count': 3 if name == TINY else 1,
                       **gains(metrics[name], metrics['old_benchmark'], metrics['announced_benchmark']),
                       'baseline_mse': metrics['old_benchmark']['endpoint_mse'],
                       'announced_baseline_mse': metrics['announced_benchmark']['endpoint_mse'],
                       'aggregation': 'per_origin_mean_seed_loss' if name == TINY else 'pooled_origin_loss',
                       'scope': 'recomputed_six_long_scale_tasks_exposed_development'})
    summary = {**{key: task[key] for key in ('task', 'profile', 'horizon', 'target')},
               'origins': n, 'completed_folds': 5,
               **{key: sum(r[key] for r in rows) for key in ('cross_event_origins', 'known_event_origins',
                   'cross_event_path_cells', 'known_event_path_cells', 'adjusted_endpoint_origins')},
               'old_benchmark': metrics['old_benchmark'], 'announced_benchmark': metrics['announced_benchmark'],
               TINY: next(r for r in pooled if r['name'] == TINY),
               'scope': 'recomputed_six_long_scale_tasks_exposed_development'}
    return pooled, summary


def run(args):
    started = time.time()
    os.sched_setaffinity(0, {26})
    require(os.sched_getaffinity(0) == {26}, 'Independent replay must use CPU26 only')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGUSR1, stop_requested)
    protocol_path = args.protocol.resolve()
    output = args.output.resolve()
    require(protocol_path.parent == HERE and output.is_relative_to(HERE), 'Separate audit stage required')
    protocol, identity = read(protocol_path), digest(protocol_path)
    final_receipt = output / 'INDEPENDENT_REPLAY_SCHEMA_R1.json'
    require(not final_receipt.exists(), 'Earlier independent replay receipt must be retained')
    comparison = Comparison()
    checked_folds = []
    try:
        check_time(protocol)
        require(protocol['maximum_seconds'] <= 300 and protocol['training_fits'] == 0
                and protocol['new_model_inferences'] == 0 and protocol['GPU'] is False
                and protocol['Q3_values_read_allowed'] is False, 'Frozen computational scope differs')
        require(protocol['known_ns'] == 1596254400000000000 and protocol['effective_date'] == '2020-08-31'
                and protocol['ratio'] == 4, 'Reviewed event boundary differs')
        adapter = read(HERE / 'SCHEMA_ADAPTER_PROTOCOL.json')
        require(adapter['original_protocol_sha256'] == identity, 'Schema adapter original protocol differs')
        require(adapter['deadline_epoch'] == protocol['deadline_epoch'], 'Schema adapter cannot reset deadline')
        require(adapter['adapter_script_sha256'] == digest(Path(__file__).resolve()), 'Adapter source differs')
        original_verifier = HERE / 'verify_saved_audit.py'
        require(adapter['original_verifier_sha256'] == digest(original_verifier)
                == protocol['inputs'][str(original_verifier)], 'Original frozen verifier differs')
        require((adapter['rtol'], adapter['atol'], adapter['gain_atol_percentage_points'])
                == (RTOL, ATOL, GAIN_ATOL_PERCENTAGE_POINTS), 'Numerical tolerances cannot change')
        require((output / 'INDEPENDENT_REPLAY.json').is_file(), 'Original failure receipt must be retained')
        for raw, expected_hash in sorted(protocol['inputs'].items()):
            check_time(protocol)
            path = Path(raw)
            require(path.is_absolute() and path.resolve().is_relative_to(ROOT)
                    and 'holdout_sealed' not in path.parts, 'Unadmitted input path')
            require(digest(path) == expected_hash, 'Input identity differs: ' + raw)
        require(len(protocol['task_records']) == 6 and len(protocol['inherited_tasks']) == 14,
                'Expected six replayed and fourteen inherited tasks')
        expected_tasks = {f'{profile}_h{h}_{target}' for profile, h in
                          (('annual', 12), ('quarterly', 3), ('monthly', 21))
                          for target in ('future_close', 'future_log_return')}
        require({t['task'] for t in protocol['task_records']} == expected_tasks,
                'Original six long-scale task identities changed')
        result_path, checkpoint_path = output / 'RESULTS.json', output / 'CHECKPOINT.json'
        result, checkpoint = read(result_path), read(checkpoint_path)
        comparison.compare(identity, result['protocol_sha256'], 'result/protocol')
        comparison.compare(identity, checkpoint['protocol_sha256'], 'checkpoint/protocol')
        comparison.compare(protocol['deadline_epoch'], checkpoint['deadline_epoch'], 'checkpoint/deadline')
        require(len(checkpoint['completed_folds']) == 30 and result['completed_folds'] == 30, 'Incomplete audit')
        saved = {(r['task'], r['fold']): r for r in checkpoint['completed_folds']}
        require(len(saved) == 30, 'Duplicate committed fold')
        import numpy as np
        bars = {}
        for frequency, raw in protocol['bars'].items():
            require(raw in protocol['inputs'], 'Unfrozen bars')
            bar = load(np, raw, ('session_date', 'end_ns', 'close'))
            require(bar['session_date'].dtype.kind == 'U' and bar['close'].ndim == 1
                    and all(len(v) == len(bar['close']) for v in bar.values()), 'Original bars shape changed')
            require(str(bar['session_date'][-1]) <= '2026-06-30'
                    and int(bar['end_ns'][-1]) <= 1782878400000000000, 'Post-development bars forbidden')
            require(np.isfinite(bar['close']).all() and (bar['close'] > 0).all(), 'Invalid original bars')
            bars[frequency] = bar
        all_pooled, summaries, old_checks, original_receipts = [], [], [], []
        old_results = read(protocol['old_results'])
        for task in sorted(protocol['task_records'], key=lambda r: r['task']):
            require({r['fold'] for r in task['folds']} == set(range(1, 6)), 'Original five-fold coverage changed')
            rows = []
            for fold in sorted(task['folds'], key=lambda r: r['fold']):
                check_time(protocol)
                require(fold['targets'] in protocol['inputs'] and
                        all(m['prediction'] in protocol['inputs'] for m in fold['models']), 'Unfrozen payload')
                expected_models = {(name, 2026) for name in NAMES} | {(TINY, seed) for seed in (2026, 2027, 2028)}
                require({(m['name'], m['seed']) for m in fold['models']} == expected_models
                        and len(fold['models']) == 11, 'Original model/seed coverage changed')
                row = replay_fold(np, protocol, identity, task, fold, bars[task['frequency']])
                key = (task['task'], fold['fold'])
                comparison.compare(row, saved[key], 'checkpoint/' + task['task'] + '/' + str(fold['fold']))
                foldfile = output / 'folds' / task['task'] / f"fold_{fold['fold']}.json"
                comparison.compare(row, read(foldfile), 'foldfile/' + task['task'] + '/' + str(fold['fold']))
                rows.append(row)
                checked_folds.append({'task': task['task'], 'fold': fold['fold']})
            pooled, summary = replay_pooled(task, rows)
            all_pooled.extend(pooled)
            summaries.append(summary)
            tiny = next(r for r in pooled if r['name'] == TINY)
            originals = [r for r in old_results['pooled'] if r['task'] == task['task'] and r.get('variant') == 'long_aligned']
            require(len(originals) == 1, 'Original tiny scalar missing or ambiguous')
            old = originals[0]
            comparison.compare({key: tiny[key] for key in ('origins', 'completed_folds', 'seed_count',
                               'endpoint_mse', 'baseline_mse', 'endpoint_mae', 'path_mse')}, old, 'original/' + task['task'])
            old_checks.append(task['task'])
            saved_tiny = next(r for r in result['pooled'] if r['task'] == task['task'] and r['name'] == TINY)
            saved_checks = [r for r in result['original_tiny_scalar_checks']['rows'] if r['task'] == task['task']]
            require(len(saved_checks) == 1, 'Original scalar receipt missing or ambiguous')
            for metric in ('endpoint_mse', 'baseline_mse', 'endpoint_mae', 'path_mse'):
                require(saved_checks[0]['recomputed_minus_original'][metric] == saved_tiny[metric] - old[metric],
                        'Main residual receipt arithmetic identity failed: ' + task['task'] + '/' + metric)
            # Tiny's actual loss values were compared in their original units
            # above and again against the main pooled output below. The worker
            # residual receipt is checked as its own exact arithmetic identity,
            # not as agreement between two differently rounded near-zero sums.
            original_receipts.append({'task': task['task'], 'original_row': old,
                                      'recomputed_minus_original': {key: saved_tiny[key] - old[key] for key in
                                          ('endpoint_mse', 'baseline_mse', 'endpoint_mae', 'path_mse')}})
        comparison.compare(all_pooled, result['pooled'], 'pooled')
        comparison.compare(summaries, result['task_summaries'], 'summaries')
        comparison.compare({'status': 'matched_original_saved_pooled_scalars', 'rtol': RTOL, 'atol': ATOL,
                            'original_values_modified': False, 'rows': original_receipts},
                           result['original_tiny_scalar_checks'], 'original_scalar_receipts')
        comparison.compare({'protocol_sha256': identity, 'folds': checkpoint['completed_folds'],
                            'completed_folds': 30, 'expected_folds': 30},
                           read(output / 'FOLD_RESULTS.json'), 'all_fold_report')
        overview = list(summaries)
        for row in protocol['inherited_tasks']:
            overview.append({'task': row['task'],
                             'scope': row.get('inherited_scope', row.get('scope', 'inherited_only_not_recomputed; pre2020_event_absent_from_2021plus_intraday_history')),
                             'recomputed_in_this_audit': False, 'inherited_original_pooled_row': row,
                             'new_event_metrics_computed': False})
        require(len(overview) == len({r['task'] for r in overview}) == 20, 'Twenty-task coverage differs')
        overview.sort(key=lambda r: r['task'])
        comparison.compare(overview, result['twenty_task_overview'], 'overview')
        comparison.compare({'protocol_sha256': identity, 'tasks': overview, 'recomputed_tasks': 6,
                            'inherited_only_tasks': 14}, read(output / 'TWENTY_TASK_OVERVIEW.json'), 'overview_file')
        comparison.compare({'independent_confirmation': False, 'research_goal_achieved': False,
                            'method_candidate_admitted': False, 'new_training': False, 'new_inference': False,
                            'GPU': False, 'Q3_read': False}, result, 'scope')
        check_time(protocol)
        receipt = {'status': 'passed_independent_numerical_replay', 'protocol_sha256': identity,
                   'deadline_epoch': protocol['deadline_epoch'], 'completed_before_deadline': True,
                   'replayed_folds': 30, 'replayed_tasks': 6, 'prediction_files_replayed': 330,
                   'pooled_rows_compared': len(all_pooled), 'inherited_only_tasks': 14,
                   'original_tiny_scalar_tasks_checked': old_checks,
                   'comparison_count': comparison.count, 'numeric_comparison_count': comparison.numeric_count,
                   'max_absolute_error': comparison.max_absolute_error,
                   'max_tolerance_fraction': comparison.max_scaled_error, 'rtol': RTOL, 'atol': ATOL,
                   'gain_atol_percentage_points': GAIN_ATOL_PERCENTAGE_POINTS,
                   'residual_receipt_rule': 'main_saved_loss_minus_old_loss_arithmetic_identity; independent_losses_compared_in_original_units',
                   'independent_implementation': 'scalar event conditions; fsum losses; no worker import or call',
                   'tiny_seed_aggregation': 'mean_seed_loss_not_mean_prediction',
                   'factor_one_cells_identical': True, 'CPU': 26, 'numerical_threads': 1,
                   'new_training': False, 'new_inference': False, 'GPU': False, 'Q3_read': False,
                   'independent_market_confirmation': False, 'research_goal_achieved': False,
                   'protocol_input_codehash': {raw: h for raw, h in protocol['inputs'].items()
                                               if Path(raw).suffix == '.py'},
                   'reportinputcodehash': {str(result_path): digest(result_path),
                                           str(checkpoint_path): digest(checkpoint_path),
                                           str(output / 'FOLD_RESULTS.json'): digest(output / 'FOLD_RESULTS.json'),
                                           str(output / 'TWENTY_TASK_OVERVIEW.json'): digest(output / 'TWENTY_TASK_OVERVIEW.json'),
                                           str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
                   'elapsed_seconds': time.time() - started,
                   'completed_utc': datetime.now(timezone.utc).isoformat()}
        receipt['metadata_schema_adapter'] = True
        receipt['original_failure_receipt_preserved'] = str(output / 'INDEPENDENT_REPLAY.json')
        receipt['schema_adapter_protocol_sha256'] = digest(HERE / 'SCHEMA_ADAPTER_PROTOCOL.json')
        receipt['numeric_code_and_tolerances_unchanged'] = True
        save(final_receipt, receipt)
        print(json.dumps({k: receipt[k] for k in ('status', 'replayed_folds', 'comparison_count',
                                                'max_absolute_error', 'max_tolerance_fraction', 'elapsed_seconds')}))
        return 0
    except Exception as exc:
        failure = {'status': 'failed_or_paused_independent_replay', 'protocol_sha256': identity,
                   'deadline_epoch': protocol.get('deadline_epoch'), 'verified_folds': checked_folds,
                   'error_type': type(exc).__name__, 'error': str(exc), 'Q3_read': False,
                   'original_audit_files_modified': False, 'elapsed_seconds': time.time() - started,
                   'completed_utc': datetime.now(timezone.utc).isoformat()}
        save(final_receipt, failure)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    sys.exit(run(parser.parse_args()))
