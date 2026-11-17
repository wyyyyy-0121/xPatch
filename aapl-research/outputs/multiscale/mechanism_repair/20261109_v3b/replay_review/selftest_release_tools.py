"""Purely synthetic complete-capsule and detached-entrypoint acceptance."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
TOOLS = STUDY/'release_tools'
sys.path.insert(0, str(TOOLS))
import stage_context_release as staging


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2)+'\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def main():
    assert set(os.sched_getaffinity(0)).issubset({24, 26})
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    start = time.time(); rng = np.random.default_rng(731)
    predictor = module('frozen_predictor_synthetic_release', STUDY/'code/predictor.py')
    registered_path = STUDY.with_name('20261109_v3')/'code/portable_stats.py'
    registered = module('registered_stats_reference_synthetic', registered_path)
    with tempfile.TemporaryDirectory(prefix='aapl-v3b-release-selftest-') as temporary:
        temporary = Path(temporary); source = temporary/'synthetic_source'
        (source/'code').mkdir(parents=True)
        for filename in ('predictor.py', 'experiment.py', 'evaluate_context.py'):
            shutil.copyfile(STUDY/'code'/filename, source/'code'/filename)
        shutil.copyfile(STUDY/'PLAN_ZH.md', source/'PLAN_ZH.md')
        tasks = ['minute_h60_future_log_return', 'hourly_h8_future_log_return']
        cells = {'A_mean': [], 'B_context': list(range(9, 17)), 'C_history': list(range(9)), 'D_joint': list(range(17))}
        sources = {name: sha(source/'code'/name) for name in ('predictor.py', 'experiment.py')}
        rules = {'tasks': tasks, 'cells': cells, 'candidate': 'D_joint_shrink', 'sources': sources,
            'plan_sha256': sha(source/'PLAN_ZH.md'), 'Q3_values_read': False, 'independent_confirmation': False}
        write(source/'protocol/RULES.json', rules)
        gate = {'tasks': {task: {'inner_gate_passed': False} for task in tasks}}
        write(source/'protocol/GATE_AND_ADMISSION.json', gate)
        evaluation = {'rules_sha256': sha(source/'protocol/RULES.json'),
            'gate_sha256': sha(source/'protocol/GATE_AND_ADMISSION.json'), 'candidate': rules['candidate'],
            'evaluator_sha256': sha(source/'code/evaluate_context.py'), 'portable_stats_sha256': sha(registered_path),
            'blocks': {'minute': [1, 5], 'hourly': [2, 5]}, 'repetitions': 5000,
            'statistics_seed': 2026, 'Q3_values_read': False}
        write(source/'protocol/EVALUATION_FREEZE.json', evaluation)
        rule_sha = sha(source/'protocol/RULES.json'); eval_sha = sha(source/'protocol/EVALUATION_FREEZE.json')
        stats = {'rules_sha256': rule_sha, 'evaluation_freeze_sha256': eval_sha,
            'candidate': rules['candidate'], 'tasks': {}, 'Q3_values_read': False,
            'independent_confirmation': False, 'both_primary_performance_gates_passed': False,
            'independent_confirmation_eligible': False}
        names = ['zero', 'archived_default_raw', 'archived_default_shrink', 'fit_mean_raw', 'fit_mean_shrink']
        names += ['ridge_'+mode for mode in ('history', 'history_anchor', 'history_selected', 'history_anchor_selected')]
        for kind in ('default', 'repair'):
            names += [f'{kind}_seed_{seed}_{mode}' for seed in (2026, 2027, 2028) for mode in ('raw', 'shrink')]
            names += [kind+'_three_seed_'+mode for mode in ('raw', 'shrink')]
        names += [cell+'_'+mode for cell in cells for mode in ('raw', 'shrink')]
        assert len(names) == 33
        all_rows = []
        for task in tasks:
            h = 60 if task.startswith('minute') else 8
            lookback = 480 if h == 60 else 96
            groups = []; all_losses = {}; byfold = {}
            for fold in range(1, 6):
                folder = source/'fits'/task/f'fold_{fold}'; folder.mkdir(parents=True)
                identity = task+str(fold)
                np.savez_compressed(folder/'FIT_ACCUM.npz', identity=np.asarray(identity), count=np.asarray(1000),
                    cursor=np.asarray(1000), sx=np.zeros(17), sy=np.zeros(h), xx=np.eye(17), xy=np.zeros((17, h)))
                sufficient_sha = sha(folder/'FIT_ACCUM.npz')
                write(folder/'FIT_DONE.json', {'status': 'completed', 'identity': identity, 'fit_count': 1000,
                    'sufficient_statistics_sha256': sufficient_sha})
                history = np.c_[np.zeros(128), np.cumsum(rng.normal(0, .001, (128, lookback-1)), axis=1)]
                history -= history[:, -1:]
                timestamps = pd.date_range('2026-01-05 10:00', periods=128, freq='min', tz='America/New_York').tz_convert('UTC').as_unit('ns').asi8
                features = predictor.features(history, timestamps, h)
                hashes = {}; inner = {}
                for cell, columns in cells.items():
                    columns = np.asarray(columns, dtype=np.int64)
                    np.savez_compressed(folder/(cell+'.npz'), coef=rng.normal(0, .0001, (len(columns), h)),
                        x_mean=features[:, columns].mean(0), x_std=np.maximum(features[:, columns].std(0), 1e-8),
                        y_mean=rng.normal(0, .0001, h), columns=columns, horizon=np.asarray(h), alpha=np.asarray(.25))
                    model = predictor.Predictor(folder/(cell+'.npz'))
                    raw = model.predict(history, timestamps)
                    np.savez_compressed(folder/(cell+'_EXAMPLE.npz'), history=history, completion_ns=timestamps,
                        origins=np.arange(128), expected_raw=raw, expected_shrink=.25*raw)
                    delivery = source/'delivery'/task/f'fold_{fold}'; delivery.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(delivery/(cell+'.npz'), history=history, completion_ns=timestamps,
                        origins=np.arange(128), expected_raw=raw, expected_shrink=.25*raw)
                    hashes[cell+'.npz'] = sha(folder/(cell+'.npz'))
                    inner[cell] = {'alpha': .25, 'raw_endpoint_mse': 2. if cell == 'D_joint' else 1.}
                write(folder/'INNER_DONE.json', {'status': 'completed', 'identity': identity, 'task': task,
                    'fold': fold, 'rules_sha256': rule_sha, 'source_hashes': sources,
                    'seed': None, 'solver_deterministic': True, 'outer_labels_constructed': False,
                    'sufficient_statistics_sha256': sufficient_sha, 'fit_count': 1000, 'stop_count': 128,
                    'inner_metrics': inner, 'model_hashes': hashes, 'zero_inner_endpoint_mse': 1.})
                grid = pd.bdate_range('2025-01-06', periods=17).strftime('%Y-%m-%d').to_numpy(dtype='U10')
                # A real grid day with zero origins tests the portable day-grid contract.
                dates = np.repeat(np.delete(grid, 5), 3)
                origins = np.arange(len(dates)) + fold*100
                common = rng.uniform(.0001, .0003, len(dates))
                losses = {name: common*rng.uniform(.5, 1.5, len(dates)) for name in names}
                lossfile = source/'analysis/paired_losses'/task/f'fold_{fold}.npz'
                lossfile.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(lossfile, origins=origins, session_date=dates, day_grid=grid, **losses)
                rows = []; paths = {}
                for cell in cells:
                    for mode in ('raw', 'shrink'):
                        name = cell+'_'+mode
                        paths[name] = losses[name]*rng.uniform(.2, 1., len(dates))
                        rows.append({'task': task, 'fold': fold, 'method': name, 'origins': len(dates),
                            'endpoint_mse': float(losses[name].mean()), 'baseline_mse': float(losses['zero'].mean()),
                            'improvement_pct': float(100*(1-losses[name].mean()/losses['zero'].mean())),
                            'endpoint_mae': float(np.sqrt(losses[name]).mean()), 'path_mse': float(paths[name].mean())})
                pathfile = source/'analysis/path_losses'/task/f'fold_{fold}.npz'
                pathfile.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(pathfile, origins=origins, **paths)
                write(source/'analysis/folds'/task/f'fold_{fold}.json', {'origins': len(dates), 'metrics': rows,
                    'paired_loss_sha256': sha(lossfile), 'path_loss_sha256': sha(pathfile),
                    'evaluation_freeze_sha256': eval_sha, 'loss_names': names, 'model_hashes': hashes})
                all_rows.extend(rows)
                fake_bars = type('Bars', (), {'session_date': grid})()
                # Expand origin-index mapping while retaining an empty grid day.
                indexed = np.searchsorted(grid, dates)
                groups.append(registered.day_groups(fake_bars, indexed, losses))
                for name in names:
                    all_losses.setdefault(name, []).append(losses[name])
                    byfold.setdefault(name, []).append(float(losses[name].mean()))
            merged = {name: np.concatenate(value) for name, value in all_losses.items()}
            summary = {name: {'origins': len(value), 'endpoint_mse': float(value.mean()),
                'improvement_pct': float(100*(1-value.mean()/merged['zero'].mean())), 'fold_mse': byfold[name],
                'positive_folds': sum(a < b for a, b in zip(byfold[name], byfold['zero']))} for name, value in merged.items()}
            controls = [name for name in names if name not in (rules['candidate'], 'D_joint_raw')]
            diagnostic = [('D_joint_raw', 'C_history_raw'), ('D_joint_raw', 'B_context_raw'), ('B_context_raw', 'A_mean_raw')]
            pairs = [(rules['candidate'], control) for control in controls]+diagnostic
            intervals = []
            for block in evaluation['blocks'][task.split('_h')[0]]:
                intervals += registered.paired_intervals(groups, pairs, block, task)
            comparisons = {control: [row for row in intervals if row['primary'] == rules['candidate'] and row['reference'] == control] for control in controls}
            checks = {'minimum_one_percent': summary[rules['candidate']]['improvement_pct'] >= 1.,
                'four_positive_folds': summary[rules['candidate']]['positive_folds'] >= 4,
                'all_control_intervals_both_blocks': all(all(row['supports_improvement'] for row in rows) for rows in comparisons.values())}
            stats['tasks'][task] = {'summary': summary, 'comparisons': comparisons,
                'diagnostic_contrasts': [row for row in intervals if (row['primary'], row['reference']) in diagnostic],
                'checks': checks, 'performance_gate_passed': all(checks.values()),
                'inner_direction_gate_passed': False, 'independent_confirmation_eligible': False}
        stats['both_primary_performance_gates_passed'] = all(value['performance_gate_passed'] for value in stats['tasks'].values())
        write(source/'analysis/STATISTICS.json', stats)
        write(source/'analysis/OUTER_RESULTS.json', {'matrix_cells': 40, 'metrics': all_rows,
            'statistics_sha256': sha(source/'analysis/STATISTICS.json'), 'Q3_values_read': False,
            'independent_confirmation': False})
        package = temporary/'release/SYNTHETIC_CONTEXT_CAPSULE'
        files = staging.stage(source, package, registered_path)
        result = staging.archive_and_verify(package, temporary/'SYNTHETIC_RELEASE.json')
        assert result['detached_checks'][0]['receipt']['predictors'] == 40
        assert result['detached_checks'][0]['receipt']['streaming_comparisons'] == 40
        assert result['detached_checks'][1]['receipt']['path_metric_rows'] == 80
        result = {'status': 'passed_synthetic_release_tool_acceptance', 'manifest_files': len(files),
            'detached_checks': result['detached_checks'], 'original_workspace_and_network_access_blocked': True,
            'synthetic_zero_origin_day_included': True, 'real_market_arrays_read': False,
            'Q3_values_read': False, 'GPU': False, 'affinity': sorted(os.sched_getaffinity(0)),
            'elapsed_seconds': time.time()-start,
            'tool_sha256': {path.name: sha(path) for path in sorted(TOOLS.glob('*.py'))}}
    write(HERE/'SYNTHETIC_RELEASE_TOOL_ACCEPTANCE.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
