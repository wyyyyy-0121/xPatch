"""Detached saved-loss statistics replay, without targets, training or GPU.

Imports only pure numerical functions from the bundled independent reviewer.
Its date-specific market guard is replaced in memory with this replay's signal
and optional deadline guard. The immutable review script is never edited and
its market entry/data loader is never called. Future offline replay has no
expired authorization deadline; today's release passes its explicit deadline.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import signal
import sys
import time
import numpy as np

HORIZONS = (5, 15, 30, 60)
STOP = False
DEADLINE = None


def guard():
    if STOP or (DEADLINE is not None and time.time() >= DEADLINE):
        raise InterruptedError('Detached replay interrupted or explicit deadline reached')


def interrupted(*_):
    global STOP
    STOP = True


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            guard(); value.update(block)
    return value.hexdigest()


def read(path):
    guard(); return json.loads(Path(path).read_text())


def install_guard(denied, self_test):
    counts = {'network_attempts': 0, 'forbidden_path_attempts': 0}; tests = {}
    def audit(event, values):
        if event in ('socket.connect', 'socket.connect_ex', 'socket.getaddrinfo', 'socket.bind'):
            counts['network_attempts'] += 1; raise PermissionError('Network disabled during detached replay')
        if event in ('open', 'os.listdir', 'os.scandir') and values:
            value = values[0]
            if isinstance(value, (str, bytes, os.PathLike)):
                path = Path(os.fsdecode(value)).resolve()
                if any(path == base or path.is_relative_to(base) for base in denied):
                    counts['forbidden_path_attempts'] += 1
                    raise PermissionError('Original workspace disabled during detached replay')
    sys.addaudithook(audit)
    if self_test:
        assert denied
        try: (denied[0]/'DO_NOT_READ_STATISTICS_GUARD_PROBE').read_bytes()
        except PermissionError: tests['original_workspace_open_blocked'] = True
        else: raise AssertionError('Workspace guard did not fire')
        import socket
        sock = socket.socket()
        try: sock.connect(('127.0.0.1', 9))
        except PermissionError: tests['socket_connect_blocked_before_network'] = True
        else: raise AssertionError('Network guard did not fire')
        finally: sock.close()
    return counts, tests


def metadata_gate(root, manifest, synthetic):
    essential = {'BUNDLE_SCOPE.json', 'replay_statistics.py', 'review/independent_statistics.py',
        'review/INDEPENDENT_STATISTICS_REVIEW.json', 'analysis/ANALYSIS_COMPLETE.json',
        'analysis/STATS.json', 'analysis/RESULTS.json', 'protocol/RULES.json',
        'protocol/EVALUATION_FREEZE.json', 'audit/COMPLETE.json'}
    for h in HORIZONS:
        task = f'minute_h{h}_future_log_return'
        for fold in range(1, 6):
            essential.add(f'analysis/folds/{task}/fold_{fold}.json')
            for family in ('paired_losses', 'path_losses', 'absolute_errors'):
                essential.add(f'analysis/{family}/{task}/fold_{fold}.npz')
    assert essential.issubset(manifest['files']), 'Incomplete saved-loss statistical capsule'
    scope = read(root/'BUNDLE_SCOPE.json')
    assert scope['Q3_values_read'] is False and scope['independent_confirmation'] is False
    if synthetic:
        assert scope['synthetic_statistics_fixture'] is True and scope['formal_market_bundles'] == 0
        assert scope['research_experiment_complete'] is False and manifest['market_bundles'] == 0
    else:
        assert scope['formal_market_bundles'] == manifest['market_bundles'] == 30
        assert scope['research_experiment_complete'] is True
    rules = read(root/'protocol/RULES.json')
    assert rules['admitted'] is True and rules['all_cells_required'] is True
    assert rules['Q3_values_read'] is False and rules['independent_confirmation'] is False
    assert rules['folds'] == [1, 2, 3, 4, 5] and rules['seeds'] == [2026, 2027, 2028]
    assert rules['fixed_epochs'] == 20 and rules['horizons'] == list(HORIZONS)
    assert len(rules['jobs']) == 30 and len({j['id'] for j in rules['jobs']}) == 30
    assert {(j['arm'], j['fold'], j['seed']) for j in rules['jobs']} == {
        (a, f, s) for a in ('tied', 'untied') for f in range(1, 6) for s in (2026, 2027, 2028)}
    paired_steps = {}
    for job in rules['jobs']:
        relative = f"runs/{job['id']}/DONE.json"
        assert relative in manifest['files']
        done = read(root/relative)
        assert done['status'] == 'completed' and done['job'] == job and done['fixed_epochs'] == 20
        assert done['rules_sha256'] == digest(root/'protocol/RULES.json')
        identity = hashlib.sha256(json.dumps({'job': job, 'rule': done['rules_sha256']}, sort_keys=True).encode()).hexdigest()
        assert done['identity'] == identity
        assert [row['epoch'] for row in done['history']] == list(range(1, 21))
        assert done['history'][-1]['steps'] == done['steps']
        scores = [row['aggregate_standardized_inner_endpoint_mse'] for row in done['history']]
        assert all(np.isfinite(value) for value in scores)
        assert done['best_epoch'] == min(range(20), key=lambda index: scores[index])+1
        assert set(done['alphas']) == {'last', 'selected'}
        for mode in ('last', 'selected'):
            assert set(done['alphas'][mode]) == {str(h) for h in HORIZONS}
            assert all(np.isfinite(value) and 0 <= value <= 1 for value in done['alphas'][mode].values())
        paired_steps.setdefault((job['fold'], job['seed']), []).append(done['steps'])
        assert done['Q3_values_read'] is False and done['independent_confirmation'] is False
    assert all(len(values) == 2 and values[0] == values[1] for values in paired_steps.values())
    freeze = read(root/'protocol/EVALUATION_FREEZE.json')
    assert freeze['status'] == 'all_30_bundles_fixed_before_new_outer_scores'
    assert freeze['rules_sha256'] == digest(root/'protocol/RULES.json')
    assert freeze['candidate'] == 'tied_selected_shrink' and freeze['horizons'] == list(HORIZONS)
    assert freeze['seeds'] == [2026, 2027, 2028] and freeze['blocks'] == [1, 5]
    assert freeze['repetitions'] == 5000 and freeze['statistics_seed'] == 2026 and freeze['legacy_h60_columns'] == 33
    complete = read(root/'audit/COMPLETE.json')
    assert complete['status'] == 'completed' and complete['bundles'] == 30 and complete['Q3_values_read'] is False
    analysis = read(root/'analysis/ANALYSIS_COMPLETE.json')
    assert analysis['status'] == 'completed' and analysis['complete_bundles'] == 30
    assert analysis['market_training_complete'] is True and analysis['all_five_folds_three_seeds'] is True
    for field, name in (('rules_sha256', 'protocol/RULES.json'), ('statistics_sha256', 'analysis/STATS.json'),
                        ('results_sha256', 'analysis/RESULTS.json'), ('evaluation_freeze_sha256', 'protocol/EVALUATION_FREEZE.json')):
        assert analysis[field] == digest(root/name), field
    required_analysis = {name for name in essential if name.startswith('analysis/') and name != 'analysis/ANALYSIS_COMPLETE.json'}
    assert required_analysis.issubset(analysis['files']) and len(required_analysis) == 82
    for name, value in analysis['files'].items():
        assert manifest['files'].get(name) == value
    review = read(root/'review/INDEPENDENT_STATISTICS_REVIEW.json')
    assert review['status'] == 'passed'
    for field, name in (('rules_sha256', 'protocol/RULES.json'), ('analysis_complete_sha256', 'analysis/ANALYSIS_COMPLETE.json'),
                        ('evaluation_freeze_sha256', 'protocol/EVALUATION_FREEZE.json'), ('script_sha256', 'review/independent_statistics.py')):
        assert review[field] == digest(root/name), field
    for receipt in (review, analysis):
        assert receipt['Q3_values_read'] is False and receipt['independent_confirmation'] is False
        assert receipt['research_goal_achieved'] is False
    assert set(review['tasks']) == {f'minute_h{h}_future_log_return' for h in HORIZONS}
    assert len(review['fold_checks']) == 20 and all(row['status'] == 'passed' for row in review['fold_checks'])
    assert {(row['horizon'], row['fold']) for row in review['fold_checks']} == {(h, f) for h in HORIZONS for f in range(1, 6)}
    return analysis, review


def replay(root, manifest, synthetic):
    analysis, review = metadata_gate(root, manifest, synthetic)
    source = root/'review/independent_statistics.py'
    spec = importlib.util.spec_from_file_location('bundled_independent_saved_loss_arithmetic', source)
    numerical = importlib.util.module_from_spec(spec); spec.loader.exec_module(numerical)
    assert Path(numerical.__file__).resolve().is_relative_to(root)
    assert numerical.HORIZONS == HORIZONS
    # Only pure arithmetic is used; do not invoke run_market, identity_gate,
    # checkpoint_metadata, data_for, or the original date-limited main function.
    numerical.guard = guard
    numerical.MAX_ERROR = 0.; numerical.COMPARISONS = 0
    official = read(root/'analysis/STATS.json'); results = read(root/'analysis/RESULTS.json')
    assert official['Q3_values_read'] is False and official['independent_confirmation'] is False
    assert official['candidate'] == numerical.CANDIDATE == 'tied_selected_shrink'
    assert results['bundles'] == 30 and results['fold_horizon_cells'] == 20
    assert results['statistics_sha256'] == digest(root/'analysis/STATS.json')
    assert len(results['metrics']) == 20
    cells = {(row['horizon'], row['fold']): row for row in results['metrics']}
    tasks = {}; checks = []
    for h in HORIZONS:
        task = f'minute_h{h}_future_log_return'; folds = []
        for fold in range(1, 6):
            guard(); cell = read(root/f'analysis/folds/{task}/fold_{fold}.json')
            assert cell == cells[(h, fold)] and cell['status'] == 'completed'
            arrays = {}
            for field, family in (('paired_loss_sha256', 'paired_losses'), ('path_loss_sha256', 'path_losses'),
                                  ('absolute_error_sha256', 'absolute_errors')):
                path = root/f'analysis/{family}/{task}/fold_{fold}.npz'
                assert cell[field] == digest(path)
                with np.load(path, allow_pickle=False) as saved:
                    arrays[family] = {name: saved[name].copy() for name in saved.files}
            endpoint = arrays['paired_losses']; origins = endpoint.pop('origins')
            dates = endpoint.pop('session_date'); grid = endpoint.pop('day_grid')
            assert origins.ndim == dates.ndim == grid.ndim == 1
            assert len(origins) == len(dates) == cell['origins'] > 0
            assert np.all(origins[1:] > origins[:-1])
            assert set(endpoint) == set(official['tasks'][task]['summary'])
            assert len(endpoint) == (76 if h == 60 else 43)
            expected_names = {'zero', 'fit_mean_raw', 'fit_mean_shrink'} | set(numerical.METHODS)
            expected_names |= {f'{method}_seed_{seed}' for method in numerical.METHODS for seed in numerical.SEEDS}
            expected_names |= {method+'_prediction_ensemble' for method in numerical.METHODS}
            if h == 60:
                legacy_names = {name for name in endpoint if name.startswith('legacy_v3b_')}
                assert len(legacy_names) == 33 and 'legacy_v3b_zero' in legacy_names
                assert np.array_equal(endpoint['legacy_v3b_zero'], endpoint['zero'])
                expected_names |= legacy_names
            assert set(endpoint) == expected_names
            for values in endpoint.values():
                assert values.shape == (len(origins),) and np.isfinite(values).all() and (values >= 0).all()
            paths = arrays['path_losses']; absolute = arrays['absolute_errors']
            assert np.array_equal(paths.pop('origins'), origins) and np.array_equal(absolute.pop('origins'), origins)
            assert set(paths) == set(absolute) and set(paths).issubset(endpoint)
            assert set(paths) == {name for name in expected_names if not name.startswith('legacy_v3b_')}
            assert len(paths) == 43 and len(cell['metrics']) == 43
            assert {row['method'] for row in cell['metrics']} == set(paths)
            for values in list(paths.values())+list(absolute.values()):
                assert values.shape == (len(origins),) and np.isfinite(values).all() and (values >= 0).all()
            for row in cell['metrics']:
                name = row['method']
                numerical.close(float(endpoint[name].mean()), row['endpoint_mse'], (task, fold, name, 'endpoint'))
                numerical.close(float(paths[name].mean()), row['path_mse'], (task, fold, name, 'path'))
                numerical.close(float(absolute[name].mean()), row['endpoint_mae'], (task, fold, name, 'absolute'))
            for family, values in (('endpoint', endpoint), ('path', paths), ('absolute', absolute)):
                for method in numerical.METHODS:
                    mean = sum((values[f'{method}_seed_{seed}'] for seed in numerical.SEEDS), np.zeros(len(origins)))/3.
                    numerical.close(values[method], mean, (task, fold, method, family, 'mean_seed_loss'))
            folds.append({'origins': origins, 'session_date': dates, 'day_grid': grid, 'losses': endpoint})
            checks.append({'horizon': h, 'fold': fold, 'status': 'passed', 'origins': len(origins),
                'endpoint_loss_columns': len(endpoint), 'path_absolute_columns': len(paths)})
        result = numerical.independent_summary(folds, h, repetitions=5000)
        numerical.compare_summary(result, official['tasks'][task], task+'/official')
        numerical.compare_summary(result, review['tasks'][task], task+'/independent_review')
        tasks[task] = result
    primary = tasks['minute_h60_future_log_return']
    assert primary['mechanism']['passed'] == official['primary_mechanism_gate_passed']
    assert primary['candidate_performance_gate_passed'] == official['primary_candidate_performance_gate_passed'] == analysis['predictive_gate_passed']
    return {'status': 'passed', 'scope': 'detached saved-loss arithmetic replication; not independent market confirmation',
        'synthetic_fixture': synthetic, 'tasks': tasks, 'fold_checks': checks,
        'repetitions_per_block': 5000, 'block_days': [1, 5], 'statistics_seed': 2026,
        'numerical_comparisons': numerical.COMPARISONS, 'maximum_absolute_difference': numerical.MAX_ERROR,
        'independent_arithmetic_script_sha256': digest(source),
        'date_specific_market_guard_not_called': True,
        'replay_guard': 'signals plus explicit deadline when supplied; otherwise offline replay has no expired market-date guard',
        'deadline_epoch': DEADLINE, 'GPU': False, 'raw_targets_or_market_predictions_loaded': False,
        'data_used': 'bundled per-origin endpoint/path/absolute losses plus origin/session day grids',
        'market_training_or_scoring': False, 'independent_confirmation': False, 'Q3_values_read': False,
        'research_goal_achieved': False}


def main():
    global DEADLINE
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--deny-path', action='append', default=[])
    parser.add_argument('--guard-self-test', action='store_true')
    parser.add_argument('--deadline-epoch', type=float)
    parser.add_argument('--allow-synthetic-fixture', action='store_true')
    args = parser.parse_args(); root = args.root.resolve(); DEADLINE = args.deadline_epoch
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    assert len(os.sched_getaffinity(0)) <= 2
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT): signal.signal(sig, interrupted)
    denied = tuple(Path(p).resolve() for p in args.deny_path)
    counts, tests = install_guard(denied, args.guard_self_test)
    manifest = read(root/'MANIFEST.json')
    for name, expected in manifest['files'].items():
        path = (root/name).resolve(); assert path.is_relative_to(root), name
        assert digest(path) == expected, f'Capsule file hash mismatch: {name}'
    result = replay(root, manifest, args.allow_synthetic_fixture)
    result.update(manifest_files_verified=len(manifest['files']), affinity=sorted(os.sched_getaffinity(0)),
        forbidden_roots=[str(p) for p in denied], audit_guard=counts, audit_guard_positive_controls=tests)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__': main()
