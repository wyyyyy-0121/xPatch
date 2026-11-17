"""Generated saved-loss capsule test; never reads running market artifacts."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import numpy as np

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[3]
os.sched_setaffinity(0, {24, 26}); os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result); return result


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2)+'\n')


def update_manifest(root):
    write(root/'MANIFEST.json', {'market_bundles': 0, 'files': {str(p.relative_to(root)): sha(p)
        for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'MANIFEST.json'}})


def fixture(root):
    old = module(BASE/'review/release_tool_acceptance.py', 'synthetic_metadata_factory')
    old.market_fixture(root)
    for path in (root/'runs').glob('*/DONE.json'):
        done = read(path)
        for row in done['history']:
            row['aggregate_standardized_inner_endpoint_mse'] = 1./row['epoch']
        done['best_epoch'] = 20
        done['alphas'] = {mode: {str(h): .5 for h in (5, 15, 30, 60)} for mode in ('last', 'selected')}
        write(path, done)
    freeze = read(root/'protocol/EVALUATION_FREEZE.json')
    freeze.update(horizons=[5, 15, 30, 60], seeds=[2026, 2027, 2028], blocks=[1, 5],
                  repetitions=5000, statistics_seed=2026, legacy_h60_columns=33)
    freeze['input_hashes'] = {name: sha(name) for name in freeze['input_hashes']}
    write(root/'protocol/EVALUATION_FREEZE.json', freeze)
    (root/'review').mkdir()
    shutil.copy2(BASE/'review/independent_statistics.py', root/'review/independent_statistics.py')
    shutil.copy2(BASE/'release_tools/replay_statistics.py', root/'replay_statistics.py')
    numerical = module(root/'review/independent_statistics.py', 'generated_statistics_expected')
    numerical.guard = lambda: None
    rng = np.random.default_rng(719); tasks = {}; rows = []; checks = []
    freeze_hash = sha(root/'protocol/EVALUATION_FREEZE.json')
    for h in numerical.HORIZONS:
        folds = []; task = f'minute_h{h}_future_log_return'
        for fold in range(1, 6):
            n = 24; actual = rng.normal(0., .01, (n, h)); predicted = {}
            for method in numerical.METHODS:
                factor = .95 if method == numerical.CANDIDATE else .8 if method == 'tied_last' else .5
                for seed in numerical.SEEDS:
                    predicted[(method, seed)] = factor*actual+rng.normal(0, .0001, (n, h))
            legacy = None
            if h == 60:
                legacy = {'zero': actual[:, -1]**2}
                legacy.update({f'generated_control_{i:02d}': (2+i/10)*actual[:, -1]**2+1e-9 for i in range(32)})
            losses, paths, absolute = numerical.independent_losses(predicted, actual, np.full(h, .02), .5, legacy)
            origins = np.arange(n)+fold*100
            grid = np.array([f'2025-01-{day+1:02d}' for day in range(8)])
            dates = np.repeat(grid, 3)
            for family, values in (('paired_losses', dict(origins=origins, session_date=dates, day_grid=grid, **losses)),
                                   ('path_losses', dict(origins=origins, **paths)),
                                   ('absolute_errors', dict(origins=origins, **absolute))):
                np.savez(root/f'analysis/{family}/{task}/fold_{fold}.npz', **values)
            row = {'status': 'completed', 'horizon': h, 'fold': fold, 'origins': n,
                'evaluation_freeze_sha256': freeze_hash, 'Q3_values_read': False,
                'metrics': [{'method': name, 'endpoint_mse': float(losses[name].mean()),
                    'path_mse': float(paths[name].mean()), 'endpoint_mae': float(absolute[name].mean())} for name in paths]}
            for field, family in (('paired_loss_sha256', 'paired_losses'), ('path_loss_sha256', 'path_losses'),
                                  ('absolute_error_sha256', 'absolute_errors')):
                row[field] = sha(root/f'analysis/{family}/{task}/fold_{fold}.npz')
            write(root/f'analysis/folds/{task}/fold_{fold}.json', row); rows.append(row)
            folds.append({'origins': origins, 'session_date': dates, 'day_grid': grid, 'losses': losses})
            checks.append({'horizon': h, 'fold': fold, 'status': 'passed'})
        tasks[task] = numerical.independent_summary(folds, h, repetitions=5000)
    stats = read(root/'analysis/STATS.json'); stats['tasks'] = tasks
    primary = tasks['minute_h60_future_log_return']
    stats['primary_mechanism_gate_passed'] = primary['mechanism']['passed']
    stats['primary_candidate_performance_gate_passed'] = primary['candidate_performance_gate_passed']
    write(root/'analysis/STATS.json', stats)
    results = read(root/'analysis/RESULTS.json'); results.update(metrics=rows, statistics_sha256=sha(root/'analysis/STATS.json'))
    write(root/'analysis/RESULTS.json', results)
    analysis = read(root/'analysis/ANALYSIS_COMPLETE.json')
    analysis.update(statistics_sha256=sha(root/'analysis/STATS.json'), results_sha256=sha(root/'analysis/RESULTS.json'),
        evaluation_freeze_sha256=freeze_hash, predictive_gate_passed=primary['candidate_performance_gate_passed'])
    analysis['files'] = {name: sha(root/name) for name in analysis['files']}
    write(root/'analysis/ANALYSIS_COMPLETE.json', analysis)
    write(root/'audit/COMPLETE.json', {'status': 'completed', 'bundles': 30, 'Q3_values_read': False})
    review = {'status': 'passed', 'rules_sha256': sha(root/'protocol/RULES.json'),
        'analysis_complete_sha256': sha(root/'analysis/ANALYSIS_COMPLETE.json'),
        'evaluation_freeze_sha256': sha(root/'protocol/EVALUATION_FREEZE.json'),
        'script_sha256': sha(root/'review/independent_statistics.py'), 'tasks': tasks, 'fold_checks': checks,
        'Q3_values_read': False, 'independent_confirmation': False, 'research_goal_achieved': False}
    write(root/'review/INDEPENDENT_STATISTICS_REVIEW.json', review)
    write(root/'BUNDLE_SCOPE.json', {'formal_market_bundles': 0, 'research_experiment_complete': False,
        'synthetic_statistics_fixture': True, 'Q3_values_read': False, 'independent_confirmation': False})
    update_manifest(root)


def run(root, output, extra=()):
    env = dict(os.environ, PYTHONPATH='', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    return subprocess.run([sys.executable, str(root/'replay_statistics.py'), '--root', str(root),
        '--output', str(output), '--deny-path', str(REPO), '--guard-self-test', '--allow-synthetic-fixture', *extra],
        cwd=root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)


def main():
    report = {'scope': 'generated arrays only; no running market arrays or scores read',
        'GPU': False, 'Q3_values_read': False, 'market_arrays_or_results_read': False,
        'script_sha256': sha(BASE/'release_tools/replay_statistics.py'), 'rejections': {}}
    with tempfile.TemporaryDirectory(prefix='aapl_SYNTHETIC_saved_loss_') as temporary:
        root = Path(temporary)/'extracted'; fixture(root)
        completed = run(root, Path(temporary)/'statistics.json')
        assert completed.returncode == 0, completed.stderr
        replay = read(Path(temporary)/'statistics.json')
        assert len(replay['tasks']) == 4 and len(replay['fold_checks']) == 20
        assert replay['repetitions_per_block'] == 5000 and replay['block_days'] == [1, 5]
        report['detached_saved_loss_replay'] = replay
        for case in ('expired_explicit_deadline', 'manifest_tamper', 'missing_loss_file', 'changed_gate_with_rehashed_closure'):
            broken = Path(temporary)/case; shutil.copytree(root, broken); extra = ()
            if case == 'expired_explicit_deadline': extra = ('--deadline-epoch', '1')
            elif case == 'manifest_tamper': (broken/'audit/COMPLETE.json').write_text('{}')
            elif case == 'missing_loss_file':
                path = broken/'analysis/paired_losses/minute_h5_future_log_return/fold_1.npz'
                path.unlink(); update_manifest(broken)
            else:
                path = broken/'analysis/STATS.json'; value = read(path)
                value['primary_mechanism_gate_passed'] = not value['primary_mechanism_gate_passed']; write(path, value)
                path = broken/'analysis/RESULTS.json'; value = read(path)
                value['statistics_sha256'] = sha(broken/'analysis/STATS.json'); write(path, value)
                path = broken/'analysis/ANALYSIS_COMPLETE.json'; value = read(path)
                value['statistics_sha256'] = sha(broken/'analysis/STATS.json')
                value['results_sha256'] = sha(broken/'analysis/RESULTS.json')
                value['files'] = {name: sha(broken/name) for name in value['files']}; write(path, value)
                path = broken/'review/INDEPENDENT_STATISTICS_REVIEW.json'; value = read(path)
                value['analysis_complete_sha256'] = sha(broken/'analysis/ANALYSIS_COMPLETE.json'); write(path, value)
                update_manifest(broken)
            completed = run(broken, Path(temporary)/f'{case}.json', extra)
            assert completed.returncode != 0, f'Invalid capsule accepted: {case}'
            report['rejections'][case] = {'rejected': True, 'last_error': completed.stderr.strip().splitlines()[-1]}
    stage = module(BASE/'code/stage_release.py', 'generated_release_deadline_guard')
    stage_checks = {}
    with tempfile.TemporaryDirectory(prefix='aapl_SYNTHETIC_release_deadline_') as temporary:
        root = Path(temporary); original = root/'input'; original.write_bytes(b'generated')
        stage.DEADLINE = 1.
        for name, operation in (
            ('hash_after_deadline', lambda: stage.sha(original)),
            ('copy_after_deadline', lambda: stage.copy(original, root/'copy')),
            ('receipt_after_deadline', lambda: stage.write(root/'PASSED.json', {'status': 'passed'}))):
            try: operation()
            except InterruptedError: stage_checks[name] = 'rejected'
            else: raise AssertionError(f'Deadline guard did not reject {name}')
        stage.DEADLINE = None; stage.interrupted()
        try: stage.write(root/'SIGNAL_PASSED.json', {'status': 'passed'})
        except InterruptedError: stage_checks['receipt_after_signal'] = 'rejected'
        else: raise AssertionError('Signal guard did not stop receipt')
        assert not (root/'PASSED.json').exists() and not (root/'SIGNAL_PASSED.json').exists()
    report['stage_deadline_signal_guards'] = stage_checks
    report['stage_release_sha256'] = sha(BASE/'code/stage_release.py')
    report['status'] = 'passed'
    write(BASE/'review/DETACHED_STATISTICS_REPLAY_ACCEPTANCE.json', report)
    print(json.dumps({'status': 'passed', 'source_sha256': report['script_sha256'],
        'comparisons': replay['numerical_comparisons'], 'maximum_absolute_difference': replay['maximum_absolute_difference'],
        'tasks': 4, 'folds': 20, 'rejections': report['rejections']}), flush=True)


if __name__ == '__main__': main()
