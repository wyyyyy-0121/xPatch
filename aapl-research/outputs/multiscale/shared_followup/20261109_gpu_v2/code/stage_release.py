"""Stage readiness or complete-market capsules without starting experiments.

Readiness preserves synthetic engineering evidence only. Market mode fails
closed unless all frozen jobs and the explicit analysis-completion receipt
are present. This file is a release tool, outside training_sources().
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[3]
REFERENCE = REPO/'outputs/multiscale/controlled_followup/20261108_v2/code/reference_source'
REFERENCE_FILES = ('multiscale/__init__.py', 'multiscale/models.py', 'aapl/__init__.py',
    'aapl/models.py', 'aapl/common.py', 'aapl/vendor/__init__.py',
    'aapl/vendor/revin.py', 'aapl/vendor/xpatch_network.py',
    'aapl/vendor/patchtst_backbone.py', 'aapl/vendor/patchtst_layers.py',
    'aapl/vendor/SOURCE.json', 'aapl/vendor/XPATCH_LICENSE.txt', 'aapl/vendor/PATCHTST_LICENSE.txt')
STOP = False
DEADLINE = None


def guard():
    if STOP or (DEADLINE is not None and time.time() >= DEADLINE):
        raise InterruptedError('Release interrupted or authorized deadline reached')


def interrupted(*_):
    global STOP
    STOP = True


def sha(path):
    guard()
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            guard()
            value.update(block)
    return value.hexdigest()


def read(path):
    guard()
    return json.loads(Path(path).read_text())


def write(path, value):
    guard()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def copy(source, destination):
    guard()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    guard()


def copy_tree(source, destination):
    for path in sorted(source.rglob('*')):
        guard()
        if path.is_file() and '__pycache__' not in path.parts:
            copy(path, destination/path.relative_to(source))


def verify_complete_market(analysis_receipt):
    rules_path = BASE/'protocol/RULES.json'
    rules = read(rules_path)
    assert rules['admitted'] is True, 'Formal market release requires cost admission'
    assert len(rules['jobs']) == 30 and len({j['id'] for j in rules['jobs']}) == 30
    expected = {(arm, fold, seed) for arm in ('tied', 'untied')
                for fold in range(1, 6) for seed in (2026, 2027, 2028)}
    assert {(j['arm'], j['fold'], j['seed']) for j in rules['jobs']} == expected
    assert rules['all_cells_required'] is True and rules['GPU'] is True
    assert rules['fixed_epochs'] == 20
    assert rules['horizons'] == [5, 15, 30, 60]
    assert rules['folds'] == [1, 2, 3, 4, 5] and rules['seeds'] == [2026, 2027, 2028]
    assert rules['Q3_values_read'] is False and rules['independent_confirmation'] is False
    for name, expected_hash in rules['sources'].items():
        assert sha(BASE/'code'/name) == expected_hash, f'Frozen source changed: {name}'
    assert sha(BASE/'PLAN_ZH.md') == rules['plan_sha256']
    assert sha(BASE/rules['cost_file']) == rules['cost_sha256']
    assert sha(BASE/'protocol/AUTHORIZATION.json') == rules['authorization_sha256']
    rule_hash = sha(rules_path)
    required_evaluation_inputs = {}
    paired_steps = {}
    for job in rules['jobs']:
        folder = BASE/'runs'/job['id']; done = read(folder/'DONE.json')
        assert done['status'] == 'completed' and done['job'] == job
        assert done['rules_sha256'] == rule_hash
        identity = hashlib.sha256(json.dumps({'job': job, 'rule': rule_hash}, sort_keys=True).encode()).hexdigest()
        assert done['identity'] == identity
        assert done['fixed_epochs'] == rules['fixed_epochs']
        assert [row['epoch'] for row in done['history']] == list(range(1, rules['fixed_epochs']+1))
        assert done['history'][-1]['steps'] == done['steps']
        paired_steps.setdefault((job['fold'], job['seed']), []).append(done['steps'])
        assert done['Q3_values_read'] is False and done['independent_confirmation'] is False
        assert sha(folder/'predictions.npz') == done['prediction_sha256']
        assert set(done['checkpoint_hashes']) == {'last.pt', 'best.pt', 'initial.pt'}
        for name in ('last.pt', 'best.pt', 'initial.pt'):
            assert sha(folder/name) == done['checkpoint_hashes'][name]
            required_evaluation_inputs[str(folder/name)] = done['checkpoint_hashes'][name]
        required_evaluation_inputs[str(folder/'DONE.json')] = sha(folder/'DONE.json')
        required_evaluation_inputs[str(folder/'predictions.npz')] = done['prediction_sha256']
        assert (folder/'EXAMPLE.npz').is_file()
    assert all(len(values) == 2 and values[0] == values[1] for values in paired_steps.values())
    assert analysis_receipt.resolve().is_relative_to((BASE/'analysis').resolve())
    analysis = read(analysis_receipt)
    assert analysis['status'] == 'completed', 'Analysis must explicitly report completed'
    assert analysis['rules_sha256'] == sha(rules_path), 'Analysis must use the frozen rules'
    assert analysis['complete_bundles'] == 30
    assert analysis['market_training_complete'] is True
    assert analysis['independent_confirmation'] is False and analysis['Q3_values_read'] is False
    assert analysis['all_five_folds_three_seeds'] is True
    assert 'predictive_gate_passed' in analysis and isinstance(analysis['predictive_gate_passed'], bool)
    assert analysis['research_goal_achieved'] is False
    assert analysis['files'], 'Analysis must hash its tables, intervals, and receipt inputs'
    essential = {'analysis/STATS.json', 'analysis/RESULTS.json'}
    cells = {(h, fold) for h in (5, 15, 30, 60) for fold in range(1, 6)}
    for h, fold in cells:
        task = f'minute_h{h}_future_log_return'
        essential.add(f'analysis/folds/{task}/fold_{fold}.json')
        for family in ('paired_losses', 'path_losses', 'absolute_errors'):
            essential.add(f'analysis/{family}/{task}/fold_{fold}.npz')
    assert essential.issubset(analysis['files']), 'All 20 cells and all three loss families are required'
    for name, expected_hash in analysis['files'].items():
        path = (BASE/name).resolve()
        assert path.is_relative_to(BASE.resolve()) and path != analysis_receipt.resolve()
        assert sha(path) == expected_hash
    statistics_path = BASE/'analysis/STATS.json'
    results_path = BASE/'analysis/RESULTS.json'
    freeze_path = BASE/'protocol/EVALUATION_FREEZE.json'
    assert analysis['statistics_sha256'] == sha(statistics_path)
    assert analysis['results_sha256'] == sha(results_path)
    assert analysis['evaluation_freeze_sha256'] == sha(freeze_path)
    evaluation = read(freeze_path)
    assert evaluation['status'] == 'all_30_bundles_fixed_before_new_outer_scores'
    assert evaluation['rules_sha256'] == rule_hash
    assert evaluation['evaluator_sha256'] == sha(BASE/'code/evaluate_shared.py')
    assert all(evaluation['input_hashes'].get(name) == value
               for name, value in required_evaluation_inputs.items())
    stats = read(statistics_path); results = read(results_path)
    assert stats['status'] == 'complete_30_bundle_five_fold_three_seed_development_analysis'
    assert stats['rules_sha256'] == rule_hash
    assert stats['evaluation_freeze_sha256'] == sha(freeze_path)
    assert stats['candidate'] == evaluation['candidate'] == 'tied_selected_shrink'
    assert stats['primary_task'] == 'minute_h60_future_log_return'
    assert set(stats['tasks']) == {f'minute_h{h}_future_log_return' for h in (5, 15, 30, 60)}
    assert isinstance(stats['primary_mechanism_gate_passed'], bool)
    assert stats['primary_candidate_performance_gate_passed'] is analysis['predictive_gate_passed']
    assert stats['tasks'][stats['primary_task']]['candidate_performance_gate_passed'] is analysis['predictive_gate_passed']
    assert stats['research_goal_achieved'] is False and stats['hourly_primary_goal_completed_by_this_study'] is False
    assert results['status'] == 'completed' and results['fold_horizon_cells'] == 20 and results['bundles'] == 30
    assert results['statistics_sha256'] == sha(statistics_path)
    assert len(results['metrics']) == 20
    assert {(row['horizon'], row['fold']) for row in results['metrics']} == cells
    for receipt in (stats, results):
        assert receipt['Q3_values_read'] is False and receipt['independent_confirmation'] is False
    for row in results['metrics']:
        h, fold = row['horizon'], row['fold']; task = f'minute_h{h}_future_log_return'
        cell = read(BASE/f'analysis/folds/{task}/fold_{fold}.json')
        assert cell == row and cell['status'] == 'completed' and cell['origins'] > 0
        assert cell['Q3_values_read'] is False and cell['evaluation_freeze_sha256'] == sha(freeze_path)
        for field, family in (('paired_loss_sha256', 'paired_losses'),
                              ('path_loss_sha256', 'path_losses'),
                              ('absolute_error_sha256', 'absolute_errors')):
            assert cell[field] == sha(BASE/f'analysis/{family}/{task}/fold_{fold}.npz')
    return rules, analysis


def verify_independent_statistics():
    complete = read(BASE/'audit/COMPLETE.json')
    assert complete['status'] == 'completed' and complete['bundles'] == 30 and complete['Q3_values_read'] is False
    review = read(BASE/'review/INDEPENDENT_STATISTICS_REVIEW.json')
    assert review['status'] == 'passed'
    for field, name in (('rules_sha256', 'protocol/RULES.json'),
                        ('analysis_complete_sha256', 'analysis/ANALYSIS_COMPLETE.json'),
                        ('evaluation_freeze_sha256', 'protocol/EVALUATION_FREEZE.json'),
                        ('script_sha256', 'review/independent_statistics.py')):
        assert review[field] == sha(BASE/name), field
    assert review['Q3_values_read'] is False and review['independent_confirmation'] is False
    assert review['research_goal_achieved'] is False
    assert set(review['tasks']) == {f'minute_h{h}_future_log_return' for h in (5, 15, 30, 60)}
    assert len(review['fold_checks']) == 20
    assert {(row['horizon'], row['fold']) for row in review['fold_checks']} == {
        (h, f) for h in (5, 15, 30, 60) for f in range(1, 6)}
    assert all(row['status'] == 'passed' for row in review['fold_checks'])
    return review


def main():
    global DEADLINE
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('ready', 'market'), required=True)
    parser.add_argument('--name')
    parser.add_argument('--analysis-receipt', type=Path, default=BASE/'analysis/ANALYSIS_COMPLETE.json')
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    assert set(os.sched_getaffinity(0)).issubset({24, 26})
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT): signal.signal(sig, interrupted)
    name = args.name or ('READY_CAPSULE' if args.mode == 'ready' else 'MARKET_CONTEXT')
    assert name and Path(name).name == name and name not in ('.', '..')
    destination = BASE/'release'/name; archive = destination.with_suffix('.zip')
    assert not destination.exists() and not archive.exists(), 'Never overwrite an existing release'
    rules = analysis = None
    if args.mode == 'market':
        authorization = read(BASE/'protocol/AUTHORIZATION.json')
        DEADLINE = float(authorization['deadline_epoch'])
        assert DEADLINE == float(read(BASE/'protocol/RULES.json')['deadline_epoch'])
        guard()
        rules, analysis = verify_complete_market(args.analysis_receipt)
        verify_independent_statistics()
    else:
        acceptance = read(BASE/'review/PORTABLE_FOUR_HEAD_ACCEPTANCE.json')
        assert acceptance['status'] == 'passed' and acceptance['market_training_or_scoring'] is False
        assert acceptance['research_experiment_complete'] is False
        assert acceptance['sources']['shared_network.py'] == sha(BASE/'code/shared_network.py')
        readiness_path = BASE/'audit/READINESS.json'
        readiness = read(readiness_path)
        assert readiness['formal_market_training_started'] is False
        assert readiness['new_market_stop_or_outer_scoring_started'] is False
        assert readiness['research_goal_achieved'] is False and readiness['goal_status'] == 'active'
    destination.mkdir(parents=True)
    for path in sorted((BASE/'code').glob('*.py')):
        copy(path, destination/'code'/path.name)
    for name in REFERENCE_FILES:
        copy(REFERENCE/name, destination/'code/reference_source'/name)
    copy(BASE/'release_tools/replay.py', destination/'replay.py')
    copy(BASE/'release_tools/replay_statistics.py', destination/'replay_statistics.py')
    copy(BASE/'PLAN_ZH.md', destination/'PLAN_ZH.md')
    for path in sorted((BASE/'review').glob('*')):
        if path.is_file() and path.suffix in ('.json', '.md', '.py'):
            copy(path, destination/'review'/path.name)
    for directory in ('audit', 'protocol'):
        if (BASE/directory).exists():
            copy_tree(BASE/directory, destination/directory)
    if args.mode == 'ready':
        # Runs in the readiness capsule are unmistakably synthetic fixtures.
        fixture = BASE/'review/synthetic_portable_bundle_v2'
        copy_tree(fixture/'runs', destination/'runs')
        recovery_source = BASE/'review/synthetic_only_retry_2'
        recovery_destination = destination/'review/synthetic_recovery'
        copy_tree(recovery_source, recovery_destination)
        recovery_review = read(destination/'review/SYNTHETIC_EXPERIMENT_REVIEW.json')
        recovery_review['resume']['evidence_root'] = 'review/synthetic_recovery'
        write(destination/'review/SYNTHETIC_EXPERIMENT_REVIEW.json', recovery_review)
        synthetic_review_sources = {
            'SYNTHETIC_EXPERIMENT_REVIEW.json': sha(destination/'review/SYNTHETIC_EXPERIMENT_REVIEW.json'),
            'FROZEN_SIGNATURE_REVIEW.json': sha(destination/'review/FROZEN_SIGNATURE_REVIEW.json'),
            'EXPERIMENT_REVIEW_ZH.md': sha(destination/'review/EXPERIMENT_REVIEW_ZH.md')}
        count = 0
        scope = {'status': 'engineering_ready_only_no_formal_market_runs',
            'formal_market_bundles': 0, 'synthetic_example_bundles': 2,
            'research_experiment_complete': False, 'predictive_gate_passed': False,
            'predictive_claim': 'No market training or predictive improvement claim is made',
            'independent_confirmation': False, 'Q3_values_read': False,
            'cost_admitted': False, 'formal_market_training_started': False,
            'new_market_stop_or_outer_scoring_started': False,
            'readiness_snapshot_sha256': sha(BASE/'audit/READINESS.json'),
            'synthetic_review_sources': synthetic_review_sources,
            'fit_only_probe_weights_retained_as_candidate': False}
    else:
        for job in rules['jobs']:
            for name in ('last.pt', 'best.pt', 'initial.pt', 'EXAMPLE.npz', 'DONE.json', 'predictions.npz'):
                copy(BASE/'runs'/job['id']/name, destination/'runs'/job['id']/name)
        copy_tree(BASE/'analysis', destination/'analysis')
        count = 30
        scope = {'status': 'complete_frozen_historical_market_development_matrix',
            'formal_market_bundles': 30, 'synthetic_example_bundles': 0,
            'research_experiment_complete': True,
            'predictive_gate_passed': analysis['predictive_gate_passed'],
            'predictive_claim': 'Use the complete analysis receipt; completion alone implies no predictive gain',
            'independent_confirmation': False, 'Q3_values_read': False}
    scope.update(original_title='Multi-Horizon Forecasting of Apple Stock Prices and Returns Across Temporal Scales',
        GPU=True, release_tool_sha256=sha(Path(__file__)),
        native_source_snapshots={name: sha(REFERENCE/name) for name in REFERENCE_FILES})
    write(destination/'BUNDLE_SCOPE.json', scope)
    (destination/'REPLAY_README.md').write_text('''# Detached reproduction

Use Python with NumPy and PyTorch for input-only inference, or NumPy alone for
saved-loss statistics. Restrict execution to two logical CPUs and set
CUDA_VISIBLE_DEVICES=-1, OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=2.

Run `python replay.py --root . --output INFERENCE_REPLAY.json` for input-only
weights replay. Run `python replay_statistics.py --root . --output STATISTICS_REPLAY.json`
for all four horizons, five folds, 5,000 bootstrap draws, one-/five-day blocks,
mean seed losses, full registered controls and both separate decision gates.
Add `--deny-path ORIGINAL_WORKSPACE --guard-self-test` to test detached access
guards. Both entrypoints reject file-manifest hash mismatches.

The statistics replay uses bundled saved endpoint/path/absolute losses and day
grids, not raw labels or loaded market prediction matrices. It wraps the
unchanged independent numerical source and replaces only its in-memory guard;
the original absolute market-authorization deadline does not prevent future
offline replay. An optional `--deadline-epoch` limits a replay run; today's
release acceptance passes the original authorized deadline explicitly.
This is saved-loss numerical reproduction on exposed development history,
not independent market confirmation or evidence of predictive improvement.
''')
    write(destination/'MANIFEST.json', {'files': {str(path.relative_to(destination)): sha(path)
        for path in sorted(destination.rglob('*')) if path.is_file()}, 'market_bundles': count})
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(destination.rglob('*')):
            guard()
            if path.is_file():
                zipped.write(path, str(path.relative_to(destination)),
                    compress_type=zipfile.ZIP_STORED if path.suffix == '.npz' else zipfile.ZIP_DEFLATED)
            guard()
    statistics_replay = None
    with tempfile.TemporaryDirectory(prefix='aapl-shared-capsule-extracted-') as temporary:
        outside = Path(temporary)
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                guard(); zipped.extract(member, outside); guard()
        output = outside/'REPLAY_ACCEPTANCE.json'
        env = dict(os.environ, PYTHONPATH='', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1')
        guard()
        process = subprocess.run([sys.executable, str(outside/'replay.py'), '--root', str(outside),
            '--deny-path', str(REPO), '--guard-self-test', '--output', str(output)],
            cwd=outside, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
        guard()
        assert process.returncode == 0, process.stderr
        replay = read(output)
        if args.mode == 'market':
            output = outside/'STATISTICS_REPLAY_ACCEPTANCE.json'
            process = subprocess.run([sys.executable, str(outside/'replay_statistics.py'), '--root', str(outside),
                '--deny-path', str(REPO), '--guard-self-test', '--deadline-epoch', str(rules['deadline_epoch']),
                '--output', str(output)], cwd=outside, env=env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
            guard()
            assert process.returncode == 0, process.stderr
            statistics_replay = read(output)
    receipt = {'status': 'passed', 'mode': args.mode, 'scope': scope,
        'archive': str(archive), 'bytes': archive.stat().st_size, 'sha256': sha(archive),
        'external_replay': replay, 'external_saved_loss_statistics_replay': statistics_replay,
        'market_training_started_by_release_tool': False}
    guard()
    write(archive.with_suffix('.RELEASE.json'), receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
