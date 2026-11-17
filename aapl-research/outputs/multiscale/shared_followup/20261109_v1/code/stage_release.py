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
import shutil
import subprocess
import sys
import tempfile
import zipfile

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[3]
REFERENCE = REPO/'outputs/multiscale/controlled_followup/20261108_v2/code/reference_source'
REFERENCE_FILES = ('multiscale/__init__.py', 'multiscale/models.py', 'aapl/__init__.py',
    'aapl/models.py', 'aapl/common.py', 'aapl/vendor/__init__.py',
    'aapl/vendor/revin.py', 'aapl/vendor/xpatch_network.py',
    'aapl/vendor/patchtst_backbone.py', 'aapl/vendor/patchtst_layers.py',
    'aapl/vendor/SOURCE.json', 'aapl/vendor/XPATCH_LICENSE.txt', 'aapl/vendor/PATCHTST_LICENSE.txt')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source, destination):
    for path in sorted(source.rglob('*')):
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
    assert rules['all_cells_required'] is True and rules['GPU'] is False
    assert rules['Q3_values_read'] is False and rules['independent_confirmation'] is False
    for name, expected_hash in rules['sources'].items():
        assert sha(BASE/'code'/name) == expected_hash, f'Frozen source changed: {name}'
    assert sha(BASE/'PLAN_ZH.md') == rules['plan_sha256']
    assert sha(BASE/'audit/FIT_ONLY_COST.json') == rules['cost_sha256']
    assert sha(BASE/'audit/CONCURRENT_FIT_COST.json') == rules['concurrent_cost_sha256']
    for job in rules['jobs']:
        folder = BASE/'runs'/job['id']; done = read(folder/'DONE.json')
        assert done['status'] == 'completed' and done['job'] == job
        assert done['rules_sha256'] == sha(rules_path)
        assert done['fixed_epochs'] == rules['fixed_epochs']
        assert done['Q3_values_read'] is False and done['independent_confirmation'] is False
        assert sha(folder/'predictions.npz') == done['prediction_sha256']
        for name in ('last.pt', 'best.pt', 'initial.pt'):
            assert sha(folder/name) == done['checkpoint_hashes'][name]
        assert (folder/'EXAMPLE.npz').is_file()
    assert analysis_receipt.resolve().is_relative_to((BASE/'analysis').resolve())
    analysis = read(analysis_receipt)
    assert analysis['status'] == 'completed', 'Analysis must explicitly report completed'
    assert analysis['rules_sha256'] == sha(rules_path), 'Analysis must use the frozen rules'
    assert analysis['complete_bundles'] == 30
    assert analysis['market_training_complete'] is True
    assert analysis['independent_confirmation'] is False and analysis['Q3_values_read'] is False
    assert analysis['all_five_folds_three_seeds'] is True
    assert 'predictive_gate_passed' in analysis and isinstance(analysis['predictive_gate_passed'], bool)
    assert analysis['files'], 'Analysis must hash its tables, intervals, and receipt inputs'
    for name, expected_hash in analysis['files'].items():
        path = (BASE/name).resolve()
        assert path.is_relative_to(BASE.resolve()) and path != analysis_receipt.resolve()
        assert sha(path) == expected_hash
    return rules, analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('ready', 'market'), required=True)
    parser.add_argument('--name')
    parser.add_argument('--analysis-receipt', type=Path, default=BASE/'analysis/ANALYSIS_COMPLETE.json')
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    assert set(os.sched_getaffinity(0)).issubset({24, 26})
    name = args.name or ('READY_CAPSULE' if args.mode == 'ready' else 'MARKET_CONTEXT')
    assert name and Path(name).name == name and name not in ('.', '..')
    destination = BASE/'release'/name; archive = destination.with_suffix('.zip')
    assert not destination.exists() and not archive.exists(), 'Never overwrite an existing release'
    rules = analysis = None
    if args.mode == 'market':
        rules, analysis = verify_complete_market(args.analysis_receipt)
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
        GPU=False, release_tool_sha256=sha(Path(__file__)),
        native_source_snapshots={name: sha(REFERENCE/name) for name in REFERENCE_FILES})
    write(destination/'BUNDLE_SCOPE.json', scope)
    write(destination/'MANIFEST.json', {'files': {str(path.relative_to(destination)): sha(path)
        for path in sorted(destination.rglob('*')) if path.is_file()}, 'market_bundles': count})
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(destination.rglob('*')):
            if path.is_file():
                zipped.write(path, str(path.relative_to(destination)))
    with tempfile.TemporaryDirectory(prefix='aapl-shared-capsule-extracted-') as temporary:
        outside = Path(temporary)
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(outside)
        output = outside/'REPLAY_ACCEPTANCE.json'
        env = dict(os.environ, PYTHONPATH='', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1')
        process = subprocess.run([sys.executable, str(outside/'replay.py'), '--root', str(outside),
            '--deny-path', str(REPO), '--guard-self-test', '--output', str(output)],
            cwd=outside, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
        assert process.returncode == 0, process.stderr
        replay = read(output)
    receipt = {'status': 'passed', 'mode': args.mode, 'scope': scope,
        'archive': str(archive), 'bytes': archive.stat().st_size, 'sha256': sha(archive),
        'external_replay': replay, 'market_training_started_by_release_tool': False}
    write(archive.with_suffix('.RELEASE.json'), receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
