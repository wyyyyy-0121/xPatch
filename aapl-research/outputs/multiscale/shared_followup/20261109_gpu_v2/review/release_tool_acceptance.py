"""CPU-only release acceptance using existing synthetic GPU weights and fake metadata.

Never imports market training, opens market arrays, or initializes CUDA.
The fake market gate fixture contains byte placeholders rather than market data.
"""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

os.sched_setaffinity(0, {24, 26})
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[3]
REFERENCE = REPO/'outputs/multiscale/controlled_followup/20261108_v2/code/reference_source'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def read(path):
    return json.loads(path.read_text())


def touch(path, value=b'synthetic-placeholder-not-an-array'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def manifest(root):
    write(root/'MANIFEST.json', {'market_bundles': 0,
        'files': {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*'))
                  if p.is_file() and p.name != 'MANIFEST.json'}})


def invoke(root, output):
    env = dict(os.environ, PYTHONPATH='', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    process = subprocess.run([sys.executable, str(root/'replay.py'), '--root', str(root),
        '--deny-path', str(REPO), '--guard-self-test', '--output', str(output)],
        cwd=root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    return process


def market_fixture(root):
    rules = {'admitted': True, 'fixed_epochs': 20, 'all_cells_required': True,
             'GPU': True, 'Q3_values_read': False, 'independent_confirmation': False,
             'horizons': [5, 15, 30, 60], 'folds': [1, 2, 3, 4, 5],
             'seeds': [2026, 2027, 2028], 'jobs': []}
    for name in ('PLAN_ZH.md', 'code/evaluate_shared.py', 'protocol/AUTHORIZATION.json', 'audit/COST.json'):
        touch(root/name)
    rules.update(sources={'evaluate_shared.py': sha(root/'code/evaluate_shared.py')},
                 plan_sha256=sha(root/'PLAN_ZH.md'), cost_file='audit/COST.json',
                 cost_sha256=sha(root/'audit/COST.json'),
                 authorization_sha256=sha(root/'protocol/AUTHORIZATION.json'))
    for arm in ('tied', 'untied'):
        for fold in range(1, 6):
            for seed in (2026, 2027, 2028):
                rules['jobs'].append({'id': f'{arm}_f{fold}_s{seed}', 'arm': arm, 'fold': fold, 'seed': seed})
    write(root/'protocol/RULES.json', rules)
    rule_hash = sha(root/'protocol/RULES.json'); inputs = {}
    for job in rules['jobs']:
        folder = root/'runs'/job['id']
        for name in ('initial.pt', 'last.pt', 'best.pt', 'EXAMPLE.npz', 'predictions.npz'):
            touch(folder/name)
        done = {'status': 'completed', 'job': job,
            'identity': hashlib.sha256(json.dumps({'job': job, 'rule': rule_hash}, sort_keys=True).encode()).hexdigest(),
            'rules_sha256': rule_hash, 'fixed_epochs': 20, 'steps': 80,
            'history': [{'epoch': epoch, 'steps': epoch*4} for epoch in range(1, 21)],
            'Q3_values_read': False, 'independent_confirmation': False,
            'prediction_sha256': sha(folder/'predictions.npz'),
            'checkpoint_hashes': {name: sha(folder/name) for name in ('initial.pt', 'last.pt', 'best.pt')}}
        write(folder/'DONE.json', done)
        for name in ('initial.pt', 'last.pt', 'best.pt', 'DONE.json', 'predictions.npz'):
            inputs[str(folder/name)] = sha(folder/name)
    freeze = {'status': 'all_30_bundles_fixed_before_new_outer_scores', 'rules_sha256': rule_hash,
        'evaluator_sha256': sha(root/'code/evaluate_shared.py'), 'input_hashes': inputs,
        'candidate': 'tied_selected_shrink'}
    write(root/'protocol/EVALUATION_FREEZE.json', freeze)
    freeze_hash = sha(root/'protocol/EVALUATION_FREEZE.json'); rows = []
    for horizon in (5, 15, 30, 60):
        for fold in range(1, 6):
            task = f'minute_h{horizon}_future_log_return'
            row = {'status': 'completed', 'horizon': horizon, 'fold': fold, 'origins': 1,
                   'Q3_values_read': False, 'evaluation_freeze_sha256': freeze_hash}
            for field, family in (('paired_loss_sha256', 'paired_losses'), ('path_loss_sha256', 'path_losses'),
                                  ('absolute_error_sha256', 'absolute_errors')):
                path = root/f'analysis/{family}/{task}/fold_{fold}.npz'
                touch(path); row[field] = sha(path)
            write(root/f'analysis/folds/{task}/fold_{fold}.json', row); rows.append(row)
    stats = {'status': 'complete_30_bundle_five_fold_three_seed_development_analysis',
        'rules_sha256': rule_hash, 'evaluation_freeze_sha256': freeze_hash,
        'candidate': 'tied_selected_shrink', 'primary_task': 'minute_h60_future_log_return',
        'primary_mechanism_gate_passed': False, 'primary_candidate_performance_gate_passed': False,
        'tasks': {f'minute_h{h}_future_log_return': {'candidate_performance_gate_passed': False}
                  for h in (5, 15, 30, 60)},
        'research_goal_achieved': False, 'hourly_primary_goal_completed_by_this_study': False,
        'Q3_values_read': False, 'independent_confirmation': False}
    write(root/'analysis/STATS.json', stats)
    results = {'status': 'completed', 'fold_horizon_cells': 20, 'bundles': 30, 'metrics': rows,
               'statistics_sha256': sha(root/'analysis/STATS.json'),
               'Q3_values_read': False, 'independent_confirmation': False}
    write(root/'analysis/RESULTS.json', results)
    analysis = {'status': 'completed', 'rules_sha256': rule_hash, 'complete_bundles': 30,
        'market_training_complete': True, 'all_five_folds_three_seeds': True,
        'predictive_gate_passed': False, 'research_goal_achieved': False,
        'Q3_values_read': False, 'independent_confirmation': False,
        'statistics_sha256': sha(root/'analysis/STATS.json'), 'results_sha256': sha(root/'analysis/RESULTS.json'),
        'evaluation_freeze_sha256': freeze_hash,
        'files': {str(p.relative_to(root)): sha(p) for p in sorted((root/'analysis').rglob('*')) if p.is_file()}}
    write(root/'analysis/ANALYSIS_COMPLETE.json', analysis)


def main():
    for source in (BASE/'code/stage_release.py', BASE/'release_tools/replay.py'):
        ast.parse(source.read_text())
    spec = importlib.util.spec_from_file_location('release_review_target', BASE/'code/stage_release.py')
    stage = importlib.util.module_from_spec(spec); spec.loader.exec_module(stage)
    acceptance = {'scope': 'existing_synthetic_GPU_weights_and_fake_metadata_only',
        'GPU': False, 'market_outer_arrays_read': False, 'market_training_or_scoring': False,
        'Q3_values_read': False, 'affinity': sorted(os.sched_getaffinity(0)),
        'sources': {str(p.relative_to(BASE)): sha(p) for p in (BASE/'code/stage_release.py', BASE/'release_tools/replay.py')},
        'rejection_checks': {}, 'synthetic_gate_checks': {}}
    with tempfile.TemporaryDirectory(prefix='aapl-release-review-') as temporary:
        temp = Path(temporary); source = temp/'source'; source.mkdir()
        shutil.copy2(BASE/'release_tools/replay.py', source/'replay.py')
        (source/'code').mkdir()
        shutil.copy2(BASE/'code/shared_network.py', source/'code/shared_network.py')
        for name in stage.REFERENCE_FILES:
            target = source/'code/reference_source'/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REFERENCE/name, target)
        evidence = Path(read(BASE/'review/GPU_SYNTHETIC_ACCEPTANCE.json')['actual_process_resume']['evidence_root'])
        for arm in ('tied', 'untied'):
            job = f'{arm}_synthetic_gpu'; folder = source/'runs'/job; folder.mkdir(parents=True)
            original = evidence/arm/'uninterrupted/runs'/job
            for name in ('initial.pt', 'last.pt', 'best.pt', 'EXAMPLE.npz', 'DONE.json'):
                shutil.copy2(original/name, folder/name)
            done = read(folder/'DONE.json'); done['synthetic_only'] = True; write(folder/'DONE.json', done)
        write(source/'BUNDLE_SCOPE.json', {'status': 'synthetic_engineering_fixture_only',
            'formal_market_bundles': 0, 'synthetic_example_bundles': 2,
            'research_experiment_complete': False, 'independent_confirmation': False, 'Q3_values_read': False})
        manifest(source)
        archive = temp/'synthetic.zip'
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as zipped:
            for path in source.rglob('*'):
                if path.is_file(): zipped.write(path, str(path.relative_to(source)))
        outside = temp/'extracted'; outside.mkdir()
        with zipfile.ZipFile(archive) as zipped: zipped.extractall(outside)
        process = invoke(outside, temp/'CPU_REPLAY.json')
        assert process.returncode == 0, process.stderr
        acceptance['detached_GPU_weights_CPU_replay'] = read(temp/'CPU_REPLAY.json')
        acceptance['detached_GPU_weights_CPU_replay']['synthetic_archive_sha256'] = sha(archive)
        for case in ('missing_run', 'missing_checkpoint_manifest', 'wrong_declared_scope', 'perturbed_expected_prediction'):
            broken = temp/case; shutil.copytree(source, broken)
            if case == 'missing_run':
                shutil.rmtree(broken/'runs/tied_synthetic_gpu'); manifest(broken)
            elif case == 'missing_checkpoint_manifest':
                value = read(broken/'MANIFEST.json'); del value['files']['runs/tied_synthetic_gpu/last.pt']
                write(broken/'MANIFEST.json', value)
            elif case == 'wrong_declared_scope':
                value = read(broken/'BUNDLE_SCOPE.json'); value['formal_market_bundles'] = 30
                value['synthetic_example_bundles'] = 0; value['research_experiment_complete'] = True
                write(broken/'BUNDLE_SCOPE.json', value); manifest(broken)
                value = read(broken/'MANIFEST.json'); value['market_bundles'] = 30; write(broken/'MANIFEST.json', value)
            else:
                import numpy as np
                path = broken/'runs/tied_synthetic_gpu/EXAMPLE.npz'
                with np.load(path, allow_pickle=False) as a: arrays = {key: a[key].copy() for key in a.files}
                arrays['last_h5'][0, 0] += 0.001; np.savez(path, **arrays); manifest(broken)
            process = invoke(broken, temp/f'{case}.json')
            assert process.returncode != 0, f'{case} was accepted'
            acceptance['rejection_checks'][case] = {'rejected': True, 'last_error': process.stderr.strip().splitlines()[-1]}
        fake = temp/'fake_market_metadata'; market_fixture(fake); stage.BASE = fake
        stage.verify_complete_market(fake/'analysis/ANALYSIS_COMPLETE.json')
        acceptance['synthetic_gate_checks']['complete_30_run_82_analysis_file_metadata'] = 'accepted'
        receipt_path = fake/'analysis/ANALYSIS_COMPLETE.json'; pristine = receipt_path.read_text()
        changes = {
            'missing_fold_receipt': lambda a: a['files'].pop('analysis/folds/minute_h5_future_log_return/fold_1.json'),
            'missing_loss_family': lambda a: a['files'].pop('analysis/absolute_errors/minute_h60_future_log_return/fold_5.npz'),
            'wrong_statistics_hash': lambda a: a.update(statistics_sha256='0'*64),
            'wrong_results_hash': lambda a: a.update(results_sha256='0'*64),
            'wrong_evaluation_freeze_hash': lambda a: a.update(evaluation_freeze_sha256='0'*64),
            'gate_disagrees_with_statistics': lambda a: a.update(predictive_gate_passed=True),
        }
        for case, mutate in changes.items():
            value = json.loads(pristine); mutate(value); write(receipt_path, value)
            try: stage.verify_complete_market(receipt_path)
            except (AssertionError, KeyError, FileNotFoundError):
                acceptance['synthetic_gate_checks'][case] = 'rejected'
            else: raise AssertionError(f'Market gate incorrectly accepted {case}')
        receipt_path.write_text(pristine)
    acceptance['status'] = 'passed'
    write(BASE/'review/RELEASE_TOOL_REVIEW.json', acceptance)
    print(json.dumps(acceptance), flush=True)


if __name__ == '__main__':
    main()
