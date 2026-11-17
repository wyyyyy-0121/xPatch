"""Package the review version and execute detached inference/statistics checks."""
from study import *
import shutil, zipfile, subprocess, tempfile, platform
import torch


def copy_tree(source, destination):
    shutil.copytree(source, destination, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'lo_profile_*'))


def main():
    assert len(os.sched_getaffinity(0)) <= 2 and os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
    result = read(BASE/'analysis/CANDIDATE_RESULTS.json')
    assert result['task_count'] == 20
    assert read(BASE/'audit/INPUT_ONLY_REPLAY.json')['status'] == 'passed'
    assert read(BASE/'audit/INPUT_ONLY_REPLAY.json')['predictors'] == 375
    assert read(BASE/'audit/VISUAL_REVIEW.json')['status'] == 'passed'
    assert read(BASE/'audit/NUMERIC_AND_DOCUMENT_CHECK.json')['status'] == 'passed'
    assert read(BASE/'audit/DOCUMENT_BUILD.json')['main_page_target_4_to_6']
    assert source_hashes() == read(BASE/'protocol/CODE_FREEZE.json')['hashes']
    release = BASE/'release'; release.mkdir(exist_ok=True)
    root = release/'AAPL_MULTISCALE_MECHANISM_REVIEW'; root.mkdir(exist_ok=True)
    for folder in ('code', 'protocol', 'analysis', 'delivery', 'data'):
        copy_tree(BASE/folder, root/folder)
    manuscript = root/'manuscript'; manuscript.mkdir(exist_ok=True)
    for pattern in ('*_EN.md', '*_EN.pdf', '*_EN.docx', '*.json'):
        for path in (BASE/'manuscript').glob(pattern):
            shutil.copy2(path, manuscript/path.name)
    copy_tree(BASE/'manuscript/figures', manuscript/'figures')
    copy_tree(BASE/'manuscript/preceding_version', manuscript/'preceding_version')
    (root/'audit').mkdir(exist_ok=True)
    for path in (BASE/'audit').glob('*.json'):
        if path.name not in ('RELEASE.json', 'FINAL_AUDIT.json'):
            shutil.copy2(path, root/'audit'/path.name)
    jobs = {j['id']: j for j in read(BASE/'protocol/JOBS.json')['jobs']}
    jobs.update({j['id']: j for j in read(BASE/'protocol/REPLICATION_JOBS.json')['jobs']})
    jobs.update({j['id']: j for j in read(BASE/'protocol/CANDIDATE_JOBS.json')['jobs']})
    jobs.update({j['cfg']['id']: j['cfg'] for j in read(BASE/'protocol/CAP_JOBS.json')['jobs']})
    assert len(jobs) == 375
    for cfg in jobs.values():
        folder = BASE/'runs'/cfg['id']; done = read(folder/'DONE.json')
        assert done['status'] == 'completed' and sha(folder/'best_trained.pt') == done['checkpoint_sha256']
        out = root/'models'/cfg['id']; out.mkdir(parents=True, exist_ok=True)
        for name in ('best_trained.pt', 'initial.pt', 'coupled_best.pt', 'DONE.json', 'INITIALIZATION.json',
                     'PREFIX_COPY.json', 'CAP_COMPARISON.json'):
            if (folder/name).exists():
                shutil.copy2(folder/name, out/name)
    for folder in ('calibration', 'calibration_candidate'):
        for receipt in (BASE/folder).glob('*/DONE.json'):
            out = root/folder/receipt.parent.name; out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(receipt, out/'DONE.json')
    for task in read(PRIOR/'protocol/DATA.json')['task_files']:
        dest = root/'partitions'/task; dest.mkdir(parents=True, exist_ok=True)
        for pattern in ('fold_*.npz', 'scaler_*.json', 'TASK.json'):
            for path in (PRIOR/'cache'/task).glob(pattern):
                shutil.copy2(path, dest/path.name)
    for folder in ('controls', 'controls_cross_scale'):
        copy_tree(BASE/folder, root/folder)
    for name in ('PLAN_ZH.md', 'PROGRESS_ZH.md'):
        if (BASE/name).exists():
            shutil.copy2(BASE/name, root/name)
    previous = PRIOR/'release/AAPL_MULTISCALE_STUDY.zip'
    prior_receipt = read(PRIOR/'audit/RELEASE.json')
    assert sha(previous) == prior_receipt['archive_sha256']
    shutil.copy2(previous, root/'PRECEDING_FULL_STUDY.zip')
    atomic(root/'ENVIRONMENT.json', {'python': sys.version, 'platform': platform.platform(),
           'numpy': np.__version__, 'torch': str(torch.__version__), 'device': 'cpu',
           'torch_build': torch.__config__.show(),
           'gpu_training_used': False, 'threads_per_training_process': 2, 'max_training_processes': 8})
    (root/'requirements.txt').write_text(f'numpy=={np.__version__}\ntorch=={torch.__version__}\n')
    (root/'README.md').write_text('''# AAPL multi-scale mechanism review

The title remains **Multi-Horizon Forecasting of Apple Stock Prices and Returns Across Temporal Scales**. The new manuscript compares all twenty original price/return tasks and adds exact-recipe restorations, factorial initialization/anchor interventions, a copied-prefix training-cap diagnostic, and one fixed repair recipe. The author is Wang Yang, High School Affiliated to Shanghai Jiao Tong University, Shanghai 200439, China; the author fills the email.

Read `manuscript/MAIN_PAPER_EN.pdf` and `SUPPLEMENT_EN.pdf`; editable Word and Markdown are included. `PRECEDING_FULL_STUDY.zip` is the unchanged earlier full package, retaining the original eight-method comparison and the earlier convolution/fusion study. The new package does not overwrite or relabel those results.

**Research status:** matched development mechanisms are supported for particular implementation choices. The designated stable predictive-improvement goal is not achieved. Independent confirmation and the original shared-versus-independent horizon experiment remain unfinished. The protected 2026Q3 series is neither read nor included. No strategy, profitability, external submission, deposit or DOI is asserted.

## Input-only inference

Use the recorded CPU Python environment. From this extracted directory:

```python
from pathlib import Path
import json, sys, numpy as np, torch
torch.set_num_threads(2)
sys.path.insert(0, str(Path('code').resolve()))
from portable_predictor import Predictor
examples = json.loads(Path('delivery/EXAMPLES.json').read_text())['examples']
for example in examples:
    sample = np.load(example['file'])
    model = Predictor(example['archive_checkpoint'])
    forecast = model.predict(sample['completed_closes'], shrink_alpha=example['shrink_alpha'])
    assert np.allclose(forecast, sample['expected'], rtol=1e-6, atol=1e-9)
```

Examples fix final fold, seed2026 and first two origins for all twenty tasks, without score selection. Inputs are positive completed chronological closes on the share basis known at the origin. The caller must enforce timestamps, expected grids and past-only corporate-action adjustment; a naked price array cannot establish those conditions. These are archived fold functions, not refits on current history.

## Saved-loss statistical reproduction

Run `python code/reproduce_statistics.py --root .` under two CPU cores. The script recomputes every mechanism, cap and twenty-task pooled endpoint loss, paired interval and factorial interaction using included per-origin losses and hash-identified pre-cutoff bar calendars. It uses no network or original workspace files. The release receipt records the actual detached run.

## Boundaries

`models/` retains all 345 newly trained configurations and 30 cap checks, with best, initial and relevant coupled checkpoints plus receipts. The cap checks include 13 already-stopped fits that correctly add no updates. Full original downloads, large feature caches, complete neural prediction arrays and latest Adam progress remain in the originating workspace. The included best checkpoints and twenty examples support detached inference; paired losses support detached statistics. Retraining requires the registered source/data identities, named environment, partition files and path remapping of historical absolute provenance paths. Training/orchestration scripts keep those provenance paths and are not claimed directly portable. The input-only predictor and statistical reproduction entrypoints are portable.

The manifest hashes all included files except itself. The prior archive has its own manifest and prior acceptance receipt. Successful checks prove saved-function and arithmetic reproduction, not independent predictive success.
''', encoding='utf-8')
    hashes = {str(path.relative_to(root)): sha(path) for path in sorted(root.rglob('*'))
              if path.is_file() and path.name != 'MANIFEST.json'}
    atomic(root/'MANIFEST.json', {'files': hashes, 'manifest_excludes_itself': True,
                                'Q3_values_read': False, 'research_goal_achieved': False})
    archive = release/'AAPL_MULTISCALE_MECHANISM_REVIEW.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=4) as stream:
        for path in sorted(root.rglob('*')):
            if path.is_file():
                stream.write(path, str(path.relative_to(root.parent)),
                             compress_type=zipfile.ZIP_STORED if path.suffix == '.zip' else zipfile.ZIP_DEFLATED)
    with tempfile.TemporaryDirectory(prefix='aapl-mechanism-detached-') as temporary:
        with zipfile.ZipFile(archive) as stream:
            assert stream.testzip() is None
            stream.extractall(temporary)
        extracted = Path(temporary)/root.name
        assert all(sha(extracted/name) == digest for name, digest in read(extracted/'MANIFEST.json')['files'].items())
        audit_hook = """from pathlib import Path
import sys,os
forbidden=Path('/home/ayaka/wslcode/aapl-research').resolve()
runtime=Path(sys.prefix).resolve()
def boundary(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        path=Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(forbidden) and not path.is_relative_to(runtime):
            raise RuntimeError('Detached access to original research: '+str(path))
    if event in ('socket.connect','socket.connect_ex','socket.getaddrinfo'):
        raise RuntimeError('Detached network access')
sys.addaudithook(boundary)
"""
        examples = audit_hook+"""import json,numpy as np,torch
torch.set_num_threads(2)
sys.path.insert(0,str(Path('code').resolve()))
from portable_predictor import Predictor
records=json.loads(Path('delivery/EXAMPLES.json').read_text())['examples']
assert len(records)==20
for record in records:
    a=np.load(record['file'])
    p=Predictor(record['archive_checkpoint']).predict(a['completed_closes'],shrink_alpha=record['shrink_alpha'])
    assert np.allclose(p,a['expected'],rtol=1e-6,atol=1e-9)
print('DETACHED_TWENTY_TASK_INFERENCE_PASSED')
"""
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
        run = subprocess.run([sys.executable, '-c', examples], cwd=extracted, env=env,
                             capture_output=True, text=True, check=True)
        assert 'DETACHED_TWENTY_TASK_INFERENCE_PASSED' in run.stdout
        statistics = audit_hook+"""import runpy
sys.path.insert(0,str(Path('code').resolve()))
sys.argv=['code/reproduce_statistics.py','--root','.']
runpy.run_path('code/reproduce_statistics.py',run_name='__main__')
"""
        run = subprocess.run([sys.executable, '-c', statistics], cwd=extracted, env=env,
                             capture_output=True, text=True, check=True)
        statistics_receipt = json.loads(run.stdout.strip().splitlines()[-1])
        assert statistics_receipt['status'] == 'passed'
    # Finalize the human-facing ledger after real detached acceptance. Only
    # these reports and the acceptance receipt change; tested functions/losses
    # retain their exact hashes when the final archive is rebuilt.
    ledger = read(BASE/'analysis/GOAL_LEDGER.json')
    for row in ledger['requirements']:
        if row['requirement'] == '论文增补与计算复现':
            row['status'] = '已验证交付'
            row['evidence'] = '新稿逐页查看通过；375 个保存模型精确复放；二十任务独立推理及保存损失统计复算通过'
    ledger['at_utc'] = now(); atomic(BASE/'analysis/GOAL_LEDGER.json', ledger)
    progress = (BASE/'PROGRESS_ZH.md').read_text()
    progress = progress.replace('| 论文增补与计算复现 | 进行中 | 以文档、逐项预测复放和独立统计回执为完成依据 |',
                                '| 论文增补与计算复现 | 已验证交付 | 双栏新稿、375 个模型复放及脱离原目录的二十任务推理／统计复算通过 |')
    progress = progress.replace('本轮补齐新稿、数值检查、逐页查看和保存模型／独立统计复算。成功交付这些材料只完成论文与工程部分，不会自动关闭预测研究 Goal。',
                                '本轮已完成新稿、数值检查、逐页查看和保存模型／独立统计复算。这些材料完成论文与工程交付，预测研究 Goal 继续开启。')
    (BASE/'PROGRESS_ZH.md').write_text(progress, encoding='utf-8')
    shutil.copy2(BASE/'analysis/GOAL_LEDGER.json', root/'analysis/GOAL_LEDGER.json')
    shutil.copy2(BASE/'PROGRESS_ZH.md', root/'PROGRESS_ZH.md')
    atomic(root/'audit/DETACHED_VALIDATION.json', {'status': 'passed', 'twenty_task_inference': True,
           'statistics': statistics_receipt, 'workspace_and_network_blocked': True,
           'research_goal_achieved': False, 'at_utc': now()})
    report_updates = {'analysis/GOAL_LEDGER.json', 'PROGRESS_ZH.md'}
    assert all(sha(root/name) == digest for name, digest in hashes.items() if name not in report_updates)
    hashes = {str(path.relative_to(root)): sha(path) for path in sorted(root.rglob('*'))
              if path.is_file() and path.name != 'MANIFEST.json'}
    atomic(root/'MANIFEST.json', {'files': hashes, 'manifest_excludes_itself': True,
           'Q3_values_read': False, 'research_goal_achieved': False})
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=4) as stream:
        for path in sorted(root.rglob('*')):
            if path.is_file():
                stream.write(path, str(path.relative_to(root.parent)),
                             compress_type=zipfile.ZIP_STORED if path.suffix == '.zip' else zipfile.ZIP_DEFLATED)
    with tempfile.TemporaryDirectory(prefix='aapl-mechanism-final-hashes-') as temporary:
        with zipfile.ZipFile(archive) as stream:
            assert stream.testzip() is None
            stream.extractall(temporary)
        extracted = Path(temporary)/root.name
        assert all(sha(extracted/name) == digest for name, digest in read(extracted/'MANIFEST.json')['files'].items())
    atomic(BASE/'audit/RELEASE.json', {'status': 'passed', 'archive': str(archive),
           'archive_sha256': sha(archive), 'bytes': archive.stat().st_size, 'manifest_files': len(hashes),
           'manifest_verified_after_extraction': True, 'zip_integrity_passed': True,
           'detached_twenty_task_inference': True, 'detached_statistics': statistics_receipt,
           'original_workspace_and_network_access_blocked': True,
           'final_ledger_updates_only_tested_functions_unchanged': True,
           'runtime_exception': str(Path(sys.prefix).resolve()),
           'runtime_exception_scope': 'installed Python environment only',
           'unchanged_prior_archive_sha256': prior_receipt['archive_sha256'],
           'research_goal_achieved': False, 'Q3_values_read': False, 'at_utc': now()})
    print(json.dumps({'status': 'passed', 'archive': str(archive), 'statistics': statistics_receipt}), flush=True)


if __name__ == '__main__':
    main()
