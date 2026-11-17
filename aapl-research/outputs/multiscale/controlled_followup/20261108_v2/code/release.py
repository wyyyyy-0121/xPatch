"""Create a portable local review package after actual document and predictor checks."""
from common import *
import shutil,zipfile,subprocess,tempfile,torch,scipy,pandas,platform
from datetime import datetime,timezone

def tree(src,dst):
    shutil.copytree(src,dst,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','lo_profile_*'))

def main():
    result=read(BASE/'analysis/RESULTS.json');build=read(BASE/'audit/DOCUMENT_BUILD.json')
    assert build['main_page_target_4_to_6']
    visual=read(BASE/'audit/VISUAL_REVIEW.json');assert visual['status']=='passed'
    assert read(BASE/'audit/PREDICTOR_REPLAY.json')['status']=='passed'
    assert read(BASE/'audit/RECOVERY_CHECK.json')['status']=='passed'
    assert read(BASE/'audit/NUMERIC_AND_DOCUMENT_CHECK.json')['status']=='passed'
    assert code_hashes()==read(BASE/'protocol/CODE_FREEZE.json')['hashes']
    jobs=read(BASE/'protocol/SELECTED_JOBS.json')['jobs'];root=BASE/'release/AAPL_MULTISCALE_STUDY'
    root.mkdir(exist_ok=True)
    for folder in ('code','protocol','diagnostics','analysis','literature','delivery'):
        tree(BASE/folder,root/folder)
    (root/'manuscript').mkdir(exist_ok=True)
    for pattern in ('*_EN.md','*_EN.docx','*_EN.pdf','AUTHOR_INFORMATION.json','SOURCE_AUDIT.json'):
        for p in (BASE/'manuscript').glob(pattern):shutil.copy2(p,root/'manuscript'/p.name)
    tree(BASE/'manuscript/figures',root/'manuscript/figures')
    (root/'audit').mkdir(exist_ok=True)
    for p in (BASE/'audit').glob('*.json'):
        if p.name not in ('RELEASE.json','FINAL_COMPLETION_AUDIT.json'):
            shutil.copy2(p,root/'audit'/p.name)
    for job in jobs:
        out=root/'models'/job['id'];out.mkdir(parents=True,exist_ok=True)
        for name in ('best.pt','DONE.json'):shutil.copy2(BASE/'runs'/job['id']/name,out/name)
    for task in read(BASE/'protocol/DATA.json')['task_files']:
        dest=root/'partitions'/task;dest.mkdir(parents=True,exist_ok=True)
        for pattern in ('fold_*.npz','scaler_*.json','TASK.json'):
            for p in (BASE/'cache'/task).glob(pattern):shutil.copy2(p,dest/p.name)
    tree(BASE/'fusion',root/'fusion');tree(BASE/'controls',root/'controls')
    original=root/'original_core';original.mkdir(exist_ok=True)
    tree(CORE/'results',original/'results')
    oldmain=ROOT/'outputs/multiscale/double_column_manuscripts/20261108_parallel8_full_core'
    for name in ('MAIN_PAPER_EN.md','MAIN_PAPER_EN.pdf','AUTHOR_INFORMATION.json'):
        shutil.copy2(oldmain/name,original/name)
    prior=root/'prior_minute_substudy';prior.mkdir(exist_ok=True)
    for folder in ('analysis','diagnostics','protocol'):
        tree(PREVIOUS/folder,prior/folder)
    (prior/'manuscript').mkdir(exist_ok=True)
    for name in ('MAIN_PAPER_EN.md','MAIN_PAPER_EN.pdf'):
        shutil.copy2(PREVIOUS/'manuscript'/name,prior/'manuscript'/name)
    source=root/'data';source.mkdir(exist_ok=True)
    for name in ('admission.json','bars_month.npz','bars_day.npz','bars_hour.npz','bars_minute.npz'):
        shutil.copy2(SOURCE/name,source/name)
    # Keep all original experts used in replay, not merely the small gate.
    experts=root/'original_experts';experts.mkdir(exist_ok=True)
    for task in read(BASE/'protocol/DATA.json')['task_files']:
        meta=read(BASE/'cache'/task/'TASK.json')
        for fn,f in meta['folds'].items():
            for model,rec in f['models'].items():
                assert sha(rec['checkpoint'])==rec['checkpoint_sha256']
                assert sha(rec['summary'])==rec['summary_sha256']
                out=experts/task/f'fold_{fn}'/model;out.mkdir(parents=True,exist_ok=True)
                shutil.copy2(rec['checkpoint'],out/'best.pt')
                shutil.copy2(rec['summary'],out/'training_summary.json')
    shutil.copy2(BASE/'EXECUTION_PLAN_ZH.md',root/'EXECUTION_PLAN_ZH.md')
    import exchange_calendars
    (root/'requirements.txt').write_text(f'numpy=={np.__version__}\nscipy=={scipy.__version__}\npandas=={pandas.__version__}\ntorch=={torch.__version__}\nexchange-calendars=={exchange_calendars.__version__}\n')
    atomic(root/'ENVIRONMENT.json',{'python':sys.version,'platform':platform.platform(),
        'numpy':np.__version__,'scipy':scipy.__version__,'pandas':pandas.__version__,
        'torch':torch.__version__,'exchange_calendars':exchange_calendars.__version__,
        'torch_build_configuration':torch.__config__.show(),'training_device':'cpu',
        'gpu_training_used':False,'cpu_threads_per_training_worker':2,
        'requirements_note':'Recorded installed distributions. The installed Torch build has CUDA support, but this study uses its CPU path only. Detached checks reuse this Python environment.'})
    (root/'README.md').write_text(f'''# AAPL multi-scale historical development study

Title: Multi-Horizon Forecasting of Apple Stock Prices and Returns Across Temporal Scales.
Author: Wang Yang, High School Affiliated to Shanghai Jiao Tong University, Shanghai 200439, China. The author fills the email before submission.

Read `manuscript/MAIN_PAPER_EN.pdf` and `SUPPLEMENT_EN.pdf`. Editable Word/Markdown, full original and new tables, diagnostics, registered protocol, all new model best checkpoints, the original expert checkpoints, fusion/linear controls and pre-July derived bar files are included. Neither a public deposit nor an external submission has been made.

New neural fits: {result['completed_new_fits']}. Primary candidate across twenty tasks; detailed factorial at two return tasks. Both-task registered predictive objective met: {result['both_mechanism_tasks_pass']}. These are exposed historical development results, not independent confirmation or trading-profit evidence. The protected 2026Q3 series is neither opened nor included.

## Portable input-only inference

With the matching CPU Python environment, from the extracted release root:

```python
from pathlib import Path
import sys, numpy as np, torch
torch.set_num_threads(2)
sys.path.insert(0, str(Path('code').resolve()))
from predictor import Predictor
sample = np.load('delivery/minute_example.npz')
model = Predictor('delivery/models/minute.pt')
forecast = model.predict(sample['closes'])
assert np.allclose(forecast, sample['expected'], rtol=1e-6, atol=1e-10)
```

The interface accepts positive chronological completed closes on the share basis known at the origin. The caller must enforce the expected session grid and past-only split adjustment. It cannot infer timestamps or missing bars from a naked price array. Output is the separately fitted nominal-price or cumulative-log-return path in raw units, according to checkpoint cfg. These are archived fold predictors, not current-history refits.

Five fixed final-fold/seed-2026 examples span all scales. Saved arrays for every new predictor were replayed with the original batch size and exact equality. Original expert restoration relies on its prior verified audit plus retained identities. Fusion gates require their four fixed experts; the 25 gate parameters do not describe total inference cost.

## Recompute released statistics

From the extracted release root, run `python code/reproduce_statistics.py`. It reconstructs all new pooled mean losses, paired intervals, factorial interactions and fusion replay statistics from included per-origin losses and local pre-cutoff calendars. It verifies the bar hashes and never accesses the originating workspace or a network source. `partitions/` retains the exact fit/stop/validation indices and preprocessing parameters for each task and fold.

## Reproduction boundaries

`analysis/paired_losses/` and `fusion_paired_losses/` contain per-origin losses for the documented paired statistics. `models/` retains every new best checkpoint and its receipt; `original_experts/` retains the original neural experts. `data/` contains the hash-identified pre-July derived bar arrays and admission metadata. The full provider raw downloads, large feature caches, full neural prediction arrays and latest Adam progress remain in the original workspace. The smaller control and fusion prediction arrays are included. This review package is sufficient for detached inference and saved-loss statistics; full retraining needs path remapping in origin-workspace protocols and the named environment. Original absolute paths record provenance; the portable predictor itself requires no original workspace path.

The earlier minute study remains a substudy, not a replacement title. The original shared-versus-independent multi-horizon market experiment remains unrun. The manifest identifies all included files and the external-source scope; no result is inferred from that unfinished comparison.
''',encoding='utf-8')
    manifest={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='MANIFEST.json'}
    atomic(root/'MANIFEST.json',{'files':manifest,'manifest_excludes_itself':True})
    archive=BASE/'release/AAPL_MULTISCALE_STUDY.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file():z.write(p,str(p.relative_to(root.parent)))
    with tempfile.TemporaryDirectory(prefix='aapl-multiscale-detached-') as tmp:
        extracted=Path(tmp)
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None;z.extractall(extracted)
        dest=extracted/root.name;m=read(dest/'MANIFEST.json');assert all(sha(dest/n)==v for n,v in m['files'].items())
        audit_hook="""from pathlib import Path
import sys,os
forbidden=Path('/home/ayaka/wslcode/aapl-research').resolve()
runtime=Path(sys.prefix).resolve()
def boundary(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        path=Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(forbidden) and not path.is_relative_to(runtime):
            raise RuntimeError('Detached check accessed original research files: '+str(path))
    if event in ('socket.connect','socket.connect_ex','socket.getaddrinfo'):
        raise RuntimeError('Detached check attempted network access')
sys.addaudithook(boundary)
"""
        code=audit_hook+"""import numpy as np,torch
torch.set_num_threads(2)
sys.path.insert(0,str(Path('code').resolve()))
from predictor import Predictor
for profile in ('annual','quarterly','monthly','hourly','minute'):
    a=np.load('delivery/'+profile+'_example.npz')
    p=Predictor('delivery/models/'+profile+'.pt').predict(a['closes'])
    assert np.allclose(p,a['expected'],rtol=1e-6,atol=1e-10)
print('DETACHED_FIVE_SCALE_INFERENCE_PASSED')
"""
        run=subprocess.run([sys.executable,'-c',code],cwd=dest,env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1'),text=True,capture_output=True,check=True)
        assert 'DETACHED_FIVE_SCALE_INFERENCE_PASSED' in run.stdout
        statcode=audit_hook+"""import runpy
sys.path.insert(0,str(Path('code').resolve()))
runpy.run_path('code/reproduce_statistics.py',run_name='__main__')
"""
        stats=subprocess.run([sys.executable,'-c',statcode],cwd=dest,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2'),
            text=True,capture_output=True,check=True)
        stats_receipt=json.loads(stats.stdout.strip().splitlines()[-1]);assert stats_receipt['status']=='passed'
    atomic(BASE/'audit/RELEASE.json',{'status':'passed','archive':str(archive),'bytes':archive.stat().st_size,
        'archive_sha256':sha(archive),'manifest_files':len(manifest),'extracted_hash_check':True,
        'detached_five_scale_inference_executed':True,'new_best_predictors_included':len(jobs),
        'detached_statistics_recomputed':stats_receipt,'detached_workspace_and_network_access_blocked':True,
        'detached_boundary_runtime_exception':str(Path(sys.prefix).resolve()),
        'detached_boundary_runtime_exception_scope':'installed Python environment only; original research code/data/results remain blocked',
        'original_experts_included':400,'raw_provider_downloads_all_included':False,
        'derived_pre_cutoff_bars_included':True,'Q3_values_read':False,'at_utc':datetime.now(timezone.utc).isoformat()})

if __name__=='__main__':main()
