"""Assemble a portable review release only after analysis and manual page review."""
from common import *
from datetime import datetime,timezone
import shutil,zipfile,subprocess,tempfile,csv


def copytree(source, dest):
    shutil.copytree(source,dest,ignore=shutil.ignore_patterns('__pycache__','*.pyc'),dirs_exist_ok=True)


def main():
    r=read(BASE/'analysis/RESULTS.json')
    visual=read(BASE/'audit/VISUAL_REVIEW.json'); assert visual['status']=='passed'
    build=read(BASE/'audit/DOCUMENT_BUILD.json'); assert build['main_page_target_4_to_6']
    replay=read(BASE/'audit/PREDICTOR_REPLAY.json'); assert replay['status']=='passed'
    assert read(BASE/'audit/REFERENCE_REPLAY.json')['status']=='passed'
    assert read(BASE/'audit/ENGINE_RECOVERY.json')['status']=='passed'
    root=BASE/'release/AAPL_MECHANISM_STUDY';root.mkdir(parents=True,exist_ok=True)
    for folder in ('code','protocol','diagnostics','literature','delivery'):
        copytree(BASE/folder,root/folder)
    (root/'analysis').mkdir(exist_ok=True)
    for pattern in ('*.csv','*.json','paired_losses*.npz'):
        for p in (BASE/'analysis').glob(pattern):shutil.copy2(p,root/'analysis'/p.name)
    (root/'manuscript').mkdir(exist_ok=True)
    for pattern in ('*_EN.md','*_EN.docx','*_EN.pdf','SOURCE_AUDIT.json'):
        for p in (BASE/'manuscript').glob(pattern):shutil.copy2(p,root/'manuscript'/p.name)
    copytree(BASE/'manuscript/figures',root/'manuscript/figures')
    (root/'audit').mkdir(exist_ok=True)
    for p in (BASE/'audit').glob('*.json'):
        if p.name not in ('RELEASE.json','FINAL_COMPLETION_AUDIT.json'):
            shutil.copy2(p,root/'audit'/p.name)
    # Keep every market fit's identity, history and artifact hashes, without all large arrays.
    (root/'run_receipts').mkdir(exist_ok=True)
    for job in read(BASE/'protocol/JOBS.json')['jobs']:
        shutil.copy2(BASE/'runs'/job['id']/'DONE.json',root/'run_receipts'/f'{job["id"]}.json')
    copytree(BASE/'controls',root/'controls')
    # Controls' full prediction arrays stay in the workspace. Retain coefficients/receipts.
    for p in (root/'controls').glob('*/prediction.npz'):p.unlink()
    background=ROOT/'outputs/multiscale/double_column_manuscripts/20261108_parallel8_full_core'
    (root/'previous_study').mkdir(exist_ok=True)
    for name in ('MAIN_PAPER_EN.md','MAIN_PAPER_EN.pdf'):
        shutil.copy2(background/name,root/'previous_study'/name)
    copytree(background/'figures',root/'previous_study/figures')
    shutil.copy2(BASE/'CURRENT_PLAN_ZH.md',root/'CURRENT_PLAN_ZH.md')
    # Record actual installed versions for the inference runtime.
    import torch,scipy
    (root/'requirements.txt').write_text(f'numpy=={np.__version__}\nscipy=={scipy.__version__}\ntorch=={torch.__version__}\n')
    primary=r['primary']; result=next(x for x in r['pooled'] if x['kind']==primary)
    (root/'README.md').write_text(f'''# AAPL structural-diagnostics study

This release contains historical development evidence, not an independently confirmed or trading-validated model. The protected 2026Q3 series remains unscored.

Start with `manuscript/MAIN_PAPER_EN.pdf`, `manuscript/SUPPLEMENT_EN.pdf`, and `diagnostics/MECHANISM_REPORT_ZH.md`. The editable Word and Markdown sources are included. The author supplies the email before submission.

Primary: `{primary}`. Pooled mean-seed-loss H60 improvement over zero: {result['h60_improvement_pct']:+.6f}%. Registered performance requirements met: {r['performance_goal_passed']}.

The predeclared two-stream complementarity screen failed. The formal experiment therefore used the predeclared local/constant/state scaling branch. No two-stream performance result is claimed.

## Standalone inference

Install the versions in requirements.txt in a compatible CPU Python environment. This archive was accepted using Python {sys.version.split()[0]} and a CPU-selected PyTorch execution path. From this release root:

```python
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path('delivery').resolve()))
from predictor import Predictor
p = Predictor('delivery/models/formal_{primary}_f5_s2026')
sample = np.load('delivery/example.npz')
forecast = p.predict(sample['closes'])
assert np.allclose(forecast, sample['expected'], rtol=1e-6, atol=1e-10)
```

Supply exactly 480 chronologically ordered, completed regular-session closes on one consistent share basis using only splits effective at the origin. Apply no future split adjustments. The interface cannot infer observation timestamps or missing minutes from a naked price array; the caller must enforce the admitted trading grid and historical availability. Output shape is batch × 60 cumulative log returns in raw units. The package has no target argument and does not retrieve data. The selected demonstration is the final chronological fold at seed 2026, fixed independently of performance.

These are archived fold/seed predictors, not a refit using all current history. The archive includes every small-model formal predictor (all branches actually evaluated). Inactive parameters remain in the archived common module layout. Sigmoid factors are not calibrated confidence or a hard bound on the unconstrained head.

## Reproduction boundaries

`analysis/paired_losses_fold*.npz` supports the paired session-block analysis without raw prices. `run_receipts/` binds every declared fit to its checkpoint and prediction SHA256. `code/` includes the isolated implementation and immutable reference-source snapshot. Absolute source paths in protocol files describe the originating workspace and require remapping for full retraining.

Raw licensed inputs, the feature cache, Adam/latest checkpoints, and the large full prediction arrays remain in the original workspace; they are not all included in this portable review archive. The package is sufficient for its standalone inference demonstration and evidence review, not a self-contained raw-data training archive. The source versions, partitions, preprocessing and scope limits are preserved in protocol files. No Q3 values are included or scored. No archival DOI or external submission has been made.
''')
    manifest={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
    atomic(root/'MANIFEST.json',{'files':manifest,'manifest_excludes_itself':True})
    archive=BASE/'release/AAPL_MECHANISM_STUDY.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file():z.write(p,str(p.relative_to(root.parent)))
    # Actually extract and run the input-only demo with no repository directory on sys.path.
    with tempfile.TemporaryDirectory(prefix='aapl-mechanism-standalone-') as temporary:
        test=Path(temporary)
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None;z.extractall(test)
        extracted=test/root.name
        m=read(extracted/'MANIFEST.json')
        assert all(sha(extracted/name)==digest for name,digest in m['files'].items())
        script='''from pathlib import Path
import sys,numpy as np,torch
torch.set_num_threads(2)
sys.path.insert(0,str(Path('delivery').resolve()))
from predictor import Predictor
p=Predictor('delivery/models/formal_'''+primary+'''_f5_s2026')
a=np.load('delivery/example.npz')
y=p.predict(a['closes'])
assert y.shape==(2,60) and np.isfinite(y).all()
assert np.allclose(y,a['expected'],rtol=1e-6,atol=1e-10)
print('STANDALONE_ACCEPTED')
'''
        completed=subprocess.run([sys.executable,'-c',script],cwd=extracted,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1'),text=True,capture_output=True,check=True)
        assert 'STANDALONE_ACCEPTED' in completed.stdout
    atomic(BASE/'audit/RELEASE.json',{'status':'passed','archive':str(archive),'bytes':archive.stat().st_size,
        'archive_sha256':sha(archive),'files_in_manifest':len(manifest),
        'extracted_manifest_verified':True,'isolated_standalone_inference_actually_executed':True,
        'raw_training_data_all_included':False,'Q3_values_read':False,
        'at_utc':datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'release':str(archive),'standalone':'passed','bytes':archive.stat().st_size}))


if __name__=='__main__':main()
