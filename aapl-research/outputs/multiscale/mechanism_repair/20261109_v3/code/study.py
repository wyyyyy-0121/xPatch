"""Explicit identities and paths for the third, isolated mechanism study."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, importlib.util, json, os, sys, tempfile
import numpy as np

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]
PRIOR = ROOT/'outputs/multiscale/controlled_followup/20261108_v2'
SOURCE = ROOT/'outputs/multiscale/admission/alpaca_20261107T052228573242Z'
REFERENCE = PRIOR/'code/reference_source'
sys.path.insert(0, str(REFERENCE))
PROFILES = {'annual': ('month',12,(12,)), 'quarterly': ('month',36,(3,)),
            'monthly': ('day',252,(21,)), 'hourly': ('hour',96,(1,4,8)),
            'minute': ('minute',480,(5,15,30,60))}

def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def atomic(p, value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=p.parent,delete=False) as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False,
                  default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())
        f.write('\n');f.flush();os.fsync(f.fileno());name=f.name
    os.replace(name,p)

def npz(p, **values):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w+b',dir=p.parent,delete=False) as f:
        np.savez_compressed(f,**values);f.flush();os.fsync(f.fileno());name=f.name
    os.replace(name,p)

def task_key(profile,horizon,target): return f'{profile}_h{horizon}_{target}'
def source_hashes():
    own={str(p.relative_to(BASE)):sha(p) for p in sorted((BASE/'code').glob('*.py'))
         if p.name in ('study.py','networks.py','engine.py','worker.py')}
    old=read(PRIOR/'protocol/CODE_FREEZE.json')['hashes']
    assert all(sha(PRIOR/'code'/name)==digest for name,digest in old.items())
    return {'new_training':own,'original_frozen':old}

def old_models():
    spec=importlib.util.spec_from_file_location('v2_frozen_models',PRIOR/'code/models.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def make_job(profile,h,target,model,variant,fold,seed=2026):
    frequency,lookback,_=PROFILES[profile]
    tiny=model=='tiny'
    if tiny:
        anchor='none';initialization='baseline';output='return';loss='aligned';phase='stopping'
    elif target=='future_log_return':
        anchor='constant' if variant.startswith('noanchor') else 'observed'
        initialization='baseline' if variant.endswith('baseline') else 'random'
        output='return';loss='path';phase='anchor_initialization'
    else:
        output='residual' if variant.startswith('residual') else 'price'
        initialization='baseline' if variant.endswith('baseline') else 'random'
        anchor='observed';loss='path';phase='price_parameterization'
    return {'id':f'{task_key(profile,h,target)}_{model}_{variant}_f{fold}_s{seed}',
            'profile':profile,'frequency':frequency,'lookback':lookback,'horizon':h,'target':target,
            'model':model,'variant':variant,'fold':fold,'seed':seed,'anchor':anchor,
            'initialization':initialization,'output':output,'loss':loss,'phase':phase,
            'max_epochs':20,'patience':3,'batch_size':128,'learning_rate':.001}

def screening_jobs():
    jobs=[]
    for profile,h in (('minute',60),('hourly',8)):
        for model in ('lstm','xpatch'):
            for variant in ('anchor_random','noanchor_random','anchor_baseline','noanchor_baseline'):
                for fold in (1,2): jobs.append(make_job(profile,h,'future_log_return',model,variant,fold))
        for fold in (1,2): jobs.append(make_job(profile,h,'future_log_return','tiny','long_aligned',fold))
    for model in ('lstm','xpatch'):
        for variant in ('price_random','price_baseline','residual_random','residual_baseline'):
            for fold in (1,2):jobs.append(make_job('hourly',1,'future_close',model,variant,fold))
    assert len(jobs)==52 and len({j['id'] for j in jobs})==52
    return jobs

def metrics(p,y,anchor,target):
    baseline=np.zeros_like(y) if target=='future_log_return' else np.broadcast_to(anchor[:,None],y.shape)
    e=p-y;b=baseline-y;bm=float(np.mean(b[:,-1]**2));m=float(np.mean(e[:,-1]**2))
    return {'origins':len(y),'endpoint_mse':m,'path_mse':float(np.mean(e**2)),
            'endpoint_mae':float(np.mean(abs(e[:,-1]))),'baseline_mse':bm,
            'improvement_pct':100*(1-m/bm) if bm else None}
