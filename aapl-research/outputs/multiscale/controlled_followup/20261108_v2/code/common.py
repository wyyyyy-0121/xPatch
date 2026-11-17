"""Isolated follow-up utilities; admitted pre-July bars only."""
from pathlib import Path
import csv, hashlib, json, os, sys, tempfile
import numpy as np

BASE = Path(__file__).resolve().parents[1]
ROOT = Path('/home/ayaka/wslcode/aapl-research')
SOURCE = ROOT/'outputs/multiscale/admission/alpaca_20261107T052228573242Z'
CORE = ROOT/'outputs/multiscale/completed_core_evidence/20261108_parallel8_continuation'
PREVIOUS = ROOT/'outputs/multiscale/mechanism_followup/20261108_v1'
sys.path.insert(0, str(BASE/'code/reference_source'))
PROFILES = {'annual':('month',12,(12,)), 'quarterly':('month',36,(3,)),
            'monthly':('day',252,(21,)), 'hourly':('hour',96,(1,4,8)),
            'minute':('minute',480,(5,15,30,60))}
MECHANISM = (('minute',60,'future_log_return'),('hourly',8,'future_log_return'))

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for a in iter(lambda:f.read(1024*1024),b''): h.update(a)
    return h.hexdigest()

def atomic(p, obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',dir=p.parent,delete=False,encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False,
            default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())
        f.write('\n');f.flush();os.fsync(f.fileno());tmp=f.name
    os.replace(tmp,p)

def save_npz(p, **arrays):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w+b',dir=p.parent,delete=False) as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno());tmp=f.name
    os.replace(tmp,p)

def key(profile,horizon,target): return f'{profile}_h{horizon}_{target}'

def rows_for(origins, all_origins):
    rows=np.searchsorted(all_origins,origins)
    assert (rows<len(all_origins)).all() and np.array_equal(all_origins[rows],origins)
    return rows

def admitted_bars(frequency):
    from multiscale.matrix_executor import read_bars
    return read_bars(SOURCE,read(SOURCE/'admission.json'),frequency)

def reference_records():
    with (CORE/'results/TRAINING_MODELS.csv').open(newline='') as f:return list(csv.DictReader(f))

def code_hashes():
    return {str(p.relative_to(BASE/'code')):sha(p) for p in sorted((BASE/'code').rglob('*.py'))
            if '__pycache__' not in p.parts and (p.parent.name=='code' and p.name in
              ('common.py','models.py','engine.py','worker.py') or 'reference_source' in p.parts)}

def metric(p,y,anchor,target):
    base=np.zeros_like(y) if target=='future_log_return' else np.broadcast_to(anchor[:,None],y.shape)
    e=p-y;b=base-y;bm=float(np.mean(b[:,-1]**2));m=float(np.mean(e[:,-1]**2))
    return {'origins':len(y),'endpoint_mse':m,'endpoint_mae':float(np.mean(abs(e[:,-1]))),
            'baseline_mse':bm,'relative_improvement_pct':100*(1-m/bm) if bm else None,
            'path_mse':float(np.mean(e**2)),'per_step_mse':np.mean(e**2,axis=0).tolist()}
