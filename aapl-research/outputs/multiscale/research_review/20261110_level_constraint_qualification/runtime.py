"""Scope, resource and durable checkpoint helpers. No numerical imports."""
import hashlib
import json
import os
from pathlib import Path
import resource
import tempfile
import time

BASE = Path(__file__).resolve().parent
MEMBERS = ['origins','labels','D','Z','C','zero','prefix_update_mean','same_information_Ridge']

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def setup(cpu):
    require(cpu in (24,26), 'Only analysis CPUs24/26')
    os.sched_setaffinity(0, {cpu})
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS']:
        os.environ[key] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    cap = 768*1024*1024
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    cap = min(cap, hard) if hard != resource.RLIM_INFINITY else cap
    resource.setrlimit(resource.RLIMIT_AS, (cap,cap))

def load_protocol():
    path = BASE/'EXECUTION_PROTOCOL.json'
    p = read(path)
    require(p['schema']=='level_constraint_pair_v1' and p['execution_admitted'] is True, 'No reviewed execution')
    require(p['deadline_epoch']-p['started_epoch']==180, 'Deadline must not reset')
    require(p['GPU'] is False and p['Q3_allowed'] is False, 'Scope differs')
    for file, expected in p['locked_files'].items():
        require(sha(file)==expected, 'Frozen source/design identity changed '+file)
    guard(p)
    return p

def guard(p, reserve=0):
    require(time.time() < p['deadline_epoch']-reserve, 'Absolute shared deadline reached')

def decision(records):
    require([r['state'] for r in records]==['minute_f1','minute_f2','hourly_f1','hourly_f2'], 'Exact four states')
    checks=[]
    for row in records:
        scores=row['metrics']
        for control in ['same_information_Ridge','zero','prefix_update_mean','half_Ridge','half_g']:
            for metric in ['endpoint_MSE','path_MSE']:
                denom=scores[control][metric]
                require(denom > 0, 'Positive control risk required')
                gain=1-scores['g'][metric]/denom
                checks.append({'state':row['state'],'control':control,'metric':metric,
                               'improvement_pct':100*gain,'passed':gain>=.001})
    task=[]
    for prefix in ['minute','hourly']:
        rows=[r for r in records if r['state'].startswith(prefix+'_')]
        gain=sum(1-r['metrics']['g']['endpoint_MSE']/r['metrics']['zero']['endpoint_MSE'] for r in rows)/2
        task.append({'task':prefix,'equal_fold_endpoint_improvement_over_zero_pct':100*gain,'passed':gain>=.01})
    return {'checks':checks,'tasks':task,'passed_of42':sum(x['passed'] for x in checks+task),
            'continue_to_full_development_design_review':all(x['passed'] for x in checks+task),
            'candidate_qualified':False,'independent_confirmation':False,'full_goal_achieved':False}
