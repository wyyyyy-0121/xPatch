from common import *
from datetime import datetime,timezone
import subprocess,time,signal,fcntl,shutil

SLOTS=[(0,2),(4,6),(8,10),(12,14),(16,18),(20,22),(1,3),(5,7)]
STOP=False
ACTIVE={}
def request_stop(*_):
    global STOP;STOP=True

def identity(pid):
    p=Path('/proc')/str(pid)
    try:
        s=(p/'stat').read_text().rsplit(')',1)[1].split()
        return {'pid':pid,'start_ticks':int(s[19]),'state':s[0],
            'argv_sha256':hashlib.sha256((p/'cmdline').read_bytes()).hexdigest()}
    except FileNotFoundError:return None

def reuse(cfg):
    if cfg['kind']!='lstm' or cfg['seed']!=2026:return False
    import torch
    from engine import metrics
    rec=read(BASE/'protocol/DATA.json')['folds'][str(cfg['fold'])]['models']['lstm']
    assert sha(rec['checkpoint'])==rec['checkpoint_sha256'] and sha(rec['prediction'])==rec['prediction_sha256']
    saved=torch.load(rec['checkpoint'],map_location='cpu',weights_only=True);assert saved['identity']==rec['identity']
    folder=BASE/'runs'/cfg['id'];folder.mkdir(exist_ok=True)
    shutil.copy2(rec['checkpoint'],folder/'best.pt');shutil.copy2(rec['prediction'],folder/'prediction.npz')
    with np.load(folder/'prediction.npz') as a:pred=a['prediction'];origins=a['origins']
    cache=cache_arrays();y=cache['actual'][rows_for(origins)]
    atomic(folder/'DONE.json',{'status':'completed','job':cfg,'reused_original':True,
        'identity':rec['identity'],'source':rec,'checkpoint_sha256':sha(folder/'best.pt'),
        'prediction_sha256':sha(folder/'prediction.npz'),'metrics':metrics(pred,y),
        'history':read(rec['summary'])['history'],'best_epoch':saved['epoch'],
        'Q3_values_read':False,'independent_confirmation':False,'new_training':False})
    return True

def stage(name):
    global ACTIVE
    jobs=[j for j in read(BASE/'protocol/JOBS.json')['jobs'] if j['phase']==name]
    pending=[j for j in jobs if not (BASE/'runs'/j['id']/'DONE.json').exists()]
    active={};ACTIVE=active;deadline=read(BASE/'protocol/BUDGET.json')['deadline_epoch'];last=0
    while pending or active:
        stopping=STOP or time.time()>=deadline
        for slot,rec in list(active.items()):
            proc=rec['proc']
            if stopping and not rec.get('signaled'):
                if proc.poll() is None:proc.send_signal(signal.SIGUSR1)
                rec['signaled']=True
            exitcode=proc.poll()
            if exitcode is not None:
                rec['log'].close();folder=BASE/'runs'/rec['job']['id']
                if not (folder/'DONE.json').exists() and not stopping:
                    atomic(BASE/'audit/CONTROLLER_ERROR.json',{'job':rec['job'],'exit_code':exitcode,'state':identity(proc.pid)})
                    raise RuntimeError('Worker exited without complete evidence: '+rec['job']['id'])
                del active[slot]
        free=[s for s in SLOTS if s not in active]
        if pending and free and not stopping:
            available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
            if available>2*1024**3:
                cfg=pending.pop(0)
                if (BASE/'runs'/cfg['id']/'ERROR.json').exists():raise RuntimeError('Unreviewed worker error '+cfg['id'])
                if reuse(cfg):continue
                slot=free[0];env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',PYTHONDONTWRITEBYTECODE='1',
                    OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
                log=(BASE/'logs'/f'{cfg["id"]}.log').open('a')
                proc=subprocess.Popen(['taskset','-c',','.join(map(str,slot)),sys.executable,str(BASE/'code/worker.py'),
                    '--job',cfg['id']],cwd=BASE/'code',env=env,stdout=log,stderr=subprocess.STDOUT)
                active[slot]={'proc':proc,'job':cfg,'log':log,'identity':identity(proc.pid)}
                time.sleep(.5)
        if time.monotonic()-last>=5 or stopping:
            last=time.monotonic();completed=sum((BASE/'runs'/j['id']/'DONE.json').exists() for j in jobs)
            atomic(BASE/'audit/CONTROLLER_STATE.json',{'phase':name,'at_utc':datetime.now(timezone.utc).isoformat(),
                'controller':identity(os.getpid()),'completed':completed,'planned':len(jobs),'pending':len(pending),
                'active':[{'job':v['job'],'process':v['identity'],'cpus':list(k)} for k,v in active.items()],
                'deadline_epoch':deadline,'stopping':stopping})
        if stopping and not active:return False
        time.sleep(1)
    return True

def main():
    for s in (signal.SIGTERM,signal.SIGINT):signal.signal(s,request_stop)
    with (BASE/'audit/CONTROLLER_LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert read(BASE/'audit/COMPUTE_PROJECTION.json')['admitted']
        if not stage('probe'):return
        if not (BASE/'protocol/STAGE_C_DECISION.json').exists():
            from decide import decide;decide()
        if not stage('formal'):return
        jobs=read(BASE/'protocol/JOBS.json')['jobs']
        assert all((BASE/'runs'/j['id']/'DONE.json').exists() for j in jobs)
        atomic(BASE/'audit/TRAINING_COMPLETE.json',{'status':'all_declared_jobs_complete',
            'at_utc':datetime.now(timezone.utc).isoformat(),'jobs':len(jobs),
            'new_fits':sum(not read(BASE/'runs'/j['id']/'DONE.json').get('reused_original',False) for j in jobs),
            'maximum_workers':8,'cpus_per_worker':2,'Q3_values_read':False,'gpu_training':False})
        print('Training complete; analysis and manuscript verification remain.',flush=True)

if __name__=='__main__':main()
