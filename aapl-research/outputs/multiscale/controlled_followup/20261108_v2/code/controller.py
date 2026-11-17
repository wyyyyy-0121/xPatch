from common import *
from datetime import datetime,timezone
import subprocess,time,signal,fcntl

SLOTS=[(0,2),(4,6),(8,10),(12,14),(16,18),(20,22),(1,3),(5,7)]
STOP=False
def stop(*_):
    global STOP
    STOP=True

def main():
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):signal.signal(sig,stop)
    assert read(BASE/'audit/COMPUTE_ADMISSION.json')['admitted'] and read(BASE/'audit/RECOVERY_CHECK.json')['status']=='passed'
    jobs=read(BASE/'protocol/SELECTED_JOBS.json')['jobs'];active={}
    deadline=read(BASE/'protocol/BUDGET.json')['deadline_epoch'];last=0
    with (BASE/'audit/CONTROLLER_LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pending=[j for j in jobs if not (BASE/'runs'/j['id']/'DONE.json').exists()]
        while pending or active:
            stopping=STOP or time.time()>=deadline
            for slot,rec in list(active.items()):
                proc=rec['proc']
                if stopping and not rec.get('signaled'):
                    if proc.poll() is None:proc.send_signal(signal.SIGUSR1)
                    rec['signaled']=True
                if proc.poll() is not None:
                    rec['log'].close()
                    if not (BASE/'runs'/rec['job']['id']/'DONE.json').exists() and not stopping:
                        atomic(BASE/'audit/CONTROLLER_ERROR.json',{'job':rec['job'],'exitcode':proc.returncode})
                        for r in active.values():
                            if r['proc'].poll() is None:r['proc'].send_signal(signal.SIGUSR1)
                        raise RuntimeError('Unreviewed worker failure '+rec['job']['id'])
                    del active[slot]
            free=[s for s in SLOTS if s not in active]
            while pending and free and not stopping:
                available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
                if available<2*1024**3:break
                cfg=pending.pop(0);folder=BASE/'runs'/cfg['id'];folder.mkdir(exist_ok=True)
                if (folder/'ERROR.json').exists():raise RuntimeError('Unreviewed existing error '+cfg['id'])
                slot=free.pop(0);env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',OPENBLAS_NUM_THREADS='1',
                    OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
                log=(BASE/'logs'/f'{cfg["id"]}.log').open('a')
                p=subprocess.Popen(['taskset','-c',','.join(map(str,slot)),sys.executable,str(BASE/'code/worker.py'),
                    '--job',cfg['id']],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                active[slot]={'proc':p,'log':log,'job':cfg};time.sleep(.25)
            if time.monotonic()-last>5 or stopping:
                last=time.monotonic();complete=sum((BASE/'runs'/j['id']/'DONE.json').exists() for j in jobs)
                atomic(BASE/'audit/CONTROLLER_STATE.json',{'status':'stopping' if stopping else 'running',
                    'controller_pid':os.getpid(),'completed':complete,'planned':len(jobs),'pending':len(pending),
                    'active':[{'pid':r['proc'].pid,'cpus':list(s),'job':r['job']} for s,r in active.items()],
                    'deadline_epoch':deadline,'at_utc':datetime.now(timezone.utc).isoformat()})
            if stopping and not active:
                atomic(BASE/'audit/TRAINING_PAUSED.json',{'completed':complete,'planned':len(jobs),'deadline_epoch':deadline})
                return
            time.sleep(1)
        atomic(BASE/'audit/TRAINING_COMPLETE.json',{'status':'all_selected_jobs_complete','jobs':len(jobs),
            'maximum_workers':8,'cpus_per_worker':2,'gpu':False,'deadline_epoch':deadline,
            'at_utc':datetime.now(timezone.utc).isoformat(),'Q3_values_read':False})
        print('All selected training cells complete; analysis and paper delivery remain.',flush=True)

if __name__=='__main__':main()
