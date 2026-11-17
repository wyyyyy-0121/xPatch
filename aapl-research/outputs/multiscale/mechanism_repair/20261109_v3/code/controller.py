"""A bounded eight-slot queue with signal-driven checkpoint saving."""
from study import *
import fcntl, signal, subprocess, time

SLOTS=[(0,2),(4,6),(8,10),(12,14),(16,18),(20,22),(1,3),(5,7)]
STOP=False
def stop(*_):
    global STOP
    STOP=True

def main():
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):signal.signal(sig,stop)
    assert read(BASE/'audit/COMPUTE_ADMISSION.json')['admitted']
    assert read(BASE/'audit/RECOVERY_CHECK.json')['status']=='passed'
    jobs=read(BASE/'protocol/JOBS.json')['jobs'];deadline=read(BASE/'protocol/BUDGET.json')['deadline_epoch']
    active={};pending=[j for j in jobs if not (BASE/'runs'/j['id']/'DONE.json').exists()];last=0;error=None
    with (BASE/'audit/QUEUE_LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while pending or active:
            stopping=STOP or time.time()>=deadline or error is not None
            for slot,r in list(active.items()):
                if stopping and not r.get('signaled') and r['p'].poll() is None:
                    r['p'].send_signal(signal.SIGUSR1);r['signaled']=True
                if r['p'].poll() is not None:
                    r['log'].close()
                    if not (BASE/'runs'/r['job']['id']/'DONE.json').exists() and not stopping:
                        error={'job':r['job'],'exitcode':r['p'].returncode};atomic(BASE/'audit/QUEUE_ERROR.json',error)
                    del active[slot]
            for slot in [s for s in SLOTS if s not in active]:
                if not pending or stopping or error is not None:break
                memory=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
                if memory<2*1024**3:break
                job=pending.pop(0);out=BASE/'runs'/job['id'];out.mkdir(exist_ok=True)
                assert not (out/'ERROR.json').exists(),'Existing worker error requires review'
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
                log=(BASE/'logs'/(job['id']+'.log')).open('a')
                proc=subprocess.Popen(['taskset','-c',','.join(map(str,slot)),sys.executable,str(BASE/'code/worker.py'),'--job',job['id']],
                    cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                active[slot]={'p':proc,'job':job,'log':log};time.sleep(.1)
            if time.monotonic()-last>5 or stopping:
                last=time.monotonic();done=sum((BASE/'runs'/j['id']/'DONE.json').exists() for j in jobs)
                atomic(BASE/'audit/QUEUE_STATE.json',{'status':'stopping' if stopping else 'running','pid':os.getpid(),
                    'planned':len(jobs),'completed':done,'pending':len(pending),
                    'active':[{'pid':r['p'].pid,'cpus':list(s),'job':r['job']['id']} for s,r in active.items()],
                    'deadline_epoch':deadline,'at_utc':now()})
            if stopping and not active:
                atomic(BASE/'audit/PAUSED.json',{'completed':done,'planned':len(jobs),'error':error,'deadline_epoch':deadline})
                if error:raise RuntimeError(str(error))
                return
            time.sleep(1)
    atomic(BASE/'audit/SCREENING_COMPLETE.json',{'status':'all_screening_jobs_complete','jobs':len(jobs),
            'at_utc':now(),'deadline_epoch':deadline,'Q3_values_read':False})
    print('All 52 screening jobs complete; hypothesis analysis remains.',flush=True)

if __name__=='__main__':main()
