"""Freeze the complete replication and keep its original absolute deadline."""
from common import *
import argparse, os, fcntl, signal, subprocess, time

SLOTS=[(0,2),(4,6),(8,10),(12,14),(16,18),(20,22),(1,3),(5,7)]
STOP=False


def stop(*_):
    global STOP
    STOP=True


def prepare():
    path=BASE/'protocol/JOBS.json'
    if path.exists():
        saved=read(path)
        assert saved['wrapper_hashes']==implementation_hashes()
        assert saved['training_hashes']==source_hashes()
        assert saved['budget_sha256']==sha(V3/'protocol/BUDGET.json')
        return saved
    specs=jobs(); locations={}; reuse=[]; costs=[]
    budget=read(V3/'protocol/BUDGET.json')
    original_jobs=read(V3/'protocol/CANDIDATE_JOBS.json')['jobs']
    rates={}
    for profile in ('annual','quarterly','monthly'):
        receipts=[read(V3/'runs'/j['id']/'DONE.json') for j in original_jobs if j['profile']==profile]
        rates[profile]=max(r['training_wall_seconds']/r['epochs'] for r in receipts)
    for cfg in specs:
        prior=V3/'runs'/cfg['id']
        if (prior/'DONE.json').exists():
            d=read(prior/'DONE.json')
            assert d['job']==cfg and d['status']=='completed'
            assert sha(prior/'best_trained.pt')==d['checkpoint_sha256']
            assert sha(prior/'predictions.npz')==d['prediction_sha256']
            locations[cfg['id']]=str(prior)
            reuse.append({'job':cfg['id'],'done_sha256':sha(prior/'DONE.json'),
                          'checkpoint_sha256':d['checkpoint_sha256']})
        else:
            locations[cfg['id']]=str(BASE/'runs'/cfg['id'])
            costs.append(20*rates[cfg['profile']]+15.)
    slot_times=[0.]*8
    for cost in sorted(costs,reverse=True):
        slot_times[slot_times.index(min(slot_times))]+=cost
    projected=2*max(slot_times)+300
    remaining=budget['deadline_epoch']-time.time()
    manifest={'status':'frozen_post_observation_replication','jobs':specs,'locations':locations,
        'reused':reuse,'new_cells':len(costs),'planned_cells':180,'wrapper_hashes':implementation_hashes(),
        'training_hashes':source_hashes(),'budget_sha256':sha(V3/'protocol/BUDGET.json'),
        'deadline_epoch':budget['deadline_epoch'],'new_results_seen':False,
        'original_twenty_task_candidate_results_seen':True,'primary_endpoints_unchanged':True,
        'rules':'xPatch original/random and baseline-function heads, unchanged inputs/direct targets/path loss/stopping; six long-scale tasks, five folds, all three seeds; no outer score selection.',
        'statistics':'per-origin mean seed losses; original strong controls;12/24month,3/6month,21/42day blocks; greedy and all horizon-offset nonoverlap sensitivities; no independent-confirmation claim.',
        'projected_wall_seconds_with_reserve':projected,'remaining_seconds_at_admission':remaining,
        'admitted':projected<remaining,'Q3_values_read':False,'at_utc':now()}
    assert len(reuse)==30 and len(costs)==150
    atomic(path,manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();manifest=prepare()
    print({'admitted':manifest['admitted'],'new_cells':manifest['new_cells'],
           'projected_seconds':manifest['projected_wall_seconds_with_reserve']},flush=True)
    if args.prepare_only or not manifest['admitted']:return
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGUSR1):signal.signal(sig,stop)
    (BASE/'logs').mkdir(exist_ok=True)
    pending=[j for j in manifest['jobs'] if not (folder(j,manifest)/'DONE.json').exists()]
    # Largest measured profile first; no ordering depends on a new score.
    pending.sort(key=lambda j:({'monthly':0,'annual':1,'quarterly':2}[j['profile']],j['id']))
    active={}; error=None; last=0.
    with (BASE.parent/'COMPUTE_QUEUE_LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while pending or active:
            stopping=STOP or time.time()>=manifest['deadline_epoch'] or error is not None
            for slot,item in list(active.items()):
                p=item['process']
                if stopping and not item.get('signaled') and p.poll() is None:
                    p.send_signal(signal.SIGUSR1);item['signaled']=True
                if p.poll() is not None:
                    item['log'].close()
                    if not (folder(item['job'],manifest)/'DONE.json').exists() and not stopping:
                        error={'job':item['job']['id'],'exitcode':p.returncode}
                        atomic(BASE/'audit/ERROR.json',error)
                    del active[slot]
            for slot in [s for s in SLOTS if s not in active]:
                if stopping or error or not pending:break
                cfg=pending.pop(0);log=(BASE/'logs'/(cfg['id']+'.log')).open('a')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                         OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
                p=subprocess.Popen(['taskset','-c',','.join(map(str,slot)),sys.executable,
                                    str(BASE/'code/worker.py'),'--job',cfg['id']],
                                   cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                active[slot]={'process':p,'job':cfg,'log':log}
            completed=sum((folder(j,manifest)/'DONE.json').exists() for j in manifest['jobs'])
            if time.monotonic()-last>5 or stopping:
                last=time.monotonic()
                atomic(BASE/'audit/STATE.json',{'status':'stopping' if stopping else 'running',
                    'completed':completed,'planned':180,'pending':len(pending),'pid':os.getpid(),
                    'active':[{'pid':i['process'].pid,'job':i['job']['id'],'cpus':list(s)} for s,i in active.items()],
                    'deadline_epoch':manifest['deadline_epoch'],'at_utc':now()})
            if stopping and not active:
                atomic(BASE/'audit/PAUSED.json',{'status':'paused','completed':completed,
                       'planned':180,'error':error,'deadline_epoch':manifest['deadline_epoch'],'at_utc':now()})
                if error:raise RuntimeError(str(error))
                return
            time.sleep(0.5)
    atomic(BASE/'audit/COMPLETE.json',{'status':'all_cells_complete','cells':180,'new_cells':150,
           'reused':30,'deadline_epoch':manifest['deadline_epoch'],'Q3_values_read':False,'at_utc':now()})
    print('All180 paired long-scale cells complete.',flush=True)


if __name__=='__main__':main()
