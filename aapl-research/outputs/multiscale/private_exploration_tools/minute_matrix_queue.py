"""Bounded local scheduler for the remaining minute development folds.

Existing completed or in-flight jobs are recorded and excluded. The scheduler
uses disjoint CPU slots, keeps checkpoints, and never retries a failed job.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


SLOTS=((0,2),(4,6),(8,10),(12,14),(16,18),(20,22))
EXTERNALS=(('h60_return', 'private_minute_parallel_folds/20261107T142105271032Z'),
           ('h60_return', 'private_minute_parallel_folds/20261107T143936100555Z'),
           ('breadth', 'private_minute_breadth_development/20261107T144302924248Z'),
           ('breadth', 'private_minute_breadth_development/20261107T144956615202Z'),
           ('breadth', 'private_minute_breadth_development/20261107T150838302360Z'),
           ('mixed', 'private_minute_mixed_development/20261107T151427101607Z'))


def save_new(path,value):
    with path.open('x',encoding='utf8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')


def available_memory_bytes():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    raise ValueError('Cannot assess available memory')


def alive(pid):
    if not isinstance(pid,int) or pid<=1:return False
    try:os.kill(pid,0)
    except ProcessLookupError:return False
    except PermissionError:return True
    return True


def terminate_child(child):
    if child.poll() is None:
        try:os.killpg(child.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait()
    return child.returncode


def external_entries(admission_hash):
    from multiscale.bars import file_digest
    from multiscale.spec import OUT
    excluded={(60,'future_log_return',1):{'source':'verified full-origin controls and neural first fold',
                                          'status_at_freeze':'completed'}}
    sources=[]
    for kind,relative in EXTERNALS:
        root=OUT/relative;pp=root/'PRIVATE_PROTOCOL.json';sp=root/'PRIVATE_QUEUE_STATE.json'
        plan=json.loads(pp.read_text());state=json.loads(sp.read_text())
        if plan['admission_sha256']!=admission_hash or plan.get('Q3_values_read') is not False:
            raise ValueError('External queue uses different candidate data')
        sources.append({'kind':kind,'protocol':str(pp),'state':str(sp),'protocol_sha256':file_digest(pp),
                        'state_sha256_at_freeze':file_digest(sp)})
        for job in plan['jobs']:
            if kind=='h60_return':
                worker=str(job['fold']);identity=(60,'future_log_return',job['fold'])
            else:
                worker=job['key'];identity=(job['horizon'],job['target'],job['fold'])
            entry=state['workers'][worker]
            if identity in excluded:raise ValueError('Duplicate external minute fold')
            excluded[identity]={'source':str(root),'worker':worker,'status_at_freeze':entry['status']}
    return excluded,sources


def prepare_plan():
    import numpy as np
    from multiscale.bars import eligible_origins,file_digest
    from multiscale.development import validate_split
    from multiscale.matrix_executor import read_bars
    from multiscale.spec import OUT,PROFILES,ROOT
    from multiscale.splits import rolling_folds
    from multiscale.training import digest_indices
    source=OUT/'admission/alpaca_20261107T052228573242Z';ap=source/'admission.json';a=json.loads(ap.read_text())
    terms=OUT/'source_docs/alpaca_terms.txt'
    quote='you agree to use the Services and Content solely for your own personal and non-commercial purposes'
    if (a.get('Q3_values_read') is not False or a.get('experiment_scored') is not False or
            quote not in ' '.join(terms.read_text().split())):
        raise ValueError('Candidate or recorded personal-use term is invalid')
    prior=OUT/'private_minute_full_neural/20261107T140317864464Z'
    prior_check=json.loads((prior/'PRIVATE_INDEPENDENT_CHECK.json').read_text())
    prior_plan=json.loads((prior/'PRIVATE_PROTOCOL.json').read_text())
    if (prior_check.get('status')!='private_full_minute_neural_independently_recomputed' or
            prior_check.get('result_sha256')!=file_digest(prior/'PRIVATE_RESULT.json') or
            prior_plan.get('admission_sha256')!=file_digest(ap)):
        raise ValueError('The first minute fold is not verified')
    if any(not set(slot).issubset(os.sched_getaffinity(0)) for slot in SLOTS):
        raise ValueError('A CPU slot is unavailable')
    excluded,external_sources=external_entries(file_digest(ap))
    bars=read_bars(source,a,'minute');profile=next(p for p in PROFILES if p.name=='minute')
    cache={}
    for horizon in (5,15,30,60):
        op=source/f'origins_minute_h{horizon}.npy'
        if file_digest(op)!=a['files'][op.name]:raise ValueError('Candidate origins changed')
        origins=np.load(op,allow_pickle=False)
        if not np.array_equal(origins,eligible_origins(bars,max(profile.lookbacks),horizon,same_session=True)):
            raise ValueError('Candidate minute origin set differs')
        folds=rolling_folds(bars,origins,horizon)
        if len(folds)!=5:raise ValueError('Expected five chronological folds')
        cache[horizon]=(op,folds)
    jobs=[]
    # Broaden across horizons and targets within each fold before later folds.
    for number in range(1,6):
        for horizon in (5,15,30,60):
            op,folds=cache[horizon]
            for target in ('future_log_return','future_close'):
                if (horizon,target,number) in excluded:continue
                train,fit,stop,val=validate_split(bars,folds[number-1],horizon)
                key=f'h{horizon}_{target}_f{number}'
                jobs.append({'key':key,'horizon':horizon,'target':target,'fold':number,
                             'origins_sha256':file_digest(op),'train_origins':len(train),
                             'fit_origins':len(fit),'stop_origins':len(stop),'validation_origins':len(val),
                             'split_sha256':{n:digest_indices(x) for n,x in [('train',train),('fit',fit),
                                                                              ('stop',stop),('validation',val)]}})
    if len(jobs)+len(excluded)!=40 or len({j['key'] for j in jobs})!=len(jobs):
        raise ValueError('Minute task inventory is inconsistent')
    sources=('multiscale/development.py','multiscale/training.py','multiscale/models.py',
             'multiscale/features.py','multiscale/bars.py','multiscale/baselines.py','multiscale/spec.py',
             'multiscale/splits.py','multiscale/evaluation.py','multiscale/ridge.py',
             'multiscale/matrix_executor.py','multiscale/matrix_gate.py','multiscale/verify_development.py',
             'multiscale/mechanism.py','aapl/models.py','aapl/vendor/xpatch_network.py')
    plan={'status':'private_minute_matrix_development_only_not_publication_admitted',
          'created_at_utc':datetime.now(timezone.utc).isoformat(),
          'user_request':'直接做实验；可以并行的任务就并行',
          'scope':'personal non-commercial local development','admission_sha256':file_digest(ap),
          'bar_sha256':a['files']['bars_minute.npz'],'alpaca_terms_sha256':file_digest(terms),
          'prior_first_fold_verification_sha256':file_digest(prior/'PRIVATE_INDEPENDENT_CHECK.json'),
          'models':['last_close','drift','seasonal','ridge','mlp','gru','lstm','xpatch'],
          'lookback':480,'seasonal_period':390,'max_epochs':20,'patience':3,'batch_size':128,
          'learning_rate':0.001,'ridge_alphas':[0.1,1.0,10.0],'seed':2026,
          'allowed_cpu_slots':[list(slot) for slot in SLOTS],'threads_per_worker':2,
          'maximum_workers':len(SLOTS),'maximum_wall_seconds':7200,
          'start_minimum_available_memory_bytes':4*1024**3,'stop_available_memory_bytes':2*1024**3,
          'external_sources':external_sources,
          'excluded_external_jobs':[{'horizon':h,'target':t,'fold':f,**v} for (h,t,f),v in sorted(excluded.items())],
          'jobs':jobs,'device':'cpu','data_cutoff':'2026-06-30','Q3_values_read':False,'independent_test':False,
          'source_sha256':{p:file_digest(ROOT/p) for p in sources},'script_sha256':file_digest(Path(__file__)),
          'publication_rights':'not confirmed; keep results local',
          'limitations':['All folds are previously exposed historical development, not independent confirmation.',
                         'Existing completed or running jobs are excluded; failed external jobs are not automatically retried.',
                         'The two-hour wall limit may interrupt jobs; checkpoints are preserved without automatic retry.',
                         'Account-specific rights, publication rights, convergence and historical revisions remain unconfirmed.']}
    return plan


def busy_external_slots(plan):
    from multiscale.bars import file_digest
    busy=set();statuses=[]
    for source in plan['external_sources']:
        pp=Path(source['protocol']);sp=Path(source['state'])
        if file_digest(pp)!=source['protocol_sha256']:raise ValueError('External protocol changed')
        ext_plan=json.loads(pp.read_text());state=json.loads(sp.read_text())
        for job in ext_plan['jobs']:
            worker=str(job['fold']) if source['kind']=='h60_return' else job['key']
            entry=state['workers'][worker];slot=tuple(job['cores'])
            if slot not in SLOTS:raise ValueError('External CPU slot is not recognized')
            if entry['status']=='running' and alive(entry.get('pid')):busy.add(slot)
            statuses.append({'worker':worker,'status':entry['status'],'source':str(sp)})
    return busy,statuses


def worker(plan_path,key,cores):
    plan_path=Path(plan_path).resolve();plan=json.loads(plan_path.read_text())
    job=next(j for j in plan['jobs'] if j['key']==key)
    slot=tuple(cores)
    if list(slot) not in plan['allowed_cpu_slots']:raise ValueError('CPU slot is not in the plan')
    os.sched_setaffinity(0,set(slot))
    import numpy as np
    import torch
    from multiscale.bars import file_digest
    from multiscale.development import FoldConfig,run_development_fold,validate_split
    from multiscale.matrix_executor import read_bars
    from multiscale.spec import OUT,ROOT
    from multiscale.splits import rolling_folds
    from multiscale.training import digest_indices
    from multiscale.verify_development import check_fold
    if (file_digest(Path(__file__))!=plan['script_sha256'] or
            any(file_digest(ROOT/p)!=h for p,h in plan['source_sha256'].items())):
        raise ValueError('Worker source identity changed')
    src=OUT/'admission/alpaca_20261107T052228573242Z';ap=src/'admission.json';a=json.loads(ap.read_text())
    if file_digest(ap)!=plan['admission_sha256'] or a.get('Q3_values_read') is not False:
        raise ValueError('Candidate data identity changed')
    bars=read_bars(src,a,'minute');h=job['horizon'];op=src/f'origins_minute_h{h}.npy'
    if file_digest(op)!=job['origins_sha256']:raise ValueError('Candidate origins changed')
    origins=np.load(op,allow_pickle=False);fold=rolling_folds(bars,origins,h)[job['fold']-1]
    train,fit,stop,val=validate_split(bars,fold,h)
    for name,x in [('train',train),('fit',fit),('stop',stop),('validation',val)]:
        if digest_indices(x)!=job['split_sha256'][name]:raise ValueError('Fold split identity changed')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    cfg=FoldConfig('minute',h,job['target'],plan['lookback'],plan['seasonal_period'],tuple(plan['ridge_alphas']),
                   plan['max_epochs'],plan['patience'],plan['batch_size'],plan['learning_rate'],plan['seed'],
                   file_digest(plan_path))
    directory=plan_path.parent;folder=directory/'jobs'/key
    print(json.dumps({'stage':'worker_started','job':key,'fit_origins':len(fit),
                      'validation_origins':len(val),'cores':list(slot)}),flush=True)
    result=run_development_fold(bars,fold,a['split_actions'],cfg,folder,
                                'private-minute-matrix:'+file_digest(ap),device='cpu')
    checked=check_fold(folder)
    if not all(checked['current_source_matches_recorded'].values()):raise ValueError('Worker source changed')
    checked['scope']=plan['status'];vp=directory/'verifications'/(key+'.json');save_new(vp,checked)
    print(json.dumps({'stage':'worker_completed','job':key,'receipt_sha256':file_digest(folder/'fold_receipt.json'),
                      'verification_sha256':file_digest(vp),'wall_seconds':result['wall_seconds']}),flush=True)


def orchestrate():
    from multiscale.bars import file_digest
    from multiscale.development import atomic_json
    from multiscale.spec import OUT
    parent=OUT/'private_minute_matrix_development';parent.mkdir(parents=True,exist_ok=True)
    with (parent/'active_queue.lock').open('a+') as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Another minute matrix queue is already active') from None
        plan=prepare_plan()
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');directory=parent/stamp;directory.mkdir(exist_ok=False)
        (directory/'jobs').mkdir();(directory/'logs').mkdir();(directory/'verifications').mkdir()
        exposure=OUT/'POST_CUTOFF_EXPOSURE.md';before=exposure.read_bytes();(directory/'EXPOSURE_BEFORE.md').write_bytes(before)
        plan['exposure_before_sha256']=hashlib.sha256(before).hexdigest()
        pp=directory/'PRIVATE_PROTOCOL.json';save_new(pp,plan)
        with exposure.open('a',encoding='utf8') as f:
            f.write('\n## 分钟剩余矩阵：有界并行开发队列\n\n'
                    f'{stamp}：用户要求直接并行实验，预先登记剩余 {len(plan["jobs"])} 个分钟开发折；'
                    '已完成和在运行的折不重复启动。最多六个双线程 CPU 工作进程，'
                    '总墙钟上限两小时，监控可用内存并保留检查点；失败或超时不自动重试。'
                    f'全部为已曝光的本机开发，2026Q3 保持封存。方案见 {pp}。\n')
        state={'status':'running','started_at_utc':datetime.now(timezone.utc).isoformat(),
               'protocol':str(pp),'protocol_sha256':file_digest(pp),'maximum_wall_seconds':plan['maximum_wall_seconds'],
               'total':len(plan['jobs']),'completed':0,'jobs':{j['key']:{'status':'pending'} for j in plan['jobs']}}
        state_path=directory/'PRIVATE_QUEUE_STATE.json';atomic_json(state_path,state)
        started=time.monotonic();active={};streams={};cancelled=False;reason=None
        def stop(signum,frame):
            nonlocal cancelled
            cancelled=True
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        print(json.dumps({'status':'minute_matrix_queue_started','directory':str(directory),'pending':len(plan['jobs']),
                          'max_workers':len(SLOTS),'threads_per_worker':2,'maximum_seconds':7200},ensure_ascii=False),flush=True)
        try:
            while True:
                for key,(child,slot) in list(active.items()):
                    code=child.poll()
                    if code is None:continue
                    streams[key].flush();streams[key].close();del streams[key];del active[key]
                    entry=state['jobs'][key]
                    if code==0:
                        receipt=directory/'jobs'/key/'fold_receipt.json';verification=directory/'verifications'/(key+'.json')
                        if not receipt.is_file() or not verification.is_file():raise ValueError('Completed job lacks verification')
                        entry.update(status='completed',exit_code=0,receipt_sha256=file_digest(receipt),
                                     verification_sha256=file_digest(verification))
                        state['completed']+=1
                    else:
                        entry.update(status='failed',exit_code=code);reason='worker_failed';cancelled=True
                    state['elapsed_seconds']=time.monotonic()-started;atomic_json(state_path,state)
                    print(json.dumps({'stage':'job_finished','job':key,'status':entry['status'],
                                      'completed':state['completed'],'total':state['total']}),flush=True)
                if state['completed']==state['total']:break
                elapsed=time.monotonic()-started
                low_memory=available_memory_bytes()<plan['stop_available_memory_bytes']
                if cancelled or low_memory or elapsed>=plan['maximum_wall_seconds']:
                    reason=reason or ('cancelled' if cancelled else ('low_available_memory' if low_memory else 'wall_limit'))
                    state['status']='interrupted_or_resource_bound';state['reason']=reason;atomic_json(state_path,state)
                    for key,(child,_) in active.items():
                        code=terminate_child(child)
                        state['jobs'][key].update(status='unrecorded_completion' if code==0 else 'interrupted',
                                                  exit_code=code)
                    break
                external_busy,statuses=busy_external_slots(plan)
                used=external_busy|{slot for _,slot in active.values()}
                available=[slot for slot in SLOTS if slot not in used]
                pending=[j for j in plan['jobs'] if state['jobs'][j['key']]['status']=='pending']
                while available and pending and available_memory_bytes()>=plan['start_minimum_available_memory_bytes']:
                    slot=available.pop(0);job=pending.pop(0);key=job['key']
                    log=(directory/'logs'/(key+'.log')).open('x',encoding='utf8');streams[key]=log
                    args=[sys.executable,str(Path(__file__).resolve()),'--worker','--plan',str(pp),'--job',key,
                          '--cores',','.join(map(str,slot))]
                    child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    active[key]=(child,slot)
                    state['jobs'][key]={'status':'running','pid':child.pid,'cores':list(slot),
                                        'log':str(directory/'logs'/(key+'.log'))}
                    state['elapsed_seconds']=time.monotonic()-started;atomic_json(state_path,state)
                    print(json.dumps({'stage':'job_started','job':key,'cores':list(slot),'pid':child.pid}),flush=True)
                time.sleep(3)
            if state['completed']==state['total']:
                state['status']='completed_scheduled_jobs_only'
                with exposure.open('a',encoding='utf8') as f:
                    f.write(f'\n{stamp}：剩余分钟队列完成 {state["completed"]} 折，状态见 {state_path}；仅为开发。\n')
            state['elapsed_seconds']=time.monotonic()-started;atomic_json(state_path,state)
            print(json.dumps({'status':state['status'],'directory':str(directory),'completed':state['completed'],
                              'total':state['total'],'reason':reason,'elapsed_seconds':round(state['elapsed_seconds'],1)},
                             ensure_ascii=False),flush=True)
        except Exception as exc:
            state['status']='failed';state['error']=f'{type(exc).__name__}: {exc}'
            state['elapsed_seconds']=time.monotonic()-started
            for key,(child,_) in active.items():
                code=terminate_child(child)
                state['jobs'][key].update(status='unrecorded_completion' if code==0 else 'interrupted_after_error',
                                          exit_code=code)
            atomic_json(state_path,state)
            raise
        finally:
            for child,_ in active.values():
                terminate_child(child)
            for stream in streams.values():stream.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true')
    parser.add_argument('--inspect',action='store_true');parser.add_argument('--plan')
    parser.add_argument('--job');parser.add_argument('--cores');args=parser.parse_args()
    try:
        if args.worker:worker(args.plan,args.job,[int(x) for x in args.cores.split(',')])
        elif args.inspect:
            plan=prepare_plan();busy,statuses=busy_external_slots(plan)
            print(json.dumps({'status':'read_only_queue_preflight','new_jobs':len(plan['jobs']),
                              'excluded_jobs':len(plan['excluded_external_jobs']),'busy_external_slots':sorted(busy),
                              'first_jobs':[j['key'] for j in plan['jobs'][:8]],
                              'available_memory_bytes':available_memory_bytes()},ensure_ascii=False))
        else:orchestrate()
    except Exception as exc:
        print(json.dumps({'status':'minute_matrix_queue_failed','error':f'{type(exc).__name__}: {exc}'},
                         ensure_ascii=False),file=sys.stderr,flush=True)
        raise
