"""Run both H15 targets in bounded parallel local development workers."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def save_new(path,value):
    with path.open('x',encoding='utf8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')


def available_memory_bytes():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])*1024
    raise ValueError('Cannot assess available memory')


def prepare():
    import numpy as np
    from multiscale.bars import eligible_origins,file_digest
    from multiscale.development import validate_split
    from multiscale.matrix_executor import read_bars
    from multiscale.spec import OUT,PROFILES,ROOT
    from multiscale.splits import rolling_folds
    from multiscale.training import digest_indices
    source=OUT/'admission/alpaca_20261107T052228573242Z'
    ap=source/'admission.json'
    a=json.loads(ap.read_text())
    prior=OUT/'private_minute_full_neural/20261107T140317864464Z'
    prior_check=json.loads((prior/'PRIVATE_INDEPENDENT_CHECK.json').read_text())
    prior_plan=json.loads((prior/'PRIVATE_PROTOCOL.json').read_text())
    if (a.get('Q3_values_read') is not False or
            prior_check.get('status')!='private_full_minute_neural_independently_recomputed' or
            prior_check.get('result_sha256')!=file_digest(prior/'PRIVATE_RESULT.json') or
            prior_plan.get('admission_sha256')!=file_digest(ap)):
        raise ValueError('Historical candidate or prior verification changed')
    bars=read_bars(source,a,'minute')
    op=source/'origins_minute_h15.npy'
    profile=next(p for p in PROFILES if p.name=='minute')
    if file_digest(op)!=a['files'][op.name]:raise ValueError('Origin identity changed')
    origins=np.load(op,allow_pickle=False)
    if not np.array_equal(origins,eligible_origins(bars,max(profile.lookbacks),15,same_session=True)):
        raise ValueError('Minute H15 origins differ from candidate')
    folds=rolling_folds(bars,origins,15)
    if len(folds)!=5:raise ValueError('Expected five chronological folds')
    train,fit,stop,val=validate_split(bars,folds[0],15)
    jobs=[]
    for key,target,cores in [('h15_return_f1','future_log_return',[4,6]),
                              ('h15_price_f1','future_close',[20,22])]:
        if not set(cores).issubset(os.sched_getaffinity(0)):raise ValueError('CPU affinity unavailable')
        jobs.append({'key':key,'horizon':15,'target':target,'fold':1,'cores':cores,
                     'train_origins':len(train),'fit_origins':len(fit),'stop_origins':len(stop),
                     'validation_origins':len(val),
                     'split_sha256':{n:digest_indices(x) for n,x in [('train',train),('fit',fit),
                                                                      ('stop',stop),('validation',val)]}})
    if available_memory_bytes()<4*1024**3:raise RuntimeError('Less than 4 GiB of available memory')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    parent=OUT/'private_minute_breadth_development'
    parent.mkdir(parents=True,exist_ok=True)
    directory=parent/stamp
    directory.mkdir(exist_ok=False)
    exposure=OUT/'POST_CUTOFF_EXPOSURE.md'
    before=exposure.read_bytes()
    (directory/'EXPOSURE_BEFORE.md').write_bytes(before)
    sources=('multiscale/development.py','multiscale/training.py','multiscale/models.py',
             'multiscale/features.py','multiscale/bars.py','multiscale/baselines.py',
             'multiscale/spec.py','multiscale/splits.py','multiscale/evaluation.py','multiscale/ridge.py',
             'multiscale/matrix_executor.py','multiscale/matrix_gate.py','multiscale/verify_development.py',
             'multiscale/mechanism.py','aapl/models.py','aapl/vendor/xpatch_network.py')
    plan={'status':'private_breadth_development_only_not_publication_admitted',
          'created_at_utc':datetime.now(timezone.utc).isoformat(),
          'user_request':'直接做实验; 可以并行的任务就并行',
          'scope':'personal non-commercial local development','admission_sha256':file_digest(ap),
          'bar_sha256':a['files']['bars_minute.npz'],'origins_sha256':a['files'][op.name],
          'prior_first_fold_verification_sha256':file_digest(prior/'PRIVATE_INDEPENDENT_CHECK.json'),
          'task':'minute H15 price and cumulative log return, first full-origin fold',
          'models':['last_close','drift','seasonal','ridge','mlp','gru','lstm','xpatch'],
          'lookback':480,'seasonal_period':390,'max_epochs':20,'patience':3,'batch_size':128,
          'learning_rate':0.001,'ridge_alphas':[0.1,1.0,10.0],'seed':2026,
          'jobs':jobs,'device':'cpu','threads_per_worker':2,'workers':2,'maximum_wall_seconds':2700,
          'minimum_available_memory_bytes':2*1024**3,'data_cutoff':'2026-06-30',
          'Q3_values_read':False,'independent_test':False,
          'source_sha256':{p:file_digest(ROOT/p) for p in sources},'script_sha256':file_digest(Path(__file__)),
          'exposure_before_sha256':hashlib.sha256(before).hexdigest(),
          'publication_rights':'not confirmed; keep results local',
          'limitations':['Only the first fold for each H15 target, not the full minute matrix.',
                         'The data were previously exposed; these are development results.',
                         'Stop on the wall-clock or memory bound may leave incomplete model checkpoints.',
                         'Account-specific and publication rights remain unconfirmed.']}
    pp=directory/'PRIVATE_PROTOCOL.json'
    save_new(pp,plan)
    with exposure.open('a',encoding='utf8') as f:
        f.write('\n## 个人内部分钟 H15 价格与收益首折：事后登记\n\n'
                f'{stamp}：用户要求并行实验，按原计划运行分钟 H15 价格与累计收益各一个完整起点首折；'
                '两个 CPU 工作进程各用两线程，墙钟上限 45 分钟并监控可用内存。'
                '结果是已曝光的本机个人开发，不是独立确认；发表权待核实，2026Q3 保持封存。'
                f'方案见 {pp}。\n')
    return directory,pp,plan


def worker(plan_path,key):
    plan_path=Path(plan_path).resolve()
    plan=json.loads(plan_path.read_text())
    job=next(j for j in plan['jobs'] if j['key']==key)
    os.sched_setaffinity(0,set(job['cores']))
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
    src=OUT/'admission/alpaca_20261107T052228573242Z'
    ap=src/'admission.json'
    a=json.loads(ap.read_text())
    if file_digest(ap)!=plan['admission_sha256'] or a.get('Q3_values_read') is not False:
        raise ValueError('Candidate identity changed')
    bars=read_bars(src,a,'minute')
    op=src/'origins_minute_h15.npy'
    if file_digest(op)!=plan['origins_sha256']:raise ValueError('Origin artifact changed')
    origins=np.load(op,allow_pickle=False)
    fold=rolling_folds(bars,origins,15)[job['fold']-1]
    train,fit,stop,val=validate_split(bars,fold,15)
    for name,x in [('train',train),('fit',fit),('stop',stop),('validation',val)]:
        if digest_indices(x)!=job['split_sha256'][name]:raise ValueError('Fold split identity changed')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg=FoldConfig('minute',15,job['target'],480,390,tuple(plan['ridge_alphas']),plan['max_epochs'],
                   plan['patience'],plan['batch_size'],plan['learning_rate'],plan['seed'],file_digest(plan_path))
    fold_dir=plan_path.parent/key
    print(json.dumps({'stage':'worker_started','job':key,'fit_origins':len(fit),
                      'validation_origins':len(val),'cores':job['cores']}),flush=True)
    result=run_development_fold(bars,fold,a['split_actions'],cfg,fold_dir,
                                'private-minute-breadth:'+file_digest(ap),device='cpu')
    checked=check_fold(fold_dir)
    if not all(checked['current_source_matches_recorded'].values()):raise ValueError('Worker source changed')
    checked['scope']=plan['status']
    vp=plan_path.parent/(key+'_verification.json')
    save_new(vp,checked)
    print(json.dumps({'stage':'worker_completed','job':key,'receipt_sha256':file_digest(fold_dir/'fold_receipt.json'),
                      'verification_sha256':file_digest(vp),'wall_seconds':result['wall_seconds']}),flush=True)


def orchestrate():
    from multiscale.development import atomic_json
    directory,plan_path,plan=prepare()
    started=time.monotonic()
    children={};streams={};cancelled=False
    state={'status':'running','started_at_utc':datetime.now(timezone.utc).isoformat(),
           'protocol':str(plan_path),'maximum_wall_seconds':plan['maximum_wall_seconds'],'workers':{}}
    state_path=directory/'PRIVATE_QUEUE_STATE.json'
    atomic_json(state_path,state)
    def stop(signum,frame):
        nonlocal cancelled
        cancelled=True
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    try:
        for job in plan['jobs']:
            key=job['key']
            log=(directory/(key+'.log')).open('x',encoding='utf8')
            streams[key]=log
            args=[sys.executable,str(Path(__file__).resolve()),'--worker','--plan',str(plan_path),'--job',key]
            child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children[key]=child
            state['workers'][key]={'status':'running','pid':child.pid,'log':str(directory/(key+'.log'))}
        atomic_json(state_path,state)
        print(json.dumps({'status':'minute_h15_breadth_started','directory':str(directory),'jobs':list(children),
                          'workers':2,'threads_per_worker':2,'maximum_seconds':2700},ensure_ascii=False),flush=True)
        while True:
            for key,child in children.items():
                entry=state['workers'][key]
                if entry['status']!='running':continue
                code=child.poll()
                if code is None:continue
                streams[key].flush()
                entry.update(status='completed' if code==0 else 'failed',exit_code=code)
                state['elapsed_seconds']=time.monotonic()-started
                atomic_json(state_path,state)
                print(json.dumps({'stage':'job_finished','job':key,'status':entry['status'],
                                  'elapsed_seconds':round(state['elapsed_seconds'],1)}),flush=True)
                if code!=0:cancelled=True
            if all(p.poll() is not None for p in children.values()):break
            memory_low=available_memory_bytes()<plan['minimum_available_memory_bytes']
            if cancelled or memory_low or time.monotonic()-started>=plan['maximum_wall_seconds']:
                state['status']='interrupted_or_resource_bound'
                state['reason']='cancelled_or_worker_failed' if cancelled else ('low_available_memory' if memory_low else 'wall_limit')
                atomic_json(state_path,state)
                for child in children.values():
                    if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
                for child in children.values():
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL)
                        child.wait()
                break
            time.sleep(2)
        for key,child in children.items():
            if state['workers'][key]['status']=='running':
                state['workers'][key].update(status='interrupted',exit_code=child.returncode)
        if all(v['status']=='completed' for v in state['workers'].values()):
            state['status']='completed_private_development_only'
            with (directory.parent.parent/'POST_CUTOFF_EXPOSURE.md').open('a',encoding='utf8') as f:
                f.write(f"\n{directory.name}：分钟 H15 两项目标的首折已完成，状态见 {state_path}；"
                        '仅为已曝光的本机开发结果。\n')
        elif state['status']=='running':state['status']='failed'
        state['elapsed_seconds']=time.monotonic()-started
        atomic_json(state_path,state)
        print(json.dumps({'status':state['status'],'directory':str(directory),'workers':state['workers'],
                          'elapsed_seconds':round(state['elapsed_seconds'],1)},ensure_ascii=False),flush=True)
    finally:
        for child in children.values():
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait()
        for stream in streams.values():stream.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--plan')
    parser.add_argument('--job')
    args=parser.parse_args()
    try:
        worker(args.plan,args.job) if args.worker else orchestrate()
    except Exception as exc:
        print(json.dumps({'status':'minute_h15_breadth_failed','error':f'{type(exc).__name__}: {exc}'},
                         ensure_ascii=False),file=sys.stderr,flush=True)
        raise
