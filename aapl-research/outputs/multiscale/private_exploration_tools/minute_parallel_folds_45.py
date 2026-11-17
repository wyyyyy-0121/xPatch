"""Two bounded CPU workers for predeclared minute H60 development folds 4 and 5."""
from __future__ import annotations

import argparse,json,os,signal,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path


def save_new(path,value):
    with path.open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def prepare():
    import numpy as np
    from multiscale.bars import eligible_origins,file_digest
    from multiscale.development import validate_split
    from multiscale.matrix_executor import read_bars
    from multiscale.spec import OUT,PROFILES,ROOT
    from multiscale.splits import rolling_folds
    from multiscale.training import digest_indices
    source=OUT/'admission/alpaca_20261107T052228573242Z';ap=source/'admission.json';a=json.loads(ap.read_text())
    prior=OUT/'private_minute_full_neural/20261107T140317864464Z';prior_check=json.loads((prior/'PRIVATE_INDEPENDENT_CHECK.json').read_text())
    prior_plan=json.loads((prior/'PRIVATE_PROTOCOL.json').read_text())
    if (a.get('Q3_values_read') is not False or prior_check.get('status')!='private_full_minute_neural_independently_recomputed' or
            prior_check.get('result_sha256')!=file_digest(prior/'PRIVATE_RESULT.json') or
            prior_plan.get('admission_sha256')!=file_digest(ap)):
        raise ValueError('Required historical candidate and prior verification are unavailable')
    bars=read_bars(source,a,'minute');op=source/'origins_minute_h60.npy';profile=next(p for p in PROFILES if p.name=='minute')
    if file_digest(op)!=a['files'][op.name]:raise ValueError('Origin identity changed')
    origins=np.load(op,allow_pickle=False)
    if not np.array_equal(origins,eligible_origins(bars,max(profile.lookbacks),60,same_session=True)):
        raise ValueError('Minute origins differ from the candidate')
    folds=rolling_folds(bars,origins,60)
    if len(folds)!=5:raise ValueError('Expected five chronological folds')
    jobs=[]
    for number,cores in [(4,[8,10]),(5,[12,14])]:
        if not set(cores).issubset(os.sched_getaffinity(0)):raise ValueError('CPU affinity unavailable')
        train,fit,stop,val=validate_split(bars,folds[number-1],60)
        jobs.append({'fold':number,'cores':cores,'train_origins':len(train),'fit_origins':len(fit),
                     'stop_origins':len(stop),'validation_origins':len(val),
                     'split_sha256':{n:digest_indices(x) for n,x in [('train',train),('fit',fit),('stop',stop),('validation',val)]}})
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');parent=OUT/'private_minute_parallel_folds';parent.mkdir(parents=True,exist_ok=True)
    directory=parent/stamp;directory.mkdir(exist_ok=False);exposure=OUT/'POST_CUTOFF_EXPOSURE.md';before=exposure.read_bytes();(directory/'EXPOSURE_BEFORE.md').write_bytes(before)
    sources=('multiscale/development.py','multiscale/training.py','multiscale/models.py','multiscale/features.py',
             'multiscale/bars.py','multiscale/baselines.py','multiscale/spec.py','multiscale/splits.py',
             'multiscale/evaluation.py','multiscale/ridge.py','multiscale/matrix_executor.py','multiscale/matrix_gate.py',
             'multiscale/verify_development.py','multiscale/mechanism.py','aapl/models.py','aapl/vendor/xpatch_network.py')
    plan={'status':'private_parallel_development_only_not_publication_admitted','created_at_utc':datetime.now(timezone.utc).isoformat(),
          'user_request':'我要开始做实验；可以并行的任务就并行','scope':'personal non-commercial local development',
          'admission_sha256':file_digest(ap),'bar_sha256':a['files']['bars_minute.npz'],'origins_sha256':a['files'][op.name],
          'prior_first_fold_verification_sha256':file_digest(prior/'PRIVATE_INDEPENDENT_CHECK.json'),
          'task':'minute H60 cumulative log return','models':['last_close','drift','seasonal','ridge','mlp','gru','lstm','xpatch'],
          'lookback':480,'seasonal_period':390,'max_epochs':20,'patience':3,'batch_size':128,'learning_rate':0.001,
          'ridge_alphas':[0.1,1.0,10.0],'seed':2026,'jobs':jobs,'device':'cpu','threads_per_worker':2,
          'workers':2,'maximum_wall_seconds':5400,'data_cutoff':'2026-06-30','Q3_values_read':False,'independent_test':False,
          'source_sha256':{p:file_digest(ROOT/p) for p in sources},'script_sha256':file_digest(Path(__file__)),
          'exposure_before_sha256':__import__('hashlib').sha256(before).hexdigest(),
          'publication_rights':'not confirmed; keep results local',
          'limitations':['Folds 4 and 5 were fixed after the first-fold results were seen; all are development data.',
                         'A short synthetic concurrency measurement does not guarantee full-run speedup.',
                         'A wall-clock stop may leave incomplete fits; no automatic budget reset or retry.',
                         'Actual account and publication rights remain unconfirmed.']}
    pp=directory/'PRIVATE_PROTOCOL.json';save_new(pp,plan)
    with exposure.open('a',encoding='utf8') as f:
        f.write('\n## 个人内部分钟 H60 并行开发折：事后登记\n\n'
                f'{stamp}：在已看到第一折结果后，按原计划固定完整分钟起点的第 4、5 个时间折，'
                '两组 CPU 各两线程并行运行八类模型，墙钟上限 90 分钟。'
                '这些结果都是已曝光的历史开发，不是独立确认；账号具体协议及发表权待核实，2026Q3 保持封存。'
                f'设置见 {pp}。\n')
    return directory,pp,plan


def worker(plan_path,number,cores):
    os.sched_setaffinity(0,set(cores))
    import numpy as np
    import torch
    from multiscale.bars import file_digest
    from multiscale.development import FoldConfig,run_development_fold,validate_split
    from multiscale.matrix_executor import read_bars
    from multiscale.spec import OUT,ROOT
    from multiscale.splits import rolling_folds
    from multiscale.training import digest_indices
    from multiscale.verify_development import check_fold
    plan_path=Path(plan_path).resolve();plan=json.loads(plan_path.read_text());directory=plan_path.parent
    job=next(j for j in plan['jobs'] if j['fold']==number)
    if (job['cores']!=cores or file_digest(Path(__file__))!=plan['script_sha256'] or
            any(file_digest(ROOT/p)!=h for p,h in plan['source_sha256'].items())):
        raise ValueError('Worker source or CPU identity differs')
    src=OUT/'admission/alpaca_20261107T052228573242Z';ap=src/'admission.json';a=json.loads(ap.read_text())
    if file_digest(ap)!=plan['admission_sha256'] or a.get('Q3_values_read') is not False:raise ValueError('Candidate identity changed')
    bars=read_bars(src,a,'minute');op=src/'origins_minute_h60.npy'
    if file_digest(op)!=plan['origins_sha256']:raise ValueError('Origin artifact changed')
    origins=np.load(op,allow_pickle=False);folds=rolling_folds(bars,origins,60);fold=folds[number-1]
    train,fit,stop,val=validate_split(bars,fold,60)
    for name,x in [('train',train),('fit',fit),('stop',stop),('validation',val)]:
        if digest_indices(x)!=job['split_sha256'][name]:raise ValueError('Fold split identity changed')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    cfg=FoldConfig('minute',60,'future_log_return',480,390,tuple(plan['ridge_alphas']),
                   plan['max_epochs'],plan['patience'],plan['batch_size'],plan['learning_rate'],plan['seed'],file_digest(plan_path))
    fold_dir=directory/f'fold_{number:02d}'
    print(json.dumps({'stage':'worker_started','fold':number,'fit_origins':len(fit),'validation_origins':len(val),'cores':cores}),flush=True)
    result=run_development_fold(bars,fold,a['split_actions'],cfg,fold_dir,'private-minute-parallel:'+file_digest(ap),device='cpu')
    verified=check_fold(fold_dir)
    if not all(verified['current_source_matches_recorded'].values()):raise ValueError('Worker code changed during the fold')
    verified['scope']=plan['status'];vp=directory/f'fold_{number:02d}_verification.json';save_new(vp,verified)
    print(json.dumps({'stage':'worker_completed','fold':number,'receipt_sha256':file_digest(fold_dir/'fold_receipt.json'),
                      'verification_sha256':file_digest(vp),'wall_seconds':result['wall_seconds']}),flush=True)


def orchestrate():
    from multiscale.development import atomic_json
    directory,plan_path,plan=prepare();started=time.monotonic();children={};streams={};cancelled=False
    state={'status':'running','started_at_utc':datetime.now(timezone.utc).isoformat(),
           'protocol':str(plan_path),'maximum_wall_seconds':plan['maximum_wall_seconds'],'workers':{}}
    state_path=directory/'PRIVATE_QUEUE_STATE.json';atomic_json(state_path,state)
    def stop(signum,frame):
        nonlocal cancelled
        cancelled=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for job in plan['jobs']:
            number=job['fold'];log=(directory/f'fold_{number:02d}.log').open('x',encoding='utf8');streams[number]=log
            args=[sys.executable,str(Path(__file__).resolve()),'--worker','--plan',str(plan_path),'--fold',str(number),
                  '--cores',','.join(map(str,job['cores']))]
            child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);children[number]=child
            state['workers'][str(number)]={'status':'running','pid':child.pid,'log':str(directory/f'fold_{number:02d}.log')}
        atomic_json(state_path,state)
        print(json.dumps({'status':'private_minute_parallel_started','directory':str(directory),
                          'folds':[4,5],'workers':2,'threads_per_worker':2,'maximum_seconds':5400},ensure_ascii=False),flush=True)
        while True:
            for number,child in children.items():
                entry=state['workers'][str(number)]
                if entry['status']!='running':continue
                code=child.poll()
                if code is None:continue
                streams[number].flush()
                entry.update(status='completed' if code==0 else 'failed',exit_code=code)
                state['elapsed_seconds']=time.monotonic()-started;atomic_json(state_path,state)
                print(json.dumps({'stage':'fold_finished','fold':number,'status':entry['status'],
                                  'elapsed_seconds':round(state['elapsed_seconds'],1)}),flush=True)
                if code!=0:cancelled=True
            if all(p.poll() is not None for p in children.values()):break
            if cancelled or time.monotonic()-started>=plan['maximum_wall_seconds']:
                state['status']='interrupted_or_budget_exhausted';atomic_json(state_path,state)
                for p in children.values():
                    if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
                for p in children.values():
                    try:p.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                break
            time.sleep(2)
        for number,p in children.items():
            if state['workers'][str(number)]['status']=='running':
                state['workers'][str(number)].update(status='interrupted',exit_code=p.returncode)
        if all(v['status']=='completed' for v in state['workers'].values()):
            state['status']='completed_private_development_only'
            with (directory.parent.parent/'POST_CUTOFF_EXPOSURE.md').open('a',encoding='utf8') as f:
                f.write(f"\n{directory.name}：分钟 H60 第 4、5 折个人内部探索已完成，状态见 {state_path}；"
                        '这些是已曝光的历史开发结果，不是独立确认。\n')
        elif state['status']=='running':state['status']='failed'
        state['elapsed_seconds']=time.monotonic()-started;atomic_json(state_path,state)
        print(json.dumps({'status':state['status'],'directory':str(directory),'workers':state['workers'],
                          'elapsed_seconds':round(state['elapsed_seconds'],1)},ensure_ascii=False),flush=True)
    finally:
        for p in children.values():
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        for f in streams.values():f.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');parser.add_argument('--plan');parser.add_argument('--fold',type=int);parser.add_argument('--cores');args=parser.parse_args()
    try:
        if args.worker:worker(args.plan,args.fold,[int(x) for x in args.cores.split(',')])
        else:orchestrate()
    except Exception as exc:
        print(json.dumps({'status':'private_minute_parallel_failed','error':f'{type(exc).__name__}: {exc}'},ensure_ascii=False),file=sys.stderr,flush=True);raise
