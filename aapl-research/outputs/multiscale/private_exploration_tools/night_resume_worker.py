"""Verify or resume one frozen minute fold without changing its identity."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot',required=True)
    parser.add_argument('--pid',required=True,type=int)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--cores',default='24')
    args=parser.parse_args()
    cores=[int(x) for x in args.cores.split(',')]
    os.sched_setaffinity(0,set(cores))
    import numpy as np
    import pandas as pd
    import torch
    from multiscale.bars import file_digest
    from multiscale.development import FoldConfig,bars_identity,run_development_fold,validate_split
    from multiscale.features import CloseWindows,fit_scaler
    from multiscale.matrix_executor import read_bars
    from multiscale.spec import OUT,ROOT
    from multiscale.splits import rolling_folds
    from multiscale.training import TrainConfig,digest_indices,identity as training_identity
    from multiscale.verify_development import check_fold
    torch.set_num_threads(1 if args.check else len(cores))
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    snapshot=json.loads(Path(args.snapshot).read_text())
    if snapshot['status']!='snapshot_complete':raise ValueError('Checkpoint snapshot is incomplete')
    item=next(w for w in snapshot['workers'] if w['pid']==args.pid)
    plan_path=Path(item['plan']).resolve();folder=Path(item['fold_dir']).resolve();plan=json.loads(plan_path.read_text())
    if file_digest(plan_path)!=item['plan_sha256'] or not folder.is_relative_to(OUT.resolve()):
        raise ValueError('Plan or output location changed')
    state=json.loads((folder/'fold_state.json').read_text())
    if state['fold_identity']!=item['fold_identity'] or state.get('Q3_values_read') is not False:
        raise ValueError('Fold identity or protected period changed')
    fields=state['identity_fields']
    calculated=hashlib.sha256(json.dumps(fields,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if calculated!=state['fold_identity']:raise ValueError('Fold identity digest is invalid')
    if any(file_digest(ROOT/name)!=sha for name,sha in state['source_sha256'].items()):
        raise ValueError('Source files changed since the paused fold')
    expected_runtime={'device':'cpu','python':sys.version.split()[0],'numpy':np.__version__,'pandas':pd.__version__,
                      'torch':str(torch.__version__),'cuda':torch.version.cuda,'gpu':None}
    if fields['runtime']!=expected_runtime:raise ValueError('Runtime differs from the saved fold')
    source=OUT/'admission/alpaca_20261107T052228573242Z';ap=source/'admission.json';a=json.loads(ap.read_text())
    if file_digest(ap)!=plan['admission_sha256'] or a.get('Q3_values_read') is not False:
        raise ValueError('Candidate source changed')
    script=item['script'];key=item['job']
    if script=='minute_parallel_folds_45.py':
        job=next(j for j in plan['jobs'] if j['fold']==int(key));h=60;target='future_log_return'
        expected_data='private-minute-parallel:'+file_digest(ap)
        verification=plan_path.parent/f'fold_{int(key):02d}_verification.json'
    else:
        job=next(j for j in plan['jobs'] if j['key']==key);h=job['horizon'];target=job['target']
        expected_data=('private-minute-matrix:' if script=='minute_matrix_queue.py' else 'private-minute-breadth:')+file_digest(ap)
        verification=(plan_path.parent/'verifications'/(key+'.json') if script=='minute_matrix_queue.py'
                      else plan_path.parent/(key+'_verification.json'))
    if state['data_identity']!=expected_data:raise ValueError('Fold data identity changed')
    cfg_dict=dict(state['config']);cfg_dict['ridge_alphas']=tuple(cfg_dict['ridge_alphas']);cfg=FoldConfig(**cfg_dict)
    expected_cfg={'profile':'minute','horizon':h,'target':target,'lookback':plan['lookback'],
                  'seasonal_period':plan['seasonal_period'],'ridge_alphas':tuple(plan['ridge_alphas']),
                  'max_epochs':plan['max_epochs'],'patience':plan['patience'],'batch_size':plan['batch_size'],
                  'learning_rate':plan['learning_rate'],'seed':plan['seed'],'protocol_identity':file_digest(plan_path)}
    if asdict(cfg)!=expected_cfg:raise ValueError('Fold configuration differs from its original plan')
    bars=read_bars(source,a,'minute');op=source/f'origins_minute_h{h}.npy'
    expected_origin_hash=(plan['origins_sha256'] if script in ('minute_parallel_folds_45.py','minute_breadth_h30_queue.py')
                          else job['origins_sha256'])
    if file_digest(op)!=expected_origin_hash or bars_identity(bars)!=fields['bars_sha256']:
        raise ValueError('Bar or origin identity changed')
    origins=np.load(op,allow_pickle=False)
    number=int(key) if script=='minute_parallel_folds_45.py' else job['fold']
    fold=rolling_folds(bars,origins,h)[number-1];train,fit,stop,val=validate_split(bars,fold,h)
    for name,x in [('train',train),('fit',fit),('stop',stop),('validation',val)]:
        if digest_indices(x)!=fields[name+'_indices_sha256'] or digest_indices(x)!=job['split_sha256'][name]:
            raise ValueError('Chronological split identity changed')
    actions=fields['actions'];windows=CloseWindows(bars,cfg.lookback,actions)
    active=[n for n in ('mlp','gru','lstm','xpatch') if n not in state['models'] and (folder/n/'latest.pt').is_file()]
    if len(active)!=1:raise ValueError('Could not uniquely identify the active checkpointed neural model')
    name=active[0];model_cfg=TrainConfig(name,cfg.lookback,cfg.horizon,cfg.target,cfg.max_epochs,
                                         cfg.patience,cfg.batch_size,cfg.learning_rate,cfg.seed)
    training_data=state['data_identity']+':'+cfg.protocol_identity+':'+state['fold_identity']
    expected_identity=training_identity(model_cfg,training_data,fit,stop,'cpu')
    checkpoint=torch.load(folder/name/'latest.pt',map_location='cpu',weights_only=True)
    scaler=fit_scaler(windows,fit,h,target)
    if checkpoint['identity']!=expected_identity or checkpoint['scaler']!=asdict(scaler):
        raise ValueError('Checkpoint optimizer identity or scaler changed')
    if args.check:
        print(json.dumps({'status':'checkpoint_resume_identity_verified','pid':args.pid,'job':key,'model':name,
                          'checkpoint_epoch':checkpoint['state']['epoch'],'checkpoint_steps':checkpoint['state']['steps'],
                          'data_and_split_match':True,'runtime_match':True,'training_started':False},ensure_ascii=False),flush=True)
        return
    deadline=datetime.fromisoformat(snapshot['resume_at_local'])
    if datetime.now(timezone.utc)<deadline.astimezone(timezone.utc):raise ValueError('Resume time has not arrived')
    result=run_development_fold(bars,fold,actions,cfg,folder,state['data_identity'],device='cpu',resume=True)
    checked=check_fold(folder)
    if not all(checked['current_source_matches_recorded'].values()):raise ValueError('Resumed source changed')
    checked['scope']='resumed_original_private_development'
    if verification.exists():
        existing=json.loads(verification.read_text())
        if existing.get('fold_receipt_sha256')!=file_digest(folder/'fold_receipt.json'):
            verification=verification.with_name(verification.stem+'_night_resumed.json')
    if not verification.exists():
        with verification.open('x',encoding='utf8') as f:json.dump(checked,f,indent=2,ensure_ascii=False);f.write('\n')
    print(json.dumps({'status':'night_resumed_fold_verified','pid':args.pid,'job':key,
                      'verification':str(verification),'receipt_sha256':file_digest(folder/'fold_receipt.json'),
                      'wall_seconds':result['wall_seconds']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(json.dumps({'status':'night_resume_worker_error','error':f'{type(exc).__name__}: {exc}'},
                         ensure_ascii=False),file=sys.stderr,flush=True)
        raise
