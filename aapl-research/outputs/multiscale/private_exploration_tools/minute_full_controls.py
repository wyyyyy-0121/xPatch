"""Full-origin first-fold minute controls; personal development only."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from multiscale.bars import eligible_origins,file_digest,targets
from multiscale.development import batched_baseline,validate_split
from multiscale.evaluation import endpoint_improvement,path_metrics
from multiscale.features import CloseWindows,fit_scaler
from multiscale.matrix_executor import read_bars
from multiscale.ridge import fit_many
from multiscale.spec import OUT,PROFILES,ROOT
from multiscale.splits import rolling_folds
from multiscale.training import digest_indices


def save(path,value):
    with path.open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def main():
    started=time.monotonic();source=OUT/'admission/alpaca_20261107T052228573242Z';ap=source/'admission.json'
    a=json.loads(ap.read_text());terms=OUT/'source_docs/alpaca_terms.txt'
    quote='you agree to use the Services and Content solely for your own personal and non-commercial purposes'
    if (quote not in ' '.join(terms.read_text().split()) or a.get('Q3_values_read') is not False or
            a.get('experiment_scored') is not False):raise ValueError('Unexpected source or personal-use term')
    profile=next(p for p in PROFILES if p.name=='minute');bars=read_bars(source,a,'minute');op=source/'origins_minute_h60.npy'
    if file_digest(op)!=a['files'][op.name]:raise ValueError('Origin identity changed')
    origins=np.load(op,allow_pickle=False)
    if not np.array_equal(origins,eligible_origins(bars,max(profile.lookbacks),60,same_session=True)):
        raise ValueError('Saved origins do not match the task')
    folds=rolling_folds(bars,origins,60)
    if len(folds)!=5:raise ValueError('Expected five folds')
    fold=folds[0];train,fit,stop,val=validate_split(bars,fold,60)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');parent=OUT/'private_minute_full_controls';parent.mkdir(parents=True,exist_ok=True)
    directory=parent/stamp;directory.mkdir(exist_ok=False);exposure=OUT/'POST_CUTOFF_EXPOSURE.md';before=exposure.read_bytes()
    (directory/'EXPOSURE_BEFORE.md').write_bytes(before)
    source_files=('multiscale/bars.py','multiscale/baselines.py','multiscale/features.py','multiscale/ridge.py',
                  'multiscale/evaluation.py','multiscale/splits.py','multiscale/development.py')
    plan={'status':'private_full_origin_controls_only_not_publication_admitted','created_at_utc':datetime.now(timezone.utc).isoformat(),
          'user_request':'我要开始做实验','scope':'personal non-commercial local development','personal_use_quote':quote,
          'terms_sha256':file_digest(terms),'account_specific_and_publication_rights':'not confirmed; keep results local',
          'admission_sha256':file_digest(ap),'bar_sha256':a['files']['bars_minute.npz'],'origins_sha256':a['files'][op.name],
          'task':'minute H60 cumulative log return','fold':1,'models':['last_close','drift','seasonal','ridge'],
          'lookback':480,'seasonal_period':390,'ridge_alphas':[0.1,1.0,10.0],
          'ridge_selection':'inner stopping endpoint MSE only; no refit on stop',
          'train_origins':len(train),'fit_origins':len(fit),'stop_origins':len(stop),'validation_origins':len(val),
          'split_sha256':{name:digest_indices(v) for name,v in [('train',train),('fit',fit),('stop',stop),('validation',val)]},
          'source_sha256':{p:file_digest(ROOT/p) for p in source_files},'script_sha256':file_digest(Path(__file__)),
          'exposure_before_sha256':hashlib.sha256(before).hexdigest(),'device':'cpu','cpu_threads':2,'maximum_wall_seconds':600,
          'data_cutoff':'2026-06-30','Q3_values_read':False,'independent_test':False,
          'limitations':['One previously exposed development fold and four controls only; not the eight-model matrix.',
                         'The minute sampled pilot was seen before this full-origin run.',
                         'The source rights and publication terms remain unconfirmed.']}
    plan_path=directory/'PRIVATE_PROTOCOL.json';save(plan_path,plan)
    with exposure.open('a',encoding='utf8') as f:
        f.write('\n## 个人内部分钟完整起点基线：事后登记\n\n'
                f'{stamp}：此前已看到分钟 H60 抽样首折及其他尺度结果后，预先固定原始首折全部合法起点上的'
                f'四项基线/Ridge 比较，拟合 {len(fit)}、早停 {len(stop)}、验证 {len(val)} 个起点。'
                '它不含四个神经模型，是历史开发而非独立确认；发表权仍待核实，2026Q3 保持封存。'
                f'设置见 {plan_path}。\n')
    print(json.dumps({'status':'private_minute_full_controls_started','directory':str(directory),'fit_origins':len(fit),
                      'validation_origins':len(val),'cpu_threads':2,'maximum_seconds':600},ensure_ascii=False),flush=True)
    w=CloseWindows(bars,480,a['split_actions']);predpaths={}
    for name in ('last_close','drift','seasonal'):
        pred=batched_baseline(name,w,val,60,'future_log_return',390)
        path=directory/f'prediction_{name}.npz';np.savez_compressed(path,prediction=pred,origins=val,decision_ns=bars.end_ns[val])
        predpaths[name]=path;del pred
    print(json.dumps({'stage':'three_controls_complete','elapsed_seconds':time.monotonic()-started}),flush=True)
    scaler=fit_scaler(w,fit,60,'future_log_return');models=fit_many(w,scaler,fit,60,'future_log_return',(0.1,1.,10.))
    truth_stop=targets(bars,stop,60,'future_log_return');losses={}
    for alpha,model in models.items():
        stop_pred=model.predict(w,scaler,stop);losses[alpha]=float(np.square(stop_pred[:,-1]-truth_stop[:,-1]).mean())
    chosen=min(losses,key=lambda alpha:(losses[alpha],alpha));ridge=models[chosen];pred=ridge.predict(w,scaler,val)
    path=directory/'prediction_ridge.npz';np.savez_compressed(path,prediction=pred,origins=val,decision_ns=bars.end_ns[val]);predpaths['ridge']=path
    np.savez_compressed(directory/'ridge_coefficients.npz',coefficients=ridge.coefficients)
    save(directory/'ridge_scaler.json',asdict(scaler));del pred
    actual=targets(bars,val,60,'future_log_return');anchor=bars.close[val]
    np.savez_compressed(directory/'targets.npz',actual=actual,origins=val,anchor=anchor,
                        decision_ns=bars.end_ns[val],target_completion_ns=bars.end_ns[val+60])
    with np.load(predpaths['last_close'],allow_pickle=False) as z:base=z['prediction'].copy()
    metrics={}
    for name,path in predpaths.items():
        with np.load(path,allow_pickle=False) as z:prediction=z['prediction']
        against=endpoint_improvement(actual,prediction,base);against.pop('paired_loss_difference')
        metrics[name]={'metrics':path_metrics(actual,prediction,'future_log_return',anchor=anchor),
                       'against_zero_return':against,'prediction_sha256':file_digest(path)}
    if any(file_digest(ROOT/p)!=h for p,h in plan['source_sha256'].items()):raise ValueError('Code changed while running')
    result={'status':'private_full_origin_controls_complete','protocol_sha256':file_digest(plan_path),'validation_origins':len(val),
            'models':metrics,'ridge_alpha':chosen,'ridge_inner_stop_mse_by_alpha':losses,
            'targets_sha256':file_digest(directory/'targets.npz'),'ridge_coefficients_sha256':file_digest(directory/'ridge_coefficients.npz'),
            'ridge_scaler_sha256':file_digest(directory/'ridge_scaler.json'),'elapsed_seconds':time.monotonic()-started,
            'data_cutoff':'2026-06-30','Q3_values_read':False,'independent_test':False,
            'scope':'private historical development; four controls only; publication rights not confirmed'}
    save(directory/'PRIVATE_RESULT.json',result)
    with exposure.open('a',encoding='utf8') as f:
        f.write(f'\n{stamp}：分钟完整起点四项对照首折已完成，结果保存在 {directory}/PRIVATE_RESULT.json；'
                '它不是完整神经模型矩阵或独立确认。\n')
    print(json.dumps({'status':result['status'],'directory':str(directory),'validation_origins':len(val),
                      'elapsed_seconds':result['elapsed_seconds'],'relative_improvement':
                      {k:v['against_zero_return']['relative_improvement'] for k,v in metrics.items()}},ensure_ascii=False),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(json.dumps({'status':'private_minute_controls_failed','error':f'{type(exc).__name__}: {exc}'},ensure_ascii=False),file=sys.stderr,flush=True)
        raise
