"""Complete the four neural models for the private full-origin minute fold."""
from datetime import datetime,timezone
import hashlib,json,sys,time
from pathlib import Path

import numpy as np
import torch

from multiscale.bars import eligible_origins,file_digest
from multiscale.development import atomic_json,validate_split
from multiscale.evaluation import endpoint_improvement,path_metrics
from multiscale.features import CloseWindows
from multiscale.matrix_executor import read_bars
from multiscale.spec import OUT,PROFILES,ROOT
from multiscale.splits import rolling_folds
from multiscale.training import TrainConfig,digest_indices,identity,predict_best,run_fit


def save_new(path,value):
    with path.open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def main():
    started=time.monotonic();controls=OUT/'private_minute_full_controls/20261107T140007359046Z'
    control_plan=json.loads((controls/'PRIVATE_PROTOCOL.json').read_text());control_result=json.loads((controls/'PRIVATE_RESULT.json').read_text())
    verification=json.loads((controls/'PRIVATE_INDEPENDENT_CHECK.json').read_text())
    if (control_result['status']!='private_full_origin_controls_complete' or
            verification['status']!='private_full_minute_controls_recomputed' or
            verification['result_sha256']!=file_digest(controls/'PRIVATE_RESULT.json')):
        raise ValueError('The full-origin control identities are not verified')
    source=OUT/'admission/alpaca_20261107T052228573242Z';ap=source/'admission.json';a=json.loads(ap.read_text())
    if file_digest(ap)!=control_plan['admission_sha256'] or a.get('Q3_values_read') is not False:
        raise ValueError('The candidate source identity or cutoff changed')
    profile=next(p for p in PROFILES if p.name=='minute');bars=read_bars(source,a,'minute');op=source/'origins_minute_h60.npy'
    if file_digest(op)!=a['files'][op.name]:raise ValueError('Minute origin identity changed')
    origins=np.load(op,allow_pickle=False)
    if not np.array_equal(origins,eligible_origins(bars,max(profile.lookbacks),60,same_session=True)):
        raise ValueError('Minute origins differ from the planned task')
    folds=rolling_folds(bars,origins,60)
    if len(folds)!=5:raise ValueError('Expected five chronological folds')
    fold=folds[0];train,fit,stop,val=validate_split(bars,fold,60)
    for n,x in [('train',train),('fit',fit),('stop',stop),('validation',val)]:
        if digest_indices(x)!=control_plan['split_sha256'][n]:raise ValueError('Neural and control origins differ')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');parent=OUT/'private_minute_full_neural';parent.mkdir(parents=True,exist_ok=True)
    directory=parent/stamp;directory.mkdir(exist_ok=False);exposure=OUT/'POST_CUTOFF_EXPOSURE.md';before=exposure.read_bytes();(directory/'EXPOSURE_BEFORE.md').write_bytes(before)
    sources=('multiscale/training.py','multiscale/models.py','multiscale/features.py','multiscale/bars.py',
             'multiscale/mechanism.py','multiscale/evaluation.py','aapl/models.py','aapl/vendor/xpatch_network.py')
    plan={'status':'private_full_origin_neural_only_not_publication_admitted','created_at_utc':datetime.now(timezone.utc).isoformat(),
          'user_request':'我要开始做实验','scope':'personal non-commercial local development',
          'control_protocol_sha256':file_digest(controls/'PRIVATE_PROTOCOL.json'),'control_result_sha256':file_digest(controls/'PRIVATE_RESULT.json'),
          'control_verification_sha256':file_digest(controls/'PRIVATE_INDEPENDENT_CHECK.json'),'admission_sha256':file_digest(ap),
          'task':'minute H60 cumulative log return','fold':1,'models':['mlp','gru','lstm','xpatch'],'lookback':480,
          'max_epochs':20,'patience':3,'batch_size':128,'learning_rate':0.001,'seed':2026,
          'train_origins':len(train),'fit_origins':len(fit),'stop_origins':len(stop),'validation_origins':len(val),
          'split_sha256':{n:digest_indices(x) for n,x in [('train',train),('fit',fit),('stop',stop),('validation',val)]},
          'source_sha256':{p:file_digest(ROOT/p) for p in sources},'script_sha256':file_digest(Path(__file__)),
          'exposure_before_sha256':hashlib.sha256(before).hexdigest(),'device':'cpu','cpu_threads':2,'maximum_wall_seconds':900,
          'data_cutoff':'2026-06-30','Q3_values_read':False,'independent_test':False,
          'publication_rights':'not confirmed; keep results local',
          'limitations':['One exposed development fold only, not a complete minute matrix or independent test.',
                         'Control scores and a sampled neural pilot were seen before this full-origin run.',
                         'The account-specific source and publication rights remain unconfirmed.',
                         'Periodic checkpoints are saved; timeout does not automatically resume or reset the budget.']}
    pp=directory/'PRIVATE_PROTOCOL.json';save_new(pp,plan);protocol_hash=file_digest(pp)
    with exposure.open('a',encoding='utf8') as f:
        f.write('\n## 个人内部分钟完整起点神经模型：事后登记\n\n'
                f'{stamp}：已看到同一折抽样结果和完整起点四项对照后，固定原计划的四种神经模型和训练配置，'
                f'在相同 {len(fit)} 个拟合、{len(stop)} 个早停及 {len(val)} 个验证起点上运行。'
                '该折仍是历史开发，不能作为独立确认；论文发表权待核实，2026Q3 仍封存。'
                f'设置见 {pp}。\n')
    state={'status':'running','protocol_sha256':protocol_hash,'completed_models':[],'current_model':None,'models':{}}
    state_path=directory/'PRIVATE_STATE.json';atomic_json(state_path,state)
    print(json.dumps({'status':'private_minute_full_neural_started','directory':str(directory),'fit_origins':len(fit),
                      'validation_origins':len(val),'cpu_threads':2,'maximum_seconds':900},ensure_ascii=False),flush=True)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);w=CloseWindows(bars,480,a['split_actions']);data_identity='private-minute-full:'+protocol_hash
    for name in plan['models']:
        state['current_model']=name;state['elapsed_seconds']=time.monotonic()-started;atomic_json(state_path,state)
        print(json.dumps({'stage':'training','model':name,'elapsed_seconds':state['elapsed_seconds']}),flush=True)
        cfg=TrainConfig(name,480,60,'future_log_return',20,3,128,0.001,2026)
        training_id=identity(cfg,data_identity,fit,stop,'cpu');model_dir=directory/name
        one_start=time.monotonic();summary=run_fit(w,fit,stop,cfg,model_dir,data_identity,device='cpu')
        if summary.get('status')!='completed' or summary.get('identity')!=training_id:raise ValueError('Neural training identity differs')
        prediction=predict_best(model_dir/'best.pt',w,val,training_id,device='cpu')
        if prediction.shape!=(len(val),60) or not np.isfinite(prediction).all():raise ValueError('Invalid neural predictions')
        pred_path=directory/f'prediction_{name}.npz';np.savez_compressed(pred_path,prediction=prediction,origins=val,decision_ns=bars.end_ns[val])
        restore=predict_best(model_dir/'best.pt',w,val[:32],training_id,device='cpu')
        if not np.allclose(restore,prediction[:32],rtol=1e-7,atol=1e-9):raise ValueError('Restored predictions differ')
        record={'training_identity':training_id,'epochs':summary['epochs'],'best_epoch':summary['best_epoch'],
                'steps':summary['steps'],'stopped_early':summary['stopped_early'],'model_elapsed_seconds':time.monotonic()-one_start,
                'best_checkpoint_sha256':file_digest(model_dir/'best.pt'),'latest_checkpoint_sha256':file_digest(model_dir/'latest.pt'),
                'training_summary_sha256':file_digest(model_dir/'training_summary.json'),'prediction_sha256':file_digest(pred_path),
                'restore_first_32_matches':True}
        state['models'][name]=record;state['completed_models'].append(name);state['elapsed_seconds']=time.monotonic()-started;atomic_json(state_path,state)
        print(json.dumps({'stage':'model_complete','model':name,'epochs':record['epochs'],'best_epoch':record['best_epoch'],
                          'model_elapsed_seconds':round(record['model_elapsed_seconds'],1),'elapsed_seconds':round(state['elapsed_seconds'],1)}),flush=True)
    # All four models and inner stopping decisions are fixed before reading outer targets.
    with np.load(controls/'targets.npz',allow_pickle=False) as z:
        actual=z['actual'].copy();anchor=z['anchor'].copy()
        if not np.array_equal(z['origins'],val) or not np.array_equal(z['decision_ns'],bars.end_ns[val]):raise ValueError('Control targets differ')
    with np.load(controls/'prediction_last_close.npz',allow_pickle=False) as z:baseline=z['prediction'].copy()
    metrics={}
    for name in plan['models']:
        with np.load(directory/f'prediction_{name}.npz',allow_pickle=False) as z:prediction=z['prediction']
        against=endpoint_improvement(actual,prediction,baseline);against.pop('paired_loss_difference')
        metrics[name]={'metrics':path_metrics(actual,prediction,'future_log_return',anchor=anchor),'against_zero_return':against,
                       **state['models'][name]}
    if any(file_digest(ROOT/p)!=h for p,h in plan['source_sha256'].items()):raise ValueError('Code changed during training')
    result={'status':'private_full_origin_neural_complete','protocol_sha256':protocol_hash,'validation_origins':len(val),
            'models':metrics,'controls_result_sha256':file_digest(controls/'PRIVATE_RESULT.json'),
            'elapsed_seconds':time.monotonic()-started,'data_cutoff':'2026-06-30','Q3_values_read':False,
            'independent_test':False,'scope':'private historical development; one fold; publication rights not confirmed'}
    rp=directory/'PRIVATE_RESULT.json';save_new(rp,result);state['status']='completed';state['current_model']=None;state['result_sha256']=file_digest(rp);state['elapsed_seconds']=result['elapsed_seconds'];atomic_json(state_path,state)
    with exposure.open('a',encoding='utf8') as f:
        f.write(f'\n{stamp}：分钟完整起点四个神经模型首折已完成，结果见 {rp}；'
                '它不是完整五折或独立确认。\n')
    print(json.dumps({'status':result['status'],'directory':str(directory),'validation_origins':len(val),
                      'elapsed_seconds':result['elapsed_seconds'],'relative_improvement':
                      {k:v['against_zero_return']['relative_improvement'] for k,v in metrics.items()}},ensure_ascii=False),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(json.dumps({'status':'private_minute_full_neural_failed','error':f'{type(exc).__name__}: {exc}'},ensure_ascii=False),file=sys.stderr,flush=True);raise
