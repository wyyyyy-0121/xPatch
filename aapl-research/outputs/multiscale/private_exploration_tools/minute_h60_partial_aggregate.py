"""Recalculate pooled H60 development scores from completed full-origin folds."""
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np

from multiscale.bars import file_digest
from multiscale.spec import MODELS,OUT
from multiscale.verify_development import check_fold


def read_json(path):
    return json.loads(path.read_text(encoding='utf8'))


def assert_close(a,b,label):
    if not np.isclose(a,b,rtol=1e-12,atol=1e-18):raise ValueError('Mismatch: '+label)


def main():
    controls=OUT/'private_minute_full_controls/20261107T140007359046Z'
    neural=OUT/'private_minute_full_neural/20261107T140317864464Z'
    cp=read_json(controls/'PRIVATE_PROTOCOL.json');np_=read_json(neural/'PRIVATE_PROTOCOL.json')
    cr=read_json(controls/'PRIVATE_RESULT.json');nr=read_json(neural/'PRIVATE_RESULT.json')
    cv=read_json(controls/'PRIVATE_INDEPENDENT_CHECK.json');nv=read_json(neural/'PRIVATE_INDEPENDENT_CHECK.json')
    if (cv['status']!='private_full_minute_controls_recomputed' or
            nv['status']!='private_full_minute_neural_independently_recomputed' or
            cr['protocol_sha256']!=file_digest(controls/'PRIVATE_PROTOCOL.json') or
            nr['protocol_sha256']!=file_digest(neural/'PRIVATE_PROTOCOL.json') or
            np_['control_protocol_sha256']!=file_digest(controls/'PRIVATE_PROTOCOL.json') or
            cv['result_sha256']!=file_digest(controls/'PRIVATE_RESULT.json') or
            nv['result_sha256']!=file_digest(neural/'PRIVATE_RESULT.json') or
            nr['controls_result_sha256']!=file_digest(controls/'PRIVATE_RESULT.json') or
            nv['controls_verification_sha256']!=file_digest(controls/'PRIVATE_INDEPENDENT_CHECK.json') or
            cr['targets_sha256']!=file_digest(controls/'targets.npz') or
            cp['admission_sha256']!=np_['admission_sha256'] or cp['Q3_values_read'] or np_['Q3_values_read']):
        raise ValueError('First-fold candidate identity or verification changed')
    for key in ('lookback','validation_origins'):
        if cp[key]!=np_[key]:raise ValueError('First-fold controls and neural task differ')
    for key,value in [('max_epochs',20),('patience',3),('batch_size',128),('learning_rate',0.001),('seed',2026)]:
        if np_[key]!=value:raise ValueError('First-fold neural recipe differs: '+key)
    runs=[OUT/'private_minute_parallel_folds/20261107T142105271032Z',
          OUT/'private_minute_parallel_folds/20261107T143936100555Z']
    later=[]
    for run in runs:
        state=read_json(run/'PRIVATE_QUEUE_STATE.json')
        plan=read_json(run/'PRIVATE_PROTOCOL.json')
        if (plan['admission_sha256']!=cp['admission_sha256'] or plan['lookback']!=cp['lookback'] or
                plan['seasonal_period']!=cp['seasonal_period'] or plan['Q3_values_read'] or
                plan['task']!='minute H60 cumulative log return'):
            raise ValueError('Minute queue candidate or configuration differs')
        for number,entry in state['workers'].items():
            fold=int(number)
            if entry['status']!='completed':continue
            folder=run/f'fold_{fold:02d}'
            verification=run/f'fold_{fold:02d}_verification.json'
            if not verification.is_file():raise ValueError('Missing completed-fold verification')
            checked=check_fold(folder)
            saved=read_json(verification)
            if (saved['fold_receipt_sha256']!=file_digest(folder/'fold_receipt.json') or
                    checked['fold_receipt_sha256']!=saved['fold_receipt_sha256']):
                raise ValueError('Completed-fold receipt changed')
            receipt=read_json(folder/'fold_receipt.json')
            cfg=receipt['config']
            for key,value in [('profile','minute'),('horizon',60),('target','future_log_return'),
                              ('lookback',480),('seasonal_period',390),('max_epochs',20),
                              ('patience',3),('batch_size',128),('seed',2026)]:
                if cfg[key]!=value:raise ValueError('Incompatible completed fold: '+key)
            if cfg['ridge_alphas']!=[0.1,1.0,10.0] or cfg['learning_rate']!=0.001:
                raise ValueError('Incompatible development recipe')
            later.append((fold,folder,receipt,checked,verification))
    later.sort()
    if len({f[0] for f in later})!=len(later):raise ValueError('Duplicated completed fold')
    first_models={**cr['models'],**nr['models']}
    first_checks={**cv['models'],**nv['models']}
    if set(first_models)!=set(MODELS) or set(first_checks)!=set(MODELS):
        raise ValueError('First fold lacks required models')
    first={'fold':1,'target_dir':controls,'model_dirs':{m:(controls if m in cr['models'] else neural) for m in MODELS},
           'models':first_models,'checks':first_checks,'verification':str(controls/'PRIVATE_INDEPENDENT_CHECK.json')}
    groups=[first]
    for fold,folder,receipt,checked,verification in later:
        groups.append({'fold':fold,'target_dir':folder,'model_dirs':{m:folder for m in MODELS},
                       'models':receipt['models'],'checks':checked['models'],'verification':str(verification)})
    total_count=0;total_sq={m:0.0 for m in MODELS};fold_summaries=[];last_origin=None
    for group in groups:
        with np.load(group['target_dir']/'targets.npz',allow_pickle=False) as x:
            truth=np.asarray(x['actual'][:,-1],dtype=np.float64)
            origin=x['origins'].copy();decision=x['decision_ns'].copy()
        count=len(truth)
        if count==0 or (last_origin is not None and origin[0]<=last_origin):
            raise ValueError('Empty or nonchronological fold origins')
        last_origin=int(origin[-1]);total_count+=count
        losses={}
        for name in MODELS:
            prediction_path=group['model_dirs'][name]/f'prediction_{name}.npz'
            if file_digest(prediction_path)!=group['models'][name]['prediction_sha256']:
                raise ValueError('Prediction digest changed: '+name)
            with np.load(prediction_path,allow_pickle=False) as x:
                if not np.array_equal(origin,x['origins']) or not np.array_equal(decision,x['decision_ns']):
                    raise ValueError('Prediction origin identity changed: '+name)
                pred=np.asarray(x['prediction'][:,-1],dtype=np.float64)
            sum_sq=float(np.square(pred-truth).sum())
            mse=sum_sq/count
            recorded=(group['checks'][name]['endpoint_mse'] if group['fold']==1
                      else group['checks'][name]['metrics']['endpoint_mse'])
            assert_close(mse,recorded,f"fold {group['fold']} {name}")
            losses[name]=mse;total_sq[name]+=sum_sq
        base=losses['last_close']
        fold_summaries.append({'fold':group['fold'],'validation_origins':count,'endpoint_mse':losses,
                               'relative_improvement':{m:1-losses[m]/base if base else None for m in MODELS},
                               'verification':group['verification']})
    pooled={m:total_sq[m]/total_count for m in MODELS};baseline=pooled['last_close']
    result={'status':'partial_h60_full_origin_development_only','created_at_utc':datetime.now(timezone.utc).isoformat(),
            'completed_folds':[x['fold'] for x in fold_summaries],'expected_folds':5,
            'validation_origins':total_count,'folds':fold_summaries,
            'models':{m:{'endpoint_mse':pooled[m],'relative_improvement':1-pooled[m]/baseline if baseline else None}
                      for m in MODELS},
            'pooling':'sum of saved per-origin endpoint squared errors divided by total origins',
            'independent_test':False,'publication_rights':'not confirmed','Q3_values_read':False,
            'limits':['Partial and previously exposed development folds, not independent confirmation.',
                      'Validation paths overlap; these origins are not independent observations.',
                      'One seed and at most 20 epochs do not establish convergence.']}
    parent=OUT/'private_minute_h60_aggregate';parent.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');directory=parent/stamp;directory.mkdir(exist_ok=False)
    path=directory/'PRIVATE_PARTIAL_AGGREGATE.json'
    with path.open('x',encoding='utf8') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(json.dumps({'status':result['status'],'path':str(path),'sha256':file_digest(path),
                      'completed_folds':result['completed_folds'],'validation_origins':total_count},ensure_ascii=False))


if __name__=='__main__':main()
