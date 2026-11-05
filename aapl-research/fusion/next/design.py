"""Deterministic candidate domains; only F2 may admit counts and freeze the search."""
from itertools import product
import math
import numpy as np

from .common import OUT, ROOT, PLAN, digest, identity, read_json, write_json, utc_now,source_identity
from .splits import VALIDATION


MAX_COUNTS={'lightgbm':24,'xpatch':16,'patchtst':16,'simple':12,'fusion':24,'transfer_pairs':8,'lora':8}


def previous_configs():
    path=OUT/'previous_selected_hyperparameters.json'
    if not path.exists():
        original=read_json(ROOT/'outputs/fusion/protocol.json')['expert_selections']
        value={name:row['trial'] for name,row in original.items()}
        write_json(path,value)
    return read_json(path)


def candidates(family,count,seed=2026):
    rng=np.random.default_rng(seed)
    shared={'family':family,'information':'ohlcv','training_scope':'multistock','weight_decay':.01,
            'dropout':0.,'lr':3e-4,'precision':'fp32','loss':'raw','lookback':240}
    if family=='lightgbm':
        domain={'lookback':[48,96,240],'num_leaves':[7,15,31],'max_depth':[3,5,-1],
                'min_child_samples':[100,500],'learning_rate':[.01,.03,.05],
                'reg_lambda':[1.,10.,100.],'colsample_bytree':[.7,1.]}
        shared.update(trees=1000,rows_per_day=16,cpu_threads=8)
    elif family in ('xpatch','patchtst'):
        domain={'lookback':[48,96,240],'patch':[8,16],'stride':[4,8],'lr':[1e-4,3e-4,1e-3],
                'weight_decay':[.001,.01,.1],'loss':['raw','scaled','step_mse']}
        if family=='xpatch':domain['alpha']=[.01,.1]
        else:domain.update(d_model=[32,64],dropout=[0.,.1])
    elif family in ('fusion','lora'):
        shared['family']='fusion';shared.update(foundation=True,market=True,projection=64,hidden=32,
            market_encoder='tcn',foundation_mode='lora' if family=='lora' else 'frozen',rank=8,
            residual_cap=.25,adaptation_fraction=.2,adapt_lr_ratio=.3,gate_smooth_penalty=1e-4)
        domain={'lookback':[48,96,240],'projection':[64,128],'hidden':[32,64],
                'market_encoder':['tcn','transformer'],'lr':[1e-4,3e-4,1e-3],
                'dropout':[0.,.1],'weight_decay':[.001,.01,.1],'residual_cap':[0.,.25,.5],'loss':['raw','scaled']}
        if family=='lora':domain.update(rank=[8,16],backbone_lr=[1e-5,3e-5])
    else:raise ValueError(family)
    configs=[]
    if family in ('lightgbm','xpatch','patchtst'):
        previous=previous_configs()[family]
        previous={k:v for k,v in previous.items() if k in domain}
        if family=='xpatch':previous['loss']='step_mse'
        if family=='patchtst':previous['loss']='raw'
        configs.append({**shared,**{k:values[0] for k,values in domain.items()},**previous})
    while len(configs)<count:
        candidate={**shared,**{key:values[int(rng.integers(len(values)))] for key,values in domain.items()}}
        if candidate not in configs:configs.append(candidate)
    return [{**config,'candidate_id':family+'_'+identity(config)[:12]} for config in configs[:count]]


def simple_candidates():
    return ([{'family':'nonnegative_ridge','alpha':a} for a in (.01,.1,1.,10.)]
            +[{'family':'convex','alpha':a} for a in (0.,.01,.1,1.)]
            +[{'family':'shrink_best','factor':a} for a in (0.,.25,.5,1.)])


def halving_units(count):
    return .1*count+.2*math.ceil(count/3)+.7*min(2,math.ceil(count/3))


def draft():
    return {'counts':MAX_COUNTS,'validation':list(VALIDATION),'seeds':[2026,2027,2028],
            'groups':{f:candidates(f,MAX_COUNTS[f]) for f in ('lightgbm','xpatch','patchtst','fusion','lora')},
            'simple':simple_candidates(),'transfer_pairs':8,
            'halving':{'prefix_fractions':[.1,.3,1.],'first_survivors':'ceil(n/3)','final_survivors':2,
                       'resume_optimizer_rng_sampler':True},
            'mechanism_pairs':[['gate','plain_mlp'],['foundation','no_foundation'],
                 ['pretrained_frozen','random_frozen'],['frozen','lora'],['aapl','multistock_then_adapt'],
                 ['market','no_market'],['close','ohlcv']],
            'new_scope_validation_scores_read':False,'frozen':False,'plan_sha256':digest(PLAN)}


def freeze(specification,projection):
    # This entry intentionally cannot invent performance data or waive a failing gate.
    admission=read_json(OUT/'performance_admission.json')
    if not admission.get('numerical_and_resource_checks_passed'):raise ValueError('F2 actual GPU admission has not passed')
    if projection['margin_multiplier']!=1.5 or projection['total_gpu_card_hours_with_margin']>24:
        raise ValueError('Complete 50% margin projection must fit 24 card-hours')
    if not projection.get('all_required_work_included'):raise ValueError('OOF/seeds/mechanisms/retraining/deployment must all be budgeted')
    path=OUT/'search_manifest.json'
    if path.exists():raise ValueError('Search is already frozen; do not silently replace it')
    value={**specification,'source_identity':source_identity(),'frozen':True,'frozen_at_utc':utc_now(),'projection':projection,
           'performance_admission_sha256':digest(OUT/'performance_admission.json')}
    value['identity']=identity(value);write_json(path,value)
    return value


if __name__=='__main__':
    value=draft();write_json(OUT/'search_design_draft.json',value)
    print('Draft only; no search is authorized until F2 and its full time projection pass.')
