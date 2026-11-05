"""F3-F5 execution: fixed forward folds, continuous halving, OOF and paired mechanisms."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .common import OUT, PLAN, digest, identity, read_json, write_json, utc_now, source_identity,status
from .hardware import runtime
from .panel import Panel
from .splits import fold,VALIDATION
from .models import NeuralFusion
from .expert_models import NativeExpert
from .experts import NAMES,fit_cpu,predict_cpu,scope_key,write_oof
from .trainer import Trainer,ExpertStore
from .evaluation import point_scores,seed_system,minimum_detectable_improvement
from .accounting import Usage,totals
from .design import halving_survivors
from . import simple


def admitted():
    spec=read_json(OUT/'search_manifest.json')
    if not spec.get('frozen') or spec['plan_sha256']!=digest(PLAN):raise ValueError('Immutable search and user plan do not match')
    if spec.get('source_identity')!=source_identity():raise ValueError('Execution code differs from the pre-score frozen source')
    if not read_json(OUT/'data_audit.json')['passed']:raise ValueError('F1 audit not complete')
    if not read_json(OUT/'performance_admission.json').get('passed'):raise ValueError('Full F2 and budget projection not admitted')
    return spec


def model_for(panel,config,device):
    channels=4 if config.get('information')=='close' else 12
    market=channels*len(panel.background) if config.get('market_information',True) else 0
    if config['family']=='fusion':return NeuralFusion(config,channels,market,len(panel.stocks),device)
    return NativeExpert(config,channels+market,len(panel.stocks))


def score_rows(panel,quarter):
    f=fold(panel.dates,quarter);rows=panel.origins['AAPL']
    return f,rows[(panel.dates[rows]>=f['score_start'])&(panel.dates[rows]<=f['score_end'])]


def save_prediction(directory,rows,prediction,panel,details):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    path=directory/'prediction.npz';np.savez_compressed(path,rows=rows,prediction=prediction)
    report={'scores':point_scores(panel.targets['AAPL'][rows],prediction),'details':details,
            'prediction_sha256':digest(path),'completed_at_utc':utc_now()}
    write_json(directory/'result.json',report);return report


def read_prediction(directory,rows,details):
    directory=Path(directory);report=read_json(directory/'result.json');path=directory/'prediction.npz'
    if report['prediction_sha256']!=digest(path) or report['details']!=details:raise ValueError('Saved prediction identity or experiment recipe differs')
    with np.load(path) as saved:
        if not np.array_equal(rows,saved['rows']):raise ValueError('Saved prediction changed its common origins')
        prediction=saved['prediction'].copy()
    return report,prediction


def neural_trial(panel,config,quarter,seed,prefix,namespace,store=None,category='fusion_search'):
    f,rows=score_rows(panel,quarter);updates=int(math.ceil(prefix*config['total_updates']))
    trial=namespace+'/'+config['candidate_id']+'/'+quarter
    directory=OUT/'trials'/trial/f'seed_{seed}'/f'updates_{updates}'
    details={'config':config,'fold':f,'seed':seed,'updates':updates,'source':source_identity()}
    if (directory/'result.json').exists():return read_prediction(directory,rows,details)
    device=runtime(seed)
    if device!='cuda':raise RuntimeError('Formal neural training is assigned to the cloud GPU')
    with Usage(category,trial+f'/seed_{seed}/updates_{updates}') as usage:
        cfg={**config,'seed':seed};model=model_for(panel,cfg,device)
        trainer=Trainer(panel,model,cfg,f['fit_end'],trial,seed,experts=store,device=device)
        if trainer.path.exists():trainer.resume()
        trainer.run_until(updates,usage)
        experts=store.predict(np.zeros(len(rows),dtype=np.int64),rows) if store is not None else None
        prediction=trainer.predict(np.zeros(len(rows),dtype=np.int64),rows,experts,usage)
        result=save_prediction(directory,rows,prediction,panel,details)
        result['parameters']={'total':sum(p.numel() for p in model.parameters()),
                              'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad)}
        write_json(directory/'result.json',result)
        del trainer,model
    gc.collect();torch.cuda.empty_cache();return result,prediction


def cpu_trial(panel,config,quarter,namespace):
    f,rows=score_rows(panel,quarter);directory=OUT/'trials'/namespace/config['candidate_id']/quarter
    details={'config':config,'fold':f,'seed':2026,'source':source_identity()}
    if (directory/'result.json').exists():return read_prediction(directory,rows,details)
    with Usage('experts_and_cache',namespace+'/'+config['candidate_id']+'/'+quarter,gpu=False):
        predict,scalers,_=fit_cpu(panel,config,f['fit_end'],directory/'model')
        prediction=predict_cpu(predict,panel,np.zeros(len(rows),dtype=np.int64),rows,config,scalers)
        result=save_prediction(directory,rows,prediction,panel,details)
    return result,prediction


def pooled_score(results):
    return sum(r['scores']['mse']*r['scores']['n'] for r in results)/sum(r['scores']['n'] for r in results)


def search(panel,configs,namespace,store=None,category='fusion_search'):
    path=OUT/'selections'/(namespace+'.json')
    if path.exists():return read_json(path)
    neural=configs[0]['family'] in ('fusion','xpatch','patchtst')
    current=list(configs);rungs=[]
    for prefix in ((.1,.3,1.) if neural else (1.,)):
        rankings=[]
        for cfg in current:
            status('F3_F5_search',namespace=namespace,candidate_id=cfg['candidate_id'],prefix=prefix)
            results=[]
            for quarter in VALIDATION:
                result,_=neural_trial(panel,cfg,quarter,2026,prefix,namespace,store,category) if neural else cpu_trial(panel,cfg,quarter,namespace)
                results.append(result)
            rankings.append({'config':cfg,'mse':pooled_score(results),'fold_scores':{q:r['scores'] for q,r in zip(VALIDATION,results)}})
        rankings.sort(key=lambda r:(r['mse'],r['config']['candidate_id']))
        rungs.append({'prefix':prefix,'rankings':rankings})
        write_json(OUT/'search_progress'/(namespace+'.json'),{'rungs':rungs,'source':source_identity()})
        count=halving_survivors(len(configs)) if prefix==.1 else min(2,len(current))
        current=[r['config'] for r in rankings[:count]]
    value={'namespace':namespace,'selected':rungs[-1]['rankings'][0]['config'],
           'top_two':[r['config'] for r in rungs[-1]['rankings'][:2]],'rungs':rungs,
           'primary_metric':'pooled raw return MSE over fixed common AAPL origins',
           'source_identity':source_identity(),'completed_at_utc':utc_now()}
    write_json(path,value);return value


def expert_configurations(spec,information='ohlcv',training_scope='multistock',market_information=True):
    selected={name:read_json(OUT/'selections'/('single_'+name+'.json'))['selected'] for name in NAMES}
    return {name:{**cfg,'information':information,'training_scope':training_scope,'market_information':market_information}
            for name,cfg in selected.items()}


def build_oof(panel,configs,category='experts_and_cache'):
    example=next(iter(configs.values()));info=example['information'];scope=example['training_scope'];market=example.get('market_information',True)
    key=scope_key(info,scope,market);names=panel.stocks if scope=='multistock' else ['AAPL']
    for period in pd.period_range('2023Q1','2026Q1',freq='Q'):
        f=fold(panel.dates,str(period));receipt=OUT/'oof'/key/(str(period)+'.json')
        if receipt.exists():continue
        predictions={}
        for name in names:
            rows=panel.origins[name];rows=rows[(panel.dates[rows]>=f['score_start'])&(panel.dates[rows]<=f['score_end'])]
            predictions[name]=(rows,np.empty((len(rows),4),dtype=np.float32))
        def cpu_column(expert):
            cfg=configs[expert];column=NAMES.index(expert);namespace='oof_models/'+key+'/'+expert+'/'+str(period)
            directory=OUT/namespace;directory.mkdir(parents=True,exist_ok=True)
            values={}
            with Usage(category,namespace,gpu=False):
                predict,scalers,_=fit_cpu(panel,cfg,f['fit_end'],directory)
                for name,(rows,_) in predictions.items():
                    sid=panel.stocks.index(name)
                    values[name]=predict_cpu(predict,panel,np.full(len(rows),sid),rows,cfg,scalers)
            return column,values
        # One CPU worker leaves room for the GPU feed process and its thread pools.
        with ThreadPoolExecutor(max_workers=1) as pool:
            futures=[pool.submit(cpu_column,expert) for expert in ('ridge','lightgbm')]
            for expert in ('xpatch','patchtst'):
                column=NAMES.index(expert)
                cfg=configs[expert];namespace='oof_models/'+key+'/'+expert+'/'+str(period)
                directory=OUT/namespace;directory.mkdir(parents=True,exist_ok=True)
                status('F3_F5_OOF',scope=key,quarter=str(period),expert=expert)
                device=runtime(2026)
                if device!='cuda':raise RuntimeError('OOF neural experts require the specified cloud GPU')
                with Usage(category,namespace) as usage:
                    model=model_for(panel,cfg,device)
                    trainer=Trainer(panel,model,cfg,f['fit_end'],namespace,device=device)
                    if trainer.path.exists():trainer.resume()
                    trainer.run_until(cfg['total_updates'],usage)
                    for name,(rows,values) in predictions.items():
                        sid=panel.stocks.index(name);values[:,column]=trainer.predict(np.full(len(rows),sid),rows,usage=usage)
                    del trainer,model
                gc.collect();torch.cuda.empty_cache()
            for future in futures:
                column,values=future.result()
                for name,result in values.items():predictions[name][1][:,column]=result
        write_oof(panel,info,f,predictions,configs,scope,market)
    return ExpertStore(panel,info,scope,market)


def simple_controls(panel,spec,store):
    report_path=OUT/'selections/simple.json'
    if report_path.exists():return read_json(report_path)
    rankings=[];sid=panel.stocks.index('AAPL')
    for index,cfg in enumerate(spec['simple']):
        results=[]
        for quarter in VALIDATION:
            f,rows=score_rows(panel,quarter)
            train=panel.origins['AAPL'];train=train[(panel.dates[train]<=f['fit_end'])&store.available[sid][train]]
            x=store.predict(np.full(len(train),sid),train);target=panel.targets['AAPL'][train]
            # Equal day weights also when early-close sessions have fewer eligible origins.
            _,inverse,counts=np.unique(panel.dates[train],return_inverse=True,return_counts=True)
            fitted=simple.fit(x,target,cfg,weights=1/counts[inverse])
            pred=simple.predict(fitted,store.predict(np.full(len(rows),sid),rows))
            directory=OUT/'trials/simple'/str(index)/quarter
            results.append(save_prediction(directory,rows,pred,panel,{'config':cfg,'fold':f,'fitted':fitted}))
        rankings.append({'config':cfg,'index':index,'mse':pooled_score(results),'fold_scores':[r['scores'] for r in results]})
    rankings.sort(key=lambda v:(v['mse'],v['index']))
    report={'rankings':rankings,'selected':rankings[0],'source_identity':source_identity()};write_json(report_path,report)
    return report


def verify_seeds(panel,configs,namespace,store,category='adaptation_and_mechanisms'):
    reports=[]
    for cfg in configs:
        targets=[];seed_predictions=[[] for _ in range(3)];quarters=[];origin_rows=[]
        for quarter in VALIDATION:
            _,rows=score_rows(panel,quarter);targets.append(panel.targets['AAPL'][rows]);quarters.extend([quarter]*len(rows));origin_rows.append(rows)
            for i,seed in enumerate((2026,2027,2028)):
                _,pred=neural_trial(panel,cfg,quarter,seed,1.,namespace,store,category);seed_predictions[i].append(pred)
        target=np.concatenate(targets);predictions=np.stack([np.concatenate(parts) for parts in seed_predictions])
        report,ensemble=seed_system(target,predictions)
        report.update(config=cfg,prediction_namespace=namespace,
                      quarter_scores={q:point_scores(target[np.asarray(quarters)==q],ensemble[np.asarray(quarters)==q]) for q in VALIDATION})
        path=OUT/'seed_verification'/namespace/cfg['candidate_id'];path.mkdir(parents=True,exist_ok=True)
        rows=np.concatenate(origin_rows)
        np.savez_compressed(path/'predictions.npz',rows=rows,timestamps_ns=panel.stamps[rows].as_unit('ns').asi8,
                            target=target,predictions=predictions,ensemble=ensemble,quarters=np.asarray(quarters))
        report['prediction_sha256']=digest(path/'predictions.npz');write_json(path/'report.json',report);reports.append(report)
    reports.sort(key=lambda r:(r['system']['mse'],r['config']['candidate_id']));return reports


def mechanism_configs(top,variant):
    result=[]
    for cfg in top:
        copy={**cfg}
        if variant=='plain_mlp':copy['fusion_type']='plain_mlp'
        elif variant=='no_foundation':copy.update(foundation=False,representation_cache=False)
        elif variant=='random_frozen':copy.update(foundation_mode='random_frozen',representation_cache=False)
        elif variant=='no_market':copy.update(market=False,market_information=False)
        elif variant=='close':copy['information']='close'
        elif variant=='aapl':copy.update(training_scope='aapl',adaptation_fraction=0.)
        elif variant!='full':raise ValueError(variant)
        copy['candidate_id']=variant+'_'+identity({k:v for k,v in copy.items() if k!='candidate_id'})[:12];result.append(copy)
    return result


def execute(stage):
    spec=admitted();panel=Panel.load()
    if stage=='F3':
        def cpu_search():
            return [search(panel,spec['groups'][family],'single_'+family,category='experts_and_cache') for family in ('ridge','lightgbm')]
        with ThreadPoolExecutor(max_workers=1) as pool:
            cpu=pool.submit(cpu_search)
            for family in ('xpatch','patchtst'):search(panel,spec['groups'][family],'single_'+family,category='experts_and_cache')
            cpu.result()
        store=build_oof(panel,expert_configurations(spec));simple_controls(panel,spec,store)
        write_json(OUT/'F3_complete.json',{'source_identity':source_identity(),'at_utc':utc_now(),'main_OOF':store.key})
        return
    if not (OUT/'F3_complete.json').exists():raise ValueError('F3 controls must complete first')
    store=ExpertStore(panel)
    if stage=='F4':
        search(panel,spec['groups']['fusion'],'fusion',store)
        # Pairwise selection uses only the multistock member; both members advance together.
        pair_configs=spec['transfer_configs'];paired_search(panel,pair_configs,store,spec)
        write_json(OUT/'F4_complete.json',{'source_identity':source_identity(),'at_utc':utc_now()});return
    if stage=='F5':
        if not (OUT/'F4_complete.json').exists():raise ValueError('F4 core fusion and paired transfer runs must complete first')
        top=read_json(OUT/'selections/fusion.json')['top_two']
        # LoRA starts from a reproducible fixed recipe; every promoted configuration is retrained.
        lora=[]
        for index,adapter in enumerate(spec['groups']['lora']):
            base=top[index%len(top)];cfg={**base,'foundation_mode':'lora','rank':adapter['rank'],
                'backbone_lr':adapter['backbone_lr'],'representation_cache':False}
            cfg['candidate_id']='lora_'+identity(cfg)[:12];lora.append(cfg)
        selected_lora=search(panel,lora,'lora',store,'adaptation_and_mechanisms')
        frozen_seed=verify_seeds(panel,top,'fusion',store)
        lora_seed=verify_seeds(panel,selected_lora['top_two'],'lora',store)
        mechanisms={}
        for variant in ('full','plain_mlp','no_foundation','random_frozen','no_market','close','aapl'):
            variant_store=store
            if variant in ('no_market','close','aapl'):
                configs=expert_configurations(spec,'close' if variant=='close' else 'ohlcv',
                    'aapl' if variant=='aapl' else 'multistock',variant!='no_market')
                variant_store=build_oof(panel,configs,'adaptation_and_mechanisms')
            variants=mechanism_configs(top,variant)
            # Every paired variant gets both fixed structural recipes and all three seeds.
            mechanisms[variant]=verify_seeds(panel,variants,'mechanism_'+variant,variant_store)
        # The prespecified feature-concatenation alternative remains eligible even
        # when the first gated model fails; no new design is introduced after scores.
        candidates=frozen_seed+lora_seed+mechanisms['plain_mlp']
        transfer=read_json(OUT/'selections/transfer.json')
        candidates+=verify_seeds(panel,transfer['top_two'],'transfer_multi',store)
        candidates.sort(key=lambda r:(r['system']['mse'],r['config']['candidate_id']))
        report={'candidates':candidates,'selected':candidates[0],'mechanisms':mechanisms,
                'usage':totals(),'Q3_accessed':False,'completed_at_utc':utc_now()}
        write_json(OUT/'F5_complete.json',report);return
    raise ValueError('Only development stages F3/F4/F5 use this entry')


def paired_search(panel,configs,multistock,spec):
    path=OUT/'selections/transfer.json'
    if path.exists():return read_json(path)
    aapl=build_oof(panel,expert_configurations(spec,training_scope='aapl'))
    current=list(configs);rungs=[]
    for prefix in (.1,.3,1.):
        rankings=[]
        for config in current:
            sides={}
            for side in ('multi','aapl'):
                cfg={**config,'training_scope':'multistock' if side=='multi' else 'aapl'}
                if side=='aapl':cfg['adaptation_fraction']=0.
                results=[]
                for quarter in VALIDATION:
                    result,_=neural_trial(panel,cfg,quarter,2026,prefix,'transfer_'+side,multistock if side=='multi' else aapl)
                    results.append(result)
                sides[side]={'mse':pooled_score(results),'fold_scores':[r['scores'] for r in results]}
            rankings.append({'config':config,'sides':sides,'mse':sides['multi']['mse']})
        rankings.sort(key=lambda r:(r['mse'],r['config']['candidate_id']));rungs.append({'prefix':prefix,'rankings':rankings})
        count=halving_survivors(len(configs)) if prefix==.1 else min(2,len(current));current=[r['config'] for r in rankings[:count]]
        write_json(OUT/'search_progress/transfer.json',{'rungs':rungs})
    result={'rungs':rungs,'selected':rankings[0]['config'],'top_two':[r['config'] for r in rankings[:2]],
            'promotion_rule':'multistock MSE; both sides always advance together','completed_at_utc':utc_now()}
    write_json(path,result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('F3','F4','F5'));args=parser.parse_args()
    execute(args.stage)
