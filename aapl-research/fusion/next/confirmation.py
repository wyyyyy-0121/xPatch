"""One fixed Q2 process confirmation and an exported system even on a negative path."""
import argparse
import gc

import numpy as np
import torch

from .common import OUT,PLAN,digest,identity,read_json,write_json,utc_now,source_identity,status
from .experiment import admitted,expert_configurations,model_for
from .panel import Panel
from .splits import fold
from .hardware import runtime
from .trainer import Trainer,ExpertStore
from .experts import NAMES,fit_cpu,predict_cpu
from .accounting import Usage
from .inference import export
from . import simple
from .evaluation import point_scores,seed_system


def checked_freeze():
    spec=admitted();path=OUT/'system_freeze.json';frozen=read_json(path)
    if identity({k:v for k,v in frozen.items() if k!='identity'})!=frozen['identity']:
        raise ValueError('Fixed system freeze identity differs')
    if frozen['source_identity']!=source_identity() or frozen['plan_sha256']!=digest(PLAN):
        raise ValueError('Code or user plan changed after system freeze')
    for name,key in [('search_manifest.json','search_manifest_sha256'),('F5_complete.json','F5_sha256')]:
        if digest(OUT/name)!=frozen[key]:raise ValueError('Pre-confirmation selection artifact changed: '+name)
    return spec,frozen


def fit_frozen(panel,spec,frozen,fit_end,namespace):
    """Fit exact experts and three seeds on earlier data; no evaluation labels."""
    config=frozen['selected_system']['config'];configs=expert_configurations(spec,
        config.get('information','ohlcv'),config.get('training_scope','multistock'),config.get('market_information',True))
    references={}
    for name in NAMES:
        cfg=configs[name];directory=OUT/'frozen_fits'/namespace/name
        if name in ('ridge','lightgbm'):
            with Usage('final_and_deployment',namespace+'/'+name,gpu=False):
                fit_cpu(panel,cfg,fit_end,directory)
            references[name]={'config':cfg,'directory':str(directory)}
        else:
            device=runtime(2026)
            if device!='cuda':raise RuntimeError('Final neural retraining is assigned to the cloud GPU')
            with Usage('final_and_deployment',namespace+'/'+name) as usage:
                model=model_for(panel,cfg,device)
                trainer=Trainer(panel,model,cfg,fit_end,'frozen_fits/'+namespace+'/'+name,device=device)
                if trainer.path.exists():trainer.resume()
                trainer.run_until(cfg['total_updates'],usage)
                references[name]={'config':cfg,'checkpoint':str(trainer.path)}
                del trainer,model
            gc.collect();torch.cuda.empty_cache()
    store=ExpertStore(panel,config.get('information','ohlcv'),config.get('training_scope','multistock'),config.get('market_information',True))
    systems=[]
    for seed in (2026,2027,2028):
        cfg={**config,'seed':seed};device=runtime(seed)
        if device!='cuda':raise RuntimeError('Final system retraining requires the specified GPU')
        with Usage('final_and_deployment',namespace+f'/system/seed_{seed}') as usage:
            model=model_for(panel,cfg,device)
            trainer=Trainer(panel,model,cfg,fit_end,'frozen_fits/'+namespace+'/system',seed,store,device)
            if trainer.path.exists():trainer.resume()
            trainer.run_until(cfg['total_updates'],usage)
            systems.append({'config':cfg,'checkpoint':str(trainer.path)});del trainer,model
        gc.collect();torch.cuda.empty_cache()
    return references,systems


def fit_simple_before_confirmation(panel,config,store,fit_end):
    sid=panel.stocks.index('AAPL');rows=panel.origins['AAPL']
    rows=rows[(panel.dates[rows]<=fit_end)&store.available[sid][rows]]
    x=store.predict(np.full(len(rows),sid),rows);y=panel.targets['AAPL'][rows]
    _,inverse,counts=np.unique(panel.dates[rows],return_inverse=True,return_counts=True)
    return simple.fit(x,y,config,weights=1/counts[inverse])


def run():
    spec,frozen=checked_freeze();panel=Panel.load()
    config=frozen['selected_system']['config']
    if config.get('information','ohlcv')!='ohlcv' or not config.get('market_information',True) or config.get('training_scope')!='multistock':
        raise ValueError('Selected non-main scope needs separately frozen primary controls')
    path=OUT/'confirmation.json'
    if path.exists():raise ValueError('The fixed Q2 process has already been evaluated; do not select or score a replacement')
    f=fold(panel.dates,'2026Q2',confirmation=True)
    # Even a negative development outcome gets an executable system. It remains a
    # negative-path engineering export and does not acquire Q2/Q3 research success.
    refs,systems=fit_frozen(panel,spec,frozen,f['fit_end'],'pre_Q2')
    destination=OUT/'deployment/pre_Q2'
    if not destination.exists():export(destination,panel.stocks,panel.background,refs,systems,digest(OUT/'system_freeze.json'))
    if not frozen['development_gate_passed']:
        report={'evaluated':False,'reason':'development entry gate failed; Q2 and Q3 scoring not performed',
                'system_freeze_sha256':digest(OUT/'system_freeze.json'),'deployment_directory':str(destination),
                'Q3_opened':False,'proceed_to_final_test':False,'completed_at_utc':utc_now()}
        write_json(path,report);return report
    # Exact frozen artifacts are ready before claiming the one confirmation run.
    claim=OUT/'confirmation_started.json'
    expected={'system_freeze_sha256':digest(OUT/'system_freeze.json'),'deployment_identity':read_json(destination/'deployment.json')['identity']}
    if claim.exists():
        previous=read_json(claim)
        if any(previous[k]!=v for k,v in expected.items()):raise ValueError('Confirmation retry changed the fixed system')
    else:write_json(claim,{**expected,'started_at_utc':utc_now(),'purpose':'one fixed known-Q2 process confirmation'})
    # Q2 prediction uses actual fit artifacts once. It is kept out of development OOF arrays.
    from .inference import FrozenSystem
    device=runtime(2026)
    if device!='cuda':raise RuntimeError('Q2 computation is assigned to the training GPU')
    rows=panel.origins['AAPL'];rows=rows[(panel.dates[rows]>=f['score_start'])&(panel.dates[rows]<=f['score_end'])]
    config=frozen['selected_system']['config'];predictions=[];seed_parts=[];expert_parts=[]
    with Usage('final_and_deployment','one fixed Q2 prediction') as usage:
        model=FrozenSystem(destination,device)
        for start in range(0,len(rows),config['batch_size']):
            output=model.predict_origin(panel,rows[start:start+config['batch_size']])
            predictions.append(output['prediction']);seed_parts.append(output['seed_predictions']);expert_parts.append(output['expert_predictions'])
            if start%(20*config['batch_size'])==0:usage.record()
        del model
    gc.collect();torch.cuda.empty_cache()
    prediction=np.concatenate(predictions);seeds=np.concatenate(seed_parts,axis=1);experts=np.concatenate(expert_parts)
    target=panel.targets['AAPL'][rows]
    simple_store=ExpertStore(panel)
    simple_fit=fit_simple_before_confirmation(panel,frozen['fixed_strongest_simple']['config'],simple_store,f['fit_end'])
    # Main OHLCV controls remain fixed even when a feature-concatenation system wins.
    # The selected systems currently retain this main scope; future new scopes must
    # provide a separate exact control fit instead of reusing mismatched experts.
    single=experts[:,NAMES.index(frozen['fixed_strongest_single']['family'])];static=simple.predict(simple_fit,experts)
    seed_report,_=seed_system(target,seeds)
    scores={'system':point_scores(target,prediction),'zero':point_scores(target,np.zeros(len(target))),
            'single':point_scores(target,single),'simple':point_scores(target,static)}
    checks={'better_than_zero':scores['system']['mse']<scores['zero']['mse'],
            'better_than_fixed_single':scores['system']['mse']<scores['single']['mse'],
            'better_than_fixed_simple':scores['system']['mse']<scores['simple']['mse']}
    prediction_path=OUT/'confirmation_predictions.npz'
    np.savez_compressed(prediction_path,rows=rows,target=target,prediction=prediction,seeds=seeds,
                        experts=experts,single=single,simple=static,dates=panel.dates[rows])
    report={'evaluated':True,'Q2_status':frozen['Q2_status'],'scores':scores,'seed_report':seed_report,
            'checks':checks,'proceed_to_final_test':all(checks.values()),'Q3_opened':False,
            'system_freeze_sha256':digest(OUT/'system_freeze.json'),'prediction_sha256':digest(prediction_path),
            'deployment_directory':str(destination),'deployment_identity':read_json(destination/'deployment.json')['identity'],
            'fixed_simple_fit':simple_fit,'completed_at_utc':utc_now()}
    write_json(path,report);return report


if __name__=='__main__':
    argparse.ArgumentParser().parse_args();print(run())
