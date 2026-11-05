"""Enumerate required work and select a pre-score search that fits measured limits."""
import argparse
import copy
import math

import numpy as np
import pandas as pd

from .common import OUT,PLAN,digest,read_json,write_json,utc_now,source_identity,identity
from .design import draft,halving_units,halving_survivors,freeze
from .panel import Panel
from .splits import VALIDATION
from .accounting import LIMITS,totals


def work_items(panel,spec):
    """Conservative counts include every prescribed side, seed, OOF and prediction."""
    nfold=len(VALIDATION);all_main=0;all_oof=0;aapl_oof=0;max_fit_days=0
    for name in panel.stocks:
        rows=panel.origins[name];dates=panel.dates[rows]
        all_oof+=int(np.sum((dates>='2023-01-01')&(dates<='2026-03-31')))
        if name=='AAPL':
            aapl_oof=int(np.sum((dates>='2023-01-01')&(dates<='2026-03-31')))
            for q in VALIDATION:
                period=pd.Period(q);all_main+=int(np.sum((dates>=str(period.start_time.date()))&(dates<=str(period.end_time.date()))))
    max_fit_days=len(np.unique(panel.dates[panel.dates<'2026-04-01']))
    work=[]
    def add(category,family,fits,updates=0,predict_rows=0,description=''):
        work.append({'category':category,'family':family,'fit_equivalents':float(fits),'updates':float(updates),
                     'predict_rows':int(predict_rows),'description':description})
    def halving_calls(count):return count+halving_survivors(count)+min(2,halving_survivors(count))
    for family in ('ridge','lightgbm','xpatch','patchtst'):
        configs=spec['groups'][family];n=len(configs)
        if family in ('ridge','lightgbm'):
            add('experts_and_cache',family,n*nfold,predict_rows=n*all_main,description='same-information full CPU search')
        else:
            updates=configs[0]['total_updates']
            add('experts_and_cache',family,nfold*halving_units(n),updates*nfold*halving_units(n),
                halving_calls(n)*all_main,'supervised search 10/30/100% prefixes')
    # Four independent information/training scopes; the AAPL-only OOF supports transfer.
    quarters=13
    for family in ('ridge','lightgbm','xpatch','patchtst'):
        scopes=[('experts_and_cache',all_oof,'main OHLCV'),
                ('fusion_search',aapl_oof,'AAPL-only paired transfer'),
                ('adaptation_and_mechanisms',all_oof,'close-only mechanism'),
                ('adaptation_and_mechanisms',all_oof,'no contemporaneous market mechanism')]
        updates=spec['groups'][family][0].get('total_updates',0)
        for category,predict_rows,description in scopes:
            add(category,family,quarters,quarters*updates,predict_rows,'quarterly OOF '+description)
    for family,category,key in [('fusion','fusion_search','fusion'),('lora','adaptation_and_mechanisms','lora')]:
        n=len(spec['groups'][key]);updates=spec['updates'][family]
        add(category,family,nfold*halving_units(n),updates*nfold*halving_units(n),halving_calls(n)*all_main,'neural search')
    n=len(spec['transfer_configs']);updates=spec['updates']['fusion']
    add('fusion_search','fusion',2*nfold*halving_units(n),2*nfold*halving_units(n)*updates,
        2*halving_calls(n)*all_main,'paired transfer; both sides always advance')
    # Generously include seed 2026 again rather than rely on a cache hit/namespace identity.
    for family in ('fusion','lora'):
        add('adaptation_and_mechanisms',family,2*3*nfold,2*3*nfold*spec['updates'][family],2*3*all_main,'top two, three seeds')
    add('adaptation_and_mechanisms','fusion',2*3*nfold,2*3*nfold*updates,2*3*all_main,'transfer top two, three seeds')
    # Seven variants, each gets both fixed structural configurations and all three seeds.
    for variant in ('full','plain_mlp','no_foundation','random_frozen','no_market','close','aapl'):
        family='fusion_random' if variant=='random_frozen' else 'fusion'
        add('adaptation_and_mechanisms',family,2*3*nfold,2*3*nfold*updates,2*3*all_main,
            'mechanism '+variant+' (no speed discount for removed modules)')
    # Preserve a conservative two-fit allowance even though the frozen final-fit
    # policy now evaluates the exact pre-Q2 artifact without post-Q2 retraining.
    final_rows=3*64*378  # extra allowance for development replay and inference acceptance
    add('final_and_deployment','lora',6,6*spec['updates']['lora'],6*final_rows,'pre-Q2 system fits, conservative spare fit allowance and acceptance replay')
    for family in ('ridge','lightgbm','xpatch','patchtst'):
        u=spec['groups'][family][0].get('total_updates',0)
        add('final_and_deployment',family,2,2*u,2*final_rows,'two governed expert/control retrains')
    return work,{'validation_origins':all_main,'all_stock_OOF_origins':all_oof,
                 'AAPL_OOF_origins':aapl_oof,'maximum_fit_calendar_days':max_fit_days,
                 'stocks':len(panel.stocks)}


def project(panel,spec,gpu,cpu):
    work,counts=work_items(panel,spec);m=gpu['measurements'];c=cpu['measurements']
    # The random frozen backbone cannot reuse the pretrained cache. Use raw timing.
    rates={'xpatch':m['xpatch_end_to_end']['seconds_per_update'],
           'patchtst':m['patchtst_end_to_end']['seconds_per_update'],
           'fusion':max(m['end_to_end_cache_population']['seconds_per_update'],m['end_to_end_raw']['seconds_per_update'],m['frozen_tcn_end_to_end']['seconds_per_update']),
           'fusion_random':max(m['end_to_end_raw']['seconds_per_update'],m['frozen_tcn_end_to_end']['seconds_per_update']),
           'lora':max(m['lora_stress']['seconds_per_update'],m['lora_end_to_end']['seconds_per_update'],m['lora_tcn_end_to_end']['seconds_per_update'])}
    infer={family:m[family+'_inference']['seconds_per_origin'] for family in ('xpatch','patchtst','lora')}
    infer['fusion']=infer['fusion_random']=max(m['frozen_inference']['seconds_per_origin'],m['frozen_tcn_inference']['seconds_per_origin'])
    infer['lora']=max(infer['lora'],m['lora_tcn_inference']['seconds_per_origin'])
    current=totals();categories={name:0. for name in LIMITS};cpu_seconds=0.
    init=m['fusion_initialization_seconds']+m['scaler_fit_seconds']*counts['maximum_fit_calendar_days']/m['scaler_training_calendar_days']
    for item in work:
        family=item['family'];seconds=0.
        if family in rates:
            seconds=item['updates']*rates[family]+item['predict_rows']*infer[family]+item['fit_equivalents']*(init+2.)
        else:
            measure=c[family]
            factor=counts['maximum_fit_calendar_days']/measure['training_calendar_days']
            seconds=item['fit_equivalents']*(measure['scaler_seconds']+measure['feature_seconds']+measure['fit_seconds'])*factor
            seconds+=item['predict_rows']*measure['prediction_seconds_per_origin']
            # Until measured overlap is implemented, count CPU time in the GPU reservation too.
            cpu_seconds+=seconds
        item['projected_seconds_before_margin']=seconds;categories[item['category']]+=seconds/3600
    categories={name:value*1.5+current['by_category'].get(name,0.) for name,value in categories.items()}
    categories['performance']=max(categories['performance'],current['by_category'].get('performance',0.))
    categories['recovery']=2.
    over={name:{'hours':value,'limit':LIMITS[name]} for name,value in categories.items() if value>LIMITS[name]}
    total=sum(categories.values())
    return {'margin_multiplier':1.5,'total_gpu_card_hours_with_margin':total,
            'formal_wall_hours_projection_with_margin':total,'category_hours_with_margin':categories,
            'category_overruns':over,'within_budget':not over and total<=24,
            'CPU_work_seconds_before_margin':cpu_seconds,'CPU_GPU_overlap_credit':0,
            'all_required_work_included':True,'work':work,'data_counts':counts,
            'measurement_gpu_sha256':digest(OUT/'performance_admission.json'),
            'measurement_cpu_sha256':digest(OUT/'performance_cpu.json')}


def execution_spec(base,counts,widths,updates,batch_size):
    spec=copy.deepcopy(base);spec['counts']=dict(counts);spec['updates']=dict(updates)
    checks=read_json(OUT/'performance_admission.json')['checks']
    for family in ('lightgbm','xpatch','patchtst','fusion','lora'):
        configs=spec['groups'][family]
        if widths=='narrow':
            # Width reduction precedes count reductions; preserve other candidate diversity.
            for cfg in configs:
                if 'projection' in cfg:cfg['projection']=64
                if 'hidden' in cfg:cfg['hidden']=32
                if 'd_model' in cfg:cfg['d_model']=32
        configs=configs[:counts[family]]
        for cfg in configs:
            if family in updates:cfg.update(total_updates=updates[family],batch_size=batch_size,early_stopping_patience_updates=updates[family])
            if family in ('fusion','lora'):
                cfg['representation_cache']=family=='fusion'
                cfg['precision']='fp32'  # Lower precision optional; no automatic untested substitution.
            cfg['candidate_id']=family+'_'+identity({k:v for k,v in cfg.items() if k!='candidate_id'})[:12]
        spec['groups'][family]=configs
    spec['groups']['ridge']=[{'family':'ridge','candidate_id':'ridge_fixed_1000','lookback':96,'alpha':1000.,
                              'information':'ohlcv','training_scope':'multistock','rows_per_day':16,'cpu_threads':8}]
    spec['transfer_configs']=[]
    for index,cfg in enumerate(spec['groups']['fusion'][:counts['transfer_pairs']]):
        pair={**cfg,'adaptation_fraction':(.1,.2)[index%2],'adapt_lr_ratio':(.1,.3)[index//2%2]}
        pair['candidate_id']='transfer_'+identity(pair)[:12];spec['transfer_configs'].append(pair)
    spec['width_domain']=widths;spec['simple']=spec['simple'][:counts['simple']]
    return spec


def admit_search():
    if (OUT/'search_manifest.json').exists():raise ValueError('Search already frozen')
    gpu=read_json(OUT/'performance_admission.json');cpu=read_json(OUT/'performance_cpu.json')
    if not gpu.get('numerical_and_resource_checks_passed'):raise ValueError('Actual GPU numerical/resource admission has not passed')
    if gpu['source_identity']!=source_identity() or cpu['source_identity']!=source_identity():
        raise ValueError('Performance evidence was collected with different source; re-admit changed behavior')
    if cpu.get('hardware',{}).get('profile')!='autodl' or gpu.get('hardware',{}).get('profile')!='autodl':
        raise ValueError('Cloud budget requires both GPU and CPU timings from the cloud execution profile')
    panel=Panel.load();base=draft();attempts=[];counts=dict(base['counts'])
    measured=gpu['measurements']
    # Update budgets are a speed-based choice made before candidate counts/search.
    # The 1000..4000 range is fixed here and recorded even when the fastest recipe
    # cannot satisfy the complete category projection; no validation score is read.
    seconds_per_update={'xpatch':measured['xpatch_end_to_end']['seconds_per_update'],
        'patchtst':measured['patchtst_end_to_end']['seconds_per_update'],
        'fusion':measured['end_to_end_raw']['seconds_per_update'],
        'lora':measured['lora_end_to_end']['seconds_per_update']}
    updates={name:int(np.clip(math.floor((120 if name in ('xpatch','patchtst') else 90)/seconds/100)*100,1000,4000))
             for name,seconds in seconds_per_update.items()}
    batch=gpu['config']['batch_size'];variants=[('full',dict(counts),'full search'),('narrow',dict(counts),'reduce width choices')]
    # Ordered count reductions retain every family and every required paired mechanism.
    for factor in (.75,.5,.25):
        reduced={key:(12 if key=='simple' else (counts[key] if key=='lora' else max(2,int(value*factor)))) for key,value in counts.items()}
        variants.append(('narrow',reduced,'reduce initial counts; retain paired mechanisms and LoRA domain'))
    last=dict(variants[-1][1])
    variants.append(('narrow',dict(last),'representation work is already lazy on sampled prefixes; no full candidate precomputation to remove'))
    last['lora']=2;variants.append(('narrow',last,'finally restrict LoRA to two prespecified configurations'))
    selected=None
    for width,reduced,reduction_step in variants:
        spec=execution_spec(base,reduced,width,updates,batch);projection=project(panel,spec,gpu,cpu)
        attempts.append({'width':width,'counts':reduced,'reduction_step':reduction_step,'projection':projection})
        if projection['within_budget']:selected=(spec,projection);break
    write_json(OUT/'budget_projection.json',{'attempts':attempts,'selected':selected is not None,
                'update_budget_policy':{'updates':updates,'seconds_per_update':seconds_per_update,
                    'minimum':1000,'maximum':4000,'selection':'fixed 120s supervised / 90s fusion training allocation, rounded to 100 updates and clipped before all search reductions'},
                'created_at_utc':utc_now(),'source_identity':source_identity(),
                'policy':'No validation score is read; no required mechanism is removed; stop if measured full scope cannot fit'})
    if selected is None:raise RuntimeError('The full experiment cannot fit the fixed package budgets; no score/search has started')
    spec,projection=selected
    # Bind the full evidence before enabling the development runner.
    gpu.update(passed=True,remaining_gate=None,budget_projection_sha256=digest(OUT/'budget_projection.json'))
    write_json(OUT/'performance_admission.json',gpu)
    projection['measurement_gpu_sha256']=digest(OUT/'performance_admission.json')
    return freeze(spec,projection)


if __name__=='__main__':
    argparse.ArgumentParser().parse_args();admit_search()
