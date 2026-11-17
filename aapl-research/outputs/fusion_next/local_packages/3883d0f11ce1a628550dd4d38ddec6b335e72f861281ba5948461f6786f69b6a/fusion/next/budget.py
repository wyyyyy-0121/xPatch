"""Enumerate full required work and freeze coverage budgets before any scores."""
import argparse
import copy
import math

import numpy as np
import pandas as pd

from .common import OUT,PLAN,digest,read_json,write_json,utc_now,source_identity,identity
from .design import draft,halving_units,halving_survivors,freeze
from .panel import Panel,CoverageSampler
from .splits import VALIDATION,fold
from .accounting import LIMITS,totals,COMPUTE_POLICY


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
    def sum_updates(family,quarters):
        config=spec['groups'][family][0];table=config.get('updates_by_fit_end')
        if table is None:return len(quarters)*config.get('total_updates',spec.get('updates',{}).get(family,0))
        return sum(table[fold(panel.dates,str(q),confirmation=str(q)=='2026Q2')['fit_end']] for q in quarters)
    for family in ('ridge','lightgbm','xpatch','patchtst'):
        configs=spec['groups'][family];n=len(configs)
        if family in ('ridge','lightgbm'):
            add('experts_and_cache',family,n*nfold,predict_rows=n*all_main,description='same-information full CPU search')
        else:
            updates=configs[0]['total_updates']
            add('experts_and_cache',family,nfold*halving_units(n),sum_updates(family,VALIDATION)*halving_units(n),
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
            add(category,family,quarters,sum_updates(family,list(pd.period_range('2023Q1','2026Q1',freq='Q'))),predict_rows,'quarterly OOF '+description)
    for family,category,key in [('fusion','fusion_search','fusion'),('lora','adaptation_and_mechanisms','lora')]:
        n=len(spec['groups'][key]);updates=spec['updates'][family]
        add(category,family,nfold*halving_units(n),sum_updates(family,VALIDATION)*halving_units(n),halving_calls(n)*all_main,'neural search')
    n=len(spec['transfer_configs']);updates=spec['updates']['fusion']
    add('fusion_search','fusion',2*nfold*halving_units(n),2*sum_updates('fusion',VALIDATION)*halving_units(n),
        2*halving_calls(n)*all_main,'paired transfer; both sides always advance')
    # Generously include seed 2026 again rather than rely on a cache hit/namespace identity.
    for family in ('fusion','lora'):
        add('adaptation_and_mechanisms',family,2*3*nfold,2*3*sum_updates(family,VALIDATION),2*3*all_main,'top two, three seeds')
    add('adaptation_and_mechanisms','fusion',2*3*nfold,2*3*sum_updates('fusion',VALIDATION),2*3*all_main,'transfer top two, three seeds')
    # Seven variants, each gets both fixed structural configurations and all three seeds.
    for variant in ('full','plain_mlp','no_foundation','random_frozen','no_market','close','aapl'):
        family='fusion_random' if variant=='random_frozen' else 'fusion'
        add('adaptation_and_mechanisms',family,2*3*nfold,2*3*sum_updates('fusion',VALIDATION),2*3*all_main,
            'mechanism '+variant+' (no speed discount for removed modules)')
    # Preserve a conservative two-fit allowance even though the frozen final-fit
    # policy now evaluates the exact pre-Q2 artifact without post-Q2 retraining.
    final_rows=3*64*378  # extra allowance for development replay and inference acceptance
    add('final_and_deployment','lora',6,6*sum_updates('lora',['2026Q2']),6*final_rows,'pre-Q2 system fits, conservative spare fit allowance and acceptance replay')
    for family in ('ridge','lightgbm','xpatch','patchtst'):
        u=spec['groups'][family][0].get('total_updates',0)
        add('final_and_deployment',family,2,2*sum_updates(family,['2026Q2']),2*final_rows,'two governed expert/control retrains')
    return work,{'validation_origins':all_main,'all_stock_OOF_origins':all_oof,
                 'AAPL_OOF_origins':aapl_oof,'maximum_fit_calendar_days':max_fit_days,
                 'stocks':len(panel.stocks)}


def project(panel,spec,gpu,cpu):
    work,counts=work_items(panel,spec);m=gpu['measurements'];c=cpu['measurements']
    fp32=spec.get('groups',{}).get('fusion',[{}])[0].get('precision')=='fp32'
    suffix='_fp32' if fp32 else ''
    def timing(name):return m.get(name+suffix,m[name])
    # The random frozen backbone cannot reuse the pretrained cache. Use raw timing.
    rates={'xpatch':m['xpatch_end_to_end']['seconds_per_update'],
           'patchtst':m['patchtst_end_to_end']['seconds_per_update'],
           'fusion':max(timing('end_to_end_cache_population')['seconds_per_update'],timing('end_to_end_raw')['seconds_per_update'],timing('frozen_tcn_end_to_end')['seconds_per_update']),
           'fusion_random':max(timing('end_to_end_raw')['seconds_per_update'],timing('frozen_tcn_end_to_end')['seconds_per_update']),
           'lora':max(m['lora_stress']['seconds_per_update'],m['lora_end_to_end']['seconds_per_update'],m['lora_tcn_end_to_end']['seconds_per_update'])}
    infer={family:m[family+'_inference']['seconds_per_origin'] for family in ('xpatch','patchtst','lora')}
    infer['fusion']=infer['fusion_random']=max(timing('frozen_inference')['seconds_per_origin'],timing('frozen_tcn_inference')['seconds_per_origin'])
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
    over={name:{'hours':value,'limit':LIMITS[name]} for name,value in categories.items()
          if LIMITS[name] is not None and value>LIMITS[name]}
    total=sum(categories.values())
    return {'margin_multiplier':1.5,'total_gpu_card_hours_with_margin':total,
            'formal_wall_hours_projection_with_margin':total,'category_hours_with_margin':categories,
            'category_overruns':over,'within_budget':not over,'compute_policy':COMPUTE_POLICY,
            'CPU_work_seconds_before_margin':cpu_seconds,'CPU_GPU_overlap_credit':0,
            'all_required_work_included':True,'work':work,'data_counts':counts,
            'measurement_gpu_sha256':digest(OUT/'performance_admission.json'),
            'measurement_cpu_sha256':digest(OUT/'performance_cpu.json')}


def execution_spec(base,counts,widths,updates,batch_size,tables=None):
    spec=copy.deepcopy(base);spec['counts']=dict(counts);spec['updates']=dict(updates)
    admission=read_json(OUT/'performance_admission.json');checks=admission['checks']
    measured=admission['measurements']
    frozen_precision=admission['config']['precision']
    if ('end_to_end_cache_reuse_fp32' in measured and measured['end_to_end_cache_reuse_fp32']['seconds_per_update']
            <=.95*measured['end_to_end_cache_reuse']['seconds_per_update']):frozen_precision='fp32'
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
            if family in updates:
                size=admission.get('native_batch_sizes',{}).get(family,batch_size)
                cfg.update(total_updates=updates[family],batch_size=size,early_stopping_patience_updates=updates[family],
                    sampler_policy=CoverageSampler.policy,checkpoint_every_updates=1000,fused_optimizer=True,
                    precision=admission['config']['precision'],prefetch_batches=2,pin_memory=True)
                if tables is not None:cfg['updates_by_fit_end']=tables[family]
            if family in ('fusion','lora'):
                cfg['representation_cache']=family=='fusion'
                if family=='fusion':cfg['precision']=frozen_precision
                cfg['fp16_cache_admitted']=bool(checks.get('fp16_cache_prediction_rmse'))
                cfg['cache_dtype']='float16' if cfg['fp16_cache_admitted'] else 'float32'
            cfg['candidate_id']=family+'_'+identity({k:v for k,v in cfg.items() if k!='candidate_id'})[:12]
        spec['groups'][family]=configs
    spec['groups']['ridge']=[{'family':'ridge','candidate_id':'ridge_fixed_1000','lookback':96,'alpha':1000.,
                              'information':'ohlcv','training_scope':'multistock','rows_per_day':None,'cpu_threads':8}]
    spec['transfer_configs']=[]
    for index,cfg in enumerate(spec['groups']['fusion'][:counts['transfer_pairs']]):
        pair={**cfg,'adaptation_fraction':(.1,.2)[index%2],'adapt_lr_ratio':(.1,.3)[index//2%2]}
        pair['candidate_id']='transfer_'+identity(pair)[:12];spec['transfer_configs'].append(pair)
    spec['width_domain']=widths;spec['simple']=spec['simple'][:counts['simple']]
    return spec


def coverage_budget(panel,batch_size,epochs=2,native_batch_sizes=None):
    # Use the largest allowed fit, including the exact pre-Q2 final artifact.
    # Every earlier fold has at most this many balanced examples per epoch.
    from .splits import fold
    fit_end=fold(panel.dates,'2026Q2',confirmation=True)['fit_end']
    sampler=CoverageSampler(panel,fit_end,2026)
    tables={family:{} for family in ('xpatch','patchtst','fusion','lora')};folds={}
    sizes={family:(native_batch_sizes or {}).get(family,batch_size) for family in tables}
    for quarter in pd.period_range('2023Q1','2026Q2',freq='Q'):
        boundary=fold(panel.dates,str(quarter),confirmation=str(quarter)=='2026Q2')['fit_end']
        eligible=CoverageSampler(panel,boundary,2026)
        for family in tables:tables[family][boundary]=int(math.ceil(epochs*eligible.epoch_size/sizes[family]))
        steps=tables['fusion'][boundary]
        folds[str(quarter)]={'fit_end':boundary,'balanced_epoch_examples':eligible.epoch_size,
                            'eligible_unique_windows':eligible.unique_total,'total_updates':steps,
                            'updates_by_family':{family:table[boundary] for family,table in tables.items()}}
    updates={family:max(table.values()) for family,table in tables.items()}
    policy={'total_coverage_equivalent_epochs':epochs,'largest_fit_end':fit_end,
        'largest_fit_data':sampler.summary(),'maximum_adaptation_fraction':.2,
        'updates':updates,'updates_by_fit_end':tables,'folds':folds,
        'batch_sizes':sizes,'full_sample_budget':{name:value*sizes[name] for name,value in updates.items()},
        'sampling_policy':CoverageSampler.policy,'paired_variants_same_batch_and_updates':True,
        'AAPL_only_pair':'same sample/update budget as the multistock member; extra AAPL reuse reported',
        'adaptation_included_in_total':True,'additional_epochs_after_two':False,
        'CPU_rows_per_day':None,'CPU_training_rows':'all eligible windows with balanced stock/day weights',
        'halving_10_percent':'0.2 coverage-equivalent epochs; no full-coverage claim at the first rung',
        'training_diagnostics':'save mean training-loss traces and actual unique coverage; fixed update prefixes, no unimplemented early stopping',
        'budget_selection_uses_validation_scores':False}
    return updates,policy


def admit_search():
    if (OUT/'search_manifest.json').exists():raise ValueError('Search already frozen')
    gpu=read_json(OUT/'performance_admission.json');cpu=read_json(OUT/'performance_cpu.json')
    if not gpu.get('numerical_and_resource_checks_passed'):raise ValueError('Actual GPU numerical/resource admission has not passed')
    if gpu['source_identity']!=source_identity() or cpu['source_identity']!=source_identity():
        raise ValueError('Performance evidence was collected with different source; re-admit changed behavior')
    if cpu.get('hardware',{}).get('profile')!='autodl' or gpu.get('hardware',{}).get('profile')!='autodl':
        raise ValueError('Cloud budget requires both GPU and CPU timings from the cloud execution profile')
    panel=Panel.load();base=draft();counts=dict(base['counts']);batch=gpu['config']['batch_size']
    updates,policy=coverage_budget(panel,batch,native_batch_sizes=gpu.get('native_batch_sizes'))
    spec=execution_spec(base,counts,'full',updates,batch,policy['updates_by_fit_end'])
    spec['training_budget_policy']=policy;spec['compute_policy']=COMPUTE_POLICY
    projection=project(panel,spec,gpu,cpu)
    write_json(OUT/'budget_projection.json',{'attempts':[{'width':'full','counts':counts,
                    'reduction_step':'none: user removed time limits','projection':projection}],'selected':True,
                'update_budget_policy':policy,'compute_policy':COMPUTE_POLICY,
                'created_at_utc':utc_now(),'source_identity':source_identity(),
                'policy':'Full candidate counts and widths; complete coverage; measured costs retained without a GPU-hour cap'})
    # Bind the full evidence before enabling the development runner.
    gpu.update(passed=True,remaining_gate=None,budget_projection_sha256=digest(OUT/'budget_projection.json'))
    write_json(OUT/'performance_admission.json',gpu)
    projection['measurement_gpu_sha256']=digest(OUT/'performance_admission.json')
    return freeze(spec,projection)


if __name__=='__main__':
    argparse.ArgumentParser().parse_args();admit_search()
