"""Full-control arithmetic audit of the frozen BN intervention, not model selection."""
from pathlib import Path
import os,sys,time,json,hashlib
import numpy as np
import torch
import diagnostic_v2 as d
from portable_stats import day_groups,paired_intervals

HERE=Path(__file__).resolve().parent
MEAN=HERE.parent/'mean_controls'

def main():
    assert sorted(os.sched_getaffinity(0))==[24,26] and os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    assert d.sha(HERE/'diagnostic.py')==d.read(HERE/'NUMERICAL_ACCEPTANCE_REVISION.json')['original_source_sha256']
    result=d.read(HERE/'RESULTS.json');gate=d.read(HERE/'GATE_AND_ADMISSION.json')
    source_hashes={};task_results={};control_names=['zero','repair_original_raw','repair_original_shrink',
        'default_original_raw','default_original_shrink','ridge_history','ridge_history_anchor',
        'ridge_history_selected','ridge_history_anchor_selected','mean_raw','mean_shrink']
    for task in gate['full_tasks']:
        profile,rest=task.split('_h',1);h=int(rest.split('_',1)[0]);groups=[]
        seeds={str(s):{'adapted':0.,'adapted_shrink':0.} for s in (2026,2027,2028)}
        totals={};folds=[];n=0
        for fold in range(1,6):
            source=d.V3/'analysis/candidate_paired_losses'/task/f'fold_{fold}.npz'
            source_hashes[str(source)]=d.sha(source)
            with np.load(source,allow_pickle=False) as a:
                idx=a['origins'];losses={'zero':a['loss_zero'],'default_original_raw':a['loss_original'],
                    'default_original_shrink':a['loss_original_shrink']}
                for name in ('ridge_history','ridge_history_anchor','ridge_history_selected','ridge_history_anchor_selected'):
                    losses[name]=a['loss_'+name]
                reference_repair_raw=a['loss_trained'];reference_repair_shrink=a['loss_candidate']
            modes={name:[] for name in ('original','original_shrink','adapted','adapted_shrink')}
            for seed in (2026,2027,2028):
                cfg=d.cfg_for(profile,h,fold,seed);folder=HERE/'cells'/cfg['id'];receipt=d.read(folder/'OUTER_DONE.json')
                path=folder/'outer_losses.npz';assert d.sha(path)==receipt['losses_sha256']
                source_hashes[str(path)]=receipt['losses_sha256']
                _,done,_=d.source(cfg)
                src=d.V3/'runs'/cfg['id']/'predictions.npz'
                assert d.sha(src)==done['prediction_sha256'];source_hashes[str(src)]=done['prediction_sha256']
                with np.load(path,allow_pickle=False) as a:
                    assert np.array_equal(idx,a['origins'])
                    assert np.array_equal(losses['zero'],a['zero'])
                    for mode in modes:modes[mode].append(a[mode])
                    for mode in ('adapted','adapted_shrink'):seeds[str(seed)][mode]+=float(a[mode].sum())
            losses['repair_original_raw']=np.mean(modes['original'],axis=0)
            losses['repair_original_shrink']=np.mean(modes['original_shrink'],axis=0)
            assert np.allclose(losses['repair_original_raw'],reference_repair_raw,rtol=1e-12,atol=1e-14)
            assert np.allclose(losses['repair_original_shrink'],reference_repair_shrink,rtol=1e-12,atol=1e-14)
            losses['adapted']=np.mean(modes['adapted'],axis=0)
            losses['adapted_shrink']=np.mean(modes['adapted_shrink'],axis=0)
            mean_path=MEAN/'predictions'/task/f'fold_{fold}.npz'
            mean_done=d.read(MEAN/'fits'/task/f'fold_{fold}'/'DONE.json')
            assert d.sha(mean_path)==mean_done['prediction_sha256']
            source_hashes[str(mean_path)]=d.sha(mean_path)
            with np.load(mean_path,allow_pickle=False) as a:
                assert np.array_equal(idx,a['origins']) and np.array_equal(losses['zero'],a['loss_baseline'])
                losses['mean_raw']=a['loss_raw'];losses['mean_shrink']=a['loss_shrink']
            d.npz(HERE/'all_control_losses'/task/f'fold_{fold}.npz',origins=idx,**losses)
            for name,value in losses.items():totals[name]=totals.get(name,0.)+float(value.sum())
            folds.append({'fold':fold,'origins':len(idx),'adapted_gain_pct':100*(1-losses['adapted'].mean()/losses['zero'].mean()),
                'adapted_shrink_gain_pct':100*(1-losses['adapted_shrink'].mean()/losses['zero'].mean())})
            data=d.Data(d.cfg_for(profile,h,fold,2026));groups.append(day_groups(data.bars,idx,losses));n+=len(idx)
        pooled={name:value/n for name,value in totals.items()}
        gains={name:100*(1-value/pooled['zero']) for name,value in pooled.items()}
        seed_gains={seed:{mode:100*(1-value/totals['zero']) for mode,value in values.items()} for seed,values in seeds.items()}
        intervals=[]
        for block in ((1,5) if profile=='minute' else (2,5)):
            intervals+=paired_intervals(groups,[(mode,control) for mode in ('adapted','adapted_shrink') for control in control_names],block,task)
        modes={}
        for mode in ('adapted','adapted_shrink'):
            checks={'minimum_one_percent_over_zero':gains[mode]>=1.,
                'all_three_seed_directions_positive':all(v[mode]>0 for v in seed_gains.values()),
                'four_positive_fold_means':sum(r[mode+'_gain_pct']>0 for r in folds)>=4,
                'all_controls_positive_both_block_intervals':all(r['supports_improvement'] for r in intervals if r['primary']==mode)}
            modes[mode]={'checks':checks,'performance_gate_passed':all(checks.values())}
        task_results[task]={'scope':'five_fold_three_seed_exposed_development','origins':n,
            'pooled_mse':pooled,'improvement_over_zero_pct':gains,'per_seed_improvement_over_zero_pct':seed_gains,
            'folds':folds,'controls':control_names,'paired_intervals':intervals,'performance_modes':modes}
    receipt={'status':'completed_all_registered_controls_for_admitted_tasks','tasks':task_results,
        'screen_only_tasks':[task for task in result['tasks'] if task not in gate['full_tasks']],
        'original_and_repair_controls_restored':True,'source_file_hashes':source_hashes,
        'numerical_tolerance_revision_recorded':True,'original_failed_tolerance_retained':True,
        'independent_confirmation':False,'Q3_values_read':False,'method_success_claimed':False,
        'statistical_scope':'Exposed development; mean-seed loss by origin; nominal intervals without multiplicity adjustment',
        'at_utc':d.now(),'at_epoch':time.time()}
    d.atomic(HERE/'CONTROL_AUDIT.json',receipt)
    print(json.dumps({'status':receipt['status'],'tasks':{t:r['performance_modes'] for t,r in task_results.items()}}),flush=True)

if __name__=='__main__':main()
