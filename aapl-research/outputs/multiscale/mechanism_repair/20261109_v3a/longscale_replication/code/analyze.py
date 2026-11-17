"""Complete paired long-scale analysis; no new primary endpoint selection."""
from common import *
import csv,os
import numpy as np
import torch
from engine import Data
from portable_stats import day_groups,paired_intervals


def summarize(losses,mask,baseline):
    count=int(np.sum(mask))
    if not count:return {'status':'not_estimable','origins':0}
    bm=float(np.mean(losses[baseline][mask]))
    return {'status':'descriptive','origins':count,'methods':{name:{
        'endpoint_mse':float(np.mean(value[mask])),
        'improvement_pct':float(100*(1-np.mean(value[mask])/bm)) if bm else None}
        for name,value in losses.items()}}


def main():
    assert len(os.sched_getaffinity(0))<=2 and os.environ['CUDA_VISIBLE_DEVICES']=='-1'
    manifest=read(BASE/'protocol/JOBS.json')
    assert read(BASE/'audit/COMPLETE.json')['cells']==180
    assert read(BASE/'audit/CALIBRATION_COMPLETE.json')['cells']==180
    freeze=BASE/'protocol/ANALYSIS_FREEZE.json'
    frozen={'source_sha256':sha(__file__),'portable_statistics_sha256':sha(V3/'code/portable_stats.py'),
        'new_results_aggregated':False,'post_observation_development':True,
        'subset_rules':'global earliest greedy with gap>=H; every absolute origin moduloH phase; leave each origin calendar year out; future split dates (origin_date,endpoint_date]; no groups declared independent',
        'baseline_controls':'same3seed default, raw/shrunk unconditional fit-only means, four original same-info Ridge variants, zero/persistence',
        'at_utc':now()}
    if freeze.exists():
        assert read(freeze)['source_sha256']==sha(__file__)
    else:atomic(freeze,frozen)
    pooled=[];rows=[];intervals=[];sensitivity=[];reproductions=[]
    mean_root=BASE.parent/'mean_controls'
    for profile,h in (('annual',12),('quarterly',3),('monthly',21)):
        for target in ('future_close','future_log_return'):
            task=task_key(profile,h,target);cfgs=[j for j in manifest['jobs'] if task_key(j['profile'],j['horizon'],j['target'])==task]
            data=Data(cfgs[0]);baseline_name='zero' if target=='future_log_return' else 'persistence'
            all_origins=[];all_fold=[];groups=[];loss_lists={};seed_lists={};fold_scores={};path_sums={};mae_sums={}
            for fold in range(1,6):
                meta=read(PRIOR/'cache'/task/'TASK.json');rec=meta['folds'][str(fold)]
                assert sha(rec['targets'])==rec['targets_sha256']
                with np.load(rec['targets'],allow_pickle=False) as a:
                    origins=a['origins'];y=a['actual'];anchor=a['anchor']
                baseline=np.zeros_like(y) if target=='future_log_return' else np.broadcast_to(anchor[:,None],y.shape)
                losses={baseline_name:np.square(baseline[:,-1]-y[:,-1])};local={}
                for cfg in [j for j in cfgs if j['fold']==fold]:
                    src=folder(cfg,manifest);done=read(src/'DONE.json')
                    assert done['job']==cfg and done['status']=='completed'
                    assert sha(src/'predictions.npz')==done['prediction_sha256']
                    with np.load(src/'predictions.npz',allow_pickle=False) as a:
                        assert np.array_equal(a['origins'],origins);trained=a['trained']
                    if cfg['seed']==2026 and cfg['initialization']=='random':
                        old=rec['models']['xpatch']
                        assert sha(old['checkpoint'])==old['checkpoint_sha256']
                        ck=torch.load(old['checkpoint'],map_location='cpu',weights_only=True)
                        new=torch.load(src/'best_trained.pt',map_location='cpu',weights_only=True)
                        weights_exact=set(ck['model'])==set(new['model']) and all(
                            torch.equal(ck['model'][k],new['model'][k]) for k in ck['model'])
                        assert sha(old['prediction'])==old['prediction_sha256']
                        with np.load(old['prediction']) as a:
                            assert np.array_equal(a['origins'],origins)
                            predictions_exact=np.array_equal(a['prediction'],trained)
                        assert weights_exact and ck['epoch']==new['epoch']
                        same_original_batch_exact=predictions_exact
                        if not predictions_exact:
                            explanation=read(BASE/'audit/RESTORATION_BATCH_EXPLANATION.json')
                            same_original_batch_exact=next(r['archived_512_batch_output_exact'] for r in
                                explanation['checks'] if r['job']==cfg['id'])
                        assert same_original_batch_exact
                        reproductions.append({'task':task,'fold':fold,'seed':2026,
                            'weights_exact':True,'prediction_exact':predictions_exact,'best_epoch_exact':True,
                            'same_original_512_batch_prediction_exact':same_original_batch_exact})
                    cal_dir=BASE/'calibration'/cfg['id'];cal=read(cal_dir/'DONE.json')
                    assert sha(cal_dir/'predictions.npz')==cal['prediction_sha256']
                    with np.load(cal_dir/'predictions.npz') as a:
                        assert np.array_equal(a['origins'],origins);calibrated=a['prediction']
                    kind='repair' if cfg['initialization']=='baseline' else 'default'
                    for mode,p in (('trained',trained),('shrink',calibrated)):
                        name=kind+'_'+mode;error=np.square(p[:,-1]-y[:,-1]);local.setdefault(name,[]).append(error)
                        seed_lists.setdefault(name,{}).setdefault(cfg['seed'],[]).append(error)
                        metric=original.metrics(p,y,anchor,target)
                        path_sums[name]=path_sums.get(name,0.)+metric['path_mse']*len(origins)/3
                        mae_sums[name]=mae_sums.get(name,0.)+metric['endpoint_mae']*len(origins)/3
                        rows.append({'task':task,'fold':fold,'seed':cfg['seed'],'method':name,
                            **metric,'alpha':cal['alpha'],'epochs':done['epochs'],'best_epoch':done['trained_best_epoch']})
                assert all(len(values)==3 for values in local.values())
                losses.update({name:np.mean(values,axis=0) for name,values in local.items()})
                control_dir=V3/'controls_cross_scale'/task/f'fold_{fold}'
                control=read(control_dir/'DONE.json')
                assert sha(control_dir/'predictions.npz')==control['prediction_sha256']
                with np.load(control_dir/'predictions.npz',allow_pickle=False) as a:
                    assert np.array_equal(a['origins'],origins)
                    for name in ('history','history_anchor','history_selected','history_anchor_selected'):
                        losses['ridge_'+name]=np.square(a[name][:,-1]-y[:,-1])
                mean_file=mean_root/'predictions'/task/f'fold_{fold}.npz'
                with np.load(mean_file,allow_pickle=False) as a:
                    assert np.array_equal(a['origins'],origins)
                    for key,name in (('raw','fit_mean'),('shrink','fit_mean_shrink')):
                        losses[name]=np.square(a[key][:,-1]-y[:,-1])
                for name,value in losses.items():
                    loss_lists.setdefault(name,[]).append(value)
                    fold_scores.setdefault(name,[]).append(float(np.mean(value)))
                all_origins.append(origins);all_fold.extend([fold]*len(origins))
                groups.append(day_groups(data.bars,origins,losses))
            origins=np.concatenate(all_origins);losses={name:np.concatenate(values) for name,values in loss_lists.items()}
            assert np.all(np.diff(origins)>0)
            n=len(origins);bm=float(np.mean(losses[baseline_name]))
            for name,value in losses.items():
                seed_values={str(seed):float(100*(1-np.mean(np.concatenate(v))/bm))
                             for seed,v in seed_lists.get(name,{}).items()}
                pooled.append({'task':task,'method':name,'origins':n,'endpoint_mse':float(np.mean(value)),
                    'baseline_mse':bm,'improvement_pct':float(100*(1-np.mean(value)/bm)),
                    'positive_folds':sum(a<b for a,b in zip(fold_scores[name],fold_scores[baseline_name])),
                    'seed_improvements_pct':seed_values,'seed_count':len(seed_values),
                    'path_mse':float(path_sums[name]/n) if name in path_sums else None,
                    'endpoint_mae':float(mae_sums[name]/n) if name in mae_sums else None})
            joined=[{'names':groups[0]['names'],'counts':np.concatenate([g['counts'] for g in groups]),
                     'sums':np.concatenate([g['sums'] for g in groups])}]
            controls=[baseline_name,'default_trained','default_shrink','fit_mean','fit_mean_shrink',
                      'ridge_history','ridge_history_anchor','ridge_history_selected','ridge_history_anchor_selected']
            pairs=[('repair_shrink',name) for name in controls]+[('repair_trained','default_trained')]
            for block in {'annual':(12,24),'quarterly':(3,6),'monthly':(21,42)}[profile]:
                intervals.extend(paired_intervals(joined,pairs,block,task))
            greedy=np.zeros(n,dtype=bool);last=-10**10
            for i,o in enumerate(origins):
                if o>=last+h:greedy[i]=True;last=int(o)
            sensitivity.append({'task':task,'rule':'original_global_earliest_greedy',**summarize(losses,greedy,baseline_name)})
            for offset in range(h):
                sensitivity.append({'task':task,'rule':'absolute_bar_phase','offset':offset,
                                    **summarize(losses,origins%h==offset,baseline_name)})
            dates=data.bars.session_date[origins].astype('U10');end_dates=data.bars.session_date[origins+h].astype('U10')
            for year in sorted({str(d)[:4] for d in dates}):
                mask=np.asarray([str(d)[:4]!=str(year) for d in dates])
                sensitivity.append({'task':task,'rule':'leave_origin_year_out','year':str(year),
                                    **summarize(losses,mask,baseline_name)})
            actions=read(original.SOURCE/'admission.json')['split_actions'];cross=np.zeros(n,dtype=bool)
            for action in actions:cross|=(dates<action['date'])&(end_dates>=action['date'])
            for name,mask in (('crosses_split',cross),('no_split',~cross)):
                sensitivity.append({'task':task,'rule':name,'split_dates':[a['date'] for a in actions],
                                    'distinct_events':sum(bool(np.any((dates<a['date'])&(end_dates>=a['date']))) for a in actions),
                                    **summarize(losses,mask,baseline_name)})
            npz(BASE/'analysis/paired_losses'/f'{task}.npz',origins=origins,fold=np.asarray(all_fold),
                session_date=dates,endpoint_date=end_dates,**{'loss_'+name:value for name,value in losses.items()})
            print(task,'analyzed',flush=True)
    recovery=read(BASE/'audit/RECOVERY_PROBE.json');done=read(BASE/'runs'/recovery['job']/'DONE.json')
    assert done['identity']==recovery['identity'] and done['steps']>=recovery['steps']
    assert sha(BASE/'runs'/recovery['job']/'initial.pt')==recovery['initial_sha256']
    atomic(BASE/'audit/RECOVERY_COMPLETE.json',{'status':'passed','actual_exit_and_resume':True,
           'identity_unchanged':True,'initial_function_unchanged':True,'prefix_steps':2,'final_steps':done['steps'],'at_utc':now()})
    result={'status':'complete_post_observation_longscale_replication','six_tasks':6,'paired_cells':180,
        'new_cells':150,'seeds':[2026,2027,2028],'pooled':pooled,'rows':rows,'paired_intervals':intervals,
        'sensitivity':sensitivity,'original_default_seed_restorations':reproductions,
        'primary_return_gates_unchanged_and_previously_failed':True,
        'independent_confirmation':False,'research_goal_achieved':False,'Q3_values_read':False,'at_utc':now()}
    atomic(BASE/'analysis/RESULTS.json',result)
    for filename,values in (('POOLED.csv',pooled),('ROWS.csv',rows),('INTERVALS.csv',intervals)):
        fields=list(dict.fromkeys(k for row in values for k in row))
        with (BASE/'analysis'/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fields);w.writeheader();w.writerows(values)
    print('Complete paired long-scale analysis saved.',flush=True)


if __name__=='__main__':main()
