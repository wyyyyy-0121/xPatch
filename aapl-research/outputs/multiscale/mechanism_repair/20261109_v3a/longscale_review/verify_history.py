"""Replay fixed-weight history counterfactuals without using the probe engine."""
from pathlib import Path
import os,sys,traceback
import numpy as np
import torch

BASE=Path(__file__).resolve().parent
PHASE=BASE.parent;PROBE=PHASE/'history_sensitivity';LONG=PHASE/'longscale_replication';V3=PHASE.with_name('20261109_v3')
sys.path.insert(0,str(BASE));sys.path.insert(0,str(V3/'code'))
from study import PRIOR,SOURCE,read,sha,atomic,now,source_hashes
from portable_predictor import Predictor
from portable_stats import day_groups,paired_intervals
from multiscale.matrix_executor import read_bars
from multiscale.features import CloseWindows
from multiscale.bars import targets
from verify import equal,close


def main():
    assert set(os.sched_getaffinity(0)).issubset({24,26}) and os.environ['CUDA_VISIBLE_DEVICES']=='-1'
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    rules=read(PROBE/'protocol/RULES.json');result=read(PROBE/'analysis/RESULTS.json')
    assert sha(PROBE/'code/history_probe.py')==rules['source_sha256']
    assert sha(PROBE/'protocol/RULES.json')==result['rules_sha256']
    manifest=read(LONG/'protocol/JOBS.json');assert sha(LONG/'protocol/JOBS.json')==rules['original_jobs_sha256']
    assert rules['frozen_training_hashes']==source_hashes()
    assert not result['independent_confirmation'] and not result['research_goal_achieved'] and not result['Q3_values_read']
    configs=[j for j in manifest['jobs'] if j['initialization']=='baseline'];assert len(configs)==90
    checked=[];pooled_checks=[];interval_checks=[];sensitivity_checks=[];cache={};zero_alpha=0
    admission=read(SOURCE/'admission.json')
    for profile,h in [('annual',12),('quarterly',3),('monthly',21)]:
        for target in ['future_close','future_log_return']:
            task=f'{profile}_h{h}_{target}';cfgs=[j for j in configs if j['profile']==profile and j['target']==target]
            meta=read(PRIOR/'cache'/task/'TASK.json');freq=meta['frequency']
            if freq not in cache:cache[freq]=read_bars(SOURCE,admission,freq)
            bars=cache[freq];assert bars.session_date[-1]<='2026-06-30'
            windows=CloseWindows(bars,meta['lookback'],admission['split_actions'])
            all_losses={};origin_chunks=[];groups=[]
            for fold in range(1,6):
                rec=meta['folds'][str(fold)]
                with np.load(rec['split_file'],allow_pickle=False) as a:fit=a['fit'].copy();stop=a['stop'].copy();val=a['validation'].copy()
                assert bars.end_ns[fit[-1]+h]<bars.end_ns[stop[0]] and bars.end_ns[stop[-1]+h]<bars.end_ns[val[0]]
                direct_sum=np.zeros(meta['lookback'])
                for start in range(0,len(fit),127):
                    x,_=windows.arrays(fit[start:start+127]);direct_sum+=x.sum(0)
                mean=direct_sum/len(fit)
                sample=PROBE/'analysis'/task/f'fold_{fold}.npz'
                with np.load(sample,allow_pickle=False) as a:saved={k:a[k].copy() for k in a.files}
                equal(saved['origins'],val,'history origins')
                assert np.allclose(mean,saved['fit_mean_history'],rtol=1e-12,atol=1e-12)
                # Use the independently verified saved mean for exact float32 replay.
                mean=saved['fit_mean_history']
                y=targets(bars,val,h,target);equal(saved['actual'],y,'history labels')
                baseline=np.zeros_like(y) if target=='future_log_return' else np.broadcast_to(bars.close[val,None],y.shape)
                equal(saved['baseline'],baseline,'history baseline')
                local={}
                for cfg in [c for c in cfgs if c['fold']==fold]:
                    src=Path(manifest['locations'][cfg['id']]);done=read(src/'DONE.json')
                    assert sha(src/'best_trained.pt')==done['checkpoint_sha256']
                    p=Predictor(src/'best_trained.pt');assert p.cfg==cfg
                    cal=read(LONG/'calibration'/cfg['id']/'DONE.json');alpha=cal['alpha']
                    assert cal['source_checkpoint_sha256']==done['checkpoint_sha256'] and cal['inner_stop_only']
                    if alpha==0:zero_alpha+=1
                    chunks={name:[] for name in ['actual','fit_mean_history','flat_history']}
                    for start in range(0,len(val),128):
                        idx=val[start:start+128];x,anchor=windows.arrays(idx)
                        for name,history in [('actual',x),('fit_mean_history',np.broadcast_to(mean,x.shape)),('flat_history',np.zeros_like(x))]:
                            # Same current-close anchor is passed to all interventions.
                            chunks[name].append(p.predict_prepared(history,anchor))
                    raw={name:np.concatenate(values) for name,values in chunks.items()}
                    with np.load(src/'predictions.npz',allow_pickle=False) as a:equal(raw['actual'],a['trained'],'true history replay')
                    predictions={name:baseline+alpha*(v-baseline) for name,v in raw.items()}
                    row=next(r for r in result['rows'] if r['task']==task and r['fold']==fold and r['seed']==cfg['seed'])
                    assert row['alpha']==alpha and row['fit_windows']==len(fit) and row['origins']==len(val)
                    for name,pred in predictions.items():
                        equal(saved[f'{name}_seed_{cfg["seed"]}'],pred,cfg['id']+' '+name)
                        errors=(pred[:,-1]-y[:,-1])**2;local.setdefault(name,[]).append(errors)
                        change=pred[:,-1]-baseline[:,-1];den=np.mean((baseline[:,-1]-y[:,-1])**2)
                        measures={'endpoint_mse':float(errors.mean()),'improvement_pct':float(100*(1-errors.mean()/den)),
                                  'forecast_change_mean':float(change.mean()),'forecast_change_std':float(change.std())}
                        for key,value in measures.items():close(row['metrics'][name][key],value,'history row '+key)
                    diff=predictions['actual'][:,-1]-predictions['fit_mean_history'][:,-1]
                    close(row['history_effect_rms'],float(np.sqrt(np.mean(diff**2))),'history RMS')
                    close(row['history_effect_mean'],float(diff.mean()),'history mean')
                    assert row['actual_replay_exact'] and row['calibrated_replay_exact']
                    checked.append({'job':cfg['id'],'all_three_history_interventions_exact':True,'fit_only_mean_verified':True,
                                    'same_anchor_parameters_scalers_original_alpha':True,'alpha':alpha})
                losses={name:np.mean(values,0) for name,values in local.items()};losses['baseline']=(baseline[:,-1]-y[:,-1])**2
                for name,value in losses.items():all_losses.setdefault(name,[]).append(value)
                groups.append(day_groups(bars,val,losses));origin_chunks.append(val)
            origins=np.concatenate(origin_chunks);losses={name:np.concatenate(values) for name,values in all_losses.items()};bm=float(losses['baseline'].mean())
            for name,value in losses.items():
                row=next(r for r in result['pooled'] if r['task']==task and r['method']==name)
                assert row['origins']==len(origins);close(row['endpoint_mse'],float(value.mean()),'history pooled MSE')
                close(row['improvement_over_baseline_pct'],float(100*(1-value.mean()/bm)),'history pooled gain')
                pooled_checks.append({'task':task,'method':name,'verified':True})
            merged=[{'names':groups[0]['names'],'counts':np.concatenate([g['counts'] for g in groups]),'sums':np.concatenate([g['sums'] for g in groups])}]
            for block in {'annual':[12,24],'quarterly':[3,6],'monthly':[21,42]}[profile]:
                for row in paired_intervals(merged,[('actual','fit_mean_history'),('actual','flat_history')],block,task):
                    saved_row=next(r for r in result['paired_intervals'] if r['task']==task and r['block']==block and r['reference']==row['reference'])
                    assert row==saved_row;interval_checks.append({'task':task,'block':block,'reference':row['reference'],'verified':True})
            keep=np.zeros(len(origins),dtype=bool);last=None
            for i,o in enumerate(origins):
                if last is None or o>=last+h:keep[i]=True;last=int(o)
            row=next(r for r in result['nonoverlap_sensitivity'] if r['task']==task)
            assert row['origins']==int(keep.sum());bm=float(losses['baseline'][keep].mean())
            for name,value in losses.items():close(row['improvement_over_baseline_pct'][name],float(100*(1-value[keep].mean()/bm)),'history nonoverlap gain')
            sensitivity_checks.append({'task':task,'origins':int(keep.sum()),'verified':True})
            print(task,'history replay verified',flush=True)
    assert len(checked)==90 and len(pooled_checks)==24 and len(interval_checks)==24 and len(sensitivity_checks)==6
    report={'status':'passed','configs':90,'fixed_weight_interventions_per_config':3,'all_saved_paths_exact':True,
            'fit_mean_recomputed_different_batch_order':True,'fit_stop_outer_purging_verified':True,
            'pooled_rows':24,'paired_intervals':24,'nonoverlap_sensitivities':6,
            'cell_checks':checked,'pooled_checks':pooled_checks,'interval_checks':interval_checks,'sensitivity_checks':sensitivity_checks,
            'zero_alpha_configs':zero_alpha,
            'causal_wording_review':{'acceptable':'Post-observation, unrefitted fixed-weight input counterfactual; both substitutions may be out of distribution. It identifies sensitivity of an archived prediction function under those substitutions.',
                                   'unsupported':'A causal market mechanism, proof of learned sequence signal, or a newly developed deployable method selected from outer scores.',
                                   'additional_limit':'All reported sensitivity is after the original shrinkage. Alpha zero masks raw network history sensitivity; invariance does not prove that the unshrunk network ignores history.'},
            'source_results_sha256':sha(PROBE/'analysis/RESULTS.json'),'rules_sha256':sha(PROBE/'protocol/RULES.json'),
            'verifier_sha256':sha(__file__),'review_affinity':sorted(os.sched_getaffinity(0)),
            'Q3_values_read':False,'research_goal_achieved':False,'at_utc':now()}
    atomic(BASE/'HISTORY_REVIEW.json',report)
    print({'status':'passed','configs':90,'interventions':270,'zero_alpha_configs':zero_alpha},flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        atomic(BASE/'HISTORY_FAILURE.json',{'status':'failed','traceback':traceback.format_exc(),'at_utc':now()})
        raise
