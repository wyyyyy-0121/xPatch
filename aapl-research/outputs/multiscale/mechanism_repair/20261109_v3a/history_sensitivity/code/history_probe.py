"""Post-observation, fixed-weight history-window counterfactuals on six long-scale tasks."""
from pathlib import Path
import os, sys, hashlib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
V3A = ROOT.parent
V3 = V3A.parent/'20261109_v3'
sys.path.insert(0, str(V3/'code'))
from study import read, sha, now, atomic, npz, source_hashes, PRIOR
from engine import Data
from networks import create
from portable_stats import day_groups, paired_intervals


def metrics(y, predictions, baseline):
    actual=y[:,-1]; b=baseline[:,-1]
    denominator=float(np.mean((b-actual)**2))
    return {name:{'endpoint_mse':float(np.mean((p[:,-1]-actual)**2)),
                  'improvement_pct':float(100*(1-np.mean((p[:,-1]-actual)**2)/denominator)),
                  'forecast_change_mean':float(np.mean(p[:,-1]-b)),
                  'forecast_change_std':float(np.std(p[:,-1]-b))}
            for name,p in predictions.items()}


def main():
    assert len(os.sched_getaffinity(0))<=2 and os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    manifest=read(V3A/'longscale_replication/protocol/JOBS.json')
    assert source_hashes()==manifest['training_hashes']
    configs=[j for j in manifest['jobs'] if j['initialization']=='baseline']
    assert len(configs)==90
    protocol=ROOT/'protocol/RULES.json'
    rules={'study':'post-observation fixed-weight input sensitivity; not independent validation or refitted ablation',
        'source_sha256':sha(__file__),'original_jobs_sha256':sha(V3A/'longscale_replication/protocol/JOBS.json'),
        'frozen_training_hashes':manifest['training_hashes'],
        'tasks':'annual H12, quarterly H3, monthly H21; both original targets, five folds, three seeds',
        'comparison':['actual past-only log-relative close window',
                      'pointwise mean of purged inner-fit log-relative windows for the fold',
                      'flat history whose log-relative close is zero at every lag'],
        'kept_fixed':'same saved parameters, anchor at each origin, input/target scalers, stop-fitted scalar, target and evaluation origins',
        'history_mean':'mean from inner fit inputs only, never fit/stop/validation future targets',
        'calibration':'use original inner-stop alpha even on substituted inputs, to isolate input changes; not re-optimized',
        'pooled':'per-origin mean of the three seed losses, not loss of average prediction',
        'intervals':'same exposed-development paired blocks as v3; descriptive, no multiplicity correction',
        'limits':'fit-mean and flat windows can be out of distribution; unrefitted counterfactual, not a market causal mechanism or a deployable method',
        'Q3_values_read':False,'independent_confirmation':False,'at_utc':now()}
    if protocol.exists():
        old=read(protocol)
        assert all(old[k]==v for k,v in rules.items() if k!='at_utc')
    else:atomic(protocol,rules)
    rows=[]; pooled=[]; intervals=[]; sensitivities=[]
    for profile,h in (('annual',12),('quarterly',3),('monthly',21)):
        for target in ('future_close','future_log_return'):
            task=f'{profile}_h{h}_{target}';selected=[c for c in configs if c['profile']==profile and c['target']==target]
            all_origins=[];all_losses={};groups=[]
            for fold in range(1,6):
                fold_cfgs=[c for c in selected if c['fold']==fold]
                data=Data(fold_cfgs[0]);fit=data.splits['fit'];validation=data.splits['validation']
                # Only fit-period inputs are used in constructing this counterfactual window.
                sum_x=np.zeros(data.cfg['lookback'],dtype=np.float64)
                for i in range(0,len(fit),4096):
                    x,_=data.windows.arrays(fit[i:i+4096]);sum_x+=x.sum(axis=0,dtype=np.float64)
                mean_x=sum_x/len(fit)
                rec=read(PRIOR/'cache'/task/'TASK.json')['folds'][str(fold)]
                assert sha(rec['targets'])==rec['targets_sha256']
                with np.load(rec['targets'],allow_pickle=False) as a:
                    origins=a['origins'];y=a['actual'];anchor=a['anchor']
                assert np.array_equal(origins,validation)
                baseline=data.baseline(origins);local={};seed_pred=[]
                for cfg in fold_cfgs:
                    src=Path(manifest['locations'][cfg['id']]);done=read(src/'DONE.json')
                    assert done['status']=='completed' and sha(src/'best_trained.pt')==done['checkpoint_sha256']
                    saved=torch.load(src/'best_trained.pt',map_location='cpu',weights_only=True)
                    model=create(cfg,data.scaler);model.load_state_dict(saved['model']);model.eval()
                    preds={name:[] for name in ('actual','fit_mean_history','flat_history')}
                    with torch.inference_mode():
                        for start in range(0,len(origins),128):
                            pos=np.arange(start,min(start+128,len(origins)));idx=origins[pos]
                            x,a=data.windows.arrays(idx)
                            a=((a-data.scaler['anchor_mean'])/data.scaler['anchor_std']).astype('float32')
                            for name,source in (('actual',x),('fit_mean_history',np.broadcast_to(mean_x,x.shape)),
                                                ('flat_history',np.zeros_like(x))):
                                scaled=((source-data.scaler['x_mean'])/data.scaler['x_std']).astype('float32')
                                p=model(torch.from_numpy(scaled[:,:,None]),torch.from_numpy(a)).numpy()
                                preds[name].append(data.invert(p,idx))
                    preds={name:np.concatenate(values) for name,values in preds.items()}
                    with np.load(src/'predictions.npz',allow_pickle=False) as a:
                        assert np.array_equal(a['origins'],origins) and np.array_equal(a['trained'],preds['actual'])
                    cal=read(V3A/'longscale_replication/calibration'/cfg['id']/'DONE.json');alpha=cal['alpha']
                    predictions={name:baseline+alpha*(p-baseline) for name,p in preds.items()}
                    calfile=V3A/'longscale_replication/calibration'/cfg['id']/'predictions.npz'
                    assert sha(calfile)==cal['prediction_sha256']
                    with np.load(calfile,allow_pickle=False) as a:assert np.array_equal(a['prediction'],predictions['actual'])
                    seed_pred.append({name:value for name,value in predictions.items()})
                    for name,p in predictions.items():local.setdefault(name,[]).append((p[:,-1]-y[:,-1])**2)
                    diff=predictions['actual'][:,-1]-predictions['fit_mean_history'][:,-1]
                    rows.append({'task':task,'fold':fold,'seed':cfg['seed'],'alpha':alpha,
                        'fit_windows':len(fit),'origins':len(origins),'metrics':metrics(y,predictions,baseline),
                        'history_effect_rms':float(np.sqrt(np.mean(diff**2))),
                        'history_effect_mean':float(np.mean(diff)),
                        'actual_replay_exact':True,'calibrated_replay_exact':True})
                losses={name:np.mean(vals,axis=0) for name,vals in local.items()}
                losses['baseline']=(baseline[:,-1]-y[:,-1])**2
                for name,value in losses.items():all_losses.setdefault(name,[]).append(value)
                groups.append(day_groups(data.bars,origins,losses));all_origins.append(origins)
                npz(ROOT/'analysis'/task/f'fold_{fold}.npz',origins=origins,fit_mean_history=mean_x,
                    baseline=baseline,actual=y,**{f'{name}_seed_{cfg["seed"]}':seed_pred[i][name]
                    for i,cfg in enumerate(fold_cfgs) for name in ('actual','fit_mean_history','flat_history')})
            origins=np.concatenate(all_origins);losses={k:np.concatenate(v) for k,v in all_losses.items()}
            baseline_mse=float(np.mean(losses['baseline']))
            for name,value in losses.items():
                pooled.append({'task':task,'method':name,'origins':len(origins),'endpoint_mse':float(np.mean(value)),
                    'improvement_over_baseline_pct':float(100*(1-np.mean(value)/baseline_mse))})
            merged=[{'names':groups[0]['names'],'counts':np.concatenate([g['counts'] for g in groups]),
                     'sums':np.concatenate([g['sums'] for g in groups])}]
            for block in {'annual':(12,24),'quarterly':(3,6),'monthly':(21,42)}[profile]:
                intervals.extend(paired_intervals(merged,[('actual','fit_mean_history'),('actual','flat_history')],block,task))
            keep=np.zeros(len(origins),dtype=bool);last=-10**10
            for i,o in enumerate(origins):
                if o>=last+h:keep[i]=True;last=int(o)
            d=float(np.mean(losses['baseline'][keep]))
            sensitivities.append({'task':task,'rule':'earliest_origin_greedy','origins':int(keep.sum()),
                'improvement_over_baseline_pct':{name:float(100*(1-np.mean(v[keep])/d)) for name,v in losses.items()}})
            print(task,'done',flush=True)
    atomic(ROOT/'analysis/RESULTS.json',{'status':'complete','rows':rows,'pooled':pooled,
        'paired_intervals':intervals,'nonoverlap_sensitivity':sensitivities,
        'limits':rules['limits'],'research_goal_achieved':False,'independent_confirmation':False,
        'Q3_values_read':False,'rules_sha256':sha(protocol),'at_utc':now()})


if __name__=='__main__':main()
