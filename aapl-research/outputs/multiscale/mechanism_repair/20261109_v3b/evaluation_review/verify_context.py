"""Independent analytic fit and complete-path replay for the frozen40-cell study.

Prepared before scoring; execute only after root announces frozen full results.
Uses its own feature implementation and fit accumulation order. No training,
source edits, network requests, Q3 access or scoring before a frozen protocol.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
import sys
import time
import numpy as np
import pandas as pd

R=Path(__file__).resolve().parent
B=R.parent
V=B.with_name('20261109_v3')
sys.path.insert(0,str(V/'code'))
from engine import Data
from study import make_job,source_hashes
from multiscale.bars import targets
from legacy_controls import source_controls

COLS={'A_mean':np.arange(0,dtype=int),'B_context':np.arange(9,17),'C_history':np.arange(9),'D_joint':np.arange(17)}

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def near(a,b,msg,rtol=5e-10,atol=3e-13):assert np.allclose(a,b,rtol=rtol,atol=atol),(msg,np.max(np.abs(np.asarray(a)-np.asarray(b))))
def same(a,b,msg):assert np.array_equal(a,b),msg
def stamp(x):return datetime.fromisoformat(x).timestamp()

def reviewer_features(history,completion_ns,h):
    history=np.asarray(history,dtype=np.float64);L=history.shape[1]
    # Derive differences independently rather than assuming the last value0.
    lags=[1,max(2,h//4),max(3,h//2),h,2*h,4*h,L-1]
    assert len(set(lags))==7 and all(1<=k<L for k in lags)
    historical=np.column_stack([history[:,-1]-history[:,-k-1] for k in lags])
    d=np.diff(history,axis=1)
    rms=np.column_stack([np.sqrt(np.mean(d[:,-k:]**2,axis=1)) for k in [h,4*h]])
    ticks=np.asarray(completion_ns,dtype=np.int64)
    assert (ticks%60_000_000_000==0).all()
    clock=pd.to_datetime(ticks,utc=True).tz_convert('America/New_York')
    fraction=(np.asarray(clock.hour)*60+np.asarray(clock.minute)-570)/390
    harmonic=np.column_stack([np.sin(2*np.pi*fraction),np.cos(2*np.pi*fraction),np.sin(4*np.pi*fraction),np.cos(4*np.pi*fraction)])
    weekday=np.column_stack([(np.asarray(clock.dayofweek)==day).astype(float) for day in range(4)])
    return np.concatenate([historical,rms,harmonic,weekday],axis=1)

def predict(model,data,origins,batch=3079):
    result=[]
    for start in range(0,len(origins),batch):
        idx=origins[start:start+batch];x,_=data.windows.arrays(idx)
        f=reviewer_features(x,data.bars.end_ns[idx],data.cfg['horizon'])
        z=(f[:,model['columns']]-model['x_mean'])/model['x_std']
        result.append(model['y_mean']+z@model['coef'])
    return np.concatenate(result)

def fit_from_raw_past(data):
    idx=data.splits['fit'];h=data.cfg['horizon'];n=len(idx)
    sums=np.zeros(17);squares=np.zeros((17,17));cross=np.zeros((17,h));ys=np.zeros(h)
    #4093 differs from the implementation's frozen2048, so receipt arrays are
    #not accepted as fit truth. Every feature and target is reconstructed.
    for start in range(0,n,4093):
        origins=idx[start:start+4093];x,_=data.windows.arrays(origins)
        f=reviewer_features(x,data.bars.end_ns[origins],h)
        y=targets(data.bars,origins,h,'future_log_return')
        sums+=f.sum(0);squares+=f.T@f;cross+=f.T@y;ys+=y.sum(0)
    means=sums/n;scales=np.sqrt(np.maximum(np.diag(squares)/n-means**2,0));scales=np.where(scales>1e-12,scales,1.)
    centered_gram=(squares/n-np.outer(means,means))/np.outer(scales,scales)
    centered_cross=(cross/n-np.outer(means,ys/n))/scales[:,None]
    models={}
    for cell,columns in COLS.items():
        weights=np.linalg.solve(centered_gram[np.ix_(columns,columns)]+.1*np.eye(len(columns)),centered_cross[columns]) if len(columns) else np.empty((0,h))
        models[cell]={'coef':weights,'x_mean':means[columns],'x_std':scales[columns],'y_mean':ys/n,'columns':columns}
    return models,{'sx':sums,'sy':ys,'xx':squares,'xy':cross,'count':n,'cursor':n}

def independently_bootstrap(groups,names,pairs,block):
    rng=np.random.default_rng(2026);sums=np.zeros((5000,len(names)));counts=np.zeros(5000)
    point=np.zeros(len(names));n=0
    for count,total in groups:
        nd=len(count);assert nd>block
        point+=total.sum(0);n+=count.sum()
        for cursor in range(0,5000,100):
            begins=rng.integers(0,nd-block+1,size=(100,(nd+block-1)//block))
            grid=(begins[:,:,None]+np.arange(block)).reshape(100,-1)[:,:nd]
            sums[cursor:cursor+100]+=total[grid].sum(1)
            counts[cursor:cursor+100]+=count[grid].sum(1)
    values=sums/counts[:,None];point/=n;out={}
    for primary,reference in pairs:
        i,j=names.index(primary),names.index(reference);delta=values[:,j]-values[:,i]
        varying=bool(np.any(delta!=delta[0]));ci=np.quantile(delta,[.025,.975]).tolist() if varying else None
        relative=100*(1-values[:,i]/values[:,j])
        out[(primary,reference,block)]={'difference_ci95':ci,'relative_ci95_pct':np.quantile(relative,[.025,.975]).tolist() if varying else None,
            'mse_reference_minus_primary':float(point[j]-point[i]),'relative_improvement_pct':float(100*(1-point[i]/point[j])),
            'supports_improvement':bool(ci and ci[0]>0),'status':'descriptive' if varying else 'degenerate'}
    return out

def main():
    start=time.time()
    assert sorted(os.sched_getaffinity(0))==[24,26] and os.environ['CUDA_VISIBLE_DEVICES']=='-1'
    rules=read(B/'protocol/RULES.json');frozen_epoch=rules['at_epoch']
    assert source_hashes()==rules['preceding_training_sources']==read(V/'protocol/CODE_FREEZE.json')['hashes']
    assert rules['deadline_epoch']==1794205909.6228158
    assert sha(B/'PLAN_ZH.md')==rules['plan_sha256']
    assert all(sha(B/'code'/name)==digest for name,digest in rules['sources'].items())
    assert not rules['Q3_values_read'] and not rules['independent_confirmation']
    assert rules['candidate']=='D_joint_shrink'
    evaluation=read(B/'protocol/EVALUATION_FREEZE.json')
    assert evaluation['evaluator_sha256']==sha(B/'code/evaluate_context.py')
    assert evaluation['portable_stats_sha256']==sha(V/'code/portable_stats.py')
    assert evaluation['rules_sha256']==sha(B/'protocol/RULES.json') and evaluation['at_epoch']>=frozen_epoch
    assert all(sha(Path(path))==digest for path,digest in evaluation['input_hashes'].items())
    outer_results=read(B/'analysis/OUTER_RESULTS.json');statistics=read(B/'analysis/STATISTICS.json')
    assert outer_results['status']=='completed' and outer_results['matrix_cells']==40 and len(outer_results['metrics'])==80
    assert outer_results['statistics_sha256']==sha(B/'analysis/STATISTICS.json')
    spec=importlib.util.spec_from_file_location('frozen_context_predictor',B/'code/predictor.py')
    predictor_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(predictor_module)
    cells=[];independent_metrics=[];loss_paths=[];inner_rows={};source_identities={};path_loss_checks=0;owner_metric_checks=0
    for profile,h in [('minute',60),('hourly',8)]:
        task=f'{profile}_h{h}_future_log_return';inner_rows[task]=[]
        for fold in range(1,6):
            cfg=make_job(profile,h,'future_log_return','xpatch','anchor_baseline',fold)
            data=Data(cfg);assert data.bars.session_date[-1]<='2026-06-30'
            folder=B/'fits'/task/f'fold_{fold}';done=read(folder/'INNER_DONE.json')
            assert done['rules_sha256']==sha(B/'protocol/RULES.json') and done['seed'] is None and done['solver_deterministic']
            assert stamp(done['at_utc'])>=frozen_epoch
            assert stamp(done['at_utc'])<evaluation['at_epoch']
            fold_receipt=read(B/'analysis/folds'/task/f'fold_{fold}.json')
            assert fold_receipt['evaluation_freeze_sha256']==sha(B/'protocol/EVALUATION_FREEZE.json')
            assert evaluation['at_epoch']<=stamp(fold_receipt['at_utc'])<rules['deadline_epoch']
            fit,stop,val=(data.splits[p] for p in ['fit','stop','validation'])
            assert len(fit)==done['fit_count'] and len(stop)==done['stop_count']
            assert data.bars.end_ns[fit[-1]+h]<data.bars.end_ns[stop[0]] and data.bars.end_ns[stop[-1]+h]<data.bars.end_ns[val[0]]
            rebuilt,stats=fit_from_raw_past(data)
            with np.load(folder/'FIT_ACCUM.npz',allow_pickle=False) as a:
                assert sha(folder/'FIT_ACCUM.npz')==done['sufficient_statistics_sha256']
                for key,value in stats.items():near(a[key],value,task+f'fold{fold} sufficient '+key)
            sy=targets(data.bars,stop,h,'future_log_return');y=targets(data.bars,val,h,'future_log_return')
            local,identities=source_controls(task,fold,val,y);source_identities.update(identities)
            local_paths={};alphas={};inner_errors={};verified_paths={}
            pathfile=B/'analysis/path_losses'/task/f'fold_{fold}.npz'
            endpointfile=B/'analysis/paired_losses'/task/f'fold_{fold}.npz'
            assert sha(pathfile)==fold_receipt['path_loss_sha256'] and sha(endpointfile)==fold_receipt['paired_loss_sha256']
            with np.load(pathfile,allow_pickle=False) as a:
                same(a['origins'],val,task+' path loss origins');owner_paths={k:a[k].copy() for k in a.files if k!='origins'}
            assert len(owner_paths)==8
            for cell in COLS:
                model_file=folder/(cell+'.npz');assert sha(model_file)==done['model_hashes'][cell+'.npz']
                with np.load(model_file,allow_pickle=False) as a:model={key:a[key].copy() for key in a.files}
                for key in rebuilt[cell]:near(model[key],rebuilt[cell][key],task+f'fold{fold} '+cell+' '+key)
                assert int(model['horizon'])==h
                # Full stopping and outer functions are rebuilt from our own
                #analytic solution. Compare against frozen portable predictions.
                sp=predict(rebuilt[cell],data,stop);vp=predict(rebuilt[cell],data,val)
                frozen=predictor_module.Predictor(model_file)
                x,_=data.windows.arrays(val[:257])
                near(frozen.predict(x,data.bars.end_ns[val[:257]]),vp[:257],cell+' portable replay')
                num=float(sp[:,-1]@sy[:,-1]);den=float(sp[:,-1]@sp[:,-1]);alpha=float(np.clip(num/den,0,1)) if den else 0.
                near(alpha,float(model['alpha']),cell+' alpha',atol=1e-11)
                im=done['inner_metrics'][cell];near(np.mean((sp[:,-1]-sy[:,-1])**2),im['raw_endpoint_mse'],cell+' inner raw')
                near(np.mean((alpha*sp[:,-1]-sy[:,-1])**2),im['shrink_endpoint_mse'],cell+' inner shrink')
                near(np.mean((sp-sy)**2),im['raw_path_mse'],cell+' inner path')
                inner_errors[cell]=float(np.mean((sp[:,-1]-sy[:,-1])**2))
                for mode,p in [('raw',vp),('shrink',alpha*vp)]:
                    name=cell+'_'+mode;local[name]=(p[:,-1]-y[:,-1])**2;local_paths[name]=p
                    path_loss=np.mean((p-y)**2,axis=1);verified_paths[name]=path_loss
                    near(path_loss,owner_paths[name],name+' every origin fullpath loss')
                    path_loss_checks+=1
                    metric={'task':task,'fold':fold,'cell':cell,'mode':mode,'origins':len(val),'endpoint_mse':float(local[name].mean()),'endpoint_mae':float(np.mean(np.abs(p[:,-1]-y[:,-1]))),'path_mse':float(np.mean((p-y)**2)),'baseline_mse':float(local['zero'].mean()),'improvement_pct':float(100*(1-local[name].mean()/local['zero'].mean()))}
                    independent_metrics.append(metric)
                    owner=next(row for row in outer_results['metrics'] if row['task']==task and row['fold']==fold and row['method']==name)
                    for k in ('origins','endpoint_mse','endpoint_mae','path_mse','baseline_mse','improvement_pct'):near(metric[k],owner[k],name+' owner metric '+k)
                    assert owner['training_seed'] is None and owner['deterministic_solver']
                    owner_metric_checks+=1
                delivery=B/'delivery'/task/f'fold_{fold}'/(cell+'.npz')
                with np.load(delivery,allow_pickle=False) as a:
                    near(a['expected_raw'],vp[:len(a['origins'])],cell+' delivery raw')
                    near(a['expected_shrink'],alpha*vp[:len(a['origins'])],cell+' delivery shrink')
                alphas[cell]=alpha
                cells.append({'task':task,'fold':fold,'cell':cell,'independent_analytic_fit':True,'fit_accumulation_batch':4093,'coefficient_and_standardization_match':True,'inner_only_alpha_match':True,'outer_paths_rebuilt':True})
            with np.load(endpointfile,allow_pickle=False) as a:
                assert set(a.files)-{'origins','session_date','day_grid'}==set(local) and len(local)==33
                same(a['origins'],val,'all endpoint origins');same(a['session_date'],data.bars.session_date[val],'all session dates')
                dates=data.bars.session_date[val];grid=np.unique(data.bars.session_date[(data.bars.session_date>=dates[0])&(data.bars.session_date<=dates[-1])])
                same(a['day_grid'],grid,'full session grid including zero-origin days')
                for name,value in local.items():near(value,a[name],name+' every endpoint loss')
            out=R/'verified_losses'/task;out.mkdir(parents=True,exist_ok=True)
            dest=out/f'fold_{fold}.npz';np.savez_compressed(dest,origins=val,session_date=data.bars.session_date[val],day_grid=grid,**local)
            np.savez_compressed(out/f'fold_{fold}_path.npz',origins=val,**verified_paths)
            loss_paths.append(str(dest.relative_to(R)))
            inner_rows[task].append({'fold':fold,'gate_raw_D_lt_zero_and_C':inner_errors['D_joint']<np.mean(sy[:,-1]**2) and inner_errors['D_joint']<inner_errors['C_history'],'alphas':alphas})
            print(json.dumps({'verified_task':task,'fold':fold,'elapsed_seconds':round(time.time()-start,2)}),flush=True)
    assert len(cells)==40 and len(independent_metrics)==80
    receipt={'status':'analytic_fits_and_paths_passed_full_statistics_pending','cells':cells,'metrics':independent_metrics,'inner_gates':inner_rows,'verified_loss_files':loss_paths,'per_origin_path_loss_checks':path_loss_checks,'owner_metric_rows_checked':owner_metric_checks,'source_control_hashes':source_identities,'rules_sha256':sha(B/'protocol/RULES.json'),'evaluation_freeze_sha256':sha(B/'protocol/EVALUATION_FREEZE.json'),'source_sha256':{name:sha(B/'code'/name) for name in rules['sources']},'reviewer_sha256':sha(Path(__file__)),'elapsed_seconds':time.time()-start,'Q3_values_read':False,'independent_confirmation':False,'research_goal_achieved':False,'at_utc':datetime.now(timezone.utc).isoformat()}
    (R/'FIT_AND_PATH_REVIEW.json').write_text(json.dumps(receipt,indent=2,default=lambda x:bool(x) if isinstance(x,np.bool_) else float(x))+'\n')
    print(json.dumps({'status':receipt['status'],'cells':len(cells),'metrics':len(independent_metrics),'elapsed_seconds':receipt['elapsed_seconds']}),flush=True)

if __name__=='__main__':main()
