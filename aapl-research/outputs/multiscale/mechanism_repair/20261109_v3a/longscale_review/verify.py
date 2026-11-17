"""Independent inference and arithmetic acceptance of the six-task replication.

Writes only longscale_review; no training or original study modification.
"""
from pathlib import Path
from datetime import datetime
import ast, json, os, sys, traceback
import numpy as np
import torch

BASE=Path(__file__).resolve().parent
PHASE=BASE.parent
STUDY=PHASE/'longscale_replication'
V3=PHASE.with_name('20261109_v3')
sys.path.insert(0,str(V3/'code'))
from study import PRIOR,SOURCE,read,sha,atomic,now,source_hashes
from portable_predictor import Predictor
from portable_stats import day_groups,paired_intervals
from multiscale.matrix_executor import read_bars
from multiscale.features import CloseWindows
from multiscale.bars import targets


def equal(a,b,message):
    assert np.array_equal(a,b),message+': '+str(float(np.max(np.abs(np.asarray(a)-np.asarray(b)))))


def close(a,b,message):
    assert np.isclose(a,b,rtol=2e-13,atol=2e-14),message+': '+str((a,b))


def metric(pred,y,baseline):
    e=pred-y;bm=float(np.mean((baseline[:,-1]-y[:,-1])**2));m=float(np.mean(e[:,-1]**2))
    return {'endpoint_mse':m,'endpoint_mae':float(np.mean(abs(e[:,-1]))),
            'path_mse':float(np.mean(e**2)),'baseline_mse':bm,
            'improvement_pct':float(100*(1-m/bm))}


def infer(predictor,windows,origins,batch):
    chunks=[]
    for start in range(0,len(origins),batch):
        idx=origins[start:start+batch]
        # Predictor receives only past relative histories and current closes.
        history,anchor=windows.arrays(idx)
        chunks.append(predictor.predict_prepared(history,anchor))
    return np.concatenate(chunks)


def main():
    assert set(os.sched_getaffinity(0)).issubset({24,26}) and os.environ['CUDA_VISIBLE_DEVICES']=='-1'
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    manifest=read(STUDY/'protocol/JOBS.json');result=read(STUDY/'analysis/RESULTS.json')
    source=source_hashes();assert source==manifest['training_hashes']==read(V3/'protocol/CODE_FREEZE.json')['hashes']
    for name,digest in manifest['wrapper_hashes'].items():assert sha(STUDY/'code'/name)==digest,name
    for name in ('ANALYSIS_FREEZE.json','CANDIDATE_ANALYSIS_FREEZE.json'):
        assert all(sha(V3/'code'/n)==d for n,d in read(V3/'protocol'/name)['files'].items())
    assert manifest['deadline_epoch']==read(V3/'protocol/BUDGET.json')['deadline_epoch']==1794205909.6228158
    assert sha(V3/'protocol/BUDGET.json')==manifest['budget_sha256']
    assert read(STUDY/'audit/COMPLETE.json')['cells']==180
    assert read(STUDY/'audit/CALIBRATION_COMPLETE.json')['cells']==180
    freeze=read(STUDY/'protocol/ANALYSIS_FREEZE.json')
    assert sha(STUDY/'code/analyze.py')==freeze['source_sha256']
    assert sha(V3/'code/portable_stats.py')==freeze['portable_statistics_sha256']
    calibration_rules=read(STUDY/'protocol/CALIBRATION.json')['rules']
    assert sha(STUDY/'code/calibrate.py')==calibration_rules['code_sha256']
    assert sha(STUDY/'protocol/JOBS.json')==calibration_rules['jobs_sha256']
    assert not result['independent_confirmation'] and not result['research_goal_achieved'] and not result['Q3_values_read']
    jobs=manifest['jobs'];assert len(jobs)==180 and len({j['id'] for j in jobs})==180
    admitted=datetime.fromisoformat(manifest['at_utc']).timestamp()
    source_data=read(PRIOR/'protocol/DATA.json');admission=read(SOURCE/'admission.json')
    cell_checks=[];restorations=[];pooled_checks=[];interval_checks=[];sensitivity_checks=[]
    origins_checked=0;recomputed={};bar_cache={};latest_fit=0.
    for task in sorted({f"{j['profile']}_h{j['horizon']}_{j['target']}" for j in jobs}):
        tcfg=[j for j in jobs if f"{j['profile']}_h{j['horizon']}_{j['target']}"==task]
        meta_path=PRIOR/'cache'/task/'TASK.json';assert sha(meta_path)==source_data['task_files'][task]
        meta=read(meta_path);h=meta['horizon'];target=meta['target'];profile=meta['profile'];freq=meta['frequency']
        if freq not in bar_cache:bar_cache[freq]=read_bars(SOURCE,admission,freq)
        bars=bar_cache[freq];assert bars.session_date[-1]<='2026-06-30'
        windows=CloseWindows(bars,meta['lookback'],admission['split_actions'])
        names=[];losses_folds={};seed_losses={};path_totals={};mae_totals={};fold_mses={};group_list=[];origin_chunks=[];fold_labels=[]
        base_name='zero' if target=='future_log_return' else 'persistence'
        for fold in range(1,6):
            rec=meta['folds'][str(fold)]
            with np.load(rec['split_file'],allow_pickle=False) as a:splits={k:a[k].copy() for k in a.files}
            for part,values in splits.items():
                import hashlib
                assert hashlib.sha256(np.asarray(values,dtype='<i8').tobytes()).hexdigest()==rec['split_sha256'][part]
            fit,stop,val=(splits[k] for k in ('fit','stop','validation'))
            assert bars.end_ns[fit[-1]+h]<bars.end_ns[stop[0]]
            assert bars.end_ns[stop[-1]+h]<bars.end_ns[val[0]]
            assert sha(rec['targets'])==rec['targets_sha256']
            with np.load(rec['targets'],allow_pickle=False) as a:
                y=a['actual'].copy();origins=a['origins'].copy();anchor=a['anchor'].copy()
            equal(origins,val,'saved validation origins')
            equal(y,targets(bars,val,h,target),'saved original labels')
            baseline=np.zeros_like(y) if target=='future_log_return' else np.broadcast_to(anchor[:,None],y.shape)
            local={base_name:(baseline[:,-1]-y[:,-1])**2};neural={}
            for cfg in [j for j in tcfg if j['fold']==fold]:
                src=Path(manifest['locations'][cfg['id']]);done=read(src/'DONE.json')
                assert done['job']==cfg and done['status']=='completed'
                assert done['deadline_epoch']==manifest['deadline_epoch'] and len(done['affinity'])<=2
                assert not done['Q3_values_read']
                completed=datetime.fromisoformat(done['at_utc']).timestamp();latest_fit=max(latest_fit,completed)
                assert completed<manifest['deadline_epoch']
                if src.is_relative_to(STUDY):assert completed>admitted
                assert sha(src/'best_trained.pt')==done['checkpoint_sha256']
                assert sha(src/'predictions.npz')==done['prediction_sha256']
                p=Predictor(src/'best_trained.pt');assert p.cfg==cfg
                trained=infer(p,windows,val,128)
                with np.load(src/'predictions.npz',allow_pickle=False) as a:
                    equal(a['origins'],val,cfg['id']+' origins');equal(trained,a['trained'],cfg['id']+' trained replay')
                # Exact batch128 stop inference followed by fixed calibration arithmetic.
                stopped=infer(p,windows,stop,128)
                sy=targets(bars,stop,h,target)
                sb=np.zeros_like(sy) if target=='future_log_return' else np.broadcast_to(bars.close[stop,None],sy.shape)
                num=den=0.
                for start in range(0,len(stop),128):
                    sl=slice(start,start+128);q=stopped[sl,-1]-sb[sl,-1];r=sy[sl,-1]-sb[sl,-1]
                    num+=float(np.dot(q,r));den+=float(np.dot(q,q))
                alpha=float(np.clip(num/den,0,1)) if den else 0.
                cal_dir=STUDY/'calibration'/cfg['id'];cal=read(cal_dir/'DONE.json')
                assert alpha==cal['alpha'],cfg['id']+' alpha'
                assert cal['source_checkpoint_sha256']==done['checkpoint_sha256']
                assert cal['inner_stop_only'] and not cal['Q3_values_read']
                calibrated=baseline+alpha*(trained-baseline)
                assert sha(cal_dir/'predictions.npz')==cal['prediction_sha256']
                with np.load(cal_dir/'predictions.npz',allow_pickle=False) as a:
                    equal(val,a['origins'],cfg['id']+' calibrated origins');equal(calibrated,a['prediction'],cfg['id']+' calibrated replay')
                kind='repair' if cfg['initialization']=='baseline' else 'default'
                for mode,pred in [('trained',trained),('shrink',calibrated)]:
                    name=kind+'_'+mode;error=(pred[:,-1]-y[:,-1])**2;neural.setdefault(name,[]).append(error)
                    seed_losses.setdefault(name,{}).setdefault(cfg['seed'],[]).append(error)
                    m=metric(pred,y,baseline)
                    row=next(r for r in result['rows'] if r['task']==task and r['fold']==fold and r['seed']==cfg['seed'] and r['method']==name)
                    for key,value in m.items():close(row[key],value,cfg['id']+' row '+key)
                    assert row['alpha']==alpha and row['epochs']==done['epochs'] and row['best_epoch']==done['trained_best_epoch']
                    path_totals[name]=path_totals.get(name,0.)+m['path_mse']*len(val)/3
                    mae_totals[name]=mae_totals.get(name,0.)+m['endpoint_mae']*len(val)/3
                if cfg['seed']==2026 and cfg['initialization']=='random':
                    old=rec['models']['xpatch'];assert sha(old['checkpoint'])==old['checkpoint_sha256']
                    old_ck=torch.load(old['checkpoint'],map_location='cpu',weights_only=True)
                    new_ck=torch.load(src/'best_trained.pt',map_location='cpu',weights_only=True)
                    assert set(old_ck['model'])==set(new_ck['model'])
                    assert all(torch.equal(old_ck['model'][k],new_ck['model'][k]) for k in old_ck['model'])
                    assert old_ck['epoch']==new_ck['epoch']==done['trained_best_epoch']
                    summary=read(old['summary']);assert sha(old['summary'])==old['summary_sha256']
                    assert summary['epochs']==done['epochs'] and summary['best_epoch']==done['trained_best_epoch']
                    assert sha(old['prediction'])==old['prediction_sha256']
                    with np.load(old['prediction'],allow_pickle=False) as a:old_pred=a['prediction'].copy();equal(a['origins'],val,'original origins')
                    original_batch=infer(p,windows,val,512)
                    equal(original_batch,old_pred,cfg['id']+' archived512batch replay')
                    exact128=np.array_equal(trained,old_pred)
                    assert exact128 if profile in ('annual','quarterly') else not exact128
                    claimed=next(r for r in result['original_default_seed_restorations'] if r['task']==task and r['fold']==fold)
                    assert claimed['prediction_exact']==exact128 and claimed['weights_exact'] and claimed['best_epoch_exact']
                    restorations.append({'task':task,'fold':fold,'weights_exact':True,'best_and_stop_epochs_exact':True,
                                         'same128batch_equals_archive':exact128,'archived512batch_replay_exact':True,
                                         'max128versus512difference':float(abs(trained-old_pred).max())})
                origins_checked+=len(val)
                cell_checks.append({'job':cfg['id'],'trained_and_shrink_batch128_exact':True,'alpha_exact_from_inner_stop':True,
                                    'fit_stop_outer_purging_checked':True,'checkpoint_and_prediction_hashes_match':True,
                                    'deadline_unchanged_and_completion_before_deadline':True,'affinity':done['affinity']})
            assert all(len(v)==3 for v in neural.values())
            local.update({name:np.mean(values,axis=0) for name,values in neural.items()})
            cdir=V3/'controls_cross_scale'/task/f'fold_{fold}';c=read(cdir/'DONE.json')
            assert sha(cdir/'predictions.npz')==c['prediction_sha256']
            with np.load(cdir/'predictions.npz',allow_pickle=False) as a:
                equal(a['origins'],val,'Ridge origins')
                for name in ('history','history_anchor','history_selected','history_anchor_selected'):
                    local['ridge_'+name]=(a[name][:,-1]-y[:,-1])**2
            mf=PHASE/'mean_controls/predictions'/task/f'fold_{fold}.npz';md=read(PHASE/'mean_controls/fits'/task/f'fold_{fold}'/'DONE.json')
            assert sha(mf)==md['prediction_sha256']
            with np.load(mf,allow_pickle=False) as a:
                equal(a['origins'],val,'mean control origins')
                for key,name in [('raw','fit_mean'),('shrink','fit_mean_shrink')]:local[name]=(a[key][:,-1]-y[:,-1])**2
            for name,values in local.items():
                losses_folds.setdefault(name,[]).append(values);fold_mses.setdefault(name,[]).append(float(values.mean()))
            group_list.append(day_groups(bars,val,local));origin_chunks.append(val);fold_labels.extend([fold]*len(val))
        all_origins=np.concatenate(origin_chunks);losses={name:np.concatenate(values) for name,values in losses_folds.items()};n=len(all_origins)
        saved_path=STUDY/'analysis/paired_losses'/f'{task}.npz'
        with np.load(saved_path,allow_pickle=False) as a:
            equal(all_origins,a['origins'],'pooled origins');equal(np.array(fold_labels),a['fold'],'pooled folds')
            for name,values in losses.items():equal(a['loss_'+name],values,task+' per-origin '+name)
        bm=float(losses[base_name].mean())
        for name,values in losses.items():
            row=next(r for r in result['pooled'] if r['task']==task and r['method']==name)
            close(row['endpoint_mse'],float(values.mean()),'pooled endpoint MSE')
            close(row['baseline_mse'],bm,'pooled baseline MSE')
            close(row['improvement_pct'],float(100*(1-values.mean()/bm)),'pooled gain')
            positive=sum(x<y for x,y in zip(fold_mses[name],fold_mses[base_name]));assert row['positive_folds']==positive
            if name in path_totals:
                close(row['path_mse'],path_totals[name]/n,'pooled path MSE');close(row['endpoint_mae'],mae_totals[name]/n,'pooled MAE')
            expected_seeds={str(seed):float(100*(1-np.concatenate(values).mean()/bm)) for seed,values in seed_losses.get(name,{}).items()}
            assert row['seed_count']==len(expected_seeds)
            for seed,value in expected_seeds.items():close(row['seed_improvements_pct'][seed],value,'per-seed gain')
            pooled_checks.append({'task':task,'method':name,'per_origin_losses_exact':True,'pooled_metrics_verified':True})
        joined=[{'names':group_list[0]['names'],'counts':np.concatenate([g['counts'] for g in group_list]),'sums':np.concatenate([g['sums'] for g in group_list])}]
        controls=[base_name,'default_trained','default_shrink','fit_mean','fit_mean_shrink','ridge_history','ridge_history_anchor','ridge_history_selected','ridge_history_anchor_selected']
        pairs=[('repair_shrink',r) for r in controls]+[('repair_trained','default_trained')]
        for block in {'annual':[12,24],'quarterly':[3,6],'monthly':[21,42]}[profile]:
            recomputed_intervals=paired_intervals(joined,pairs,block,task)
            for row in recomputed_intervals:
                saved=next(r for r in result['paired_intervals'] if r['task']==task and r['block']==block and r['primary']==row['primary'] and r['reference']==row['reference'])
                assert row==saved,(task,block,row['primary'],row['reference'])
                interval_checks.append({'task':task,'block':block,'primary':row['primary'],'reference':row['reference'],'recomputed_exact':True})
        dates=bars.session_date[all_origins];end=bars.session_date[all_origins+h]
        for row in [r for r in result['sensitivity'] if r['task']==task]:
            if row['rule']=='original_global_earliest_greedy':
                mask=np.zeros(n,dtype=bool);last=None
                for i,o in enumerate(all_origins):
                    if last is None or o>=last+h:mask[i]=True;last=int(o)
            elif row['rule']=='absolute_bar_phase':mask=all_origins%h==row['offset']
            elif row['rule']=='leave_origin_year_out':mask=np.asarray([str(d)[:4]!=row['year'] for d in dates])
            elif row['rule'] in ('crosses_split','no_split'):
                cross=np.zeros(n,dtype=bool)
                for action in admission['split_actions']:cross|=(dates<action['date'])&(end>=action['date'])
                mask=cross if row['rule']=='crosses_split' else ~cross
            else:raise AssertionError(row['rule'])
            assert row['origins']==int(mask.sum())
            if mask.any():
                for name,values in losses.items():
                    mse=float(values[mask].mean());base=float(losses[base_name][mask].mean())
                    close(row['methods'][name]['endpoint_mse'],mse,'sensitivity MSE')
                    close(row['methods'][name]['improvement_pct'],100*(1-mse/base),'sensitivity gain')
            else:assert row['status']=='not_estimable'
            sensitivity_checks.append({'task':task,'rule':row['rule'],'origins':row['origins'],'verified':True})
        recomputed[task]={'origins':n,'methods':len(losses)}
        print(json.dumps({'task_verified':task,'configs':len(tcfg),'pooled_methods':len(losses)}),flush=True)
    assert len(cell_checks)==180 and len(restorations)==30 and len(result['rows'])==360
    assert len(pooled_checks)==len(result['pooled']) and len(interval_checks)==len(result['paired_intervals'])
    original_copy=read(STUDY/'audit/ANALYSIS_FREEZE_BEFORE_REPLAY_FIX.json')
    repair=read(STUDY/'audit/ANALYSIS_ASSERTION_REPAIR.json')
    assert repair['before']==original_copy
    assert repair['statistics_and_subset_rules_unchanged'] and repair['training_unchanged']
    for key in ('portable_statistics_sha256','subset_rules','baseline_controls'):
        assert repair['before'][key]==repair['after'][key]
    controller=ast.parse((STUDY/'code/controller.py').read_text())
    slots=next(ast.literal_eval(n.value) for n in controller.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SLOTS' for t in n.targets))
    assert len(slots)==8 and all(len(s)==2 for s in slots)
    report={'status':'passed','independent_saved_input_inference':True,'all180alphas_recomputed_inner_stop_only':True,
            'all180trained_and_shrink_paths_exact_samebatch128':True,'origin_model_pairs':origins_checked,
            'task_summary':recomputed,'cell_checks':cell_checks,'original_default_seed_restorations':restorations,
            'pooled_checks':pooled_checks,'paired_interval_checks':interval_checks,'sensitivity_checks':sensitivity_checks,
            'pooled_endpoint_mae_path_mse_and_seed_losses_verified':True,
            'both_v3_analysis_freezes_and_all_training_sources_unchanged':True,
            'analysis_assertion_revision_transparent_and_no_statistics_or_gate_change':True,
            'resource_evidence':{'controller_slots':slots,'max_configured_training_processes':8,'cores_per_fit_at_most2_receipts_verified':True,
                                 'same_original_deadline':manifest['deadline_epoch'],'latest_fit_completion_epoch':latest_fit,
                                 'review_affinity':sorted(os.sched_getaffinity(0)),
                                 'limit':'Concurrency ceiling is verified from frozen controller, queue locking and per-fit affinity receipts; no claim of complete OS process sampling.'},
            'limits':['Post-observation development; all historical periods remain exposed.','Same-batch replay proves saved-function correctness, not predictive success.','Monthly original512 versus128 batch outputs differ numerically; weights and training/stopping epochs match exactly.','Annual nonoverlap paths and starting phases do not create additional independent histories.'],
            'source_results_sha256':sha(STUDY/'analysis/RESULTS.json'),'verifier_sha256':sha(__file__),
            'Q3_values_read':False,'research_goal_achieved':False,'at_utc':now()}
    atomic(BASE/'REVIEW.json',report)
    print(json.dumps({'status':'passed','cells':180,'default_restorations':30,'pooled_rows':len(pooled_checks),'paired_intervals':len(interval_checks)}),flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        atomic(BASE/'FAILURE.json',{'status':'failed','traceback':traceback.format_exc(),'at_utc':now()})
        raise
