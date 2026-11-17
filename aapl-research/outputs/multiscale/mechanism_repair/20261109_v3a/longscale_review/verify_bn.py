"""Independent BN boundary, inner gate, saved loss and full-control audit."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import sys
import time
import numpy as np
import torch

R=Path(__file__).resolve().parent
B=R.parent
N=B/'bn_recalibration'
V=B.with_name('20261109_v3')
sys.path.insert(0,str(V/'code'))
from study import source_hashes
from engine import Data
from portable_predictor import Predictor
from multiscale.bars import targets

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def equal(a,b,msg):assert np.array_equal(a,b),msg
def near(a,b,msg):assert np.allclose(a,b,rtol=3e-12,atol=3e-14), (msg,a,b)
def stamp(s):return datetime.fromisoformat(s).timestamp()
def predict(p,windows,idx,batch=128):
    return np.concatenate([p.predict_prepared(*windows.arrays(idx[start:start+batch])) for start in range(0,len(idx),batch)])
def inner_metric(p,y):
    p=p[:,-1];y=y[:,-1];den=float(p@p);num=float(p@y)
    alpha=float(np.clip(num/den,0,1)) if den else 0.
    return {'n':len(y),'raw_mse':float(np.mean((p-y)**2)),'shrunk_mse':float(np.mean((alpha*p-y)**2)),
        'zero_mse':float(y@y)/len(y),'alpha':alpha,'forecast_energy':den/len(y),'alignment':num/len(y)}

def independently_bootstrap(groups,names,pairs,block):
    rng=np.random.default_rng(2026);sums=np.zeros((5000,len(names)));counts=np.zeros(5000)
    total=np.zeros(len(names));n=0
    for group in groups:
        local_count,local_sum=group;days=len(local_count)
        assert days>block
        total+=local_sum.sum(0);n+=local_count.sum()
        for cursor in range(0,5000,100):
            starts=rng.integers(0,days-block+1,size=(100,(days+block-1)//block))
            draw=(starts[:,:,None]+np.arange(block)).reshape(100,-1)[:,:days]
            sums[cursor:cursor+100]+=local_sum[draw].sum(1)
            counts[cursor:cursor+100]+=local_count[draw].sum(1)
    samples=sums/counts[:,None];point=total/n;result={}
    for primary,reference in pairs:
        i,j=names.index(primary),names.index(reference)
        diff=samples[:,j]-samples[:,i];relative=100*(1-samples[:,i]/samples[:,j])
        ci=np.quantile(diff,[.025,.975]);rci=np.quantile(relative,[.025,.975])
        result[(primary,reference,block)]={'mse_reference_minus_primary':point[j]-point[i],
            'difference_ci95':ci,'relative_improvement_pct':100*(1-point[i]/point[j]),'relative_ci95_pct':rci,
            'supports_improvement':bool(ci[0]>0)}
    return result

def main():
    start=time.time()
    assert set(os.sched_getaffinity(0))=={24,26} and os.environ['CUDA_VISIBLE_DEVICES']=='-1'
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    frozen=read(N/'SOURCE_FREEZE.json');revision=read(N/'NUMERICAL_ACCEPTANCE_REVISION.json')
    gate=read(N/'GATE_AND_ADMISSION.json');outerstart=read(N/'OUTER_START.json')
    runner=read(N/'RUNNER_V2_FREEZE.json');aux=read(N/'AUXILIARY_SOURCE_FREEZE.json')
    controls=read(N/'CONTROL_AUDIT.json');results=read(N/'RESULTS.json')
    boundary_checks={}
    def check(k,v):assert v,k;boundary_checks[k]=True
    check('frozen_original_sources',source_hashes()==frozen['identity']['original_source_hashes']==read(V/'protocol/CODE_FREEZE.json')['hashes'])
    check('fixed_budget_deadline',frozen['identity']['deadline_epoch']==gate['deadline_epoch']==read(V/'protocol/BUDGET.json')['deadline_epoch']==1794205909.6228158)
    check('original_diagnostic_and_rules',sha(N/'diagnostic.py')==frozen['identity']['diagnostic.py']==revision['original_source_sha256'] and sha(N/'BN_RULES.json')==frozen['identity']['BN_RULES.json']==revision['original_rules_sha256'])
    diff=revision['literal_diff']
    check('only_versioned_tolerance_literal_changed',(N/'diagnostic_v2.py').read_text()==(N/'diagnostic.py').read_text().replace(diff['old'],diff['new']) and sha(N/'diagnostic_v2.py')==revision['revised_source_sha256'])
    check('freeze_failure_revision_gate_before_outer',frozen['frozen_epoch']<revision['at_epoch']<runner['at_epoch']<gate['at_epoch']<outerstart['at_epoch'] and aux['at_epoch']<outerstart['at_epoch'])
    check('freeze_hash_chain',sha(N/'SOURCE_FREEZE.json')==revision['original_freeze_sha256']==gate['source_freeze_sha256']==outerstart['original_source_freeze_sha256'] and sha(N/'GATE_AND_ADMISSION.json')==outerstart['gate_sha256'] and sha(N/'NUMERICAL_ACCEPTANCE_REVISION.json')==runner['identity']['revision_sha256'] and sha(N/'run_v2.py')==runner['identity']['run_v2.py'])
    check('failed_tolerance_preserved',sha(N/'BATCH_NUMERIC_FAILURE.json')==revision['failure_receipt_sha256'])
    check('auxiliary_sources_frozen',all(sha(N/name)==digest for name,digest in aux['files'].items()))
    check('control_source_hashes',all(sha(Path(path))==digest for path,digest in controls['source_file_hashes'].items()))
    check('moment_estimator_source_excludes_targets','data.windows.arrays(indices)' in (N/'diagnostic.py').read_text() and 'x,a=data.windows.arrays(indices)' in (N/'diagnostic.py').read_text())
    allowed={f'backbone.bn{i}.{suffix}' for i in (1,2,3) for suffix in ('running_mean','running_var')}
    cells=[];data_cache={};raw_seed_losses={};screenrows={};decomposition_count=0;all_cell_losses={}
    dirs=sorted(p.parent for p in (N/'cells').glob('*/INNER_DONE.json'))
    check('exact_expected17cells',len(dirs)==17)
    for folder in dirs:
        inner=read(folder/'INNER_DONE.json');moment=read(folder/'MOMENTS_DONE.json');outer=read(folder/'OUTER_DONE.json')
        cfg=inner['identity']['cfg'];job=cfg['id'];task=f"{cfg['profile']}_h{cfg['horizon']}_future_log_return"
        oldfolder=V/'runs'/job;done=read(oldfolder/'DONE.json')
        check(job+'_checkpoint_hashes',sha(oldfolder/'best_trained.pt')==done['checkpoint_sha256']==inner['identity']['checkpoint_sha256'] and sha(folder/'adapted.pt')==inner['adapted_checkpoint_sha256']==moment['adapted_checkpoint_sha256'])
        old=torch.load(oldfolder/'best_trained.pt',map_location='cpu',weights_only=True)
        new=torch.load(folder/'adapted.pt',map_location='cpu',weights_only=True)
        check(job+'_cfg_same',old['cfg']==new['cfg']==cfg)
        check(job+'_model_keys_same',set(old['model'])==set(new['model']))
        changed=[k for k in old['model'] if not torch.equal(old['model'][k],new['model'][k])]
        check(job+'_only6runningbuffers_changed',set(changed)==allowed and changed==moment['changed_buffers'])
        for layer in moment['layers']:
            name=layer['layer']
            equal(torch.tensor(layer['pooled_mean_float64'],dtype=new['model'][name+'.running_mean'].dtype).numpy(),new['model'][name+'.running_mean'].numpy(),job+' moment mean cast')
            equal(torch.tensor(layer['pooled_unbiased_var_float64'],dtype=new['model'][name+'.running_var'].dtype).numpy(),new['model'][name+'.running_var'].numpy(),job+' moment variance cast')
        key=(cfg['profile'],cfg['fold'])
        if key not in data_cache:data_cache[key]=Data(cfg)
        data=data_cache[key];stop=data.splits['stop'];val=data.splits['validation']
        check(job+'_admitted_date_boundary',data.bars.session_date[-1]<='2026-06-30')
        check(job+'_fitstop_partition_counts',inner['fit_origins']==len(data.splits['fit']) and inner['stop_origins']==len(stop))
        check(job+'_purge',data.bars.end_ns[data.splits['fit'][-1]+cfg['horizon']]<data.bars.end_ns[stop[0]] and data.bars.end_ns[stop[-1]+cfg['horizon']]<data.bars.end_ns[val[0]])
        sy=targets(data.bars,stop,cfg['horizon'],cfg['target'])
        for mode,path in [('original',oldfolder/'best_trained.pt'),('adapted',folder/'adapted.pt')]:
            predictor=Predictor(path);pp=predict(predictor,data.windows,stop)
            mm=inner_metric(pp,sy)
            check(job+'_'+mode+'_all_inner_metrics_exact',mm==inner[mode])
            reference=predict(predictor,data.windows,stop[:19],19)
            for batch in (1,7):
                check(job+'_'+mode+f'_batch{batch}',np.allclose(predict(predictor,data.windows,stop[:19],batch),reference,rtol=1e-6,atol=1e-8))
        if cfg['fold'] in (1,2) and cfg['seed']==2026:screenrows.setdefault(task,[]).append(inner)
        check(job+'_outer_chronology',stamp(outer['at_utc'])>=outerstart['at_epoch'] and stamp(outer['at_utc'])<gate['deadline_epoch'] and stamp(inner['at_utc'])<outerstart['at_epoch'])
        check(job+'_loss_prediction_hashes',sha(folder/'outer_losses.npz')==outer['losses_sha256'] and sha(folder/'adapted_predictions.npz')==outer['predictions_sha256'] and sha(oldfolder/'predictions.npz')==done['prediction_sha256'])
        y=targets(data.bars,val,cfg['horizon'],cfg['target'])[:,-1]
        with np.load(folder/'adapted_predictions.npz',allow_pickle=False) as a:
            equal(a['origins'],val,job+' adapted origins');ap=a['prediction'][:,-1]
        with np.load(oldfolder/'predictions.npz',allow_pickle=False) as a:
            equal(a['origins'],val,job+' original origins');op=a['trained'][:,-1]
        p={'zero':np.zeros(len(y)),'original':op,'original_shrink':inner['original']['alpha']*op,'adapted':ap,'adapted_shrink':inner['adapted']['alpha']*ap}
        with np.load(folder/'outer_losses.npz',allow_pickle=False) as a:
            equal(a['origins'],val,job+' loss origins')
            all_cell_losses[(task,cfg['fold'],cfg['seed'])]={name:a[name].copy() for name in p}
            for name,value in p.items():
                loss=(value-y)**2;equal(a[name],loss,job+' '+name+' endpoint loss')
                dec=outer['decomposition'][name]
                near([dec['mse'],dec['truth_energy'],dec['forecast_energy'],dec['twice_alignment']],[loss.mean(),np.mean(y*y),np.mean(value*value),2*np.mean(value*y)],job+' decomposition')
                decomposition_count+=1
                if cfg['profile']=='minute':raw_seed_losses.setdefault((cfg['fold'],cfg['seed']),{})[name]=loss
        cells.append({'job':job,'only_running_buffers_changed':True,'all_inner_metrics_exact':True,'canonical_crossbatch_passed':True,'saved_full_path_endpoint_losses_exact':True})
        print(json.dumps({'verified_cell':job,'elapsed_seconds':round(time.time()-start,1)}),flush=True)
    admitted=[]
    for task,rows in screenrows.items():
        rows.sort(key=lambda r:r['identity']['cfg']['fold'])
        directions=[r['adapted']['raw_mse']<r['original']['raw_mse'] for r in rows]
        check(task+'_admission_directions',directions==gate['tasks'][task]['screen_raw_directions'] and all(directions)==gate['tasks'][task]['inner_direction_gate'])
        if all(directions):admitted.append(task)
        for r in rows:check(r['identity']['cfg']['id']+'_screen_receipt_hash',sha(N/'cells'/r['identity']['cfg']['id']/'INNER_DONE.json')==gate['screen_inner_receipt_hashes'][r['identity']['cfg']['id']])
    check('admitted_task_set',admitted==gate['mechanism_admitted_tasks']==gate['full_tasks']==['minute_h60_future_log_return'])
    check('compute_admission',gate['estimated_seconds']+300<gate['remaining_seconds_at_freeze'] and gate['compute_admitted'])
    # Independently pool both reported task scopes, including hourly screen-only.
    for task,reported in results['tasks'].items():
        keys=[key for key in all_cell_losses if key[0]==task]
        check(task+'_summary_cell_count',len(keys)==reported['cells'])
        summary_losses={};summary_fold_gains=[]
        for fold in sorted(set(k[1] for k in keys)):
            selected=[all_cell_losses[k] for k in keys if k[1]==fold]
            local={name:np.mean([a[name] for a in selected],axis=0) for name in selected[0]}
            for name,values in local.items():summary_losses.setdefault(name,[]).append(values)
            summary_fold_gains.append(100*(1-local['adapted_shrink'].mean()/local['zero'].mean()))
        pooled_summary={name:np.concatenate(arrays).mean() for name,arrays in summary_losses.items()}
        for name,value in pooled_summary.items():
            near(value,reported['pooled_mse'][name],task+' summary MSE')
            near(100*(1-value/pooled_summary['zero']),reported['improvement_over_zero_pct'][name],task+' summary improvement')
        near(summary_fold_gains,reported['fold_improvements'],task+' summary folds')
        check(task+'_positive_fold_count',sum(v>0 for v in summary_fold_gains)==reported['adapted_shrink_positive_fold_count'])
    task='minute_h60_future_log_return';audit=controls['tasks'][task]
    totals={};groups=[];seedtotals={s:{m:0. for m in ('adapted','adapted_shrink')} for s in (2026,2027,2028)};folds=[];n=0
    for fold in range(1,6):
        file=N/'all_control_losses'/task/f'fold_{fold}.npz'
        with np.load(file,allow_pickle=False) as a:idx=a['origins'].copy();loss={k:a[k].copy() for k in a.files if k!='origins'}
        data=data_cache[('minute',fold)];equal(idx,data.splits['validation'],'all-control origins')
        # Rebuild every control independently from the immutable prior arrays.
        with np.load(V/'analysis/candidate_paired_losses'/task/f'fold_{fold}.npz',allow_pickle=False) as a:
            pairs={'zero':'zero','default_original_raw':'original','default_original_shrink':'original_shrink','ridge_history':'ridge_history','ridge_history_anchor':'ridge_history_anchor','ridge_history_selected':'ridge_history_selected','ridge_history_anchor_selected':'ridge_history_anchor_selected'}
            for dest,source in pairs.items():equal(loss[dest],a['loss_'+source],dest+' lineage')
        for dest,source in [('repair_original_raw','original'),('repair_original_shrink','original_shrink'),('adapted','adapted'),('adapted_shrink','adapted_shrink')]:
            rebuilt=np.mean([raw_seed_losses[(fold,s)][source] for s in (2026,2027,2028)],axis=0);equal(loss[dest],rebuilt,dest+' mean seed loss')
        with np.load(B/'mean_controls/predictions'/task/f'fold_{fold}.npz',allow_pickle=False) as a:
            equal(loss['mean_raw'],a['loss_raw'],'mean raw lineage');equal(loss['mean_shrink'],a['loss_shrink'],'mean shrink lineage')
        for name,values in loss.items():totals[name]=totals.get(name,0.)+float(values.sum())
        for seed in (2026,2027,2028):
            for m in ('adapted','adapted_shrink'):seedtotals[seed][m]+=float(raw_seed_losses[(fold,seed)][m].sum())
        folds.append({'fold':fold,'origins':len(idx),'adapted_gain_pct':100*(1-loss['adapted'].mean()/loss['zero'].mean()),'adapted_shrink_gain_pct':100*(1-loss['adapted_shrink'].mean()/loss['zero'].mean())})
        names=list(loss);dates=data.bars.session_date[idx];calendar=np.unique(data.bars.session_date[(data.bars.session_date>=dates[0])&(data.bars.session_date<=dates[-1])]);pos=np.searchsorted(calendar,dates)
        counts=np.bincount(pos,minlength=len(calendar));day_sums=np.column_stack([np.bincount(pos,weights=loss[name],minlength=len(calendar)) for name in names]);groups.append((counts,day_sums));n+=len(idx)
    check('all13controls_all5folds',len(names)==13 and n==audit['origins']==376890)
    pooled={name:value/n for name,value in totals.items()};gain={name:100*(1-value/pooled['zero']) for name,value in pooled.items()}
    for name,value in pooled.items():near(value,audit['pooled_mse'][name],name+' pooled');near(gain[name],audit['improvement_over_zero_pct'][name],name+' gain')
    for got,want in zip(folds,audit['folds']):
        for k in got:near(got[k],want[k],'fold '+k)
    seedgains={str(s):{m:100*(1-value/totals['zero']) for m,value in modes.items()} for s,modes in seedtotals.items()}
    for seed in seedgains:
        for m,value in seedgains[seed].items():near(value,audit['per_seed_improvement_over_zero_pct'][seed][m],'seed gain')
    pairs=[(m,c) for m in ('adapted','adapted_shrink') for c in audit['controls']]
    recomputed={}
    for block in (1,5):recomputed.update(independently_bootstrap(groups,names,pairs,block))
    check('all44paired_intervals',len(recomputed)==len(audit['paired_intervals'])==44)
    for row in audit['paired_intervals']:
        key=(row['primary'],row['reference'],row['block']);new=recomputed[key]
        for k,v in new.items():near(v,row[k],str(key)+' '+k)
    perf={}
    for mode in ('adapted','adapted_shrink'):
        tests={'minimum_one_percent_over_zero':gain[mode]>=1,'all_three_seed_directions_positive':all(seedgains[s][mode]>0 for s in seedgains),'four_positive_fold_means':sum(r[mode+'_gain_pct']>0 for r in folds)>=4,'all_controls_positive_both_block_intervals':all(v['supports_improvement'] for k,v in recomputed.items() if k[0]==mode)}
        perf[mode]={'checks':tests,'performance_gate_passed':all(tests.values())}
        check(mode+'_gate_recomputed',perf[mode]==audit['performance_modes'][mode])
    check('hourly_only_screen',results['tasks']['hourly_h8_future_log_return']['scope']=='two_fold_single_seed_screen_only' and results['tasks']['hourly_h8_future_log_return']['cells']==2 and controls['screen_only_tasks']==['hourly_h8_future_log_return'])
    check('no_success_no_independent_no_Q3',all(not x['Q3_values_read'] and not x['independent_confirmation'] and not x['method_success_claimed'] for x in (results,controls)))
    receipt={'status':'passed','checks':boundary_checks,'cells':cells,'all_control_methods':names,'pooled_mse':pooled,'gain_over_zero_pct':gain,'per_seed_gains':seedgains,'performance_modes':perf,'paired_intervals_independently_recomputed':len(recomputed),'decompositions_recomputed':decomposition_count,'source_sha256':{str(p.relative_to(B)):sha(p) for p in [N/'BN_RULES.json',N/'SOURCE_FREEZE.json',N/'NUMERICAL_ACCEPTANCE_REVISION.json',N/'GATE_AND_ADMISSION.json',N/'OUTER_START.json',N/'RESULTS.json',N/'CONTROL_AUDIT.json']},'reviewer_sha256':sha(Path(__file__)),'scope_limits':['All17innerstop predictions independently replayed; fullouterlosses recomputed from saved full paths and original admitted targets, not all outer model paths replayed by this reviewer.','Moment float64 records cast back to all six saved buffers; fixed-weight boundary verified directly. All-fit activation moment estimator inspected, not rerun for every cell.','All44 control intervals independently rebuilt with reviewer block sampling code; no new market history.'],'research_goal_achieved':False,'independent_confirmation':False,'Q3_values_read':False,'elapsed_seconds':time.time()-start,'at_utc':datetime.now(timezone.utc).isoformat()}
    (R/'BN_REVIEW.json').write_text(json.dumps(receipt,indent=2,default=lambda x:bool(x) if isinstance(x,np.bool_) else float(x))+'\n')
    print(json.dumps({'status':'passed','cells':len(cells),'checks':len(boundary_checks),'paired_intervals':len(recomputed),'gain_adapted':gain['adapted'],'gain_adapted_shrink':gain['adapted_shrink'],'elapsed_seconds':time.time()-start}),flush=True)

if __name__=='__main__':main()
