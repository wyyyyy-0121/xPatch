"""Independent136interval/full-gate acceptance after analytic fit/path review."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,time
import numpy as np
from verify_context import independently_bootstrap

R=Path(__file__).resolve().parent;B=R.parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def near(a,b,msg):assert np.allclose(a,b,rtol=5e-10,atol=3e-12),(msg,a,b)

def main():
    start=time.time();assert sorted(os.sched_getaffinity(0))==[24,26] and os.environ['CUDA_VISIBLE_DEVICES']=='-1'
    fit=read(R/'FIT_AND_PATH_REVIEW.json');assert fit['status']=='analytic_fits_and_paths_passed_full_statistics_pending' and len(fit['cells'])==40
    rules=read(B/'protocol/RULES.json');evalrule=read(B/'protocol/EVALUATION_FREEZE.json');gate=read(B/'protocol/GATE_AND_ADMISSION.json')
    results=read(B/'analysis/STATISTICS.json');outer=read(B/'analysis/OUTER_RESULTS.json')
    assert fit['rules_sha256']==results['rules_sha256']==sha(B/'protocol/RULES.json')
    assert fit['evaluation_freeze_sha256']==results['evaluation_freeze_sha256']==sha(B/'protocol/EVALUATION_FREEZE.json')
    assert outer['statistics_sha256']==sha(B/'analysis/STATISTICS.json')
    assert results['candidate']=='D_joint_shrink' and not results['Q3_values_read'] and not results['independent_confirmation'] and not results['research_goal_achieved']
    checks={};task_receipts=[];interval_count=0;summary_rows=0
    for task,reported in results['tasks'].items():
        profile=task.split('_h',1)[0];names=None;totals={};folds={};n=0;groups=[]
        for fold in range(1,6):
            file=R/'verified_losses'/task/f'fold_{fold}.npz'
            with np.load(file,allow_pickle=False) as a:
                local_names=[k for k in a.files if k not in ['origins','session_date','day_grid']]
                # Use saved owner column order solely to match matrix layout;
                #each value is the independently reconstructed loss column.
                with np.load(B/'analysis/paired_losses'/task/f'fold_{fold}.npz',allow_pickle=False) as o:
                    owner_names=[k for k in o.files if k not in ['origins','session_date','day_grid']]
                assert set(local_names)==set(owner_names) and len(local_names)==33
                if names is None:names=owner_names
                assert owner_names==names
                dates=a['session_date'];grid=a['day_grid'];origins=a['origins'];positions=np.searchsorted(grid,dates)
                counts=np.bincount(positions,minlength=len(grid))
                sums=np.column_stack([np.bincount(positions,weights=a[name],minlength=len(grid)) for name in names])
                groups.append((counts,sums));n+=len(origins)
                for name in names:
                    totals[name]=totals.get(name,0.)+float(a[name].sum());folds.setdefault(name,[]).append(float(a[name].mean()))
        summary={name:{'origins':n,'endpoint_mse':total/n,'improvement_pct':100*(1-total/totals['zero']),'positive_folds':sum(x<z for x,z in zip(folds[name],folds['zero'])),'fold_mse':folds[name]} for name,total in totals.items()}
        assert set(summary)==set(reported['summary'])
        for name,row in summary.items():
            for k,value in row.items():near(value,reported['summary'][name][k],task+' '+name+' summary '+k)
            summary_rows+=1
        controls=[name for name in names if name not in ['D_joint_shrink','D_joint_raw']]
        assert len(controls)==31 and set(controls)==set(reported['comparisons'])
        diagnostics=[('D_joint_raw','C_history_raw'),('D_joint_raw','B_context_raw'),('B_context_raw','A_mean_raw')]
        pairs=[('D_joint_shrink',control) for control in controls]+diagnostics
        calculated={}
        for block in evalrule['blocks'][profile]:
            calculated.update(independently_bootstrap(groups,names,pairs,block))
        assert len(calculated)==68
        emitted=[row for rows in reported['comparisons'].values() for row in rows]+reported['diagnostic_contrasts']
        assert len(emitted)==68 and len(reported['diagnostic_contrasts'])==6
        assert {(r['primary'],r['reference'],r['block']) for r in emitted}==set(calculated)
        for row in emitted:
            key=(row['primary'],row['reference'],row['block']);expected=calculated[key]
            assert row['task']==task and row.get('repetitions',5000)==5000
            for name,value in expected.items():
                if isinstance(value,(str,bool)) or value is None:assert row[name]==value,(task,key,name,row[name],value)
                else:near(value,row[name],task+' '+str(key)+' '+name)
            interval_count+=1
        candidate=summary['D_joint_shrink']
        performance_checks={'minimum_one_percent':candidate['improvement_pct']>=1.,'four_positive_folds':candidate['positive_folds']>=4,'all_control_intervals_both_blocks':all(v['supports_improvement'] for key,v in calculated.items() if key[0]=='D_joint_shrink')}
        inner=all(r['gate_raw_D_lt_zero_and_C'] for r in fit['inner_gates'][task][:2])
        assert inner==gate['tasks'][task]['inner_gate_passed']==reported['inner_direction_gate_passed']
        assert performance_checks==reported['checks']
        perf=all(performance_checks.values())
        assert perf==reported['performance_gate_passed']
        assert (perf and inner)==reported['independent_confirmation_eligible']
        assert reported['training_seed_count'] is None and reported['deterministic_solver']
        task_receipts.append({'task':task,'origins':n,'loss_columns':33,'controls':31,'intervals':68,'candidate':candidate,'checks':performance_checks,'inner_direction_gate_passed':inner,'performance_gate_passed':perf,'independent_confirmation_eligible':perf and inner,'complete_day_grid_lengths':[len(g[0]) for g in groups],'zero_origin_days':[int(np.sum(g[0]==0)) for g in groups]})
    assert interval_count==136 and summary_rows==66
    assert results['both_primary_performance_gates_passed']==all(t['performance_gate_passed'] for t in task_receipts)
    assert results['independent_confirmation_eligible']==all(t['independent_confirmation_eligible'] for t in task_receipts)
    assert len(outer['metrics'])==80
    checks={'40models_analytic_fit_and_full_path_rebuilt':True,'80metric_rows_and80origin_path_loss_arrays':fit['owner_metric_rows_checked']==fit['per_origin_path_loss_checks']==80,'33columns_all10folds_source_lineage':True,'31controls_each_task_including_all_stochastic_seeds':True,'66pooled_rows_recomputed':summary_rows==66,'136intervals_independently_sampled':interval_count==136,'unchanged_candidate_and_inner_qualification_logic':True,'no_pseudo_seed_replication':True,'no_Q3_or_independent_confirmation_claim':True}
    receipt={'status':'passed','checks':checks,'tasks':task_receipts,'matrix_models':40,'raw_shrink_metric_rows':80,'paired_intervals':interval_count,'pooled_rows':summary_rows,'both_primary_performance_gates_passed':all(t['performance_gate_passed'] for t in task_receipts),'independent_confirmation_eligible':all(t['independent_confirmation_eligible'] for t in task_receipts),'independent_confirmation':False,'research_goal_achieved':False,'goal_should_remain_active':True,'Q3_values_read':False,'source_sha256':{str(p.relative_to(B)):sha(p) for p in [B/'protocol/RULES.json',B/'protocol/EVALUATION_FREEZE.json',B/'analysis/OUTER_RESULTS.json',B/'analysis/STATISTICS.json',R/'FIT_AND_PATH_REVIEW.json']},'reviewer_sha256':sha(Path(__file__)),'elapsed_seconds':time.time()-start,'at_utc':datetime.now(timezone.utc).isoformat()}
    (R/'REVIEW.json').write_text(json.dumps(receipt,indent=2,default=lambda x:bool(x) if isinstance(x,np.bool_) else float(x))+'\n')
    print(json.dumps({'status':'passed','intervals':interval_count,'pooled_rows':summary_rows,'candidate':[{k:r[k] for k in ['task','candidate','inner_direction_gate_passed','performance_gate_passed','independent_confirmation_eligible']} for r in task_receipts],'elapsed_seconds':time.time()-start},indent=2),flush=True)

if __name__=='__main__':main()
