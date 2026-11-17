"""Third-role saved-small-JSON and source review only; never opens arrays."""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path

B=Path(__file__).resolve().parent
checks=[]
def check(ok,name):
    if not ok:raise RuntimeError(name)
    checks.append(name)
def read(name):return json.loads((B/name).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def near(a,b,rel=1e-11,atol=0.):return abs(a-b)<=max(atol,rel*max(abs(a),abs(b),1e-300))

def main():
    os.sched_setaffinity(0,{26})
    spec=read('DESIGN_SPEC.json');p=read('EXECUTION_PROTOCOL.json')
    result=read('run/RESULTS.json');decision=read('run/DECISION.json')
    ind=read('run/INDEPENDENT_REPLAY.json');life=read('run/LIFECYCLE_ACCEPTANCE.json')
    pause=read('run/PAUSED.json');historical=read('run/PAUSED_CHECKPOINT.json');final=read('run/CHECKPOINT.json')
    identity=sha(B/'EXECUTION_PROTOCOL.json')
    check(os.sched_getaffinity(0)=={26},'review CPU26')
    check(p['deadline_epoch']-p['started_epoch']==180,'single180second window')
    check(p['GPU'] is False and p['Q3_allowed'] is False,'protocol scope')
    for name,digest in p['locked_files'].items():check(sha(name)==digest,'frozen source '+name)
    for obj,label in [(result,'result'),(ind,'independent'),(life,'lifecycle'),(pause,'pause'),(historical,'historical'),(final,'terminal')]:
        check(obj['protocol_sha256']==identity,label+' protocol identity')
    check(ind['passed'] is True,'independent passed')
    check(life['status']=='passed' and life['actual_shared_seconds']<180,'lifecycle within window')
    check(life['deadline_epoch']==p['deadline_epoch']==ind['deadline_epoch'],'same absolute deadline')
    check(life['signal']=='SIGUSR1' and life['first_exit']==75 and life['resumed_exit']==0,'real signal75 resume0')
    check(life['initial_pid']!=life['resumed_pid'] and historical['worker_pids']==[life['initial_pid']] and final['worker_pids']==[life['initial_pid'],life['resumed_pid']],'historical newPID chain')
    check(sha(B/'run/PAUSED_CHECKPOINT.json')==life['paused_checkpoint_sha256']==pause['checkpoint_sha256'],'historical paused bytes hash')
    check(len(historical['completed'])==1 and historical['completed']==final['completed'][:1],'first commit unchanged')
    check(len(final['completed'])==4 and life['all_children_terminal'] is True,'all four states terminal')
    for item in final['completed']:check(sha(B/('run/'+item['state']+'.json'))==item['sha256'],'committed state '+item['state'])
    check(not any('FAILURE' in f.name for f in (B/'run').iterdir()),'no formal failure artifacts')
    rows=result['records'];controls=['zero','prefix_update_mean','same_information_Ridge']
    check([x['state'] for x in rows]==[x['state'] for x in spec['states']],'all original states')
    check(sum(x['horizon'] for x in rows)==136==ind['state_horizon_positions'],'all136h')
    check(spec['controls_for_gate']==controls,'three fixed controls')
    for row,entry in zip(rows,spec['states']):
        name=row['state'];hs=row['all_horizons'];H=entry['horizon'];N=entry['count']
        check(row['count']==N and row['horizon']==H and row['tail_sha256']==entry['tail_sha256'],'state identity '+name)
        check(len(hs)==H and [h['horizon'] for h in hs]==list(range(1,H+1)),'full horizon order '+name)
        check(row['opened_prediction_members']==[x+'.npy' for x in spec['prediction_decode_members']],'five prediction members '+name)
        check(row['bar_identity']['full_planned_grid_equal'] and row['bar_identity']['close_member_opened'] is False,'planned grid no close '+name)
        check(row['bar_identity']['opened_members']==[x+'.npy' for x in spec['bar_decode_members']],'four bar members '+name)
        for h in hs:
            tag=name+':'+str(h['horizon']);F=h['floor_MSE'];R_b=h['b_MSE'];inside=h['support_b_MSE'];E=h['risk_error_bounds']
            check(0<=h['support_count']<=N,'support count '+tag)
            check(F>=0 and inside>=0 and R_b>0,'risk signs '+tag)
            check(abs(R_b-F-inside)<=h['identity_tolerance'],'raw risk partition '+tag)
            check(near(R_b,h['MSE']['prefix_update_mean']),'bcontrol '+tag)
            check(E['floor']==E['prefix_update_mean'] and all(v>0 for v in E.values()),'positive bound '+tag)
            for c in controls:
                R=h['MSE'][c];raw=100*(1-F/R);safe=100*(1-max(0,F-E['floor'])/(R+E[c]))
                check(R>0 and math.isfinite(R),'positive reference '+tag+c)
                check(near(raw,h['maximum_improvement_pct'][c],atol=1e-10),'raw formula '+tag+c)
                check(near(safe,h['safe_upper_maximum_improvement_pct'][c],atol=1e-10) and safe>=raw,'safe formula direction '+tag+c)
        for c in controls:
            metric=row['metrics'][c];Fpath=math.fsum(h['floor_MSE'] for h in hs)/H
            Rpath=math.fsum(h['MSE'][c] for h in hs)/H
            check(near(metric['endpoint_MSE'],hs[-1]['MSE'][c]) and near(metric['path_MSE'],Rpath),'equal horizon risk '+name+c)
            check(near(row['metrics']['floor']['path_MSE'],Fpath),'equal horizon floor '+name+c)
            check(near(metric['path_maximum_improvement_pct'],100*(1-Fpath/Rpath),atol=1e-10),'ratio after aggregation '+name+c)
        if name.startswith('minute'):
            check(all(h['support_count']==0 for h in hs[:-1]),'minute first59zero '+name)
            check(hs[-1]['support_count']==(39 if name.endswith('1') else 79),'minute endpoint support '+name)
            path=row['metrics']['prefix_update_mean']['path_MSE']-row['metrics']['floor']['path_MSE']
            endpoint=row['metrics']['prefix_update_mean']['endpoint_MSE']-row['metrics']['floor']['endpoint_MSE']
            check(near(path,endpoint/60,rel=1e-10,atol=1e-20),'minute1over60 dilution '+name)
        else:
            check(hs[-1]['support_count']==N and hs[-1]['floor_MSE']==0,'hour endpoint unconstrained '+name)
            check(row['metrics']['zero']['endpoint_maximum_improvement_pct']==100.,'hour100oracle only '+name)
    check(len(decision['necessary_checks'])==24 and len(decision['tasks'])==2 and decision['necessary_check_count']==26,'26necessary subset not58')
    check(len({(x['state'],x['control'],x['metric']) for x in decision['necessary_checks']})==24,'no duplicate necessary gates')
    excluded=[]
    for item in decision['necessary_checks']+decision['tasks']:
        raw=item.get('maximum_improvement_pct',item.get('equal_fold_endpoint_maximum_improvement_over_zero_pct'))
        safe=item['safe_upper_maximum_improvement_pct'];threshold=item['threshold_pct']
        status='excluded' if safe<threshold-1e-8 else ('within_numerical_margin' if raw<=threshold+1e-8 else 'not_ruled_out')
        check(status==item['status'],'gate classification '+str(item))
        if status=='excluded':excluded.append(item)
    for task in decision['tasks']:
        subset=[r for r in rows if r['state'].startswith(task['task'])]
        expected=math.fsum(r['metrics']['zero']['endpoint_maximum_improvement_pct'] for r in subset)/2
        check(near(task['equal_fold_endpoint_maximum_improvement_over_zero_pct'],expected,atol=1e-12),'equal fold '+task['task'])
    check(len(excluded)==6==decision['excluded_necessary_conditions'],'six exclusions')
    check(decision['fixed_support_class_ruled_out_on_development_gates'] is True,'fixed class excluded')
    for key in ['model_fitting_admitted','candidate_admitted','independent_market_confirmation','original_58_gate_recipe_changed','full_research_goal_achieved']:
        check(decision[key] is False,'boundary '+key)
    for kind in ['necessary_checks','tasks']:
        for main,other in zip(decision[kind],ind['decision'][kind]):
            for key,value in main.items():
                if isinstance(value,float):check(near(value,other[key],atol=1e-8),'independent decision numerical '+key)
                else:check(value==other[key],'independent decision identity '+key)
    for key in decision:
        if key not in ['necessary_checks','tasks']:check(decision[key]==ind['decision'][key],'independent terminal '+key)
    check(result['market_fits']==result['training_updates']==result['model_inference']==0,'zero fitupdatesinference')
    check(result['close_member_opened'] is False and result['Q3_read'] is False and result['GPU'] is False,'recorded no closeQ3GPU')
    for key in ['Q3_read','GPU','new_model_fit','close_member_decoded','q_computed','market_confirmation']:
        check(ind['scope'][key] is False,'independent scope '+key)
    check(ind['scope']['CPU']==[26] and ind['scope']['numeric_threads']==1,'independent oneCPU')
    report={'status':'artifact_checks_passed_waiting_document_claims','reviewer':'/root/opportunity_result_review',
            'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol_sha256':identity,
            'artifact_checks_passed':len(checks),'checks':checks,'excluded_necessary_conditions':excluded,
            'independent_checks':ind['checks'],'independent_check_count_includes_scope_identity_structural_and_numeric':True,
            'max_differences':ind['max_differences'],'minute_task':decision['tasks'][0],'hourly_task':decision['tasks'][1],
            'historical_paused_checkpoint_sha256':sha(B/'run/PAUSED_CHECKPOINT.json'),
            'actual_shared_seconds':life['actual_shared_seconds'],'source_results_unchanged':True,
            'review_scope':{'small_JSON_source_only':True,'market_array_decode':False,'numeric_library_import':False,
                           'market_refit':False,'CPU':[26],'Q3_read':False,'GPU':False}}
    (B/'ARTIFACT_RESULT_REVIEW_PENDING_CLAIMS.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print({'checks':len(checks),'excluded':len(excluded),'protocol':identity})

if __name__=='__main__':main()
