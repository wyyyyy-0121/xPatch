"""Metadata-only independent cost and frozen identity review; no market scores."""
import ast
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
assert set(os.sched_getaffinity(0)).issubset({24,26})
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
rules=read(BASE/'protocol/RULES.json');auth=read(BASE/'protocol/AUTHORIZATION.json')
cost=read(BASE/'audit/GPU_COST.json');admission=read(BASE/'audit/ADMISSION.json')
acceptance=read(BASE/'review/GPU_SYNTHETIC_ACCEPTANCE.json')
fit=read(BASE/'audit/GPU_FIT_ONLY_COST.json')
expected={(a,f,s) for a in ('tied','untied') for f in range(1,6) for s in (2026,2027,2028)}
checks={}
def check(name,value):
    checks[name]=bool(value)
    assert value,name
check('CPU_review_affinity_24_26',set(os.sched_getaffinity(0)).issubset({24,26}))
check('explicit_authorization_same_30_bundles_20_epochs',auth['authorized'] and auth['local_gpu_authorized'] and
    not auth['paid_resources'] and not auth['cloud_gpu_authorized'] and auth['planned_bundles']==30 and auth['fixed_epochs']==20)
check('same_fixed_deadline_everywhere',rules['deadline_epoch']==auth['deadline_epoch']==cost['deadline_epoch']==admission['deadline_epoch']==1794216709.)
check('full_cross_product_rules_and_cost',len(rules['jobs'])==len(cost['jobs'])==30 and
    {(j['arm'],j['fold'],j['seed']) for j in rules['jobs']}==expected and
    {(r['job']['arm'],r['job']['fold'],r['job']['seed']) for r in cost['jobs']}==expected)
check('20_epochs_horizons_seeds_folds_fixed',rules['fixed_epochs']==20 and rules['horizons']==[5,15,30,60] and
    rules['seeds']==[2026,2027,2028] and rules['folds']==[1,2,3,4,5] and rules['all_cells_required'])
check('same_authorization_plan_and_cost_hashes',rules['authorization_sha256']==sha(BASE/'protocol/AUTHORIZATION.json') and
    rules['plan_sha256']==sha(BASE/'PLAN_ZH.md') and rules['cost_file']=='audit/GPU_COST.json' and
    rules['cost_sha256']==sha(BASE/'audit/GPU_COST.json'))
check('all_training_sources_match_rules_cost_and_GPU_acceptance',all(
    sha(BASE/'code'/name)==digest==cost['sources'][name]==acceptance['source_hashes_after'][name]
    for name,digest in rules['sources'].items()))
check('all_final_acceptance_code_sources_unchanged',acceptance['status']=='passed' and
    acceptance['same_sources_during_audit'] and all(sha(BASE/'code'/name)==digest
    for name,digest in acceptance['source_hashes_after'].items()))
check('GPU_acceptance_identity_bound',cost['synthetic_acceptance_sha256']==sha(BASE/'review/GPU_SYNTHETIC_ACCEPTANCE.json'))
check('all_cost_measurement_file_hashes_match',all(sha(BASE/name)==digest for name,digest in cost['measurement_hashes'].items()) and
    cost['single_fit_only_hash']==sha(BASE/'audit/GPU_FIT_ONLY_COST.json'))
training=[];inference=[]
for name in cost['measurement_hashes']:
    rec=read(BASE/name)
    check('measurement_no_market_scores_'+name,rec['Q3_values_read'] is False and
        not rec.get('market_stop_or_outer_scored',False) and not rec.get('stop_or_outer_targets_constructed',False))
    check('measurement_training_identity_'+name,rec['experiment_sha256']==rules['sources']['experiment.py'])
    (inference if '/gpu_inference_' in name else training).extend(rec['rows'])
for arm in ('tied','untied'):
    tr=[r for r in training if r['arm']==arm];inf=[r for r in inference if r['arm']==arm]
    check('training_cost_max_measured_'+arm,cost['training_step_seconds'][arm]==max(r['step_seconds_mean'] for r in tr))
    check('checkpoint_cost_max_measured_'+arm,cost['checkpoint_max_seconds'][arm]==max(r['checkpoint_cpu_snapshot_atomic_save_seconds_max'] for r in tr))
    check('metric_cost_max_measured_'+arm,cost['metric_batch_seconds'][arm]==max(r['paths']['metric']['mean_seconds'] for r in inf))
    check('collect_cost_max_measured_'+arm,cost['collect_batch_seconds'][arm]==max(r['paths']['collect']['mean_seconds'] for r in inf))
check('mixed_two_slots_fixed',cost['slots']==rules['maximum_concurrent_gpu_workers']==2 and
    cost['slot_arms']==rules['slot_arms']==['tied','untied'] and rules['slots']==[[0,2],[4,6]])
check('resource_bounds',rules['maximum_concurrent_gpu_workers']<=auth['maximum_training_processes'] and
    auth['logical_cpus_per_training_process']==2 and auth['analysis_cpus_total']==2)
check('fixed_cost_constants',cost['inflation']==1.25 and cost['fixed_prepare_and_output_seconds_per_job']==20 and
    cost['reserved_analysis_and_acceptance_seconds']==rules['reserve_seconds']==600)
rows=[];arm_totals={'tied':0.,'untied':0.};max_diff=0.
for row in cost['jobs']:
    job=row['job'];arm=job['arm'];counts=fit['fold_counts'][str(job['fold'])]
    # The real loop merges a one-observation remainder into the preceding batch.
    fit_batches=(counts['fit']+127)//128-(1 if counts['fit']%128==1 else 0)
    updates=20*fit_batches;stop_batches=(counts['stop']+127)//128;outer_batches=(counts['validation']+127)//128
    check('count_identity_'+job['id'],row['training_updates']==updates and
        row['stop_metric_batches_per_epoch']==stop_batches and row['outer_input_only_batches_per_pass']==outer_batches)
    st=cost['training_step_seconds'][arm];mt=cost['metric_batch_seconds'][arm];co=cost['collect_batch_seconds'][arm]
    compute=updates*st+20*stop_batches*mt+2*stop_batches*(mt+co)+2*outer_batches*co
    checkpoints=(updates//100+20)*cost['checkpoint_max_seconds'][arm]
    seconds=1.25*compute+checkpoints+20
    diff=abs(seconds-row['projected_seconds']);max_diff=max(max_diff,diff)
    check('projection_arithmetic_'+job['id'],diff<1e-9)
    arm_totals[arm]+=seconds
    rows.append({'job':job,'count_source':counts,'training_updates':updates,'stop_batches':stop_batches,
        'outer_batches':outer_batches,'compute_with_25pct_margin_seconds':1.25*compute,
        'periodic_and_epoch_latest_checkpoint_seconds':checkpoints,'fixed_prepare_output_allowance_seconds':20,
        'independently_recomputed_seconds':seconds,'recorded_seconds':row['projected_seconds'],'absolute_difference':diff})
for arm,total in arm_totals.items():check('arm_total_'+arm,abs(total-cost['arm_projected_seconds'][arm])<1e-9)
total=max(arm_totals.values())+600
check('complete_cost_and_remaining_budget',abs(total-cost['projected_total_seconds_including_reserve'])<1e-9 and
    cost['remaining_seconds_at_admission']==rules['remaining_wall_seconds_at_freeze'] and total<rules['remaining_wall_seconds_at_freeze'])
check('freeze_time_budget_consistency_within_100ms_metadata_write',abs(rules['at_epoch']+rules['remaining_wall_seconds_at_freeze']-rules['deadline_epoch'])<0.1)
check('admission_receipt_identity',admission['status']=='passed' and admission['rules_sha256']==sha(BASE/'protocol/RULES.json') and
    abs(admission['projected_seconds_including_reserve']-total)<1e-9 and admission['remaining_seconds']==rules['remaining_wall_seconds_at_freeze'])
check('same_arm_cost_descending_order',all(
    [j['fold'] for j in rules['jobs'] if j['arm']==arm]==[5,5,5,4,4,4,3,3,3,2,2,2,1,1,1]
    for arm in ('tied','untied')))
# Validate the controller constants without importing/executing it.
tree=ast.parse((BASE/'code/controller.py').read_text());slot_constants=[]
for node in tree.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SLOTS' for t in node.targets):
        slot_constants=ast.literal_eval(node.value)
check('controller_slot_constants_match',slot_constants[:2]==[(0,2),(4,6)])
source=(BASE/'code/controller.py').read_text()
check('controller_enforces_arm_and_uses_full_worker_source',"assert slot_arms == ['tied','untied']" in source and
    "j['arm'] == arm" in source and "str(BASE / 'code/experiment.py')" in source)
queue=read(BASE/'audit/QUEUE_STATE.json');active=queue['active'];jobmap={j['id']:j for j in rules['jobs']}
check('live_queue_same_deadline_and_fixed_arm_affinity',queue['pid']==989224 and queue['deadline_epoch']==rules['deadline_epoch'] and
    all((jobmap[r['job']]['arm'],r['cpus']) in (('tied',[0,2]),('untied',[4,6])) for r in active))
check('live_controller_CPU_24_26',set(os.sched_getaffinity(queue['pid'])).issubset({24,26}))
check('live_worker_CPU_affinities',all(set(os.sched_getaffinity(r['pid']))==set(r['cpus']) for r in active))
check('no_Q3_confirmation_or_Goal_success_claim',rules['Q3_values_read'] is False and rules['independent_confirmation'] is False and
    rules['candidate']=='tied_selected_shrink' and rules['early_termination'] is False)
result={'status':'passed','scope':'read-only metadata, source identities and arithmetic; no market scores, targets or prediction arrays opened',
    'at_utc':datetime.now(timezone.utc).isoformat(),'rules_sha256':sha(BASE/'protocol/RULES.json'),
    'cost_sha256':sha(BASE/'audit/GPU_COST.json'),'source_hashes':rules['sources'],'checks':checks,
    'independent_job_arithmetic':rows,'maximum_job_projection_absolute_difference':max_diff,
    'independent_arm_totals_seconds':arm_totals,'independent_complete_projection_seconds':total,
    'remaining_seconds_at_freeze':rules['remaining_wall_seconds_at_freeze'],
    'unallocated_seconds_beyond_projection':rules['remaining_wall_seconds_at_freeze']-total,
    'checkpoint_cost_note':'Periodic+epoch latest saves are explicit. Initial/best/last saves, capture/setup and compressed outputs share the fixed 20s/job allowance; 25% compute margin and 600s analysis reserve are separate. This is a projection, not a runtime guarantee.',
    'live_queue_metadata_snapshot':queue,'market_stop_or_outer_arrays_opened':False,'GPU_used_by_review':False,'Q3_values_read':False,
    'independent_statistics_review_design':'review/INDEPENDENT_STATISTICS_REVIEW_PLAN_ZH.md',
    'statistics_review_not_started':True}
(BASE/'review/FINAL_ADMISSION_REVIEW.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'checks':len(checks),'projection_seconds':total,
    'remaining_seconds':result['remaining_seconds_at_freeze'],'margin_seconds':result['unallocated_seconds_beyond_projection'],
    'maximum_job_error':max_diff,'market_arrays_opened':False},indent=2))
