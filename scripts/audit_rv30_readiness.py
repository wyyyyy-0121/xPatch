"""Audit an explicitly unfinished prospective handoff without claiming delivery."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from finance.io import ROOT,read_json,sha,stable_hash,write_json
from finance.runtime import configure,exclusive_run


def audit():
    from finance.provenance import verify_freeze
    from finance.prospective import read_plan,observation_directory
    from finance.verify import require_verifications
    from finance.confirmation import confirmation_evidence
    from finance.jobs import receipt
    from finance.acceptance import artifact_index,search_coverage
    config=read_json(ROOT/'configs/rv30.json');frozen=verify_freeze(config);plan=read_plan(config)
    directory=ROOT/plan['payload']['run_directory'];setup=read_json(directory/'setup.json')
    checks={};details={}
    checks['research_and_prospective_signatures_valid']=True
    import hashlib
    for path,digest in {**frozen['effective_source_files'],**frozen['postprocessing_lock_files'],
            'delivery/prospective_lock.json':sha(ROOT/'delivery/prospective_lock.json')}.items():
        blob=subprocess.check_output(['git','show',f'HEAD:{path}'],cwd=ROOT)
        if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('Signed source or plan not locally committed: '+path)
    checks['signed_implementation_and_plan_locally_committed']=True
    source=read_json(ROOT/'delivery/data.json');data={}
    for asset,entry in source['assets'].items():
        for key in ('csv','receipt','audit','audit20'):
            if sha(ROOT/entry[key])!=entry[key+'_sha256']:raise ValueError('Changed audited input: '+asset+'/'+key)
        data[asset]={k:entry[k] for k in ('regular_rows','expected_rows','coverage')}
    checks['all_source_and_audit_fingerprints_match']=len(data)==3
    details['data']=data
    history=read_json(ROOT/'delivery/history.json')
    for path,digest in history['historical_file_hashes'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Changed historical experiment: '+path)
    checks['original_experiments_preserved']=history['v2_result']=='completed_negative_result'
    for scope in ('development','confirmation'):
        proof=require_verifications(config,directory,scope)
        checks['two_independent_'+scope+'_rounds']=len(proof)==2
        details[scope+'_verification']=proof
    prerequisites=confirmation_evidence(config,directory,frozen['signature'])
    checks['confirmed_model_prerequisites_exact']=prerequisites==plan['payload']['historical_prerequisites']
    coverage=search_coverage(directory,config,setup);write_json(directory/'search_coverage.json',coverage)
    checks['registered_search_complete']=coverage['passed'] and not coverage['failed_trials']
    index=artifact_index(directory,frozen['signature']);write_json(directory/'artifact_index.json',index)
    counts={};allocated,gpu=0.,0.
    for path in (ROOT/'artifacts/rv30'/frozen['signature']).rglob('result.json'):
        r=receipt(path.parent);k=r['phase']+'/'+r['status'];counts[k]=counts.get(k,0)+1
        allocated+=r.get('allocated_cpu_seconds',0.)
        if r['model'] in ('xpatch','chronos2'):gpu+=r.get('wall_seconds',0.)
    charges=read_json(ROOT/'artifacts/rv30'/frozen['signature']/'interrupted_resource_charges.json')
    allocated+=sum(r['allocated_cpu_seconds'] for r in charges.values());gpu+=sum(r['gpu_wall_seconds'] for r in charges.values())
    checks['training_search_12200_complete']=counts.get('inner/complete',0)+counts.get('outer/complete',0)==12200
    checks['all_200_formal_outer_models']=counts.get('outer/complete',0)==200
    checks['original_42_replay_failures_preserved']=counts.get('verify/failed',0)==42
    checks['total_budget_preserved_and_within_limits']=allocated/3600<config['runtime']['total_cpu_core_hours'] and gpu/3600<config['runtime']['total_gpu_hours']
    details['receipt_counts']=counts;details['budget']={'allocated_cpu_core_hours':allocated/3600,'gpu_wall_hours':gpu/3600}
    checks['artifact_index_covers_all_run_receipts']=index['job_count']==len(list(directory.rglob('result.json')))
    examples=read_json(ROOT/'delivery/cli_examples.json')
    checks['real_cli_examples_and_exit_codes']=examples['passed'] and all(r['passed'] for r in examples['commands'])
    checks['latest_engineering_suite_passed']=read_json(ROOT/'artifacts/rv30/setup/prospective_20261104/validation.json')['passed']
    model=read_json(ROOT/'delivery/model.json');example=read_json(ROOT/'delivery/example_prediction.json')
    checks['confirmed_candidate_explicitly_unqualified_for_forward_use']=(model['role']=='historically_confirmed_candidate_pending_prospective' and
        model['panel']=='aapl' and model['model']=='ridge' and not model['prospective_qualified'] and not model['active_predictive_edge_established'])
    checks['actual_inference_has_units_timing_and_no_live_claim']=(example['status']=='historical_replay_candidate' and example['rv_mean']>0 and
        not example['eligible_signal'] and not example['direction_or_profit_prediction'] and example['received_at'] is None and
        (datetime.fromisoformat(example['as_of'])-datetime.fromisoformat(example['data_cutoff_at'])).total_seconds()==60 and
        (datetime.fromisoformat(example['target_window_end'])-datetime.fromisoformat(example['as_of'])).total_seconds()==1800)
    admission=read_json(observation_directory(plan)/'admission.json')
    details['admission']=admission
    checks['external_permission_is_explicit']=not admission['passed'] and admission['http_status']==403
    # Historical wheel evidence can name documentation snapshots that have since
    # advanced. Those snapshots must remain recoverable by their exact Git blob.
    restoration=read_json(ROOT/'delivery/environment_restore.json');snapshots={}
    for path,digest in restoration['evidence_files'].items():
        if sha(ROOT/path)==digest:continue
        if path not in ('delivery/README.md','delivery/ACCEPTANCE_CHECKLIST.md','delivery/ENVIRONMENT_RESTORE.md'):
            raise ValueError('Environment recovery evidence changed: '+path)
        for revision in subprocess.check_output(['git','log','--format=%H','--',path],cwd=ROOT,text=True).splitlines():
            blob=subprocess.run(['git','show',f'{revision}:{path}'],cwd=ROOT,capture_output=True)
            if blob.returncode==0 and hashlib.sha256(blob.stdout).hexdigest()==digest:
                snapshots[path]={'sha256':digest,'git_snapshot':revision};break
        else:raise ValueError('Historical documentation snapshot lost: '+path)
    checks['environment_recovery_scope_and_evidence_preserved']=(restoration['passed'] and restoration['cpu_replay_jobs']==8 and not restoration['gpu_execution_tested'])
    details['historical_document_snapshots']=snapshots
    return {'passed':bool(checks) and all(checks.values()),'completed_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'historical_research_complete_and_prospective_tooling_ready_only','checks':checks,'details':details,
        'signature':frozen['signature'],'prospective_plan_signature':plan['signature'],'final_acceptance':False,
        'goal_can_complete':False,'remaining':['timely SIP permission','60 real complete qualified sessions','two full forward verification rounds','final delivery acceptance and commit']}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args()
    p=ROOT/args.output
    if p.exists():parser.error('Preserve prior audits; use a new output file')
    configure(1)
    try:
        with exclusive_run(ROOT/'artifacts/rv30/driver.lock'):result=audit()
    except (ValueError,KeyError,OSError,AssertionError,TypeError,RuntimeError) as error:
        result={'passed':False,'error':str(error),'final_acceptance':False,'goal_can_complete':False}
    write_json(p,result);print({k:v for k,v in result.items() if k not in ('details','checks')});return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
