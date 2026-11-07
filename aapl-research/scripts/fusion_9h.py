"""Fixed A-only three-seed/two-epoch execution under the user's nine-hour amendment."""
import argparse
from datetime import datetime
import gc
import math
import os
from pathlib import Path
import sys
import tarfile
import time

import numpy as np
import psutil
import torch

from fusion.next.common import ROOT, OUT as BASE, digest, identity, read_json, utc_now, write_json
from scripts import fusion_9h_guard as guard
from scripts.followup_admission import check_frozen, identified
from scripts.followup_runtime import FOLLOW, isolated_outputs
from scripts.run_fusion_followup import original_idle, execution_locks, runtime_signature

ENTRY=Path(__file__).resolve()
PARENT=FOLLOW/'preparation/frozen_6e143bef48e2fe6f.json'
PLAN=ROOT/'FUSION_9H_AMENDMENT.md'
SEEDS=(2026,2027,2028)
QUARTERS=('2025Q1','2025Q3','2026Q1')
DIAGNOSTIC=FOLLOW/'runs/6e143bef48e2fe6f5f970dbfd35e0bfa31165ac8510bc4b8657225c5f981d533/attempts/probe_04_sustained_9h'


def immutable(path,value):
    if path.exists():
        if read_json(path)!=value:raise ValueError('Preserve conflicting artifact: '+str(path))
    else:write_json(path,value)


def diagnostic(parent):
    result=identified(DIAGNOSTIC/'diagnostic_result.json')
    request=identified(DIAGNOSTIC/'guard_request.json');state=read_json(DIAGNOSTIC/'guard_state.json')
    if (result['parent_manifest_identity']!=parent['identity'] or result['entry_sha256']!=digest(ROOT/'scripts/diagnose_fusion_9h.py')
            or state['stage']!='completed' or state['worker_returncode']!=0 or state['live_group_members']
            or result['guard_request_identity']!=request['identity'] or guard.exact_alive(state['guard'])
            or guard.group_members(state['worker']) or not result['exact_resume_passed']
            or result['FP32_repeat_max_difference']!=0 or not result['training_only']
            or result['updates_per_block']!=256 or len(result['block_seconds'])!=4
            or result['seconds_per_update_upper']!=max(result['block_seconds'])/256):
        raise ValueError('Need complete, exited, exact fixed sustained diagnostic')
    return result


def freeze():
    parent=check_frozen(PARENT);probe=diagnostic(parent)
    candidate=next(c for c in parent['candidates'] if c['name']=='A_foundation_raw')
    value={'schema':1,'artifact_type':'user_shortened_GPU_focus_protocol',
        'authorization':'User: GPU has nine hours left; change plan and execute on 5090 now',
        'parent_manifest':str(PARENT.relative_to(ROOT)),'parent_sha256':digest(PARENT),
        'parent_identity':parent['identity'],'amendment_sha256':digest(PLAN),
        'implementation':{p:digest(ROOT/p) for p in ('scripts/fusion_9h.py','scripts/fusion_9h_guard.py','scripts/diagnose_fusion_9h.py')},
        'source_identity':parent['source_identity'],'scope_identity':parent['scope_identity'],
        'plan_sha256':parent['plan_sha256'],'limit':parent['limit'],
        'GPU_limit':{'total_wall_hours':9,'anchor_utc':'2026-11-07T02:11:20+00:00','deadline_utc':guard.DEADLINE},
        'candidate':candidate,'seeds':list(SEEDS),'validation':list(QUARTERS),
        'total_coverage_equivalent_epochs':2,'cancelled_full_fit_candidates':['B_no_foundation_scaled','C_no_foundation_raw'],
        'diagnostic_identity':probe['identity'],'diagnostic_sha256':digest(DIAGNOSTIC/'diagnostic_result.json'),
        'selection_by_new_development_score':False,'engineering_fit_despite_negative_development_is_only_engineering':True,
        'Q2_Q3_scoring_authorized':False,'independent_success_claimed':False}
    value['identity']=identity(value)
    path=FOLLOW/'preparation'/('nine_hour_'+value['identity'][:16]+'.json')
    immutable(path,value)
    write_json(FOLLOW/'preparation/nine_hour_latest.json',{'manifest':str(path.relative_to(ROOT)),'identity':value['identity']})
    print({'manifest':str(path),'identity':value['identity']},flush=True)
    return path,value


def checked(path):
    value=identified(path)
    parent_path=ROOT/value['parent_manifest']
    if parent_path.resolve()!=PARENT.resolve() or digest(parent_path)!=value['parent_sha256']:
        raise ValueError('Parent manifest changed')
    parent=check_frozen(parent_path);probe=diagnostic(parent)
    if (value['parent_identity']!=parent['identity'] or value['amendment_sha256']!=digest(PLAN)
            or value['candidate']!=next(c for c in parent['candidates'] if c['name']=='A_foundation_raw')
            or value['seeds']!=list(SEEDS) or value['validation']!=list(QUARTERS)
            or value['total_coverage_equivalent_epochs']!=2 or value['Q2_Q3_scoring_authorized'] is not False
            or value['GPU_limit']!={'total_wall_hours':9,'anchor_utc':'2026-11-07T02:11:20+00:00','deadline_utc':guard.DEADLINE}
            or probe['identity']!=value['diagnostic_identity'] or digest(DIAGNOSTIC/'diagnostic_result.json')!=value['diagnostic_sha256']):
        raise ValueError('Nine-hour amendment or diagnostic binding changed')
    for name,sha in value['implementation'].items():
        if digest(ROOT/name)!=sha:raise ValueError('Frozen new GPU implementation changed: '+name)
    return value,parent,probe


def project(parent,probe,engineering_only=False):
    updates=3*parent['deployment_fit_policy']['total_updates']
    origins=0;fits=3
    if not engineering_only:
        updates+=3*sum(parent['folds'][q]['total_updates'] for q in QUARTERS)
        origins=3*sum(parent['folds'][q]['score_origins'] for q in QUARTERS);fits+=9
    copying=sum(r['bytes'] for name,r in parent['original_inputs'].items() if name.startswith('deployment/pre_Q2/'))
    copy_time=copying/probe['checkpoint_copy_bytes']*probe['checkpoint_copy_seconds']
    raw=updates*probe['seconds_per_update_upper']+origins*probe['prediction_seconds_per_origin_upper']+fits*probe['initialization_seconds']+copy_time
    return {'updates':updates,'full_fits':fits,'prediction_origins':origins,
        'raw_seconds':raw,'safety_multiplier':1.5,'required_seconds':raw*1.5+60,
        'copy_seconds':copy_time,'return_reserve_seconds':900}


def budget(parent,probe,engineering_only=False):
    projection=project(parent,probe,engineering_only)
    cutoff=datetime.fromisoformat(guard.DEADLINE).timestamp()-900
    remaining=cutoff-time.time()
    if not math.isfinite(projection['required_seconds']) or projection['required_seconds']>remaining:
        raise TimeoutError('Full fixed scope cannot fit remaining shortened window: '+repr({'projection':projection,'remaining':remaining}))
    return projection,remaining


def bind(attempt):
    request=identified(attempt/'guard_request.json');state=read_json(attempt/'guard_state.json')
    current=guard.process_identity()
    if (request['guard_sha256']!=digest(ROOT/'scripts/fusion_9h_guard.py') or request['purpose']!='exploration'
            or request['command']!=[sys.executable,str(ENTRY),*sys.argv[1:]]
            or state['stage']!='running' or state['request_identity']!=request['identity']
            or state['worker']!=current or state['guard']['pid']!=os.getppid()
            or not guard.exact_alive(state['guard']) or not guard.exact_alive(request['owner'])
            or os.getsid(0)!=os.getpid() or os.getpgid(0)!=os.getpid()
            or psutil.Process(os.getppid()).cmdline()[1:]!=[str(ROOT/'scripts/fusion_9h_guard.py'),'--request',str(attempt/'guard_request.json')]
            or time.time()>=request['stop_by_unix']):
        raise RuntimeError('GPU worker lacks its exact live shortened-deadline guard')
    return request


def review(manifest,parent,panel):
    from fusion.next.evaluation import seed_system,point_scores
    from scripts.review_fusion_pilot import full_prediction,original_seeds,robustness,time_segments
    run=FOLLOW/'runs'/manifest['identity'];config=manifest['candidate']['config']
    provenance={};parts=[];common={};targets=[]
    directory=run/'gpu/trials/focus_9h'/config['candidate_id']
    for seed in SEEDS:
        seed_parts=[]
        for q in QUARTERS:
            rows,p,_=full_prediction(directory,config,q,parent,panel,provenance,seed)
            if q not in common:common[q]=rows;targets.append(panel.targets['AAPL'][rows])
            elif not np.array_equal(common[q],rows):raise ValueError('Common origins changed')
            seed_parts.append(p)
        parts.append(np.concatenate(seed_parts))
    rows=np.concatenate([common[q] for q in QUARTERS]);target=np.concatenate(targets)
    quarters=np.concatenate([np.full(len(common[q]),q) for q in QUARTERS])
    scores,mean=seed_system(target,np.stack(parts));fixed=read_json(BASE/'system_freeze.json')
    controls={}
    for name,key in [('fixed_Chronos_scaled','original_frozen_predictions'),('original_LoRA','original_selected_predictions')]:
        values=original_seeds(BASE/parent[key],parent,panel,rows,quarters,target)
        control_scores,control_mean=seed_system(target,values)
        controls[name]={'scores':control_scores,'robustness':robustness(target,mean,control_mean,panel.dates[rows],time_segments(panel.stamps[rows]))}
    by_quarter={q:point_scores(target[quarters==q],mean[quarters==q]) for q in QUARTERS}
    gates={'at_least_one_percent_vs_zero':scores['system']['mse_improvement_vs_zero']>=.01,
        'beats_fixed_strongest_single':scores['system']['mse']<fixed['fixed_strongest_single']['mse'],
        'beats_fixed_strongest_simple':scores['system']['mse']<fixed['fixed_strongest_simple']['mse'],
        'at_least_two_positive_quarters':sum(r['mse_improvement_vs_zero']>0 for r in by_quarter.values())>=2,
        'all_three_seeds_positive':all(s['mse_improvement_vs_zero']>0 for s in scores['individual_seeds'].values())}
    result={'manifest_identity':manifest['identity'],'candidate':manifest['candidate']['name'],
        'seeds':list(SEEDS),'quarters':by_quarter,'selected':scores,'controls':controls,
        'fixed_strongest_single':fixed['fixed_strongest_single'],'fixed_strongest_simple':fixed['fixed_strongest_simple'],
        'development_checks':gates,'development_qualification_passed':all(gates.values()),
        'prediction_provenance_sha256':provenance,'Q2_Q3_scored':False,
        'independent_success_claimed':False,'foundation_ablation_complete':False}
    result['identity']=identity(result);immutable(run/'review.json',result)
    return result


def gpu_worker(path,attempt):
    request=bind(attempt);manifest,parent,probe=checked(path)
    if request['limit']!=manifest['GPU_limit']:raise ValueError('GPU guard limit changed')
    original_idle(parent,query_gpu=False)
    if runtime_signature()!=probe['runtime_signature']:raise ValueError('Diagnostic environment differs')
    budget(parent,probe)
    from scripts import portable_inference_runtime as numerical,future_speed_runtime as speed
    from fusion.next.panel import Panel
    from fusion.next.trainer import ExpertStore
    from fusion.next.experiment import neural_trial
    from scripts.fit_fusion_followup import original_experts,fit_and_export
    if numerical._installed or speed._installed:raise RuntimeError('Use a clean GPU interpreter')
    panel=Panel.load();store=ExpertStore(panel)
    if store.sources!=parent['expert_sources'] or store.key!=parent['expert_scope_key']:raise ValueError('OOF source changed')
    run=FOLLOW/'runs'/manifest['identity'];done=[]
    # The speed recorder captures its output root at installation. Restore it
    # before changing namespaces so engineering evidence cannot use the old fence.
    from fusion.next.trainer import Trainer
    original_run_until,original_windows=Trainer.run_until,Panel.windows
    try:
        with isolated_outputs(run/'gpu'):
            numerical.install();speed.install()
            for seed in SEEDS:
                for q in QUARTERS:
                    result,_=neural_trial(panel,manifest['candidate']['config'],q,seed,1.,'focus_9h',store,'fusion_search')
                    done.append({'seed':seed,'quarter':q,'prediction_sha256':result['prediction_sha256']})
                    write_json(run/'progress.json',{'stage':'development_training','completed_fits':done,'expected_fits':9,'updated_at_utc':utc_now()})
    finally:
        speed.clear();Trainer.run_until=original_run_until;Panel.windows=original_windows;speed._installed=False
    result=review(manifest,parent,panel)
    budget(parent,probe,engineering_only=True)
    frozen={'schema':1,'artifact_type':'nine_hour_development_engineering_freeze',
        'manifest_identity':manifest['identity'],'source_identity':parent['source_identity'],
        'plan_sha256':parent['plan_sha256'],'development_review_identity':result['identity'],
        'selected_candidate':manifest['candidate']['name'],'config':manifest['candidate']['config'],
        'development_qualification_passed':result['development_qualification_passed'],
        'fit_policy':parent['deployment_fit_policy'],'original_experts':original_experts(parent),
        'expert_sources':parent['expert_sources'],'limit':parent['limit'],'GPU_limit':manifest['GPU_limit'],
        'engineering_fit_only':True,'actual_CPU_acceptance_passed':False,
        'Q2_Q3_scoring_authorized':False,'independent_research_success_claimed':False}
    frozen['identity']=identity(frozen);freeze_path=run/'delivery/engineering_freeze.json';immutable(freeze_path,frozen)
    # Reuse the same numeric/input adapters, with a new fenced output namespace.
    with isolated_outputs(run/'delivery_fit'):
        speed.install()
        fit=fit_and_export(panel,store,frozen,freeze_path,run/'delivery_fit')
    completion={'stage':'GPU_complete_exported_actual_local_acceptance_required',
        'manifest_identity':manifest['identity'],'completed_validation_fits':9,
        'completed_engineering_seeds':list(SEEDS),'model_identity':fit['model_identity'],
        'review_identity':result['identity'],'actual_CPU_acceptance_passed':False,
        'Q2_Q3_scored':False,'independent_success_claimed':False,'completed_at_utc':utc_now()}
    completion['identity']=identity(completion);immutable(run/'gpu_completion.json',completion)
    return completion


def bundle(path,manifest):
    run=FOLLOW/'runs'/manifest['identity'];complete=identified(run/'gpu_completion.json')
    directory=run/'return';directory.mkdir(exist_ok=True)
    receipt_path=directory/'receipt.json';archive=directory/'bundle.tar'
    if receipt_path.exists():
        receipt=identified(receipt_path)
        if digest(archive)!=receipt['sha256']:raise ValueError('Existing return differs')
        return receipt
    members={}
    for top in ('gpu/trials','gpu/training','gpu/usage_events','delivery','delivery_fit/training','delivery_fit/deployment'):
        for source in sorted((run/top).rglob('*')):
            if source.is_symlink() or not source.is_file():continue
            relative=str(source.relative_to(ROOT))
            members[relative]={'sha256':digest(source),'bytes':source.stat().st_size}
    for source in (path,run/'review.json',run/'gpu_completion.json',run/'delivery_fit/fit_result.json',
            run/'attempts/gpu_01/guard_request.json',run/'attempts/gpu_01/guard_state.json',
            DIAGNOSTIC/'diagnostic_result.json',DIAGNOSTIC/'guard_request.json',DIAGNOSTIC/'guard_state.json'):
        members[str(source.relative_to(ROOT))]={'sha256':digest(source),'bytes':source.stat().st_size}
    partial=directory/'bundle.partial.tar'
    if partial.exists():partial.rename(directory/('interrupted_'+str(time.time_ns())+'.tar'))
    with tarfile.open(partial,'w') as tar:
        for relative,row in members.items():
            source=ROOT/relative
            if digest(source)!=row['sha256']:raise ValueError('Source changed while archiving')
            tar.add(source,arcname=relative,recursive=False)
    partial.rename(archive)
    receipt={'schema':1,'manifest_identity':manifest['identity'],'archive':str(archive.relative_to(ROOT)),
        'sha256':digest(archive),'bytes':archive.stat().st_size,'members':members,
        'GPU_completion_identity':complete['identity'],'actual_CPU_acceptance_passed':False,
        'Q2_Q3_scored':False,'created_at_utc':utc_now()}
    receipt['identity']=identity(receipt);write_json(receipt_path,receipt)
    return receipt


def launch(path):
    manifest,parent,probe=checked(path);run=FOLLOW/'runs'/manifest['identity']
    attempt=run/'attempts/gpu_01'
    if attempt.exists():raise RuntimeError('GPU attempt already exists; inspect exact guard/checkpoints before recovery')
    with execution_locks():
        original_idle(parent)
        if runtime_signature()!=probe['runtime_signature']:raise ValueError('Runtime signature changed')
        projection,remaining=budget(parent,probe)
        attempt.mkdir(parents=True)
        write_json(attempt/'admission.json',{'manifest_identity':manifest['identity'],'projection':projection,
            'remaining_seconds':remaining,'admitted_at_utc':utc_now()})
        process,receipt=guard.start(attempt,[sys.executable,str(ENTRY),'--mode','gpu_worker','--manifest',str(path),
            '--attempt',str(attempt)],manifest['GPU_limit'],min(9*3600,remaining))
        write_json(run/'scheduler_dispatch.json',{'scheduler':guard.process_identity(),'guard':receipt,'manifest_identity':manifest['identity']})
        code=process.wait()
        if code:raise RuntimeError('GPU stage stopped; retain checkpoint and all guard evidence')
        if read_json(attempt/'guard_state.json')['stage']!='completed':raise RuntimeError('Guard did not complete')
        print(bundle(path,manifest),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('freeze','run','gpu_worker'),required=True)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--attempt',type=Path)
    args=parser.parse_args()
    if args.mode=='freeze':freeze()
    elif args.mode=='run':launch(args.manifest.resolve())
    else:gpu_worker(args.manifest.resolve(),args.attempt.resolve())
