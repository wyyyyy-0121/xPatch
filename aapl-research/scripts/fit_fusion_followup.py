"""Fit the one reviewed follow-up system and export it without final-period scoring.

Original experts are reused only with exact recipes, fit boundaries and bytes.
This entry stops at export; a returned model still needs actual local CPU F8.
"""
import argparse
from datetime import datetime,timezone
import gc
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from fusion.next.common import ROOT,OUT as BASE,PLAN,digest,identity,read_json,utc_now,write_json
from fusion.next.recipe import for_fit
from scripts import followup_guard as guard
from scripts.followup_admission import check_frozen,checked_file,identified
from scripts.followup_runtime import FOLLOW,DEADLINE,EARLIEST,isolated_outputs
from scripts.review_fusion_extension import checked_review
from scripts.run_fusion_followup import (original_idle,execution_locks,runtime_signature,
    probe_table,stage_remaining,attempts,bind_guarded_worker,stop_guard)


ENTRY = Path(__file__).resolve()


def original_experts(manifest):
    """Check actual exported experts, not merely a matching model family name."""
    from fusion.next.inference import checked_manifest
    from fusion.next.experts import NAMES
    policy = manifest['deployment_fit_policy']
    def bound(relative):
        path = checked_file(BASE,relative);row = manifest['original_inputs'].get(relative)
        if row is None or digest(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:
            raise ValueError('Expert reuse source is not bound to the frozen matrix: '+relative)
        return path
    directory = 'deployment/pre_Q2'
    model = checked_manifest(bound(directory+'/deployment.json').parent)
    if set(model['experts'])!=set(NAMES):raise ValueError('Reuse requires all four original experts')
    reference = manifest['reference'];experts,files = {},{}
    for name in NAMES:
        selection = read_json(bound('selections/single_'+name+'.json'))['selected']
        expected = for_fit({**selection,'information':reference.get('information','ohlcv'),
            'training_scope':reference.get('training_scope','multistock'),
            'market_information':reference.get('market_information',True)},policy['fit_end'])
        entry = model['experts'][name]
        if entry['config']!=expected:raise ValueError('Original expert recipe or two-epoch fit differs: '+name)
        if name in ('xpatch','patchtst'):
            names = [entry['checkpoint'],entry['checkpoint']+'.json']
            receipt = read_json(bound(directory+'/'+names[1]));recipe = receipt['recipe']
            if (recipe['config']!=expected or recipe['fit_end']!=policy['fit_end']
                    or recipe['source_identity']!=manifest['source_identity']
                    or recipe['plan_sha256']!=manifest['plan_sha256'] or receipt['step']!=expected['total_updates']
                    or receipt['sha256']!=digest(bound(directory+'/'+names[0]))):
                raise ValueError('Original neural expert is incomplete or has another fit boundary')
            experts[name] = {'config':expected,'checkpoint':directory+'/'+entry['checkpoint']}
        else:
            names = [entry['directory']+'/'+part for part in
                     ('summary.json','scalers.json','ridge.npz' if name=='ridge' else 'lightgbm.txt')]
            summary = read_json(bound(directory+'/'+names[0]))
            if (summary['config']!=expected or summary['fit_end']!=policy['fit_end']
                    or summary['model_sha256']!=digest(bound(directory+'/'+names[-1]))):
                raise ValueError('Original CPU expert has another recipe, model or fit boundary')
            experts[name] = {'config':expected,'directory':directory+'/'+entry['directory']}
        for name_in_model in names:
            relative = directory+'/'+name_in_model;source = bound(relative)
            files[relative] = {'sha256':digest(source),'bytes':source.stat().st_size}
    return {'model_identity':model['identity'],'deployment_manifest_sha256':digest(BASE/directory/'deployment.json'),
            'stocks':model['stocks'],'background':model['background'],'experts':experts,'files':files}


def freeze_delivery(manifest_path,manifest=None):
    manifest = manifest or check_frozen(manifest_path)
    saved,review = checked_review(manifest_path,manifest)
    name = review['selected_candidate']
    selected = next((r for r in manifest['candidates'] if r['name']==name),None)
    if (selected is None or review['extension_plan']['selected_candidate']!=name
            or review['seeds']!=[2026,2027,2028] or review['total_coverage_equivalent_epochs']!=2
            or review['Q2_Q3_prediction_arrays_loaded'] is not False
            or review['independent_research_success_claimed'] is not False
            or review['development_checks_authorize_final_data_access'] is not False):
        raise ValueError('Engineering fit needs the fixed complete three-seed review')
    body = {'schema':1,'artifact_type':'followup_engineering_fit_freeze','manifest_identity':manifest['identity'],
            'source_identity':manifest['source_identity'],'plan_sha256':manifest['plan_sha256'],
            'extension_review':saved,'selected_candidate':name,'config':selected['config'],
            'development_qualification_passed':review['development_qualification_passed'],
            'fit_policy':manifest['deployment_fit_policy'],'original_experts':original_experts(manifest),
            'expert_sources':manifest['expert_sources'],'limit':manifest['limit'],
            'independent_research_success_claimed':False,'Q2_Q3_scoring_authorized':False,
            'engineering_fit_only':True,'actual_CPU_acceptance_passed':False}
    path = FOLLOW/'runs'/manifest['identity']/'delivery/engineering_freeze.json'
    if path.exists():
        old = identified(path)
        if {k:v for k,v in old.items() if k not in ('identity','frozen_at_utc')}!=body:
            raise ValueError('Preserve the existing engineering freeze; its recipe or evidence differs')
        return path,old
    value = {**body,'frozen_at_utc':utc_now()};value['identity'] = identity(value)
    write_json(path,value)
    return path,value


def system_config(frozen,seed):
    return for_fit({**frozen['config'],'seed':seed},frozen['fit_policy']['fit_end'])


def checkpoint_path(destination,frozen,seed):
    return Path(destination)/'training/delivery_final'/frozen['config']['candidate_id']/('seed_'+str(seed)+'.pt')


def checkpoint_progress(destination,frozen):
    from fusion.next.checkpoint import resolve_checkpoint
    progress = {}
    for seed in (2026,2027,2028):
        path = checkpoint_path(destination,frozen,seed)
        if not path.with_suffix('.pt.json').exists():
            progress[str(seed)] = {'step':0,'checkpoint_sha256':None};continue
        committed,receipt = resolve_checkpoint(path)
        state = torch.load(committed,map_location='cpu',weights_only=True);recipe = receipt['recipe']
        if (recipe['config']!=system_config(frozen,seed) or recipe['fit_end']!=frozen['fit_policy']['fit_end']
                or recipe['seed']!=seed or recipe['source_identity']!=frozen['source_identity']
                or recipe['plan_sha256']!=frozen['plan_sha256'] or recipe['expert_sources']!=frozen['expert_sources']
                or state['recipe']!=recipe or state['step']!=receipt['step']
                or type(state['step']) is not int or not 0<=state['step']<=frozen['fit_policy']['total_updates']):
            raise ValueError('Engineering checkpoint changes recipe, fit boundary, seed or update count')
        progress[str(seed)] = {'step':state['step'],'checkpoint_sha256':receipt['sha256']}
    return progress


def delivery_budget(manifest,run,frozen,signature,now=None,current_attempt=None):
    now = now or datetime.now(timezone.utc)
    if now<datetime.fromisoformat(EARLIEST):raise RuntimeError('Next large computation is scheduled for November 8')
    if (frozen['limit']!=manifest['limit'] or manifest['limit']['deadline_utc']!=DEADLINE
            or manifest['limit']['total_wall_hours']!=100):
        raise ValueError('Original absolute deadline differs')
    destination = run/'delivery_fit'
    if (destination/'fit_result.json').exists():raise RuntimeError('Engineering fit already completed; return and verify it')
    table = probe_table(run,manifest,signature);measured = table[frozen['selected_candidate']]
    for key in ('checkpoint_copy_bytes','checkpoint_copy_seconds_upper'):
        if key not in measured or not np.isfinite(measured[key]) or measured[key]<=0:
            raise ValueError('Need actual checkpoint copy bytes and timing from the admitted CUDA probe')
    progress = checkpoint_progress(destination,frozen)
    total = frozen['fit_policy']['total_updates'];pending = sum(total-r['step'] for r in progress.values())
    unfinished = sum(r['step']<total for r in progress.values())
    # Adam state is initialized in the probe. A factor of two allows the larger
    # final training sampler state; this is a projection, not an IO guarantee.
    copy_bytes = sum(r['bytes'] for r in frozen['original_experts']['files'].values())+6*measured['checkpoint_copy_bytes']
    copy_seconds = copy_bytes*measured['checkpoint_copy_seconds_upper']/measured['checkpoint_copy_bytes']
    seconds = (unfinished*measured['initialization_seconds_upper']+pending*measured['seconds_per_update_upper']+copy_seconds)
    left = (datetime.fromisoformat(DEADLINE)-now).total_seconds()
    if current_attempt is None:
        remaining = stage_remaining(run,'fit')
    else:
        from scripts.run_fusion_followup import spent_seconds
        current_attempt = Path(current_attempt).resolve()
        known = attempts(run,'fit')
        if current_attempt/'guard_request.json' not in [p.resolve() for p in known]:
            raise ValueError('Current engineering reservation is outside its fixed fit attempts')
        previous = [p for p in known if p.parent.resolve()!=current_attempt and p.exists()]
        current = identified(current_attempt/'guard_request.json')
        if (current['purpose']!='delivery_fit' or current['limit']!=manifest['limit']
                or current['delivery_reserve_seconds']!=guard.CPU_DELIVERY_RESERVE_SECONDS
                or not np.isfinite(current['stage_seconds']) or not 0<current['stage_seconds']<=guard.MAX_STAGE_SECONDS
                or not np.isfinite(current['requested_at_unix']) or current['requested_at_unix']>now.timestamp()):
            raise ValueError('Current engineering reservation changes purpose, cutoff or cumulative cap')
        elapsed = now.timestamp()-current['requested_at_unix']
        remaining = max(0,min(current['stage_seconds'],
            guard.MAX_STAGE_SECONDS-sum(spent_seconds(p,guard.MAX_STAGE_SECONDS) for p in previous))-elapsed)
    required = 1.5*seconds+60
    if required>remaining:raise RuntimeError('Engineering fit exceeds its remaining cumulative six-hour cap')
    if required+guard.CPU_DELIVERY_RESERVE_SECONDS>left:
        raise RuntimeError('Engineering fit leaves insufficient time for the two-hour local delivery reserve')
    return {'mode':'delivery_fit','stage_seconds':remaining,'checkpoint_progress':progress,
            'projection':{'seconds':seconds,'pending_updates':pending,'unfinished_seeds':unfinished,
                          'export_copy_bytes_upper':copy_bytes,'export_copy_seconds':copy_seconds,
                          'checkpoint_size_projection_multiplier':2,'margin_multiplier':1.5},
            'local_delivery_reserve_seconds':guard.CPU_DELIVERY_RESERVE_SECONDS,'remaining_absolute_seconds':left,
            'budget_resets_on_retry':False,'new_research_evaluation_authorized':False}


def verify_export(destination,frozen,freeze_sha256,systems):
    from fusion.next.inference import checked_manifest
    from fusion.next.checkpoint import resolve_checkpoint
    value = checked_manifest(destination)
    if (value['freeze_sha256']!=freeze_sha256 or value['stocks']!=frozen['original_experts']['stocks']
            or value['background']!=frozen['original_experts']['background']
            or set(value['experts'])!=set(frozen['original_experts']['experts'])
            or len(value['system_checkpoints'])!=3):raise ValueError('Engineering export identity or model membership differs')
    for name,source in frozen['original_experts']['experts'].items():
        target = value['experts'][name]
        if target['config']!=source['config']:raise ValueError('Export changed an original expert recipe')
        if 'checkpoint' in source:
            pairs = [(source['checkpoint'],target['checkpoint']),(source['checkpoint']+'.json',target['checkpoint']+'.json')]
        else:
            pairs = [(source['directory']+'/'+part,target['directory']+'/'+part) for part in
                     ('summary.json','scalers.json','ridge.npz' if name=='ridge' else 'lightgbm.txt')]
        for original,copy in pairs:
            if value['files'][copy]!=frozen['original_experts']['files'][original]:
                raise ValueError('Exported expert bytes differ from original admitted fits')
    for seed,source,target in zip((2026,2027,2028),systems,value['system_checkpoints']):
        committed,receipt = resolve_checkpoint(source['checkpoint'])
        if (target['config']!=system_config(frozen,seed) or target['checkpoint']!='system/seed_'+str(seed)+'.pt'
                or value['files'][target['checkpoint']]['sha256']!=receipt['sha256']
                or value['files'][target['checkpoint']+'.json']['sha256']!=digest(Path(source['checkpoint']).with_suffix('.pt.json'))
                or committed.stat().st_size!=value['files'][target['checkpoint']]['bytes']):
            raise ValueError('Exported system is not the exact three fresh engineering fits')
    return value


def fit_and_export(panel,store,frozen,freeze_path,destination,device='cuda'):
    """Run only after admission; CPU is exposed solely for isolated integration tests."""
    from fusion.next.experiment import model_for
    from fusion.next.trainer import Trainer
    from fusion.next.hardware import runtime
    from fusion.next.accounting import Usage
    from fusion.next.inference import export
    progress = checkpoint_progress(destination,frozen);systems = []
    for seed in (2026,2027,2028):
        config = system_config(frozen,seed);path = checkpoint_path(destination,frozen,seed)
        if progress[str(seed)]['step']<config['total_updates']:
            actual = runtime(seed)
            if actual!=device:raise RuntimeError('Engineering worker device differs from its admitted runtime')
            with Usage('final_and_deployment','followup '+frozen['selected_candidate']+' seed '+str(seed),gpu=device=='cuda') as usage:
                model = model_for(panel,config,device)
                trainer = Trainer(panel,model,config,frozen['fit_policy']['fit_end'],
                                  'delivery_final/'+config['candidate_id'],seed,store,device)
                if path.with_suffix('.pt.json').exists():trainer.resume()
                trainer.run_until(config['total_updates'],usage)
                if trainer.path!=path:raise ValueError('Engineering trainer left its isolated checkpoint namespace')
                del trainer,model
            gc.collect()
            if device=='cuda':torch.cuda.empty_cache()
        systems.append({'config':config,'checkpoint':str(path)})
        write_json(Path(destination)/'fit_progress.json',{'engineering_freeze_identity':frozen['identity'],
                   'complete_seeds':list(range(2026,seed+1)),'expected_seeds':[2026,2027,2028],
                   'updated_at_utc':utc_now(),'Q2_Q3_scored':False})
    if any(r['step']!=frozen['fit_policy']['total_updates'] for r in checkpoint_progress(destination,frozen).values()):
        raise ValueError('A fresh engineering seed remains incomplete')
    target = Path(destination)/'deployment/pre_Q2'
    if target.exists() and not (target/'deployment.json').exists():
        preserved = target.parent/('incomplete_export_'+str(time.time_ns()));target.rename(preserved)
        write_json(Path(destination)/'export_recovery'/(str(time.time_ns())+'.json'),
                   {'preserved_directory':str(preserved.relative_to(ROOT)),
                    'reason':'Interrupted export had no committed deployment manifest; original bytes retained'})
    references = {name:{k:(str(BASE/v) if k in ('checkpoint','directory') else v) for k,v in row.items()}
                  for name,row in frozen['original_experts']['experts'].items()}
    if not target.exists():export(target,panel.stocks,panel.background,references,systems,digest(freeze_path))
    exported = verify_export(target,frozen,digest(freeze_path),systems)
    result = {'schema':1,'manifest_identity':frozen['manifest_identity'],'engineering_freeze_identity':frozen['identity'],
              'engineering_freeze_sha256':digest(freeze_path),'model_identity':exported['identity'],
              'deployment_directory':str(target.relative_to(ROOT)),
              'deployment_manifest_sha256':digest(target/'deployment.json'),
              'stage':'engineering_fit_export_complete_local_acceptance_required','fresh_system_seeds':[2026,2027,2028],
              'fit_end':frozen['fit_policy']['fit_end'],'total_coverage_equivalent_epochs':2,
              'original_expert_bytes_reused':True,'development_qualification_passed':frozen['development_qualification_passed'],
              'Q2_Q3_scored':False,'actual_CPU_acceptance_passed':False,
              'independent_research_success_claimed':False,'completed_at_utc':utc_now()}
    result['identity'] = identity(result);write_json(Path(destination)/'fit_result.json',result)
    return result


def worker(manifest_path,attempt):
    request = bind_guarded_worker(attempt,ENTRY,'delivery_fit');manifest = check_frozen(manifest_path)
    original_idle(manifest,query_gpu=False)
    from scripts import portable_inference_runtime as numerical,future_speed_runtime as speed
    if numerical._installed or speed._installed:raise RuntimeError('Use a clean independent engineering worker')
    freeze_path,frozen = freeze_delivery(manifest_path,manifest)
    run = FOLLOW/'runs'/manifest['identity'];destination = run/'delivery_fit'
    signature = runtime_signature();admission = identified(Path(attempt)/'launch_admission.json')
    budget = delivery_budget(manifest,run,frozen,signature,current_attempt=attempt)
    # Recompute useful-work costs while keeping the scheduler's original time
    # reservation. Startup elapsed time cannot grant a new stage budget.
    if (request['limit']!=manifest['limit'] or admission['mode']!='delivery_fit'
            or admission['manifest_identity']!=manifest['identity'] or admission['entry_sha256']!=digest(ENTRY)
            or admission['engineering_freeze_sha256']!=digest(freeze_path) or admission['runtime_signature']!=signature
            or request['stage_seconds']!=admission['budget']['stage_seconds']
            or budget['checkpoint_progress']!=admission['budget']['checkpoint_progress']
            or budget['projection']!=admission['budget']['projection']):
        raise ValueError('Engineering worker admission changed before model loading')
    from fusion.next.panel import Panel
    from fusion.next.trainer import ExpertStore
    panel = Panel.load();store = ExpertStore(panel)
    if (store.sources!=manifest['expert_sources'] or store.key!=manifest['expert_scope_key']
            or panel.stocks!=frozen['original_experts']['stocks'] or panel.background!=frozen['original_experts']['background']):
        raise ValueError('Engineering OOF, stock or background ordering changed')
    with isolated_outputs(destination):
        numerical.install();speed.install()
        return fit_and_export(panel,store,frozen,freeze_path,destination)


def launch(manifest_path):
    manifest_path = Path(manifest_path).resolve();manifest = check_frozen(manifest_path)
    with execution_locks():
        from scripts import portable_inference_runtime as numerical,future_speed_runtime as speed
        if numerical._installed or speed._installed:raise RuntimeError('Use a clean independent engineering launcher')
        original_idle(manifest);signature = runtime_signature()
        freeze_path,frozen = freeze_delivery(manifest_path,manifest)
        run = FOLLOW/'runs'/manifest['identity'];budget = delivery_budget(manifest,run,frozen,signature)
        attempt = run/'attempts'/('fit_attempt_'+str(len(attempts(run,'fit'))+1).zfill(2))
        if attempt.exists():raise RuntimeError('Preserve the existing partial engineering launch')
        admission = {'manifest_identity':manifest['identity'],'mode':'delivery_fit','runtime_signature':signature,
                     'engineering_freeze_sha256':digest(freeze_path),'budget':budget,'entry_sha256':digest(ENTRY),
                     'issued_at_utc':utc_now()}
        admission['identity'] = identity(admission);write_json(attempt/'launch_admission.json',admission)
        command = [sys.executable,str(ENTRY),'--worker','--manifest',str(manifest_path),'--attempt',str(attempt)]
        os.environ['PYTHONPATH'] = str(ROOT)
        process,dispatch = guard.start(attempt,command,manifest['limit'],budget['stage_seconds'],purpose='delivery_fit')
        try:code = process.wait(timeout=max(1,dispatch['stop_by_unix']-time.time()+10))
        finally:stop_guard(process,attempt)
        state = read_json(attempt/'guard_state.json')
        if code!=0 or state['stage']!='completed':raise RuntimeError('Engineering guard stopped or failed; inspect retained evidence')
        result = identified(run/'delivery_fit/fit_result.json')
        if result['engineering_freeze_sha256']!=digest(freeze_path):raise ValueError('Export completion belongs to another freeze')
        print({'fit_export_complete':True,'model_identity':result['model_identity'],'actual_CPU_acceptance_passed':False})
        return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--attempt',type=Path,help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if not args.attempt:parser.error('Internal workers require their exact guard attempt')
        worker(args.manifest,args.attempt)
    else:
        if args.attempt:parser.error('The engineering attempt directory is fixed by its matrix')
        launch(args.manifest)
