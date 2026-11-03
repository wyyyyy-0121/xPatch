"""Finalize genuine forward evidence while preserving the historical conclusion."""
from datetime import datetime,timezone
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from .io import ROOT,read_json,write_json,sha,stable_hash
from .observation import BarStore,Journal,complete_days,immutable_json,session_origins,stamp


def prospective_evidence(config, run_directory):
    """Rebuild dates, labels, scores and gates; a summary's passed flag is insufficient."""
    from .prospective import read_plan,observation_directory
    from .verify import reference_targets,reference_score,reference_mz,reference_comparison,assert_reference_equal
    lock=read_plan(config);payload=lock['payload'];settings=payload['settings']
    if str(Path(run_directory).relative_to(ROOT))!=payload['run_directory']:
        raise ValueError('Forward evidence belongs to another historical run')
    directory=observation_directory(lock);journal=Journal(directory,lock['signature'])
    quality=complete_days(journal.events,settings)
    if not quality['ready_for_final_evaluation']:
        raise ValueError('Final acceptance requires 60 actual complete qualified prospective sessions')
    if settings['minimum_complete_sessions'] != 60:
        raise ValueError('Final acceptance cannot shorten the registered 60-session minimum')
    dates=quality['complete_dates'][:settings['minimum_complete_sessions']]
    expected=[t.isoformat() for day in dates for t in session_origins(day)]
    admission_path=directory/'admission.json';admission=read_json(admission_path)
    if (admission.get('passed') is not True or admission.get('http_status')!=200 or
            admission.get('plan_signature')!=lock['signature'] or
            admission.get('status')!='timely_sip_admitted' or
            admission.get('recent_all_three_assets') is not True or admission.get('regular_session') is not True):
        raise ValueError('Actual timely SIP admission is required before prospective issuance')
    admitted=next((e for e in journal.events if e['hash']==admission['last_event_hash']),None)
    if admitted is None or admitted['kind']!='bars' or admitted['payload']['purpose']!='admission':
        raise ValueError('Admission does not bind an actual SIP response')
    admitted_at=stamp(admission['checked_utc'])
    if not stamp(admitted['recorded_at'])<=admitted_at<stamp(expected[0]):
        raise ValueError('Timely admission must precede actual prospective issuance')
    admission_store=BarStore()
    for event in journal.events[:admitted['sequence']]:
        if event['kind']=='bars' and event['payload']['purpose']=='admission':admission_store.apply(event)
    end=stamp(admitted['payload']['end'])
    if not all(admission_store.bars[a] and max(admission_store.bars[a])>=end-pd.Timedelta(minutes=1) for a in ('AAPL','SPY','QQQ')):
        raise ValueError('Admission did not receive timely bars for all three assets')
    admitted_grid=session_origins(admitted_at.tz_convert('America/New_York').strftime('%Y-%m-%d'))
    if (not len(admitted_grid) or not admitted_grid[0]-pd.Timedelta(minutes=15)<=admitted_at<=admitted_grid[-1]+pd.Timedelta(minutes=30) or
            not admitted_at.floor('min')-pd.Timedelta(minutes=2)<=end<=admitted_at):
        raise ValueError('The admitted response was stale or outside a regular exchange session')
    selected=set(dates);store=BarStore();records=[]
    for event in journal.events:
        if stamp(event['recorded_at'])<stamp(payload['created_utc']):
            raise ValueError('Observation predates the frozen prospective protocol')
        if event['kind']=='bars':store.apply(event)
        elif event['kind']=='prediction':
            p=event['payload'];origin=stamp(p['origin'])
            if origin.tz_convert('America/New_York').strftime('%Y-%m-%d') not in selected:continue
            if (p['status']!='issued' or p['data_event_hash']!=store.last_event_hash or
                    stamp(p['received_at'])!=store.last_received_at):
                raise ValueError('Prediction differs from its actual received-data prefix')
            if set(p['predictions'])!=set(payload['models']):raise ValueError('Prospective model family changed')
            for key,entry in payload['models'].items():
                if p['predictions'][key]['source_result_sha256']!=entry['source_result_sha256']:
                    raise ValueError('Prospective prediction used another frozen model')
            records.append(p)
    if [p['origin'] for p in records]!=expected:
        raise ValueError('The earliest complete prospective grid differs from its declared sources')
    times=pd.DatetimeIndex(expected).as_unit('ns')
    day_labels=np.asarray(times.tz_convert('America/New_York').strftime('%Y-%m-%d'))
    closes=pd.Series(store.first_live_aapl_closes,dtype=float).sort_index()
    targets=reference_targets(closes,times)
    values,descriptions={},{}
    for key in payload['models']:
        mean=np.asarray([p['predictions'][key]['rv_mean'] for p in records])
        levels=config['calibration']['quantiles']
        if any(set(p['predictions'][key]['quantiles'])!={str(q) for q in levels} for p in records):
            raise ValueError('Prospective quantile levels changed')
        quantiles=np.asarray([[p['predictions'][key]['quantiles'][str(q)] for q in levels] for p in records])
        if (not np.isfinite(mean).all() or (mean<=0).any() or not np.isfinite(quantiles).all() or
                (quantiles<=0).any() or (np.diff(quantiles,axis=1)<0).any()):
            raise ValueError('Nonfinite or nonpositive prospective predictions')
        values[key]={'origins':np.arange(len(records)),'timestamps':times.asi8,'dates':day_labels,'y':targets,
            'mean':mean,'levels':levels,'quantiles':quantiles,'seeds':{'2026':mean}}
        descriptions[key]={**reference_score(values[key]),'mincer_zarnowitz':reference_mz(targets,mean,day_labels)}
    comparisons={}
    for key in settings['candidate_keys']:
        item=reference_comparison(values[key],values[key.split('/')[0]+'/har'],settings)
        comparisons[key]={field:item[field] for field in ('mean_difference','dm','bootstraps','p_conservative')}
    family=[f'{panel}/{name}' for panel in config['feature_sets'] for name in config['statistics']['candidate_models']]
    ordered=sorted((comparisons[k]['p_conservative'] if k in comparisons else 1.,k) for k in family)
    greatest=0.
    for number,(p,key) in enumerate(ordered):
        greatest=max(greatest,(len(family)-number)*p)
        if key in comparisons:
            item=comparisons[key];item['holm_p']=min(1.,greatest)
            item['passed_predictive_gate']=bool(item['mean_difference']<0 and item['holm_p']<settings['alpha'] and all(b['ci95'][1]<0 for b in item['bootstraps']))
    qualified=[k for k in settings['primary_candidate_preference'] if comparisons[k]['passed_predictive_gate']]
    proofs={};previous_completed=max(stamp(e['recorded_at']) for e in journal.events)
    for number in (1,2):
        path=directory/f'verify_{number}.json';proof=read_json(path)
        if (proof.get('passed') is not True or proof.get('round')!=number or proof.get('plan_signature')!=lock['signature'] or
                proof.get('last_event_hash')!=journal.last_hash or proof.get('sessions')!=len(dates) or
                proof.get('origins')!=len(expected) or proof.get('first_60_complete_dates')!=dates or
                proof.get('full_model_replay') is not True or proof.get('all_labels_independently_rebuilt') is not True or
                stamp(proof['completed_utc'])<previous_completed):
            raise ValueError('Both full independent forward rounds must match the exact observation chain and grid')
        assert_reference_equal(proof['descriptions'],descriptions,'prospective_descriptions')
        assert_reference_equal(proof['comparisons'],comparisons,'prospective_comparisons')
        assert_reference_equal(proof['qualified_candidates'],qualified,'prospective_qualified_candidates')
        if proof['primary_model']!=(qualified[0] if qualified else None):raise ValueError('Primary selection violates the registered information-set preference')
        proofs[str(number)]={'path':str(path.relative_to(ROOT)),'sha256':sha(path)}
        previous_completed=stamp(proof['completed_utc'])
    return {'plan_signature':lock['signature'],'last_event_hash':journal.last_hash,'complete_dates':dates,
        'sessions':len(dates),'origins':len(expected),'verifications':proofs,'qualified_candidates':qualified,
        'primary_model':qualified[0] if qualified else None,'descriptions':descriptions,'comparisons':comparisons,
        'admission':{'path':str(admission_path.relative_to(ROOT)),'sha256':sha(admission_path)},
        'historical_prerequisites':payload['historical_prerequisites'],'not_before_utc':previous_completed.isoformat()}


def prospective_artifact_index(lock, evidence):
    """Bind every raw response, event, and replay proof, including excluded days."""
    from .prospective import observation_directory
    directory=observation_directory(lock);journal=Journal(directory,lock['signature'])
    if journal.last_hash!=evidence['last_event_hash']:
        raise ValueError('Observation changed during finalization')
    paths=list((directory/'events').glob('*.json'))+[directory/'admission.json']
    paths += [ROOT/entry['path'] for entry in evidence['verifications'].values()]
    return {'schema_version':1,'plan_signature':lock['signature'],'last_event_hash':journal.last_hash,
        'event_count':len(journal.events),'raw_responses_embedded_in_event_files':True,
        'files':{str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}}


def selected_model(lock,evidence):
    qualified=bool(evidence['qualified_candidates'])
    key=evidence['primary_model'] if qualified else 'aapl/har'
    if qualified and key!=next(k for k in lock['payload']['settings']['primary_candidate_preference'] if k in evidence['qualified_candidates']):
        raise ValueError('The final model was chosen by prospective score ranking')
    entry=lock['payload']['models'][key]
    return {**entry,'schema_version':1,'signature':lock['payload']['base_signature'],
        'outcome':'completed_prospective_validation' if qualified else 'completed_no_prospective_edge',
        'role':'prospectively_qualified_rv_model' if qualified else 'reference_har_baseline_after_prospective_failure',
        'historically_qualified':qualified,'active_predictive_edge_established':qualified,'prospective_qualified':qualified,
        'prospective_plan_signature':lock['signature'],'direction_or_profit_claim':False,'default_mode':'historical_replay',
        'selection_policy':'registered_primary_information_set_preference_after_full_forward_evaluation',
        'prerequisites':lock['payload']['historical_prerequisites'],'prospective_evidence':evidence}


def accept_forward(config,directory):
    from .acceptance import artifact_index,search_coverage
    from .confirmation import confirmation_evidence
    from .evaluation import evaluate
    from .inference import infer
    from .provenance import verify_freeze
    from .prospective import read_plan
    from .prospective import observation_directory
    from .forward_runtime import resource_evidence
    from .verify import require_verifications
    checks,errors={},[];manifest=None;directory=Path(directory)
    try:
        # This gate runs first, so an early refusal cannot relabel the candidate
        # or overwrite the existing historical report and inference example.
        evidence=prospective_evidence(config,directory)
        lock=read_plan(config);frozen=verify_freeze(config)
        from .forward_runtime import ACTIVE
        if ACTIVE is None or ACTIVE.action not in ('finalize','accept'):
            raise ValueError('Final acceptance requires the bounded finance.forward_runtime launcher')
        checks['two_complete_actual_prospective_verifications']=evidence['sessions']>=60
        journal=Journal(observation_directory(lock),lock['signature'])
        resources=resource_evidence(config,lock,journal,evidence['verifications'])
        checks['all_forward_usage_metered_within_original_budget']=resources['passed']
        forward_index=prospective_artifact_index(lock,evidence)
        forward_index['resource_files']=resources['files']
        model=read_json(ROOT/'delivery/model.json');expected=selected_model(lock,evidence)
        checks['final_frozen_model_matches_forward_selection']=model==expected
        setup=read_json(directory/'setup.json');run=read_json(directory/'run.json')
        checks['all_registered_model_groups_finished']=run['finished_groups']==120 and run['search_finished']
        checks['registered_budget_not_exhausted']=not run['budget_exhausted']
        coverage=search_coverage(directory,config,setup)
        checks['full_search_coverage']=coverage['passed'] and not coverage['failed_trials']
        write_json(directory/'search_coverage.json',coverage)
        before=read_json(directory/'evaluation.json');rebuilt=evaluate(directory,config)
        checks['historical_metrics_and_gates_recomputed']=stable_hash(before)==stable_hash(rebuilt) and rebuilt['complete_model_panel_results']==12 and not rebuilt['errors']
        development=require_verifications(config,directory)
        confirmation=confirmation_evidence(config,directory,frozen['signature'])
        checks['two_development_and_confirmation_verifications']=len(development)==2 and len(confirmation['verifications'])==2 and confirmation==evidence['historical_prerequisites']
        checks['setup_and_search_match_original_freeze']=setup['signature']==frozen['signature'] and setup['config']==config and setup['search']==read_json(ROOT/'configs/rv30_search.json')
        history=read_json(ROOT/'delivery/history.json')
        checks['original_experiments_preserved']=all(sha(ROOT/p)==digest for p,digest in history['historical_file_hashes'].items())
        example=infer(config);write_json(ROOT/'delivery/example_prediction.json',example)
        checks['runnable_inference_with_correct_qualification']=example['rv_mean'] is not None and example['rv_mean']>0 and not example['eligible_signal'] and not example['direction_or_profit_prediction'] and example['prospective_qualified']==model['prospective_qualified']
        checks['report_and_user_documentation']=all((ROOT/p).is_file() for p in ('delivery/REPORT.md','delivery/README.md','delivery/PROSPECTIVE_PLAN.md','delivery/results.json'))
        import hashlib
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        sources={**frozen['effective_source_files'],**frozen['postprocessing_lock_files'],
            'delivery/rv30_freeze.json':sha(ROOT/'delivery/rv30_freeze.json'),'delivery/prospective_lock.json':sha(ROOT/'delivery/prospective_lock.json')}
        checks['signed_code_and_plan_locally_committed']=all(hashlib.sha256(subprocess.check_output(['git','show',f'{commit}:{p}'],cwd=ROOT)).hexdigest()==digest for p,digest in sources.items())
        index=artifact_index(directory,frozen['signature']);write_json(directory/'artifact_index.json',index)
        checks['complete_historical_artifact_chain']=index['job_count']>=12200
        if all(checks.values()):
            result=read_json(ROOT/'delivery/results.json');result.update(engineering_final_acceptance=True,outcome=model['outcome'],
                prospective_status='evaluated_60_complete_sessions',prospective_complete_sessions=evidence['sessions'],
                prospective_qualified_candidates=evidence['qualified_candidates'],active_predictive_edge_established=model['active_predictive_edge_established'])
            write_json(ROOT/'delivery/results.json',result)
            paths=['GOAL.md','RESEARCH_PROTOCOL.md','configs/rv30.json','configs/rv30_search.json','configs/rv30_model_lock.json','configs/rv30_prospective.json',
                'delivery/model.json','delivery/results.json','delivery/README.md','delivery/REPORT.md','delivery/PROSPECTIVE_PLAN.md',
                'delivery/prospective_lock.json','delivery/rv30_freeze.json','delivery/rv30_dependencies.json','delivery/data.json','delivery/history.json',
                'delivery/example_prediction.json','delivery/external_dependencies.json']
            paths+=list(frozen['postprocessing_lock_files'])
            paths+=[str((directory/p).relative_to(ROOT)) for p in ('evaluation.json','daily_losses.csv','search_coverage.json','artifact_index.json','confirmation/evaluation.json')]
            for group in (development,confirmation['verifications'],evidence['verifications']):paths.extend(p['path'] for p in group.values())
            paths.append('delivery/prospective_results.json')
            write_json(ROOT/'delivery/prospective_results.json',evidence)
            manifest={'schema_version':1,'signature':frozen['signature'],'prospective_plan_signature':lock['signature'],'outcome':model['outcome'],
                'code_commit':commit,'created_utc':datetime.now(timezone.utc).isoformat(),'engineering_accepted':True,'final_acceptance':True,
                'active_predictive_edge_established':model['active_predictive_edge_established'],'prospective_qualified':model['prospective_qualified'],
                'goal_can_complete':True,
                'model':'delivery/model.json','report':'delivery/REPORT.md','readme':'delivery/README.md','results':'delivery/results.json',
                'prospective_results':'delivery/prospective_results.json','run_directory':str(directory.relative_to(ROOT)),
                'artifact_index':str((directory/'artifact_index.json').relative_to(ROOT)),
                'acceptance':str((directory/'acceptance.json').relative_to(ROOT)),
                'files':{p:sha(ROOT/p) for p in sorted(set(paths))}}
            write_json(ROOT/'delivery/prospective_artifact_index.json',forward_index)
            write_json(ROOT/'delivery/final_resource_usage.json',resources)
            for p in ('delivery/prospective_artifact_index.json','delivery/final_resource_usage.json'):
                manifest['files'][p]=sha(ROOT/p)
            manifest['prospective_artifact_index']='delivery/prospective_artifact_index.json'
    except (ValueError,KeyError,OSError,AssertionError,TypeError,IndexError,StopIteration,subprocess.CalledProcessError) as error:
        errors.append(str(error))
    result={'passed':bool(checks) and all(checks.values()) and not errors,'checks':checks,'errors':errors,
        'outcome':manifest['outcome'] if manifest else 'waiting_prospective_acceptance','goal_can_complete':bool(checks) and all(checks.values()) and not errors}
    old=directory/'acceptance.json'
    if old.exists():
        previous=read_json(old);archive=directory/'acceptance_attempts'/f'{stable_hash(previous)}.json'
        if not archive.exists():immutable_json(archive,previous)
    write_json(directory/'acceptance.json',result)
    if result['passed']:
        manifest['files'][str((directory/'acceptance.json').relative_to(ROOT))]=sha(directory/'acceptance.json')
        write_json(ROOT/'delivery/manifest.json',manifest)
    return result


def finalize_forward(config,directory):
    from .prospective import read_plan,observation_directory,verify_observation
    lock=read_plan(config);path=observation_directory(lock)
    quality=complete_days(Journal(path,lock['signature']).events,lock['payload']['settings'])
    if not quality['ready_for_final_evaluation']:
        return {'passed':False,'stage':'waiting_prospective_acceptance','prospective_complete_sessions':quality['complete_sessions'],
            'prospective_required_sessions':lock['payload']['settings']['minimum_complete_sessions'],'final_acceptance':False}
    from .forward_runtime import ACTIVE
    if ACTIVE is None or ACTIVE.action!='finalize':
        raise ValueError('Use python -m finance.forward_runtime finalize to preserve resource accounting')
    for number in (1,2):
        proof=path/f'verify_{number}.json'
        if not proof.exists():
            raise ValueError('Run both metered full prospective verification rounds before finalization')
        result=verify_observation(config,number)
        if not result['passed']:return {'passed':False,'stage':f'prospective_verification_{number}_failed','final_acceptance':False}
    evidence=prospective_evidence(config,directory)
    previous=read_json(ROOT/'delivery/model.json');target=selected_model(lock,evidence)
    if previous!=target:
        archive=path/'model_before_final_selection.json'
        if not archive.exists():immutable_json(archive,previous)
        write_json(ROOT/'delivery/model.json',target)
    report=ROOT/'delivery/REPORT.md';marker='\n## 真实前瞻最终结果\n'
    text=report.read_text().split(marker)[0]+marker+'\n已按预注册最早六十个完整合格交易日一次性评价，两轮完整独立复核通过；原历史结果保留。\n\n'
    text+='| 候选 | QLIKE差值 | Holm p | 前瞻门槛通过 |\n|---|---:|---:|---|\n'
    for key,item in evidence['comparisons'].items():text+=f"| {key} | {item['mean_difference']:.6f} | {item['holm_p']:.6g} | {'是' if item['passed_predictive_gate'] else '否'} |\n"
    text+=f"\n交付结论：`{target['outcome']}`。原主信息集优先于固定消融；不根据前瞻最小分数挑模型。波动率预测不构成价格方向、执行成本或盈利能力证明，不自动下单。\n"
    report.write_text(text)
    return accept_forward(config,directory)
