"""Bind a prospective follow-up matrix to actual original research/delivery.

This entry freezes a candidate matrix only. It cannot launch training and does
not turn a draft, engineering receipt or date check into GPU admission.
"""
import argparse
import fcntl
import os
from pathlib import Path

import numpy as np

from fusion.next.common import ROOT, OUT as BASE, PLAN, digest, identity, read_json, source_identity, utc_now, write_json
from scripts.followup_runtime import FOLLOW, DEADLINE, EARLIEST, candidates
from scripts.followup_guard import exact_alive
from scripts.review_fusion_development import QUARTERS, validate_rows


ENGINE = 'fb0a03b4841e9bde2b555440214b2b4308bc89e8ce2f3473b22a73b6ffd063cb'
SCOPE = '2b93e6c04d1fe15f1fd9cb47f424846014b80d88f4f5c8679db0c918b6c60207'
NUMERICAL = '9cf552a6c334bee8226411e1e989a25c11af89d8305bb29feb74ff2276098872'
SPEED = '0c433a556af0a5d944c7a6afdfdbda35b9a7ac372610b7492295dcc487f06fab'
IMPLEMENTATIONS = ('followup_runtime.py','followup_guard.py','followup_admission.py','run_fusion_followup.py')


def identified(path):
    row = read_json(path)
    if row['identity'] != identity({k:v for k,v in row.items() if k != 'identity'}):
        raise ValueError('Artifact identity differs: '+str(path))
    return row


def checked_file(root,relative):
    path = (root/relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('Evidence path is missing or leaves the original run: '+str(relative))
    return path


def verify_delivery(root=BASE):
    """Inspect actual receipt chains, not merely a terminal-looking state file.

    Called on the local research host after the original receiver has exited.
    No confirmation/final prediction arrays are loaded or selected here.
    """
    root = Path(root).resolve()
    state = read_json(root/'cloud/local_completion/state.json')
    binding = read_json(root/'cloud/local_completion/binding.json')
    if (state.get('stage') not in ('delivery_evidence_ready','delivery_evidence_ready_and_summary_pushed')
            or state.get('engineering_acceptance_passed') is not True
            or binding['source_identity'] != source_identity() or binding['source_identity'] != ENGINE
            or binding['scope_identity'] != SCOPE or binding['plan_sha256'] != digest(PLAN)
            or binding['numerical_policy_identity'] != NUMERICAL
            or binding['speed_policy_identity'] != SPEED
            or state['source_identity'] != binding['source_identity']
            or state['scope_identity'] != binding['scope_identity']):
        raise RuntimeError('The original research and actual CPU delivery are not complete')
    receiver = read_json(root/'cloud/local_completion/handoff_receiver_dispatch.json')
    if exact_alive(receiver):
        raise RuntimeError('Original local receiver is still alive; wait for its natural exit')
    limit = binding['limit']
    if limit['total_wall_hours'] != 100 or limit['deadline_utc'] != DEADLINE:
        raise ValueError('Original absolute deadline changed')
    files = {}
    def note(relative):
        path = checked_file(root,relative)
        files[str(path.relative_to(root))] = {'sha256':digest(path),'bytes':path.stat().st_size}
        return path
    for name in ('cloud/local_completion/state.json','cloud/local_completion/binding.json',
                 'cloud/local_completion/handoff_receiver_dispatch.json'):
        note(name)
    scope = identified(note('performance/scope_100h.json'))
    if (scope['identity'] != SCOPE or scope['source_identity'] != ENGINE
            or scope['seeds'] != [2026,2027,2028] or scope['validation'] != list(QUARTERS)
            or scope['training_budget_policy']['total_coverage_equivalent_epochs'] != 2):
        raise ValueError('Original scientific scope differs')
    for name in ('F3_complete.json','F4_complete.json','F5_complete.json'):
        row = read_json(note(name))
        if (row['source_identity'] != ENGINE or row['scope_identity'] != SCOPE
                or row['scope_amendment_sha256'] != files['performance/scope_100h.json']['sha256']
                or row.get('original_complete_scope_claimed') is not False
                or row.get('numerical_policy_identity') != NUMERICAL
                or row.get('evaluation_precision') != 'float32'):
            raise ValueError('Original stage is incomplete or belongs to another run: '+name)
    for filename,expected in [('portable_inference_policy.json',NUMERICAL),('future_speed_policy.json',SPEED)]:
        if identified(note('performance/'+filename))['identity'] != expected:
            raise ValueError('Original numerical/speed policy differs')
    frozen = identified(note('system_freeze.json'))
    confirmation = read_json(note('confirmation.json'))
    if (frozen['source_identity'] != ENGINE or frozen['plan_sha256'] != digest(PLAN)
            or frozen['F5_sha256'] != files['F5_complete.json']['sha256']
            or confirmation['system_freeze_sha256'] != files['system_freeze.json']['sha256']):
        raise ValueError('Original freeze/confirmation identity differs')
    from scripts.finish_fusion_100h import choose_final
    proceed = choose_final(frozen,confirmation)
    if not proceed and (root/'final_test/entry.json').exists():
        raise ValueError('Negative original gate unexpectedly opened the final entry')
    report = read_json(note('research_report.json'))
    required = ('F0','F1','F2','F3','F4','F5','F6','F8_engineering_passed')
    if (report['source_identity'] != ENGINE or report['plan_sha256'] != digest(PLAN)
            or report['violations'] or any(report['stages'].get(k) is not True for k in required)
            or report['stages']['F7_evaluated'] is not proceed
            or report['research_status'] not in (('final_gate_passed','final_gate_failed') if proceed
                    else ('stopped_before_final_by_prespecified_rule',))):
        raise ValueError('Actual research report is incomplete or inconsistent')
    for name,sha in report['source_artifacts_sha256'].items():
        if digest(note(name)) != sha:
            raise ValueError('Original research report artifact changed: '+name)
    if any(name not in report['source_artifacts_sha256'] for name in
           ('F3_complete.json','F4_complete.json','F5_complete.json','system_freeze.json','confirmation.json')):
        raise ValueError('Original report omits required source provenance')
    if proceed:
        note('final_test/report.json')
        if 'final_test/report.json' not in report['source_artifacts_sha256']:
            raise ValueError('Actual final report is not bound to the research report')
    elif report['Q3_preserved_on_negative_path'] is not True:
        raise ValueError('Negative original path did not preserve Q3')
    if frozen['final_fit_policy']['artifact'] != 'deployment/pre_Q2':
        raise ValueError('Original deployment policy changed')
    deployment = identified(note('deployment/pre_Q2/deployment.json'))
    model = deployment['identity']
    if deployment['freeze_sha256'] != files['system_freeze.json']['sha256'] or report['deployment_identity'] != model:
        raise ValueError('Actual deployment does not match the frozen system')
    for name,row in deployment['files'].items():
        path = checked_file(root/'deployment/pre_Q2',name)
        relative = str(path.relative_to(root))
        if digest(note(relative)) != row['sha256'] or path.stat().st_size != row['bytes']:
            raise ValueError('Original deployed model bytes changed')
    accept = 'deployment_acceptance/'+model+'/'
    acceptance = read_json(note(accept+'report.json'))
    progress = read_json(note(accept+'progress.json'))
    if (acceptance['model_identity'] != model or acceptance['engineering_acceptance_passed'] is not True
            or report['engineering_acceptance'] != acceptance or acceptance['origins'] != 256
            or acceptance['timing_complete'] is not True or acceptance['timed_origins'] != 256
            or acceptance['resume_without_duplicate_prediction'] is not True
            or acceptance['distinct_exit_and_resume_processes'] is not True
            or acceptance['cold_load_seconds'] > 60 or acceptance['warm_single_origin_p95_seconds'] > 1
            or acceptance['batch_stream_max_prediction_difference'] > acceptance['batch_stream_tolerance']
            or acceptance['Q3_accessed'] is not False
            or progress['recipe']['source_identity'] != ENGINE
            or progress['recipe']['system_freeze_sha256'] != files['system_freeze.json']['sha256']
            or progress['recipe']['model_identity'] != model
            or len({row['pid'] for row in progress['processes']}) < 2):
        raise ValueError('Actual 256-origin replay, latency or process recovery is not verified')
    package = 'local_packages/'+model
    archive = note(package+'.tar')
    package_receipt = read_json(note(package+'.receipt.json'))
    verification = read_json(note(package+'.verification.json'))
    if (package_receipt['archive_sha256'] != digest(archive)
            or package_receipt['archive_bytes'] != archive.stat().st_size
            or package_receipt['model_identity'] != model
            or verification['archive_sha256'] != digest(archive)
            or verification['package_identity'] != package_receipt['package_identity']
            or verification['model_identity'] != model or verification['Q3_accessed'] is not False
            or verification['isolated_example_and_fresh_process_resume_passed'] is not True
            or [r['arguments'] for r in verification['processes']] != [['--verify-only'],['--example'],['--example']]):
        raise ValueError('Standalone CPU package has not been actually verified')
    returned = read_json(note('cloud/local_completion/return_receipt.json'))
    restored = read_json(note('cloud/restores/'+returned['sha256']+'/receipt.json'))
    if (restored['archive_sha256'] != returned['sha256'] or restored['all_file_hashes_verified'] is not True
            or restored['source_identity'] != ENGINE or restored['plan_sha256'] != digest(PLAN)):
        raise ValueError('Formal cloud return has not been verified/restored')
    value = {'schema':1,'checked_at_utc':utc_now(),'source_identity':ENGINE,'plan_sha256':digest(PLAN),
             'scope_identity':SCOPE,'numerical_policy_identity':NUMERICAL,'speed_policy_identity':SPEED,
             'deployment_identity':model,'limit':limit,'original_receiver_observed_exited':True,
             'research_status':report['research_status'],'original_final_evaluated':proceed,
             'engineering_acceptance_passed':True,'files':files,
             'Q2_Q3_prediction_arrays_loaded':False,'new_training_launched':False}
    value['identity'] = identity(value)
    return value


def fixed_reference(root=BASE,panel=None):
    """One existing frozen reference, never a search across new scores."""
    root = Path(root).resolve()
    reference = read_json(root/'selections/fusion.json')['selected']
    proposed = candidates(reference)
    # The no-foundation/changed-loss matrix must retain exact original row count,
    # fit-specific two-epoch updates and training inputs, not just its name.
    from fusion.next.recipe import for_fit
    from fusion.next.splits import fold
    if panel is None:
        from fusion.next.panel import Panel
        panel = Panel.load()
    dates = panel.dates
    files = {}
    quarters = {}
    for quarter in QUARTERS:
        spec = fold(dates,quarter)
        effective = for_fit(reference,spec['fit_end'])
        relative = ('trials/fusion/'+reference['candidate_id']+'/'+quarter+
                    '/seed_2026/updates_'+str(effective['total_updates']))
        result_path = checked_file(root,relative+'/result.json')
        pred_path = checked_file(root,relative+'/prediction.npz')
        report = read_json(result_path)
        details = report['details']
        if (details['config'] != effective or details['seed'] != 2026 or details['fold'] != spec
                or details['source'] != ENGINE or details['updates'] != effective['total_updates']
                or details['evaluation_precision'] != 'float32' or details['numerical_policy_identity'] != NUMERICAL
                or report['prediction_sha256'] != digest(pred_path)):
            raise ValueError('Reference is not the exact complete original recipe')
        with np.load(pred_path,allow_pickle=False) as saved:
            rows,prediction = saved['rows'],saved['prediction']
            validate_rows(rows,prediction,dates,quarter)
            origins = panel.origins['AAPL']
            expected = origins[(dates[origins]>=spec['score_start'])&(dates[origins]<=spec['score_end'])]
            if not np.array_equal(rows,expected):
                raise ValueError('Reference omitted eligible common origins')
            from fusion.next.evaluation import point_scores
            scores = point_scores(panel.targets['AAPL'][rows],prediction)
            if scores != report['scores']:
                raise ValueError('Reference scores differ from local development targets')
        for path in (result_path,pred_path):
            files[str(path.relative_to(root))] = {'sha256':digest(path),'bytes':path.stat().st_size}
        quarters[quarter] = {'fit_end':spec['fit_end'],'total_updates':effective['total_updates'],
                             'score_origins':len(expected),
                             'reference_result':str(result_path.relative_to(root)),
                             'reference_prediction':str(pred_path.relative_to(root))}
    path = checked_file(root,'selections/fusion.json')
    files['selections/fusion.json'] = {'sha256':digest(path),'bytes':path.stat().st_size}
    return reference,proposed,quarters,files


def _freeze():
    if os.environ.get('FUSION_EXECUTION_PROFILE','local') != 'local':
        raise RuntimeError('Verify actual original CPU delivery on the local research host')
    root = BASE
    delivery = verify_delivery(root)
    from fusion.next import panel,trainer
    original_panel = panel.Panel.load()
    reference,proposed,quarters,files = fixed_reference(root,original_panel)
    # These artifacts are present on both hosts. Local replay/package receipts
    # stay in the delivery prerequisite, while the cloud's original fits,
    # policies and freeze must still match their verified local counterparts.
    for name,row in delivery['files'].items():
        if name.split('/')[0] in ('performance','deployment') or name in (
                'F3_complete.json','F4_complete.json','F5_complete.json','system_freeze.json','confirmation.json'):
            files[name] = row
    store = trainer.ExpertStore(original_panel)
    for name in ['data/dates.npy','data/timestamps_ns.npy','data_admission.json','foundation_weights.json']:
        path = checked_file(Path(root),name)
        files[name] = {'sha256':digest(path),'bytes':path.stat().st_size}
    for name in set(original_panel.stocks+original_panel.background):
        path = checked_file(Path(root),'data/'+name+'/manifest.json')
        files[str(path.relative_to(root))] = {'sha256':digest(path),'bytes':path.stat().st_size}
        for relative,sha in read_json(path)['files'].items():
            array = checked_file(path.parent,relative)
            if digest(array) != sha:
                raise ValueError('Original prepared data changed: '+name+'/'+relative)
            files[str(array.relative_to(root))] = {'sha256':sha,'bytes':array.stat().st_size}
    for path in sorted((root/'oof'/store.key).glob('*')):
        if path.suffix in ('.npy','.npz','.json'):
            files[str(path.relative_to(root))] = {'sha256':digest(path),'bytes':path.stat().st_size}
    implementation = {name:digest(ROOT/'scripts'/name) for name in IMPLEMENTATIONS}
    value = {'schema':1,'status':'matrix_frozen_benchmark_and_launch_admission_pending',
             'frozen_at_utc':utc_now(),'source_identity':ENGINE,'scope_identity':SCOPE,
             'plan_sha256':digest(PLAN),'followup_plan_sha256':digest(ROOT/'FUSION_FOLLOWUP_PLAN.md'),
             'delivery':delivery,'reference':reference,'candidates':proposed,'folds':quarters,
             'original_inputs':files,'expert_sources':store.sources,'expert_scope_key':store.key,
             'implementation':implementation,'earliest_large_run':EARLIEST,
             'limit':delivery['limit'],'validation':list(QUARTERS),'pilot_seeds':[2026],
             'expanded_seeds':[2026,2027,2028],'total_coverage_equivalent_epochs':2,
             'pilot_full_fits':9,'pilot_GPU_hours_ceiling':6,'extension_GPU_hours_ceiling':6,
             'training_admitted':False,'new_training_launched':False,'Q2_Q3_prediction_arrays_loaded':False,
             'interpretation':'prospective development comparison; independent future evidence still required'}
    value['identity'] = identity(value)
    path = FOLLOW/'preparation'/('frozen_'+value['identity'][:16]+'.json')
    write_json(path,value)
    write_json(FOLLOW/'preparation/frozen_latest.json',{'manifest':str(path.relative_to(ROOT)),
               'identity':value['identity'],'status':value['status'],'training_admitted':False})
    return path,value


def freeze():
    directory = FOLLOW/'preparation'
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'freeze.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another prospective matrix freeze is still running')
        latest = directory/'frozen_latest.json'
        if latest.exists():
            pointer = read_json(latest)
            path = ROOT/pointer['manifest']
            value = check_frozen(path)
            if pointer['identity'] != value['identity']:
                raise ValueError('Frozen matrix pointer identity differs')
            return path,value
        return _freeze()


def check_frozen(path,root=BASE):
    """Portable identity/input check; still not a date or GPU benchmark gate."""
    path = Path(path).resolve()
    if not path.is_relative_to(FOLLOW/'preparation'):
        raise ValueError('Use the isolated frozen follow-up manifest')
    value = identified(path)
    if (value['status'] != 'matrix_frozen_benchmark_and_launch_admission_pending'
            or value['training_admitted'] is not False or value['new_training_launched'] is not False
            or value['source_identity'] != source_identity() or value['source_identity'] != ENGINE
            or value['scope_identity'] != SCOPE or value['plan_sha256'] != digest(PLAN)
            or value['followup_plan_sha256'] != digest(ROOT/'FUSION_FOLLOWUP_PLAN.md')
            or value['limit']['total_wall_hours'] != 100 or value['limit']['deadline_utc'] != DEADLINE
            or value['validation'] != list(QUARTERS) or value['pilot_seeds'] != [2026]
            or value['expanded_seeds'] != [2026,2027,2028] or value['total_coverage_equivalent_epochs'] != 2
            or value['pilot_full_fits'] != 9 or value['pilot_GPU_hours_ceiling'] != 6
            or value['extension_GPU_hours_ceiling'] != 6 or value['earliest_large_run'] != EARLIEST
            or value['candidates'] != candidates(value['reference'])):
        raise ValueError('Frozen prospective matrix or original protocol differs')
    if set(value['implementation']) != set(IMPLEMENTATIONS):
        raise ValueError('Frozen manifest omits a follow-up implementation')
    for name,sha in value['implementation'].items():
        if digest(ROOT/'scripts'/name) != sha:
            raise ValueError('Follow-up implementation changed after matrix freeze')
    if (value['delivery']['identity'] != identity({k:v for k,v in value['delivery'].items() if k != 'identity'})
            or value['delivery']['source_identity'] != ENGINE or value['delivery']['scope_identity'] != SCOPE
            or value['delivery']['numerical_policy_identity'] != NUMERICAL
            or value['delivery']['speed_policy_identity'] != SPEED
            or value['delivery']['plan_sha256'] != digest(PLAN)
            or value['delivery']['engineering_acceptance_passed'] is not True
            or value['delivery']['original_receiver_observed_exited'] is not True
            or value['delivery']['limit'] != value['limit']):
        raise ValueError('Original actual-delivery prerequisite differs')
    for name,row in value['original_inputs'].items():
        p = checked_file(Path(root),name)
        if digest(p) != row['sha256'] or p.stat().st_size != row['bytes']:
            raise ValueError('Original fixed input or reference changed: '+name)
    if read_json(Path(root)/'selections/fusion.json')['selected'] != value['reference']:
        raise ValueError('Frozen reference differs from the fixed original selection')
    from fusion.next.recipe import for_fit
    from fusion.next.splits import fold
    dates = np.load(Path(root)/'data/dates.npy',mmap_mode='r')
    if set(value['folds']) != set(QUARTERS):
        raise ValueError('Frozen manifest omitted a validation quarter')
    for quarter,row in value['folds'].items():
        fit_end = fold(dates,quarter)['fit_end']
        if row['fit_end'] != fit_end or row['total_updates'] != for_fit(value['reference'],fit_end)['total_updates']:
            raise ValueError('Frozen manifest changed a two-epoch fit boundary')
        if not isinstance(row['score_origins'],int) or row['score_origins'] <= 0:
            raise ValueError('Frozen manifest must bind all common scoring origins')
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Freeze prospective matrix only after actual original CPU delivery')
    parser.add_argument('--freeze',action='store_true',required=True)
    parser.parse_args()
    path,value = freeze()
    print({'manifest':str(path),'status':value['status'],'training_admitted':False})
