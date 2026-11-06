"""Assemble an evidence-bound follow-up handover after real local delivery.

Recompute development decisions through the existing verifiers. This entry has
no model fitting, final-data access, alternative selection or Git publication.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import os
import sys
import time

from fusion.next.common import ROOT, OUT as BASE, digest, identity, read_json, utc_now, write_json
from scripts import deliver_fusion_followup as delivery
from scripts import followup_guard as guard
from scripts import review_fusion_pilot as pilot
from scripts import review_fusion_extension as extension
from scripts.followup_admission import check_frozen, identified
from scripts.followup_runtime import FOLLOW, DEADLINE, EARLIEST
from scripts.run_fusion_followup import execution_locks, bind_guarded_worker, stop_guard


ENTRY = Path(__file__).resolve()
CONTRACTS = {'exact_retained_context_reproduces_prediction', 'duplicate_bar_ignored',
    'conflicting_duplicate', 'nonfinite', 'inconsistent_OHLCV', 'unfinished_bar',
    'missing_current_is_failed_and_durable', 'two_context_gaps_follow_admitted_policy',
    'excessive_context_gaps_rejected', 'delayed_inputs_cannot_claim_timely_forecast',
    'controlled_model_failure_is_journaled'}


def note(path, provenance):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Report evidence leaves the project')
    provenance[str(path.relative_to(ROOT))] = digest(path)
    return read_json(path)


def budgets(run, manifest, provenance):
    """Retain failed attempts; stage wall time is not measured GPU compute time."""
    groups = {}
    deadline = datetime.fromisoformat(DEADLINE).timestamp()
    earliest = datetime.fromisoformat(EARLIEST).timestamp()
    for prefix, cap, purpose in (('probe_',600,'exploration'), ('pilot_attempt_',21600,'exploration'),
            ('extension_attempt_',21600,'exploration'), ('fit_attempt_',21600,'delivery_fit')):
        records = []
        for directory in sorted((run/'attempts').glob(prefix+'*')):
            if not directory.is_dir():continue
            path = directory/'guard_request.json'
            if not path.exists():
                records.append({'directory':str(directory.relative_to(ROOT)),
                    'status':'reservation_without_guard_request','charged_seconds':cap})
                continue
            request = identified(path);note(path,provenance)
            state = note(directory/'guard_state.json',provenance)
            start = request['requested_at_unix']
            end = datetime.fromisoformat(state['updated_at_utc']).timestamp()
            if (request['limit']!=manifest['limit'] or request['purpose']!=purpose
                    or not 0<request['stage_seconds']<=cap or request['stop_by_unix']>deadline
                    or state['request_identity']!=request['identity']
                    or state['stage'] not in ('completed','stopped','failed','refused')
                    or not start<=end<=deadline or end-start>request['stage_seconds']
                    or (prefix!='probe_' and start<earliest)):
                raise ValueError('A follow-up attempt lacks a terminal identity/date/budget record')
            records.append({'directory':str(directory.relative_to(ROOT)), 'status':state['stage'],
                'charged_seconds':end-start, 'requested_at_unix':start,'finished_at_unix':end,
                'reason':state.get('reason'),'worker_returncode':state.get('worker_returncode')})
        charged = sum(r['charged_seconds'] for r in records)
        ceiling = 3600 if prefix=='probe_' else cap
        if charged>ceiling or (prefix=='probe_' and len(records)>5):
            raise ValueError('The cumulative follow-up stage or probe reservation cap was exceeded')
        groups[prefix.rstrip('_')] = {'records':records,'charged_wall_seconds':charged,
            'ceiling_seconds':ceiling,'includes_failed_attempts':True,
            'missing_guard_reservations_charged_at_full_cap':True}
    return {'groups':groups,'absolute_deadline_utc':DEADLINE,
            'stage_wall_time_is_not_measured_GPU_card_hours':True,
            'actual_provider_billed_cost_cny':None,'rental_shutdown_performed':False}


def local_evidence(run, binding, provenance):
    directory = run/'local_delivery'
    saved = identified(directory/'binding.json');note(directory/'binding.json',provenance)
    complete = identified(directory/'completion.json');note(directory/'completion.json',provenance)
    if (saved!=binding or complete['manifest_identity']!=binding['manifest_identity']
            or complete['binding_identity']!=binding['identity']
            or complete['independent_research_success_claimed'] is not False
            or complete['Q2_Q3_scored'] is not False):
        raise ValueError('Local completion is missing, changed or claims independent follow-up success')
    found = False
    for attempt in sorted((run/'attempts').glob('CPU_delivery_*')):
        request = identified(attempt/'guard_request.json');note(attempt/'guard_request.json',provenance)
        state = note(attempt/'guard_state.json',provenance)
        admission = identified(attempt/'launch_admission.json');note(attempt/'launch_admission.json',provenance)
        if (guard.exact_alive(state['guard']) or (state.get('worker') and guard.group_members(state['worker']))):
            raise RuntimeError('The local delivery guard or worker group is still alive')
        if (state['stage']=='completed' and state['worker_returncode']==0
                and state['request_identity']==request['identity'] and request['purpose']=='delivery_cpu'
                and request['limit']==binding['limit'] and request['stage_seconds']<=7200
                and request['command'][1]==str(delivery.ENTRY)
                and admission['binding']==binding
                and datetime.fromisoformat(state['updated_at_utc']).timestamp()<=datetime.fromisoformat(DEADLINE).timestamp()):
            found = True
    if not found:raise ValueError('No exited exact successful CPU delivery guard exists')
    outcome = binding['outcome']
    if not outcome['new_model_exported']:
        if (complete['stage']!='followup_negative_pilot_complete_original_package_retained'
                or complete['model_identity']!=outcome['retained_original_model_identity']
                or complete['package_sha256']!=outcome['retained_original_package_sha256']
                or complete['actual_new_model_CPU_acceptance_passed'] is not False
                or complete['new_model_exported'] is not False
                or complete['original_actual_CPU_package_retained'] is not True):
            raise ValueError('Negative pilot did not retain the original actual CPU package')
        return {'new_model_exported':False,'original_actual_CPU_package_retained':True,
                'package_sha256':complete['package_sha256'],'actual_new_model_CPU_acceptance_passed':False}
    report = delivery.checked_replay(directory,binding)
    report_path = directory/'deployment_acceptance'/outcome['deployment_identity']/'report.json'
    note(report_path,provenance)
    contracts = identified(directory/'input_contracts.json');note(directory/'input_contracts.json',provenance)
    if (set(contracts['checks'])!=CONTRACTS or any(v is not True for v in contracts['checks'].values())
            or contracts['all_passed'] is not True or contracts['model_identity']!=report['model_identity']
            or contracts['independent_research_success_claimed'] is not False
            or contracts['Q3_accessed'] is not False or contracts['controlled_failures_are_not_research_scores'] is not True
            or any(Path(name).name!=name or digest(directory/'contract_states'/name)!=sha
                   for name,sha in contracts['state_sha256'].items())):
        raise ValueError('The complete set of real input and durable failure checks is not bound')
    archive = directory/'local_packages'/(report['model_identity']+'.tar')
    receipt = note(archive.with_suffix('.receipt.json'),provenance)
    verified = note(archive.with_suffix('.verification.json'),provenance)
    processes = verified['actual_processes']
    process_logs = {p:read_json(p) for p in (directory/'package_processes').glob('*.json')}
    for process in processes:
        match = next((path for path,value in process_logs.items() if value==process),None)
        if match is None:raise ValueError('Package execution lacks its retained process log')
        note(match,provenance)
    actual_hash = digest(archive);provenance[str(archive.relative_to(ROOT))] = actual_hash
    if (complete['stage']!='followup_CPU_delivery_complete' or complete['model_identity']!=report['model_identity']
            or complete['CPU_acceptance_report_sha256']!=digest(report_path)
            or complete['input_contracts_sha256']!=digest(directory/'input_contracts.json')
            or complete['package']!=str(archive.relative_to(ROOT)) or complete['package_sha256']!=actual_hash
            or complete['package_verification_sha256']!=digest(archive.with_suffix('.verification.json'))
            or complete['standalone_package_actually_executed'] is not True
            or complete['actual_new_model_CPU_acceptance_passed']!=report['engineering_acceptance_passed']
            or report['actual_input_contract_checks_pending'] is not False
            or report['engineering_acceptance_passed']!=delivery.qualified(report,True)
            or receipt['archive_sha256']!=actual_hash or receipt['archive_bytes']!=archive.stat().st_size
            or verified['archive_sha256']!=actual_hash or verified['model_identity']!=report['model_identity']
            or verified['followup_binding_identity']!=binding['identity']
            or verified['isolated_example_and_fresh_process_resume_passed'] is not True
            or verified['independent_research_success_claimed'] is not False or verified['Q2_Q3_scored'] is not False
            or len(processes)!=3 or [p['command'][-1] for p in processes]!=['--verify-only','--example','--example']
            or any(p['returncode']!=0 or guard.exact_alive(p['process']) for p in processes)
            or len({(p['process']['pid'],p['process']['created_unix']) for p in processes})!=3):
        raise ValueError('Actual complete CPU replay/package evidence differs or is missing')
    return {'new_model_exported':True,'model_identity':report['model_identity'],
            'actual_new_model_CPU_acceptance_passed':report['engineering_acceptance_passed'],
            'full_cold_start_seconds':report['cold_load_seconds'],
            'warm_single_origin_p95_seconds':report['warm_single_origin_p95_seconds'],
            'origins':report['origins'],'package':complete['package'],'package_sha256':actual_hash,
            'standalone_package_actually_executed':True,'all_named_input_contracts_passed':True}


def collect(manifest_path):
    manifest_path = Path(manifest_path).resolve();manifest = check_frozen(manifest_path)
    provenance = {str(manifest_path.relative_to(ROOT)):digest(manifest_path)}
    binding = delivery.checked_return(manifest_path,manifest);binding['identity'] = identity(binding)
    run = FOLLOW/'runs'/manifest['identity']
    original = note(BASE/'research_report.json',provenance)
    pilot_source,pilot_report = pilot.checked_report(manifest_path,manifest)
    note(ROOT/pilot_source['report'],provenance)
    local = local_evidence(run,binding,provenance)
    extension_report = None
    if binding['outcome']['new_model_exported']:
        source,extension_report = extension.checked_review(manifest_path,manifest)
        note(ROOT/source['report'],provenance)
    selected = pilot_report['selected_for_possible_expansion']
    if bool(selected) != local['new_model_exported']:
        raise ValueError('The frozen pilot decision and actual delivered path differ')
    budget = budgets(run,manifest,provenance)
    value = {'manifest_identity':manifest['identity'],'source_identity':manifest['source_identity'],
        'limit':manifest['limit'],'original_research_status':original['research_status'],
        'original_final_test_evaluated':original['stages']['F7_evaluated'],
        'original_Q3_preserved_on_negative_path':original['Q3_preserved_on_negative_path'],
        'original_development_system':original['selected_development_system'],
        'original_fixed_controls':original['fixed_controls'],
        'original_cost_and_cleanup':original['cost_and_cleanup'],
        'followup_pilot':pilot_report,'followup_extension':extension_report,'local_delivery':local,
        'budget':budget,'all_required_delivery_evidence_present':True,
        'engineering_acceptance_passed':bool(not local['new_model_exported'] or local['actual_new_model_CPU_acceptance_passed']),
        'followup_development_qualification_passed':None if extension_report is None else extension_report['development_qualification_passed'],
        'independent_followup_research_success_claimed':False,'new_final_data_access_authorized':False,
        'followup_final_independent_validation':'requires separately fixed future independent data',
        'automatic_Goal_completion_or_monitor_shutdown':False,
        'source_artifacts_sha256':provenance}
    value['decision_identity'] = identity(value)
    return value


def markdown(value):
    lines = ['# AAPL 后续开发与实际交付汇总','',
        f'原研究状态：`{value["original_research_status"]}`。后续工程验收通过：`{value["engineering_acceptance_passed"]}`。',
        '后续比较使用已经参与选择的开发季度，独立未来验证仍待完成。','',
        '| 后续配方 | 种子数 | 相对零收益 MSE 改善 | 探索性扩展条件 |',
        '| --- | ---: | ---: | --- |']
    for name,row in value['followup_pilot']['candidates'].items():
        lines.append(f'| {name} | 1 | {100*row["pooled"]["mse_improvement_vs_zero"]:+.6f}% | {row["eligible_for_seed_expansion"]} |')
    extended = value['followup_extension']
    if extended:
        lines += ['',f'固定选择：{extended["selected_candidate"]}；固定配对对照：{extended["paired_control"]}。',
            f'三种子均值改善：{100*extended["selected"]["system"]["mse_improvement_vs_zero"]:+.6f}%；开发门槛通过：{extended["development_qualification_passed"]}。']
    else:lines += ['','没有配方满足预定扩展条件，已保留原实际 CPU 运行包。']
    local = value['local_delivery']
    if local['new_model_exported']:
        lines += ['',f'实际 CPU 重放 {local["origins"]} 个起点；完整冷启动 {local["full_cold_start_seconds"]:.6f} 秒，p95 {local["warm_single_origin_p95_seconds"]:.6f} 秒。',
            '完整包已在三个独立进程中执行身份检查、示例预测和示例恢复。']
    lines += ['',f'运行包 SHA256：`{local["package_sha256"]}`。',
        f'原绝对截止：{value["limit"]["deadline_utc"]}。失败尝试和阶段墙钟计入 JSON；阶段墙钟不等于实际 GPU 计算卡时。',
        '供应商实际账单未取得；未操作租赁关机。完整 JSON 保留原成本/清理记录、配对开发结果、逐种子与季度及溯源。',
        '本报告不会替换原最终检验结论，也不会自动完成 Goal 或停用复查。','']
    return '\n'.join(lines)


def publish(manifest_path):
    value = collect(manifest_path)
    destination = FOLLOW/'runs'/value['manifest_identity']/'handover'/value['decision_identity']
    path = destination/'report.json'
    if path.exists():
        saved = identified(path)
        if {k:v for k,v in saved.items() if k not in ('identity','reported_at_utc')}!=value:
            raise ValueError('Preserve the previous handover; its evidence differs')
        value = saved
    else:
        value['reported_at_utc'] = utc_now();value['identity'] = identity(value);write_json(path,value)
    text = markdown(value);document = destination/'report.md'
    if document.exists() and document.read_text()!=text:raise ValueError('Preserve the previous handover document')
    if not document.exists():document.write_text(text,encoding='utf-8')
    write_json(destination.parent/'latest.json',{'report':str(path.relative_to(ROOT)),'identity':value['identity'],
        'decision_identity':value['decision_identity'],'Goal_completion_not_automatic':True})
    return path,value


def worker(manifest_path,attempt):
    request = bind_guarded_worker(attempt,ENTRY,'delivery_cpu')
    manifest = check_frozen(manifest_path)
    admission = identified(Path(attempt)/'launch_admission.json')
    if (request['limit']!=manifest['limit'] or request['stage_seconds']>7200
            or admission['manifest_identity']!=manifest['identity']
            or admission['matrix_sha256']!=digest(manifest_path) or admission['entry_sha256']!=digest(ENTRY)):
        raise ValueError('Final handover differs from its exact CPU report admission')
    return publish(manifest_path)


def launch(manifest_path):
    if os.environ.get('FUSION_EXECUTION_PROFILE','local')!='local':raise RuntimeError('Assemble the handover on the local CPU host')
    manifest_path = Path(manifest_path).resolve();manifest = check_frozen(manifest_path)
    with execution_locks():
        left = (datetime.fromisoformat(DEADLINE)-datetime.now(timezone.utc)).total_seconds()
        if left<=65:raise TimeoutError('Original cutoff leaves no guarded handover time')
        attempts = FOLLOW/'runs'/manifest['identity']/'attempts'
        previous = sorted(attempts.glob('handover_report_*'))
        from scripts.run_fusion_followup import spent_seconds
        for prior in previous:
            if (prior/'guard_request.json').exists():spent_seconds(prior/'guard_request.json',7200)
        attempt = attempts/('handover_report_'+str(len(previous)+1).zfill(2))
        if attempt.exists():raise RuntimeError('Preserve the previous partial handover launch')
        admission = {'manifest_identity':manifest['identity'],'matrix_sha256':digest(manifest_path),
                     'entry_sha256':digest(ENTRY)}
        admission['identity'] = identity(admission);write_json(attempt/'launch_admission.json',admission)
        command = [sys.executable,str(ENTRY),'--manifest',str(manifest_path),'--worker','--attempt',str(attempt)]
        os.environ['PYTHONPATH'] = str(ROOT)
        process,dispatch = guard.start(attempt,command,manifest['limit'],min(7200,left),purpose='delivery_cpu')
        try:code = process.wait(timeout=max(1,dispatch['stop_by_unix']-time.time()+10))
        finally:stop_guard(process,attempt)
        if code or read_json(attempt/'guard_state.json')['stage']!='completed':
            raise RuntimeError('Final handover failed/stopped; preserve and inspect its evidence')
        print(read_json(FOLLOW/'runs'/manifest['identity']/'handover/latest.json'))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--attempt',type=Path,help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if not args.attempt:parser.error('Worker requires its exact guard attempt')
        worker(args.manifest,args.attempt)
    else:
        if args.attempt:parser.error('Attempt paths are fixed by the launcher')
        launch(args.manifest)
