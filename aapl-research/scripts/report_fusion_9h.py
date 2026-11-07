"""Audit actual A-only delivery and publish a bounded, local CPU handover.

No training, final-data evaluation, candidate selection or automatic Goal close.
The old full-matrix report remains intact; cancelled B/C work is disclosed here.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile

from fusion.next.common import ROOT,OUT as BASE,digest,identity,read_json,utc_now,write_json
from scripts import followup_guard as guard
from scripts import fusion_9h as focus
from scripts import deliver_fusion_9h as returned
from scripts import deliver_fusion_followup as cpu
from scripts.followup_admission import identified
from scripts.report_fusion_followup import CONTRACTS,note
from scripts.run_fusion_followup import execution_locks,bind_guarded_worker

ENTRY=Path(__file__).resolve()


def exited_attempt(attempt,limit,entry,provenance):
    request=identified(attempt/'guard_request.json');note(attempt/'guard_request.json',provenance)
    state=note(attempt/'guard_state.json',provenance)
    if guard.exact_alive(state['guard']) or (state.get('worker') and guard.group_members(state['worker'])):
        raise RuntimeError('Delivery/report dependency has a live guard or worker group')
    cwd=Path(request['cwd'])
    if str(cwd) not in (str(ROOT),returned.REMOTE_ROOT):raise ValueError('Dependency root is not an admitted local/cloud project')
    expected_entry=str(cwd/entry.relative_to(ROOT))
    if (state['stage']!='completed' or state['worker_returncode']!=0
            or state['request_identity']!=request['identity'] or request['limit']!=limit
            or request['command'][1]!=expected_entry
            or state.get('live_group_members')!=[]):
        raise ValueError('Need an actually exited successful dependency guard')
    end=datetime.fromisoformat(state['updated_at_utc']).timestamp()
    if not request['requested_at_unix']<=end<=request['stop_by_unix']+1:
        raise ValueError('Dependency completion lies outside its guard cutoff')
    return {'request_identity':request['identity'],'stage':state['stage'],
        'stage_wall_seconds':end-request['requested_at_unix'],'limit':limit,
        'wall_seconds_are_not_GPU_compute_hours_or_billed_time':True}


def package_bytes(archive,receipt,model,provenance):
    """Rehash the complete actual package, including the original engine copy."""
    if digest(archive)!=receipt['archive_sha256'] or archive.stat().st_size!=receipt['archive_bytes']:
        raise ValueError('Actual CPU package bytes differ')
    provenance[str(archive.relative_to(ROOT))]=digest(archive)
    with tarfile.open(archive,'r') as tar:
        members=tar.getmembers();names=[m.name for m in members]
        if len(set(names))!=len(names) or any(not m.isfile() for m in members):
            raise ValueError('Package contains duplicate or non-regular members')
        for name in names:returned.safe_relative(name)
        package=json.load(tar.extractfile('aapl-cpu/package.json'))
        if (package['identity']!=identity({k:v for k,v in package.items() if k!='identity'})
                or package['identity']!=receipt['package_identity'] or package['model_identity']!=model
                or package['credentials_included'] is not False or package['Q3_values_included'] is not False):
            raise ValueError('Independent package identity or confidentiality declaration differs')
        expected={'aapl-cpu/'+name for name in package['files']}|{'aapl-cpu/package.json'}
        if set(names)!=expected:raise ValueError('Independent package membership differs')
        for name,row in package['files'].items():
            returned.safe_relative(name);member=tar.getmember('aapl-cpu/'+name)
            stream=tar.extractfile(member);sha=hashlib.sha256()
            for block in iter(lambda:stream.read(1024*1024),b''):sha.update(block)
            if member.size!=row['bytes'] or sha.hexdigest()!=row['sha256']:
                raise ValueError('Independent package member differs: '+name)
    return package


def actual_delivery(run,body,provenance):
    directory=run/'local_delivery';saved=identified(directory/'binding.json')
    complete=identified(directory/'completion.json')
    note(directory/'binding.json',provenance);note(directory/'completion.json',provenance)
    if (saved!=body or complete['manifest_identity']!=body['manifest_identity']
            or complete['binding_identity']!=body['identity'] or complete['stage']!='nine_hour_CPU_delivery_complete'
            or complete['independent_success_claimed'] is not False or complete['Q2_Q3_scored'] is not False):
        raise ValueError('Actual local delivery is incomplete or claims independent success')
    report=cpu.checked_replay(directory,body)
    model=body['outcome']['deployment_identity'];report_path=directory/'deployment_acceptance'/model/'report.json'
    note(report_path,provenance)
    contracts=identified(directory/'input_contracts.json');note(directory/'input_contracts.json',provenance)
    if (set(contracts['checks'])!=CONTRACTS or any(v is not True for v in contracts['checks'].values())
            or contracts['all_passed'] is not True or contracts['model_identity']!=model
            or contracts['independent_research_success_claimed'] is not False or contracts['Q3_accessed'] is not False
            or contracts['controlled_failures_are_not_research_scores'] is not True
            or any(Path(name).name!=name or digest(directory/'contract_states'/name)!=sha
                   for name,sha in contracts['state_sha256'].items())):
        raise ValueError('Actual named input contracts or durable states are incomplete')
    if (report.get('cold_timing_complete') is not True or report['cold_timing_binding']!=body['cold_timing']
            or report['actual_input_contract_checks_pending'] is not False):
        raise ValueError('Full-process birth timing or actual input checks are incomplete')
    for row in report['cold_load_measurements_by_process']:
        birth=row['kernel_process_birth']
        expected=row['first_durable_output_boottime_seconds']-birth['start_ticks']/birth['ticks_per_second']
        if expected!=row['full_process_to_first_output_seconds'] or expected<0:
            raise ValueError('Kernel process-birth cold measurement changed')
    archive=directory/'local_packages'/(model+'.tar')
    receipt=note(archive.with_suffix('.receipt.json'),provenance)
    verified=note(archive.with_suffix('.verification.json'),provenance)
    package=package_bytes(archive,receipt,model,provenance)
    processes=verified['actual_processes']
    if (len(processes)!=3 or [r['command'][-1] for r in processes]!=['--verify-only','--example','--example']
            or len({(r['process']['pid'],r['process']['created_unix']) for r in processes})!=3
            or any(r['returncode']!=0 or guard.exact_alive(r['process']) for r in processes)):
        raise ValueError('Package did not actually run in three exited successful processes')
    logs={p:read_json(p) for p in (directory/'package_processes').glob('*.json')}
    for process in processes:
        match=next((p for p,row in logs.items() if row==process),None)
        if match is None:raise ValueError('Actual independent package process lacks its raw log')
        note(match,provenance)
    if (complete['model_identity']!=model or complete['CPU_report_sha256']!=digest(report_path)
            or complete['package']!=str(archive.relative_to(ROOT)) or complete['package_sha256']!=digest(archive)
            or complete['verification_sha256']!=digest(archive.with_suffix('.verification.json'))
            or complete['actual_new_model_CPU_acceptance_passed']!=report['engineering_acceptance_passed']
            or report['engineering_acceptance_passed']!=cpu.qualified(report,True)
            or complete['standalone_package_actually_executed'] is not True
            or verified['followup_binding_identity']!=body['identity'] or verified['model_identity']!=model
            or verified['archive_sha256']!=digest(archive) or verified['isolated_example_and_fresh_process_resume_passed'] is not True
            or verified['independent_research_success_claimed'] is not False or verified['Q2_Q3_scored'] is not False
            or package['engineering_acceptance_passed']!=report['engineering_acceptance_passed']):
        raise ValueError('New-model CPU completion and actual package evidence disagree')
    return {'model_identity':model,'origins':report['origins'],
        'actual_new_model_CPU_acceptance_passed':report['engineering_acceptance_passed'],
        'full_process_cold_seconds':report['cold_load_seconds'],
        'warm_single_origin_p95_seconds':report['warm_single_origin_p95_seconds'],
        'batch_stream_max_difference':report['batch_stream_max_prediction_difference'],
        'all_named_input_contracts_passed':True,'package_files_rehashed':len(package['files']),
        'three_distinct_actual_package_processes_verified':True,
        'package':complete['package'],'package_sha256':complete['package_sha256']}


def usage(run,provenance):
    jobs=[]
    for prefix in ('gpu','delivery_fit'):
        for path in sorted((run/prefix/'usage_events').glob('*.json')):
            row=note(path,provenance)
            if row['status']!='completed' or not row['gpu'] or row['wall_seconds']<0:
                raise ValueError('A required completed GPU fit has missing or nonterminal accounting')
            jobs.append(row)
    if len(jobs)!=12:raise ValueError('Accounting must contain nine development and three engineering fits')
    return {'GPU_job_records':jobs,'accounted_GPU_job_wall_hours':sum(r['wall_seconds'] for r in jobs)/3600,
        'job_wall_time_is_not_measured_GPU_kernel_time':True,'actual_provider_billed_cost_cny':None,
        'rental_shutdown_performed':False}


def collect(path):
    manifest,parent,_=focus.checked(path);run=focus.FOLLOW/'runs'/manifest['identity']
    provenance={str(path.relative_to(ROOT)):digest(path),str(ENTRY.relative_to(ROOT)):digest(ENTRY)}
    # Revalidate the original prerequisite, all nine saved prediction arrays,
    # fixed controls, three fresh engineering checkpoints and exported bytes.
    body=returned.binding(path,manifest,parent)
    local_guard=exited_attempt(run/'attempts/local_cpu_01',parent['limit'],returned.ENTRY,provenance)
    gpu_guard=exited_attempt(run/'attempts/gpu_01',manifest['GPU_limit'],focus.ENTRY,provenance)
    if read_json(run/'attempts/local_cpu_01/guard_request.json')['purpose']!='delivery_cpu':
        raise ValueError('Actual CPU delivery did not use its reserved purpose')
    receipt=identified(focus.FOLLOW/'returns'/manifest['identity']/'receipt.json')
    archive=focus.FOLLOW/'returns'/manifest['identity']/'bundle.tar'
    returned.inspect(archive,receipt,manifest)
    returned.fetch_accounting(manifest,receipt)
    note(archive.parent/'receipt.json',provenance);note(archive.parent/'restored_receipt.json',provenance)
    note(archive.parent/'accounting_receipt.json',provenance)
    actual=actual_delivery(run,body,provenance)
    review=note(run/'review.json',provenance);original=note(BASE/'research_report.json',provenance)
    budget=usage(run,provenance)
    from scripts.run_fusion_followup import ENTRY as old_entry
    records=[]
    prior_attempts=focus.DIAGNOSTIC.parent
    for name in ('probe_01','probe_02','probe_03','probe_04_sustained_9h'):
        attempt=prior_attempts/name
        entry=ROOT/'scripts/diagnose_fusion_9h.py' if name.endswith('9h') else old_entry
        record=exited_attempt(attempt,parent['limit'],entry,provenance)
        request=identified(attempt/'guard_request.json')
        if request['stage_seconds']!=600 or request['purpose']!='exploration':raise ValueError('Original probe reservation changed')
        records.append({'attempt':name,**record})
    charged=sum(row['stage_wall_seconds'] for row in records)
    if len(list(prior_attempts.glob('probe_*')))!=4 or charged>3600:raise ValueError('Probe count or cumulative budget differs')
    budget.update(original_and_sustained_probe_records=records,probe_reservations_used=4,
        probe_charged_wall_seconds=charged,
        GPU_guard=gpu_guard,CPU_delivery_guard=local_guard)
    result={'manifest_identity':manifest['identity'],'source_identity':manifest['source_identity'],
        'original_limit':parent['limit'],'shortened_GPU_limit':manifest['GPU_limit'],
        'fixed_candidate':manifest['candidate']['name'],'seeds':list(focus.SEEDS),'validation':list(focus.QUARTERS),
        'total_coverage_equivalent_epochs':2,'cancelled_full_fit_candidates':manifest['cancelled_full_fit_candidates'],
        'full_factorial_or_foundation_removal_attribution_completed':False,
        'original_research_status':original['research_status'],
        'original_final_test_evaluated':original['stages']['F7_evaluated'],
        'original_Q3_preserved_on_negative_path':original['Q3_preserved_on_negative_path'],
        'original_cost_and_cleanup':original['cost_and_cleanup'],
        'development':review,'actual_local_delivery':actual,'budget':budget,
        'new_development_qualification_passed':review['development_qualification_passed'],
        'new_engineering_acceptance_passed':actual['actual_new_model_CPU_acceptance_passed'],
        'independent_research_success_claimed':False,'Q2_Q3_scoring_authorized':False,
        'remaining_work':['separately prespecified future independent validation',
            'cancelled B/C full factorial contrasts remain unperformed',
            'provider billing cannot be inferred from process wall time'],
        'automatic_Goal_completion_or_monitor_shutdown':False,'source_artifacts_sha256':provenance}
    result['decision_identity']=identity(result)
    return result


def markdown(value):
    actual=value['actual_local_delivery'];development=value['development']
    lines=['# 九小时 A 方案实际交付审计','',
        f'固定配方：{value["fixed_candidate"]}；三季度、三个种子、完整两轮。',
        '本次因资源缩短取消 B/C 完整拟合，未完成基础支路移除对照或 2×2 归因。',
        f'开发门槛通过：{value["new_development_qualification_passed"]}；新模型实际工程验收通过：{value["new_engineering_acceptance_passed"]}。',
        '开发比较使用已经参与研究的季度，不能据此声称独立预测成功。','',
        '| 开发结果 | 相对零收益 MSE 改善 |','| --- | ---: |',
        f'| 三种子均值预测 | {100*development["selected"]["system"]["mse_improvement_vs_zero"]:+.6f}% |']
    for seed,row in development['selected']['individual_seeds'].items():
        lines.append(f'| seed {seed} | {100*row["mse_improvement_vs_zero"]:+.6f}% |')
    for q,row in development['quarters'].items():lines.append(f'| {q} 三种子均值 | {100*row["mse_improvement_vs_zero"]:+.6f}% |')
    lines+=['',f'实际 CPU 重放 {actual["origins"]} 起点，两个退出恢复进程；完整冷启动 {actual["full_process_cold_seconds"]:.6f} 秒，warm p95 {actual["warm_single_origin_p95_seconds"]:.6f} 秒。',
        '已核验全部命名输入契约、持久失败、完整包逐文件字节和三个独立包进程的原始日志。',
        f'运行包：`{actual["package"]}`。SHA256：`{actual["package_sha256"]}`。','',
        f'GPU 截止：{value["shortened_GPU_limit"]["deadline_utc"]}；原总截止：{value["original_limit"]["deadline_utc"]}。',
        '预算同时保留原成本/清理与后续作业记录。阶段和作业墙钟不等于 GPU 内核计算时长或供应商账单；未操作租赁关机。',
        '原负门槛、F7 未评估与 Q3 保留结论不被本模型替换。未来独立验证和取消的对照仍是未完成事项。',
        '本报告不自动完成 Goal 或停用复查。','']
    return '\n'.join(lines)


def publish(path):
    value=collect(path);destination=focus.FOLLOW/'runs'/value['manifest_identity']/'handover'/value['decision_identity']
    target=destination/'report.json'
    if target.exists():
        saved=identified(target)
        if {k:v for k,v in saved.items() if k not in ('identity','reported_at_utc')}!=value:raise ValueError('Preserve conflicting handover')
        value=saved
    else:
        value['reported_at_utc']=utc_now();value['identity']=identity(value);write_json(target,value)
    document=target.with_suffix('.md');text=markdown(value)
    if document.exists() and document.read_text()!=text:raise ValueError('Preserve conflicting handover document')
    if not document.exists():document.write_text(text,encoding='utf-8')
    write_json(destination.parent/'latest.json',{'report':str(target.relative_to(ROOT)),
        'identity':value['identity'],'decision_identity':value['decision_identity'],'Goal_completion_not_automatic':True})
    return target,value


def worker(path,attempt):
    request=bind_guarded_worker(attempt,ENTRY,'delivery_cpu')
    manifest,_,_=focus.checked(path)
    admission=identified(attempt/'admission.json')
    if (request['limit']!=manifest['limit'] or request['stage_seconds']>7200
            or admission['entry_sha256']!=digest(ENTRY) or admission['manifest_sha256']!=digest(path)):
        raise ValueError('Handover admission changed')
    return publish(path)


def launch(path):
    if os.environ.get('FUSION_EXECUTION_PROFILE','local')!='local':raise RuntimeError('Report runs only on local CPU')
    manifest,_,_=focus.checked(path);run=focus.FOLLOW/'runs'/manifest['identity']
    with execution_locks():
        attempt=run/'attempts/handover_01'
        if attempt.exists():raise RuntimeError('Report attempt exists; inspect exact process and immutable result')
        # Verify delivery has actually exited before starting another CPU guard.
        exited_attempt(run/'attempts/local_cpu_01',manifest['limit'],returned.ENTRY,{})
        left=datetime.fromisoformat(manifest['limit']['deadline_utc']).timestamp()-datetime.now(timezone.utc).timestamp()
        if left<=65:raise TimeoutError('Original absolute deadline leaves no guarded reporting time')
        attempt.mkdir(parents=True)
        admission={'entry_sha256':digest(ENTRY),'manifest_sha256':digest(path)};admission['identity']=identity(admission)
        write_json(attempt/'admission.json',admission)
        process,receipt=guard.start(attempt,[sys.executable,str(ENTRY),'--manifest',str(path),'--worker','--attempt',str(attempt)],
            manifest['limit'],min(7200,left),purpose='delivery_cpu')
        write_json(attempt/'scheduler_dispatch.json',{'scheduler':guard.process_identity(),'guard':receipt})
        if process.wait()!=0:raise RuntimeError('Actual handover failed; preserve evidence')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--attempt',type=Path)
    args=parser.parse_args()
    worker(args.manifest.resolve(),args.attempt.resolve()) if args.worker else launch(args.manifest.resolve())
