"""Handover aggregation with synthetic verifier boundaries and retained files.

Actual replay/model/package execution is covered by test_followup_delivery;
these cases test the final evidence chain and refusal to manufacture completion.
"""
import copy
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys

import pytest

from fusion.next.common import ROOT,digest,identity,read_json,write_json
from scripts import report_fusion_followup as report
from scripts import followup_guard as guard
from scripts.followup_runtime import DEADLINE


def seal(row):
    value=copy.deepcopy(row);value.pop('identity',None);value['identity']=identity(value)
    return value


@pytest.fixture
def evidence(monkeypatch):
    with TemporaryDirectory(dir=ROOT/'tmp',prefix='handover_unit_') as temporary:
        root=Path(temporary);base=root/'original';follow=root/'followup';run=follow/'runs/synthetic_matrix'
        local=run/'local_delivery';path=follow/'preparation/matrix.json'
        limit={'total_wall_hours':100,'deadline_utc':DEADLINE}
        manifest={'identity':'synthetic_matrix','source_identity':'synthetic_engine','limit':limit}
        write_json(path,manifest)
        original={'research_status':'stopped_before_final_by_prespecified_rule','stages':{'F7_evaluated':False},
            'Q3_preserved_on_negative_path':True,'selected_development_system':{'synthetic':True},
            'fixed_controls':{},'cost_and_cleanup':{'actual_provider_billed_cost_cny':None}}
        write_json(base/'research_report.json',original)
        pilot={'selected_for_possible_expansion':None,'candidates':{
            name:{'pooled':{'mse_improvement_vs_zero':-.001},'eligible_for_seed_expansion':False}
            for name in ('A_foundation_raw','B_no_foundation_scaled','C_no_foundation_raw')}}
        pilot_path=run/'reviews/synthetic/report.json';write_json(pilot_path,pilot)
        body={'manifest_identity':manifest['identity'],'limit':limit,
            'outcome':{'new_model_exported':False,'retained_original_model_identity':'original_model',
                'retained_original_package_sha256':'synthetic_original_archive_only'}}
        complete={'manifest_identity':manifest['identity'],'binding_identity':seal(body)['identity'],
            'independent_research_success_claimed':False,'Q2_Q3_scored':False,
            'stage':'followup_negative_pilot_complete_original_package_retained',
            'model_identity':'original_model','package_sha256':'synthetic_original_archive_only',
            'actual_new_model_CPU_acceptance_passed':False,'new_model_exported':False,
            'original_actual_CPU_package_retained':True}
        write_json(local/'binding.json',seal(body));write_json(local/'completion.json',seal(complete))
        start=datetime.fromisoformat('2026-11-08T01:00:00+00:00').timestamp()
        def attempt(name,purpose='exploration',seconds=30):
            directory=run/'attempts'/name
            request=seal({'limit':limit,'purpose':purpose,'requested_at_unix':start,
                'stage_seconds':600 if name.startswith('probe_') else 7200,
                'stop_by_unix':start+(600 if name.startswith('probe_') else 7200),
                'command':[sys.executable,str(report.delivery.ENTRY)]})
            write_json(directory/'guard_request.json',request)
            state={'request_identity':request['identity'],'stage':'completed','worker_returncode':0,
                'guard':{'pid':99999991,'created_unix':1.},'worker':None,
                'updated_at_utc':datetime.fromtimestamp(start+seconds).astimezone().isoformat()}
            write_json(directory/'guard_state.json',state)
            write_json(directory/'launch_admission.json',seal({'binding':seal(body)}))
            return directory
        attempt('pilot_attempt_01')
        for number in range(3):attempt('probe_'+str(number+1).zfill(2))
        cpu=attempt('CPU_delivery_01','delivery_cpu')
        monkeypatch.setattr(report,'BASE',base);monkeypatch.setattr(report,'FOLLOW',follow)
        monkeypatch.setattr(report,'check_frozen',lambda path:manifest)
        monkeypatch.setattr(report.delivery,'checked_return',lambda path,manifest:copy.deepcopy(body))
        monkeypatch.setattr(report.pilot,'checked_report',lambda path,manifest:({'report':str(pilot_path.relative_to(ROOT))},copy.deepcopy(pilot)))
        monkeypatch.setattr(report.guard,'exact_alive',lambda row:False)
        monkeypatch.setattr(report.guard,'group_members',lambda row:[])
        yield SimpleNamespace(root=root,base=base,follow=follow,run=run,local=local,path=path,
            body=body,pilot=pilot,complete=complete,cpu=cpu,attempt=attempt)


def test_negative_handover_retains_original_package_and_repeated_report_bytes(evidence):
    e=evidence;original={str(p):digest(p) for p in e.base.rglob('*') if p.is_file()}
    path,value=report.publish(e.path);first=digest(path)
    assert value['engineering_acceptance_passed'] and not value['local_delivery']['new_model_exported']
    assert value['original_Q3_preserved_on_negative_path'] and not value['original_final_test_evaluated']
    assert not value['independent_followup_research_success_claimed'] and not value['new_final_data_access_authorized']
    assert not value['automatic_Goal_completion_or_monitor_shutdown']
    assert '已保留原实际 CPU 运行包' in path.with_suffix('.md').read_text()
    assert report.publish(e.path)==(path,value) and digest(path)==first
    assert {str(p):digest(p) for p in e.base.rglob('*') if p.is_file()}==original


@pytest.mark.parametrize('change',[{'independent_research_success_claimed':True},
    {'binding_identity':'different'}, {'package_sha256':'different'}, {'new_model_exported':True}])
def test_false_success_changed_binding_and_changed_package_refuse_publication(evidence,change):
    e=evidence;write_json(e.local/'completion.json',seal({**e.complete,**change}))
    with pytest.raises(ValueError):report.publish(e.path)
    assert not (e.run/'handover').exists()


def test_running_cpu_guard_or_missing_actual_completion_cannot_publish(evidence,monkeypatch):
    e=evidence;monkeypatch.setattr(report.guard,'exact_alive',lambda row:True)
    with pytest.raises(RuntimeError,match='still alive'):report.publish(e.path)
    monkeypatch.setattr(report.guard,'exact_alive',lambda row:False)
    (e.local/'completion.json').unlink()
    with pytest.raises(FileNotFoundError):report.publish(e.path)
    assert not (e.run/'handover').exists()


def test_stage_caps_dates_and_failed_probe_reservations_are_not_erased(evidence):
    e=evidence
    directory=e.attempt('probe_04',seconds=100)
    state=read_json(directory/'guard_state.json');state.update(stage='stopped',worker_returncode=1,reason='controlled_failure')
    write_json(directory/'guard_state.json',state)
    result=report.collect(e.path)
    probes=result['budget']['groups']['probe']
    assert len(probes['records'])==4 and probes['charged_wall_seconds']==190
    assert probes['records'][-1]['reason']=='controlled_failure'
    request=directory/'guard_request.json';saved=read_json(request)
    changed=seal({**saved,'requested_at_unix':datetime.fromisoformat(DEADLINE).timestamp()+1})
    write_json(request,changed)
    with pytest.raises(ValueError,match='terminal identity/date/budget'):report.publish(e.path)
    assert not (e.run/'handover').exists()


def new_model(e,monkeypatch,*,p95=.2):
    e.body['outcome']={'new_model_exported':True,'deployment_identity':'followup_model'}
    e.pilot['selected_for_possible_expansion']='A_foundation_raw'
    binding=seal(e.body);write_json(e.local/'binding.json',binding)
    write_json(e.cpu/'launch_admission.json',seal({'binding':binding}))
    replay={'model_identity':'followup_model','origins':256,'timed_origins':256,'timing_complete':True,
        'cold_load_seconds':10.,'warm_single_origin_p95_seconds':p95,'batch_stream_max_prediction_difference':0.,
        'batch_stream_tolerance':1e-8,'resume_without_duplicate_prediction':True,
        'distinct_exit_and_resume_processes':True,'actual_input_contract_checks_pending':False,
        'engineering_acceptance_passed':p95<=1}
    acceptance=e.local/'deployment_acceptance/followup_model/report.json';write_json(acceptance,replay)
    monkeypatch.setattr(report.delivery,'checked_replay',lambda directory,body:copy.deepcopy(replay))
    state=e.local/'contract_states/durable.json';write_json(state,{'synthetic':'durable fault fixture only'})
    contracts={'model_identity':'followup_model','all_passed':True,'independent_research_success_claimed':False,
        'Q3_accessed':False,'controlled_failures_are_not_research_scores':True,
        'checks':{name:True for name in report.CONTRACTS},'state_sha256':{state.name:digest(state)}}
    write_json(e.local/'input_contracts.json',seal(contracts))
    archive=e.local/'local_packages/followup_model.tar';archive.parent.mkdir();archive.write_bytes(b'synthetic byte identity only')
    write_json(archive.with_suffix('.receipt.json'),{'archive_sha256':digest(archive),'archive_bytes':archive.stat().st_size})
    processes=[{'command':[sys.executable,'run.py',arg],'returncode':0,
        'process':{'pid':99999992+i,'created_unix':1.}} for i,arg in enumerate(('--verify-only','--example','--example'))]
    for i,row in enumerate(processes):write_json(e.local/'package_processes'/(str(i)+'.json'),row)
    verification={'archive_sha256':digest(archive),'model_identity':'followup_model',
        'followup_binding_identity':binding['identity'],'isolated_example_and_fresh_process_resume_passed':True,
        'independent_research_success_claimed':False,'Q2_Q3_scored':False,'actual_processes':processes}
    write_json(archive.with_suffix('.verification.json'),verification)
    complete={'stage':'followup_CPU_delivery_complete','manifest_identity':'synthetic_matrix',
        'binding_identity':binding['identity'],'model_identity':'followup_model','independent_research_success_claimed':False,
        'Q2_Q3_scored':False,'actual_new_model_CPU_acceptance_passed':p95<=1,
        'CPU_acceptance_report_sha256':digest(acceptance),'input_contracts_sha256':digest(e.local/'input_contracts.json'),
        'package':str(archive.relative_to(ROOT)),'package_sha256':digest(archive),
        'package_verification_sha256':digest(archive.with_suffix('.verification.json')),'standalone_package_actually_executed':True}
    write_json(e.local/'completion.json',seal(complete))
    ext={'selected_candidate':'A_foundation_raw','paired_control':'C_no_foundation_raw',
        'selected':{'system':{'mse_improvement_vs_zero':.02}},'development_qualification_passed':True}
    path=e.run/'extension_reviews/synthetic/report.json';write_json(path,ext)
    monkeypatch.setattr(report.extension,'checked_review',lambda path,manifest:({'report':str((e.run/'extension_reviews/synthetic/report.json').relative_to(ROOT))},ext))
    return archive


def test_new_model_handover_preserves_metrics_and_rejects_lost_named_input_checks(evidence,monkeypatch):
    e=evidence;archive=new_model(e,monkeypatch)
    value=report.collect(e.path)
    assert value['engineering_acceptance_passed'] and value['followup_development_qualification_passed']
    assert value['local_delivery']['origins']==256 and value['local_delivery']['standalone_package_actually_executed']
    assert value['local_delivery']['package_sha256']==digest(archive)
    contracts=e.local/'input_contracts.json';row=read_json(contracts);row['checks']={};write_json(contracts,seal(row))
    with pytest.raises(ValueError,match='complete set'):report.publish(e.path)
    assert not (e.run/'handover').exists()


def test_failed_real_latency_cannot_be_replaced_by_good_development_scores(evidence,monkeypatch):
    e=evidence;new_model(e,monkeypatch,p95=1.5)
    _,value=report.publish(e.path)
    assert value['followup_development_qualification_passed'] and not value['engineering_acceptance_passed']
    assert value['local_delivery']['warm_single_origin_p95_seconds']==1.5
    assert not value['automatic_Goal_completion_or_monitor_shutdown']


def test_actual_package_process_logs_are_required_before_publication(evidence,monkeypatch):
    e=evidence;new_model(e,monkeypatch)
    (e.local/'package_processes/2.json').unlink()
    with pytest.raises(ValueError,match='retained process log'):report.publish(e.path)
    assert not (e.run/'handover').exists()


def test_real_report_worker_requires_its_CPU_guard_and_does_not_load_an_unadmitted_matrix():
    guard.RUNS.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=guard.RUNS,prefix='handover_guard_test_') as temporary:
        attempt=Path(temporary)/'attempt'
        missing=ROOT/'outputs/fusion_followup/preparation/no_such_handover_matrix.json'
        command=[sys.executable,str(report.ENTRY),'--manifest',str(missing),'--worker','--attempt',str(attempt)]
        process,_=guard.start(attempt,command,{'total_wall_hours':100,'deadline_utc':DEADLINE},10,
            checkpoint_grace=2,termination_grace=.2,purpose='delivery_cpu')
        try:
            assert process.wait(timeout=12)==1
            log=(attempt/'worker.log').read_text()
            assert 'FileNotFoundError' in log and missing.name in log and 'Worker is not bound' not in log
        finally:report.stop_guard(process,attempt)
