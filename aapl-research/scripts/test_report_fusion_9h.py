"""Audit refusals plus actual synthetic replay/package provenance verification."""
from datetime import datetime,timezone
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from fusion.next.common import ROOT,digest,identity,read_json,write_json
from scripts import report_fusion_9h as report
from scripts import deliver_fusion_9h as returned
from scripts import test_deliver_fusion_9h as integration
from scripts.test_followup_return import evidence,prepared,fitted
from scripts.test_followup_delivery import local_model


def seal(value):
    value=copy.deepcopy(value);value.pop('identity',None);value['identity']=identity(value)
    return value


def test_actual_synthetic_package_is_audited_with_raw_process_logs(local_model):
    e=local_model
    integration.test_actual_birth_clock_two_process_replay_and_three_package_processes(e)
    directory=e.local;model=e.binding['outcome']['deployment_identity']
    report_path=directory/'deployment_acceptance'/model/'report.json'
    result=read_json(report_path);archive=directory/'local_packages'/(model+'.tar')
    verified=archive.with_suffix('.verification.json')
    complete=seal({'stage':'nine_hour_CPU_delivery_complete','manifest_identity':e.manifest['identity'],
        'binding_identity':e.binding['identity'],'model_identity':model,
        'CPU_report_sha256':digest(report_path),'package':str(archive.relative_to(ROOT)),
        'package_sha256':digest(archive),'verification_sha256':digest(verified),
        'actual_new_model_CPU_acceptance_passed':result['engineering_acceptance_passed'],
        'standalone_package_actually_executed':True,'independent_success_claimed':False,'Q2_Q3_scored':False})
    write_json(directory/'completion.json',complete)
    provenance={};value=report.actual_delivery(e.run,e.binding,provenance)
    assert value['three_distinct_actual_package_processes_verified']
    assert value['all_named_input_contracts_passed'] and value['origins']==256
    assert value['package_files_rehashed']>50
    log=next((directory/'package_processes').glob('*.json'));old=log.read_bytes();log.unlink()
    with pytest.raises(ValueError,match='raw log'):report.actual_delivery(e.run,e.binding,{})
    log.write_bytes(old)
    write_json(directory/'completion.json',seal({**complete,'actual_new_model_CPU_acceptance_passed':not result['engineering_acceptance_passed']}))
    with pytest.raises(ValueError,match='disagree'):report.actual_delivery(e.run,e.binding,{})


@pytest.mark.parametrize('cloud',[False,True])
def test_dependency_root_mapping_and_actual_exit_are_checked(tmp_path,monkeypatch,cloud):
    monkeypatch.setattr(report,'note',lambda path,provenance:read_json(path))
    monkeypatch.setattr(report.guard,'exact_alive',lambda row:False)
    monkeypatch.setattr(report.guard,'group_members',lambda row:[])
    start=datetime.now(timezone.utc).timestamp()-5
    cwd=returned.REMOTE_ROOT if cloud else str(ROOT)
    request=seal({'cwd':cwd,'command':[sys.executable,str(Path(cwd)/'scripts/fusion_9h.py')],
        'limit':{'fixture':True},'requested_at_unix':start,'stop_by_unix':start+60})
    write_json(tmp_path/'guard_request.json',request)
    state={'guard':{},'worker':{},'stage':'completed','worker_returncode':0,'live_group_members':[],
        'request_identity':request['identity'],'updated_at_utc':datetime.now(timezone.utc).isoformat()}
    write_json(tmp_path/'guard_state.json',state)
    assert report.exited_attempt(tmp_path,request['limit'],report.focus.ENTRY,{})['stage_wall_seconds']>0
    monkeypatch.setattr(report.guard,'exact_alive',lambda row:True)
    with pytest.raises(RuntimeError,match='live guard'):report.exited_attempt(tmp_path,request['limit'],report.focus.ENTRY,{})


def test_usage_requires_every_complete_development_and_engineering_fit(tmp_path,monkeypatch):
    monkeypatch.setattr(report,'note',lambda path,provenance:read_json(path))
    for group,count in [('gpu',9),('delivery_fit',3)]:
        for i in range(count):write_json(tmp_path/group/'usage_events'/(str(i)+'.json'),{'status':'completed','gpu':True,'wall_seconds':2.})
    assert report.usage(tmp_path,{})['accounted_GPU_job_wall_hours']==24/3600
    (tmp_path/'delivery_fit/usage_events/0.json').unlink()
    with pytest.raises(ValueError,match='nine development'):report.usage(tmp_path,{})


def test_accounting_remote_program_preserves_fixed_only_allowlist(tmp_path,monkeypatch):
    root=tmp_path/'cloud';scripts=root/'scripts';scripts.mkdir(parents=True)
    (scripts/'__init__.py').write_text('')
    (scripts/'fusion_9h_guard.py').write_text('def exact_alive(row): return False\ndef group_members(row): return []\n')
    run=root/'outputs/fusion_followup/runs/new';prior=root/'outputs/fusion_followup/runs/parent/attempts'
    write_json(run/'attempts/gpu_01/guard_state.json',{'stage':'completed','worker_returncode':0,'guard':{},'worker':{}})
    write_json(run/'scheduler_dispatch.json',{'scheduler':{}})
    write_json(run/'return/receipt.json',{'identity':'receipt'})
    for i in range(3):write_json(run/'delivery_fit/usage_events'/(str(i)+'.json'),{'status':'completed'})
    for name in ('probe_01','probe_02','probe_03'):
        for filename in ('guard_request.json','guard_state.json','probe_result.json'):write_json(prior/name/filename,{'fixture':True})
    monkeypatch.setattr(returned,'REMOTE_ROOT',str(root))
    result=subprocess.run([sys.executable,'-'],input=returned.remote_return_program({'identity':'new','parent_identity':'parent'},True),
        cwd=tmp_path,capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stderr
    proof=json.loads(result.stdout)
    assert len(proof['files'])==12 and proof['return_receipt_identity']=='receipt'
    assert all('/usage_events/' in p or '/attempts/probe_' in p for p in proof['files'])


def test_report_cannot_be_collected_before_actual_delivery_completes(tmp_path,monkeypatch):
    monkeypatch.setattr(report.focus,'checked',lambda path:({'identity':'fixture'}, {}, {}))
    path=ROOT/'tmp/nonexistent_nine_hour_fixture.json'
    # A missing actual model/CPU binding must fail, rather than publish success.
    with pytest.raises(FileNotFoundError):report.collect(path)


@pytest.mark.parametrize('conflict',[False,True])
def test_accounting_restore_checks_all_conflicts_before_writing(tmp_path,monkeypatch,conflict):
    import hashlib
    from types import SimpleNamespace
    monkeypatch.setattr(returned,'ROOT',tmp_path)
    monkeypatch.setattr(returned.focus,'FOLLOW',tmp_path/'outputs/fusion_followup')
    manifest={'identity':'new','parent_identity':'parent'}
    names=['outputs/fusion_followup/runs/new/delivery_fit/usage_events/'+str(i)+'.json' for i in range(3)]
    names += ['outputs/fusion_followup/runs/parent/attempts/'+name+'/'+filename
        for name in ('probe_01','probe_02','probe_03') for filename in ('guard_request.json','guard_state.json','probe_result.json')]
    content='{"synthetic_accounting":true}\n';data=content.encode()
    proof={'manifest_identity':'new','return_receipt_identity':'receipt',
        'files':{name:{'content':content,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()} for name in names}}
    monkeypatch.setattr(returned.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout=json.dumps(proof)))
    if conflict:
        bad=tmp_path/names[-1];bad.parent.mkdir(parents=True);bad.write_text('preserve me')
        with pytest.raises(ValueError,match='Preserve different'):returned.fetch_accounting(manifest,{'identity':'receipt'})
        assert bad.read_text()=='preserve me'
        assert not (tmp_path/names[0]).exists()
    else:
        first=returned.fetch_accounting(manifest,{'identity':'receipt'})
        assert all((tmp_path/name).read_text()==content for name in names)
        # A verified receipt is reused locally without making another SSH call.
        monkeypatch.setattr(returned.subprocess,'run',lambda *a,**k:pytest.fail('Unexpected duplicate SSH fetch'))
        assert returned.fetch_accounting(manifest,{'identity':'receipt'})==first
