"""Immutable return tests with real tiny fits and explicitly synthetic admission.

No test calls a cloud host, acquires market data or asserts actual research F8.
"""
import copy
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
from types import SimpleNamespace

import pytest

from fusion.next.common import ROOT,digest,identity,read_json,write_json
from scripts import return_fusion_followup as returned
from scripts import fit_fusion_followup as fit
from scripts import followup_runtime as runtime
from scripts import followup_guard as guard
from scripts.test_fit_fusion_followup import evidence,freeze
from scripts.test_run_fusion_followup import NOW


def seal(value):
    value=copy.deepcopy(value);value.pop('identity',None);value['identity']=identity(value)
    return value


@pytest.fixture
def prepared(evidence,monkeypatch):
    e=evidence
    original_model=read_json(e.original/'deployment/pre_Q2/deployment.json')['identity']
    e.manifest['delivery']={'deployment_identity':original_model,'files':{
        'local_packages/'+original_model+'.tar':{'sha256':'synthetic_original_package_fixture_only','bytes':1}}}
    write_json(e.path,e.manifest)
    monkeypatch.setattr(returned,'FOLLOW',e.follow);monkeypatch.setattr(returned,'BASE',e.original)
    monkeypatch.setattr(returned,'check_frozen',lambda path:e.manifest)
    e.pilot={'report':'synthetic_pilot_fixture_only','identity':'synthetic','decision_identity':'synthetic'}
    e.pilot_review={'selected_for_possible_expansion':'C_no_foundation_raw'}
    monkeypatch.setattr(returned.pilot,'checked_report',lambda path,manifest:(e.pilot,e.pilot_review))
    return e


@pytest.fixture
def fitted(prepared):
    e=prepared;path,frozen,run=freeze(e);destination=run/'delivery_fit'
    with runtime.isolated_outputs(destination,e.original):
        fit.fit_and_export(e.panel,e.store,frozen,path,destination,device='cpu')
    # Synthetic control receipts exercise exact validation only. The actual fit
    # above uses two fabricated sessions; it is not an admitted GPU experiment.
    attempt=run/'attempts/fit_attempt_01'
    request=seal({'requested_at_unix':NOW.timestamp()-10,'purpose':'delivery_fit',
        'delivery_reserve_seconds':guard.CPU_DELIVERY_RESERVE_SECONDS,'limit':e.manifest['limit']})
    write_json(attempt/'guard_request.json',request)
    write_json(attempt/'guard_state.json',{'request_identity':request['identity'],'stage':'completed',
        'worker_returncode':0,'worker':None,'updated_at_utc':NOW.isoformat()})
    write_json(attempt/'launch_admission.json',seal({'mode':'delivery_fit','manifest_identity':e.manifest['identity'],
        'entry_sha256':digest(fit.ENTRY),'engineering_freeze_sha256':digest(path)}))
    e.run=run
    return e


def originals(e):
    return {str(p.relative_to(e.original)):digest(p) for p in e.original.rglob('*') if p.is_file()}


def test_real_three_seed_export_returns_restores_exact_bytes_and_keeps_original_inputs(fitted):
    e=fitted;before=originals(e)
    # These excluded paths can exist in real workers. They must not enter a
    # return even if they have a superficially acceptable JSON extension.
    write_json(e.run/'pilot/data/fake.json',{'synthetic_raw_values':'excluded'})
    write_json(e.run/'pilot/representation_cache/fake.json',{'synthetic_latents':'excluded'})
    write_json(e.run/'attempts/return_bundle_01/guard_state.json',{'stage':'running'})
    write_json(e.run/'attempts/return_bundle_01/guard_request.json',{'immutable':'request preserved'})
    for q in ('2026Q2','2026Q3'):
        excluded=e.run/'pilot/trials/pilot/forbidden'/q/'seed_2026/updates_2/prediction.npz'
        excluded.parent.mkdir(parents=True,exist_ok=True);excluded.write_bytes(b'synthetic excluded final array')
    archive,receipt=returned.build(e.path,e.manifest)
    record=returned.inspect_archive(archive,receipt,e.path,e.manifest)
    assert record['outcome']['fresh_system_seeds']==[2026,2027,2028]
    assert not record['CPU_acceptance_proven_by_this_archive']
    assert all(not any(s in n for s in ('/data/','/representation_cache/','return_bundle_01/guard_state','2026Q2','2026Q3')) for n in record['files'])
    assert 'run/attempts/return_bundle_01/guard_request.json' in record['files']
    original_receipt=dict(receipt)
    assert returned.build(e.path,e.manifest)==(archive,original_receipt)
    preserved=e.root/'retained_remote_run';e.run.rename(preserved)
    result=returned.restore(archive,archive.parent/'receipt.json',e.path,e.manifest)
    assert result['all_file_hashes_verified'] and result['outcome']==record['outcome']
    assert not result['actual_new_model_CPU_acceptance_passed'] and not result['independent_research_success_claimed']
    for name,row in record['files'].items():
        if name.startswith('run/'):
            assert digest(e.run/name[4:])==row['sha256']
    assert originals(e)==before
    # A replayed restore may verify identical evidence, never overwrite bytes.
    repeated=returned.restore(archive,archive.parent/'receipt.json',e.path,e.manifest)
    assert repeated==result


def test_archive_hash_corruption_is_rejected_before_any_canonical_restore(fitted):
    e=fitted;archive,receipt=returned.build(e.path,e.manifest)
    corrupted=archive.with_name('corrupted.tar');shutil.copyfile(archive,corrupted)
    with corrupted.open('r+b') as stream:stream.seek(2048);stream.write(b'changed')
    e.run.rename(e.root/'retained_run')
    with pytest.raises(ValueError,match='archive or matrix binding'):
        returned.restore(corrupted,archive.parent/'receipt.json',e.path,e.manifest)
    assert not e.run.exists() and not (archive.parent/'restored_receipt.json').exists()


@pytest.mark.parametrize('name,kind',[
    ('run/../outside.json','file'),('run/pilot/representation_cache/latents.json','file'),
    ('run/delivery_fit/deployment/pre_Q2/unsafe.pt','link'),('matrix.json','file')])
def test_excluded_paths_symlinks_traversal_and_duplicate_members_are_rejected(fitted,name,kind):
    e=fitted;archive,receipt=returned.build(e.path,e.manifest)
    altered=archive.with_name('altered.tar');shutil.copyfile(archive,altered)
    with tarfile.open(altered,'a') as stream:
        info=tarfile.TarInfo(name)
        if kind=='link':info.type=tarfile.SYMTYPE;info.linkname='../../../../outside';stream.addfile(info)
        else:info.size=2;stream.addfile(info,io.BytesIO(b'{}'))
    changed=seal({**receipt,'sha256':digest(altered),'bytes':altered.stat().st_size})
    with pytest.raises(ValueError,match='Unsafe, excluded or duplicate'):
        returned.inspect_archive(altered,changed,e.path,e.manifest)


def test_all_conflicts_are_checked_before_restoring_any_missing_file(fitted):
    e=fitted;archive,receipt=returned.build(e.path,e.manifest)
    record=returned.inspect_archive(archive,receipt,e.path,e.manifest)
    file=next(n for n in record['files'] if n.endswith('experts/xpatch.pt'))
    target=e.run/file[4:];target.write_bytes(b'preserve differing local model evidence')
    missing=e.run/'delivery_fit/fit_progress.json';missing.unlink()
    before=originals(e)
    with pytest.raises(ValueError,match='Preserve differing existing'):
        returned.restore(archive,archive.parent/'receipt.json',e.path,e.manifest)
    assert not missing.exists() and target.read_bytes()==b'preserve differing local model evidence'
    assert originals(e)==before and not (archive.parent/'restored_receipt.json').exists()


def test_negative_pilot_retains_original_package_without_exporting_a_single_seed(prepared):
    e=prepared;e.pilot_review['selected_for_possible_expansion']=None
    run=e.follow/'runs'/e.manifest['identity']
    write_json(run/'pilot/pilot_result.json',{'fixture':'synthetic complete negative pilot metadata only'})
    archive,receipt=returned.build(e.path,e.manifest)
    outcome=returned.inspect_archive(archive,receipt,e.path,e.manifest)['outcome']
    assert outcome['path']=='negative_pilot_retaining_original_package' and not outcome['new_model_exported']
    assert outcome['retained_original_package_sha256']=='synthetic_original_package_fixture_only'
    assert not (run/'delivery_fit').exists()
    (run/'extension').mkdir()
    with pytest.raises(ValueError,match='cannot discard'):returned.completion(e.path,e.manifest)


def test_incomplete_fit_wrong_guard_and_missing_fixed_seed_cannot_be_returned(fitted):
    e=fitted;result=e.run/'delivery_fit/fit_result.json';saved=read_json(result)
    result.unlink()
    with pytest.raises(RuntimeError,match='complete reviewed three-seed'):returned.completion(e.path,e.manifest)
    write_json(result,seal({**saved,'fresh_system_seeds':[2026]}))
    with pytest.raises(ValueError,match='completion differs'):returned.completion(e.path,e.manifest)
    write_json(result,saved)
    path=e.run/'attempts/fit_attempt_01/guard_request.json';value=read_json(path)
    request=seal({**value,'purpose':'exploration'});write_json(path,request)
    state=e.run/'attempts/fit_attempt_01/guard_state.json';row=read_json(state)
    write_json(state,{**row,'request_identity':request['identity']})
    with pytest.raises(ValueError,match='No completed exact'):returned.completion(e.path,e.manifest)


def test_guarded_fetch_binds_remote_receipt_and_hash_then_reuses_verified_download(fitted,monkeypatch):
    e=fitted;archive,receipt=returned.build(e.path,e.manifest)
    remote=e.root/'synthetic_remote_archive.tar';archive.rename(remote);calls=[]
    def network(command,**kwargs):
        calls.append(command)
        assert kwargs['check'] is True and kwargs['timeout']>0
        if command[0]=='ssh':return SimpleNamespace(stdout=json.dumps(receipt))
        assert command[0]=='scp' and command[-2].endswith(receipt['archive'])
        shutil.copyfile(remote,command[-1]);return SimpleNamespace(stdout='')
    monkeypatch.setattr(returned.subprocess,'run',network)
    result=returned.fetch(e.path,e.manifest)
    assert result['all_file_hashes_verified'] and not result['actual_new_model_CPU_acceptance_passed']
    assert [c[0] for c in calls]==['ssh','scp']
    downloaded=remote.parent/'followup/returns'/e.manifest['identity']/'downloaded_bundle.tar'
    assert digest(downloaded)==receipt['sha256']
    assert returned.fetch(e.path,e.manifest)['outcome']==result['outcome']
    assert [c[0] for c in calls]==['ssh','scp','ssh']


def test_fetch_rejects_changed_remote_archive_path_before_scp(fitted,monkeypatch):
    e=fitted;archive,receipt=returned.build(e.path,e.manifest)
    changed=seal({**receipt,'archive':'outputs/fusion_next/raw_prices.tar'});calls=[]
    def network(command,**kwargs):
        calls.append(command);assert command[0]=='ssh'
        return SimpleNamespace(stdout=json.dumps(changed))
    monkeypatch.setattr(returned.subprocess,'run',network)
    with pytest.raises(ValueError,match='fixed archive path'):returned.fetch(e.path,e.manifest)
    assert len(calls)==1 and not (archive.parent/'source_receipt.json').exists()


def test_failed_scp_retains_partial_download_and_retry_restores_verified_bytes(fitted,monkeypatch):
    e=fitted;archive,receipt=returned.build(e.path,e.manifest);failed=[]
    def network(command,**kwargs):
        if command[0]=='ssh':return SimpleNamespace(stdout=json.dumps(receipt))
        partial=Path(command[-1])
        if not failed:
            partial.write_bytes(b'retained interrupted transfer')
            failed.append(partial)
            raise returned.subprocess.CalledProcessError(1,command,stderr='synthetic interrupted transport')
        shutil.copyfile(archive,partial);return SimpleNamespace(stdout='')
    monkeypatch.setattr(returned.subprocess,'run',network)
    with pytest.raises(returned.subprocess.CalledProcessError):returned.fetch(e.path,e.manifest)
    assert failed[0].read_bytes()==b'retained interrupted transfer'
    assert not (archive.parent/'restored_receipt.json').exists()
    result=returned.fetch(e.path,e.manifest)
    assert result['all_file_hashes_verified'] and not result['actual_new_model_CPU_acceptance_passed']
    assert failed[0].read_bytes()==b'retained interrupted transfer'


@pytest.mark.parametrize('purpose',['delivery_fit','delivery_cpu'])
def test_real_return_worker_requires_its_exact_CPU_guard_before_reading_matrix(prepared,purpose):
    e=prepared;attempt=e.root/'return_identity_attempt'
    missing=runtime.FOLLOW/'preparation'/('missing_return_'+e.root.name+'.json')
    command=[sys.executable,str(returned.ENTRY),'--worker','--manifest',str(missing),
        '--mode','bundle','--attempt',str(attempt)]
    process,_=guard.start(attempt,command,e.manifest['limit'],10,checkpoint_grace=2,termination_grace=.2,purpose=purpose)
    try:
        assert process.wait(timeout=12)==1
        log=(attempt/'worker.log').read_text()
        if purpose=='delivery_fit':
            assert 'Delivery CPU worker requires its exact delivery CPU guard purpose' in log
            assert 'FileNotFoundError' not in log
        else:
            assert 'FileNotFoundError' in log and missing.name in log
            assert 'Worker is not bound' not in log
    finally:fit.stop_guard(process,attempt)
