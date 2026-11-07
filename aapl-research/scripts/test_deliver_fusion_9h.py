"""Real synthetic CPU replay/package and malicious/conflicting return checks."""
import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from fusion.next.common import ROOT,digest,identity,read_json,write_json
from scripts import deliver_fusion_9h as nine
from scripts import deliver_fusion_followup as old
from scripts.test_followup_return import evidence,prepared,fitted
from scripts.test_followup_delivery import local_model


PROGRAM='''
import sys
from pathlib import Path
from fusion.next import panel,data
from fusion.next.common import read_json
from scripts import deliver_fusion_followup as d,deliver_fusion_9h as nine
base,follow,local,freeze,deployment,phase,resume,stop=sys.argv[1:]
d.BASE=Path(base);d.FOLLOW=Path(follow);panel.OUT=Path(base);data.OUT=Path(base)
nine.focus.FOLLOW=Path(follow);d.local_runtime=nine.measured_runtime
local=Path(local)
d.run_phase(phase,local,read_json(freeze),Path(deployment),read_json(local/'binding.json'),
    resume=resume=='yes',stop_after=None if stop=='none' else int(stop))
'''


def test_actual_birth_clock_two_process_replay_and_three_package_processes(local_model):
    e=local_model
    e.binding.pop('identity')
    e.binding['cold_timing']={'clock':'CLOCK_BOOTTIME minus kernel birth ticks','full_process_to_first_durable_output':True}
    e.binding['identity']=identity(e.binding);write_json(e.local/'binding.json',e.binding)
    path=e.follow/'preparation'/('nine_hour_'+e.manifest['identity'][:16]+'.json')
    write_json(path,{'synthetic_CPU_fixture_only':True})
    before={str(p.relative_to(e.original)):digest(p) for p in e.original.rglob('*') if p.is_file()}
    def run(phase,resume='no',stop='none'):
        args=[sys.executable,'-c',PROGRAM,str(e.original),str(e.follow),str(e.local),
            str(e.run/'delivery/engineering_freeze.json'),str(e.deployment),phase,resume,stop]
        result=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,timeout=90,
            env={**os.environ,'PYTHONPATH':str(ROOT),'FUSION_EXECUTION_PROFILE':'local','CUDA_VISIBLE_DEVICES':''})
        assert result.returncode==0,result.stderr[-5000:]
    run('replay',stop='128');run('replay',resume='yes')
    report=old.checked_replay(e.local,e.binding)
    rows=report['cold_load_measurements_by_process']
    assert report['cold_timing_complete'] and len(rows)==2
    assert len({(r['kernel_process_birth']['boot_id'],r['pid'],r['kernel_process_birth']['start_ticks']) for r in rows})==2
    for r in rows:
        birth=r['kernel_process_birth']
        assert r['full_process_to_first_output_seconds']==r['first_durable_output_boottime_seconds']-birth['start_ticks']/birth['ticks_per_second']
        assert r['full_process_to_first_output_seconds']>=r['model_cold_load_seconds']
    run('package')
    report=old.checked_replay(e.local,e.binding)
    archive=e.local/'local_packages'/(report['model_identity']+'.tar')
    verified=read_json(archive.with_suffix('.verification.json'))
    assert verified['isolated_example_and_fresh_process_resume_passed']
    assert len(verified['actual_processes'])==3
    assert read_json(e.local/'input_contracts.json')['all_passed']
    assert {str(p.relative_to(e.original)):digest(p) for p in e.original.rglob('*') if p.is_file()}==before


@pytest.mark.parametrize('name',['../secret','/etc/passwd','outputs/../secret','outputs/./secret','outputs\\secret'])
def test_return_paths_cannot_escape(name):
    with pytest.raises(ValueError):nine.safe_relative(name)


def test_return_excludes_raw_data_and_final_period_arrays():
    manifest={'identity':'fixture'}
    for suffix in ('data/target.npy','gpu/data/target.npy','gpu/representation_cache/cached.npy','raw/AAPL.csv'):
        assert not nine.permitted('outputs/fusion_followup/runs/fixture/'+suffix,manifest)


def test_duplicate_archive_member_is_refused_before_restoration(tmp_path):
    path=tmp_path/'duplicate.tar';name='outputs/fusion_followup/runs/fixture/review.json'
    with tarfile.open(path,'w') as tar:
        for _ in range(2):
            entry=tarfile.TarInfo(name);entry.size=2;tar.addfile(entry,io.BytesIO(b'{}'))
    import hashlib
    receipt={'manifest_identity':'fixture','sha256':digest(path),'bytes':path.stat().st_size,
        'Q2_Q3_scored':False,'members':{name:{'bytes':2,'sha256':hashlib.sha256(b'{}').hexdigest()}}}
    with pytest.raises(ValueError,match='duplicate'):nine.inspect(path,receipt,{'identity':'fixture'})


def test_remote_return_imports_from_admitted_root_when_SSH_starts_elsewhere(tmp_path,monkeypatch):
    root=tmp_path/'cloud';scripts=root/'scripts';scripts.mkdir(parents=True)
    (scripts/'__init__.py').write_text('')
    (scripts/'fusion_9h_guard.py').write_text('def exact_alive(row): return False\ndef group_members(row): return []\n')
    run=root/'outputs/fusion_followup/runs/fixture'
    write_json(run/'attempts/gpu_01/guard_state.json',{'stage':'completed','worker_returncode':0,'guard':{},'worker':{}})
    write_json(run/'scheduler_dispatch.json',{'scheduler':{}})
    write_json(run/'return/receipt.json',{'synthetic_receipt_only':True})
    monkeypatch.setattr(nine,'REMOTE_ROOT',str(root))
    env={k:v for k,v in os.environ.items() if k!='PYTHONPATH'}
    result=subprocess.run([sys.executable,'-'],input=nine.remote_return_program({'identity':'fixture'}),
        cwd=tmp_path,env=env,capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stderr
    import json
    assert json.loads(result.stdout)=={'synthetic_receipt_only':True}
