"""Guard against wrong-run downloads, accidental Q3 entry and orphaned CPU work."""
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

import psutil
import pytest

from scripts import finish_fusion_100h as finish


def records():
    dispatch = {'pid':123, 'created_unix':456., 'scope_identity':'scope', 'script_sha256':'runner'}
    limit = {'deadline_utc':'2099-01-01T00:00:00+00:00', 'total_wall_hours':100}
    binding = {'dispatch':dispatch, 'limit':limit, 'scope_identity':'scope', 'source_identity':'source'}
    snapshot = {'dispatch':copy.deepcopy(dispatch), 'limit':copy.deepcopy(limit), 'runner_alive':True, 'limiter_alive':True,
        'state':{'stage':'F3_running', 'scope_identity':'scope', 'source_identity':'source'}, 'receipt':None}
    return snapshot, binding


def test_observer_binds_exact_run_and_rejects_silent_exit():
    snapshot, binding = records()
    assert finish.check_binding(snapshot, binding) is False
    snapshot['dispatch']['created_unix'] += 1
    with pytest.raises(ValueError, match='identity changed'):
        finish.check_binding(snapshot, binding)
    snapshot, binding = records();snapshot['runner_alive'] = False
    with pytest.raises(RuntimeError, match='without a completed return'):
        finish.check_binding(snapshot, binding)


def test_ready_requires_exact_return_artifact_but_not_live_runner():
    snapshot, binding = records();snapshot['state']['stage'] = 'F6_complete_return_ready'
    snapshot['runner_alive'] = False
    with pytest.raises(ValueError, match='receipt'):
        finish.check_binding(snapshot, binding)
    snapshot['receipt'] = {'source_identity':'source', 'archive':'outputs/fusion_next/cloud/return_bundle.tar'}
    assert finish.check_binding(snapshot, binding)
    snapshot['receipt']['archive'] = '../other.tar'
    with pytest.raises(ValueError, match='Unexpected'):
        finish.check_binding(snapshot, binding)


def test_observer_stops_on_failure_or_changed_deadline():
    snapshot, binding = records();snapshot['state']['stage'] = 'stopped_for_inspection'
    with pytest.raises(RuntimeError, match='stopped for inspection'):
        finish.check_binding(snapshot, binding)


def test_observer_rejects_changed_numerical_policy():
    snapshot,binding=records();binding['numerical_policy_identity']='fp32-policy'
    snapshot['numerical_policy']={'identity':'other-policy'}
    with pytest.raises(ValueError,match='inference policy changed'):
        finish.check_binding(snapshot,binding)
    snapshot['numerical_policy']['identity']='fp32-policy'
    assert finish.check_binding(snapshot,binding) is False
    snapshot, binding = records();snapshot['limit']['total_wall_hours'] = 101
    with pytest.raises(ValueError, match='time limit changed'):
        finish.check_binding(snapshot, binding)
    snapshot, binding = records();snapshot['limiter_alive'] = False
    with pytest.raises(RuntimeError, match='limiter is no longer alive'):
        finish.check_binding(snapshot, binding)


def test_negative_development_never_enters_final_test():
    frozen = {'development_gate_passed':False}
    confirmation = {'evaluated':False, 'proceed_to_final_test':False, 'Q3_opened':False}
    assert finish.choose_final(frozen, confirmation) is False
    confirmation['proceed_to_final_test'] = True
    with pytest.raises(ValueError, match='contradicts'):
        finish.choose_final(frozen, confirmation)


def test_final_dispatch_requires_both_gates_and_no_cloud_Q3_access():
    frozen = {'development_gate_passed':True}
    confirmation = {'evaluated':True, 'proceed_to_final_test':False, 'Q3_opened':False}
    assert finish.choose_final(frozen, confirmation) is False
    confirmation['proceed_to_final_test'] = True
    assert finish.choose_final(frozen, confirmation) is True
    confirmation['Q3_opened'] = True
    with pytest.raises(ValueError, match='Q3 access'):
        finish.choose_final(frozen, confirmation)


def test_external_compatibility_only_adds_exported_diagnostic():
    from fusion.next import restore
    original = set(restore.FIXED)
    before = finish.source_identity()
    with pytest.raises(RuntimeError):
        with finish.return_compatibility() as checked:
            assert checked.FIXED-original == {'batch_scaling.json'}
            assert checked.allowed('outputs/fusion_next/batch_scaling.json')
            assert not checked.allowed('outputs/fusion_next/arbitrary.json')
            assert not checked.allowed('outputs/fusion_next/../../private.json')
            raise RuntimeError('controlled exit')
    assert restore.FIXED == original
    assert finish.source_identity() == before


def test_local_deadline_terminates_replay_children(monkeypatch):
    # Actual process tree: termination must include the child, not only its parent.
    with tempfile.TemporaryDirectory(dir=finish.ROOT/'tmp') as temporary:
        directory = Path(temporary)
        monkeypatch.setattr(finish, 'HERE', directory)
        deadline = datetime.now(timezone.utc)+timedelta(seconds=1.5)
        binding = {'limit':{'deadline_utc':deadline.isoformat()}}
        pidfile = directory/'child.json'
        program = "import subprocess,sys,time,json,pathlib; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); pathlib.Path(sys.argv[1]).write_text(json.dumps({'pid':p.pid})); time.sleep(30)"
        with pytest.raises(subprocess.TimeoutExpired):
            finish.Completion(binding).command([sys.executable, '-c', program, str(pidfile)], 'timeout')
        assert pidfile.exists()
        pid = json.loads(pidfile.read_text())['pid']
        try:
            process = psutil.Process(pid)
            assert process.status() == psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            pass


def test_nested_completion_deadline_keeps_grandchildren_in_outer_group(monkeypatch):
    with tempfile.TemporaryDirectory(dir=finish.ROOT/'tmp') as temporary:
        directory = Path(temporary);monkeypatch.setattr(finish, 'HERE', directory)
        deadline = datetime.now(timezone.utc)+timedelta(seconds=2)
        binding = {'limit':{'deadline_utc':deadline.isoformat()}}
        pidfile = directory/'grandchild.json'
        program = '''import json,sys,pathlib,os
from scripts import finish_fusion_100h as finish
finish.HERE=pathlib.Path(sys.argv[1])
binding=json.loads(sys.argv[2])
inner="import subprocess,sys,time,json,pathlib; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); pathlib.Path(sys.argv[1]).write_text(json.dumps({'pid':p.pid,'group':__import__('os').getpgid(p.pid)})); time.sleep(30)"
finish.Completion(binding).command([sys.executable,'-c',inner,sys.argv[3]],'nested')
'''
        with pytest.raises(subprocess.TimeoutExpired):
            finish.Completion(binding).command([sys.executable, '-c', program, str(directory), json.dumps(binding), str(pidfile)],
                'outer', env={**os.environ, 'AAPL_COMPLETION_MANAGED_GROUP':'1'})
        assert pidfile.exists()
        pid = json.loads(pidfile.read_text())['pid']
        try:
            process = psutil.Process(pid)
            assert process.status() == psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            pass


def archive_fixture(directory, mode=None):
    payloads = {'outputs/fusion_next/confirmation.json':b'{"evaluated":false}',
        'outputs/fusion_next/batch_scaling.json':b'{"diagnostic":true}'}
    binding = {'source_identity':'source', 'plan_sha256':'plan', 'scope_identity':'scope'}
    manifest = {'source_identity':'source', 'plan_sha256':'plan', 'Q3_data_included':False,
        'credentials_included':False, 'raw_development_market_data_included':False,
        'files':{name:{'bytes':len(data), 'sha256':hashlib.sha256(data).hexdigest()} for name,data in payloads.items()}}
    if mode == 'changed_member':payloads['outputs/fusion_next/confirmation.json'] = b'{"evaluated":true}'
    if mode == 'unknown_member':payloads['outputs/fusion_next/private.json'] = b'{}'
    payloads['outputs/fusion_next/cloud/return_manifest.json'] = json.dumps(manifest).encode()
    archive = directory/'return.tar'
    with tarfile.open(archive, 'w') as stream:
        for name,data in payloads.items():
            member = tarfile.TarInfo(name);member.size = len(data);stream.addfile(member, io.BytesIO(data))
        if mode == 'duplicate':
            name,data = next(iter(payloads.items()))
            member = tarfile.TarInfo(name);member.size = len(data);stream.addfile(member, io.BytesIO(data))
    receipt = {'sha256':finish.digest(archive), 'bytes':archive.stat().st_size}
    return archive, receipt, binding


def test_archive_preflight_accepts_exported_diagnostic_without_restoring(monkeypatch):
    from scripts import report_fusion_scope
    with tempfile.TemporaryDirectory(dir=finish.ROOT/'tmp') as temporary:
        directory = Path(temporary);monkeypatch.setattr(finish, 'HERE', directory)
        monkeypatch.setattr(report_fusion_scope, 'inspect', lambda _: {
            'scope_identity':'scope', 'stages':{stage:{'completed':True} for stage in ('F3','F4','F5')}})
        archive,receipt,binding = archive_fixture(directory)
        finish.preflight_archive(archive,receipt,binding)
        assert (directory/'archive_preflight.json').exists()
        assert not (directory/'outputs').exists()


@pytest.mark.parametrize('mode', ['changed_member', 'unknown_member', 'duplicate'])
def test_archive_preflight_rejects_invalid_members_before_restoring(monkeypatch, mode):
    with tempfile.TemporaryDirectory(dir=finish.ROOT/'tmp') as temporary:
        directory = Path(temporary);monkeypatch.setattr(finish, 'HERE', directory)
        archive,receipt,binding = archive_fixture(directory, mode)
        with pytest.raises(ValueError, match='(member identity|Unexpected or duplicate)'):
            finish.preflight_archive(archive,receipt,binding)
        assert not (directory/'archive_preflight.json').exists()
