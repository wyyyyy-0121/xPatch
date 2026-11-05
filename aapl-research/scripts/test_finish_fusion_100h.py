"""Guard against wrong-run downloads, accidental Q3 entry and orphaned CPU work."""
import copy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
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
