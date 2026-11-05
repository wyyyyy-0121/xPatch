import copy
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory

import psutil
import pytest

from fusion.next.common import ROOT, identity, digest, read_json, write_json
from scripts import handoff_fusion_speed as handoff
from scripts import start_handoff_completion as receiver


@pytest.fixture
def records(monkeypatch):
    with TemporaryDirectory(dir=ROOT/'tmp', prefix='unit_handoff_') as temporary:
        root = Path(temporary)
        out = root/'outputs'
        monkeypatch.setattr(handoff, 'ROOT', root)
        monkeypatch.setattr(handoff, 'OUT', out)
        monkeypatch.setattr(handoff, 'REQUEST', out/'cloud/speed_handoff_request.json')
        monkeypatch.setattr(handoff, 'STATE', out/'cloud/speed_handoff_state.json')
        monkeypatch.setattr(receiver, 'ROOT', root)
        monkeypatch.setattr(receiver, 'OUT', out)
        monkeypatch.setattr(receiver.finish, 'HERE', out/'cloud/local_completion')
        limit = {'total_wall_hours':100,'deadline_utc':(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()}
        old = {'pid':123,'created_unix':4.,'scope_identity':'scope','script_sha256':'runner'}
        request = {'old_dispatch':old,'limit':limit,'source_identity':'source',
                   'speed_policy_identity':'speed','numerical_policy_identity':'numerical',
                   'stage_receipt':'F4_complete.json',
                   'completion_wrapper_sha256':digest(Path(receiver.__file__))}
        request['identity'] = identity(request)
        write_json(out/'cloud/speed_handoff_request.json',request)
        binding = {'dispatch':copy.deepcopy(old),'limit':copy.deepcopy(limit),'source_identity':'source',
                   'scope_identity':'scope','numerical_policy_identity':'numerical'}
        snapshot = {'dispatch':copy.deepcopy(old),'limit':copy.deepcopy(limit),
                    'runner_alive':True,'limiter_alive':True,
                    'numerical_policy':{'identity':'numerical'},
                    'state':{'stage':'F4_running','source_identity':'source','scope_identity':'scope'}}
        write_json(out/'cloud/bounded_run_state.json',snapshot['state'])
        yield out, request, binding, snapshot


def phase(request, name, **extra):
    return {'request_identity':request['identity'],'phase':name,
            'at_utc':datetime.now(timezone.utc).isoformat(), **extra}


def test_no_signal_until_bound_stage_receipt_exists(records):
    out, request, _, _ = records
    assert handoff.complete_boundary(request) is False
    write_json(out/'F4_complete.json',{'source_identity':'another source'})
    with pytest.raises(ValueError, match='bound experiment'):
        handoff.complete_boundary(request)


def test_pending_observer_accepts_only_planned_interrupt_and_bounded_window(records):
    _, request, binding, snapshot = records
    snapshot['handoff'] = phase(request,'stopping_old_runner')
    snapshot['runner_alive'] = False
    snapshot['state'].update(stage='stopped_for_inspection',error_type='KeyboardInterrupt')
    assert receiver.check(snapshot,binding) is False
    snapshot['state']['error_type'] = 'RuntimeError'
    with pytest.raises(RuntimeError, match='Unplanned'):
        receiver.check(snapshot,binding)
    snapshot['state']['error_type'] = 'KeyboardInterrupt'
    snapshot['handoff']['at_utc'] = (datetime.now(timezone.utc)-timedelta(seconds=181)).isoformat()
    with pytest.raises(TimeoutError, match='transition window'):
        receiver.check(snapshot,binding)


def test_missing_runner_is_not_excused_outside_handoff(records):
    _, request, binding, snapshot = records
    snapshot['runner_alive'] = False
    snapshot['handoff'] = phase(request,'waiting_for_stage')
    with pytest.raises(RuntimeError, match='without a completed return'):
        receiver.check(snapshot,binding)


def test_exact_completed_chain_rebinds_and_resumes_original_checks(records):
    out, request, binding, snapshot = records
    new = {**request['old_dispatch'],'pid':321,'created_unix':8.,
           'previous_dispatch':request['old_dispatch'], 'speed_policy_identity':'speed',
           'handoff_request_identity':request['identity']}
    snapshot['dispatch'] = new
    snapshot['handoff'] = phase(request,'complete',new_dispatch=new,old_runner_confirmed_exited=True)
    snapshot['state']['stage'] = 'F5_running'
    assert receiver.check(snapshot,binding) is False
    assert read_json(out/'cloud/local_completion/binding.json')['dispatch'] == new
    assert read_json(out/'cloud/local_completion/before_speed_handoff_binding.json')['dispatch'] == request['old_dispatch']
    snapshot['runner_alive'] = False
    with pytest.raises(RuntimeError, match='without a completed return'):
        receiver.check(snapshot,binding)


def test_unrelated_dispatch_or_changed_budget_is_rejected(records):
    _, request, binding, snapshot = records
    snapshot['dispatch']['pid'] = 999
    snapshot['handoff'] = phase(request,'complete')
    with pytest.raises(ValueError, match='Unapproved'):
        receiver.check(snapshot,binding)
    snapshot['limit']['total_wall_hours'] = 101
    with pytest.raises(ValueError, match='time limit'):
        receiver.check(snapshot,binding)


def test_limiter_must_stay_alive_through_transition(records):
    _, request, binding, snapshot = records
    snapshot['handoff'] = phase(request,'starting_fast_runner')
    snapshot['limiter_alive'] = False
    with pytest.raises(RuntimeError, match='limiter'):
        receiver.check(snapshot,binding)


def test_pid_reuse_cannot_signal_another_process(records):
    _, _, _, _ = records
    row = {'pid':os.getpid(),'created_unix':psutil.Process().create_time()+1}
    with pytest.raises(RuntimeError, match='disappeared'):
        handoff.stop_exact(row)


def test_actual_signal_waits_for_completed_work_and_preserves_exit_file(records):
    out, _, _, _ = records
    script = handoff.ROOT/'scripts/run_fusion_100h.py'
    script.parent.mkdir()
    committed = handoff.ROOT/'committed.json'
    exited = handoff.ROOT/'exited.json'
    script.write_text('import pathlib,signal,time\n'
        'pathlib.Path("committed.json").write_text("saved stage")\n'
        'def stop(*a):\n'
        '    pathlib.Path("exited.json").write_text("checkpoint preserved")\n'
        '    raise SystemExit(0)\n'
        'signal.signal(signal.SIGINT,stop)\n'
        'while True: time.sleep(.01)\n')
    p = subprocess.Popen([sys.executable,'scripts/run_fusion_100h.py'],cwd=handoff.ROOT)
    try:
        import time
        deadline = time.monotonic()+5
        while not committed.exists() and time.monotonic()<deadline:
            time.sleep(.01)
        assert committed.exists()
        handoff.stop_exact({'pid':p.pid,'created_unix':psutil.Process(p.pid).create_time()})
        p.wait(timeout=5)
        assert committed.read_text() == 'saved stage'
        assert exited.read_text() == 'checkpoint preserved'
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()


def test_real_process_handoff_starts_one_successor_after_old_exit(records, monkeypatch):
    out, request, _, _ = records
    script = handoff.ROOT/'scripts/run_fusion_100h.py'
    script.parent.mkdir()
    script.write_text('import pathlib,signal,time\n'
        'signal.signal(signal.SIGINT,lambda *a: exit(0))\n'
        'pathlib.Path("ready").touch()\n'
        'while True: time.sleep(.01)\n')
    raw_popen = subprocess.Popen
    old = raw_popen([sys.executable,'scripts/run_fusion_100h.py'],cwd=handoff.ROOT)
    children = []
    try:
        import time
        deadline = time.monotonic()+5
        while not (handoff.ROOT/'ready').exists() and time.monotonic()<deadline:
            time.sleep(.01)
        assert (handoff.ROOT/'ready').exists()
        request.update(old_dispatch={'pid':old.pid,'created_unix':psutil.Process(old.pid).create_time()},
                       cpu_ids='12',cpu_threads=2)
        write_json(handoff.REQUEST,request)
        write_json(out/'F4_complete.json',{'committed':True})
        (out/'cloud/logs').mkdir()
        monkeypatch.setenv('FUSION_EXECUTION_PROFILE','autodl')
        monkeypatch.setattr(handoff,'check_request',lambda request:request['limit'])
        monkeypatch.setattr(handoff,'complete_boundary',lambda request:(out/'F4_complete.json').exists())
        def launch(command, **kwargs):
            assert not handoff.alive(request['old_dispatch'])
            assert command[-2:] == ['scripts/start_fast_fusion.py','scripts/run_fusion_100h.py']
            program = ('import pathlib,json,time,os; '
                'b=pathlib.Path("outputs"); '
                '(b/"cloud/state.pending").write_text(json.dumps({"stage":"F5_running"})); '
                'os.replace(b/"cloud/state.pending",b/"cloud/bounded_run_state.json"); '
                '(b/"performance/speed_activations").mkdir(parents=True); '
                '(b/"performance/speed_activations/test.json").write_text("{}"); '
                'time.sleep(30)')
            child=raw_popen([sys.executable,'-c',program],**kwargs)
            children.append(child)
            return child
        monkeypatch.setattr(handoff.subprocess,'Popen',launch)
        handoff.run()
        assert len(children)==1
        result=read_json(handoff.STATE)
        assert result['phase']=='complete' and result['old_runner_confirmed_exited']
        assert result['new_dispatch']['previous_dispatch']==request['old_dispatch']
        assert result['new_dispatch']['speed_policy_identity']=='speed'
    finally:
        for child in [old,*children]:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
