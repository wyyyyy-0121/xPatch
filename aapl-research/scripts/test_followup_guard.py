"""Real CPU process tests; no model, GPU or original-run mutation."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

import psutil
import pytest

from fusion.next.common import ROOT,read_json
from scripts import followup_guard as guard


LIMIT = {'total_wall_hours':100,'deadline_utc':guard.DEADLINE}


@pytest.fixture
def directory():
    guard.RUNS.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=guard.RUNS,prefix='guard_fixture_') as value:
        yield Path(value)


def until(predicate,seconds=6):
    deadline = time.monotonic()+seconds
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(.03)
    raise AssertionError('CPU process fixture did not reach its expected state')


def ignoring_tree(directory):
    child = ("import os,signal,time,pathlib,json,sys; "
             "signal.signal(signal.SIGTERM,signal.SIG_IGN); signal.signal(signal.SIGINT,signal.SIG_IGN); "
             "pathlib.Path(sys.argv[1]).write_text(json.dumps({'pid':os.getpid()})); time.sleep(60)")
    parent = ("import os,subprocess,sys,time,signal,pathlib,json; "
              "signal.signal(signal.SIGTERM,signal.SIG_IGN); signal.signal(signal.SIGINT,signal.SIG_IGN); "
              f"subprocess.Popen([sys.executable,'-c',{child!r},sys.argv[2]]); "
              "pathlib.Path(sys.argv[1]).write_text(json.dumps({'pid':os.getpid()})); time.sleep(60)")
    return [sys.executable,'-c',parent,str(directory/'parent.json'),str(directory/'child.json')]


def terminate_guard(process):
    if process.poll() is None:
        process.terminate()
        process.wait(timeout=5)


def assert_gone(pid):
    try:
        assert psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        pass


def test_stage_cap_stops_parent_and_child_with_grace_inside_the_cutoff(directory):
    process,dispatch = guard.start(directory,ignoring_tree(directory),LIMIT,2.5,
                                  checkpoint_grace=.6,termination_grace=.25)
    try:
        until(lambda:(directory/'child.json').exists())
        assert process.wait(timeout=7) == 1
        state = read_json(directory/'guard_state.json')
        assert state['reason'] == 'stage_time_limit' and state['stage'] == 'stopped'
        assert state['live_group_members'] == []
        assert [row['signal'] for row in state['signal_events']] == [signal.SIGINT,signal.SIGTERM,signal.SIGKILL]
        kill = state['signal_events'][-1]
        assert kill['sent'] and kill['sent_at_unix']-dispatch['stop_by_unix'] < .3
        for name in ('parent.json','child.json'):
            assert_gone(read_json(directory/name)['pid'])
    finally:
        terminate_guard(process)


@pytest.mark.parametrize('purpose',['exploration','delivery_fit','delivery_cpu'])
def test_normal_worker_exits_successfully_without_a_stop_signal(directory,purpose):
    process,_ = guard.start(directory,[sys.executable,'-c','pass'],LIMIT,4,
                            checkpoint_grace=.5,termination_grace=.2,purpose=purpose)
    assert process.wait(timeout=6) == 0
    state = read_json(directory/'guard_state.json')
    assert state['stage'] == 'completed' and state['worker_returncode'] == 0
    assert state['purpose']==purpose and state['delivery_reserve_seconds']==guard.PURPOSE_RESERVES[purpose]
    assert state['signal_events'] == []
    with pytest.raises(RuntimeError,match='previous guard request'):
        guard.start(directory,[sys.executable,'-c','pass'],LIMIT,4,
                    checkpoint_grace=.5,termination_grace=.2)


def test_exploration_and_delivery_fit_have_separate_cutoffs_without_moving_the_deadline():
    from datetime import datetime
    deadline = datetime.fromisoformat(guard.DEADLINE).timestamp()
    exploration,reserve = guard.stage_cutoff(deadline-9*3600,6*3600)
    assert exploration==deadline-8*3600 and reserve==8*3600
    delivery,reserve = guard.stage_cutoff(deadline-7*3600,6*3600,'delivery_fit')
    assert delivery==deadline-2*3600 and reserve==2*3600
    too_late,_ = guard.stage_cutoff(deadline-3600,6*3600,'delivery_fit')
    assert too_late<=deadline-3600  # A launcher must reject this expired window.
    cpu,reserve=guard.stage_cutoff(deadline-3600,2*3600,'delivery_cpu')
    assert cpu==deadline and reserve==0
    with pytest.raises(ValueError,match='Unknown'):guard.stage_cutoff(deadline,10,'unbounded')
    with pytest.raises(ValueError,match='finite stage cap'):
        guard.stage_cutoff(deadline,7*3600,'delivery_fit')


def test_worker_exit_does_not_leave_a_background_child(directory):
    child = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"
    parent = ("import subprocess,sys,json,pathlib; "
              f"p=subprocess.Popen([sys.executable,'-c',{child!r}]); "
              "pathlib.Path(sys.argv[1]).write_text(json.dumps({'pid':p.pid}))")
    process,_ = guard.start(directory,[sys.executable,'-c',parent,str(directory/'child.json')],LIMIT,4,
                            checkpoint_grace=.5,termination_grace=.2)
    try:
        assert process.wait(timeout=6) == 1
        state = read_json(directory/'guard_state.json')
        assert state['reason'] == 'leader_exited_with_live_descendants'
        assert state['worker_returncode'] == 0 and state['live_group_members'] == []
        assert_gone(read_json(directory/'child.json')['pid'])
    finally:
        terminate_guard(process)


def test_killed_calling_scheduler_cannot_orphan_the_worker_group(directory):
    command = ignoring_tree(directory)
    program = ("from scripts.followup_guard import start; import json,pathlib,time; "
               f"p,r=start({str(directory)!r},{command!r},{LIMIT!r},10,checkpoint_grace=.5,termination_grace=.2); "
               f"pathlib.Path({str(directory/'caller_ready.json')!r}).write_text(json.dumps(r)); time.sleep(60)")
    owner = subprocess.Popen([sys.executable,'-c',program],cwd=ROOT,start_new_session=True)
    dispatch = None
    try:
        until(lambda:(directory/'caller_ready.json').exists() and (directory/'child.json').exists())
        dispatch = read_json(directory/'caller_ready.json')
        assert guard.exact_alive(dispatch['guard'])
        owner.kill();owner.wait(timeout=2)
        def stopped():
            path = directory/'guard_state.json'
            return path.exists() and read_json(path)['stage'] == 'stopped'
        until(stopped)
        state = read_json(directory/'guard_state.json')
        assert state['reason'] == 'calling_scheduler_exited' and state['live_group_members'] == []
        for name in ('parent.json','child.json'):
            assert_gone(read_json(directory/name)['pid'])
    finally:
        if owner.poll() is None:
            owner.kill();owner.wait(timeout=2)
        if dispatch is not None and guard.exact_alive(dispatch['guard']):
            os.kill(dispatch['guard']['pid'],signal.SIGTERM)
            until(lambda:not guard.exact_alive(dispatch['guard']))


def test_only_one_guard_can_run_even_in_different_output_directories(directory):
    first_dir,second_dir = directory/'first',directory/'second'
    first,_ = guard.start(first_dir,[sys.executable,'-c','import time; time.sleep(60)'],LIMIT,8,
                          checkpoint_grace=.5,termination_grace=.2)
    second = None
    try:
        until(lambda:(first_dir/'guard_state.json').exists() and read_json(first_dir/'guard_state.json')['stage'] == 'running')
        second,_ = guard.start(second_dir,[sys.executable,'-c','pass'],LIMIT,4,
                               checkpoint_grace=.5,termination_grace=.2)
        assert second.wait(timeout=4) == 1
        state = read_json(second_dir/'guard_state.json')
        assert state['stage'] == 'refused' and state['worker_started'] is False
        assert state['reason'] == 'another_followup_guard_is_active'
        assert first.poll() is None
    finally:
        if second is not None:
            terminate_guard(second)
        terminate_guard(first)
    assert read_json(first_dir/'guard_state.json')['reason'] == 'guard_received_stop_signal'


def test_wrong_identity_extended_deadline_or_stage_cap_is_rejected(directory):
    row = guard.process_identity()
    assert guard.exact_alive(row)
    assert not guard.exact_alive({**row,'created_unix':row['created_unix']-1})
    for wrong in ({**LIMIT,'total_wall_hours':101},
                  {**LIMIT,'deadline_utc':'2026-11-10T09:34:22+00:00'}):
        with pytest.raises(ValueError,match='must not change'):
            guard.start(directory,[sys.executable,'-c','pass'],wrong,10)
    for seconds in (float('nan'),float('inf'),0,guard.MAX_STAGE_SECONDS+1):
        with pytest.raises(ValueError,match='finite stage cap'):
            guard.start(directory,[sys.executable,'-c','pass'],LIMIT,seconds)
    assert not (directory/'guard_request.json').exists()
