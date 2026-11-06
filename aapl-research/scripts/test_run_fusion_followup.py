"""CPU-only controller checks; synthetic timing never admits an actual GPU run."""
from datetime import datetime,timezone
import os
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
import time

import pytest

from fusion.next.common import ROOT,digest,identity,read_json,write_json
from scripts import followup_guard as guard
from scripts import run_fusion_followup as run


LIMIT = {'total_wall_hours':100,'deadline_utc':run.DEADLINE}
NOW = datetime.fromisoformat('2026-11-08T09:00:00+08:00')
SIGNATURE = {'fixture':'synthetic runtime; no real GPU benchmark'}


@pytest.fixture
def fixture():
    guard.RUNS.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=guard.RUNS,prefix='pilot_fixture_') as temporary:
        directory = Path(temporary)
        manifest = {'identity':'synthetic_matrix','limit':LIMIT,
                    'candidates':[{'name':name,'config':{'factor':i,'lookback':240,'batch_size':256}}
                                  for i,name in enumerate(('A','B','C'))],
                    'folds':{q:{'fit_end':end,'total_updates':100,'score_origins':200} for q,end in
                             zip(run.QUARTERS,('2024-12-30','2025-06-27','2025-12-30'))}}
        yield directory,manifest


def seal(value):
    value['identity'] = identity(value)
    return value


def synthetic_probe(directory,manifest,index):
    attempt = directory/'attempts'/('probe_'+str(index+1).zfill(2))
    candidate = manifest['candidates'][index]
    request = seal({'requested_at_unix':NOW.timestamp()-100,'stage_seconds':600})
    write_json(attempt/'guard_request.json',request)
    worker = {'pid':123,'created_unix':1.}
    write_json(attempt/'guard_state.json',{'request_identity':request['identity'],'stage':'completed',
               'worker_returncode':0,'worker':worker,'updated_at_utc':NOW.isoformat()})
    report = {'manifest_identity':manifest['identity'],'candidate_name':candidate['name'],
              'candidate_config_identity':identity(candidate['config']),'entry_sha256':digest(run.ENTRY),
              'guard_request_identity':request['identity'],'worker':worker,'runtime_signature':SIGNATURE,
              'fit_end':manifest['folds']['2026Q1']['fit_end'],'lookback':240,'batch_size':256,
              'actual_CUDA_probe':True,'training_data_only':True,'updates_measured':run.PROBE_UPDATES,
              'exact_checkpoint_resume_passed':True,'prediction_parity_max_difference':0,
              'initialization_seconds_upper':1.,'seconds_per_update_upper':.5,
              'seconds_per_prediction_origin_upper':.005}
    write_json(attempt/'probe_result.json',seal(report))
    return attempt


def until(predicate,seconds=8):
    end = time.monotonic()+seconds
    while time.monotonic()<end:
        if predicate():return
        time.sleep(.03)
    raise AssertionError('CPU process fixture did not reach its expected state')


def test_pilot_requires_the_date_and_all_three_complete_guarded_probes(fixture):
    directory,manifest = fixture
    with pytest.raises(RuntimeError,match='November 8'):
        run.check_budget('pilot',manifest,directory,SIGNATURE,datetime.fromisoformat('2026-11-08T08:59:59+08:00'))
    with pytest.raises(RuntimeError,match='all three'):
        run.check_budget('pilot',manifest,directory,SIGNATURE,NOW)
    for i in range(3):synthetic_probe(directory,manifest,i)
    budget = run.check_budget('pilot',manifest,directory,SIGNATURE,NOW)
    assert budget['large_training_admitted'] is True and len(budget['projection']['fits']) == 9
    assert budget['projection']['seconds'] == 9*(1+100*.5+200*.005)
    assert budget['stage_seconds'] == run.PILOT_SECONDS
    # This is a pure calculation on synthetic receipts, not an actual launch.
    assert not (directory/'pilot').exists()
    write_json(directory/'pilot/pilot_result.json',{'fixture':'already complete'})
    with pytest.raises(RuntimeError,match='already completed'):
        run.check_budget('pilot',manifest,directory,SIGNATURE,NOW)


@pytest.mark.parametrize('field,value',[
    ('actual_CUDA_probe',False),('training_data_only',False),('exact_checkpoint_resume_passed',False),
    ('prediction_parity_max_difference',1e-8),('updates_measured',1),
    ('runtime_signature',{'fixture':'another driver/CPU runtime'}),('candidate_config_identity','changed'),
    ('worker',{'pid':124,'created_unix':1.}),('fit_end','2026-06-30'),('lookback',48),
    ('seconds_per_update_upper',-1),('seconds_per_prediction_origin_upper',0),
])
def test_wrong_stale_incomplete_or_out_of_scope_benchmark_is_refused(fixture,field,value):
    directory,manifest = fixture
    for i in range(3):synthetic_probe(directory,manifest,i)
    path = directory/'attempts/probe_01/probe_result.json'
    report = read_json(path);report.pop('identity');report[field] = value;write_json(path,seal(report))
    with pytest.raises(ValueError):
        run.check_budget('pilot',manifest,directory,SIGNATURE,NOW)


def test_probe_reservations_are_bounded_even_if_startup_did_not_finish(fixture):
    directory,manifest = fixture
    early = datetime.fromisoformat('2026-11-06T12:00:00+08:00')
    assert run.check_budget('probe',manifest,directory,SIGNATURE,early)['large_training_admitted'] is False
    for i in range(run.MAX_PROBES):
        (directory/'attempts'/('probe_'+str(i+1).zfill(2))).mkdir(parents=True)
    with pytest.raises(RuntimeError,match='five'):
        run.check_budget('probe',manifest,directory,SIGNATURE,early)


def test_stage_failure_or_resume_does_not_reset_the_six_hour_budget(fixture):
    directory,manifest = fixture
    for i in range(3):synthetic_probe(directory,manifest,i)
    attempt = directory/'attempts/pilot_attempt_01'
    request = seal({'requested_at_unix':NOW.timestamp()-run.PILOT_SECONDS+60,'stage_seconds':run.PILOT_SECONDS})
    write_json(attempt/'guard_request.json',request)
    write_json(attempt/'guard_state.json',{'request_identity':request['identity'],'stage':'stopped',
               'worker':None,'updated_at_utc':NOW.isoformat()})
    assert run.stage_remaining(directory) == 60
    with pytest.raises(RuntimeError,match='remaining 6-hour cap'):
        run.check_budget('pilot',manifest,directory,SIGNATURE,NOW)
    (attempt/'guard_state.json').unlink()
    assert run.stage_remaining(directory) == 0  # Unknown work is charged conservatively.


def test_both_absolute_delivery_reserve_and_stage_margin_are_enforced(fixture):
    directory,manifest = fixture
    for i in range(3):synthetic_probe(directory,manifest,i)
    for q in manifest['folds'].values():q['total_updates'] = 10000
    with pytest.raises(RuntimeError,match='remaining 6-hour cap'):
        run.check_budget('pilot',manifest,directory,SIGNATURE,NOW)
    with pytest.raises(RuntimeError,match='reserve'):
        run.check_budget('probe',manifest,directory,SIGNATURE,datetime.fromisoformat('2026-11-09T02:00:00+00:00'))


def test_exact_original_runner_and_limiter_identity_are_required(fixture,monkeypatch):
    directory,manifest = fixture
    monkeypatch.setattr(run,'BASE',directory)
    monkeypatch.setenv('FUSION_EXECUTION_PROFILE','autodl')
    write_json(directory/'cloud/hard_time_limit.json',LIMIT)
    manifest.update(scope_identity='scope',source_identity='engine')
    write_json(directory/'cloud/bounded_run_state.json',{
        'stage':'F6_complete_return_ready','scope_identity':'scope','source_identity':'engine'})
    current = guard.process_identity()
    write_json(directory/'cloud/bounded_run_dispatch.json',{**current,'created_unix':current['created_unix']-10})
    write_json(directory/'cloud/hard_time_limit_state.json',current)
    run.original_idle(manifest,query_gpu=False)
    write_json(directory/'cloud/bounded_run_dispatch.json',current)
    with pytest.raises(RuntimeError,match='actually completed and exited'):
        run.original_idle(manifest,query_gpu=False)
    write_json(directory/'cloud/bounded_run_dispatch.json',{**current,'created_unix':current['created_unix']-10})
    write_json(directory/'cloud/hard_time_limit_state.json',{**current,'created_unix':current['created_unix']-10})
    with pytest.raises(RuntimeError,match='limiter'):
        run.original_idle(manifest,query_gpu=False)


def test_original_and_followup_locks_are_held_together_and_released(fixture,monkeypatch):
    directory,_ = fixture
    (directory/'cloud').mkdir()
    monkeypatch.setattr(run,'BASE',directory)
    monkeypatch.setattr(run,'FOLLOW',directory/'followup')
    with run.execution_locks():
        with pytest.raises(RuntimeError,match='holds its lock'):
            with run.execution_locks():pass
    with run.execution_locks():pass


def test_guard_bound_worker_passes_identity_check_before_refusing_an_absent_matrix(fixture):
    directory,_ = fixture
    attempt = directory/'worker_identity'
    command = [sys.executable,str(run.ENTRY),'--worker','--manifest',str(run.FOLLOW/'preparation/no_such_matrix.json'),
               '--attempt',str(attempt),'--destination',str(attempt),'--mode','probe','--candidate','A']
    process,_ = guard.start(attempt,command,LIMIT,10,checkpoint_grace=2,termination_grace=.2)
    try:
        assert process.wait(timeout=12) == 1
        log = (attempt/'worker.log').read_text()
        assert 'no_such_matrix.json' in log and 'FileNotFoundError' in log
        assert 'Worker is not bound' not in log and 'Worker arguments differ' not in log
        assert read_json(attempt/'guard_state.json')['reason'] == 'worker_failed'
    finally:
        run.stop_guard(process,attempt)


def test_scheduler_cleans_the_exact_worker_group_if_its_guard_was_killed(fixture):
    directory,_ = fixture
    attempt = directory/'killed_guard'
    program = ("import pathlib,json,os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
               "pathlib.Path(__import__('sys').argv[1]).write_text(json.dumps({'pid':os.getpid()})); time.sleep(60)")
    process,_ = guard.start(attempt,[sys.executable,'-c',program,str(directory/'worker.json')],LIMIT,10,
                            checkpoint_grace=2,termination_grace=.2)
    try:
        until(lambda:(directory/'worker.json').exists())
        state = read_json(attempt/'guard_state.json')
        assert guard.group_members(state['worker'])
        process.kill();process.wait(timeout=3)
        run.stop_guard(process,attempt)
        until(lambda:not guard.group_members(state['worker']))
    finally:
        run.stop_guard(process,attempt)
