"""Forward runtime keeps the old budget and fails closed on missing usage."""
from types import SimpleNamespace

import pytest

from finance import forward_runtime as runtime, prospective, observation
from finance.io import read_json, sha, write_json


@pytest.fixture
def meter_setup(monkeypatch, tmp_path):
    config = {'runtime': {'total_cpu_core_hours': 2000, 'total_gpu_hours': 240,
                          'memory_budget_gib':10,'max_artifact_gib':60}}
    lock = {'signature': 'isolated-plan', 'payload': {'base_signature': 'historical'}}
    monkeypatch.setattr(runtime, 'ROOT', tmp_path)
    monkeypatch.setattr(prospective, 'observation_directory', lambda *a: tmp_path / 'observation')
    monkeypatch.setattr(runtime, 'historical_usage', lambda *a: (120 * 3600., 20 * 3600.))
    monkeypatch.setattr(runtime, 'ACTIVE', None)
    monkeypatch.setattr(runtime.os,'sched_getaffinity',lambda pid:{0,1,2,3})
    return config, lock, tmp_path


def test_explicit_idle_sleep_releases_allocation_but_all_work_is_charged(meter_setup, monkeypatch):
    config, lock, root = meter_setup
    clock = [10.]
    monkeypatch.setattr(runtime.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(runtime.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    meter = runtime.Meter(config, lock, 'observe')
    clock[0] += 2
    meter.sleep(3600)
    clock[0] += 3
    meter.finish(True)
    rows, files = runtime.resource_records(lock)
    assert rows[0]['charged_cpu_seconds'] == 20
    assert rows[0]['end']['gpu_wall_seconds'] == 0
    assert len(files) == 2


def test_interrupted_action_retains_its_entire_reservation(meter_setup):
    config, lock, root = meter_setup
    meter = runtime.Meter(config, lock, 'observe')
    rows, files = runtime.resource_records(lock)
    assert rows[0]['end'] is None
    assert rows[0]['charged_cpu_seconds'] == 4 * runtime.ACTION_LIMITS['observe']
    with pytest.raises(ValueError, match='Unresolved'):
        runtime.resource_evidence(config, lock, observation.Journal(root / 'observation', lock['signature']), {})


def test_original_budget_is_checked_before_starting_an_action(meter_setup):
    config, lock, root = meter_setup
    config['runtime']['total_cpu_core_hours'] = 121
    with pytest.raises(ValueError, match='Insufficient original budget'):
        runtime.Meter(config, lock, 'observe')
    assert not list(root.rglob('start.json'))


def test_changed_or_negative_resource_receipt_is_rejected(meter_setup):
    config, lock, root = meter_setup
    meter = runtime.Meter(config, lock, 'verify_1');meter.finish(True)
    path = meter.path / 'end.json';end = read_json(path);end['allocated_cpu_seconds'] = -1
    write_json(path, end)
    with pytest.raises(ValueError, match='receipt differs'):
        runtime.resource_records(lock)


def test_both_full_replays_and_every_observation_event_need_metered_coverage(meter_setup, monkeypatch):
    config, lock, root = meter_setup
    journal = observation.Journal(root / 'observation', lock['signature'])
    meter = runtime.Meter(config, lock, 'observe')
    journal.append('prediction', {'origin': 'synthetic'}, '2026-11-04T00:00:00Z')
    meter.finish(True)
    proofs = {}
    for number in (1, 2):
        meter = runtime.Meter(config, lock, f'verify_{number}')
        p = root / 'observation' / f'verify_{number}.json';write_json(p, {'fixture_round': number})
        proofs[str(number)] = {'path': str(p.relative_to(root)), 'sha256': sha(p)}
        meter.finish(True)
    evidence = runtime.resource_evidence(config, lock, journal, proofs)
    assert evidence['passed'] and evidence['metered_prediction_and_commit_events'] == 1
    journal.append('prediction', {'origin': 'unmetered'}, '2026-11-04T00:00:01Z')
    with pytest.raises(ValueError, match='metered receipts'):
        runtime.resource_evidence(config, lock, journal, proofs)


def test_cached_verification_is_not_a_new_full_replay(meter_setup):
    config, lock, root = meter_setup
    p = root / 'observation/verify_1.json';write_json(p, {'passed': True})
    meter = runtime.Meter(config, lock, 'verify_1');meter.finish(True)
    assert not read_json(meter.path / 'start.json')['full_replay_required']
    with pytest.raises(ValueError, match='metered receipts'):
        runtime.resource_evidence(config, lock, observation.Journal(root / 'observation', lock['signature']),
                                  {'1': {'path': str(p.relative_to(root)), 'sha256': sha(p)}, '2': {}})


def test_recovery_preserves_unknown_cost_and_never_replaces_a_live_process(meter_setup):
    config,lock,root=meter_setup
    meter=runtime.Meter(config,lock,'observe')
    with pytest.raises(RuntimeError,match='still alive'):runtime.recover_interrupted(lock)
    path=meter.path/'start.json';start=read_json(path);start['kernel_start_ticks']='old-process-identity'
    write_json(path,start)
    runtime.recover_interrupted(lock)
    rows,_=runtime.resource_records(lock)
    assert rows[0]['end']['interrupted']
    assert rows[0]['charged_cpu_seconds']==start['reserved_cpu_seconds']
    assert not rows[0]['end']['success']
