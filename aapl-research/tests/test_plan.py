"""Protect the revised dates, data firewall and single-device time accounting."""
from types import SimpleNamespace
import json

import numpy as np
import pytest

from aapl.common import ROOT
from aapl.data import development_splits, read_source, sessions
from aapl import accounting, sealed


def test_fixed_development_calendar_and_purges():
    dates = np.asarray(list(sessions('2021-01-04', '2026-06-30')))
    split = development_splits(SimpleNamespace(dates=dates))
    assert [i['quarter'] for i in split['inner']] == ['2024Q4', '2025Q2', '2025Q4']
    for inner in split['inner']:
        assert max(inner['fit_dates']) < min(inner['purge_dates']) < min(inner['validation_dates'])
        assert len(inner['purge_dates']) == 1
        assert set(inner['fit_dates']).isdisjoint(inner['validation_dates'])
    d = split['confirmation']
    assert max(d['fit_dates']) < min(d['calibration_dates']) < min(d['evaluation_dates'])
    assert len(d['calibration_dates']) == 20
    assert set(d['fit_dates']).isdisjoint(d['calibration_dates'] + d['evaluation_dates'] + d['purge_dates'])
    assert max(split['final_fit_dates']) == '2026-05-29'
    assert len(split['calibration_dates']) == 20
    assert all(x.startswith('2026-06') for x in split['calibration_dates'])
    assert len(split['holdout_dates_calendar_only']) == 64


def test_generic_loader_rejects_sealed_path_before_reading(monkeypatch):
    events = []
    monkeypatch.setattr(sealed, 'event', lambda *args: events.append(args))
    with pytest.raises(PermissionError, match='sealed'):
        read_source(ROOT / 'holdout_sealed/AAPL_sip_1min_raw.csv')
    assert events[0][0] == 'denied'


def test_gpu_busy_time_counts_union_of_parallel_jobs(tmp_path, monkeypatch):
    fake = tmp_path / 'formal'
    fake.mkdir()
    now = 1000.0
    (fake / 'run.json').write_text(json.dumps({'started_epoch': 100.0}))
    for number, (start, end, gpu) in enumerate([(100, 300, True), (150, 350, True), (400, 500, True), (300, 900, False)]):
        directory = fake / str(number)
        directory.mkdir()
        (directory / 'execution_charge.json').write_text(json.dumps({'attempts': [dict(start_epoch=start, end_epoch=end, gpu=gpu)]}))
    monkeypatch.setattr(accounting, 'FORMAL', fake)
    monkeypatch.setattr(accounting.time, 'time', lambda: now)
    result = accounting.usage()
    assert result['physical_gpu_busy_hours'] * 3600 == 350
    assert result['sum_gpu_job_hours'] * 3600 == 500
