"""Synthetic forward evidence exercises acceptance; it is never live research."""
from copy import deepcopy
from functools import cache
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from finance import completion, observation, prospective, forward_runtime
from finance.io import ROOT, read_json, sha, write_json


CACHED_SESSION_GRID = cache(observation.session_origins)


@pytest.fixture
def forward_evidence(monkeypatch, tmp_path, request):
    config = read_json(ROOT / 'configs/rv30.json')
    settings = read_json(ROOT / 'configs/rv30_prospective.json')
    settings['bootstrap_repeats'] = 499
    directory = tmp_path / 'artifacts/prospective/synthetic'
    models = {key: {'source_result_sha256': key, 'model': key.split('/')[1],
                   'panel': key.split('/')[0]} for key in settings['candidate_keys'] + settings['reference_keys']}
    lock = {'signature': 'synthetic-only', 'payload': {'settings': settings, 'models': models,
            'created_utc': '2026-01-01T00:00:00Z', 'run_directory': 'artifacts/development',
            'base_signature': 'historical', 'historical_prerequisites': {'preserved': True,
                'verifications': {'1': {'path': 'history_1.json'}, '2': {'path': 'history_2.json'}}}}}
    for module in (completion, prospective, forward_runtime):
        monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(prospective, 'read_plan', lambda *a, **kw: lock)
    monkeypatch.setattr(prospective, 'observation_directory', lambda *a: directory)
    original_grid = CACHED_SESSION_GRID
    # Keep sixty actual session dates but shrink the intraday grid in this
    # isolated fixture. Production grid correctness has its own calendar tests.
    def small_grid(day):
        return original_grid(day)[:2]
    monkeypatch.setattr(observation, 'session_origins', small_grid)
    monkeypatch.setattr(completion, 'session_origins', small_grid)
    monkeypatch.setattr(prospective, 'session_origins', small_grid)
    dates = [day.strftime('%Y-%m-%d') for day in pd.bdate_range('2026-02-02', '2026-06-01')
             if len(original_grid(day.strftime('%Y-%m-%d')))][:60]
    journal = observation.Journal(directory, lock['signature'])
    admitted_at = pd.Timestamp('2026-01-30T14:31:02Z')
    bar = {'t': '2026-01-30T14:30:00Z', 'o': 100., 'h': 101., 'l': 99., 'c': 100., 'v': 1}
    event = journal.append('bars', {'http_status': 200, 'feed': 'sip', 'adjustment': 'raw', 'purpose': 'admission',
        'end': bar['t'], 'received_at': '2026-01-30T14:31:01Z',
        'response': {'bars': {asset: [bar] for asset in observation.ASSETS}}}, admitted_at.isoformat())
    write_json(directory / 'admission.json', {'passed': True, 'http_status': 200, 'status': 'timely_sip_admitted',
        'plan_signature': lock['signature'], 'last_event_hash': event['hash'], 'checked_utc': admitted_at.isoformat(),
        'regular_session': True, 'recent_all_three_assets': True})
    forecasts = {}
    for number, day in enumerate(dates):
        origins = small_grid(day)
        minute_times = pd.date_range(origins[0] - pd.Timedelta(minutes=1), origins[-1] + pd.Timedelta(minutes=29), freq='min')
        closes = 100 * np.exp(np.arange(len(minute_times)) * (0.0001 + number * 0.0000008))
        for origin in origins:
            received = origin - pd.Timedelta(seconds=10)
            data = journal.append('bars', {'http_status': 200, 'feed': 'sip', 'adjustment': 'raw', 'purpose': 'live',
                'received_at': received.isoformat(), 'end': (origin - pd.Timedelta(minutes=2)).isoformat(),
                'response': {'bars': {}}}, received.isoformat())
            target = 30 * (0.0001 + number * 0.0000008) ** 2
            predictions = {}
            for key in models:
                factor=(getattr(request, 'param', 1.05)+.01*np.sin(number)) if key.endswith('ridge') else (1.6+.02*np.cos(number))
                mean = target * factor
                predictions[key] = {'raw': mean, 'rv_mean': mean, 'source_result_sha256': key,
                    'quantiles': {str(q): mean * (0.5 + q) for q in config['calibration']['quantiles']}}
            forecasts[origin.isoformat()] = deepcopy(predictions)
            p = journal.append('prediction', {'origin': origin.isoformat(), 'status': 'issued',
                'received_at': received.isoformat(), 'issued_at': (origin - pd.Timedelta(seconds=3)).isoformat(),
                'data_cutoff_at': (origin - pd.Timedelta(minutes=1)).isoformat(), 'data_event_hash': data['hash'],
                'predictions': predictions}, (origin - pd.Timedelta(seconds=2)).isoformat())
            journal.append('issuance_commit', {'prediction_hash': p['hash']}, (origin - pd.Timedelta(seconds=1)).isoformat())
        bars = [{'t': t.isoformat(), 'o': float(c), 'h': float(c), 'l': float(c), 'c': float(c), 'v': 1}
                for t, c in zip(minute_times, closes)]
        received = minute_times[-1] + pd.Timedelta(minutes=2)
        journal.append('bars', {'http_status': 200, 'feed': 'sip', 'adjustment': 'raw', 'purpose': 'live',
            'received_at': received.isoformat(), 'end': minute_times[-1].isoformat(),
            'response': {'bars': {'AAPL': bars}}}, received.isoformat())
    class Predictor:
        def __init__(self, *args):
            pass
        def predict(self, store, origin):
            return forecasts[origin.isoformat()]
    monkeypatch.setattr(prospective, 'Predictor', Predictor)
    assert prospective.verify_observation(config, 1)['passed']
    assert prospective.verify_observation(config, 2)['passed']
    return config, lock, directory, tmp_path / 'artifacts/development'


def test_rebuilds_the_exact_earliest_grid_scores_and_model_preference(forward_evidence):
    config, lock, directory, run = forward_evidence
    evidence = completion.prospective_evidence(config, run)
    assert evidence['sessions'] == 60 and evidence['origins'] == 120
    assert evidence['qualified_candidates'] == ['aapl/ridge', 'market/ridge']
    selected = completion.selected_model(lock, evidence)
    assert selected['panel'] == 'aapl' and selected['prospective_qualified']
    assert selected['outcome'] == 'completed_prospective_validation'
    index = completion.prospective_artifact_index(lock, evidence)
    assert index['event_count'] == len(observation.Journal(directory, lock['signature']).events)
    assert all(sha(completion.ROOT / p) == digest for p, digest in index['files'].items())


@pytest.mark.parametrize('corruption', ['missing_round', 'false_pass', 'round_identity', 'dates', 'counts',
    'chain', 'statistics', 'preference', 'labels_not_rebuilt', 'admission', 'admission_prefix', 'late_admission'])
def test_final_acceptance_rejects_incomplete_or_altered_evidence(forward_evidence, corruption):
    config, lock, directory, run = forward_evidence
    path = directory / 'verify_2.json'
    if corruption == 'missing_round':
        path.unlink()
    elif corruption.startswith('admission') or corruption == 'late_admission':
        path = directory / 'admission.json'; value = read_json(path)
        if corruption == 'admission': value['http_status'] = 403
        elif corruption == 'admission_prefix': value['last_event_hash'] = 'invented'
        else: value['checked_utc'] = '2026-10-30T14:31:02Z'
        write_json(path, value)
    else:
        value = read_json(path)
        if corruption == 'false_pass': value['passed'] = False
        elif corruption == 'round_identity': value['round'] = 1
        elif corruption == 'dates': value['first_60_complete_dates'] = list(reversed(value['first_60_complete_dates']))
        elif corruption == 'counts': value['origins'] -= 1
        elif corruption == 'chain': value['last_event_hash'] = 'changed'
        elif corruption == 'statistics': value['comparisons']['aapl/ridge']['mean_difference'] -= .01
        elif corruption == 'preference': value['primary_model'] = 'market/ridge'
        elif corruption == 'labels_not_rebuilt': value['all_labels_independently_rebuilt'] = False
        write_json(path, value)
    with pytest.raises((ValueError, FileNotFoundError, AssertionError)):
        completion.prospective_evidence(config, run)


def test_current_empty_run_refuses_without_relabeling_or_overwriting_reports(monkeypatch, tmp_path):
    settings = read_json(ROOT / 'configs/rv30_prospective.json')
    lock = {'signature': 'isolated', 'payload': {'settings': settings, 'run_directory': 'run'}}
    monkeypatch.setattr(completion, 'ROOT', tmp_path)
    monkeypatch.setattr(prospective, 'read_plan', lambda *a: lock)
    monkeypatch.setattr(prospective, 'observation_directory', lambda *a: tmp_path / 'observation')
    paths = ['delivery/model.json', 'delivery/REPORT.md', 'delivery/example_prediction.json']
    for name in paths:
        p = tmp_path / name;p.parent.mkdir(parents=True, exist_ok=True);p.write_text('preserve me')
    before = {name: sha(tmp_path / name) for name in paths}
    result = completion.accept_forward({}, tmp_path / 'run')
    assert not result['passed'] and not result['goal_can_complete']
    assert '60 actual' in result['errors'][0]
    assert completion.finalize_forward({}, tmp_path / 'run')['prospective_complete_sessions'] == 0
    assert before == {name: sha(tmp_path / name) for name in paths}


def test_no_forward_winner_delivers_exact_reference_baseline(forward_evidence):
    config, lock, directory, run = forward_evidence
    evidence = completion.prospective_evidence(config, run)
    evidence.update(qualified_candidates=[], primary_model=None)
    selected = completion.selected_model(lock, evidence)
    assert selected['source_result_sha256'] == lock['payload']['models']['aapl/har']['source_result_sha256']
    assert selected['outcome'] == 'completed_no_prospective_edge'
    assert selected['role'] == 'reference_har_baseline_after_prospective_failure'
    assert not selected['active_predictive_edge_established'] and not selected['prospective_qualified']
    evidence.update(qualified_candidates=['market/ridge'], primary_model='market/ridge')
    assert completion.selected_model(lock, evidence)['panel'] == 'market'
    evidence.update(qualified_candidates=['aapl/ridge', 'market/ridge'], primary_model='market/ridge')
    with pytest.raises(ValueError, match='score ranking'):
        completion.selected_model(lock, evidence)


def test_tampered_received_prefix_is_rejected_even_with_passed_proofs(forward_evidence, monkeypatch):
    config, lock, directory, run = forward_evidence
    original = completion.Journal
    def altered(*args):
        journal = original(*args)
        event = next(e for e in journal.events if e['kind'] == 'prediction')
        event['payload']['data_event_hash'] = 'another-prefix'
        return journal
    monkeypatch.setattr(completion, 'Journal', altered)
    with pytest.raises(ValueError, match='received-data prefix'):
        completion.prospective_evidence(config, run)


@pytest.mark.parametrize('forward_evidence', [1.05, 2.0], indirect=True)
def test_isolated_success_path_preserves_both_possible_delivery_outcomes(forward_evidence, monkeypatch):
    from finance import acceptance, confirmation, evaluation, inference, provenance, verify
    config, lock, directory, run = forward_evidence
    root = completion.ROOT
    evidence = completion.prospective_evidence(config, run)
    selected = completion.selected_model(lock, evidence)
    write_json(root / 'delivery/model.json', selected)
    write_json(run / 'setup.json', {'signature': 'historical', 'config': config, 'search': {}})
    write_json(run / 'run.json', {'finished_groups': 120, 'search_finished': True, 'budget_exhausted': False})
    write_json(run / 'evaluation.json', {'complete_model_panel_results': 12, 'errors': []})
    write_json(root / 'delivery/history.json', {'historical_file_hashes': {}})
    write_json(root / 'configs/rv30_search.json', {})
    write_json(root / 'delivery/results.json', {'engineering_final_acceptance': False})
    required = ['GOAL.md', 'RESEARCH_PROTOCOL.md', 'configs/rv30.json', 'configs/rv30_model_lock.json',
        'configs/rv30_prospective.json', 'delivery/README.md', 'delivery/REPORT.md', 'delivery/PROSPECTIVE_PLAN.md',
        'delivery/prospective_lock.json', 'delivery/rv30_freeze.json', 'delivery/rv30_dependencies.json',
        'delivery/data.json', 'delivery/external_dependencies.json', 'history_1.json', 'history_2.json']
    for name in required:
        p = root / name;p.parent.mkdir(parents=True, exist_ok=True);p.write_text('isolated fixture')
    for name in ('daily_losses.csv', 'confirmation/evaluation.json'):
        p = run / name;p.parent.mkdir(parents=True, exist_ok=True);p.write_text('isolated fixture')
    frozen = {'signature': 'historical', 'effective_source_files': {}, 'postprocessing_lock_files': {}}
    # Only expensive already-covered historical prerequisites are stubbed here;
    # the forward grid, raw labels, two proofs and all statistics are real fixture data.
    monkeypatch.setattr(provenance, 'verify_freeze', lambda *a: frozen)
    monkeypatch.setattr(acceptance, 'search_coverage', lambda *a: {'passed': True, 'failed_trials': []})
    monkeypatch.setattr(acceptance, 'artifact_index', lambda *a: {'job_count': 12812})
    monkeypatch.setattr(evaluation, 'evaluate', lambda *a: read_json(run / 'evaluation.json'))
    monkeypatch.setattr(confirmation, 'confirmation_evidence', lambda *a: lock['payload']['historical_prerequisites'])
    monkeypatch.setattr(verify, 'require_verifications', lambda *a: lock['payload']['historical_prerequisites']['verifications'])
    monkeypatch.setattr(inference, 'infer', lambda *a: {'rv_mean': 1e-5, 'eligible_signal': False,
        'direction_or_profit_prediction': False, 'prospective_qualified': selected['prospective_qualified']})
    monkeypatch.setattr(forward_runtime, 'ACTIVE', SimpleNamespace(action='accept'))
    monkeypatch.setattr(forward_runtime, 'resource_evidence', lambda *a: {'passed': True, 'files': {}})
    original = completion.subprocess.check_output
    def committed(args, **kw):
        if args[:3] == ['git', 'rev-parse', 'HEAD']: return 'fixture-commit'
        return (root / args[-1].split(':', 1)[1]).read_bytes()
    monkeypatch.setattr(completion.subprocess, 'check_output', committed)
    result = completion.accept_forward(config, run)
    assert result['passed'] and result['goal_can_complete']
    manifest = read_json(root / 'delivery/manifest.json')
    assert manifest['final_acceptance'] and manifest['goal_can_complete']
    assert manifest['outcome'] == selected['outcome']
    assert all(sha(root / name) == digest for name, digest in manifest['files'].items())
    index = read_json(root / manifest['prospective_artifact_index'])
    assert index['event_count'] == len(observation.Journal(directory, lock['signature']).events)
