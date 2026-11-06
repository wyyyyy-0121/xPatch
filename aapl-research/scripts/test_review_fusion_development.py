import io
import json
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from fusion.next.common import ROOT, PLAN, digest, identity, write_json
from fusion.next.evaluation import point_scores
from scripts import review_fusion_development as review


def test_snapshot_allowlist_excludes_confirmation_holdout_and_weights():
    assert review.allowed('trials/lora/model/2025Q1/seed_2026/updates_42/prediction.npz')
    assert review.allowed('training/fusion/model/2026Q1/seed_2028.progress.json')
    for path in ('trials/lora/model/2026Q2/seed_2026/updates_42/prediction.npz',
                 'trials/lora/model/2026Q3/seed_2026/updates_42/prediction.npz',
                 'trials/lora/model/2025Q1/seed_2026/updates_42/weights.pt',
                 '../trials/simple/1/2025Q1/result.json', 'confirmation_predictions.npz',
                 'final_test/report.json', '/cloud/bounded_run_state.json'):
        assert not review.allowed(path)


def test_date_validation_precedes_label_lookup_and_rejects_partial_rows():
    dates = np.array(['2025-01-02', '2025-01-03', '2026-07-01'])
    review.validate_rows(np.array([0, 1]), np.array([.1, .2]), dates, '2025Q1')
    with pytest.raises(ValueError, match='leaves'):
        review.validate_rows(np.array([0, 2]), np.array([.1, .2]), dates, '2025Q1')
    for rows, prediction in [(np.array([1, 1]), np.ones(2)),
                             (np.array([0, 1]), np.array([.1, np.nan])),
                             (np.array([0, 1]), np.ones(1))]:
        with pytest.raises(ValueError, match='Invalid'):
            review.validate_rows(rows, prediction, dates, '2025Q1')


def test_forward_shrink_coefficients_do_not_use_current_or_future_labels():
    parts = {q: (np.array([1., -1.]), np.array([2., -2.])) for q in review.QUARTERS}
    original = review.forward_shrink(parts)['quarters']
    assert [row['alpha'] for row in original] == [.5, .5]
    changed = {**parts, '2026Q1': (np.array([100., -100.]), parts['2026Q1'][1])}
    future = review.forward_shrink(changed)['quarters']
    assert [row['alpha'] for row in future] == [.5, .5]
    changed = {**parts, '2025Q3': (np.array([10., -10.]), parts['2025Q3'][1])}
    current = review.forward_shrink(changed)['quarters']
    assert current[0]['alpha'] == .5
    assert current[1]['alpha'] == 1.
    assert original[0]['fit_quarters'] == ['2025Q1']


def test_error_decomposition_matches_actual_mse_change():
    result = review.metrics(np.array([1., -2., 1.]), np.array([.2, -.3, .4]))
    assert result['target_alignment_pct_zero'] - result['forecast_energy_pct_zero'] == pytest.approx(result['improvement_pct'])


def test_archive_rejects_holdout_or_hash_changes_before_writing(tmp_path, monkeypatch):
    binding = {'source_identity':'source', 'plan_sha256':'plan', 'scope_identity':'scope'}
    payload = b'{}'
    name = 'cloud/bounded_run_state.json'
    manifest = {**binding, 'snapshot':{}, 'Q2_Q3_predictions_included':False, 'training_launched':False,
                'files':{name:{'sha256':'wrong', 'bytes':len(payload)}}}
    manifest['identity'] = identity(manifest)
    archive = tmp_path/'bad.tar'
    with tarfile.open(archive, 'w') as stream:
        for filename, data in [(name, payload), ('manifest.json', json.dumps(manifest).encode())]:
            info = tarfile.TarInfo(filename);info.size=len(data)
            stream.addfile(info, io.BytesIO(data))
    monkeypatch.setattr(review, 'check_binding', lambda snapshot,binding:False)
    destination = tmp_path/'unpacked'
    with pytest.raises(ValueError, match='hash differs'):
        review.unpack(archive, destination, binding)
    assert not destination.exists()
    with tarfile.open(archive, 'a') as stream:
        info = tarfile.TarInfo('final_test/report.json');info.size=2
        stream.addfile(info, io.BytesIO(b'{}'))
    with pytest.raises(ValueError, match='Unexpected'):
        review.unpack(archive, destination, binding)
    assert not destination.exists()


def test_actual_report_keeps_partial_seed_results_incomplete(monkeypatch):
    with TemporaryDirectory(dir=ROOT/'tmp', prefix='followup_test_') as temporary:
        base = Path(temporary)
        canonical, follow = base/'canonical', base/'followup'
        snapshot = follow/'snapshots'/'test'
        data = canonical/'data';(data/'AAPL').mkdir(parents=True)
        dates = np.array(['2025-01-02', '2025-01-03', '2025-07-01', '2025-07-02',
                          '2026-01-02', '2026-01-05'])
        target = np.array([1., -1., 2., -2., 1.5, -1.5])
        np.save(data/'dates.npy', dates)
        np.save(data/'timestamps_ns.npy', dates.astype('datetime64[ns]').view('i8'))
        np.save(data/'AAPL/target.npy', target)
        files = {}
        policy = snapshot/'performance/portable_inference_policy.json'
        write_json(policy, {'identity':'numeric'})
        files[str(policy.relative_to(snapshot))] = {'bytes':policy.stat().st_size, 'sha256':digest(policy)}
        for seed, quarters in [(2026, review.QUARTERS), (2027, review.QUARTERS),
                               (2028, review.QUARTERS[:1])]:
            for quarter in quarters:
                index = review.QUARTERS.index(quarter)
                rows = np.arange(2*index, 2*index+2)
                prediction = target[rows]*.5
                directory = snapshot/'trials/fusion/candidate'/quarter/f'seed_{seed}'/'updates_2'
                directory.mkdir(parents=True)
                path = directory/'prediction.npz'
                np.savez(path, rows=rows, prediction=prediction)
                result = {'details':{'config':{'total_updates':2}, 'updates':2, 'seed':seed,
                          'fold':{'quarter':quarter}, 'evaluation_precision':'float32',
                          'numerical_policy_identity':'numeric'}, 'scores':point_scores(target[rows], prediction),
                          'prediction_sha256':digest(path)}
                write_json(directory/'result.json', result)
                for p in directory.iterdir():
                    files[str(p.relative_to(snapshot))] = {'bytes':p.stat().st_size, 'sha256':digest(p)}
        manifest = {'captured_at_utc':'2026-11-06T00:00:00+00:00', 'source_identity':'source',
                    'scope_identity':'scope', 'plan_sha256':digest(PLAN), 'files':files,
                    'purpose':'development only', 'Q2_Q3_predictions_included':False,
                    'training_launched':False}
        manifest['identity'] = identity(manifest)
        write_json(snapshot/'manifest.json', manifest)
        monkeypatch.setattr(review, 'OUT', canonical)
        monkeypatch.setattr(review, 'FOLLOW', follow)
        monkeypatch.setattr(review, 'source_identity', lambda:'source')
        report = review.analyze(snapshot)
        assert set(report['systems']) == {'fusion/candidate/seed_2026', 'fusion/candidate/seed_2027'}
        assert report['incomplete_systems_not_scored_as_complete'] == [
            {'model':'fusion/candidate/seed_2028', 'completed_quarters':['2025Q1']}]
        assert report['common_origins'] == {q:2 for q in review.QUARTERS}
        assert report['Q2_Q3_values_read'] is False
