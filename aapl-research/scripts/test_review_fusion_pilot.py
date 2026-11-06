"""Development allocation tests with fixed synthetic forecasts, never market evidence."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fusion.next.common import ROOT,digest,identity,read_json,write_json
from fusion.next.evaluation import point_scores
from fusion.next.recipe import for_fit
from fusion.next.splits import fold
from scripts import review_fusion_pilot as review


def vectors():
    dates = np.repeat(np.concatenate([pd.bdate_range(start,periods=20).strftime('%Y-%m-%d').to_numpy()
                                     for start in ('2025-01-02','2025-07-01','2026-01-02')]),3)
    quarters = np.repeat(np.asarray(review.QUARTERS),60)
    segments = np.tile(review.REVIEW_POLICY['segments'],60)
    target = np.where(np.arange(len(dates))%2,1.,-1.)*.001
    return target,quarters,dates,segments


def test_stable_gain_selects_only_one_fixed_recipe_and_discloses_seed_counts():
    y,q,dates,times = vectors()
    result = review.evaluate(y,q,dates,times,{'B':y*.5,'A':y*.5,'C':-y*.2},y*.1,y*.2)
    assert result['selected_for_possible_expansion'] == 'A'
    assert result['candidates']['A']['eligible_for_seed_expansion']
    assert not result['candidates']['C']['eligible_for_seed_expansion']
    assert all(r['positive_after_top_days_removed'] for r in result['candidates']['A']['paired_robustness'].values())
    assert result['training_started'] is False and result['original_research_gates_replaced'] is False
    assert 'three-seed mean' in result['seed_count_disclosure']


def test_gain_from_only_five_extreme_days_is_reported_but_does_not_trigger_expansion():
    y,q,dates,times = vectors()
    top = np.isin(dates,np.unique(dates)[:5])
    y[top] *= 50
    prediction = np.where(top,y*.5,-y*.01)
    result = review.evaluate(y,q,dates,times,{'A':prediction},np.zeros_like(y),np.zeros_like(y))
    row = result['candidates']['A']
    assert row['pooled']['mse_improvement_vs_zero'] > 0
    paired = row['paired_robustness']['zero']
    assert len(paired['removed_top_positive_days']) == 5
    assert paired['mean_loss_reduction_after_top_days_removed'] < 0
    assert not row['eligible_for_seed_expansion']
    assert result['selected_for_possible_expansion'] is None


def test_gain_in_one_fixed_segment_does_not_become_an_all_day_claim():
    y,q,dates,times = vectors()
    prediction = np.where(times==review.REVIEW_POLICY['segments'][0],.5*y,-.01*y)
    result = review.evaluate(y,q,dates,times,{'A':prediction},np.zeros_like(y),np.zeros_like(y))
    row = result['candidates']['A']
    assert row['pooled']['mse_improvement_vs_zero'] > 0
    assert row['exploratory_checks']['at_least_two_positive_quarters']
    assert not row['exploratory_checks']['gain_in_at_least_two_time_segments_for_each_control']
    assert not row['eligible_for_seed_expansion']


def test_pilot_must_beat_the_actual_incumbent_and_cover_three_quarters():
    y,q,dates,times = vectors()
    result = review.evaluate(y,q,dates,times,{'A':.5*y},.1*y,.8*y)
    assert not result['candidates']['A']['exploratory_checks']['pooled_beats_original_selected_ensemble']
    with pytest.raises(ValueError,match='three validation'):
        keep = q != '2026Q1'
        review.evaluate(y[keep],q[keep],dates[keep],times[keep],{'A':.5*y[keep]},.1*y[keep],.2*y[keep])
    with pytest.raises(ValueError,match='Missing predictions'):
        prediction = .5*y;prediction[0] = np.nan
        review.evaluate(y,q,dates,times,{'A':prediction},.1*y,.2*y)


def test_fixed_time_bins_include_boundaries_and_equal_gain_day_ties_are_deterministic():
    stamps = pd.to_datetime(['2026-01-02T15:29:00Z','2026-01-02T15:30:00Z',
                             '2026-01-02T19:29:00Z','2026-01-02T19:30:00Z'],utc=True)
    assert list(review.time_segments(stamps)) == ['opening_0931_1029','midday_1030_1429',
                                                 'midday_1030_1429','closing_1430_1548']
    y,q,dates,times = vectors()
    paired = review.robustness(y,.5*y,np.zeros_like(y),dates,times)
    assert [row['date'] for row in paired['removed_top_positive_days']] == list(np.unique(dates)[:5])


def test_factorial_contrasts_use_only_the_matched_single_seed_cells():
    y,_,_,_ = vectors()
    report = review.factorial(y,{'A_foundation_raw':.7*y,'B_no_foundation_scaled':.2*y,
                                'C_no_foundation_raw':.4*y},.5*y)
    assert report['foundation_on_minus_off_under_scaled'] < 0
    assert report['foundation_on_minus_off_under_raw'] < 0
    assert report['raw_minus_scaled_with_foundation'] < 0
    assert report['raw_minus_scaled_without_foundation'] < 0
    assert report['parameter_count_and_initialization_differ_when_a_branch_is_removed']


def test_full_recipe_reader_rejects_omissions_hash_changes_scores_and_partial_updates():
    follow = ROOT/'outputs/fusion_followup'
    follow.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=follow,prefix='pilot_review_fixture_') as temporary:
        directory = Path(temporary)
        dates = np.array(['2024-12-27','2024-12-30','2024-12-31','2025-01-02','2025-01-03'])
        target = np.arange(1,6)*.0001
        panel = SimpleNamespace(dates=dates,origins={'AAPL':np.array([3,4])},targets={'AAPL':target})
        config = {'total_updates':10,'updates_by_fit_end':{'2024-12-30':8}}
        f = fold(dates,'2025Q1');effective = for_fit(config,f['fit_end'])
        path = directory/'2025Q1/seed_2026/updates_8'
        path.mkdir(parents=True)
        rows = np.array([3,4]);prediction = target[rows]*.5
        np.savez_compressed(path/'prediction.npz',rows=rows,prediction=prediction)
        manifest = {'source_identity':'synthetic_engine','folds':{'2025Q1':{'score_origins':2}}}
        original = {'details':{'config':effective,'fold':f,'seed':2026,'updates':8,
                    'source':'synthetic_engine','evaluation_precision':'float32',
                    'numerical_policy_identity':review.NUMERICAL},
                    'scores':point_scores(target[rows],prediction),'prediction_sha256':digest(path/'prediction.npz')}
        write_json(path/'result.json',original)
        origins,values,_ = review.full_prediction(directory,config,'2025Q1',manifest,panel,{})
        np.testing.assert_array_equal(origins,rows);np.testing.assert_array_equal(values,prediction)
        for bad in ('score','updates','hash'):
            import copy
            altered = copy.deepcopy(original)
            if bad == 'score':altered['scores']['mse'] *= .5
            elif bad == 'updates':altered['details']['updates'] = 7
            else:altered['prediction_sha256'] = 'wrong'
            write_json(path/'result.json',altered)
            with pytest.raises(ValueError):
                review.full_prediction(directory,config,'2025Q1',manifest,panel,{})
        original['prediction_sha256'] = 'unused'
        np.savez_compressed(path/'prediction.npz',rows=rows[:1],prediction=prediction[:1])
        original['prediction_sha256'] = digest(path/'prediction.npz')
        original['scores'] = point_scores(target[rows[:1]],prediction[:1])
        write_json(path/'result.json',original)
        with pytest.raises(ValueError,match='omits eligible'):
            review.full_prediction(directory,config,'2025Q1',manifest,panel,{})


def test_complete_pilot_review_recomputes_all_nine_fits_and_refuses_a_duplicate(monkeypatch):
    from fusion.next.panel import Panel
    from scripts.run_fusion_followup import ENTRY
    follow = ROOT/'outputs/fusion_followup'
    follow.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=follow,prefix='full_pilot_review_fixture_') as temporary:
        base = Path(temporary)/'original';matrix = 'synthetic_matrix'
        fixture_follow = Path(temporary)/'followup'
        run = fixture_follow/'runs'/matrix;pilot = run/'pilot'
        dates = np.array(['2024-12-27','2024-12-30','2024-12-31','2025-01-02',
                          '2025-06-26','2025-06-27','2025-06-30','2025-07-01',
                          '2025-12-29','2025-12-30','2025-12-31','2026-01-02'])
        stamps = pd.to_datetime([d+'T15:00:00Z' for d in dates],utc=True)
        target = np.linspace(.0001,.0012,len(dates));origins = np.array([3,7,11])
        panel = SimpleNamespace(dates=dates,stamps=stamps,origins={'AAPL':origins},targets={'AAPL':target})
        reference = {'candidate_id':'reference','total_updates':6,
                     'updates_by_fit_end':{'2024-12-30':2,'2025-06-27':4,'2025-12-30':6}}
        candidates = [{'name':name,'config':{**reference,'candidate_id':name}} for name in
                      ('A_foundation_raw','B_no_foundation_scaled','C_no_foundation_raw')]
        incumbent_path = 'seed_verification/original/selected/predictions.npz'
        manifest = {'identity':matrix,'source_identity':'synthetic_engine','reference':reference,
                    'candidates':candidates,'original_selected_predictions':incumbent_path,
                    'original_inputs':{incumbent_path:{}},'folds':{q:{'score_origins':1} for q in review.QUARTERS}}
        def save_fit(directory,config,quarter,rows,prediction):
            f = fold(dates,quarter);effective = for_fit(config,f['fit_end'])
            directory = directory/quarter/'seed_2026'/('updates_'+str(effective['total_updates']))
            directory.mkdir(parents=True)
            np.savez_compressed(directory/'prediction.npz',rows=rows,prediction=prediction)
            report = {'details':{'config':effective,'fold':f,'seed':2026,'updates':effective['total_updates'],
                       'source':'synthetic_engine','evaluation_precision':'float32','numerical_policy_identity':review.NUMERICAL},
                       'prediction_sha256':digest(directory/'prediction.npz'),'scores':point_scores(target[rows],prediction)}
            write_json(directory/'result.json',report)
            return report
        completed = []
        for q,i in zip(review.QUARTERS,origins):
            rows = np.array([i])
            save_fit(base/'trials/fusion/reference',reference,q,rows,target[rows]*.1)
            for candidate in candidates:
                report = save_fit(pilot/'trials/pilot'/candidate['config']['candidate_id'],
                                  candidate['config'],q,rows,target[rows]*.5)
                completed.append({'candidate':candidate['name'],'quarter':q,'seed':2026,
                                  'scores':report['scores'],'prediction_sha256':report['prediction_sha256']})
        path = base/incumbent_path;path.parent.mkdir(parents=True)
        seeds = np.stack([target[origins]*.2]*3)
        np.savez_compressed(path,rows=origins,quarters=np.array(review.QUARTERS),target=target[origins],
                            timestamps_ns=stamps[origins].as_unit('ns').asi8,predictions=seeds,ensemble=seeds.mean(0))
        write_json(base/'system_freeze.json',{'fixed_strongest_single':{'mse':1e-7},'fixed_strongest_simple':{'mse':2e-7}})
        completion = {'manifest_identity':matrix,'stage':'pilot_complete_review_required','expected_fits':9,
                      'total_coverage_equivalent_epochs':2,'Q2_Q3_prediction_arrays_loaded':False,
                      'extension_or_final_fit_started':False,'completed_fits':completed}
        write_json(pilot/'pilot_result.json',completion)
        attempt = run/'attempts/pilot_attempt_01'
        request = {'requested_at_unix':0.,'stage_seconds':10}
        request['identity'] = identity(request);write_json(attempt/'guard_request.json',request)
        write_json(attempt/'guard_state.json',{'request_identity':request['identity'],'stage':'completed',
                   'worker_returncode':0,'worker':None,'updated_at_utc':'2026-11-08T01:01:00+00:00'})
        admission = {'manifest_identity':matrix,'mode':'pilot','entry_sha256':digest(ENTRY)}
        admission['identity'] = identity(admission);write_json(attempt/'launch_admission.json',admission)
        monkeypatch.setattr(review,'check_frozen',lambda path:manifest)
        monkeypatch.setattr(review,'BASE',base);monkeypatch.setattr(review,'FOLLOW',fixture_follow)
        monkeypatch.setattr(Panel,'load',classmethod(lambda cls:panel))
        destination,result = review.review(Path(temporary)/'synthetic_manifest.json')
        assert result['full_pilot_fits'] == 9 and result['selected_for_possible_expansion'] is None
        assert result['training_started'] is False and result['Q2_Q3_prediction_arrays_loaded'] is False
        assert len(result['prediction_provenance_sha256']) == 26
        assert (destination/'report.md').exists()
        completion['completed_fits'][0] = completion['completed_fits'][1]
        write_json(pilot/'pilot_result.json',completion)
        with pytest.raises(ValueError,match='omitted or duplicated'):
            review.review(Path(temporary)/'synthetic_manifest.json')
