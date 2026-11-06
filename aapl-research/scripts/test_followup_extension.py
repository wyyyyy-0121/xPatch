"""Synthetic paired extension evidence and budget checks; never GPU admission."""
import copy
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
from scripts import review_fusion_pilot as pilot
from scripts import review_fusion_extension as extension
from scripts import run_fusion_followup as run
from scripts.test_run_fusion_followup import NOW,SIGNATURE,synthetic_probe


NAMES = ('A_foundation_raw','B_no_foundation_scaled','C_no_foundation_raw')


def seal(value):
    value = copy.deepcopy(value);value.pop('identity',None);value['identity'] = identity(value)
    return value


@pytest.fixture
def evidence(monkeypatch):
    from fusion.next.panel import Panel
    directory = ROOT/'outputs/fusion_followup';directory.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=directory,prefix='extension_fixture_') as temporary:
        root = Path(temporary);base = root/'original';follow = root/'followup'
        matrix = 'synthetic_matrix';location = follow/'runs'/matrix
        records = []
        for start in ('2025-01-02','2025-07-01','2026-01-02'):
            quarter_start = pd.Period(start,freq='Q').start_time
            for day in pd.bdate_range(end=quarter_start-pd.Timedelta(days=1),periods=2):
                records.append(day.strftime('%Y-%m-%dT15:00:00Z'))
            for day in pd.bdate_range(start,periods=8):
                for clock in ('14:45','17:00','20:00'):
                    records.append(day.strftime('%Y-%m-%dT')+clock+':00Z')
        stamps = pd.to_datetime(sorted(records),utc=True)
        dates = stamps.strftime('%Y-%m-%d').to_numpy(dtype=str)
        quarters = stamps.tz_localize(None).to_period('Q').astype(str).to_numpy(dtype=str)
        origins = np.flatnonzero(np.isin(quarters,pilot.QUARTERS))
        target = np.where(np.arange(len(dates))%2,1.,-1.)*.001
        panel = SimpleNamespace(dates=dates,stamps=stamps,origins={'AAPL':origins},targets={'AAPL':target})
        reference = {'candidate_id':'reference','total_updates':6,
                     'updates_by_fit_end':{fold(dates,q)['fit_end']:2*(i+1) for i,q in enumerate(pilot.QUARTERS)}}
        candidates = [{'name':name,'config':{**reference,'candidate_id':name,'lookback':240,'batch_size':256}}
                      for name in NAMES]
        manifest = {'identity':matrix,'source_identity':'synthetic_engine','reference':reference,
                    'candidates':candidates,'review_policy':pilot.REVIEW_POLICY,
                    'limit':{'total_wall_hours':100,'deadline_utc':run.DEADLINE},
                    'original_selected_predictions':'seed_verification/selected/predictions.npz',
                    'original_frozen_predictions':'seed_verification/frozen/predictions.npz',
                    'original_inputs':{},'folds':{q:{'fit_end':fold(dates,q)['fit_end'],
                        'total_updates':2*(i+1),'score_origins':int((quarters[origins]==q).sum())}
                        for i,q in enumerate(pilot.QUARTERS)}}
        def save_fit(destination,config,q,seed,factor):
            f = fold(dates,q);effective = for_fit(config,f['fit_end'])
            path = destination/q/('seed_'+str(seed))/('updates_'+str(effective['total_updates']))
            path.mkdir(parents=True,exist_ok=True)
            rows = origins[quarters[origins]==q];p = target[rows]*factor
            np.savez_compressed(path/'prediction.npz',rows=rows,prediction=p)
            result = {'details':{'config':effective,'fold':f,'seed':seed,'updates':effective['total_updates'],
                       'source':manifest['source_identity'],'evaluation_precision':'float32',
                       'numerical_policy_identity':pilot.NUMERICAL},
                       'prediction_sha256':digest(path/'prediction.npz'),'scores':point_scores(target[rows],p)}
            write_json(path/'result.json',result)
            return result
        fits = []
        for q in pilot.QUARTERS:
            save_fit(base/'trials/fusion/reference',reference,q,2026,.1)
            for candidate,factor in zip(candidates,(.5,.4,.3)):
                result = save_fit(location/'pilot/trials/pilot'/candidate['name'],candidate['config'],q,2026,factor)
                fits.append({'candidate':candidate['name'],'quarter':q,'seed':2026,
                             'scores':result['scores'],'prediction_sha256':result['prediction_sha256']})
        for field,factor in (('original_selected_predictions',.2),('original_frozen_predictions',.1)):
            path = base/manifest[field];path.parent.mkdir(parents=True)
            seeds = np.stack([target[origins]*factor]*3)
            np.savez_compressed(path,rows=origins,quarters=quarters[origins],target=target[origins],
                                timestamps_ns=stamps[origins].as_unit('ns').asi8,predictions=seeds,ensemble=seeds.mean(0))
            manifest['original_inputs'][manifest[field]] = {'sha256':digest(path)}
        write_json(base/'system_freeze.json',{'fixed_strongest_single':point_scores(target[origins],target[origins]*.1),
                                            'fixed_strongest_simple':point_scores(target[origins],target[origins]*.15)})
        write_json(location/'pilot/pilot_result.json',{'manifest_identity':matrix,'stage':'pilot_complete_review_required',
                   'expected_fits':9,'total_coverage_equivalent_epochs':2,'Q2_Q3_prediction_arrays_loaded':False,
                   'extension_or_final_fit_started':False,'completed_fits':fits})
        def complete_guard(mode,budget=None):
            attempt = location/'attempts'/({'pilot':'pilot_attempt_01','extend':'extension_attempt_01'}[mode])
            request = seal({'requested_at_unix':NOW.timestamp()-10,'stage_seconds':run.PILOT_SECONDS})
            write_json(attempt/'guard_request.json',request)
            write_json(attempt/'guard_state.json',{'request_identity':request['identity'],'stage':'completed',
                       'worker_returncode':0,'worker':None,'updated_at_utc':NOW.isoformat()})
            write_json(attempt/'launch_admission.json',seal({'manifest_identity':matrix,'mode':mode,
                       'entry_sha256':digest(run.ENTRY),'budget':budget}))
        complete_guard('pilot')
        monkeypatch.setattr(pilot,'BASE',base);monkeypatch.setattr(extension,'BASE',base)
        monkeypatch.setattr(pilot,'FOLLOW',follow);monkeypatch.setattr(extension,'FOLLOW',follow)
        monkeypatch.setattr(pilot,'check_frozen',lambda path:manifest)
        monkeypatch.setattr(extension,'check_frozen',lambda path:manifest)
        monkeypatch.setattr(Panel,'load',classmethod(lambda cls:panel))
        yield SimpleNamespace(root=root,base=base,follow=follow,run=location,manifest=manifest,
                manifest_path=root/'synthetic_manifest.json',save_fit=save_fit,complete_guard=complete_guard,
                panel=panel,origins=origins,quarters=quarters[origins],target=target[origins])


@pytest.mark.parametrize('chosen,pair,count',[(NAMES[0],NAMES[2],12),(NAMES[1],'original_frozen',6),(NAMES[2],NAMES[0],12)])
def test_fixed_selected_recipe_and_pair_are_the_only_new_fits(evidence,chosen,pair,count):
    _,report = pilot.review(evidence.manifest_path)
    report['selected_for_possible_expansion'] = chosen
    plan = pilot.expansion_members(evidence.manifest,report)
    assert plan['paired_control']==pair and plan['new_full_fits']==count
    assert plan['new_seeds']==[2027,2028] and plan['pilot_seed_reused']==2026
    assert plan['train_candidates'][0]==chosen and plan['paired_control_can_replace_selected_candidate'] is False
    report['candidates'][chosen]['eligible_for_seed_expansion'] = False
    with pytest.raises(ValueError,match='eligible'):pilot.expansion_members(evidence.manifest,report)
    report['selected_for_possible_expansion'] = None
    with pytest.raises(RuntimeError,match='No fixed'):pilot.expansion_members(evidence.manifest,report)


def test_saved_decision_is_recomputed_and_pointer_or_prediction_tampering_is_refused(evidence):
    path,report = pilot.review(evidence.manifest_path)
    saved,plan = pilot.checked_review(evidence.manifest_path,evidence.manifest)
    assert saved['decision_identity']==report['decision_identity'] and plan['selected_candidate']==NAMES[0]
    # CPU recomputation adds a fresh timestamp but leaves the decision invariant.
    assert pilot.review(evidence.manifest_path,publish=False)[1]['decision_identity']==report['decision_identity']
    altered = copy.deepcopy(report);altered['selected_for_possible_expansion']=NAMES[1]
    write_json(path/'report.json',seal(altered))
    with pytest.raises(ValueError,match='pointer or stable'):pilot.checked_review(evidence.manifest_path,evidence.manifest)
    write_json(path/'report.json',report)
    source = next((evidence.run/'pilot/trials/pilot').rglob('prediction.npz'))
    source.write_bytes(b'changed prediction bytes')
    with pytest.raises(ValueError):pilot.checked_review(evidence.manifest_path,evidence.manifest)


def test_extension_requires_own_budget_date_benchmarks_and_fixed_fits(evidence):
    pilot.review(evidence.manifest_path)
    for i in range(3):synthetic_probe(evidence.run,evidence.manifest,i)
    kwargs = {'manifest_path':evidence.manifest_path}
    budget = run.check_budget('extend',evidence.manifest,evidence.run,SIGNATURE,NOW,**kwargs)
    assert len(budget['projection']['fits'])==12 and budget['stage_seconds']==run.EXTENSION_SECONDS
    assert {r['seed'] for r in budget['projection']['fits']}=={2027,2028}
    assert {r['candidate'] for r in budget['projection']['fits']}=={NAMES[0],NAMES[2]}
    with pytest.raises(RuntimeError,match='November 8'):
        run.check_budget('extend',evidence.manifest,evidence.run,SIGNATURE,NOW.replace(day=7),**kwargs)
    evidence.complete_guard('extend',budget)
    assert run.stage_remaining(evidence.run,'extend')==run.EXTENSION_SECONDS-10
    # Unknown extension time consumes its cap, independently of the pilot cap.
    (evidence.run/'attempts/extension_attempt_01/guard_state.json').unlink()
    assert run.stage_remaining(evidence.run,'extend')==0
    assert run.stage_remaining(evidence.run,'pilot')==run.PILOT_SECONDS-10
    with pytest.raises(RuntimeError,match='remaining 6-hour cap'):
        run.check_budget('extend',evidence.manifest,evidence.run,SIGNATURE,NOW,**kwargs)


@pytest.mark.parametrize('chosen',[NAMES[0],NAMES[1]])
def test_complete_three_seed_review_reuses_pilot_and_never_promotes_pair(evidence,chosen):
    if chosen==NAMES[1]:
        # This fixture changes the pre-pilot synthetic predictions, not a saved decision.
        complete = read_json(evidence.run/'pilot/pilot_result.json')
        cfg = evidence.manifest['candidates'][1]['config']
        for q in pilot.QUARTERS:
            report = evidence.save_fit(evidence.run/'pilot/trials/pilot'/chosen,cfg,q,2026,.6)
            entry = next(r for r in complete['completed_fits'] if r['candidate']==chosen and r['quarter']==q)
            entry.update(scores=report['scores'],prediction_sha256=report['prediction_sha256'])
        write_json(evidence.run/'pilot/pilot_result.json',complete)
    pilot.review(evidence.manifest_path)
    saved,plan = pilot.checked_review(evidence.manifest_path,evidence.manifest)
    assert plan['selected_candidate']==chosen
    configs = {r['name']:r['config'] for r in evidence.manifest['candidates']};fits = []
    for name in plan['train_candidates']:
        for seed in plan['new_seeds']:
            for q in pilot.QUARTERS:
                report = evidence.save_fit(evidence.run/'extension/trials/extension'/name,configs[name],q,seed,
                                           .6 if name==chosen else .9)
                fits.append({'candidate':name,'quarter':q,'seed':seed,
                             'scores':report['scores'],'prediction_sha256':report['prediction_sha256']})
    complete = {'manifest_identity':evidence.manifest['identity'],'stage':'extension_complete_review_required',
                'extension_plan':plan,'completed_fits':fits,'expected_fits':plan['new_full_fits'],
                'total_coverage_equivalent_epochs':2,'Q2_Q3_prediction_arrays_loaded':False,
                'independent_success_claimed':False,'final_fit_started':False}
    path = evidence.run/'extension/extension_result.json';write_json(path,complete)
    evidence.complete_guard('extend',{'extension_plan':plan,'pilot_review':saved})
    destination,result = extension.review(evidence.manifest_path)
    assert result['development_qualification_passed'] is True and result['selected_candidate']==chosen
    assert result['seeds']==[2026,2027,2028] and result['new_full_fits']==plan['new_full_fits']
    assert not result['development_checks_authorize_final_data_access'] and not result['independent_research_success_claimed']
    assert (destination/'report.md').exists()
    stable,fresh=extension.checked_review(evidence.manifest_path,evidence.manifest)
    assert stable['decision_identity']==result['decision_identity']==fresh['decision_identity']
    pointer_path=evidence.run/'extension_reviews/latest.json';pointer=read_json(pointer_path)
    write_json(pointer_path,{**pointer,'development_qualification_passed':False})
    with pytest.raises(ValueError,match='pointer or stable'):extension.checked_review(evidence.manifest_path,evidence.manifest)
    write_json(pointer_path,pointer)
    complete['completed_fits'][0] = complete['completed_fits'][1];write_json(path,complete)
    with pytest.raises(ValueError,match='omitted or duplicated'):extension.review(evidence.manifest_path)
    with pytest.raises(RuntimeError,match='already completed'):
        run.check_budget('extend',evidence.manifest,evidence.run,SIGNATURE,NOW,manifest_path=evidence.manifest_path)


def test_one_bad_seed_prevents_development_qualification_even_when_ensemble_improves(evidence):
    _,pilot_report = pilot.review(evidence.manifest_path)
    plan = pilot.expansion_members(evidence.manifest,pilot_report)
    y=evidence.target;selected=np.stack([y*.9,y*.9,-y*.1]);paired=np.stack([y*.1]*3)
    result = extension.evaluate(y,evidence.quarters,evidence.panel.dates[evidence.origins],
             pilot.time_segments(evidence.panel.stamps[evidence.origins]),selected,paired,paired,
             read_json(evidence.base/'system_freeze.json'),plan)
    assert result['selected']['system']['mse_improvement_vs_zero']>.01
    assert not result['development_checks']['all_three_seeds_positive']
    assert not result['development_qualification_passed']


def test_negative_pilot_can_be_recomputed_for_return_without_admitting_extension(evidence):
    complete=read_json(evidence.run/'pilot/pilot_result.json')
    for candidate in evidence.manifest['candidates']:
        for q in pilot.QUARTERS:
            report=evidence.save_fit(evidence.run/'pilot/trials/pilot'/candidate['name'],candidate['config'],q,2026,-.2)
            item=next(r for r in complete['completed_fits'] if r['candidate']==candidate['name'] and r['quarter']==q)
            item.update(scores=report['scores'],prediction_sha256=report['prediction_sha256'])
    write_json(evidence.run/'pilot/pilot_result.json',complete)
    _,report=pilot.review(evidence.manifest_path)
    saved,fresh=pilot.checked_report(evidence.manifest_path,evidence.manifest)
    assert fresh['selected_for_possible_expansion'] is None and fresh['decision_identity']==report['decision_identity']
    assert saved['decision_identity']==report['decision_identity']
    with pytest.raises(RuntimeError,match='No fixed pilot'):pilot.checked_review(evidence.manifest_path,evidence.manifest)
