"""Forward evidence needs real receipt order, complete grids, and frozen models."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finance import observation as obs, prospective as forward
from finance.io import ROOT, read_json, write_json, sha


SETTINGS = read_json(ROOT/'configs/rv30_prospective.json')


def bar_event(day='2026-11-04', purpose='live'):
    start = pd.Timestamp(day+'T09:30:00',tz='America/New_York').tz_convert('UTC')
    close = start+pd.Timedelta(minutes=390)
    bars=[{'t':t.isoformat(),'o':100.,'h':101.,'l':99.,'c':100.+i/10000,'v':1000} for i,t in enumerate(pd.date_range(start,periods=390,freq='min'))]
    return {'kind':'bars','hash':'actual_response','recorded_at':(close+pd.Timedelta(minutes=2)).isoformat(),
        'payload':{'http_status':200,'feed':'sip','adjustment':'raw','purpose':purpose,'end':close.isoformat(),
            'received_at':(close+pd.Timedelta(minutes=1)).isoformat(),'response':{'bars':{'AAPL':bars,'SPY':[],'QQQ':[]}}}}


def full_quality_events(day='2026-11-04'):
    events=[]
    for i,t in enumerate(obs.session_origins(day)):
        event={'kind':'prediction','hash':f'p{i}','recorded_at':(t-pd.Timedelta(seconds=2)).isoformat(),
            'payload':{'origin':t.isoformat(),'status':'issued','issued_at':(t-pd.Timedelta(seconds=3)).isoformat(),
                'received_at':(t-pd.Timedelta(seconds=10)).isoformat(),'data_cutoff_at':(t-pd.Timedelta(minutes=1)).isoformat(),
                'predictions':{k:{} for k in SETTINGS['candidate_keys']+SETTINGS['reference_keys']}}}
        events.extend([event,{'kind':'issuance_commit','recorded_at':(t-pd.Timedelta(seconds=1)).isoformat(),'payload':{'prediction_hash':event['hash']}}])
    events.append(bar_event(day))
    return events


def test_exchange_grids_include_dst_and_actual_early_close():
    before=obs.session_origins('2026-10-30');after=obs.session_origins('2026-11-02');short=obs.session_origins('2026-11-27')
    assert len(before)==len(after)==346 and len(short)==166
    assert before[0].hour==13 and after[0].hour==14
    assert short[-1].tz_convert('America/New_York').strftime('%H:%M')=='12:30'
    assert len(obs.session_origins('2026-11-26'))==0


def test_chain_is_immutable_atomic_and_detects_tamper_or_missing_event(tmp_path):
    journal=obs.Journal(tmp_path,'frozen')
    first=journal.append('test',{},'2026-11-04T10:00:00Z');journal.append('test',{},'2026-11-04T10:00:01Z')
    assert len(obs.Journal(tmp_path,'frozen').events)==2
    with pytest.raises(FileExistsError):obs.immutable_json(tmp_path/'events/00000001.json',{})
    assert not list((tmp_path/'events').glob('.pending-*'))
    with pytest.raises(ValueError,match='backwards'):journal.append('test',{},'2026-11-04T09:00:00Z')
    path=tmp_path/'events/00000001.json';value=read_json(path);value['payload']={'changed':True};write_json(path,value)
    with pytest.raises(ValueError,match='chain'):obs.Journal(tmp_path,'frozen')
    path.unlink()
    with pytest.raises(ValueError,match='chain'):obs.Journal(tmp_path,'frozen')


def test_partial_future_and_wrong_feed_bars_are_rejected():
    for kind in ('future','partial','feed'):
        event=bar_event()
        if kind=='future':event['payload']['end']='2026-11-04T12:00:00Z'
        elif kind=='partial':event['payload']['received_at']='2026-11-04T14:30:00Z'
        else:event['payload']['feed']='iex'
        with pytest.raises(ValueError):obs.BarStore().apply(event)


def test_historical_warmup_cannot_backfill_actual_targets():
    store=obs.BarStore();store.apply(bar_event(purpose='warmup'))
    with pytest.raises(ValueError,match='live target'):store.target(obs.session_origins('2026-11-04')[0])
    store.apply(bar_event())
    assert np.isfinite(store.target(obs.session_origins('2026-11-04')[0]))


def test_complete_quality_depends_on_whole_day_commit_and_labels():
    events=full_quality_events();good=obs.complete_days(events,SETTINGS)
    assert good['complete_sessions']==1 and not good['ready_for_final_evaluation']
    assert 'comparisons' not in good and not good['cumulative_predictive_scores_disclosed']
    assert good['days']['2026-11-04']['complete_origins']==346
    for corruption in ('missing','late','missing_commit','missing_label','wrong_models'):
        changed=deepcopy(events)
        if corruption=='missing':changed=changed[2:]
        elif corruption=='late':changed[1]['recorded_at']=changed[0]['payload']['origin']
        elif corruption=='missing_commit':changed.pop(1)
        elif corruption=='missing_label':changed[-1]['payload']['response']['bars']['AAPL'].pop(100)
        else:changed[0]['payload']['predictions'].pop('market/har')
        assert obs.complete_days(changed,SETTINGS)['complete_sessions']==0
    with pytest.raises(ValueError,match='Duplicate prospective origin'):obs.complete_days(events+[events[0]],SETTINGS)


def test_early_scores_do_not_construct_a_model_or_expose_losses(monkeypatch,tmp_path):
    lock={'signature':'plan','payload':{'settings':SETTINGS}}
    monkeypatch.setattr(forward,'read_plan',lambda config:lock)
    monkeypatch.setattr(forward,'observation_directory',lambda lock:tmp_path)
    monkeypatch.setattr(forward,'Predictor',lambda *a:pytest.fail('No scoring or model replay before sixty complete days'))
    result=forward.verify_observation({},1)
    assert result['status']=='waiting_60_complete_sessions' and result['complete_sessions']==0
    assert not (tmp_path/'verify_1.json').exists() and 'comparisons' not in result


def test_fsync_after_deadline_cannot_qualify_an_issued_forecast(monkeypatch,tmp_path):
    origin=pd.Timestamp('2026-11-04T15:00:00Z');clock=[origin-pd.Timedelta(seconds=3)]
    def now():
        clock[0]+=pd.Timedelta(milliseconds=1)
        return clock[0].isoformat()
    monkeypatch.setattr(obs,'utc_now',now);monkeypatch.setattr(forward,'utc_now',now)
    store=obs.BarStore();store.last_received_at=origin-pd.Timedelta(seconds=20);store.last_event_hash='actual'
    class Predictor:
        def predict(self,*a):return {k:{} for k in SETTINGS['candidate_keys']+SETTINGS['reference_keys']}
    original=obs.immutable_json
    def slow_fsync(path,value):
        original(path,value)
        if value['kind']=='prediction':clock[0]=origin+pd.Timedelta(milliseconds=1)
    monkeypatch.setattr(obs,'immutable_json',slow_fsync)
    journal=obs.Journal(tmp_path,'plan');p=forward.issue(journal,Predictor(),store,origin)
    assert p['payload']['status']=='issued'
    assert obs.stamp(journal.events[-1]['recorded_at'])>origin
    result=obs.complete_days(journal.events,SETTINGS)
    assert result['complete_sessions']==0 and result['days']['2026-11-04']['late_origins']==1


def test_receiver_retains_actual_403_and_never_falls_back_or_records_keys(monkeypatch,tmp_path):
    class Client:
        def get(self,path,params):
            assert params['feed']=='sip' and params['adjustment']=='raw'
            return 403,b'{"message":"secret-value-must-not-be-stored"}',{}
    monkeypatch.setattr('data_acquisition.delivery_data.Client',Client)
    journal=obs.Journal(tmp_path,'plan')
    with pytest.raises(obs.DataUnavailable) as error:
        obs.Receiver(journal).fetch('2026-11-04T14:30:00Z','2026-11-04T14:31:00Z','admission')
    assert error.value.http_status==403
    assert journal.events[0]['kind']=='data_unavailable'
    assert 'secret-value' not in json.dumps(journal.events)


@pytest.fixture
def frozen_sources(monkeypatch,tmp_path):
    monkeypatch.setattr(forward,'ROOT',tmp_path)
    from finance import provenance,confirmation
    proof={'not_before_utc':'2026-11-03T22:34:53+00:00','evaluation':{},'verifications':{}}
    monkeypatch.setattr(provenance,'verify_freeze',lambda config:{'signature':'historical'})
    monkeypatch.setattr(confirmation,'confirmation_evidence',lambda *args:proof)
    for name in ('finance/prospective.py','finance/observation.py','finance/cli.py','delivery/PROSPECTIVE_PLAN.md','data_acquisition/delivery_data.py'):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('preserved-source')
    write_json(tmp_path/'configs/rv30_prospective.json',SETTINGS)
    run=tmp_path/'artifacts/development';write_json(run/'confirmation/evaluation.json',{'passed_candidates':SETTINGS['candidate_keys']})
    for key in SETTINGS['candidate_keys']+SETTINGS['reference_keys']:
        panel,name=key.split('/');source=run/'confirmation'/panel/name/'s2026'
        write_json(source/'request.json',{'model':name,'panel':panel,'params':{'alpha':.1},'config_id':'fixed','seed':2026,
            'partition':{'fit_dates':['2025-05-28'],'calibration_dates':['2025-06-27']}})
        write_json(source/'model.npz',{'actual':'frozen'})
        write_json(source/'result.json',{'status':'complete','files':{'model.npz':sha(source/'model.npz')}})
    return tmp_path,run,{'statistics':{'prospective_minimum_complete_sessions':60}}


def test_freeze_uses_exact_confirmed_models_and_never_marks_prospective_success(frozen_sources):
    root,run,config=frozen_sources;lock=forward.freeze_plan(config,run)
    assert lock['payload']['settings']['minimum_complete_sessions']==60
    assert len(lock['payload']['models'])==4
    model=read_json(root/'delivery/model.json')
    assert model['model']=='ridge' and model['panel']=='aapl' and model['historically_qualified']
    assert not model['prospective_qualified'] and not model['active_predictive_edge_established']
    assert forward.freeze_plan(config,run)==lock
    p=root/'finance/observation.py';p.write_text('altered')
    with pytest.raises(ValueError,match='implementation'):forward.read_plan(config)


def test_unconfirmed_candidate_cannot_be_added_to_frozen_family(frozen_sources):
    root,run,config=frozen_sources
    write_json(run/'confirmation/evaluation.json',{'passed_candidates':['aapl/ridge']})
    with pytest.raises(ValueError,match='exactly'):forward.freeze_plan(config,run)


def test_final_observation_replays_both_rounds_and_catches_altered_predictions(monkeypatch,tmp_path):
    settings=deepcopy(SETTINGS);settings['minimum_complete_sessions']=3;settings['bootstrap_repeats']=100
    keys=settings['candidate_keys']+settings['reference_keys'];events=[];forecast={}
    dates=['2026-11-02','2026-11-03','2026-11-04']
    for day in dates:
        for i,origin in enumerate(obs.session_origins(day)):
            receive=(origin-pd.Timedelta(seconds=5)).isoformat();digest=f'{day}_{i}'
            event={'kind':'bars','hash':digest,'recorded_at':receive,'payload':{'http_status':200,'feed':'sip','adjustment':'raw',
                'purpose':'live','received_at':receive,'end':(origin-pd.Timedelta(minutes=2)).isoformat(),'response':{'bars':{}}}}
            events.append(event)
            values={key:{'raw':1e-9 if key.endswith('ridge') else 1e-8,'rv_mean':1e-9 if key.endswith('ridge') else 1e-8,
                'quantiles':{'0.05':1e-12,'0.1':1e-11,'0.9':1e-7,'0.95':1e-6},'source_result_sha256':key} for key in keys}
            forecast[origin.isoformat()]=deepcopy(values)
            prediction={'kind':'prediction','hash':digest+'_p','recorded_at':(origin-pd.Timedelta(seconds=2)).isoformat(),
                'payload':{'origin':origin.isoformat(),'status':'issued','issued_at':(origin-pd.Timedelta(seconds=3)).isoformat(),
                    'received_at':receive,'data_cutoff_at':(origin-pd.Timedelta(minutes=1)).isoformat(),
                    'data_event_hash':digest,'predictions':values}}
            events.extend([prediction,{'kind':'issuance_commit','recorded_at':(origin-pd.Timedelta(seconds=1)).isoformat(),'payload':{'prediction_hash':prediction['hash']}}])
        events.append(bar_event(day))
    class Journal:
        def __init__(self,*args):self.events=events;self.last_hash='fixed-observation'
    class Predictor:
        def __init__(self,*args):pass
        def predict(self,store,origin):return forecast[origin.isoformat()]
    lock={'signature':'plan','payload':{'settings':settings,'models':{k:{} for k in keys},'created_utc':'2026-11-01T00:00:00Z'}}
    config=read_json(ROOT/'configs/rv30.json')
    monkeypatch.setattr(forward,'read_plan',lambda *a:lock)
    monkeypatch.setattr(forward,'observation_directory',lambda *a:tmp_path)
    monkeypatch.setattr(forward,'Journal',Journal);monkeypatch.setattr(forward,'Predictor',Predictor)
    with pytest.raises(FileNotFoundError):forward.verify_observation(config,2)
    first=forward.verify_observation(config,1)
    assert first['passed'] and first['sessions']==3 and first['origins']==1038
    second=forward.verify_observation(config,2)
    assert second['passed'] and second['comparisons']==first['comparisons']
    assert forward.verify_observation(config,1)==first
    other=tmp_path/'corruption';monkeypatch.setattr(forward,'observation_directory',lambda *a:other)
    prediction=next(e for e in events if e['kind']=='prediction')
    prediction['payload']['predictions']['aapl/ridge']['rv_mean']=1.
    corrupted=forward.verify_observation(config,1)
    assert not corrupted['passed'] and corrupted['status']=='prospective_verification_failed'
    assert read_json(other/'verify_1.json')==corrupted
