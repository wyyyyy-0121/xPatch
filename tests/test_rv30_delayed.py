"""The free flow changes publication time, never model time or real event time."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from finance import delayed, delayed_service as service, observation as obs, prospective
from finance.delayed_validation import MODE, DELAY, complete_days, session_origins
from finance.io import ROOT, read_json, sha, write_json


SETTINGS = read_json(ROOT/'configs/rv30_delayed_sip.json')
KEYS = SETTINGS['candidate_keys']+SETTINGS['reference_keys']


def events_for_day(day='2026-11-04'):
    events = []
    for i, origin in enumerate(session_origins(day)):
        deadline = origin+DELAY; received = deadline-pd.Timedelta(seconds=5)
        digest = f'{day}_{i}'
        events.append({'kind':'bars', 'hash':digest, 'recorded_at':received.isoformat(),
            'payload':{'http_status':200, 'feed':'sip', 'adjustment':'raw', 'purpose':'live',
                'end':(origin-pd.Timedelta(minutes=2)).isoformat(), 'received_at':received.isoformat(),
                'response':{'bars':{}}}})
        p = {'origin':origin.isoformat(), 'market_origin':origin.isoformat(), 'mode':MODE,
            'scheduled_publication_at':deadline.isoformat(), 'prospective_qualified':False,
            'eligible_signal':False, 'status':'issued', 'issued_at':(deadline-pd.Timedelta(seconds=3)).isoformat(),
            'received_at':received.isoformat(), 'data_event_hash':digest,
            'data_cutoff_at':(origin-pd.Timedelta(minutes=1)).isoformat(),
            'target_window_start':origin.isoformat(), 'target_window_end':(origin+pd.Timedelta(minutes=30)).isoformat(),
            'predictions':{key:{'raw':1e-9 if key.endswith('ridge') else 1e-8,
                'rv_mean':1e-9 if key.endswith('ridge') else 1e-8,
                'quantiles':{'0.05':1e-12,'0.1':1e-11,'0.9':1e-7,'0.95':1e-6},
                'source_result_sha256':key} for key in KEYS}}
        events.extend([{'kind':'prediction','hash':digest+'_p',
            'recorded_at':(deadline-pd.Timedelta(seconds=2)).isoformat(), 'payload':p},
            {'kind':'issuance_commit','recorded_at':(deadline-pd.Timedelta(seconds=1)).isoformat(),
             'payload':{'prediction_hash':digest+'_p'}}])
    start = pd.Timestamp(day+'T09:30:00',tz='America/New_York').tz_convert('UTC')
    close = session_origins(day)[-1]+pd.Timedelta(minutes=30)
    bars = [{'t':t.isoformat(), 'o':100., 'h':101., 'l':99., 'c':100.+i/10000, 'v':1000}
            for i,t in enumerate(pd.date_range(start,close-pd.Timedelta(minutes=1),freq='min'))]
    events.append({'kind':'bars','hash':day+'_labels','recorded_at':(close+DELAY+pd.Timedelta(minutes=2)).isoformat(),
        'payload':{'http_status':200,'feed':'sip','adjustment':'raw','purpose':'live',
            'end':close.isoformat(),'received_at':(close+DELAY+pd.Timedelta(minutes=1)).isoformat(),
            'response':{'bars':{'AAPL':bars}}}})
    return events


@pytest.mark.parametrize('day,count,hour', [('2026-10-30',346,13),('2026-11-02',346,14),('2026-11-27',166,14)])
def test_original_grid_and_all_four_models_qualify_only_as_replay(day,count,hour):
    origins = session_origins(day)
    assert len(origins)==count and origins[0].hour==hour
    quality = complete_days(events_for_day(day),SETTINGS)
    assert quality['complete_sessions']==1 and quality['days'][day]['complete_origins']==count
    assert quality['timely_prospective_complete_sessions']==0 and not quality['prospective_qualified']
    assert not quality['ready_for_final_evaluation'] and not quality['cumulative_predictive_scores_disclosed']


@pytest.mark.parametrize('now,market', [('2026-11-04T14:59:55Z','2026-11-04T14:45:00Z'),
    ('2026-11-04T15:00:01Z','2026-11-04T14:46:00Z')])
def test_scheduler_moves_the_entire_original_clock(now,market):
    assert delayed.next_origin(now)==obs.stamp(market)
    assert issubclass(delayed.Predictor,prospective.Predictor)


def test_fast_frame_path_matches_original_after_reordering_revision_and_future_bars():
    original=obs.BarStore();fast=delayed.ReplayStore()
    event=events_for_day()[-1]
    event['payload']['response']['bars']={asset:deepcopy(event['payload']['response']['bars']['AAPL'][::-1])
                                        for asset in obs.ASSETS}
    for change in (None,'revision','missing'):
        if change=='revision':
            event['payload']['response']['bars']['AAPL'][20]['c']+=.1
        elif change=='missing':
            event['payload']['response']['bars']['SPY'].pop(30)
        # Start fresh so the missing-minute case really has a hole.
        original=obs.BarStore();fast=delayed.ReplayStore()
        original.apply(event);fast.apply(event)
        for cutoff in (None,'2026-11-04T18:00:00Z','2026-11-04T15:00:00Z'):
            expected=original.frames(cutoff);actual=fast.frames(cutoff)
            for asset in obs.ASSETS:
                pd.testing.assert_frame_equal(expected[asset],actual[asset],check_exact=True)


def test_calendar_cache_matches_original_data_across_dst_half_day_and_price_updates():
    from finance.data import RVData
    from finance_v2.data import calendar_grid
    for start,end in (('2026-10-23','2026-11-04'),('2026-11-20','2026-11-27')):
        grids=calendar_grid(start,end)
        index=pd.DatetimeIndex(np.concatenate([g.asi8 for g in grids.values()]),tz='UTC')
        frames={asset:pd.DataFrame({'open':100.,'high':102.,'low':98.,
            'close':100.+np.sin(np.arange(len(index))*.01),'volume':1000.},index=index) for asset in obs.ASSETS}
        for revision in (False,True):
            if revision:
                frames['SPY']=frames['SPY'].iloc[:-5].copy()
                frames['AAPL'].iloc[10:20,3]*=1.001
            expected=RVData.from_frames(frames,start,end);actual=delayed.causal_data(frames,start,end)
            pd.testing.assert_index_equal(expected.timestamps,actual.timestamps)
            for name in ('dates','days','day_index','returns','volume','log_range'):
                np.testing.assert_array_equal(getattr(expected,name),getattr(actual,name))
            assert expected.regular_audits==actual.regular_audits
            assert expected.close_by_date==actual.close_by_date


@pytest.mark.parametrize('corruption', ['cutoff','target','fake_time','late_commit','missing_commit',
    'missing_origin','wrong_models','wrong_prefix','wrong_mode','missing_label','warmup_label'])
def test_incomplete_or_misdated_day_is_not_counted(corruption):
    events = events_for_day();p = events[1]['payload']
    if corruption=='cutoff':p['data_cutoff_at']=p['origin']
    elif corruption=='target':p['target_window_start']=p['scheduled_publication_at']
    elif corruption=='fake_time':p['issued_at']=p['origin']
    elif corruption=='late_commit':events[2]['recorded_at']=p['scheduled_publication_at']
    elif corruption=='missing_commit':events.pop(2)
    elif corruption=='missing_origin':events=events[3:]
    elif corruption=='wrong_models':p['predictions'].pop('market/har')
    elif corruption=='wrong_prefix':p['data_event_hash']='another-response'
    elif corruption=='wrong_mode':p['prospective_qualified']=True
    elif corruption=='missing_label':events[-1]['payload']['response']['bars']['AAPL'].pop(100)
    else:events[-1]['payload']['purpose']='warmup'
    assert complete_days(events,SETTINGS)['complete_sessions']==0


def test_duplicate_origin_commit_and_recent_free_request_are_rejected():
    events = events_for_day()
    with pytest.raises(ValueError,match='Duplicate delayed replay'):complete_days(events+[events[1]],SETTINGS)
    with pytest.raises(ValueError,match='Duplicate delayed prediction commit'):complete_days(events+[events[2]],SETTINGS)
    events[0]['payload']['end']=events[0]['payload']['received_at']
    with pytest.raises(ValueError,match='15-minute'):complete_days(events,SETTINGS)


def test_issue_preserves_real_times_and_original_model_arguments(monkeypatch,tmp_path):
    origin = obs.stamp('2026-11-04T14:45:00Z');clock=[origin+DELAY-pd.Timedelta(seconds=4)]
    def now():
        clock[0]+=pd.Timedelta(milliseconds=1)
        return clock[0].isoformat()
    monkeypatch.setattr(obs,'utc_now',now);monkeypatch.setattr(delayed,'utc_now',now)
    store=obs.BarStore();store.last_received_at=clock[0]-pd.Timedelta(seconds=1);store.last_event_hash='actual'
    class Predictor:
        def predict(self,actual_store,actual_origin):
            assert actual_store is store and actual_origin==origin
            return {key:{'rv_mean':1e-6} for key in KEYS}
    journal=obs.Journal(tmp_path,'isolated')
    p=delayed.issue(journal,Predictor(),store,origin)['payload']
    assert p['status']=='issued' and obs.stamp(p['issued_at'])>origin
    assert obs.stamp(p['target_window_start'])==origin
    assert obs.stamp(p['target_window_end'])==origin+pd.Timedelta(minutes=30)
    clock[0]=origin+DELAY
    assert delayed.issue(journal,Predictor(),store,origin)['payload']['status']=='not_issued_deadline_passed'


def test_outside_session_admission_never_queries_provider(monkeypatch,tmp_path):
    lock={'signature':'isolated','payload':{}}
    monkeypatch.setattr(delayed,'observation_directory',lambda *a:tmp_path)
    monkeypatch.setattr(delayed,'utc_now',lambda:'2026-11-04T05:00:00Z')
    monkeypatch.setattr(delayed,'Receiver',lambda *a:pytest.fail('No off-hours provider poll'))
    assert not delayed.admission({},lock)['passed']


def test_collector_warms_full_history_then_publishes_the_original_origin(monkeypatch,tmp_path):
    clock=[obs.stamp('2026-11-04T14:59:50Z')];requests=[];statuses=[]
    monkeypatch.setattr(obs,'utc_now',lambda:clock[0].isoformat())
    monkeypatch.setattr(delayed,'utc_now',lambda:clock[0].isoformat())
    monkeypatch.setattr(delayed,'observation_directory',lambda *a:tmp_path)
    monkeypatch.setattr(delayed,'admission',lambda *a:{'passed':True})
    monkeypatch.setattr(service,'STOP',False)
    monkeypatch.setattr(service,'save_status',lambda lock,stage,quality,*a:statuses.append(stage))
    class Receiver:
        def __init__(self,journal):self.journal=journal
        def fetch(self,start,end,purpose):
            requests.append((start,end,purpose))
            bars={asset:[{'t':end.isoformat(),'o':100.,'h':101.,'l':99.,'c':100.,'v':1000}] for asset in obs.ASSETS}
            return [self.journal.append('bars',{'http_status':200,'feed':'sip','adjustment':'raw','purpose':purpose,
                'received_at':clock[0].isoformat(),'end':end.isoformat(),'response':{'bars':bars}})]
    class Predictor:
        def __init__(self,*a):pass
        def predict(self,store,origin):
            assert origin==obs.stamp('2026-11-04T14:45:00Z')
            assert max(store.bars['AAPL'])==origin-pd.Timedelta(minutes=2)
            clock[0]+=pd.Timedelta(seconds=1)
            return {key:{'rv_mean':1e-6} for key in KEYS}
    monkeypatch.setattr(obs,'Receiver',Receiver);monkeypatch.setattr(delayed,'Predictor',Predictor)
    class Meter:
        def sleep(self,seconds):
            clock[0]+=pd.Timedelta(seconds=seconds)
            if any(e['kind']=='issuance_commit' for e in obs.Journal(tmp_path,'isolated').events):
                service.STOP=True
    lock={'signature':'isolated','payload':{'settings':SETTINGS}}
    # An admission-only prefix must not be mistaken for 80 days of model warmup.
    journal=obs.Journal(tmp_path,'isolated')
    Receiver(journal).fetch(clock[0]-pd.Timedelta(minutes=20),clock[0]-pd.Timedelta(minutes=17),'admission')
    service.observe({},lock,Meter())
    warmup=next(row for row in requests if row[2]=='warmup')
    assert warmup[1]-warmup[0]==pd.Timedelta(days=80)
    predictions=[e['payload'] for e in obs.Journal(tmp_path,'isolated').events if e['kind']=='prediction']
    assert len(predictions)==1 and predictions[0]['origin']=='2026-11-04T14:45:00+00:00'
    assert predictions[0]['issued_at']=='2026-11-04T14:59:56+00:00'


def test_original_forward_namespace_and_early_scoring_remain_unchanged(monkeypatch,tmp_path):
    lock={'signature':'isolated','payload':{'settings':SETTINGS}}
    monkeypatch.setattr(delayed,'read_plan',lambda *a,**kw:lock)
    monkeypatch.setattr(delayed,'observation_directory',lambda *a:tmp_path)
    monkeypatch.setattr(prospective,'Predictor',lambda *a:pytest.fail('No model before sixty days'))
    before=prospective.read_plan,prospective.complete_days,prospective.observation_directory
    result=delayed.verify({},1)
    assert result['status']=='waiting_60_complete_sessions' and not result['prospective_qualified']
    assert before==(prospective.read_plan,prospective.complete_days,prospective.observation_directory)
    assert not (tmp_path/'verify_1.json').exists()


def test_two_numerical_replay_rounds_and_corruption_detection(monkeypatch,tmp_path):
    settings=deepcopy(SETTINGS);settings.update(minimum_complete_sessions=3,bootstrap_repeats=100)
    events=sum((events_for_day(day) for day in ('2026-11-02','2026-11-03','2026-11-04')),[])
    forecasts={e['payload']['origin']:deepcopy(e['payload']['predictions']) for e in events if e['kind']=='prediction'}
    class Journal:
        def __init__(self,*a):self.events=events;self.last_hash='same-observation'
    class Predictor:
        def __init__(self,*a):pass
        def predict(self,store,origin):return forecasts[origin.isoformat()]
    lock={'signature':'isolated','payload':{'settings':settings,'models':{k:{} for k in KEYS},'created_utc':'2026-11-01T00:00:00Z'}}
    monkeypatch.setattr(delayed,'read_plan',lambda *a:lock)
    monkeypatch.setattr(delayed,'observation_directory',lambda *a:tmp_path)
    monkeypatch.setattr(prospective,'Journal',Journal);monkeypatch.setattr(prospective,'Predictor',Predictor)
    config=read_json(ROOT/'configs/rv30.json')
    with pytest.raises(FileNotFoundError):delayed.verify(config,2)
    first=delayed.verify(config,1);second=delayed.verify(config,2)
    assert first['passed'] and second['passed'] and first['comparisons']==second['comparisons']
    assert first['sessions']==3 and first['origins']==1038 and not first['prospective_qualified']
    assert read_json(tmp_path/'verify_1.json')['mode']==MODE
    monkeypatch.setattr(delayed,'observation_directory',lambda *a:tmp_path/'corruption')
    next(e for e in events if e['kind']=='prediction')['payload']['predictions']['aapl/ridge']['rv_mean']=1.
    assert not delayed.verify(config,1)['passed']


def test_replay_budget_is_mirrored_once_and_exceptions_restore_context(monkeypatch,tmp_path):
    from finance import forward_runtime as runtime
    config={'runtime':{'total_cpu_core_hours':2000,'total_gpu_hours':240,'memory_budget_gib':10}}
    lock={'signature':'shadow','payload':{'base_signature':'base'}}
    monkeypatch.setattr(runtime,'ROOT',tmp_path);monkeypatch.setattr(service,'ROOT',tmp_path)
    monkeypatch.setattr(delayed,'observation_directory',lambda *a:tmp_path/'shadow')
    monkeypatch.setattr(runtime.os,'sched_getaffinity',lambda *a:{0,1,2,3})
    ledger=tmp_path/'artifacts/rv30/base/interrupted_resource_charges.json';write_json(ledger,{})
    def usage(signature):return 120*3600+sum(r['allocated_cpu_seconds'] for r in read_json(ledger).values()),20*3600
    monkeypatch.setattr(runtime,'historical_usage',usage)
    original_dir=prospective.observation_directory
    service.execute(config,lock,'observe',lambda meter:None)
    assert prospective.observation_directory is original_dir and runtime.historical_usage is usage
    with delayed.resource_context():
        rows,_=runtime.resource_records(lock)
        assert service.mirrored_cpu(lock)==sum(r['charged_cpu_seconds'] for r in rows)
    assert usage('base')[0]==120*3600+sum(r['charged_cpu_seconds'] for r in rows)
    def fail(meter):raise RuntimeError('controlled failure')
    with pytest.raises(RuntimeError,match='controlled'):service.execute(config,lock,'verify_1',fail)
    assert runtime.ACTIVE is None and runtime.historical_usage is usage
    assert prospective.observation_directory is original_dir
    # A hard interruption stays fully reserved until a real end receipt exists.
    with delayed.resource_context():
        meter=runtime.Meter(config,lock,'observe');service.sync_charges(lock)
        charges=read_json(ledger);key='delayed_replay_resource:'+str((meter.path/'start.json').relative_to(tmp_path))
        assert charges[key]['allocated_cpu_seconds']==4*runtime.ACTION_LIMITS['observe']
        with pytest.raises(RuntimeError,match='still alive'):runtime.recover_interrupted(lock)
        record=read_json(meter.path/'start.json');record['kernel_start_ticks']='old-identity';write_json(meter.path/'start.json',record)
        runtime.recover_interrupted(lock);service.sync_charges(lock)
        assert read_json(ledger)[key]['allocated_cpu_seconds']==record['reserved_cpu_seconds']
