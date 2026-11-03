"""Freeze confirmed RV models and collect genuine, unrevised forward evidence."""
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .io import ROOT, read_json, write_json, sha, stable_hash
from .jobs import receipt
from .observation import ASSETS, BarStore, DataUnavailable, Journal, Receiver, complete_days, immutable_json, session_origins, stamp, utc_now


PLAN = 'delivery/prospective_lock.json'
OBSERVATION = 'artifacts/rv30/prospective'


def read_plan(config, full=True):
    from .provenance import verify_freeze
    lock = read_json(ROOT/PLAN)
    payload = lock['payload']
    if stable_hash(payload) != lock['signature']:
        raise ValueError('Prospective preregistration changed')
    frozen = verify_freeze(config)
    if payload['base_signature'] != frozen['signature']:
        raise ValueError('Prospective plan belongs to another research protocol')
    if payload['settings'] != read_json(ROOT/'configs/rv30_prospective.json'):
        raise ValueError('Prospective settings changed after freezing')
    if payload['settings']['minimum_complete_sessions'] != config['statistics']['prospective_minimum_complete_sessions']:
        raise ValueError('Prospective duration differs from the registered minimum')
    for path, digest in payload['source_files'].items():
        if sha(ROOT/path) != digest:
            raise ValueError('Prospective implementation or protocol changed: '+path)
    if full:
        from .confirmation import confirmation_evidence
        proof = confirmation_evidence(config, ROOT/payload['run_directory'], frozen['signature'])
        if proof != payload['historical_prerequisites']:
            raise ValueError('Historical prerequisites changed')
        for entry in payload['models'].values():
            source = ROOT/entry['source_directory']
            result = receipt(source)
            if result['status'] != 'complete' or sha(source/'result.json') != entry['source_result_sha256']:
                raise ValueError('A frozen confirmed model changed')
    return lock


def freeze_plan(config, directory):
    from .confirmation import confirmation_evidence
    from .provenance import verify_freeze
    if (ROOT/PLAN).exists():
        return read_plan(config)
    frozen = verify_freeze(config)
    directory = Path(directory)
    proof = confirmation_evidence(config,directory,frozen['signature'])
    confirmation = read_json(directory/'confirmation/evaluation.json')
    settings = read_json(ROOT/'configs/rv30_prospective.json')
    if sorted(confirmation['passed_candidates']) != sorted(settings['candidate_keys']):
        raise ValueError('The prospective family must contain exactly the confirmed candidates')
    models = {}
    for key in settings['candidate_keys']+settings['reference_keys']:
        panel, name = key.split('/')
        source = directory/'confirmation'/panel/name/'s2026'
        request = read_json(source/'request.json');result=receipt(source)
        if result['status'] != 'complete' or request['model'] != name or request['panel'] != panel:
            raise ValueError('Frozen model is not its confirmed source')
        models[key] = {'model':name,'panel':panel,'params':request['params'],'config_id':request['config_id'],'seed':request['seed'],
            'source_directory':str(source.relative_to(ROOT)),'source_result_sha256':sha(source/'result.json'),
            'trained_through':request['partition']['fit_dates'][-1],'calibration_through':request['partition']['calibration_dates'][-1]}
    sources = {name:sha(ROOT/name) for name in ('finance/prospective.py','finance/observation.py','finance/cli.py',
        'configs/rv30_prospective.json','delivery/PROSPECTIVE_PLAN.md','data_acquisition/delivery_data.py')}
    payload = {'schema_version':1,'created_utc':utc_now(),'base_signature':frozen['signature'],
        'run_directory':str(directory.relative_to(ROOT)),'settings':settings,'models':models,'source_files':sources,
        'historical_prerequisites':proof,'no_prospective_observations_before_freeze':True,
        'historical_scores_exposed':True,'prospective_scores_exposed':False,'prospective_data_received':False}
    lock = {'signature':stable_hash(payload),'payload':payload}
    immutable_json(ROOT/PLAN,lock)
    primary = next(k for k in settings['primary_candidate_preference'] if k in models)
    model = {**models[primary], 'schema_version':1,'signature':frozen['signature'],'outcome':'waiting_prospective_acceptance',
        'role':'historically_confirmed_candidate_pending_prospective','historically_qualified':True,
        'active_predictive_edge_established':False,'prospective_qualified':False,
        'prospective_plan_signature':lock['signature'],'direction_or_profit_claim':False,'default_mode':'historical_replay',
        'selection_policy':'original_primary_information_set_preference_not_forecast_score_ranking',
        'prerequisites':proof}
    write_json(ROOT/'delivery/model.json',model)
    write_json(ROOT/'delivery/status.json',{'goal_status':'active','stage':'waiting_timely_SIP_admission',
        'historically_confirmed_candidates':settings['candidate_keys'],'prospective_complete_sessions':0,
        'prospective_required_sessions':settings['minimum_complete_sessions'],'final_acceptance':False,
        'model':'delivery/model.json','prospective_plan':PLAN})
    return lock


def observation_directory(lock):
    return ROOT/OBSERVATION/lock['signature']


def admission(config):
    lock = read_plan(config)
    directory = observation_directory(lock)
    journal = Journal(directory,lock['signature'])
    now = stamp(utc_now());end=now.floor('min')-pd.Timedelta(minutes=1)
    try:
        events = Receiver(journal).fetch(end-pd.Timedelta(minutes=5),end,'admission')
        store=BarStore()
        for event in events:store.apply(event)
        recent = all(store.bars[a] and max(store.bars[a]) >= end-pd.Timedelta(minutes=1) for a in ASSETS)
        local_day=now.tz_convert('America/New_York').strftime('%Y-%m-%d')
        origins=session_origins(local_day)
        in_session=len(origins)>0 and origins[0]-pd.Timedelta(minutes=15)<=now<=origins[-1]+pd.Timedelta(minutes=30)
        passed=bool(recent and in_session)
        result={'passed':passed,'status':'timely_sip_admitted' if passed else 'waiting_recent_regular_session_SIP',
            'checked_utc':utc_now(),'plan_signature':lock['signature'],'last_event_hash':journal.last_hash,
            'recent_all_three_assets':bool(recent),'regular_session':bool(in_session),'http_status':200}
    except DataUnavailable as error:
        result={'passed':False,'status':'blocked_timely_SIP_permission' if error.http_status==403 else 'SIP_request_unavailable','http_status':error.http_status,
            'checked_utc':utc_now(),'plan_signature':lock['signature'],'last_event_hash':journal.last_hash,
            'action':'User handles subscription permissions; no purchase or feed switch performed'}
    write_json(directory/'admission.json',result)
    return result


class Predictor:
    def __init__(self, config, lock):
        from .features import FitState
        from .models import load_model
        from .calibration import Calibration
        self.config,self.lock=config,lock
        self.models={}
        for key,entry in lock['payload']['models'].items():
            source=ROOT/entry['source_directory']
            self.models[key]=(load_model(entry['model'],source,entry['params'],config,'cpu'),
                FitState(**read_json(source/'fit_state.json')),Calibration(**read_json(source/'calibration.json')))

    def predict(self, store, origin):
        from .data import RVData
        from .features import FeatureStore
        from .models import raw_predictions
        origin=stamp(origin)
        cutoff=origin-pd.Timedelta(minutes=2)
        frames=store.frames(cutoff)
        # Bound per-minute work during a multi-month observation. Eighty calendar
        # days retain far more than the registered 22-session feature history.
        history_start=origin.normalize()-pd.Timedelta(days=80)
        frames={asset:frame.loc[frame.index>=history_start] for asset,frame in frames.items()}
        start=min(f.index.min() for f in frames.values()).tz_convert('America/New_York').strftime('%Y-%m-%d')
        day=origin.tz_convert('America/New_York').strftime('%Y-%m-%d')
        data=RVData.from_frames(frames,start,day)
        location=data.timestamps.get_indexer([origin])[0]
        if location<0 or location not in data.origins(require_target=False):
            raise ValueError('Missing minutes or insufficient causal warmup history')
        features=FeatureStore(data,self.config)
        output={}
        for key,(model,state,calibration) in self.models.items():
            entry=self.lock['payload']['models'][key]
            if day<=entry['calibration_through']:
                raise ValueError('Prospective origin predates the frozen model availability')
            raw=raw_predictions(entry['model'],model,data,features,state,[location],entry['params'],entry['panel'],self.config,'cpu')
            mean,quantiles=calibration.predict(raw)
            output[key]={'raw':float(raw[0]),'rv_mean':float(mean[0]),
                'quantiles':{str(k):float(v) for k,v in zip(calibration.levels,quantiles[0])},
                'source_result_sha256':entry['source_result_sha256']}
        return output


def issue(journal, predictor, store, origin):
    origin=stamp(origin);at=stamp(utc_now())
    if at>=origin:
        return journal.append('prediction',{'origin':origin.isoformat(),'status':'not_issued_deadline_passed'})
    common={'origin':origin.isoformat(),'received_at':store.last_received_at.isoformat() if store.last_received_at is not None else None,
        'data_cutoff_at':(origin-pd.Timedelta(minutes=1)).isoformat(),'data_event_hash':store.last_event_hash,
        'target_window_end':(origin+pd.Timedelta(minutes=30)).isoformat(),
        'unit':'sum_of_squared_log_returns_over_30_minutes','eligible_signal':False,'direction_or_profit_prediction':False}
    try:
        forecasts=predictor.predict(store,origin)
        issued=stamp(utc_now())
        if store.last_received_at is None or issued>=origin or store.last_received_at>issued:
            raise ValueError('Response or prediction arrived after the fixed deadline')
        event=journal.append('prediction',{**common,'status':'issued','issued_at':issued.isoformat(),'predictions':forecasts})
        # This timestamp follows fsync of the actual prediction; it establishes
        # whether the prediction was committed before the target began.
        journal.append('issuance_commit',{'prediction_hash':event['hash']})
        return event
    except (ValueError,KeyError,IndexError) as error:
        return journal.append('prediction',{**common,'status':'unavailable','reason':str(error)})


def observe(config):
    """Long-lived local service; it never fabricates missed issue timestamps."""
    lock=read_plan(config)
    directory=observation_directory(lock);settings=lock['payload']['settings']
    admitted=read_json(directory/'admission.json')
    if not admitted['passed'] or admitted['plan_signature']!=lock['signature']:
        raise PermissionError('Actual timely SIP admission is required before observing')
    journal=Journal(directory,lock['signature']);store=BarStore()
    for event in journal.events:
        if event['kind']=='bars':store.apply(event)
    now=stamp(utc_now());end=now.floor('min')-pd.Timedelta(minutes=1)
    receiver=Receiver(journal)
    # Eighty calendar days include at least forty normal exchange sessions.
    for event in receiver.fetch(end-pd.Timedelta(days=80),end,'warmup'):store.apply(event)
    predictor=Predictor(config,lock)
    seen={e['payload']['origin'] for e in journal.events if e['kind']=='prediction'}
    previous_wall=stamp(utc_now())
    quality=complete_days(journal.events,settings)
    quality_day=None
    if quality['ready_for_final_evaluation']:
        return {'passed':True,'status':'ready_for_two_independent_prospective_verifications',**quality}
    while True:
        now=stamp(utc_now())
        if now<previous_wall:raise RuntimeError('Wall clock moved backwards; observation stopped')
        previous_wall=now
        day=now.tz_convert('America/New_York').strftime('%Y-%m-%d')
        origins=session_origins(day)
        next_origin=now.floor('min')+pd.Timedelta(minutes=1)
        wake=next_origin-pd.Timedelta(seconds=settings['forecast_lead_seconds'])
        if now<wake:
            time.sleep(min((wake-now).total_seconds(),30))
            continue
        # Collect through the close plus two minutes to receive final labels.
        if len(origins) and origins[0]-pd.Timedelta(minutes=15)<=now<=origins[-1]+pd.Timedelta(minutes=33):
            end=next_origin-pd.Timedelta(minutes=2)
            for event in receiver.fetch(end-pd.Timedelta(minutes=3),end,'live'):store.apply(event)
            if next_origin in origins and next_origin.isoformat() not in seen:
                issue(journal,predictor,store,next_origin)
                seen.add(next_origin.isoformat())
            if now >= origins[-1]+pd.Timedelta(minutes=31) and quality_day != day:
                quality=complete_days(journal.events,settings)
                quality_day=day
            write_json(directory/'status.json',{'plan_signature':lock['signature'],'updated_utc':utc_now(),
                'pid':__import__('os').getpid(),'last_event_hash':journal.last_hash,**quality})
            if quality['ready_for_final_evaluation']:
                return {'passed':True,'status':'ready_for_two_independent_prospective_verifications',**quality}
        # A bounded sleep keeps cancellation responsive, without agent polling.
        delay=(next_origin+pd.Timedelta(seconds=1)-stamp(utc_now())).total_seconds()
        if delay>0:time.sleep(min(delay,30))


def status(config):
    lock=read_plan(config,full=False)
    directory=observation_directory(lock)
    journal=Journal(directory,lock['signature'])
    result={'plan_signature':lock['signature'],'events':len(journal.events),
        **complete_days(journal.events,lock['payload']['settings'])}
    if (directory/'admission.json').exists():result['admission']=read_json(directory/'admission.json')
    return result


def verify_observation(config, round_number):
    """No score is available before sixty complete sessions; failures are immutable."""
    from .verify import reference_targets, reference_comparison, assert_reference_equal
    from .metrics import qlike, daily_mean, dm_test, block_comparison, holm, score, mincer_zarnowitz
    if round_number not in (1,2):raise ValueError('Independent round must be 1 or 2')
    lock=read_plan(config);settings=lock['payload']['settings'];directory=observation_directory(lock)
    journal=Journal(directory,lock['signature']);quality=complete_days(journal.events,settings)
    if not quality['ready_for_final_evaluation']:
        return {'passed':False,'status':'waiting_60_complete_sessions',**quality}
    path=directory/f'verify_{round_number}.json'
    if path.exists():
        previous=read_json(path)
        if previous['plan_signature']!=lock['signature'] or previous['last_event_hash']!=journal.last_hash:
            raise ValueError('Existing verification belongs to changed observation evidence')
        return previous
    if round_number==2:
        previous=read_json(directory/'verify_1.json')
        if not previous['passed'] or previous['last_event_hash']!=journal.last_hash:
            raise ValueError('The first independent prospective round must pass on identical evidence')
    first_dates=quality['complete_dates'][:settings['minimum_complete_sessions']]
    selected=set(first_dates);predictor=Predictor(config,lock);store=BarStore();records=[]
    try:
        for event in journal.events:
            if stamp(event['recorded_at'])<stamp(lock['payload']['created_utc']):
                raise ValueError('An observation event predates prospective freezing')
            if event['kind']=='bars':store.apply(event)
            elif event['kind']=='prediction':
                p=event['payload'];origin=stamp(p['origin']);day=origin.tz_convert('America/New_York').strftime('%Y-%m-%d')
                if day not in selected:continue
                if p['status']!='issued' or p['data_event_hash']!=store.last_event_hash or stamp(p['received_at'])!=store.last_received_at:
                    raise ValueError('Forecast does not match its actual received-data prefix')
                replay=predictor.predict(store,origin)
                if set(replay)!=set(p['predictions']):raise ValueError('Model family differs from its frozen plan')
                for key in replay:
                    for number in ('raw','rv_mean'):
                        np.testing.assert_allclose(replay[key][number],p['predictions'][key][number],rtol=1e-6,atol=1e-12)
                    if replay[key]['source_result_sha256']!=p['predictions'][key]['source_result_sha256']:
                        raise ValueError('Forecast model receipt differs')
                    np.testing.assert_allclose(list(replay[key]['quantiles'].values()),list(p['predictions'][key]['quantiles'].values()),rtol=1e-6,atol=1e-12)
                records.append(p)
        expected=[t.isoformat() for day in first_dates for t in session_origins(day)]
        if [p['origin'] for p in records]!=expected:
            raise ValueError('Complete prospective grid is missing, reordered or duplicated')
        times=pd.DatetimeIndex([p['origin'] for p in records]);dates=np.asarray(times.tz_convert('America/New_York').strftime('%Y-%m-%d'))
        closes=pd.Series(store.first_live_aapl_closes,dtype=float).sort_index()
        labels=reference_targets(closes,times)
        np.testing.assert_allclose(labels,[store.target(t) for t in times],rtol=1e-8,atol=1e-14)
        values={};descriptions={}
        for key in lock['payload']['models']:
            means=np.asarray([p['predictions'][key]['rv_mean'] for p in records])
            levels=sorted(float(x) for x in records[0]['predictions'][key]['quantiles'])
            quantiles=np.asarray([[p['predictions'][key]['quantiles'][str(q)] for q in levels] for p in records])
            values[key]={'origins':np.arange(len(times)),'timestamps':times.asi8,'dates':dates,'y':labels,
                'mean':means,'quantiles':quantiles,'levels':levels,'seeds':{'2026':means}}
            descriptions[key]={**score(labels,means,dates,quantiles,levels),'mincer_zarnowitz':mincer_zarnowitz(labels,means,dates)}
        comparisons={}
        family=[f'{panel}/{name}' for panel in config['feature_sets'] for name in config['statistics']['candidate_models']]
        for key in settings['candidate_keys']:
            baseline=key.split('/')[0]+'/har';v=values[key];reference=values[baseline]
            _,difference=daily_mean(qlike(labels,v['mean'])-qlike(labels,reference['mean']),dates)
            dm=dm_test(difference,settings['dm_hac_lags'])
            blocks=[block_comparison(difference,length,settings['bootstrap_repeats'],settings['bootstrap_seed']) for length in settings['block_lengths_sessions']]
            result={'mean_difference':float(difference.mean()),'dm':dm,'bootstraps':blocks,
                'p_conservative':max(dm['p_less'],*(b['p_less_centered_null'] for b in blocks))}
            independent=reference_comparison(v,reference,settings)
            for field in result:
                assert_reference_equal(result[field],independent[field],field)
            comparisons[key]=result
        adjusted=holm([comparisons[k]['p_conservative'] if k in comparisons else 1. for k in family])
        # Separate scalar step-down check, without sharing the evaluator's Holm routine.
        ordered=sorted((comparisons[k]['p_conservative'] if k in comparisons else 1.,k) for k in family)
        greatest=0.;independent_holm={}
        for i,(p,key) in enumerate(ordered):
            greatest=max(greatest,(len(family)-i)*p);independent_holm[key]=min(greatest,1.)
        for key,item in comparisons.items():
            item['holm_p']=float(adjusted[family.index(key)])
            np.testing.assert_allclose(item['holm_p'],independent_holm[key],rtol=1e-12,atol=1e-14)
            item['passed_predictive_gate']=bool(item['mean_difference']<0 and item['holm_p']<settings['alpha'] and all(b['ci95'][1]<0 for b in item['bootstraps']))
        passed=[k for k in settings['primary_candidate_preference'] if comparisons[k]['passed_predictive_gate']]
        result={'passed':True,'round':round_number,'plan_signature':lock['signature'],'last_event_hash':journal.last_hash,
            'completed_utc':utc_now(),'first_60_complete_dates':first_dates,'sessions':len(first_dates),'origins':len(records),
            'full_model_replay':True,'all_labels_independently_rebuilt':True,'descriptions':descriptions,'comparisons':comparisons,
            'qualified_candidates':passed,'primary_model':passed[0] if passed else None,'quality':quality,
            'direction_or_profit_claim':False,'verification_implementation':'full_causal_prefix_replay_and_independent_labels_HAC_block_sums_Holm'}
        if round_number==2:
            prior=read_json(directory/'verify_1.json')
            for key in ('first_60_complete_dates','sessions','origins','descriptions','comparisons','qualified_candidates','primary_model'):
                assert_reference_equal(result[key],prior[key],key)
    except (ValueError,KeyError,AssertionError,IndexError) as error:
        result={'passed':False,'round':round_number,'plan_signature':lock['signature'],'last_event_hash':journal.last_hash,
            'completed_utc':utc_now(),'status':'prospective_verification_failed','error':str(error)}
    immutable_json(path,result)
    return result
