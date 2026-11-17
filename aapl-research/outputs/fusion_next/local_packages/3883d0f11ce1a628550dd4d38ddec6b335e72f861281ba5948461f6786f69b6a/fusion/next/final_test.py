"""Separate all-day F7: gated local acquisition, one fixed artifact, resumable scoring.

The original afternoon holdout and its failed-development guard stay untouched.
This module never opens that CSV. New all-day access requires the new study's
frozen development gate and independently checked, single fixed Q2 confirmation.
"""
import argparse
from contextlib import contextmanager
import fcntl
import gzip
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError,URLError
from urllib.parse import urlencode
from urllib.request import Request,build_opener

import numpy as np
import pandas as pd

from aapl.data import sessions
from .common import ROOT,OUT,PLAN,digest,identity,read_json,write_json,utc_now
from .confirmation import checked_freeze
from .freeze import recheck_seed_predictions,recheck_controls
from .panel import Panel
from .data import FIELDS,POLICY,causal_transform,eligible_origins
from .splits import fold
from .experts import NAMES
from .evaluation import point_scores,seed_system,compare
from .acquire import credentials,NoRedirect,ENDPOINT,bounds
from .hardware import runtime,ResourceGuard
from . import simple

FINAL=OUT/'final_test'
START='2026-07-01'
END='2026-09-30'


@contextmanager
def exclusive():
    FINAL.mkdir(parents=True,exist_ok=True)
    with (FINAL/'execution.lock').open('a') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('The one fixed final evaluation is already running') from None
        yield


def recheck_confirmation(panel,frozen,confirmation):
    """Q2 summary flags alone cannot authorize reading any final market data."""
    if not confirmation.get('evaluated') or not confirmation.get('proceed_to_final_test'):
        raise PermissionError('Fixed Q2 confirmation failed or did not run; Q3 remains sealed')
    if confirmation['system_freeze_sha256']!=digest(OUT/'system_freeze.json'):
        raise ValueError('Q2 belongs to a different frozen system')
    path=OUT/'confirmation_predictions.npz'
    if digest(path)!=confirmation['prediction_sha256']:raise ValueError('Q2 prediction artifact changed')
    f=fold(panel.dates,'2026Q2',confirmation=True);rows=panel.origins['AAPL']
    rows=rows[(panel.dates[rows]>=f['score_start'])&(panel.dates[rows]<=f['score_end'])]
    with np.load(path,allow_pickle=False) as saved:
        for key,expected in [('rows',rows),('dates',panel.dates[rows]),('target',panel.targets['AAPL'][rows])]:
            if not np.array_equal(saved[key],expected):raise ValueError('Q2 raw identity changed: '+key)
        target=saved['target'];prediction=saved['prediction'];expert=saved['experts']
        seed_report,ensemble=seed_system(target,saved['seeds'])
        if not np.array_equal(ensemble.astype(prediction.dtype),prediction):raise ValueError('Q2 ensemble changed')
        single=expert[:,NAMES.index(frozen['fixed_strongest_single']['family'])]
        static=simple.predict(confirmation['fixed_simple_fit'],expert)
        if not np.array_equal(single,saved['single']) or not np.array_equal(static,saved['simple']):
            raise ValueError('Q2 fixed control predictions changed')
        scores={'system':point_scores(target,prediction),'zero':point_scores(target,np.zeros(len(target))),
                'single':point_scores(target,single),'simple':point_scores(target,static)}
    if scores!=confirmation['scores'] or seed_report!=confirmation['seed_report']:
        raise ValueError('Q2 reported scores differ from its raw saved predictions')
    if not all(scores['system']['mse']<scores[k]['mse'] for k in ('zero','single','simple')):
        raise PermissionError('Recomputed Q2 gate failed; Q3 remains sealed')


def entry(deployment):
    """All inputs here are development, Q2, or metadata; no Q3 values are opened."""
    _,frozen=checked_freeze()
    if not frozen['development_gate_passed']:raise PermissionError('Development gate failed; Q3 remains sealed')
    panel=Panel.load();selected,_,_,_=recheck_seed_predictions(panel,frozen['selected_system'])
    controls,static=recheck_controls(panel)
    mse=selected['system']['mse'];zero=selected['system']['zero_mse']
    gates=[1-mse/zero>=.01,mse<controls[0]['mse'],mse<static['mse'],
           sum(s['mse_improvement_vs_zero']>0 for s in selected['quarter_scores'].values())>=2,
           all(s['mse_improvement_vs_zero']>0 for s in selected['individual_seeds'].values())]
    if not all(gates) or controls[0]!=frozen['fixed_strongest_single'] or static!=frozen['fixed_strongest_simple']:
        raise PermissionError('Recomputed frozen development gate or controls differ')
    confirmation=read_json(OUT/'confirmation.json');recheck_confirmation(panel,frozen,confirmation)
    deployment=Path(deployment).resolve();expected=(OUT/frozen['final_fit_policy']['artifact']).resolve()
    if deployment!=expected or frozen['final_fit_policy']['retrain_after_Q2']:
        raise ValueError('Use the exact pre-Q2 artifact; no post-confirmation retraining')
    manifest=read_json(deployment/'deployment.json')
    if manifest['freeze_sha256']!=digest(OUT/'system_freeze.json') or manifest['identity']!=confirmation['deployment_identity']:
        raise ValueError('Final artifact differs from the one actually confirmed in Q2')
    if identity({k:v for k,v in manifest.items() if k!='identity'})!=manifest['identity']:
        raise ValueError('Deployment manifest changed')
    for name,row in manifest['files'].items():
        path=(deployment/name).resolve()
        if not path.is_relative_to(deployment) or digest(path)!=row['sha256']:raise ValueError('Final fitted model file changed')
    inventory=read_json(OUT/'inventory.json');legacy=ROOT/'holdout_manifest.json'
    if digest(legacy)!=inventory['Q3_isolation']['old_AAPL_manifest_sha256']:
        raise PermissionError('Original sealed identity changed; independent scope needs review')
    # Inspect access-event metadata, never the original sealed CSV or statistics.
    if any(read_json(p)['kind']=='authorized_read' for p in (ROOT/'outputs/holdout_access').glob('*.json')):
        raise PermissionError('Prior model access to Q3 is recorded; do not claim an independent final test')
    sources=['system_freeze.json','confirmation.json','confirmation_predictions.npz','data_admission.json','data_audit.json']
    claim={'system_identity':manifest['identity'],'deployment_manifest_sha256':digest(deployment/'deployment.json'),
           'input_artifacts':{name:digest(OUT/name) for name in sources},'legacy_manifest_sha256':digest(legacy),
           'statistics':frozen['statistics'],'dates':[START,END],
           'symbols':list(dict.fromkeys(['AAPL']+manifest['background'])),
           'authorization_basis':'User instruction to execute the implementation plan; all new conditional gates verified',
           'legacy_afternoon_holdout_opened':False,'training_after_confirmation':False}
    claim['identity']=identity(claim)
    path=FINAL/'entry.json'
    if path.exists():
        previous=read_json(path)
        if previous['claim']!=claim:raise PermissionError('One final evaluation already belongs to a different frozen system')
    else:write_json(path,{'claim':claim,'authorized_at_utc':utc_now(),'Q3_values_read_at_claim':False})
    return claim,frozen,confirmation,manifest


def assert_entry(claim,require_preflight=True):
    saved=read_json(FINAL/'entry.json')
    if saved['claim']!=claim or identity({k:v for k,v in claim.items() if k!='identity'})!=claim['identity']:
        raise PermissionError('All-day final-entry authorization differs')
    for name,sha in claim['input_artifacts'].items():
        if digest(OUT/name)!=sha:raise PermissionError('A frozen final-entry input changed: '+name)
    # Recheck the source/plan lock even on a resumed acquisition.
    checked_freeze()
    if require_preflight:
        preflight=read_json(FINAL/'local_preflight.json')
        if preflight['entry_identity']!=claim['identity'] or not preflight['passed']:
            raise PermissionError('Local loading/numerical preflight has not passed; Q3 remains unread')


def local_preflight(system,claim,confirmation,batch_size):
    """Load and compare the real CPU artifact on known Q2 before any final data IO."""
    assert_entry(claim,require_preflight=False)
    if system.model_identity!=claim['system_identity']:raise ValueError('Local preflight loaded another artifact')
    panel=Panel.load();path=OUT/'confirmation_predictions.npz'
    with np.load(path,allow_pickle=False) as saved:
        indices=np.unique(np.linspace(0,len(saved['rows'])-1,min(128,len(saved['rows'])),dtype=np.int64))
        rows=saved['rows'][indices].copy()
        expected={'prediction':saved['prediction'][indices].copy(),
                  'seed_predictions':saved['seeds'][:,indices].copy(),'expert_predictions':saved['experts'][indices].copy()}
    parts=[system.predict_origin(panel,rows[first:first+batch_size]) for first in range(0,len(rows),batch_size)]
    observed={key:np.concatenate([p[key] for p in parts],axis=1 if key=='seed_predictions' else 0) for key in expected}
    tolerance=max(1e-8,float(np.sqrt(confirmation['scores']['zero']['mse']))*1e-4)
    checks={key:bool(np.allclose(observed[key],values,rtol=1e-4,atol=tolerance)) for key,values in expected.items()}
    report={'entry_identity':claim['identity'],'system_identity':system.model_identity,'origins':len(rows),
            'data':'already evaluated, known Q2 only','absolute_tolerance':tolerance,'relative_tolerance':1e-4,
            'checks':checks,'maximum_absolute_difference':{key:float(np.max(np.abs(observed[key]-values))) for key,values in expected.items()},
            'passed':all(checks.values()),'Q3_values_read':False,'completed_at_utc':utc_now()}
    write_json(FINAL/'local_preflight.json',report)
    if not report['passed']:raise RuntimeError('CPU/cloud prediction parity failed before opening Q3')
    return report


def final_request(params,claim,key,secret):
    assert_entry(claim)
    low,high=bounds(START,END)
    if params['start']!=low or params['end']!=high or params['symbols'] not in claim['symbols']:
        raise PermissionError('Final request exceeds its exact frozen scope')
    opener=build_opener(NoRedirect)
    for attempt in range(5):
        request=Request(ENDPOINT+'?'+urlencode(params),headers={
            'APCA-API-KEY-ID':key,'APCA-API-SECRET-KEY':secret,'User-Agent':'Fixed-all-day-final-evaluation/1.0'})
        try:
            with opener.open(request,timeout=45) as response:
                if response.url.split('?')[0]!=ENDPOINT:raise RuntimeError('Unexpected final data endpoint')
                body=response.read(30_000_001)
                if len(body)>30_000_000:raise RuntimeError('Final page exceeds bounded size')
                return body
        except HTTPError as exc:
            if (exc.code==429 or exc.code>=500) and attempt<4:time.sleep(min(30,3*(attempt+1)));continue
            raise RuntimeError(f'Final data request rejected HTTP {exc.code}; no feed fallback; body suppressed') from None
        except (URLError,TimeoutError):
            if attempt<4:time.sleep(min(20,2**attempt));continue
            raise RuntimeError('Final data network retries exhausted') from None


def acquire_symbol(symbol,claim):
    assert_entry(claim)
    if symbol not in claim['symbols']:raise PermissionError('Stock outside frozen final inputs')
    directory=FINAL/'raw'/symbol;directory.mkdir(parents=True,exist_ok=True);path=directory/'receipt.json'
    receipt=read_json(path) if path.exists() else {'symbol':symbol,'entry_identity':claim['identity'],'pages':[],'complete':False}
    if receipt['entry_identity']!=claim['identity']:raise PermissionError('Final raw data belongs to another entry')
    records=[];token=None;last=None;seen=set();low,high=bounds(START,END)
    def consume(body):
        nonlocal token,last
        data=json.loads(body)
        if set(data['bars'])-{symbol}:raise ValueError('Unexpected stock in final response')
        for b in data['bars'].get(symbol,[]) or []:
            stamp=pd.Timestamp(b['t'])
            if not pd.Timestamp(low)<=stamp<=pd.Timestamp(high) or (last is not None and stamp<=last):
                raise ValueError('Final raw timestamps exceed scope or are repeated/unordered')
            last=stamp;records.append((b['t'],b['o'],b['h'],b['l'],b['c'],b['v'],b['n'],b['vw']))
        token=data.get('next_page_token')
        if token and token in seen:raise ValueError('Repeated final pagination token')
        if token:seen.add(token)
    for i,row in enumerate(receipt['pages']):
        page=directory/row['file']
        if i!=row['page'] or digest(page)!=row['sha256']:raise ValueError('Final raw page changed')
        with gzip.open(page,'rb') as stream:consume(stream.read())
    if receipt['complete'] and token:raise ValueError('Completed final receipt retains pagination token')
    if not receipt['complete']:
        # Credentials are loaded only after all gates have passed, never bundled.
        key,secret=credentials()
        for i in range(len(receipt['pages']),100):
            if receipt['pages'] and token is None:break
            params={'symbols':symbol,'start':low,'end':high,'feed':'sip','adjustment':'raw',
                    'timeframe':'1Min','limit':10000,'sort':'asc'}
            if token:params['page_token']=token
            body=final_request(params,claim,key,secret);consume(body);page=directory/f'page_{i:04d}.json.gz'
            with gzip.open(page,'wb',compresslevel=3) as stream:stream.write(body)
            receipt['pages'].append({'file':page.name,'page':i,'sha256':digest(page),'received_at_utc':utc_now()})
            receipt['complete']=token is None;write_json(path,receipt)
            if not token:break
            time.sleep(.45)
        if not receipt['complete']:
            if token is None:receipt['complete']=True;write_json(path,receipt)
            else:raise RuntimeError('Final page count exceeded its guard')
    frame=pd.DataFrame.from_records(records,columns=['timestamp',*FIELDS])
    frame.index=pd.DatetimeIndex(pd.to_datetime(frame.pop('timestamp'),utc=True))
    if frame.empty:raise ValueError('No final bars were acquired; no substitute period allowed')
    return frame.astype(np.float64)


def extend_features(prefix_features,prefix_observed,previous_close,raw,stamps,starts):
    """Match a full-history transform using a fixed development prefix, without refitting."""
    aligned=raw.reindex(stamps)
    anchor=pd.DataFrame([{k:(0. if k in ('volume','trade_count') else previous_close) for k in FIELDS}],
                        index=[stamps[0]-pd.Timedelta(minutes=1)])
    feature,observed,close,_=causal_transform(pd.concat([anchor,aligned]),np.r_[False,starts])
    return np.concatenate([prefix_features,feature[1:]]),np.r_[prefix_observed,observed[1:]],close[1:]


def load_panel(claim,manifest):
    assert_entry(claim)
    grids=sessions(START,END);stamps=pd.DatetimeIndex(np.concatenate([g.as_unit('ns').asi8 for g in grids.values()]),tz='UTC')
    dates=stamps.tz_convert('America/New_York').strftime('%Y-%m-%d').to_numpy(dtype='U10')
    starts=np.r_[True,dates[1:]!=dates[:-1]]
    development=OUT/'data';audit=read_json(OUT/'data_audit.json')
    if digest(development/'timestamps_ns.npy')!=audit['global_files_sha256']['timestamps_ns.npy']:
        raise ValueError('Audited development calendar changed')
    prior_stamps=pd.to_datetime(np.load(development/'timestamps_ns.npy',mmap_mode='r')[-241:],utc=True)
    all_stamps=prior_stamps.append(stamps);all_dates=all_stamps.tz_convert('America/New_York').strftime('%Y-%m-%d').to_numpy(dtype='U10')
    boundaries=np.r_[np.flatnonzero(all_dates[1:]!=all_dates[:-1]),len(all_dates)-1]
    ends=np.repeat(boundaries,np.diff(np.r_[-1,boundaries]))
    features={};observed={};origins={};targets={};source_hashes={};coverage={}
    for symbol in claim['symbols']:
        if digest(development/symbol/'manifest.json')!=audit['symbols'][symbol]['manifest_sha256']:
            raise ValueError('Audited development stock manifest changed')
        receipt=read_json(development/symbol/'manifest.json')
        for field in ('features.npy','observed.npy','close.npy'):
            if digest(development/symbol/field)!=receipt['files'][field]:raise ValueError('Fixed development context changed')
        raw=acquire_symbol(symbol,claim);prefix=np.load(development/symbol/'features.npy',mmap_mode='r')[-241:]
        prior_observed=np.load(development/symbol/'observed.npy',mmap_mode='r')[-241:]
        previous_close=float(np.load(development/symbol/'close.npy',mmap_mode='r')[-1])
        features[symbol],observed[symbol],close=extend_features(prefix,prior_observed,previous_close,raw,stamps,starts)
        source_hashes[symbol]=digest(FINAL/'raw'/symbol/'receipt.json');coverage[symbol]=float(observed[symbol][241:].mean())
        candidate=eligible_origins(observed[symbol],ends);origins[symbol]=candidate[candidate>=241]
        targets[symbol]=np.full(len(all_dates),np.nan,dtype=np.float32)
        if symbol=='AAPL':
            row=origins[symbol]-241;targets[symbol][origins[symbol]]=np.log(close[row+12]/close[row]).astype(np.float32)
    common=origins['AAPL']
    for name in manifest['background']:
        mask=observed[name];missing=np.r_[0,np.cumsum(~mask)]
        common=common[mask[common]&(missing[common+1]-missing[common-239]<=POLICY['maximum_missing_context_bars'])]
    origins['AAPL']=common
    if set(all_dates[common].astype('U7'))!={'2026-07','2026-08','2026-09'}:
        raise ValueError('The fixed complete three-month final scope is unavailable; do not substitute a subset')
    write_json(FINAL/'data_receipt.json',{'entry_identity':claim['identity'],'source_receipt_sha256':source_hashes,
               'observed_coverage':coverage,'common_origins':len(common),'sessions':len(np.unique(all_dates[common])),
               'origin_identity':identity({'bar_start_ns':all_stamps[common].asi8.tolist()}),
               'eligibility_policy':POLICY,'outcome_based_exclusion':False,'completed_at_utc':utc_now()})
    return Panel(manifest['stocks'],manifest['background'],all_dates,all_stamps,features,observed,origins,targets)


def run(deployment):
    if os.environ.get('FUSION_EXECUTION_PROFILE','local')!='local':raise RuntimeError('Final data and evaluation stay local')
    device=runtime(2026)
    if device!='cpu':raise RuntimeError('Final evaluation must run locally')
    with exclusive():
        claim,frozen,confirmation,manifest=entry(deployment)
        report_path=FINAL/'report.json'
        if report_path.exists():raise ValueError('The one fixed final evaluation already completed; no replacement or rescoring')
        config=frozen['selected_system']['config']
        from .inference import FrozenSystem
        system=FrozenSystem(deployment,device);local_preflight(system,claim,confirmation,config['batch_size'])
        panel=load_panel(claim,manifest);rows=panel.origins['AAPL'];guard=ResourceGuard();parts=[]
        for start in range(0,len(rows),config['batch_size']):
            selected_rows=rows[start:start+config['batch_size']];path=FINAL/'prediction_chunks'/f'{start:08d}.npz';receipt=path.with_suffix('.json')
            if receipt.exists():
                saved=read_json(receipt)
                if saved['entry_identity']!=claim['identity'] or digest(path)!=saved['sha256']:raise ValueError('Final prediction checkpoint changed')
                with np.load(path,allow_pickle=False) as values:output={key:values[key].copy() for key in values.files if key!='rows'};stored_rows=values['rows'].copy()
                if not np.array_equal(selected_rows,stored_rows):raise ValueError('Final prediction checkpoint origins changed')
            else:
                output=system.predict_origin(panel,selected_rows);path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path,rows=selected_rows,**output)
                write_json(receipt,{'entry_identity':claim['identity'],'sha256':digest(path),'generated_at_utc':utc_now()})
            parts.append(output);guard.check()
        prediction=np.concatenate([p['prediction'] for p in parts]);seeds=np.concatenate([p['seed_predictions'] for p in parts],axis=1)
        experts=np.concatenate([p['expert_predictions'] for p in parts]);weights=np.concatenate([p['weights'] for p in parts])
        single=experts[:,NAMES.index(frozen['fixed_strongest_single']['family'])];static=simple.predict(confirmation['fixed_simple_fit'],experts)
        target=panel.targets['AAPL'][rows];decision=panel.stamps[rows]+pd.Timedelta(minutes=1)
        controls={'zero':np.zeros(len(rows)),'single':single,'simple':static}
        result=compare(target,prediction,controls,panel.dates[rows],decision)
        seed_report,ensemble=seed_system(target,seeds)
        if not np.array_equal(prediction,ensemble):raise ValueError('Final saved output differs from the frozen three-seed mean')
        months=panel.dates[rows].astype('U7');local=decision.tz_convert('America/New_York')
        minute=local.hour.to_numpy()*60+local.minute.to_numpy()
        masks={'opening_0931_1029':minute<630,'midday_1030_1429':(minute>=630)&(minute<870),'closing_1430_1548':minute>=870}
        seed_report['monthly']={str(seed):{m:point_scores(target[months==m],seeds[i,months==m]) for m in np.unique(months)}
                                for i,seed in enumerate((2026,2027,2028))}
        seed_report['time_of_day']={str(seed):{name:point_scores(target[mask],seeds[i,mask]) for name,mask in masks.items() if mask.any()}
                                    for i,seed in enumerate((2026,2027,2028))}
        prediction_path=FINAL/'predictions.npz'
        np.savez_compressed(prediction_path,rows=rows,bar_start_ns=panel.stamps[rows].asi8,decision_time_ns=decision.asi8,
                            target_completed_time_ns=(panel.stamps[rows+12]+pd.Timedelta(minutes=1)).asi8,
                            dates=panel.dates[rows],target=target,prediction=prediction,seeds=seeds,experts=experts,
                            weights=weights,single=single,simple=static)
        result.update(entry_identity=claim['identity'],system_identity=system.model_identity,seed_report=seed_report,
                      raw_prediction_sha256=digest(prediction_path),data_receipt_sha256=digest(FINAL/'data_receipt.json'),
                      mode='offline held-out historical evaluation; not timely historical trading',
                      receive_time='original live receipt timestamps unavailable; raw acquisition receipts are preserved',
                      Q2_status=frozen['Q2_status'],resources=guard.check(),legacy_holdout_unchanged=True,
                      completed_at_utc=utc_now())
        write_json(report_path,result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--deployment',type=Path,required=True);args=parser.parse_args()
    try:result=run(args.deployment)
    except Exception as exc:
        # Preserve failures; retries retain the same frozen entry and completed
        # prediction chunks. No failed eligible origin can be dropped for a score.
        write_json(FINAL/'failures'/f'{time.time_ns()}.json',{'error_type':type(exc).__name__,
                   'error':str(exc),'recorded_at_utc':utc_now(),'configuration_changes_allowed':False})
        raise
    print({'research_success':result['research_success'],'system':result['system']})
