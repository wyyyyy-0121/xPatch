"""All-day completed-bar windows, explicit sessions/gaps, causal missing-data rules."""
import argparse
import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from aapl.data import sessions,read_source
from .common import ROOT,OUT,PLAN,STOCKS,BACKGROUND,DEV_START,DEV_END,digest,read_json,write_json,utc_now,status

FIELDS=('open','high','low','close','volume','trade_count','vwap')
CHANNELS=('open_return','high_return','low_return','close_return','log_volume','range',
          'body','session_start','overnight_gap','missing','possible_action','close_only_return')
POLICY={'minimum_observed_regular_coverage':.995,'maximum_missing_context_bars':2,
        'labels_require_all12_future_bars_observed':True,'targets_within_same_session':True,
        'history_bars':240,'horizon':12,'missing_fill':'past close only; zero volume; explicit mask',
        'no_values_after':'2026-06-30T23:59:59.999999999-04:00',
        'market_peers':['MSFT','NVDA','GOOGL'],'no_outcome_based_symbol_substitution':True}


def calendar():
    grids=sessions(DEV_START,DEV_END)
    stamps=pd.DatetimeIndex(np.concatenate([g.as_unit('ns').asi8 for g in grids.values()]),tz='UTC')
    dates=stamps.tz_convert('America/New_York').strftime('%Y-%m-%d').to_numpy(dtype='U10')
    start=np.r_[True,dates[1:]!=dates[:-1]]
    first=np.maximum.accumulate(np.where(start,np.arange(len(dates)),0))
    last=np.r_[np.flatnonzero(start)[1:]-1,len(dates)-1]
    end=np.repeat(last,np.diff(np.r_[0,last+1]))
    return stamps,dates,start,first,end


def pages_frame(symbol):
    directory=OUT/'raw'/symbol;receipt=read_json(directory/'receipt.json')
    if not receipt['download_complete']:raise ValueError(f'{symbol} acquisition incomplete')
    records=[];verified=0;counts=0
    for entry in receipt['pages']:
        p=directory/entry['file']
        if digest(p)!=entry['sha256']:raise ValueError('Raw SIP page identity mismatch')
        with gzip.open(p,'rb') as f:payload=json.load(f)
        bars=payload['bars'].get(symbol,[]) or []
        if len(bars)!=entry['rows']:raise ValueError('Raw page row count differs')
        for b in bars:records.append((b['t'],b['o'],b['h'],b['l'],b['c'],b['v'],b['n'],b['vw']))
        counts+=len(bars);verified+=1
    if counts!=receipt['rows']:raise ValueError('Acquisition total count differs')
    frame=pd.DataFrame.from_records(records,columns=['timestamp',*FIELDS])
    index=pd.DatetimeIndex(pd.to_datetime(frame.pop('timestamp'),utc=True))
    if index.has_duplicates or not index.is_monotonic_increasing:raise ValueError('Raw minute ordering invalid')
    frame.index=index
    if index.max().tz_convert('America/New_York').date()>pd.Timestamp(DEV_END).date():
        raise ValueError('Raw development data crosses Q3 boundary')
    return frame.astype(np.float64),{'verified_pages':verified,'verified_rows':counts,
                                  'receipt_sha256':digest(directory/'receipt.json')}


def causal_transform(aligned,session_start):
    values=aligned.loc[:,FIELDS].to_numpy(dtype=np.float64,copy=True)
    finite=np.isfinite(values).all(axis=1)
    valid=finite & (values[:,:4]>0).all(axis=1) & (values[:,4]>=0)
    bounds=values[:,[0,2,3]]
    valid &= (values[:,1]>=np.max(np.where(np.isfinite(bounds),bounds,-np.inf),axis=1)) & (values[:,2]<=np.min(np.where(np.isfinite(bounds),bounds,np.inf),axis=1))
    values[~valid]=np.nan
    # Forward-only missing-bar policy, including a mask; never interpolate backwards.
    close=pd.Series(values[:,3]).ffill().to_numpy()
    good=np.isfinite(close)
    for c in (0,1,2,3):values[~valid,c]=close[~valid]
    values[~valid,4:]=0
    open_,high,low,current=values[:,:4].T
    previous=np.r_[np.nan,close[:-1]]
    anchor=np.where(session_start,open_,previous)
    safe=np.where((anchor>0)&np.isfinite(anchor),anchor,current)
    with np.errstate(divide='ignore',invalid='ignore'):
        gap=np.where(session_start,np.log(open_/previous),0.)
        raw_close=np.log(current/previous)
        r=np.column_stack([np.log(values[:,c]/safe) for c in (0,1,2,3)])
        span=np.log(high/low)
    gap=np.nan_to_num(gap,nan=0.);raw_close=np.nan_to_num(raw_close,nan=0.)
    action=(session_start&(np.abs(gap)>.25)).astype(np.float64)
    body=(current-open_)/(high-low+1e-8)
    features=np.column_stack([r,np.log1p(values[:,4]),span,body,session_start,
                              np.clip(gap,-.3,.3),~valid,action,np.clip(raw_close,-.3,.3)])
    features[~good]=0
    features[~good,9]=1
    features[~good,7]=session_start[~good]
    if not np.isfinite(features).all():raise ValueError('Nonfinite causal features after forward fill')
    return features.astype(np.float32),valid,close,action


def eligible_origins(observed,session_end,history=240):
    n=len(observed);idx=np.arange(n)
    # A 240-return context needs a predecessor completed bar; exclude first 240 rows.
    enough=idx>=history
    future_inside=session_end-idx>=12
    # Prefix sums enforce observed future endpoints and the complete 12-minute path.
    missing=np.r_[0,np.cumsum(~observed)]
    candidates=idx[enough&future_inside&observed]
    past_bad=missing[candidates+1]-missing[candidates-history+1]
    future_bad=missing[candidates+13]-missing[candidates+1]
    return candidates[(past_bad<=POLICY['maximum_missing_context_bars'])&(future_bad==0)]


def prepare_symbol(symbol,stamps,dates,start,end):
    directory=OUT/'data'/symbol;directory.mkdir(parents=True,exist_ok=True)
    receipt_path=directory/'manifest.json'
    if receipt_path.exists():
        old=read_json(receipt_path)
        for name,checksum in old['files'].items():
            if digest(directory/name)!=checksum:raise ValueError('Prepared shard changed')
        return old
    if symbol=='AAPL':
        protocol=read_json(ROOT/'protocol_locked.json');path=Path(protocol['source_csv_read_only'])
        if digest(path)!=protocol['source_csv_expected_sha256']:raise ValueError('AAPL source identity changed')
        raw=read_source(path);source={'source_csv_read_only':str(path),'sha256':digest(path),'rows':len(raw)}
    else:raw,source=pages_frame(symbol)
    aligned=raw.reindex(stamps)
    features,observed,close,actions=causal_transform(aligned,start)
    coverage=float(np.mean(observed));origins=eligible_origins(observed,end)
    y=np.full(len(stamps),np.nan,dtype=np.float32)
    y[origins]=np.log(close[origins+12]/close[origins]).astype(np.float32)
    arrays={'features.npy':features,'observed.npy':observed,'close.npy':close.astype(np.float64),
            'origins.npy':origins,'target.npy':y}
    for name,array in arrays.items():np.save(directory/name,array)
    by_quarter={}
    for period in pd.period_range('2021Q1','2026Q2',freq='Q'):
        q=str(period);mask=(dates>=str(period.start_time.date()))&(dates<=str(period.end_time.date()))
        by_quarter[q]=float(observed[mask].mean())
    admitted=coverage>=POLICY['minimum_observed_regular_coverage'] and all(c>=POLICY['minimum_observed_regular_coverage'] for c in by_quarter.values())
    possible_actions=[{'market_date':str(dates[i]),'raw_observed_gap_flag':True} for i in np.flatnonzero(actions)]
    report={'symbol':symbol,'source':source,'regular_rows':len(stamps),'observed_rows':int(observed.sum()),
            'coverage':coverage,'coverage_by_quarter':by_quarter,'admitted':admitted,
            'exclusion_reason':None if admitted else 'Predeclared coverage threshold failed; no replacement selected',
            'eligible_origins':len(origins),'features':list(CHANNELS),'missing_policy':POLICY,
            'possible_corporate_action_or_extreme_gap_flags':possible_actions,
            'files':{name:digest(directory/name) for name in arrays},'prepared_at_utc':utc_now(),
            'Q3_values_read':False}
    write_json(receipt_path,report)
    return report


def prepare_all(wait=False):
    OUT.mkdir(parents=True,exist_ok=True);status('F1_all_day_data_preparation')
    stamps,dates,start,first,end=calendar();directory=OUT/'data';directory.mkdir(exist_ok=True)
    for name,array in [('timestamps_ns.npy',stamps.as_unit('ns').asi8),('dates.npy',dates),
                       ('session_start.npy',start),('session_end.npy',end)]:np.save(directory/name,array)
    write_json(OUT/'data_policy.json',{'fixed_before_coverage_scoring_at_utc':utc_now(),'policy':POLICY,
                                     'stocks':list(STOCKS),'background':list(BACKGROUND),'plan_sha256':digest(PLAN)})
    reports={}
    pending=list(STOCKS+BACKGROUND)
    import time
    while pending:
        for symbol in list(pending):
            source=OUT/'raw'/symbol/'receipt.json'
            if symbol!='AAPL' and (not source.exists() or not read_json(source).get('download_complete')):continue
            status('F1_all_day_data_preparation',symbol=symbol,remaining=len(pending))
            reports[symbol]=prepare_symbol(symbol,stamps,dates,start,end);pending.remove(symbol)
            write_json(OUT/'prepared_symbols.json',{'symbols':reports,'pending':pending,'updated_at_utc':utc_now()})
        if not pending:break
        if not wait:return
        failures=[s for s in pending if (OUT/'data_failures'/f'{s}.json').exists()]
        if failures:raise RuntimeError('Historical acquisition failed for required symbols: '+','.join(failures))
        time.sleep(5)
    admitted=[s for s in STOCKS if reports[s]['admitted']]
    if 'AAPL' not in admitted or not all(reports[s]['admitted'] for s in BACKGROUND):
        raise RuntimeError('AAPL or required background failed coverage admission')
    # All algorithms use the same AAPL origins and background availability.
    common=np.load(directory/'AAPL/origins.npy')
    for symbol in BACKGROUND+tuple(s for s in POLICY['market_peers'] if s in admitted):
        observed=np.load(directory/symbol/'observed.npy')
        bad=np.r_[0,np.cumsum(~observed)]
        valid=observed[common] & (bad[common+1]-bad[common-239]<=POLICY['maximum_missing_context_bars'])
        common=common[valid]
    np.save(directory/'common_aapl_origins.npy',common)
    audit_examples=[]
    for i in common:
        local=stamps[i].tz_convert('America/New_York')
        if local.hour==9 and local.minute==30:
            audit_examples.append({'origin_bar_start':stamps[i].isoformat(),'decision_time':(stamps[i]+pd.Timedelta(minutes=1)).isoformat(),
                  'context_first_bar':stamps[i-239].isoformat(),'context_crosses_session':dates[i-239]!=dates[i],
                  'label_target_completed_at':(stamps[i+12]+pd.Timedelta(minutes=1)).isoformat()})
            if len(audit_examples)==3:break
    report={'passed':True,'completed_at_utc':utc_now(),'admitted_training_stocks':admitted,
            'excluded_stocks':[s for s in STOCKS if s not in admitted],'background':list(BACKGROUND),
            'market_peers':[s for s in POLICY['market_peers'] if s in admitted],
            'common_AAPL_origins':len(common),'calendar_sessions':len(set(dates)),
            'decision_time_range_normal_session':'09:31-15:48 ET','early_close_labels_stay_in_session':True,
            'overnight_returns_separated_from_intraday':True,'examples':audit_examples,'policy':POLICY,
            'all_symbols_Q3_blocked':True,'plan_sha256':digest(PLAN),'origin_file_sha256':digest(directory/'common_aapl_origins.npy')}
    write_json(OUT/'data_admission.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    prepare_all(args.wait)
