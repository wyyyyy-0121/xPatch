"""Audit public vendor sample. Never declare formal source verification."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import exchange_calendars as xc

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
root=args.run.resolve()
frame=pd.read_csv(root/'AAPL_1min_sample.csv')
required=['timestamp','open','high','low','close','volume']
assert list(frame.columns)==required, 'Unexpected CSV schema'
idx=pd.DatetimeIndex(pd.to_datetime(frame.timestamp,errors='raise')).tz_localize('America/New_York',ambiguous='raise',nonexistent='raise')
values=frame[['open','high','low','close','volume']].to_numpy(float)
invalid_numeric=~np.isfinite(values).all(axis=1)
invalid_price=(frame[['open','high','low','close']]<=0).any(axis=1)
invalid_bounds=(frame.high<frame[['open','close','low']].max(axis=1))|(frame.low>frame[['open','close','high']].min(axis=1))
assert not idx.duplicated().any(), 'Duplicate timestamps; preserve input for investigation'
assert idx.is_monotonic_increasing,'Unsorted timestamps'
assert not invalid_numeric.any() and not invalid_price.any() and not invalid_bounds.any(), 'Invalid numeric/OHLC input'
assert not (frame.volume<0).any(),'Negative volume'
calendar=xc.get_calendar('XNYS',start=idx.min().date(),end=idx.max().date())
sessions=calendar.schedule.loc[str(idx.min().date()):str(idx.max().date())]
parts=[];daily=[];windows=0
for label,row in sessions.iterrows():
    expected=pd.date_range(row.open,row.close,freq='min',inclusive='left').tz_convert('America/New_York')
    actual=idx[idx.isin(expected)]
    minute_set=set(actual)
    per_day_windows=0
    for origin_pos in range(240,len(expected)-12):
        if all(t in minute_set for t in expected[origin_pos-240:origin_pos+13]):
            per_day_windows+=1
    windows+=per_day_windows
    daily.append({'date':str(label.date()),'expected_minutes':len(expected),'observed_minutes':len(actual),'missing_minutes':len(expected.difference(actual)),'coverage':len(actual)/len(expected),'valid_240_returns_to_12_returns_windows':per_day_windows})
    parts.append(expected)
expected=parts[0]
for part in parts[1:]:expected=expected.append(part)
regular=idx.isin(expected)
adjacent=((idx[1:]-idx[:-1])==pd.Timedelta(minutes=1))&(np.asarray(idx.date)[1:]==np.asarray(idx.date)[:-1])
ret=np.diff(np.log(frame.close.to_numpy(float)))
regular_adj=adjacent&regular[1:]&regular[:-1]
rank=np.argsort(np.abs(ret[regular_adj]))[::-1][:5]
locations=np.flatnonzero(regular_adj)[rank]
top=[{'timestamp':idx[i+1].isoformat(),'previous_close':float(frame.close.iloc[i]),'close':float(frame.close.iloc[i+1]),'log_return':float(ret[i])} for i in locations]
normalized=frame.copy();normalized['timestamp']=[t.isoformat() for t in idx]
output=root/'AAPL_1min_sample_timezone_normalized.csv'
normalized_bytes=normalized.to_csv(index=False).encode()
if output.exists():
    assert output.read_bytes()==normalized_bytes, 'Existing normalized output differs'
else:
    with output.open('xb') as stream:stream.write(normalized_bytes)
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
report={'status':'provider_sample_structural_audit_complete','formal_eligible':False,'audited_utc':datetime.now(timezone.utc).isoformat(),'symbol':'AAPL','raw_csv_sha256':sha(root/'AAPL_1min_sample.csv'),'normalized_csv_sha256':sha(output),'timezone':'America/New_York','timestamp_label':'bar_start','adjustment':'Vendor sample page describes split-adjusted; ZIP readme lists three product variants but does not uniquely label this CSV. Not asserted unadjusted.','rows_all_sessions':len(frame),'start':idx.min().isoformat(),'end':idx.max().isoformat(),'regular_sessions':len(sessions),'regular_rows':int(regular.sum()),'expected_regular_rows':len(expected),'regular_coverage':float(regular.sum()/len(expected)),'minimum_session_coverage':min(d['coverage'] for d in daily),'missing_regular_rows':len(expected.difference(idx)),'outside_regular_rows':int((~regular).sum()),'zero_volume_rows':int((frame.volume==0).sum()),'invalid_numeric_rows':int(invalid_numeric.sum()),'invalid_price_rows':int(invalid_price.sum()),'invalid_ohlc_bounds_rows':int(invalid_bounds.sum()),'duplicate_timestamps':int(idx.duplicated().sum()),'max_abs_adjacent_regular_log_return':float(np.abs(ret[regular_adj]).max()),'adjacent_regular_abs_log_return_over_log1p05':int((np.abs(ret[regular_adj])>np.log(1.05)).sum()),'valid_240_returns_to_12_returns_windows':windows,'top_regular_jumps':top,'daily':daily,'limitations':[f'Only {len(sessions)} sessions; insufficient for full development or six-month history gate.','No second-provider price cross-check.','Vendor performs end-of-day anomaly checks/corrections; point-in-time causality unverified.','No DST switch or half-day present in actual sample.','Official free page dates/schema differ from this ZIP.','Raw data redistribution restricted; local evaluation only.','Overlaps previously inspected Clean dates; not independent confirmation.'],'software':{'numpy':np.__version__,'pandas':pd.__version__,'exchange_calendars':xc.__version__}}
(root/'quality_audit.json').write_text(json.dumps(report,indent=2)+chr(10))
pd.DataFrame(daily).to_csv(root/'session_coverage.csv',index=False)
(root/'sample_identity.json').write_text(json.dumps({'purpose':'local_provider_evaluation_only','source_verified':False,'formal_eligible':False,'source':'FirstRate Data public AAPL sample','provider_url':'https://firstratedata.com/i/stock/AAPL','csv_sha256':report['raw_csv_sha256'],'normalized_csv_sha256':report['normalized_csv_sha256'],'timezone':'America/New_York','timestamp_label':'bar_start','adjustment':report['adjustment'],'evidence':'download_receipts.json','audit':'quality_audit.json'},indent=2)+chr(10))
print(json.dumps({k:v for k,v in report.items() if k not in ('daily','top_regular_jumps','software')},indent=2))
