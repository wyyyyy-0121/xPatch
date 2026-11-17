"""CPU JSON-lines interface for completed bars and one-origin frozen predictions."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from aapl.data import sessions
from .common import OUT,write_json,utc_now
from .hardware import runtime,ResourceGuard
from .inference import FrozenSystem
from .stream import Stream


def process_event(stream,event,mode):
    if not isinstance(event,dict):raise ValueError('Each event must be a JSON object')
    if not isinstance(event.get('bar_start'),str):raise ValueError('bar_start must be an explicit timestamp string')
    kind=event.get('type');stamp=pd.Timestamp(event.get('bar_start'))
    if stamp.tz is None:raise ValueError('bar_start requires an explicit timezone')
    date=stamp.tz_convert('America/New_York').strftime('%Y-%m-%d')
    # Generic deployment must not bypass the new research's dedicated final entry.
    if '2026-07-01'<=date<='2026-09-30':raise PermissionError('The protected final period is available only through its controlled evaluation entry')
    if kind=='bar':
        if set(event)!={'type','symbol','bar_start','received_at','values'}:raise ValueError('Bar events require symbol, bar_start, received_at and OHLCV values')
        return stream.ingest(event['symbol'],stamp,event['values'],event['received_at'])
    if kind=='predict':
        if set(event)-{'type','bar_start','generated_at'}:raise ValueError('Unexpected prediction fields')
        return stream.predict(stamp,event.get('generated_at'),mode)
    raise ValueError('Event type must be bar or predict')


def run(deployment,start,end,state=None,resume=False,mode='delayed_historical_replay',input_stream=None,output_stream=None):
    if runtime(2026)!='cpu':raise RuntimeError('This deployment entry is assigned to local CPU inference')
    predictor=FrozenSystem(deployment,'cpu');guard=ResourceGuard();guard.check()
    grids=sessions(start,end);calendar=pd.DatetimeIndex(np.concatenate([g.as_unit('ns').asi8 for g in grids.values()]),tz='UTC')
    config=predictor.manifest['system_checkpoints'][0]['config'];scalers=predictor.models[0][1]
    state=Path(state) if state else OUT/'streams'/predictor.model_identity/'state.json'
    stream=Stream(calendar,predictor.stocks,predictor.background,config,scalers,predictor,predictor.model_identity,state)
    if state.exists() and not resume:raise ValueError('Stream state already exists; use --resume to preserve the prediction journal')
    if resume:stream.resume()
    errors=0;events=0;failed_forecasts=0
    for line in input_stream or sys.stdin:
        if not line.strip():continue
        events+=1
        try:result=process_event(stream,json.loads(line),mode);guard.check()
        except Exception as exc:
            errors+=1;result={'status':'event_rejected','event_index':events,'error_type':type(exc).__name__,'reason':str(exc),
                              'recorded_at_utc':utc_now(),'model_identity':predictor.model_identity}
            write_json(state.parent/'input_failures'/f'{time.time_ns()}.json',result)
        if result['status']=='input_or_model_failure' or (result['status']=='already_generated' and 'reason' in result):failed_forecasts+=1
        print(json.dumps(result,ensure_ascii=False,allow_nan=False),file=output_stream or sys.stdout,flush=True)
    stream.save()
    return {'events':events,'rejected_events':errors,'failed_forecasts':failed_forecasts,'model_identity':predictor.model_identity}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--deployment',type=Path,required=True)
    parser.add_argument('--calendar-start',required=True);parser.add_argument('--calendar-end',required=True)
    parser.add_argument('--state',type=Path);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--mode',choices=('delayed_historical_replay','timely_forecast'),default='delayed_historical_replay')
    args=parser.parse_args();result=run(args.deployment,args.calendar_start,args.calendar_end,args.state,args.resume,args.mode)
    print(json.dumps(result),file=sys.stderr)
    if result['rejected_events'] or result['failed_forecasts']:raise SystemExit(1)
