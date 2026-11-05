"""Completed-bar prediction/replay engine with durable state and explicit data timing."""
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ROOT,digest,identity,read_json,write_json,utc_now
from .data import FIELDS,causal_transform,POLICY
from .panel import Panel


class Stream:
    """The predictor receives features only. Its implementation owns frozen weights."""
    def __init__(self,calendar,stocks,background,config,scalers,predictor,model_identity,state_path):
        self.calendar=pd.DatetimeIndex(calendar)
        if self.calendar.tz is None or self.calendar.has_duplicates or not self.calendar.is_monotonic_increasing:
            raise ValueError('Calendar requires ordered unique timezone-aware bar starts')
        self.stocks=list(stocks);self.background=list(background);self.names=list(dict.fromkeys(['AAPL']+self.background))
        if 'AAPL' not in self.stocks:raise ValueError('The fixed primary stock is AAPL')
        self.config=config;self.scalers=scalers;self.predictor=predictor;self.model_identity=model_identity
        self.state_path=Path(state_path);self.bars={name:{} for name in self.names};self.outputs={}
        self.recipe={'model_identity':model_identity,'config':config,'stocks':self.stocks,'background':self.background,
                     'calendar_identity':identity(self.calendar.as_unit('ns').asi8.tolist())}

    def ingest(self,symbol,bar_start,values,received_at):
        if symbol not in self.names:raise ValueError('Input symbol outside the fixed deployed information set')
        stamp=pd.Timestamp(bar_start);received=pd.Timestamp(received_at)
        if stamp.tz is None or received.tz is None:raise ValueError('Input times require explicit time zones')
        stamp=stamp.tz_convert('UTC');received=received.tz_convert('UTC')
        if stamp not in self.calendar:raise ValueError('Bar is outside regular-session calendar')
        if stamp+pd.Timedelta(minutes=1)>received:raise ValueError('Bar was not completed when received')
        required={'open','high','low','close','volume'}
        if not required<=set(values) or not set(values)<=set(FIELDS):raise ValueError('Input requires OHLCV; only trade_count and vwap may be added')
        # Ancillary fields are not model features. Missing values have causal
        # placeholders so an OHLCV-only feed need not invent a trade count/VWAP.
        normalized={**values,'trade_count':values.get('trade_count',0.),'vwap':values.get('vwap',values['close'])}
        record={'values':{name:float(normalized[name]) for name in FIELDS},'received_at':received.isoformat(),
                'ancillary_fields_unavailable':[name for name in ('trade_count','vwap') if name not in values]}
        if not all(np.isfinite(v) for v in record['values'].values()):raise ValueError('Nonfinite input bar')
        o,h,l,c,v,n,vw=[record['values'][name] for name in FIELDS]
        if min(o,h,l,c,vw)<=0 or v<0 or n<0 or h<max(o,c,l) or l>min(o,c):raise ValueError('Invalid or inconsistent OHLCV bar')
        key=stamp.isoformat()
        if key in self.bars[symbol]:
            previous=self.bars[symbol][key]
            if previous['values']!=record['values']:raise ValueError('Conflicting duplicate bar; frozen prediction history is not rewritten')
            return {'status':'duplicate_ignored','symbol':symbol,'bar_start':key}
        # Old re-deliveries must not evict newer history.
        if self.bars[symbol]:
            latest=max(pd.Timestamp(k) for k in self.bars[symbol])
            if stamp<latest-pd.Timedelta(days=10):raise ValueError('Input is older than the retained replay history')
        self.bars[symbol][key]=record
        keys=sorted(self.bars[symbol])
        for old in keys[:-242]:del self.bars[symbol][old]
        return {'status':'accepted','symbol':symbol,'bar_start':key}

    def _panel(self,bar_start):
        stamp=pd.Timestamp(bar_start).tz_convert('UTC');position=self.calendar.get_indexer([stamp])[0]
        if position<240:raise ValueError('Need at least 241 completed regular-session bars')
        index=self.calendar[position-240:position+1]
        dates=index.tz_convert('America/New_York').strftime('%Y-%m-%d').to_numpy(dtype='U10')
        starts=np.r_[True,dates[1:]!=dates[:-1]]
        features={};observed={};received=[];missing={}
        for name in self.names:
            records=[]
            for time in index:
                record=self.bars[name].get(time.isoformat())
                records.append([record['values'][field] for field in FIELDS] if record else [np.nan]*len(FIELDS))
                if record:received.append(pd.Timestamp(record['received_at']))
            frame=pd.DataFrame(records,index=index,columns=FIELDS)
            # The predecessor may itself be missing in an otherwise admissible
            # 240-bar context. Seed its causal fill with the last older received
            # close, just as the full-history batch transform does.
            older=[k for k in self.bars[name] if pd.Timestamp(k)<index[0]]
            if older:
                anchor_key=max(older);anchor=self.bars[name][anchor_key]
                anchor_time=pd.Timestamp(anchor_key)
                prefix=pd.DataFrame([[anchor['values'][field] for field in FIELDS]],index=[anchor_time],columns=FIELDS)
                extended=pd.concat([prefix,frame]);extended_dates=extended.index.tz_convert('America/New_York').date
                extended_start=np.r_[True,extended_dates[1:]!=extended_dates[:-1]]
                values,obs,close,_=causal_transform(extended,extended_start)
                values=values[1:];obs=obs[1:];close=close[1:];received.append(pd.Timestamp(anchor['received_at']))
            else:values,obs,close,_=causal_transform(frame,starts)
            missing[name]=int(np.sum(~obs[-240:]))
            if not obs[-1] or missing[name]>POLICY['maximum_missing_context_bars'] or not np.isfinite(close[0]):
                raise ValueError('Missing current input, predecessor, or excessive missing history for '+name)
            features[name]=values;observed[name]=obs
        # Inference Panel never reads these placeholders; labels do not exist in this engine.
        panel=Panel(self.stocks,self.background,dates,index,features,observed,{}, {})
        return panel,missing,max(received)

    def predict(self,bar_start,generated_at=None,mode='delayed_historical_replay'):
        if mode not in ('delayed_historical_replay','timely_forecast'):raise ValueError('Unknown timing mode')
        stamp=pd.Timestamp(bar_start)
        if stamp.tz is None:raise ValueError('Bar time requires explicit zone')
        stamp=stamp.tz_convert('UTC');generated=pd.Timestamp(generated_at or utc_now())
        if generated.tz is None:raise ValueError('Generation time requires explicit zone')
        key=stamp.isoformat()
        if key in self.outputs:return {**self.outputs[key],'status':'already_generated'}
        decision=stamp+pd.Timedelta(minutes=1)
        position=self.calendar.get_indexer([stamp])[0]
        if position<0:raise ValueError('Unknown regular-session origin')
        target_position=position+12
        if target_position>=len(self.calendar) or self.calendar[target_position].tz_convert('America/New_York').date()!=stamp.tz_convert('America/New_York').date():
            raise ValueError('Twelve-minute target would cross session close')
        if generated<decision:raise ValueError('Prediction timestamp precedes completion of its input')
        try:
            panel,missing,last_received=self._panel(stamp)
            if last_received>generated:raise ValueError('A required input was received after generation time')
            if mode=='timely_forecast' and last_received>decision+pd.Timedelta(seconds=60):
                raise ValueError('Delayed inputs cannot be labeled a timely market-time forecast')
            if mode=='timely_forecast' and generated>decision+pd.Timedelta(seconds=60):
                raise ValueError('Late generation cannot be labeled a timely market-time forecast')
            if hasattr(self.predictor,'predict_origin'):
                full=self.predictor.predict_origin(panel,[240])
                result={'prediction':float(full['prediction'][0]),'weights':full['weights'][0]}
            else:
                batch=panel.windows([self.stocks.index('AAPL')],[240],self.config,self.scalers,include_labels=False)
                result=self.predictor(batch)
            prediction=float(result['prediction']);weights=np.asarray(result.get('weights',[]),dtype=float)
            if not np.isfinite(prediction) or not np.isfinite(weights).all():raise ValueError('Nonfinite frozen-model output')
            output={'status':'ok','symbol':'AAPL','horizon_minutes':12,'predicted_log_return':prediction,
                    'model_identity':self.model_identity,'bar_start':key,'input_cutoff':decision.isoformat(),
                    'market_prediction_origin':decision.isoformat(),
                    'target_market_time':(self.calendar[target_position]+pd.Timedelta(minutes=1)).isoformat(),
                    'latest_required_input_received_at':last_received.isoformat(),
                    'generated_at':generated.isoformat(),'mode':mode,
                    'input_integrity':{'missing_context_bars_by_symbol':missing},'expert_weights':weights.tolist()}
        except Exception as exc:
            output={'status':'input_or_model_failure','symbol':'AAPL','bar_start':key,'input_cutoff':decision.isoformat(),
                    'generated_at':generated.isoformat(),'mode':mode,'model_identity':self.model_identity,
                    'error_type':type(exc).__name__,'reason':str(exc)}
        # Keep the full prediction journal, including failures; never silently drop failed origins.
        output['emitted_at_wall_clock_utc']=utc_now()
        self.outputs[key]=output;self.save();return output

    def save(self):
        body={'schema':2,'recipe':self.recipe,'bars':self.bars,'outputs':self.outputs}
        # One atomic publication commits data and integrity together. There is no
        # separate receipt update in which an interrupted write can split them.
        write_json(self.state_path,{**body,'integrity':identity(body)})

    def resume(self):
        state=read_json(self.state_path)
        if state.get('schema')!=2 or state.get('integrity')!=identity({k:v for k,v in state.items() if k!='integrity'}):
            raise ValueError('Replay-state checksum or schema differs')
        if state['recipe']!=self.recipe:raise ValueError('Cannot resume replay with a changed model/configuration/calendar')
        self.bars=state['bars'];self.outputs=state['outputs']
