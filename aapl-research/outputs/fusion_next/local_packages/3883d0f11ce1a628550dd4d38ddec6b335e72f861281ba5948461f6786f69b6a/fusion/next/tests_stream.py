from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from .common import OUT
from .data import FIELDS
from .stream import Stream
from .tests_data import two_sessions
from .tests_training import make_panel


def engine(path,predictor=None):
    frame,starts,ends=two_sessions();panel=make_panel(frame,starts,ends)
    config={'lookback':240,'information':'ohlcv'}
    if predictor is None:
        predictor=lambda b:{'prediction':float(b['local'][0,-1,3])*.001,'weights':[.1,.1,.1,.1,.6]}
    return Stream(frame.index,['AAPL'],['SPY'],config,panel.scaler('2021-01-04'),predictor,'frozen_test_identity',path),frame,panel


def populate(stream,frame,first=150,last=391,omit=None):
    for i in range(first,last):
        for name in stream.names:
            if omit==(name,i):continue
            stream.ingest(name,frame.index[i],frame.iloc[i].to_dict(),frame.index[i]+pd.Timedelta(minutes=16))


def test_stream_matches_batch_causal_features_and_recovers_without_regenerating():
    with TemporaryDirectory(prefix='unit_stream_',dir=OUT) as folder:
        path=Path(folder)/'state.json';seen=[]
        def predictor(batch):
            assert 'target' not in batch
            seen.append(batch)
            return {'prediction':.0001,'weights':[0,0,0,0,1]}
        stream,frame,panel=engine(path,predictor);populate(stream,frame)
        result=stream.predict(frame.index[390],frame.index[390]+pd.Timedelta(minutes=16))
        assert result['status']=='ok' and result['mode']=='delayed_historical_replay'
        assert result['market_prediction_origin']==(frame.index[390]+pd.Timedelta(minutes=1)).isoformat()
        batch=panel.windows([0],[390],stream.config,stream.scalers,include_labels=False)
        import torch
        for name in ('local','market','foundation_context','state','sigma12'):
            torch.testing.assert_close(seen[0][name],batch[name],rtol=0,atol=0)
        resumed,_,_=engine(path,lambda b:pytest.fail('A saved origin must not be generated twice'))
        resumed.resume();assert resumed.predict(frame.index[390])['status']=='already_generated'


def test_stream_rejects_unfinished_duplicate_conflicts_and_logs_missing_inputs():
    with TemporaryDirectory(prefix='unit_stream_',dir=OUT) as folder:
        stream,frame,_=engine(Path(folder)/'state.json');populate(stream,frame,omit=('SPY',390))
        with pytest.raises(ValueError,match='completed'):
            stream.ingest('SPY',frame.index[390],frame.iloc[390].to_dict(),frame.index[390])
        result=stream.predict(frame.index[390],frame.index[390]+pd.Timedelta(minutes=16))
        assert result['status']=='input_or_model_failure' and 'SPY' in result['reason']
        values=frame.iloc[389].to_dict();stamp=frame.index[389];received=stamp+pd.Timedelta(minutes=16)
        assert stream.ingest('AAPL',stamp,values,received)['status']=='duplicate_ignored'
        values['volume']+=1
        with pytest.raises(ValueError,match='Conflicting'):stream.ingest('AAPL',stamp,values,received)


def test_delayed_data_cannot_be_labeled_a_timely_forecast():
    with TemporaryDirectory(prefix='unit_stream_',dir=OUT) as folder:
        stream,frame,_=engine(Path(folder)/'state.json');populate(stream,frame)
        result=stream.predict(frame.index[390],frame.index[390]+pd.Timedelta(minutes=16),mode='timely_forecast')
        assert result['status']=='input_or_model_failure' and 'Delayed' in result['reason']


def test_OHLCV_only_feed_is_sufficient_and_ancillary_placeholders_are_recorded():
    with TemporaryDirectory(prefix='unit_stream_',dir=OUT) as folder:
        stream,frame,_=engine(Path(folder)/'state.json');values=frame.iloc[300].to_dict()
        del values['trade_count'];del values['vwap'];stamp=frame.index[300]
        assert stream.ingest('AAPL',stamp,values,stamp+pd.Timedelta(minutes=1))['status']=='accepted'
        assert stream.bars['AAPL'][stamp.isoformat()]['ancillary_fields_unavailable']==['trade_count','vwap']


def test_missing_predecessor_uses_older_close_and_matches_batch():
    with TemporaryDirectory(prefix='unit_stream_anchor_',dir=OUT) as folder:
        seen=[]
        stream,frame,_=engine(Path(folder)/'state.json',lambda batch:seen.append(batch) or {'prediction':0.})
        frame.loc[frame.index[149],['open','high','low','close','vwap']]=105.
        missing=frame.copy();missing.iloc[150]=np.nan;starts=np.zeros(780,dtype=bool);starts[[0,390]]=True
        ends=np.r_[np.full(390,389),np.full(390,779)]
        panel=make_panel(missing,starts,ends);populate(stream,frame,first=148,last=391,omit=('AAPL',150))
        # Both inputs lack the predecessor; a single missing predecessor is outside
        # the scored 240 bars and does not make the batch origin inadmissible.
        stream.bars['SPY'].pop(frame.index[150].isoformat())
        result=stream.predict(frame.index[390],frame.index[390]+pd.Timedelta(minutes=16))
        assert result['status']=='ok'
        import torch
        expected=panel.windows([0],[390],stream.config,stream.scalers,include_labels=False)
        for key in ('local','market','foundation_context','state','sigma12'):
            torch.testing.assert_close(seen[0][key],expected[key],rtol=0,atol=0)


def test_model_runtime_failure_is_durably_journaled_and_never_dropped():
    with TemporaryDirectory(prefix='unit_stream_fault_',dir=OUT) as folder:
        def fail(batch):raise RuntimeError('injected model execution failure')
        path=Path(folder)/'state.json';stream,frame,_=engine(path,fail);populate(stream,frame)
        result=stream.predict(frame.index[390],frame.index[390]+pd.Timedelta(minutes=16))
        assert result['status']=='input_or_model_failure' and result['error_type']=='RuntimeError'
        restored,_,_=engine(path,lambda b:pytest.fail('A failed origin cannot be silently retried as a success'))
        restored.resume();assert restored.predict(frame.index[390])['status']=='already_generated'
        assert restored.outputs[frame.index[390].isoformat()]['reason']=='injected model execution failure'


def test_stream_atomic_state_keeps_prior_snapshot_on_publication_failure(monkeypatch):
    from . import stream as module
    from .common import digest
    with TemporaryDirectory(prefix='unit_stream_atomic_',dir=OUT) as folder:
        path=Path(folder)/'state.json';stream,frame,_=engine(path);populate(stream,frame);stream.save();before=digest(path)
        def fail(*args,**kwargs):raise OSError('injected before atomic state publication')
        monkeypatch.setattr(module,'write_json',fail)
        with pytest.raises(OSError):stream.predict(frame.index[390],frame.index[390]+pd.Timedelta(minutes=16))
        assert digest(path)==before
        restored,_,_=engine(path);restored.resume();assert not restored.outputs
