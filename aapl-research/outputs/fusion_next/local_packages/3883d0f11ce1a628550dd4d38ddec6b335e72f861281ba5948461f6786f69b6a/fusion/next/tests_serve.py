"""JSONL failure accounting, explicit timestamps and the generic holdout boundary."""
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

from .common import OUT
from .tests_data import two_sessions
from .tests_training import make_panel
from . import serve


def test_generic_input_rejects_protected_period_before_forwarding():
    class Unused:
        def ingest(self,*args):raise AssertionError('Protected input reached stream')
        def predict(self,*args):raise AssertionError('Protected input reached model')
    for kind in ('bar','predict'):
        with pytest.raises(PermissionError):serve.process_event(Unused(),{'type':kind,'bar_start':'2026-07-01T14:30:00Z'},'delayed_historical_replay')
    with pytest.raises(ValueError):serve.process_event(Unused(),{'type':'predict'},'delayed_historical_replay')


def test_missing_history_is_a_durable_failed_forecast_and_malformed_json_is_counted(monkeypatch):
    frame,starts,ends=two_sessions();panel=make_panel(frame,starts,ends)
    class Predictor:
        stocks=panel.stocks;background=panel.background;model_identity='unit_missing_history'
        manifest={'system_checkpoints':[{'config':{'lookback':240}}]}
        models=[(None,panel.scaler('2021-01-04'),manifest['system_checkpoints'][0]['config'])]
        def __init__(self,*args):pass
        def predict_origin(self,*args):raise AssertionError('Missing history reached model')
    monkeypatch.setattr(serve,'FrozenSystem',Predictor)
    monkeypatch.setattr(serve,'runtime',lambda *args:'cpu')
    monkeypatch.setattr(serve,'sessions',lambda *args:{'toy':frame.index})
    with TemporaryDirectory(prefix='unit_serve_',dir=OUT) as folder:
        state=Path(folder)/'state.json';output=io.StringIO()
        event=json.dumps({'type':'predict','bar_start':frame.index[300].isoformat(),'generated_at':(frame.index[300]+pd.Timedelta(minutes=16)).isoformat()})
        result=serve.run('unused','2021-01-04','2021-01-05',state,input_stream=io.StringIO(event+'\n{malformed}\n'),output_stream=output)
        assert result['rejected_events']==result['failed_forecasts']==1 and result['events']==2
        saved=json.loads(state.read_text());assert len(saved['outputs'])==1
        assert next(iter(saved['outputs'].values()))['status']=='input_or_model_failure'
        assert len(list((state.parent/'input_failures').glob('*.json')))==1
