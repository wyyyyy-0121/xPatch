"""Exercise the final gate with toy values; never read actual held-out market data."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from . import final_test
from .common import OUT,digest
from .data import causal_transform
from .evaluation import point_scores,seed_system
from .tests_data import two_sessions


def test_negative_development_stops_before_any_final_market_or_model_access(monkeypatch):
    with TemporaryDirectory(prefix='unit_final_gate_',dir=OUT) as temporary:
        root=Path(temporary);monkeypatch.setattr(final_test,'FINAL',root)
        monkeypatch.setattr(final_test,'runtime',lambda seed:'cpu')
        monkeypatch.setattr(final_test,'checked_freeze',lambda:({}, {'development_gate_passed':False}))
        def forbidden(*args,**kwargs):raise AssertionError('Negative gate accessed data')
        monkeypatch.setattr(final_test.Panel,'load',forbidden)
        monkeypatch.setattr(final_test,'credentials',forbidden)
        monkeypatch.setattr(final_test,'load_panel',forbidden)
        with pytest.raises(PermissionError,match='Development gate failed'):final_test.run(root/'deployment')
        assert not (root/'entry.json').exists() and not (root/'raw').exists()


def test_final_context_extension_matches_full_transform_at_open_and_missing_bar():
    frame,starts,ends=two_sessions();frame.iloc[390]=np.nan
    full,observed,close,_=causal_transform(frame,starts)
    joined,mask,new_close=final_test.extend_features(full[149:390],observed[149:390],close[389],
                                                    frame.iloc[390:],frame.index[390:],starts[390:])
    np.testing.assert_array_equal(joined,full[149:])
    np.testing.assert_array_equal(mask,observed[149:])
    np.testing.assert_array_equal(new_close,close[390:])


def test_final_transport_rejects_unfrozen_symbol_or_dates_before_network(monkeypatch):
    monkeypatch.setattr(final_test,'assert_entry',lambda claim:None)
    def forbidden(*args,**kwargs):raise AssertionError('Out of scope request reached network')
    monkeypatch.setattr(final_test,'build_opener',forbidden)
    for params in ({'start':'2021-01-01','end':'2021-02-01','symbols':'AAPL'},
                   {'start':final_test.bounds(final_test.START,final_test.END)[0],
                    'end':final_test.bounds(final_test.START,final_test.END)[1],'symbols':'TSLA'}):
        with pytest.raises(PermissionError,match='scope'):final_test.final_request(params,{'symbols':['AAPL']},'unused','unused')


def test_confirmation_flags_cannot_replace_recomputed_aligned_raw_gate(monkeypatch):
    with TemporaryDirectory(prefix='unit_final_confirmation_',dir=OUT) as temporary:
        root=Path(temporary);monkeypatch.setattr(final_test,'OUT',root)
        (root/'system_freeze.json').write_text('{}')
        # These are toy targets and dates, not observations from Q2 or Q3.
        dates=np.array(['2026-03-27','2026-03-30','2026-03-31','2026-04-01','2026-04-02'])
        target=np.array([1.,-1.],dtype=np.float32);expert=np.zeros((2,4),dtype=np.float32)
        expert[:,0]=target*.1;seeds=np.tile(target*.5,(3,1));seed_report,prediction=seed_system(target,seeds)
        static={'coefficient':[1.,0.,0.,0.]};control=expert[:,0]
        panel=SimpleNamespace(dates=dates,origins={'AAPL':np.array([3,4])},targets={'AAPL':np.r_[np.zeros(3,dtype=np.float32),target]})
        frozen={'fixed_strongest_single':{'family':'ridge'}};path=root/'confirmation_predictions.npz'
        arrays={'rows':np.array([3,4]),'dates':dates[3:],'target':target,'prediction':prediction,
                'seeds':seeds,'experts':expert,'single':control,'simple':final_test.simple.predict(static,expert)}
        np.savez(path,**arrays)
        report={'evaluated':True,'proceed_to_final_test':True,'system_freeze_sha256':digest(root/'system_freeze.json'),
                'prediction_sha256':digest(path),'fixed_simple_fit':static,'seed_report':seed_report,
                'scores':{name:point_scores(target,p) for name,p in [('system',prediction),('zero',np.zeros(2)),('single',control),('simple',arrays['simple'])]}}
        final_test.recheck_confirmation(panel,frozen,report)
        forged={**report,'scores':{**report['scores'],'system':{**report['scores']['system'],'mse':0.}}}
        with pytest.raises(ValueError,match='reported scores'):final_test.recheck_confirmation(panel,frozen,forged)
        np.savez(path,**{**arrays,'rows':np.array([2,4])});report['prediction_sha256']=digest(path)
        with pytest.raises(ValueError,match='identity changed'):final_test.recheck_confirmation(panel,frozen,report)


def test_real_local_artifact_must_match_saved_cloud_outputs_before_final_data(monkeypatch):
    with TemporaryDirectory(prefix='unit_cpu_preflight_',dir=OUT) as temporary:
        root=Path(temporary);monkeypatch.setattr(final_test,'OUT',root);monkeypatch.setattr(final_test,'FINAL',root/'final')
        monkeypatch.setattr(final_test,'assert_entry',lambda *args,**kwargs:None)
        monkeypatch.setattr(final_test.Panel,'load',lambda:SimpleNamespace())
        rows=np.arange(4);prediction=np.array([.1,.2,.3,.4]);seeds=np.tile(prediction,(3,1));experts=np.tile(prediction[:,None],(1,4))
        np.savez(root/'confirmation_predictions.npz',rows=rows,prediction=prediction,seeds=seeds,experts=experts)
        class Model:
            model_identity='frozen';offset=0.
            def predict_origin(self,panel,rows):
                return {'prediction':prediction[rows]+self.offset,'seed_predictions':seeds[:,rows],
                        'expert_predictions':experts[rows]}
        system=Model();claim={'identity':'claim','system_identity':'frozen'};confirmation={'scores':{'zero':{'mse':.1}}}
        assert final_test.local_preflight(system,claim,confirmation,2)['passed']
        system.offset=.01
        with pytest.raises(RuntimeError,match='before opening Q3'):final_test.local_preflight(system,claim,confirmation,2)
