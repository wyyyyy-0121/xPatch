from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pytest

from .common import OUT,write_json,read_json
from . import confirmation
from .delivery import score_journal
from .design import candidates


def test_failed_development_still_exports_but_never_scores_confirmation_or_opens_Q3(monkeypatch):
    with TemporaryDirectory(prefix='unit_confirmation_',dir=OUT) as temporary:
        root=Path(temporary);monkeypatch.setattr(confirmation,'OUT',root)
        frozen={'selected_system':{'config':{'information':'ohlcv','training_scope':'multistock'}},'development_gate_passed':False}
        write_json(root/'system_freeze.json',frozen)
        monkeypatch.setattr(confirmation,'checked_freeze',lambda:({},frozen))
        panel=SimpleNamespace(stocks=['AAPL'],background=['SPY'],dates=np.array(['2026-03-27','2026-03-30','2026-03-31','2026-04-01']))
        monkeypatch.setattr(confirmation.Panel,'load',lambda:panel)
        calls=[]
        monkeypatch.setattr(confirmation,'fit_frozen',lambda *args:(calls.append('fit') or ({},[])))
        def export(destination,*args):
            calls.append('export');destination.mkdir(parents=True)
            write_json(destination/'deployment.json',{'identity':'frozen-test'})
        monkeypatch.setattr(confirmation,'export',export)
        # Access to targets is absent from the toy panel. This branch must not need it.
        result=confirmation.run()
        assert calls==['fit','export'] and not result['evaluated'] and not result['Q3_opened']
        assert not (root/'confirmation_started.json').exists()
        with pytest.raises(ValueError,match='already been evaluated'):confirmation.run()
        assert calls==['fit','export']


def test_two_prespecified_LoRA_configs_keep_both_ranks_and_learning_rates():
    configs=candidates('lora',2)
    assert {(c['rank'],c['backbone_lr']) for c in configs}=={(8,1e-5),(16,3e-5)}


def test_saved_prediction_recomputation_rejects_missing_or_duplicated_origins():
    with TemporaryDirectory(prefix='unit_recompute_',dir=OUT) as temporary:
        path=Path(temporary)/'saved.npz'
        np.savez(path,target=[1.,-1.],prediction=[0.,0.],rows=[2,3],dates=['2021-01-04','2021-01-04'])
        assert score_journal(path)['recomputed_raw_metrics']['mse']==1.
        np.savez(path,target=[1.,-1.],prediction=[0.,0.],rows=[2,2],dates=['2021-01-04','2021-01-04'])
        with pytest.raises(ValueError,match='Repeated'):score_journal(path)
