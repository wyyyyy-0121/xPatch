"""Tests for leakage boundaries and substantive fusion failure modes."""
import numpy as np
import pandas as pd
import pytest
import torch

from .features import price_features,context_tensor
from .inference import market_from_frame
from .head import fit,predict,StateGate
from .pipeline import forward_blocks
from .config import candidate_grid


def frame():
    rng=np.random.default_rng(48)
    idx=pd.date_range('2022-01-03T14:30Z',periods=390,freq='min')
    close=100*np.exp(np.cumsum(rng.normal(0,.0001,390)))
    return pd.DataFrame({'open':close,'high':close+.02,'low':close-.02,'close':close,'volume':1000.},index=idx)


def test_inference_rejects_gaps_overnight_and_future_bars():
    f=frame();end=f.index[260]+pd.Timedelta(minutes=1)
    m=market_from_frame(f,end)
    assert m.bars.index[-1]==f.index[260] and len(m.bars)==241
    g=f.copy();g.iloc[261:,3]*=10
    other=market_from_frame(g,end)
    np.testing.assert_array_equal(context_tensor(m,[240]),context_tensor(other,[240]))
    with pytest.raises(ValueError):market_from_frame(f.drop(f.index[200]),end)
    with pytest.raises(ValueError):market_from_frame(f.iloc[:100],end)


def test_zero_expert_and_residual_bound():
    net=StateGate(4,5,8,0.)
    out=net(torch.randn(20,4),torch.zeros(20,5),torch.ones(20))
    assert torch.equal(out,torch.zeros_like(out))


def test_head_training_ignores_evaluation_targets():
    rng=np.random.default_rng(48)
    data={'experts':rng.normal(0,.001,(256,5)),'features':rng.normal(size=(256,4)),
          'embedding':rng.normal(size=(256,48)),'volatility':np.full(256,.002),'y':rng.normal(0,.002,256)}
    config=next(c for c in candidate_grid() if c['kind']=='ridge' and c['basis']=='foundation')
    state=fit(config,data)
    pred=predict(state,data)
    changed={**data,'y':data['y']+100}
    np.testing.assert_array_equal(pred,predict(state,changed))


def test_meta_fit_purges_last_session_before_validation():
    dates=np.repeat(pd.bdate_range('2023-01-01','2026-03-31').strftime('%Y-%m-%d').to_numpy(),20)
    data={'dates':dates,'y':np.zeros(len(dates)),'experts':np.zeros((len(dates),5)),
          'features':np.zeros((len(dates),4)),'embedding':np.zeros((len(dates),48)),
          'volatility':np.ones(len(dates)),'origins':np.arange(len(dates))}
    blocks=forward_blocks(data,{'validation_quarters':['2025Q1','2025Q3','2026Q1']})
    for _,tr,val in blocks:
        prior=sorted(set(dates[dates<min(val['dates'])]))
        assert max(tr['dates'])==prior[-2]
