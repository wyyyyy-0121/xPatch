"""All-stock misses use one encoder call and preserve per-origin cache values."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import numpy as np
import torch

from . import cache
from .common import OUT,write_json


def test_missing_stocks_are_encoded_together_and_existing_latents_are_reused(monkeypatch):
    with TemporaryDirectory(prefix='unit_bulk_cache_',dir=OUT) as temporary:
        root=Path(temporary);monkeypatch.setattr(cache,'OUT',root)
        (root/'data').mkdir();np.save(root/'data/timestamps_ns.npy',np.arange(10))
        for stock in ('AAPL','MSFT'):write_json(root/'data'/stock/'manifest.json',{'stock':stock})
        calls=[]
        def latent(context):calls.append(len(context));return context[:,-1,:2]
        branch=SimpleNamespace(mode='frozen',receipt={'weights':'fixed'},latent=latent,
            backbone=SimpleNamespace(config=SimpleNamespace(d_model=2)))
        panel=SimpleNamespace(stocks=['AAPL','MSFT'],dates=np.array(['2021-01-04']*10))
        store=cache.RepresentationCache(panel,branch,{'lookback':48})
        batch={'stock':torch.tensor([0,1,0,1]),'row':torch.tensor([1,2,3,4]),
            'foundation_context':torch.arange(4*48*3).reshape(4,48,3).float(),
            'local':torch.ones(4,48,12)}
        expected=batch['foundation_context'][:,-1,:2]
        torch.testing.assert_close(store.get(batch),expected,rtol=0,atol=0)
        assert calls==[4]
        torch.testing.assert_close(store.get(batch),expected,rtol=0,atol=0);assert calls==[4]
        batch['row']=torch.tensor([1,2,7,8])
        torch.testing.assert_close(store.get(batch),expected,rtol=0,atol=0);assert calls==[4,2]
