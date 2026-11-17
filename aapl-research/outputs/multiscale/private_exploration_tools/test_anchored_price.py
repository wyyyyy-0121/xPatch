"""Synthetic checks for the price-anchored residual head."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from anchored_price import AnchoredPrice
from multiscale.features import Scaler


@pytest.mark.parametrize("model",["mlp","gru","lstm","xpatch"])
@pytest.mark.parametrize("horizon",[1,8])
def test_zero_head_is_persistence_for_varied_anchors(model,horizon):
    scaler = Scaler(0.,1.,100.,20.,tuple(np.linspace(90.,110.,horizon)),
                    tuple(np.linspace(3.,9.,horizon)),20)
    torch.manual_seed(5)
    net = AnchoredPrice(model,96,horizon,scaler).eval()
    raw_anchor = np.asarray([50.,100.,160.,250.])[:,None]
    normalized_anchor = ((raw_anchor-scaler.anchor_mean)/scaler.anchor_std).astype(np.float32)
    with torch.inference_mode():
        normalized_y = net(torch.randn(4,96,1),torch.as_tensor(normalized_anchor)).numpy()
    actual = scaler.invert_y(normalized_y)
    np.testing.assert_allclose(actual,np.broadcast_to(raw_anchor,(4,horizon)),rtol=2e-6,atol=2e-5)
