"""Close-only MLP/GRU/LSTM and native xPatch adapters with variable horizons."""
from __future__ import annotations

import torch
from torch import nn

from aapl.models import CausalEMA
from aapl.vendor.xpatch_network import Network


class MLP(nn.Module):
    def __init__(self, lookback: int, horizon: int, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(lookback + 1, width), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(width, horizon))

    def forward(self, x, anchor):
        return self.net(torch.cat((x.squeeze(-1), anchor), dim=1))


class Recurrent(nn.Module):
    def __init__(self, kind: str, horizon: int, width: int = 32):
        super().__init__()
        cls = {"gru": nn.GRU, "lstm": nn.LSTM}[kind]
        self.rnn = cls(input_size=1, hidden_size=width, batch_first=True)
        self.head = nn.Linear(width + 1, horizon)

    def forward(self, x, anchor):
        hidden = self.rnn(x)[0][:, -1]
        return self.head(torch.cat((hidden, anchor), dim=1))


class NativeXPatch(nn.Module):
    def __init__(self, lookback: int, horizon: int, patch: int = 8, alpha: float = 0.1):
        super().__init__()
        if lookback < 2 or horizon < 1:
            raise ValueError("Invalid xPatch dimensions")
        patch = min(int(patch), lookback)
        if patch < 1:
            raise ValueError("Invalid patch length")
        self.horizon = horizon
        # The native backbone uses a half-horizon pooling layer; H=1 is not
        # representable. Run its H=2 head and expose the first step explicitly.
        self.native_horizon = max(2, horizon)
        self.ema = CausalEMA(alpha)
        self.backbone = Network(lookback, self.native_horizon, patch, max(1, patch // 2), "end")
        self.anchor_head = nn.Linear(1, self.native_horizon)

    def forward(self, x, anchor):
        trend = self.ema(x)
        value = self.backbone(x - trend, trend)[:, :, 0] + self.anchor_head(anchor)
        return value[:, :self.horizon]


def make_model(name: str, lookback: int, horizon: int):
    if lookback < 2 or horizon < 1:
        raise ValueError("Invalid model dimensions")
    if name == "mlp":
        return MLP(lookback, horizon)
    if name in ("gru", "lstm"):
        return Recurrent(name, horizon)
    if name == "xpatch":
        return NativeXPatch(lookback, horizon)
    raise ValueError(f"Unknown neural model: {name}")
