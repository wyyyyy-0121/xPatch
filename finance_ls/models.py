"""Asymmetric lookbacks with the original xPatch forecasting branches."""
from types import SimpleNamespace

import torch
from torch import nn

from layers.ema import EMA
from layers.network import Network
from models.xPatch import Model as OriginalXPatch


NEURAL_MODELS = {"xpatch48", "xpatch240", "ls", "ls_unpooled", "ls_no_decomp",
                 "long_only", "short_only"}
ALL_MODELS = NEURAL_MODELS | {"zero", "ridge"}


class XPatchLS(nn.Module):
    def __init__(self, config, variant="ls"):
        super().__init__()
        self.lookback = config["lookback"]
        self.short_len = config["short_len"]
        self.pool = 1 if variant == "ls_unpooled" else config["pool"]
        self.decompose = variant != "ls_no_decomp"
        self.streams = {"long_only": "long", "short_only": "short"}.get(variant, "both")
        if self.lookback % self.pool or self.short_len > self.lookback:
            raise ValueError("Long lookback must be divisible by pool and cover short history")
        self.short_ema = EMA(config["alpha"])
        self.long_ema = EMA(config["alpha"])
        self.net = Network(self.short_len, config["horizon"], config["patch_len"],
                           config["stride"], "end", trend_seq_len=self.lookback // self.pool,
                           streams=self.streams)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.lookback, 1):
            raise ValueError(f"Expected [batch, {self.lookback}, 1]")
        short = x[:, -self.short_len:, :]
        long = x
        if self.decompose:
            if self.streams != "long":
                short = short - self.short_ema(short)
            if self.streams != "short":
                long = self.long_ema(long)
        # Non-overlapping trailing-aligned bins; mean preserves return-per-minute units.
        if self.streams != "short":
            long = long.reshape(x.shape[0], self.lookback // self.pool, self.pool, 1).mean(dim=2)
        return self.net(short, long)


class OriginalAdapter(nn.Module):
    def __init__(self, config, length):
        super().__init__()
        self.length = length
        self.model = OriginalXPatch(SimpleNamespace(
            seq_len=length, pred_len=config["horizon"], enc_in=1,
            patch_len=config["patch_len"], stride=config["stride"],
            padding_patch="end", revin=False, ma_type="ema", alpha=config["alpha"], beta=0.1))
        # The upstream model creates RevIN affine weights even when disabled.
        # Exclude these inert parameters for a fair active-parameter comparison.
        del self.model.revin_layer

    def forward(self, x):
        return self.model(x[:, -self.length:, :])


def make_model(name, config):
    if name == "xpatch48":
        return OriginalAdapter(config, config["short_len"])
    if name == "xpatch240":
        return OriginalAdapter(config, config["lookback"])
    if name not in NEURAL_MODELS:
        raise ValueError(f"Not a neural model: {name}")
    return XPatchLS(config, name)
