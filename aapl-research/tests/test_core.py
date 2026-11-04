"""Tests of failure modes that can change a research conclusion."""
import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from aapl.common import ROOT, runtime
from aapl.models import CausalEMA, neural, quantile_point
from aapl.search_space import generate
from aapl.statistics import paired_blocks, holm


def test_shared_search_space_and_paired_losses():
    trials = generate()
    assert {k: len(v) for k, v in trials.items()} == dict(ridge=24, lightgbm=48, xpatch=32,
                                                       patchtst=32, chronos2_frozen=12, chronos2_lora=12)
    for model in ("xpatch", "patchtst"):
        for a, b in zip(trials[model][::2], trials[model][1::2]):
            assert a["loss"] != b["loss"]
            assert {k: v for k, v in a.items() if k not in ("loss", "trial_id")} == {
                k: v for k, v in b.items() if k not in ("loss", "trial_id")}


def test_all_native_supervised_shapes_backward():
    runtime()
    for model in ("xpatch", "patchtst"):
        for lookback in (48, 96, 240):
            for patch in (8, 12, 16):
                cfg = dict(model=model, lookback=lookback, patch=patch, stride=patch // 2,
                           alpha=0.7, revin=True, layers=3, d_model=64, dropout=0.2)
                net = neural(cfg)
                pred = net(torch.randn(2, lookback, 1))
                assert pred.shape == (2, 12) and torch.isfinite(pred).all()
                pred.square().mean().backward()


def test_quantile_point_is_not_log_of_mean_price():
    q = np.array([[[1.0], [2.0], [4.0]]])
    levels = [0.1, 0.5, 0.9]
    np.testing.assert_array_equal(quantile_point(q, levels, "native_median"), [[2.0]])
    # Exact integral under the declared flat-tail, piecewise-linear quantile rule.
    np.testing.assert_allclose(quantile_point(q, levels, "quantile_integral_tailflat"), [[2.3]])


def test_block_bootstrap_direction_and_null_coverage():
    rng = np.random.default_rng(8113)
    dates = np.repeat(np.arange(100), 4).astype(str)
    noise = np.repeat(rng.normal(0, 0.1, 100), 4)
    assert paired_blocks(noise - 0.5, dates, repeats=1000)["one_sided_p"] < 0.01
    assert paired_blocks(noise + 0.5, dates, repeats=1000)["one_sided_p"] > 0.99
    covered = 0
    for _ in range(60):
        day_loss = np.repeat(rng.normal(0, 1, 100), 4)
        low, high = paired_blocks(day_loss, dates, repeats=500)["ci95"]
        covered += low <= 0 <= high
    assert 48 <= covered <= 60


def test_holm_adjustment_known_example():
    assert holm({"a": 0.01, "b": 0.03, "c": 0.2}) == {"a": 0.03, "b": 0.06, "c": 0.2}
