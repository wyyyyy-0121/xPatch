"""Check information boundaries and economic units independently of model code."""
import numpy as np
import pandas as pd
import pytest

from finance.data import RVData
from finance.features import FeatureStore, FitState
from finance.io import ROOT, read_json
from test_finance_rv_data import frames, locate


@pytest.fixture(scope="module")
def example():
    source = frames("2022-01-03", "2022-05-02")
    # Vary both variance and sign so semivariance features have distinct meaning.
    for j, frame in enumerate(source.values()):
        n = np.arange(len(frame))
        steps = (1 + j * 0.2) * (0.0003 * np.sin(n / 5) + 0.0001)
        price = 100 * np.exp(np.cumsum(steps))
        frame.loc[:, "close"] = price
        frame.loc[:, "open"] = price
        frame.loc[:, "high"] = price * 1.0002
        frame.loc[:, "low"] = price * 0.9998
        frame.loc[:, "volume"] = 100 + n % 40
    data = RVData.from_frames(source, "2022-01-03", "2022-05-02")
    config = read_json(ROOT / "configs/rv30.json")
    state = FitState.fit(data, data.days[:50], config["seasonality"])
    return source, data, config, state


def test_feature_and_fitted_state_invariant_to_unseen_future(example):
    source, data, config, state = example
    origin = locate(data, "2022-05-02 10:30")
    changed = {asset: frame.copy() for asset, frame in source.items()}
    cutoff = pd.Timestamp("2022-05-02 10:29", tz="America/New_York").tz_convert("UTC")
    for frame in changed.values():
        mask = frame.index >= cutoff
        frame.loc[mask, ["open", "high", "low", "close"]] *= np.exp(np.arange(mask.sum()) * .003)[:, None]
        frame.loc[mask, "volume"] *= 100
    future = RVData.from_frames(changed, "2022-01-03", "2022-05-02")
    fitted_again = FitState.fit(future, data.days[:50], config["seasonality"])
    assert state.dictionary() == fitted_again.dictionary()
    for representation in ("log_rv", "log_seasonal_ratio"):
        params = {"feature_profile": "rich", "representation": representation}
        before, names = FeatureStore(data, config).matrix([origin], state, "market", "ridge", params)
        after, names_after = FeatureStore(future, config).matrix([origin], state, "market", "ridge", params)
        np.testing.assert_array_equal(before, after)
        assert names == names_after and len(set(names)) == len(names)
        # Rebinding a fitted state must invalidate its derived data cache.
        np.testing.assert_array_equal(state.series(data, representation)[:origin],
            state.series(future, representation)[:origin])
        assert not np.array_equal(data.targets([origin]), future.targets([origin]))


def test_registered_matrices_and_target_units(example):
    _, data, config, state = example
    origins = data.origins(dates=data.days[-2:], stride=10)
    assert len(origins)
    for representation in ("level_rv", "log_rv", "log_seasonal_ratio"):
        y = state.labels(data, origins, representation)
        np.testing.assert_allclose(state.inverse(data, origins, y, representation), data.targets(origins), rtol=1e-12)
        for panel in ("aapl", "market"):
            matrix, names = FeatureStore(data, config).matrix(origins, state, panel, "har",
                {"semivariance": True, "representation": representation})
            assert matrix.shape == (len(origins), len(names))
            assert np.isfinite(matrix).all()
    # Original variance scale is a sum of 30 squared returns, not its square root.
    origin = origins[0]
    expected = sum(data.returns[origin + k, 0] ** 2 for k in range(1, 31))
    assert data.targets([origin])[0] == pytest.approx(expected, rel=1e-9)
    restored = FitState(**state.dictionary())
    assert restored.dictionary() == state.dictionary()


def test_future_season_uses_target_clock_and_past_season_keeps_window(example):
    _, data, _, state = example
    profile = np.full_like(np.asarray(state.seasonal_profile), 1e-6)
    # Future target from 10:30 ends at 11:00, whose last minute has a separate bucket.
    profile[6:, :] = 2e-6
    flat = FitState(**{**state.dictionary(), "seasonal_profile": profile.tolist()})
    origin = locate(data, "2022-05-02 10:30")
    assert flat.future_season(data, [origin])[0] == pytest.approx(31e-6)
    assert flat.expected(data, 30)[origin - 1, 0] == pytest.approx(30e-6)


def test_origin_cache_filters_without_mutating_common_eligibility(example):
    _, data, _, _ = example
    all_origins = data.origins().copy()
    selected = data.origins(dates=data.days[-2:], stride=10)
    assert np.isin(selected, all_origins).all()
    np.testing.assert_array_equal(data.origins(), all_origins)
    local = data.timestamps[selected].tz_convert("America/New_York")
    assert ((local.hour * 60 + local.minute - 585) % 10 == 0).all()
