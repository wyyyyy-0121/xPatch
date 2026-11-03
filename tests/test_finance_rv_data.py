"""Economic-time invariants: actual future labels, lagged inputs and no gap fill."""
import numpy as np
import pandas as pd
import pytest

from finance.data import ASSETS, RVData, rolling_sum
from finance_v2.data import calendar_grid


def frames(start, end, overnight_factor=1):
    grids = calendar_grid(start, end)
    pieces = []
    for i, grid in enumerate(grids.values()):
        close = 100 * overnight_factor ** i * np.exp(np.arange(len(grid)) * 0.001)
        pieces.append(pd.DataFrame({"open": close, "high": close * 1.0001,
            "low": close * 0.9999, "close": close, "volume": np.full(len(grid), 100.0)}, index=grid))
    frame = pd.concat(pieces)
    return {asset: frame.copy() for asset in ASSETS}


def locate(data, local):
    return data.timestamps.get_loc(pd.Timestamp(local, tz="America/New_York").tz_convert("UTC"))


def test_exact_30_minute_label_and_one_minute_information_lag():
    data = RVData.from_frames(frames("2024-03-08", "2024-03-11"), "2024-03-08", "2024-03-11")
    origin = locate(data, "2024-03-11 09:45")
    candidates = data.origins(context=1, history_sessions=0)
    assert origin in candidates
    assert data.targets([origin])[0] == pytest.approx(30 * 0.001 ** 2)
    history = data.context([origin], 30, data.returns)
    assert np.allclose(history, 0.001)
    assert data.timestamps[origin - 1].tz_convert("America/New_York").strftime("%H:%M") == "09:44"
    assert data.timestamps[origin + 30].tz_convert("America/New_York").strftime("%H:%M") == "10:15"


def test_future_changes_do_not_change_inputs_but_change_labels():
    originals = frames("2024-03-08", "2024-03-11")
    data = RVData.from_frames(originals, "2024-03-08", "2024-03-11")
    changed = {asset: frame.copy() for asset, frame in originals.items()}
    cutoff = pd.Timestamp("2024-03-11 09:44", tz="America/New_York").tz_convert("UTC")
    for frame in changed.values():
        mask = frame.index >= cutoff
        frame.loc[mask, ["open", "high", "low", "close"]] *= np.exp(np.arange(mask.sum()) * 0.002)[:, None]
    future = RVData.from_frames(changed, "2024-03-08", "2024-03-11")
    origin = locate(data, "2024-03-11 09:45")
    assert np.array_equal(data.context([origin], 64, data.rv()), future.context([origin], 64, future.rv()))
    assert np.array_equal(data.previous_sessions_rv(1)[origin], future.previous_sessions_rv(1)[origin])
    assert data.targets([origin])[0] != pytest.approx(future.targets([origin])[0])


def test_overnight_jump_is_excluded_while_past_sessions_remain_available():
    data = RVData.from_frames(frames("2024-03-08", "2024-03-11", overnight_factor=10), "2024-03-08", "2024-03-11")
    assert np.isfinite(data.returns).all()
    assert np.allclose(data.returns, 0.001)
    assert data.daily_rv()[0, 0] == pytest.approx(389 * 0.001 ** 2)
    origin = locate(data, "2024-03-11 09:45")
    assert np.isfinite(data.context([origin], 64, data.rv())).all()


def test_dst_and_early_close_qualify_actual_exchange_endpoints():
    data = RVData.from_frames(frames("2024-03-08", "2024-03-11"), "2024-03-08", "2024-03-11")
    assert data.timestamps[locate(data, "2024-03-08 09:45")].hour == 14
    assert data.timestamps[locate(data, "2024-03-11 09:45")].hour == 13
    early = RVData.from_frames(frames("2024-11-27", "2024-11-29"), "2024-11-27", "2024-11-29")
    origins = early.origins(dates=["2024-11-29"], context=1, history_sessions=0)
    assert early.timestamps[origins[0]].tz_convert("America/New_York").strftime("%H:%M") == "09:45"
    assert early.timestamps[origins[-1]].tz_convert("America/New_York").strftime("%H:%M") == "12:30"
    assert np.allclose(early.targets(origins), 30e-6)
    with pytest.raises(ValueError, match="gap|Overnight"):
        early.targets([locate(early, "2024-11-27 15:45")])


def test_future_covariate_missingness_does_not_select_evaluation_origins():
    base = frames("2024-03-08", "2024-03-11")
    missing = pd.Timestamp("2024-03-11 10:40", tz="America/New_York").tz_convert("UTC")
    base["SPY"] = base["SPY"].drop(missing)
    data = RVData.from_frames(base, "2024-03-08", "2024-03-11")
    origin = locate(data, "2024-03-11 10:30")
    assert origin in data.origins(context=1, history_sessions=0)
    assert data.targets([origin])[0] == pytest.approx(30e-6)
    assert locate(data, "2024-03-11 10:45") not in data.origins(context=1, history_sessions=0)
    assert data.regular_audits["SPY"]["missing_rows"] == 1
    assert np.isnan(data.returns[:, 1]).sum() == 2


def test_missing_target_rejected_and_live_inputs_do_not_require_future_prices():
    base = frames("2024-03-08", "2024-03-11")
    missing = pd.Timestamp("2024-03-11 10:40", tz="America/New_York").tz_convert("UTC")
    base["AAPL"] = base["AAPL"].drop(missing)
    data = RVData.from_frames(base, "2024-03-08", "2024-03-11")
    origin = locate(data, "2024-03-11 10:30")
    assert origin not in data.origins(context=1, history_sessions=0)
    assert origin in data.origins(context=1, history_sessions=0, require_target=False)
    with pytest.raises(ValueError, match="missing"):
        data.targets([origin])


def test_completed_day_features_never_read_current_or_future_session():
    data = RVData.from_frames(frames("2024-03-08", "2024-03-12"), "2024-03-08", "2024-03-12")
    origin = locate(data, "2024-03-11 09:45")
    previous = data.previous_sessions_rv(1)
    assert previous[origin, 0] == pytest.approx(389e-6)
    assert np.isnan(data.previous_sessions_rv(22)[origin, 0])


def test_rolling_missing_values_are_not_interpolated_or_dropped():
    values = np.array([1, 2, np.nan, 4, 5, 6], dtype=float)
    result = rolling_sum(values, 3)
    assert np.isnan(result[:5]).all()
    assert result[-1] == 15
