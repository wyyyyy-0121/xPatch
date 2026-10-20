import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment import data_gate, load_ohlcv, make_samples, rolling_splits, run_profile


def test_csv_is_sorted_and_deduplicated(tmp_path):
    path = tmp_path / "bars.csv"
    path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2024-01-02 09:32,11,12,10,11,3\n"
        "2024-01-02 09:31,10,11,9,10,2\n"
        "2024-01-02 09:32,11,13,10,12,4\n"
    )
    frame, meta = load_ohlcv(path)
    assert frame.index.is_monotonic_increasing
    assert len(frame) == 2
    assert frame.iloc[-1].close == 12
    assert meta["duplicate_timestamps_removed"] == 1


def test_future_targets_are_aligned_and_intraday_sequences_stay_in_session():
    index = pd.date_range("2024-01-02 09:30", periods=5, freq="min").append(
        pd.date_range("2024-01-03 09:30", periods=5, freq="min"))
    close = np.arange(100.0, 110.0)
    bars = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                         "close": close, "volume": np.ones(10)}, index=index)
    _, returns, origins, _ = make_samples(bars, 2, 2, "1min", "future_log_return")
    assert len(origins) == 4
    assert np.allclose(returns[0], np.log(close[origins[0] + 1:origins[0] + 3] / close[origins[0]]))
    assert all(index[i].date() == index[i + 2].date() for i in origins)


def test_intraday_samples_do_not_bridge_missing_bars():
    index = pd.date_range("2024-01-02 09:30", periods=8, freq="min").delete(3)
    close = np.arange(100.0, 107.0)
    bars = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                         "close": close, "volume": np.ones(7)}, index=index)
    _, _, origins, _ = make_samples(bars, 2, 2, "1min", "future_close")
    for origin in origins:
        assert np.all(np.diff(index[origin - 1:origin + 3]) == pd.Timedelta(minutes=1))


def test_purged_walk_forward_splits_do_not_overlap_horizon_labels():
    origins = np.arange(100)
    dates = np.array(pd.date_range("2024-01-01", periods=100).date)
    splits = rolling_splits(origins, dates, folds=5, purge=4)
    assert len(splits) == 5
    for train, valid in splits:
        assert origins[train].max() + 4 < origins[valid].min()


def test_data_gate_reports_missing_intraday_provenance():
    index = pd.date_range("2024-01-01", periods=400, freq="min")
    bars = pd.DataFrame(index=index)
    meta = {"timestamp_timezone": "naive / unspecified",
            "source": "local CSV; provenance not independently verified"}
    req = {"minute": {"minimum_months": 6, "minimum_session_coverage": 0.98}}
    reasons = data_gate("minute", bars, meta, req)
    assert any("timezone" in reason for reason in reasons)
    assert any("provenance" in reason for reason in reasons)


def test_annual_profile_smoke_writes_fold_metrics(tmp_path):
    index = pd.date_range("2000-01-01", periods=360, freq="MS")
    close = 25 * np.exp(np.cumsum(0.004 + 0.02 * np.sin(np.arange(360) / 7)))
    frame = pd.DataFrame({"timestamp": index, "open": close * .999,
                          "high": close * 1.01, "low": close * .99,
                          "close": close, "volume": np.full(360, 1000)})
    source = tmp_path / "monthly.csv"
    frame.to_csv(source, index=False)
    result = run_profile("annual", source, tmp_path / "out", lookbacks=[12],
                         requested_models=["last_close", "ridge"])
    assert result["status"] == "complete"
    assert len(result["rankings"]) == 4
    assert (tmp_path / "out" / "REPORT.md").exists()
    assert result["final_test"]["status"] == "evaluated_once_after_validation_selection"
    assert len(result["final_test"]["results"]) == 2
    assert all("MASE" in fold["metrics"] for row in result["rankings"] for fold in row["folds"])
    with pytest.raises(FileExistsError):
        run_profile("annual", source, tmp_path / "out", lookbacks=[12], requested_models=["last_close"])
