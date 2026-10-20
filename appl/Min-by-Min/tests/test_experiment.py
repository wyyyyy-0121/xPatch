import sys
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment import (aggregate_bars, data_gate, holdout_indices, load_ohlcv,
                        make_samples, rolling_splits, run_profile, scale_fold)


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


def test_sidecar_requires_matching_hash_before_trusting_intraday_source(tmp_path):
    path = tmp_path / "intraday.csv"
    path.write_text("timestamp,open,high,low,close,volume\n2024-01-02 09:30,10,11,9,10,1\n")
    manifest = {"source_verified": True, "source": "Provider AAPL minute bars",
                "timezone": "America/New_York",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path.with_suffix(".csv.manifest.json").write_text(json.dumps(manifest))
    _, meta = load_ohlcv(path)
    assert meta["source_verified"] is True
    assert meta["timestamp_timezone"] == "America/New_York"
    path.write_text(path.read_text() + "2024-01-02 09:31,10,11,9,10,1\n")
    with pytest.raises(ValueError, match="SHA-256"):
        load_ohlcv(path)


def test_future_targets_are_aligned_and_intraday_sequences_stay_in_session():
    index = pd.date_range("2024-01-02 09:30", periods=5, freq="min").append(
        pd.date_range("2024-01-03 09:30", periods=5, freq="min"))
    close = np.arange(100.0, 110.0)
    bars = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                         "close": close, "volume": np.ones(10)}, index=index)
    _, returns, origins, _ = make_samples(bars, 2, 2, "1min", "future_log_return")
    _, prices, price_origins, _ = make_samples(bars, 2, 2, "1min", "future_close")
    assert len(origins) == 4
    assert np.array_equal(origins, price_origins)
    assert np.allclose(returns[0], np.log(close[origins[0] + 1:origins[0] + 3] / close[origins[0]]))
    assert np.array_equal(prices[0], close[origins[0] + 1:origins[0] + 3])
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


def test_final_holdout_is_after_development_and_purged():
    origins = np.arange(200)
    train, test = holdout_indices(origins, horizon=8)
    assert origins[test[0]] >= 160
    assert origins[train].max() + 8 < origins[test].min()
    development = np.arange(len(train))
    folds = rolling_splits(origins[train], np.arange(len(train)), folds=5, purge=8)
    assert len(folds) == 5
    assert max(origins[train[v]].max() for _, v in folds) + 8 < origins[test].min()


def test_holdout_and_validation_periods_are_shared_across_lookbacks():
    cutoff = 160
    ranges = [(65, 84), (84, 103), (103, 122), (122, 141), (141, cutoff)]
    origins_by_window = [np.arange(12, 200), np.arange(48, 200)]
    validation_origins = []
    for origins in origins_by_window:
        train, test = holdout_indices(origins, horizon=8, cutoff_origin=cutoff)
        assert origins[test[0]] == cutoff
        assert origins[train].max() + 8 < cutoff
        splits = rolling_splits(origins[train], np.arange(len(train)), purge=8,
                                validation_ranges=ranges)
        assert len(splits) == 5
        fold_origins = []
        for (start, end), (_, valid) in zip(ranges, splits):
            valid_origins = origins[train][valid]
            assert valid_origins.min() >= start
            assert valid_origins.max() < end
            fold_origins.append(valid_origins)
        validation_origins.append(fold_origins)
    for short, long in zip(validation_origins[0], validation_origins[1]):
        assert np.array_equal(short, long)


def test_scalers_are_fit_only_on_training_partition():
    rng = np.random.default_rng(10)
    x_train = rng.normal(size=(20, 4, 5)).astype("float32")
    y_train = rng.normal(size=(20, 3))
    x_eval = rng.normal(size=(5, 4, 5)).astype("float32")
    train_a, _, target_a, scaler_a = scale_fold(x_train, x_eval, y_train)
    changed_eval = x_eval + 1000
    train_b, _, target_b, scaler_b = scale_fold(x_train, changed_eval, y_train)
    assert np.array_equal(train_a, train_b)
    assert np.array_equal(target_a, target_b)
    assert np.array_equal(scaler_a.mean_, scaler_b.mean_)


def test_profiles_produce_expected_window_and_horizon_shapes():
    profiles = __import__("json").loads((Path(__file__).resolve().parents[1] / "profiles.json").read_text())
    for name, profile in profiles["profiles"].items():
        interval = profile["bar_interval"]
        lookback, horizon = profile["lookbacks"][0], profile["horizons"][0]
        if interval == "1mo":
            index = pd.date_range("2000-01-01", periods=lookback + horizon + 24, freq="MS")
        elif interval == "1d":
            index = pd.date_range("2020-01-01", periods=lookback + horizon + 24, freq="B")
        elif interval == "1h":
            index = pd.DatetimeIndex([stamp for day in pd.date_range("2024-01-01", periods=20, freq="D")
                                      for stamp in pd.date_range(day + pd.Timedelta(hours=4), periods=16, freq="h")])
        else:
            freq = "min" if interval == "1min" else "5min"
            index = pd.date_range("2024-01-02 04:00", periods=lookback + horizon + 4, freq=freq)
        close = np.linspace(100, 120, len(index))
        bars = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                             "close": close, "volume": np.full(len(index), 10)}, index=index)
        X, y, _, _ = make_samples(bars, lookback, horizon, interval, "future_close")
        assert X.shape[1:] == (lookback, 5), name
        assert y.shape[1] == horizon, name
        assert len(X) > 0, name


def test_five_minute_profile_maps_minutes_to_bar_counts():
    index = pd.date_range("2024-01-02 04:00", periods=30, freq="5min")
    close = np.arange(100.0, 130.0)
    bars = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                         "close": close, "volume": np.ones(30)}, index=index)
    X, y, _, _ = make_samples(bars, 6, 3, "5min", "future_close")
    assert X.shape[1:] == (6, 5)
    assert y.shape[1] == 3


def test_ohlcv_resampling_preserves_aggregate_fields():
    index = pd.date_range("2024-01-02 09:30", periods=120, freq="min")
    close = np.arange(100.0, 220.0)
    minute = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                           "close": close, "volume": np.ones(120)}, index=index)
    five_min = aggregate_bars(minute, "5min")
    hourly = aggregate_bars(minute, "1h")
    assert len(five_min) == 24
    assert five_min.iloc[0].open == 100 and five_min.iloc[0].close == 104
    assert five_min.iloc[0].high == 105 and five_min.iloc[0].low == 99
    assert five_min.iloc[0].volume == 5
    assert len(hourly) == 2


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
    assert len(result["experiment_config"]["validation_boundaries"]) == 5
    assert result["experiment_config"]["holdout_boundary"]
    assert all("MASE" in fold["metrics"] for row in result["rankings"] for fold in row["folds"])
    with pytest.raises(FileExistsError):
        run_profile("annual", source, tmp_path / "out", lookbacks=[12], requested_models=["last_close"])
