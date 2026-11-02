"""Independent calendar, gap, target, and leakage checks for the new data path."""
import numpy as np
import pandas as pd
import pytest

from data_acquisition.download_alpaca_sip import read_credentials
from finance_v2.data import audit_bars, calendar_grid, load_market_data, quarterly_splits, read_bars
from finance_v2.data import training_dates


def make_prices(path, days=("2024-03-08", "2024-03-11", "2024-11-29")):
    grids = calendar_grid(min(days), max(days))
    stamps = grids[days[0]]
    for day in days[1:]:
        stamps = stamps.append(grids[day])
    close = 100 * np.exp(np.arange(len(stamps), dtype=float) * 1e-5 +
                         np.sin(np.arange(len(stamps))) * 1e-4)
    frame = pd.DataFrame({"timestamp": [x.isoformat() for x in stamps], "open": close,
                          "high": close + 0.02, "low": close - 0.02,
                          "close": close, "volume": 1000 + np.arange(len(stamps))})
    frame.to_csv(path, index=False)
    return frame


def test_dst_half_day_and_same_session_targets(tmp_path):
    path = tmp_path / "prices.csv"
    frame = make_prices(path)
    data = load_market_data(path)
    assert len(data.origins) == 2 * 138
    local = data.bars.index.tz_convert("America/New_York")
    assert local[0].hour == 9 and local[390].hour == 9
    assert str(data.bars.index[0]) == "2024-03-08 14:30:00+00:00"
    assert str(data.bars.index[390]) == "2024-03-11 13:30:00+00:00"
    assert not len(data.samples(["2024-11-29"]))
    o = data.origins[0]
    assert o == 240
    expected = [np.log(frame.close.iloc[o + h] / frame.close.iloc[o]) for h in (1, 5, 12)]
    np.testing.assert_allclose(data.targets([o])[0], expected, atol=1e-15)
    x = data.inputs([o], {"center": [0], "std": [1]})
    np.testing.assert_allclose(x[0, :, 0], np.diff(np.log(frame.close.iloc[:241])), rtol=1e-6)
    regular, report = audit_bars(read_bars(path), requested_dates=["2024-03-08", "2024-03-11", "2024-11-29"])
    assert len(regular) == 990 and report["coverage_passed"]
    assert report["daily"][-1]["expected"] == 210


def test_minute_gap_never_enters_history_or_target(tmp_path):
    path = tmp_path / "gap.csv"
    frame = make_prices(path, days=("2024-03-08",))
    frame.drop(index=100).to_csv(path, index=False)
    data = load_market_data(path, lookback=20)
    for o in data.origins:
        stamps = data.bars.index[o - 20:o + 13]
        assert (np.diff(stamps.as_unit("ns").asi8) == 60 * 10**9).all()
    assert data.audit["within_session_gaps"] == 1


def test_missing_entire_session_cannot_pass_calendar_audit(tmp_path):
    path = tmp_path / "missing.csv"
    make_prices(path, days=("2024-03-08",))
    _, report = audit_bars(read_bars(path), requested_dates=["2024-03-08", "2024-03-11"])
    assert not report["coverage_passed"]
    assert report["daily"][-1]["observed"] == 0


@pytest.mark.parametrize("defect", ["naive", "duplicate", "second", "negative_volume", "ohlc"])
def test_invalid_provider_input_rejected(tmp_path, defect):
    path = tmp_path / "bad.csv"
    frame = make_prices(path, days=("2024-03-08",))
    if defect == "naive":
        frame.timestamp = frame.timestamp.str[:19]
    elif defect == "duplicate":
        frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
    elif defect == "second":
        frame.loc[1, "timestamp"] = "2024-03-08T14:31:01+00:00"
    elif defect == "negative_volume":
        frame.loc[1, "volume"] = -1
    else:
        frame.loc[1, "high"] = 1
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        read_bars(path)


def test_scaler_uses_training_only_and_features_ignore_future(tmp_path):
    path = tmp_path / "prices.csv"
    frame = make_prices(path, days=("2024-03-08", "2024-03-11"))
    data = load_market_data(path, feature_set="ohlcv")
    scale = data.scaler(["2024-03-08"])
    x = data.inputs(data.samples(["2024-03-08"]), scale)
    mask = frame.timestamp.str.startswith("2024-03-11")
    frame.loc[mask, ["open", "high", "low", "close"]] *= 3
    frame.to_csv(path, index=False)
    changed = load_market_data(path, feature_set="ohlcv")
    assert scale == changed.scaler(["2024-03-08"])
    np.testing.assert_array_equal(x, changed.inputs(changed.samples(["2024-03-08"]), scale))


def test_quarterly_splits_use_last_trading_session_purge():
    config = {"purge_sessions": 1, "periods": {"development_start": "2021-01-04",
              "development_end": "2025-06-30", "validation_start": "2023-01-01"}}
    splits = quarterly_splits(config)
    assert len(splits) == 10
    assert splits[0]["name"] == "2023Q1"
    assert splits[0]["purged_dates"] == ["2022-12-30"]
    assert splits[0]["train_dates"][-1] == "2022-12-29"
    assert splits[-1]["name"] == "2025Q2"
    assert splits[-1]["kind"] == "historical_walk_forward"
    for s in splits:
        assert max(s["train_dates"]) < min(s["purged_dates"]) < min(s["eval_dates"])
        p = training_dates(s["train_dates"], {"purge_sessions": 1,
            "training": {"calibration_sessions": 20, "inner_epoch_selection_sessions": 20}})
        groups = [p[k] for k in ("early_fit", "selection_purged", "epoch_selection",
                                  "calibration_purged", "calibration", "final_fit")]
        assert p["final_fit"] == p["early_fit"] + p["epoch_selection"]
        assert not set(p["selection_purged"]) & set(p["final_fit"])
        assert not set(p["calibration_purged"]) & set(p["final_fit"])
        assert max(p["calibration"]) < min(s["purged_dates"])


def test_credentials_literal_file_does_not_execute_shell(tmp_path, monkeypatch):
    for k in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"):
        monkeypatch.delenv(k, raising=False)
    path = tmp_path / "credentials.env"
    marker = tmp_path / "must_not_exist"
    path.write_text(f'APCA_API_KEY_ID="$(touch {marker})"\nAPCA_API_SECRET_KEY="example"\n')
    key, secret = read_credentials(path)
    assert key.startswith("$(touch") and secret == "example"
    assert not marker.exists()
