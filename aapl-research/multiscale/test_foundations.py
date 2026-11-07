"""Focused tests for task scope, label timing, missing bars and purged splits."""
import numpy as np
import pandas as pd
import pytest

from .bars import Bars, MINUTE_NS, aggregate_intraday, aggregate_months, eligible_origins, targets
from .baselines import predict as baseline_predict
from .features import CloseWindows, fit_scaler
from .spec import MODELS, PROFILES, tasks
from .splits import rolling_folds, warmup_folds


def make_minutes(lengths, dates, missing=()):
    stamps, labels = [], []
    for length, date in zip(lengths, dates):
        start = pd.Timestamp(date + " 09:30", tz="America/New_York")
        ts = pd.date_range(start, periods=length, freq="min").tz_convert("UTC").as_unit("ns").asi8
        stamps.extend(ts); labels.extend([date] * length)
    stamps = np.asarray(stamps, dtype=np.int64)
    close = np.arange(100, 100 + len(stamps), dtype=np.float64)
    observed = np.ones(len(stamps), dtype=bool)
    observed[list(missing)] = False
    return Bars("minute", stamps, stamps + MINUTE_NS, close, observed, np.asarray(labels))


def test_scope_has_twenty_tasks_and_eight_named_models():
    from datetime import date
    from .acquire_daily import date_in_scope
    assert len(tasks()) == 20
    assert len({(x["profile"], x["horizon"], x["target"]) for x in tasks()}) == 20
    assert len(MODELS) == 8 and len(PROFILES) == 5
    assert {x["horizon"] for x in tasks() if x["profile"] == "minute"} == {5, 15, 30, 60}
    assert date_in_scope(date(2026, 6, 30)) and not date_in_scope(date(2026, 7, 1))


def test_minute_labels_are_cumulative_and_do_not_cross_target_session():
    bars = make_minutes([3, 3], ["2024-11-27", "2024-11-29"])
    origins = eligible_origins(bars, 2, 2, same_session=True)
    assert origins.tolist() == [3]  # history may cross sessions; future labels may not
    np.testing.assert_array_equal(targets(bars, origins, 2, "future_close"), [[104, 105]])
    np.testing.assert_allclose(targets(bars, origins, 2, "future_log_return"),
                               [[np.log(104 / 103), np.log(105 / 103)]])
    with pytest.raises(ValueError, match="Unknown"):
        targets(bars, origins, 2, "wrong_target")
    with pytest.raises(ValueError, match="same session"):
        targets(bars, [1], 2, "future_close")


def test_missing_minute_invalidates_both_context_and_future_path():
    bars = make_minutes([12], ["2024-11-27"], missing=[7])
    origins = eligible_origins(bars, 3, 2, same_session=True)
    assert all(not (o - 2 <= 7 <= o + 2) for o in origins)
    with pytest.raises(ValueError, match="observed"):
        targets(bars, [5], 2, "future_close")


def test_hourly_regular_and_early_close_bins_have_real_completion_times():
    bars = make_minutes([390, 210], ["2024-11-27", "2024-11-29"], missing=[390])
    hours, days, metadata = aggregate_intraday(bars)
    assert len(hours.close) == 9  # six complete hours, then three on the half day
    assert metadata["residual_minutes_total"] == 60
    assert metadata["hours_per_session_counts"] == {"3": 1, "6": 1}
    assert hours.end_ns[0] - hours.start_ns[0] == 60 * MINUTE_NS
    assert hours.session_date[5] == "2024-11-27" and hours.session_date[6] == "2024-11-29"
    assert not hours.observed[6] and not days.observed[1]
    assert 5 not in eligible_origins(hours, 1, 1, same_session=False)


def test_bars_roundtrip_without_pickle(tmp_path):
    from .admission import save_bars
    bars = make_minutes([3], ["2024-11-27"])
    path = tmp_path / "bars.npz"
    save_bars(path, bars)
    with np.load(path, allow_pickle=False) as value:
        assert value["session_date"].dtype.kind == "U"
        np.testing.assert_array_equal(value["close"], bars.close)


def test_month_marks_a_partial_exchange_month_unobserved():
    import exchange_calendars as xc
    calendar = xc.get_calendar("XNYS", start="2021-01-01", end="2021-03-01").schedule
    schedule = calendar.loc["2021-01-01":"2021-02-15"]
    starts = pd.DatetimeIndex(pd.to_datetime(schedule["open"], utc=True)).as_unit("ns").asi8
    ends = pd.DatetimeIndex(pd.to_datetime(schedule["close"], utc=True)).as_unit("ns").asi8
    dates = schedule.index.strftime("%Y-%m-%d").to_numpy(dtype="U10")
    day = Bars("day", starts, ends, np.full(len(starts), 100.0), np.ones(len(starts), dtype=bool), dates)
    month = aggregate_months(day)
    assert month.session_date.tolist() == ["2021-01-29", "2021-02-12"]
    assert month.observed.tolist() == [True, False]


def test_daily_final_bar_completion_does_not_cross_q3_cutoff():
    from .admission import daily_bars
    values = pd.DataFrame({"date": ["2026-06-29", "2026-06-30"], "close": [100.0, 101.0]})
    bars, audit = daily_bars(values)
    assert bars.session_date.tolist() == ["2026-06-29", "2026-06-30"]
    assert bars.observed.tolist() == [True, False]
    assert audit["cutoff_excluded_sessions"] == ["2026-06-30"]
    assert pd.Timestamp(int(bars.end_ns[0]), tz="UTC").tz_convert("America/New_York").isoformat() == "2026-06-30T00:00:00-04:00"


def test_outer_and_inner_purge_by_real_completion_times():
    dates = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
    starts = dates.as_unit("ns").asi8
    ends = starts + 60 * MINUTE_NS
    bars = Bars("day", starts, ends, np.full(100, 100.0), np.ones(100, dtype=bool),
                dates.strftime("%Y-%m-%d").to_numpy(dtype="U10"))
    origins = eligible_origins(bars, 1, 5, same_session=False)
    folds = rolling_folds(bars, origins, 5)
    assert len(folds) == 5
    for fold in folds:
        assert len(fold["train"]) and len(fold["validation"])
        assert bars.end_ns[fold["train"] + 5].max() < bars.end_ns[fold["validation"][0]]
        inner = fold["inner"]
        if inner["status"] == "ok":
            assert bars.end_ns[inner["fit"] + 5].max() < bars.end_ns[inner["stop"][0]]


def test_warmup_five_folds_are_selected_from_sample_counts_without_scores():
    dates = pd.date_range("2016-01-31", periods=120, freq="ME", tz="UTC")
    starts = dates.as_unit("ns").asi8
    bars = Bars("month", starts, starts + 60 * MINUTE_NS, np.full(len(dates), 100.0),
                np.ones(len(dates), dtype=bool), dates.strftime("%Y-%m-%d").to_numpy(dtype="U10"))
    origins = eligible_origins(bars, 36, 3, same_session=False)
    assert len(rolling_folds(bars, origins, 3)[0]["inner"]["fit"]) < 8
    folds = warmup_folds(bars, origins, 3, min_inner_fit=24, min_validation_groups=6)
    assert len(folds) == 5 and min(len(f["inner"]["fit"]) for f in folds) >= 24
    assert min(f["validation_groups"] for f in folds) >= 6
    for fold in folds:
        assert bars.end_ns[fold["train"][-1] + 3] < bars.end_ns[fold["validation"][0]]


def test_known_split_adjusts_only_past_history_and_keeps_nominal_targets():
    dates = pd.date_range("2020-08-27", periods=6, freq="D", tz="UTC")
    start = dates.as_unit("ns").asi8
    labels = dates.strftime("%Y-%m-%d").to_numpy(dtype="U10")
    bars = Bars("day", start, start + 60 * MINUTE_NS,
                np.array([100, 102, 51, 52, 53, 55], dtype=float), np.ones(6, dtype=bool), labels)
    actions = [{"date": "2020-08-29", "ratio": 2.0}]
    windows = CloseWindows(bars, 2, actions)
    no_future_knowledge = CloseWindows(bars, 2, [])
    before, _ = windows.arrays([1]); before_without_event, _ = no_future_knowledge.arrays([1])
    np.testing.assert_allclose(before, before_without_event)
    after, anchor = windows.arrays([2])
    np.testing.assert_allclose(after, [[0.0, 0.0]], atol=1e-14)  # 102 before the split becomes 51
    np.testing.assert_allclose(anchor, [[51.0]])
    np.testing.assert_allclose(targets(bars, [1], 1, "future_close"), [[51.0]])
    np.testing.assert_allclose(targets(bars, [1], 1, "future_log_return"), [[np.log(0.5)]])


def test_scaler_ignores_values_after_the_fit_labels():
    dates = pd.date_range("2024-01-01", periods=12, freq="D", tz="UTC")
    start = dates.as_unit("ns").asi8
    labels = dates.strftime("%Y-%m-%d").to_numpy(dtype="U10")
    prices = np.arange(100, 112, dtype=float)
    first = Bars("day", start, start + 60 * MINUTE_NS, prices, np.ones(12, dtype=bool), labels)
    changed = prices.copy(); changed[6:] *= 10
    second = Bars("day", start, start + 60 * MINUTE_NS, changed, np.ones(12, dtype=bool), labels)
    origins = np.array([1, 2, 3, 4])
    a = fit_scaler(CloseWindows(first, 2), origins, 1, "future_close")
    b = fit_scaler(CloseWindows(second, 2), origins, 1, "future_close")
    assert a == b
    assert a.fitted_origin_count == 4


def test_baselines_use_origin_anchor_and_past_only():
    bars = make_minutes([9], ["2024-11-27"])
    windows = CloseWindows(bars, 4)
    last = baseline_predict("last_close", windows, [5], 2, "future_close")
    np.testing.assert_allclose(last, [[105, 105]])
    np.testing.assert_allclose(baseline_predict("last_close", windows, [5], 2, "future_log_return"), [[0, 0]])
    np.testing.assert_allclose(baseline_predict("drift", windows, [5], 2, "future_close"), [[106, 107]])
    np.testing.assert_allclose(baseline_predict("seasonal", windows, [5], 2, "future_close", period=2), [[104, 105]])


def test_streamed_ridge_matches_sklearn_with_unpenalized_intercept():
    from sklearn.linear_model import Ridge
    from .ridge import fit
    bars = make_minutes([60], ["2024-11-27"])
    windows = CloseWindows(bars, 4)
    origins = np.arange(3, 50)
    scaler = fit_scaler(windows, origins, 2, "future_close", batch_size=7)
    model = fit(windows, scaler, origins, 2, "future_close", alpha=2.0, batch_size=7)
    x, anchor = windows.arrays(origins)
    x, anchor = scaler.transform_x(x, anchor)
    y = scaler.transform_y(targets(bars, origins, 2, "future_close"))
    sklearn = Ridge(alpha=2.0).fit(np.column_stack((x, anchor)).astype(np.float64), y.astype(np.float64))
    expected = scaler.invert_y(sklearn.predict(np.column_stack((x, anchor)).astype(np.float64)))
    np.testing.assert_allclose(model.predict(windows, scaler, origins), expected, rtol=1e-10, atol=1e-9)


def test_path_metrics_separate_endpoint_path_and_zero_direction_abstention():
    from .evaluation import endpoint_improvement, path_metrics
    y = np.array([[1.0, 2.0], [-1.0, -2.0]])
    p = np.zeros_like(y)
    metrics = path_metrics(y, p, "future_log_return")
    assert metrics["endpoint_mse"] == 4.0 and metrics["path_mse"] == 2.5
    assert metrics["direction_coverage"] == 0.0 and metrics["direction_accuracy_when_active"] is None
    result = endpoint_improvement(y, p, p)
    assert result["relative_improvement"] == 0.0


def test_group_bootstrap_and_holm_use_prespecified_units():
    from .evaluation import holm, paired_block_bootstrap
    groups = np.repeat(np.arange(10), 3)
    result = paired_block_bootstrap(np.full(30, -1.0), groups, block_length=3, resamples=200, seed=5)
    assert result["groups"] == 10 and result["origins"] == 30
    np.testing.assert_allclose(result["confidence_interval"], [-1.0, -1.0])
    assert result["centered_one_sided_p_improvement"] == pytest.approx(1 / 201)
    np.testing.assert_allclose(holm([0.04, 0.001, 0.02]), [0.04, 0.003, 0.04])


def test_shared_horizon_objective_weights_prefix_paths_equally():
    import torch
    from .mechanism import shared_path_loss
    target = torch.zeros(2, 60)
    prediction = target.clone(); prediction[:, -1] = 1
    assert shared_path_loss(prediction, target).item() == pytest.approx(1 / (4 * 60))
    with pytest.raises(ValueError, match="Horizons"):
        shared_path_loss(prediction, target, (5, 60, 15))


def test_shared_xpatch_trains_on_matched_synthetic_sixty_step_paths(tmp_path):
    import torch
    from .training import TrainConfig, run_fit
    torch.set_num_threads(2)
    bars = make_minutes([240], ["2024-11-27"])
    cfg = TrainConfig("xpatch", 6, 60, "future_log_return", 1, 1, 16, 1e-3, 2026,
                      checkpoint_batches=20, shared_horizons=(5, 15, 30, 60))
    outcome = run_fit(CloseWindows(bars, 6), np.arange(5, 70), np.arange(135, 150), cfg,
                      tmp_path / "shared", "synthetic-shared-v1")
    assert outcome["status"] == "completed"
    assert outcome["inner_stop_definition"] == "equal mean standardized endpoint MSE across shared horizons"


def test_legacy_exposure_ledger_merges_and_detects_label_overlap(tmp_path):
    import json
    from .exposure import legacy_ledger, overlapping_origin_count
    p = tmp_path / "ledger.json"
    p.write_text(json.dumps([
        {"symbol": "AAPL", "start": "2024-01-02", "end": "2024-01-05", "status": "exposed"},
        {"symbol": "AAPL", "start": "2024-01-04", "end": "2024-01-07", "status": "exposed"},
    ]))
    ledger = legacy_ledger(p)
    assert ledger["merged_intervals"] == [{"start": "2024-01-02", "end": "2024-01-07"}]
    dates = np.array(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-08", "2024-01-09"])
    assert overlapping_origin_count(dates, [0, 2, 3], 1, ledger) == 2


def test_multi_epoch_resume_preserves_sampler_rng_optimizer_and_early_stopping(tmp_path):
    import json
    import subprocess
    import sys
    import torch
    from .training import TrainConfig, predict_best, run_fit
    torch.set_num_threads(2)
    bars = make_minutes([120], ["2024-11-27"])
    windows = CloseWindows(bars, 6)
    fit, stop = np.arange(5, 65), np.arange(70, 95)
    cfg = TrainConfig("mlp", 6, 3, "future_log_return", 3, 5, 8, 1e-3, 2026, checkpoint_batches=5)
    complete = run_fit(windows, fit, stop, cfg, tmp_path / "complete", "synthetic-bars-v1")
    partial = run_fit(windows, fit, stop, cfg, tmp_path / "resumed", "synthetic-bars-v1", stop_after_steps=3)
    assert partial["status"] == "interrupted_for_test" and partial["steps"] == 3
    program = '''
import json, sys
import numpy as np
import torch
from pathlib import Path
from multiscale.test_foundations import make_minutes
from multiscale.features import CloseWindows
from multiscale.training import TrainConfig, run_fit
torch.set_num_threads(2)
bars = make_minutes([120], ["2024-11-27"])
cfg = TrainConfig("mlp", 6, 3, "future_log_return", 3, 5, 8, 1e-3, 2026, checkpoint_batches=5)
result = run_fit(CloseWindows(bars, 6), np.arange(5,65), np.arange(70,95), cfg,
                 Path(sys.argv[1]), "synthetic-bars-v1", resume=True)
print(json.dumps(result))
'''
    child = subprocess.run([sys.executable, "-c", program, str(tmp_path / "resumed")],
                           capture_output=True, text=True, timeout=30, check=True)
    restored = json.loads(child.stdout)
    assert restored == complete
    a = torch.load(tmp_path / "complete/latest.pt", map_location="cpu", weights_only=True)
    b = torch.load(tmp_path / "resumed/latest.pt", map_location="cpu", weights_only=True)
    for name in a["model"]:
        torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)
    forecast = predict_best(tmp_path / "complete/best.pt", windows, [55, 60], complete["identity"])
    prices = bars.close.copy(); prices[61:] *= 7
    future_changed = Bars("minute", bars.start_ns, bars.end_ns, prices, bars.observed, bars.session_date)
    changed = predict_best(tmp_path / "complete/best.pt", CloseWindows(future_changed, 6),
                           [55, 60], complete["identity"])
    np.testing.assert_array_equal(forecast, changed)
    with pytest.raises(ValueError, match="changed"):
        run_fit(windows, fit, stop, cfg, tmp_path / "resumed", "changed-identity", resume=True)


@pytest.mark.parametrize("name", ["mlp", "gru", "lstm", "xpatch"])
@pytest.mark.parametrize("horizon", [1, 3, 12, 21, 60])
def test_neural_models_support_each_horizon_and_have_gradients(name, horizon):
    import torch
    from .models import make_model
    torch.manual_seed(2026)
    model = make_model(name, lookback=12, horizon=horizon)
    x = torch.randn(4, 12, 1)
    anchor = torch.randn(4, 1)
    prediction = model(x, anchor)
    assert prediction.shape == (4, horizon)
    prediction.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    if name == "xpatch" and horizon == 1:
        assert model.native_horizon == 2
