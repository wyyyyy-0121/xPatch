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


def test_candidate_csv_rejects_dates_after_the_development_boundary():
    from .acquire_kaggle_candidate import csv_metadata
    valid = b"Date,Close\n2026-06-29,100\n2026-06-30,101\n"
    assert csv_metadata(valid)["last"] == "2026-06-30"
    with pytest.raises(ValueError, match="outside"):
        csv_metadata(valid + b"2026-07-01,102\n")
    day_first = b"Date,Close\n04-03-2024,100\n15-08-2024,101\n"
    assert csv_metadata(day_first)["first"] == "2024-03-04"
    assert csv_metadata(day_first)["last"] == "2024-08-15"
    with pytest.raises(ValueError, match="outside"):
        csv_metadata(day_first + b"01-07-2026,102\n")


def test_kaggle_split_table_preserves_dates_and_ratios(tmp_path):
    from .audit_kaggle_candidates import kpatel_splits
    path = tmp_path / "actions.csv"
    path.write_text("Date,Stock Splits\n2020-08-31,4:1\n2014-06-09,7:1\n", encoding="utf8")
    receipt = {"files": [{"original_name": "AppleStockSplit.csv", "file": path.name}]}
    assert kpatel_splits(tmp_path, receipt) == [
        {"date": "2014-06-09", "ratio": 7.0}, {"date": "2020-08-31", "ratio": 4.0}]
    path.write_text("Date,Stock Splits\n2020-08-31,4:0\n", encoding="utf8")
    with pytest.raises(ValueError, match="ratio"):
        kpatel_splits(tmp_path, receipt)


def test_sec_range_audit_handles_fiscal_order_and_later_split_units():
    from datetime import date
    from .audit_sec_ranges import QUARTERS, audit_ranges
    filing = ("On June 21, 2000, the Company effected a two-for-one stock split\n"
              "Fiscal 2000 price range per common\n share..... $20.00-$10.00 $20.00-$10.00 $20.00-$10.00 $20.00-$10.00\n"
              "Fiscal 1999 price range per common\n share..... $20.00-$10.00 $20.00-$10.00 $20.00-$10.00 $20.00-$10.00\n")
    prices = pd.DataFrame({"Date": [date.fromisoformat(start).strftime("%d-%m-%Y") for _, _, start, _ in QUARTERS],
                           "High": [20 / 56] * 8, "Low": [10 / 56] * 8})
    events = [{"date": "2005-02-28", "ratio": 2.0}, {"date": "2014-06-09", "ratio": 7.0},
              {"date": "2020-08-31", "ratio": 4.0}]
    result = audit_ranges(prices, events, filing)
    assert len(result["quarters"]) == 8 and result["all_within_one_cent"]
    assert result["candidate_to_2000_filing_share_units_factor"] == 56
    with pytest.raises(ValueError, match="split chain"):
        audit_ranges(prices, events[:-1], filing)


def test_alpha_source_audit_checks_split_units_and_incomplete_quarters():
    from .audit_alpha_candidates import compare_filing_quarters, compare_raw_to_adjusted
    raw = pd.DataFrame({"date": ["2014-06-06", "2014-06-09"], "close": [700.0, 100.0]})
    adjusted = pd.DataFrame({"date": ["2014-06-06", "2014-06-09"], "close": [100.0, 100.0]})
    events = [{"date": "2014-06-09", "ratio": 7.0}]
    comparison = compare_raw_to_adjusted(raw, adjusted, events)
    assert comparison["overlap"] == 2 and comparison["max_relative_difference"] == 0.0
    adjusted.loc[1, "close"] = 102.0
    comparison = compare_raw_to_adjusted(raw, adjusted, events)
    assert comparison["above_1_percent_count"] == 1
    assert comparison["over_0_1_percent_dates"][0]["date"] == "2014-06-09"
    filing = ("On June 21, 2000, the Company effected a two-for-one stock split\n"
              "Fiscal 2000 price range per common\n share..... $20.00-$10.00 $20.00-$10.00 $20.00-$10.00 $20.00-$10.00\n"
              "Fiscal 1999 price range per common\n share..... $20.00-$10.00 $20.00-$10.00 $20.00-$10.00 $20.00-$10.00\n")
    incomplete = pd.DataFrame({"date": ["1999-11-01"], "high": [20.0], "low": [10.0]})
    result = compare_filing_quarters(incomplete, filing)
    assert result["complete_quarters"] == 0 and not result["all_compared_within_one_cent"]


def test_protocol_draft_preserves_all_twenty_tasks_and_is_not_runnable():
    from .protocol_draft import make_draft
    rows = [{**item, "primary_endpoint": item["primary"], "common_max_lookback": max(item["lookbacks"]),
             "lookback_candidates": item["lookbacks"], "origins": 100,
             "status": "candidate", "minimum_inner_fit_origins": 64, "known_legacy_overlap_origins": 0,
             "folds": [{"inner_fit_origins": 64}], "proposed_warmup_folds": None} for item in tasks()]
    admission = {"Q3_values_read": False, "experiment_scored": False,
                 "development_cutoff_new_york": "2026-06-30", "tasks": rows}
    draft = make_draft(admission, "synthetic-id")
    assert draft["status"] == "draft_not_runnable" and draft["frozen"] is False
    assert len(draft["tasks"]) == 20 and draft["neural_update_upper_bound_one_seed_all_tasks"] == 20 * 20 * 4
    admission["tasks"] = rows[:-1]
    with pytest.raises(ValueError, match="twenty"):
        make_draft(admission, "synthetic-id")


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
    from .ridge import fit, fit_many
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
    grid = fit_many(windows, scaler, origins, 2, "future_close", [1.0, 2.0, 10.0], batch_size=7)
    np.testing.assert_allclose(grid[2.0].coefficients, model.coefficients, rtol=1e-12, atol=1e-12)


def test_path_metrics_separate_endpoint_path_and_zero_direction_abstention():
    from .evaluation import endpoint_improvement, path_metrics
    y = np.array([[1.0, 2.0], [-1.0, -2.0]])
    p = np.zeros_like(y)
    metrics = path_metrics(y, p, "future_log_return")
    assert metrics["endpoint_mse"] == 4.0 and metrics["path_mse"] == 2.5
    assert metrics["direction_coverage"] == 0.0 and metrics["direction_accuracy_when_active"] is None
    result = endpoint_improvement(y, p, p)
    assert result["relative_improvement"] == 0.0


def test_independent_price_metrics_use_the_origin_price_for_direction():
    from .verify_development import independent_metrics
    actual = np.array([[101.0, 102.0], [99.0, 97.0]])
    prediction = np.array([[100.0, 101.0], [100.0, 98.0]])
    metric, terminal = independent_metrics(actual, prediction, "future_close", np.array([100.0, 100.0]))
    assert terminal == 1.0 and metric["path_mse"] == 1.0
    assert metric["direction_coverage"] == 1.0 and metric["direction_accuracy_when_active"] == 1.0


def test_group_bootstrap_and_holm_use_prespecified_units():
    from .evaluation import holm, paired_block_bootstrap
    groups = np.repeat(np.arange(10), 3)
    result = paired_block_bootstrap(np.full(30, -1.0), groups, block_length=3, resamples=200, seed=5)
    assert result["groups"] == 10 and result["origins"] == 30
    assert result["group_grid"] == "observed_groups_only; gaps not checked"
    np.testing.assert_allclose(result["confidence_interval"], [-1.0, -1.0])
    assert result["centered_one_sided_p_improvement"] == pytest.approx(1 / 201)
    np.testing.assert_allclose(holm([0.04, 0.001, 0.02]), [0.04, 0.003, 0.04])
    missing_day = paired_block_bootstrap([-1.0, -2.0], [1, 3], block_length=2,
                                         resamples=200, group_order=[1, 2, 3])
    assert missing_day["groups"] == 3 and missing_day["group_grid"] == "caller_supplied_full_order"
    with pytest.raises(ValueError, match="absent"):
        paired_block_bootstrap([-1.0, -2.0], [1, 3], block_length=2, resamples=200, group_order=[1, 2])


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


def test_complete_synthetic_development_fold_has_eight_aligned_models(tmp_path, monkeypatch):
    import json
    import torch
    from . import development
    from .development import FoldConfig, run_development_fold, validate_split
    from .verify_development import check_fold
    torch.set_num_threads(2)
    days = pd.bdate_range("2024-01-02", periods=12).strftime("%Y-%m-%d").tolist()
    bars = make_minutes([30] * len(days), days)
    origins = eligible_origins(bars, 6, 3, same_session=True)
    fold = rolling_folds(bars, origins, 3)[0]
    cfg = FoldConfig("minute", 3, "future_log_return", 6, 6, (0.1, 1.0),
                     1, 1, 16, 1e-3, 2026, "synthetic-protocol-v1")
    raw_targets = development.targets
    def guarded_targets(data, indices, horizon, mode):
        if np.array_equal(indices, fold["validation"]):
            assert all((tmp_path / "fold" / f"prediction_{name}.npz").is_file() for name in MODELS)
        return raw_targets(data, indices, horizon, mode)
    monkeypatch.setattr(development, "targets", guarded_targets)
    receipt = run_development_fold(bars, fold, [], cfg, tmp_path / "fold", "synthetic-data-v1")
    assert receipt["status"] == "completed" and not receipt["independent_test"]
    assert tuple(receipt["models"]) == MODELS
    actual = np.load(tmp_path / "fold/targets.npz", allow_pickle=False)["actual"]
    assert receipt["models"]["last_close"]["metrics"]["endpoint_mse"] == pytest.approx(np.square(actual[:, -1]).mean())
    for name in MODELS:
        prediction = np.load(tmp_path / f"fold/prediction_{name}.npz", allow_pickle=False)["prediction"]
        assert prediction.shape == actual.shape
    receipt_path = tmp_path / "fold/fold_receipt.json"
    original_receipt = receipt_path.read_text()
    disk_receipt = json.loads(original_receipt)
    assert disk_receipt["phase"] == "development" and not disk_receipt["Q3_values_read"]
    verification = check_fold(tmp_path / "fold")
    assert verification["status"] == "verified_saved_fold" and not verification["independent_test"]
    assert len(verification["models"]) == 8 and all(verification["current_source_matches_recorded"].values())
    disk_receipt["models"]["mlp"]["metrics"]["endpoint_mse"] += 1.0
    receipt_path.write_text(json.dumps(disk_receipt), encoding="utf8")
    with pytest.raises(ValueError, match="Reported metric differs"):
        check_fold(tmp_path / "fold")
    receipt_path.write_text(original_receipt, encoding="utf8")
    monkeypatch.setattr(development, "targets", raw_targets)
    changed_close = bars.close.copy(); changed_close[fold["validation"][0] + 1:] *= 1.25
    changed_bars = Bars("minute", bars.start_ns, bars.end_ns, changed_close, bars.observed, bars.session_date)
    changed = run_development_fold(changed_bars, fold, [], cfg, tmp_path / "future_changed", "synthetic-changed-v1")
    assert changed["models"]["ridge"]["alpha"] == receipt["models"]["ridge"]["alpha"]
    for name in MODELS:
        before = np.load(tmp_path / f"fold/prediction_{name}.npz", allow_pickle=False)["prediction"][0]
        after = np.load(tmp_path / f"future_changed/prediction_{name}.npz", allow_pickle=False)["prediction"][0]
        np.testing.assert_array_equal(before, after)
    contaminated = dict(fold)
    contaminated["train"] = np.r_[fold["train"], fold["validation"][0]]
    with pytest.raises(ValueError, match="overlap"):
        validate_split(bars, contaminated, 3)
    ridge_prediction = tmp_path / "fold/prediction_ridge.npz"
    with np.load(ridge_prediction, allow_pickle=False) as stored:
        altered = stored["prediction"].copy()
        saved_origins = stored["origins"].copy()
        saved_decisions = stored["decision_ns"].copy()
    altered[0, 0] += 1.0
    np.savez_compressed(ridge_prediction, prediction=altered)
    with pytest.raises(ValueError, match="Artifact identity mismatch"):
        check_fold(tmp_path / "fold")
    np.savez_compressed(ridge_prediction, prediction=altered, origins=saved_origins[::-1],
                        decision_ns=saved_decisions)
    from .bars import file_digest
    disk_receipt = json.loads(original_receipt)
    disk_receipt["models"]["ridge"]["prediction_sha256"] = file_digest(ridge_prediction)
    receipt_path.write_text(json.dumps(disk_receipt), encoding="utf8")
    with pytest.raises(ValueError, match="Prediction origins or decision times differ"):
        check_fold(tmp_path / "fold")


def test_fold_resume_after_abrupt_process_exit_matches_uninterrupted(tmp_path):
    import fcntl
    import json
    import subprocess
    import sys
    from .development import FoldConfig, run_development_fold
    from .verify_development import check_fold

    program = r'''
import os, sys
from pathlib import Path
import pandas as pd
import torch
import multiscale.development as development
from multiscale.bars import eligible_origins
from multiscale.development import FoldConfig
from multiscale.splits import rolling_folds
from multiscale.test_foundations import make_minutes

torch.set_num_threads(2)
dates = pd.bdate_range("2024-01-02", periods=18).strftime("%Y-%m-%d").tolist()
bars = make_minutes([30] * len(dates), dates)
origins = eligible_origins(bars, 6, 3, same_session=True)
fold = rolling_folds(bars, origins, 3)[0]
cfg = FoldConfig("minute", 3, "future_log_return", 6, 6, (0.1, 1.0),
                 2, 2, 8, 1e-3, 2026, "synthetic-resume-protocol")
if sys.argv[2] in ("interrupt", "before_checkpoint"):
    original = development.run_fit
    def interrupt_after_checkpoint(windows, fit, stop, config, directory, identity, **kwargs):
        if config.model == "mlp":
            if sys.argv[2] == "before_checkpoint":
                directory.mkdir(parents=True, exist_ok=False)
                os._exit(74)
            result = original(windows, fit, stop, config, directory, identity,
                              stop_after_steps=3, **kwargs)
            assert result["status"] == "interrupted_for_test"
            os._exit(73)
        return original(windows, fit, stop, config, directory, identity, **kwargs)
    development.run_fit = interrupt_after_checkpoint
if sys.argv[2] == "after_training":
    original_predict = development.predict_best
    def interrupt_after_training(checkpoint, *args, **kwargs):
        if Path(checkpoint).parent.name == "mlp":
            os._exit(75)
        return original_predict(checkpoint, *args, **kwargs)
    development.predict_best = interrupt_after_training
development.run_development_fold(bars, fold, [], cfg, Path(sys.argv[1]), "synthetic-resume-data",
                                 resume=(sys.argv[2] == "resume"))
'''
    interrupted = tmp_path / "interrupted"
    stopped = subprocess.run([sys.executable, "-c", program, str(interrupted), "interrupt"],
                             capture_output=True, text=True, timeout=45)
    assert stopped.returncode == 73, stopped.stderr
    state = json.loads((interrupted / "fold_state.json").read_text())
    assert state["status"] == "running" and list(state["models"]) == list(MODELS[:4])
    assert (interrupted / "mlp/latest.pt").is_file() and not (interrupted / "fold_receipt.json").exists()

    dates = pd.bdate_range("2024-01-02", periods=18).strftime("%Y-%m-%d").tolist()
    bars = make_minutes([30] * len(dates), dates)
    origins = eligible_origins(bars, 6, 3, same_session=True)
    fold = rolling_folds(bars, origins, 3)[0]
    wrong_cfg = FoldConfig("minute", 3, "future_log_return", 6, 6, (0.1, 1.0),
                           2, 2, 8, 1e-3, 2027, "synthetic-resume-protocol")
    correct_cfg = FoldConfig("minute", 3, "future_log_return", 6, 6, (0.1, 1.0),
                             2, 2, 8, 1e-3, 2026, "synthetic-resume-protocol")
    with (interrupted / "fold.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="already active"):
            run_development_fold(bars, fold, [], correct_cfg, interrupted, "synthetic-resume-data", resume=True)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    with pytest.raises(ValueError, match="data, split, source, runtime or configuration changed"):
        run_development_fold(bars, fold, [], wrong_cfg, interrupted, "synthetic-resume-data", resume=True)
    changed_close = bars.close.copy()
    changed_close[0] += 1.0
    changed_bars = Bars(bars.frequency, bars.start_ns, bars.end_ns, changed_close, bars.observed, bars.session_date)
    with pytest.raises(ValueError, match="data, split, source, runtime or configuration changed"):
        run_development_fold(changed_bars, fold, [], correct_cfg, interrupted, "synthetic-resume-data", resume=True)

    resumed = subprocess.run([sys.executable, "-c", program, str(interrupted), "resume"],
                             capture_output=True, text=True, timeout=60)
    assert resumed.returncode == 0, resumed.stderr
    fresh_dir = tmp_path / "uninterrupted"
    fresh = subprocess.run([sys.executable, "-c", program, str(fresh_dir), "fresh"],
                           capture_output=True, text=True, timeout=60)
    assert fresh.returncode == 0, fresh.stderr
    assert check_fold(interrupted)["status"] == "verified_saved_fold"
    assert check_fold(fresh_dir)["status"] == "verified_saved_fold"
    result = json.loads((interrupted / "fold_receipt.json").read_text())
    reference = json.loads((fresh_dir / "fold_receipt.json").read_text())
    assert result["resume_count"] == 1 and result["fold_identity"] == reference["fold_identity"]
    for name in MODELS:
        with np.load(interrupted / f"prediction_{name}.npz", allow_pickle=False) as left:
            with np.load(fresh_dir / f"prediction_{name}.npz", allow_pickle=False) as right:
                np.testing.assert_array_equal(left["prediction"], right["prediction"])
        assert result["models"][name]["metrics"] == reference["models"][name]["metrics"]

    from .bars import file_digest
    completed_identity = file_digest(interrupted / "fold_receipt.json")
    rerun = subprocess.run([sys.executable, "-c", program, str(interrupted), "resume"],
                           capture_output=True, text=True, timeout=45)
    assert rerun.returncode == 0, rerun.stderr
    assert file_digest(interrupted / "fold_receipt.json") == completed_identity

    early = tmp_path / "before_initial_checkpoint"
    stopped_early = subprocess.run([sys.executable, "-c", program, str(early), "before_checkpoint"],
                                   capture_output=True, text=True, timeout=45)
    assert stopped_early.returncode == 74, stopped_early.stderr
    assert (early / "mlp").is_dir() and not (early / "mlp/latest.pt").exists()
    resumed_early = subprocess.run([sys.executable, "-c", program, str(early), "resume"],
                                   capture_output=True, text=True, timeout=60)
    assert resumed_early.returncode == 0, resumed_early.stderr
    assert (early / "mlp").is_dir()  # Incomplete attempt is retained rather than overwritten.
    assert check_fold(early)["status"] == "verified_saved_fold"
    early_receipt = json.loads((early / "fold_receipt.json").read_text())
    assert early_receipt["models"]["mlp"]["checkpoint_dir"] == "mlp-retry-001"
    for name in MODELS:
        with np.load(early / f"prediction_{name}.npz", allow_pickle=False) as left:
            with np.load(fresh_dir / f"prediction_{name}.npz", allow_pickle=False) as right:
                np.testing.assert_array_equal(left["prediction"], right["prediction"])

    after_training = tmp_path / "after_training"
    stopped_after = subprocess.run([sys.executable, "-c", program, str(after_training), "after_training"],
                                   capture_output=True, text=True, timeout=45)
    assert stopped_after.returncode == 75, stopped_after.stderr
    assert (after_training / "mlp/training_summary.json").is_file()
    checkpoint_before = file_digest(after_training / "mlp/latest.pt")
    resumed_after = subprocess.run([sys.executable, "-c", program, str(after_training), "resume"],
                                   capture_output=True, text=True, timeout=60)
    assert resumed_after.returncode == 0, resumed_after.stderr
    assert file_digest(after_training / "mlp/latest.pt") == checkpoint_before
    assert check_fold(after_training)["status"] == "verified_saved_fold"
    for name in MODELS:
        with np.load(after_training / f"prediction_{name}.npz", allow_pickle=False) as left:
            with np.load(fresh_dir / f"prediction_{name}.npz", allow_pickle=False) as right:
                np.testing.assert_array_equal(left["prediction"], right["prediction"])


def test_shared_fold_uses_common_h60_purged_samples_for_every_model(tmp_path):
    import torch
    from .shared_development import SharedConfig, run_shared_fold
    torch.set_num_threads(2)
    days = pd.bdate_range("2024-01-02", periods=12).strftime("%Y-%m-%d").tolist()
    bars = make_minutes([100] * len(days), days)
    origins = eligible_origins(bars, 12, 60, same_session=True)
    fold = rolling_folds(bars, origins, 60)[0]
    cfg = SharedConfig(12, 1, 1, 16, 1e-3, 2026, "synthetic-matched-protocol")
    receipt = run_shared_fold(bars, fold, [], cfg, tmp_path / "matched", "synthetic-matched-bars")
    assert receipt["status"] == "completed" and not receipt["independent_test"]
    assert receipt["purge_horizon_for_every_model"] == 60
    assert set(receipt["comparisons"]) == {"5", "15", "30", "60"}
    assert len(receipt["models"]) == 5


@pytest.mark.parametrize("invalid_partition", ["train", "fit", "stop", "validation", "missing_history", "missing_target"])
def test_shared_fold_rejects_ineligible_h60_origins_before_training(tmp_path, monkeypatch, invalid_partition):
    from . import shared_development
    from .development import validate_split

    days = pd.bdate_range("2024-01-02", periods=6).strftime("%Y-%m-%d").tolist()
    missing = (0,) if invalid_partition == "missing_history" else (30,) if invalid_partition == "missing_target" else ()
    bars = make_minutes([100] * len(days), days, missing=missing)
    fit = np.arange(11, 19, dtype=np.int64)
    stop = np.array([201], dtype=np.int64)
    validation = np.array([401], dtype=np.int64)
    if invalid_partition == "fit":
        fit = np.r_[fit, 40]
    if invalid_partition == "stop":
        stop = np.array([240], dtype=np.int64)
    if invalid_partition == "validation":
        validation = np.array([440], dtype=np.int64)
    train = np.sort(np.r_[fit, stop, [40] if invalid_partition == "train" else []]).astype(np.int64)
    fold = {"train": train, "inner": {"fit": fit, "stop": stop}, "validation": validation}
    validate_split(bars, fold, 60)  # Chronology alone does not establish full H60 eligibility.
    monkeypatch.setattr(shared_development, "run_fit", lambda *args, **kwargs: pytest.fail("Training started"))
    cfg = shared_development.SharedConfig(12, 1, 1, 16, 1e-3, 2026, "synthetic-matched-protocol")
    destination = tmp_path / "must_not_start"
    with pytest.raises(ValueError, match="eligible H60"):
        shared_development.run_shared_fold(bars, fold, [], cfg, destination, "synthetic-matched-bars")
    assert not destination.exists()


def test_shared_fold_rejects_out_of_scope_split_before_training(tmp_path, monkeypatch):
    from . import shared_development

    days = pd.bdate_range("2024-01-02", periods=6).strftime("%Y-%m-%d").tolist()
    bars = make_minutes([100] * len(days), days)
    fit = np.arange(11, 19, dtype=np.int64)
    stop, validation = np.array([201]), np.array([401])
    fold = {"train": np.r_[fit, stop], "inner": {"fit": fit, "stop": stop}, "validation": validation}
    monkeypatch.setattr(shared_development, "run_fit", lambda *args, **kwargs: pytest.fail("Training started"))
    cfg = shared_development.SharedConfig(12, 1, 1, 16, 1e-3, 2026, "synthetic-matched-protocol")
    destination = tmp_path / "must_not_start"
    with pytest.raises(ValueError, match="out-of-scope split"):
        shared_development.run_shared_fold(bars, fold, [{"date": "2026-07-01", "ratio": 2.0}], cfg,
                                           destination, "synthetic-matched-bars")
    assert not destination.exists()


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
