"""Behavioral checks of saved-model inference on fictional exchange-grid data."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import exchange_calendars as xc
import numpy as np
import pytest
import torch

from .bars import Bars, MINUTE_NS, eligible_origins, file_digest
from .development import FoldConfig, run_development_fold
from .inference import _read_bars, export_development_bundle, load_development_bundle
from .spec import MODELS, ROOT, TARGETS, tasks
from .splits import rolling_folds


def fake_bars(frequency, start, end):
    """Fictional prices on an actual calendar; no provider or market prices."""
    schedule = xc.get_calendar("XNYS", start=start, end=end).schedule
    sessions = [(str(day.date()), int(row["open"].value), int(row["close"].value))
                for day, row in schedule.iterrows()]
    records = []
    if frequency in ("minute", "hour"):
        width = MINUTE_NS if frequency == "minute" else 60 * MINUTE_NS
        for day, opened, closed in sessions:
            records.extend((day, stamp, stamp + width) for stamp in range(opened, closed - width + 1, width))
    elif frequency == "day":
        records = sessions
    elif frequency == "month":
        groups = {}
        for day, opened, closed in sessions:
            month = day[:7]
            if month not in groups:
                groups[month] = [day, opened, closed]
            else:
                groups[month][0] = day
                groups[month][2] = closed
        records = [tuple(groups[key]) for key in sorted(groups)]
    labels = np.asarray([record[0] for record in records])
    starts = np.asarray([record[1] for record in records], dtype=np.int64)
    ends = np.asarray([record[2] for record in records], dtype=np.int64)
    position = np.arange(len(records), dtype=np.float64)
    close = 100.0 + 0.01 * position + 2.0 * np.sin(position / 11.0)
    return Bars(frequency, starts, ends, close, np.ones(len(records), dtype=bool), labels)


def before(bars, last):
    part = slice(None, last + 1)
    return Bars(bars.frequency, bars.start_ns[part], bars.end_ns[part], bars.close[part],
                bars.observed[part], bars.session_date[part])


SHAPES = [
    ("annual", "month", (12,), 12, "2000-01-01", "2024-12-31"),
    ("quarterly", "month", (3,), 6, "2012-01-01", "2024-12-31"),
    ("monthly", "day", (21,), 21, "2021-01-01", "2023-12-31"),
    ("hourly", "hour", (1, 4, 8), 12, "2024-01-02", "2024-03-01"),
    ("minute", "minute", (5, 15, 30, 60), 30, "2024-01-02", "2024-01-19"),
]
CASES = [(profile, horizon, target) for profile, _, horizons, _, _, _ in SHAPES
         for horizon in horizons for target in TARGETS]


def case_id(case):
    return f"{case[0]}-h{case[1]}-{case[2]}"


@pytest.fixture(scope="module")
def examples(tmp_path_factory):
    cache = {}
    def get(case):
        if case in cache:
            return cache[case]
        profile, horizon, target = case
        _, frequency, _, lookback, start, end = next(shape for shape in SHAPES if shape[0] == profile)
        tmp_path = tmp_path_factory.mktemp("inference-" + case_id(case))
        torch.set_num_threads(2)
        bars = fake_bars(frequency, start, end)
        origins = eligible_origins(bars, lookback, horizon, same_session=(frequency == "minute"))
        fold = rolling_folds(bars, origins, horizon)[0]
        cfg = FoldConfig(profile, horizon, target, lookback, min(lookback, 12), (0.1, 1.0),
                         1, 1, 128, 0.001, 2026, "synthetic-test-protocol")
        folder = tmp_path / "fold"
        receipt = run_development_fold(bars, fold, [], cfg, folder, "synthetic-data-only")
        assert receipt["status"] == "completed"
        bundle_path = tmp_path / "bundle"
        exported = export_development_bundle(folder, expected_receipt_sha256=file_digest(folder / "fold_receipt.json"),
                                            output_directory=bundle_path)
        assert exported["status"] == "synthetic_test_inference_bundle"
        predictor = load_development_bundle(bundle_path, expected_manifest_sha256=exported["manifest_sha256"])
        cache[case] = {"profile": profile, "frequency": frequency, "horizon": horizon, "target": target,
                       "lookback": lookback, "bars": bars, "fold": fold, "folder": folder,
                       "bundle": bundle_path, "exported": exported, "predictor": predictor}
        return cache[case]
    return get


def forecast(example, history, model):
    return example["predictor"].predict(history, profile=example["profile"], horizon=example["horizon"],
        target=example["target"], model=model, as_of_ns=int(history.end_ns[-1]),
        history_identity="fictional-synthetic-history")


def test_synthetic_cases_cover_the_whole_requested_task_inventory():
    assert set(CASES) == {(item["profile"], item["horizon"], item["target"]) for item in tasks()}


@pytest.mark.parametrize("case", CASES, ids=case_id)
def test_bundle_restores_all_eight_models_and_scheduled_paths(case, examples):
    example = examples(case)
    origin = int(example["fold"]["validation"][0])
    history = before(example["bars"], origin)
    expected_files = {"manifest.json", "ridge_coefficients.npz", "ridge_scaler.json"} | {
        name + "_best.pt" for name in ("mlp", "gru", "lstm", "xpatch")}
    assert {p.name for p in example["bundle"].iterdir()} == expected_files
    for model in MODELS:
        before_rng = torch.get_rng_state().clone()
        result = forecast(example, history, model)
        assert torch.equal(before_rng, torch.get_rng_state())
        with np.load(example["folder"] / f"prediction_{model}.npz", allow_pickle=False) as saved:
            expected = saved["prediction"][0]
        np.testing.assert_allclose(result["prediction"], expected, rtol=1e-5, atol=1e-6)
        assert len(result["future_targets"]) == example["horizon"]
        assert all(item["observed"] is None for item in result["future_targets"])
        assert result["status"] == "synthetic_test_forecast_only" and result["data_kind"] == "synthetic_test"
        assert not result["independent_test"]
        assert all(item["target_completion_ns"] is None for item in result["future_targets"])
        if example["frequency"] == "minute":
            assert all(item["session_date"] == result["origin_session_date"] for item in result["future_targets"])
        if example["profile"] == "hourly" and example["horizon"] == 8:
            assert result["future_targets"][-1]["session_date"] > result["origin_session_date"]
            assert result["future_targets"][-1]["elapsed_seconds_from_decision"] > 8 * 3600


@pytest.mark.parametrize("case", CASES, ids=case_id)
def test_bundle_rejects_incomplete_wrong_time_and_changed_artifacts(case, examples):
    example = examples(case)
    origin = int(example["fold"]["validation"][0])
    history = before(example["bars"], origin)
    earlier = before(example["bars"], origin - 1)
    with pytest.raises(ValueError, match="Decision time"):
        forecast(example, earlier, "ridge")
    with pytest.raises(ValueError, match="Requested task"):
        example["predictor"].predict(history, profile=example["profile"], horizon=example["horizon"] + 1,
            target=example["target"], model="ridge", as_of_ns=int(history.end_ns[-1]), history_identity="synthetic")
    with pytest.raises(ValueError, match="Decision time"):
        example["predictor"].predict(before(example["bars"], origin + 1), profile=example["profile"],
            horizon=example["horizon"], target=example["target"], model="ridge",
            as_of_ns=int(history.end_ns[-1]), history_identity="synthetic")
    observed = history.observed.copy()
    observed[-1] = False
    missing = Bars(history.frequency, history.start_ns, history.end_ns, history.close, observed, history.session_date)
    with pytest.raises(ValueError, match="must be observed"):
        forecast(example, missing, "ridge")
    scaler_path = example["bundle"] / "ridge_scaler.json"
    original = scaler_path.read_bytes()
    try:
        scaler_path.write_bytes(original + b" ")
        with pytest.raises(ValueError, match="changed"):
            forecast(example, history, "ridge")
    finally:
        scaler_path.write_bytes(original)


def test_minute_grid_session_boundary_and_process_restart(examples, tmp_path):
    example = examples(("minute", 5, "future_log_return"))
    origin = int(example["fold"]["validation"][0])
    history = before(example["bars"], origin)
    keep = np.ones(len(history.close), dtype=bool)
    keep[-10] = False
    missing = Bars(history.frequency, history.start_ns[keep], history.end_ns[keep], history.close[keep],
                   history.observed[keep], history.session_date[keep])
    with pytest.raises(ValueError, match="bar grid"):
        forecast(example, missing, "xpatch")
    day = history.session_date[-1]
    same_day = np.flatnonzero(example["bars"].session_date == day)
    near_close = before(example["bars"], int(same_day[-3]))
    with pytest.raises(ValueError, match="cross the regular session"):
        forecast(example, near_close, "xpatch")
    path = tmp_path / "synthetic_history.npz"
    np.savez_compressed(path, frequency=np.asarray(history.frequency), start_ns=history.start_ns,
                        end_ns=history.end_ns, close=history.close, observed=history.observed,
                        session_date=history.session_date)
    command = [sys.executable, "-m", "multiscale.inference", "forecast", "--bundle", str(example["bundle"]),
               "--manifest-sha256", example["exported"]["manifest_sha256"], "--bars", str(path),
               "--profile", "minute", "--horizon", "5", "--target", "future_log_return",
               "--model", "xpatch", "--as-of-ns", str(int(history.end_ns[-1])),
               "--history-identity", "fictional-synthetic-history"]
    child = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True, timeout=30)
    restored = json.loads(child.stdout)
    assert restored["prediction"] == forecast(example, history, "xpatch")["prediction"]
    assert restored["input_window_sha256"] == forecast(example, history, "xpatch")["input_window_sha256"]


def test_early_close_uses_the_actual_session_schedule(examples):
    example = examples(("minute", 5, "future_log_return"))
    synthetic = fake_bars("minute", "2024-11-29", "2024-11-30")
    valid = before(synthetic, len(synthetic.close) - 6)
    result = forecast(example, valid, "last_close")
    assert result["future_targets"][-1]["scheduled_session_or_bar_end_ns"] == int(synthetic.end_ns[-1])
    near_close = before(synthetic, len(synthetic.close) - 3)
    with pytest.raises(ValueError, match="cross the regular session"):
        forecast(example, near_close, "last_close")


@pytest.mark.parametrize("case", [
    ("annual", 12, "future_close"), ("quarterly", 3, "future_log_return"),
    ("monthly", 21, "future_close"), ("hourly", 1, "future_log_return"),
    ("minute", 5, "future_log_return"),
], ids=case_id)
def test_omitted_bar_cannot_be_silently_compressed(case, examples):
    example = examples(case)
    origin = int(example["fold"]["validation"][0])
    history = before(example["bars"], origin)
    keep = np.ones(len(history.close), dtype=bool)
    keep[-3] = False
    missing = Bars(history.frequency, history.start_ns[keep], history.end_ns[keep], history.close[keep],
                   history.observed[keep], history.session_date[keep])
    with pytest.raises(ValueError, match="grid|incomplete"):
        forecast(example, missing, "ridge")


def test_target_schedule_cannot_enter_protected_q3(examples):
    example = examples(("monthly", 21, "future_close"))
    synthetic = fake_bars("day", "2026-05-01", "2026-06-26")
    with pytest.raises(ValueError, match="protected cutoff"):
        forecast(example, synthetic, "last_close")


def test_protected_session_metadata_is_rejected_before_loading_price_array(tmp_path):
    path = tmp_path / "fictional-protected-date.npz"
    np.savez(path, frequency=np.asarray("day"), session_date=np.asarray(["2026-07-01"]),
             start_ns=np.asarray([0]), end_ns=np.asarray([1]), observed=np.asarray([True]),
             close=np.asarray(["SYNTHETIC_NOT_A_PRICE"], dtype=object))
    with pytest.raises(ValueError, match="protected cutoff"):
        _read_bars(path)


def test_fictional_market_identity_cannot_bypass_the_existing_review_gate(examples, tmp_path):
    example = examples(("minute", 5, "future_log_return"))
    # This is a fabricated identity on a synthetic receipt; it is never a market result.
    receipt = json.loads((example["folder"] / "fold_receipt.json").read_text(encoding="utf8"))
    claimed = "market:" + "a" * 64
    receipt["data_identity"] = claimed
    receipt["identity_fields"]["data_identity"] = claimed
    canonical = json.dumps(receipt["identity_fields"], sort_keys=True, separators=(",", ":"),
                           allow_nan=False).encode("utf8")
    receipt["fold_identity"] = hashlib.sha256(canonical).hexdigest()
    fake = tmp_path / "fictional-market-identity"
    fake.mkdir()
    path = fake / "fold_receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf8")
    output = tmp_path / "must-not-be-created"
    with pytest.raises(ValueError, match="require the reviewed admission"):
        export_development_bundle(fake, expected_receipt_sha256=file_digest(path), output_directory=output)
    assert not output.exists()
