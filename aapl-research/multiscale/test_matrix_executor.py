"""Check queue preflight, synthetic split binding, and conservative timing."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from . import matrix_executor
from .admission import save_bars
from .bars import eligible_origins, file_digest
from .splits import rolling_folds
from .spec import PROFILES
from .test_foundations import make_minutes
from .training import digest_indices


def test_reconstructs_only_the_frozen_synthetic_origins_and_folds(tmp_path):
    dates = pd.bdate_range("2024-01-02", periods=140).strftime("%Y-%m-%d").tolist()
    bars = make_minutes([100] * len(dates), dates)
    origins = eligible_origins(bars, 480, 60, same_session=True)
    fold = rolling_folds(bars, origins, 60)[0]
    path = tmp_path / "bars_minute.npz"
    save_bars(path, bars)
    origin_path = tmp_path / "origins_minute_h60.npy"
    np.save(origin_path, origins, allow_pickle=False)
    admission = {"files": {path.name: file_digest(path), origin_path.name: file_digest(origin_path)}}
    job = {"profile": "minute", "frequency": "minute", "horizon": 60, "fold": 1,
           "fold_strategy": "rolling", "warmup": {"min_inner_fit": 24, "min_inner_stop": 4,
                                                    "min_validation_groups": 6},
           "fit_origins": len(fold["inner"]["fit"]), "stop_origins": len(fold["inner"]["stop"]),
           "validation_origins": len(fold["validation"]),
           "train_indices_sha256": digest_indices(fold["train"]),
           "fit_indices_sha256": digest_indices(fold["inner"]["fit"]),
           "stop_indices_sha256": digest_indices(fold["inner"]["stop"]),
           "validation_indices_sha256": digest_indices(fold["validation"])}
    actual_bars, actual_fold = matrix_executor.materialize(tmp_path, admission, job)
    assert actual_bars.frequency == "minute"
    np.testing.assert_array_equal(actual_fold["validation"], fold["validation"])
    changed = deepcopy(job); changed["validation_indices_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="splits differ"):
        matrix_executor.materialize(tmp_path, admission, changed)


def test_cutoff_is_checked_before_price_array_is_opened(tmp_path, monkeypatch):
    path = tmp_path / "bars_day.npz"
    np.savez_compressed(path, start_ns=np.array([1, 3]), end_ns=np.array([2, 4]),
                        close=np.array([100.0, 101.0]), observed=np.array([True, True]),
                        session_date=np.array(["2026-06-30", "2026-07-01"]))
    admission = {"files": {path.name: file_digest(path)}}
    original_load = matrix_executor.np.load

    class Guard:
        def __init__(self, inner):
            self.inner = inner
            self.files = inner.files

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.inner.close()

        def __getitem__(self, key):
            if key == "close":
                raise AssertionError("Cutoff check must precede opening prices")
            return self.inner[key]

    monkeypatch.setattr(matrix_executor.np, "load", lambda *args, **kwargs: Guard(original_load(*args, **kwargs)))
    with pytest.raises(ValueError, match="protected cutoff"):
        matrix_executor.read_bars(tmp_path, admission, "day")


def test_missing_review_blocks_execution_before_run_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(matrix_executor, "OUT", tmp_path)
    admission_dir = tmp_path / "admission" / "synthetic"
    admission_dir.mkdir(parents=True)
    (admission_dir / "admission.json").write_text(json.dumps({"Q3_values_read": False, "experiment_scored": False,
                                                                "development_cutoff_new_york": "2026-06-30"}))
    review = tmp_path / "unapproved.json"
    review.write_text("{}")
    protocol = tmp_path / "synthetic-protocol.json"
    protocol.write_text(json.dumps({"selected_profiles": [p.name for p in PROFILES]}))
    with pytest.raises(ValueError, match="Matrix admission blocked"):
        matrix_executor.execute(admission_dir, review, protocol)
    assert not (tmp_path / "matrix_runs").exists()


def test_restart_counts_elapsed_time_and_refuses_backwards_clock():
    active = {"boot_id": "same", "started_monotonic": 100.0, "started_wall": 200.0}
    assert matrix_executor._recovered_elapsed(active, "same", 125.0, 900.0) == 25.0
    assert matrix_executor._recovered_elapsed(active, "new", 5.0, 260.0) == 60.0
    with pytest.raises(ValueError, match="elapsed matrix time"):
        matrix_executor._recovered_elapsed(active, "new", 5.0, 190.0)


def test_actual_observed_history_must_meet_profile_minimum():
    dates = pd.bdate_range("2024-01-02", periods=35).strftime("%Y-%m-%d").tolist()
    bars = make_minutes([100] * len(dates), dates)
    profile = next(p for p in PROFILES if p.name == "minute")
    with pytest.raises(ValueError, match="history does not meet"):
        matrix_executor._check_history(bars, profile)
