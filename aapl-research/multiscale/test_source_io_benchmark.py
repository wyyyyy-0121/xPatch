"""Test read-only I/O timing and cutoff guards with fictional bars."""
import json

import numpy as np
import pandas as pd
import pytest

from . import source_io_benchmark as io_benchmark
from .bars import file_digest


def candidate(tmp_path, dates):
    directory = tmp_path / "admission" / "synthetic"
    directory.mkdir(parents=True)
    starts = pd.to_datetime(dates, utc=True).as_unit("ns").asi8
    path = directory / "bars_day.npz"
    np.savez_compressed(path, start_ns=starts, end_ns=starts + 60_000_000_000,
                        close=np.full(len(dates), 100.0), observed=np.ones(len(dates), dtype=bool),
                        session_date=np.asarray(dates))
    receipt = {"Q3_values_read": False, "experiment_scored": False,
               "development_cutoff_new_york": "2026-06-30", "files": {path.name: file_digest(path)}}
    (directory / "admission.json").write_text(json.dumps(receipt), encoding="utf8")
    return directory


def test_io_measurement_is_labeled_and_does_not_score(tmp_path, monkeypatch):
    monkeypatch.setattr(io_benchmark, "OUT", tmp_path)
    directory = candidate(tmp_path, ["2024-01-02", "2024-01-03"])
    result = io_benchmark.measure(directory, ["day"], repetitions=1)
    assert result["status"] == "candidate_read_only_io_not_admitted"
    assert result["model_trained"] is False and result["experiment_scored"] is False
    assert result["records"][0]["bar_count"] == 2
    assert "close" not in result["records"][0]


def test_io_measurement_rejects_post_cutoff_before_reading_price_values(tmp_path, monkeypatch):
    monkeypatch.setattr(io_benchmark, "OUT", tmp_path)
    directory = candidate(tmp_path, ["2026-06-30", "2026-07-01"])
    original_load = io_benchmark.np.load

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
                raise AssertionError("Post-cutoff metadata must be rejected before price loading")
            return self.inner[key]

    monkeypatch.setattr(io_benchmark.np, "load", lambda *args, **kwargs: Guard(original_load(*args, **kwargs)))
    with pytest.raises(ValueError, match="protected cutoff"):
        io_benchmark.measure(directory, ["day"], repetitions=1)
