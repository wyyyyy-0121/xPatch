"""Economic gate invariants, independent source labels and zero lookahead claims."""
import numpy as np
import pandas as pd
import pytest

from finance.evaluation import compare, gates
from finance.io import ROOT, read_json
from finance.verify import reference_targets
from test_finance_rv_data import frames


def test_independent_csv_price_differences_match_known_30_minute_variance():
    source = frames("2024-03-08", "2024-03-11")["AAPL"]
    origins = pd.DatetimeIndex(["2024-03-08 14:45:00+00:00", "2024-03-11 13:45:00+00:00"])
    np.testing.assert_allclose(reference_targets(source.close, origins), [30e-6, 30e-6], rtol=1e-9)
    with pytest.raises(ValueError, match="nanoseconds"):
        reference_targets(source.close, origins.as_unit("us").asi8)


def test_ensemble_significance_cannot_override_a_bad_random_seed():
    dates = []
    for quarter in pd.period_range("2023Q1", "2025Q2", freq="Q"):
        dates.extend(pd.bdate_range(quarter.start_time, periods=10).strftime("%Y-%m-%d"))
    dates = np.array(dates)
    n = len(dates)
    y = np.full(n, 1e-5)
    candidate = {"origins": np.arange(n), "timestamps": pd.to_datetime(dates, utc=True).asi8,
        "dates": dates, "y": y, "mean": y, "seeds": {str(seed): y.copy() for seed in (2026, 2027, 2028)}}
    baseline = {**candidate, "mean": y * 2}
    config = read_json(ROOT / "configs/rv30.json")
    good = compare(candidate, baseline, config)
    result = gates({"aapl/xpatch": good}, config)
    assert result["aapl/xpatch"]["passed"]
    assert len(result) == 10  # The nine missing comparisons are retained at p=1.
    candidate["seeds"]["2028"] = y * 10
    bad = gates({"aapl/xpatch": compare(candidate, baseline, config)}, config)
    assert not bad["aapl/xpatch"]["passed"]
    assert not bad["aapl/xpatch"]["gate_checks"]["all_seed_pooled_and_segments_improve"]
