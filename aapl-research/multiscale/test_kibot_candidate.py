"""Check the guest-source response boundary using synthetic CSV only."""
from datetime import date
from io import BytesIO

import pytest
import pandas as pd

from .acquire_kibot_candidate import OutOfScopeDate, parse_response
from .audit_kibot_candidate import historical_unit_diagnostics, month_end_comparison


def test_guest_parser_accepts_only_bounded_valid_rows_and_sorts_them():
    data = BytesIO(b"09/30/2015,10,12,9,11,100\n01/02/2015,8,10,7,9,200\n")
    rows, size = parse_response(data, date(2015, 1, 1), date(2015, 12, 31))
    assert [row["date"] for row in rows] == ["2015-01-02", "2015-09-30"]
    assert size == len(data.getvalue())


def test_guest_parser_rejects_out_of_scope_date_without_exposing_its_values():
    data = BytesIO(b"06/30/2026,10,12,9,11,100\n07/01/2026,12345,12346,12344,12345,100\n")
    with pytest.raises(OutOfScopeDate) as failure:
        parse_response(data, date(1980, 12, 12), date(2026, 6, 30))
    assert "2026-07-01" in str(failure.value)
    assert "12345" not in str(failure.value)


def test_guest_parser_rejects_bad_prices_and_duplicate_dates():
    with pytest.raises(ValueError, match="OHLC"):
        parse_response(BytesIO(b"01/02/2015,10,9,8,11,100\n"), date(2015, 1, 1), date(2015, 1, 31))
    with pytest.raises(ValueError, match="duplicate"):
        parse_response(BytesIO(b"01/02/2015,10,12,9,11,100\n01/02/2015,10,12,9,11,100\n"),
                       date(2015, 1, 1), date(2015, 1, 31))


def test_unit_diagnostic_detects_large_change_on_synthetic_data():
    first = pd.DataFrame({"date": ["1997-12-31", "1998-01-02", "1998-01-05"],
                          "close": [1.0, 28.0, 29.0]})
    second = pd.DataFrame({"date": first.date, "close": [28.0, 28.0, 29.0]})
    result = historical_unit_diagnostics(first, second, [])
    assert result["adjacent_close_changes_above_tenfold"][0]["date"] == "1998-01-02"
    assert result["annual_ratios_vs_candidate_nominal_reconstruction"][0]["median_ratio"] == pytest.approx(1 / 28)


def test_month_end_comparison_excludes_a_partial_final_month():
    left = pd.DataFrame({"date": ["2015-09-30", "2015-10-01"], "close": [100.0, 120.0]})
    right = pd.DataFrame({"date": ["2015-09-30", "2015-10-01"], "close": [100.0, 100.0]})
    result = month_end_comparison(left, right)
    assert result["overlapping_proxy_month_ends"] == 1
    assert result["above_0_1_percent_count"] == 0
