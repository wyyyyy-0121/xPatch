"""Audit an entire completed AAPL SIP history, including absent sessions."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from finance_v2.data import load_market_data
from finance_v2.io import read_json, write_json
from finance_v2.provenance import verify_alpaca_run


def audit_history(root):
    root = Path(root)
    verified = verify_alpaca_run(root)
    receipt = read_json(root / "receipt.json")
    start, end = receipt["ranges"][0][0], receipt["ranges"][-1][1]
    data = load_market_data(root / receipt["csv"], start=start, end=end)
    audit = data.audit
    years = {}
    samples_by_date = Counter(data.dates[data.origins])
    for day in audit["daily"]:
        entry = years.setdefault(day["date"][:4], {"sessions": 0, "expected_rows": 0,
            "observed_rows": 0, "eligible_origins": 0, "half_days": 0,
            "sessions_below_98pct": 0})
        entry["sessions"] += 1
        entry["expected_rows"] += day["expected"]
        entry["observed_rows"] += day["observed"]
        entry["eligible_origins"] += samples_by_date[day["date"]]
        entry["half_days"] += int(day["early_close"])
        entry["sessions_below_98pct"] += int(day["coverage"] < .98)
    for entry in years.values():
        entry["coverage"] = entry["observed_rows"] / entry["expected_rows"]
    finite = np.flatnonzero(np.isfinite(data.returns))
    largest = finite[np.argsort(np.abs(data.returns[finite]))[-20:][::-1]]
    details = [{"timestamp_utc": data.bars.index[i].isoformat(),
                "log_return": float(data.returns[i]),
                "previous_close": float(data.bars.close.iloc[i - 1]),
                "close": float(data.bars.close.iloc[i])} for i in largest]
    audit.update({"receipt_verification": verified, "by_year": years,
        "largest_returns": details, "abs_return_over_5pct_count": int(np.sum(
            np.abs(data.returns[finite]) > np.log(1.05))),
        "historical_research_eligible": bool(audit["coverage_passed"] and len(data.origins)),
        "formal_eligible": False, "independent_test": False,
        "status": "retrospective_history_audit_complete",
        "limitations": audit["limitations"] + [
            "This history overlaps previously inspected IEX experiments; it cannot be promoted to a pristine holdout.",
            "Extreme moves are reported without deleting, clipping or interpolating any price.",
            "Historical revisions and bar publication latency are not reconstructible from final bars."]})
    write_json(root / "history_audit.json", audit)
    print({"rows": receipt["rows"], "regular_rows": audit["regular_rows"],
           "expected_rows": audit["expected_rows"], "coverage": audit["regular_coverage"],
           "minimum_session_coverage": audit["minimum_daily_coverage"],
           "eligible_origins": len(data.origins), "by_year": years,
           "historical_research_eligible": audit["historical_research_eligible"]})
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    result = audit_history(args.run)
    if not result["historical_research_eligible"]:
        raise SystemExit("History coverage gate failed; see history_audit.json")


if __name__ == "__main__":
    main()
