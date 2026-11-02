"""Verify saved Alpaca SIP pages and audit selected sessions against XNYS."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from finance_v2.data import audit_bars, read_bars
from finance_v2.io import digest, write_json
from finance_v2.provenance import verify_alpaca_run


def audit_run(root: Path) -> dict:
    root = root.resolve()
    verification = verify_alpaca_run(root)
    receipt = json.loads((root / "receipt.json").read_text())
    if (receipt.get("symbol"), receipt.get("feed"), receipt.get("adjustment"),
            receipt.get("timeframe")) != ("AAPL", "sip", "raw", "1Min"):
        raise ValueError("Receipt is not the required AAPL SIP/raw/1Min request")
    if not receipt.get("download_complete"):
        raise ValueError("Download receipt is incomplete")
    csv_path = root / receipt["csv"]
    if digest(csv_path) != receipt["csv_sha256"]:
        raise ValueError("CSV SHA256 does not match receipt")
    requested = [a for a, b in receipt["ranges"] if a == b]
    if len(requested) != len(receipt["ranges"]):
        raise ValueError("This audit expects one explicitly requested date per range")
    pages = receipt.get("pages", [])
    if len(pages) != len(requested):
        raise ValueError("Page receipt count differs from requested session count")
    page_bars = {}
    for page in pages:
        raw = root / page["file"]
        if digest(raw) != page["sha256"] or raw.stat().st_size != page["bytes"]:
            raise ValueError(f"Raw page integrity failure: {page['file']}")
        payload = json.loads(raw.read_text())
        rows = payload.get("bars", {}).get("AAPL", [])
        if not isinstance(rows, list) or len(rows) != page["rows_returned"]:
            raise ValueError(f"Raw page row count mismatch: {page['file']}")
        if page["has_next_page"]:
            raise ValueError("Audit range unexpectedly has an unrecorded next page")
        page_bars[page["range_start"]] = rows

    all_bars = read_bars(csv_path)
    selected, calendar = audit_bars(all_bars, requested_dates=requested)
    local = selected.index.tz_convert("America/New_York")
    counts = selected.groupby(local.strftime("%Y-%m-%d")).size().to_dict()
    daily = calendar["daily"]
    for day in daily:
        day["alpaca_bars"] = int(counts.get(day["date"], 0))
        start = pd.Timestamp(day["open_utc"])
        stop = pd.Timestamp(day["close_exclusive_utc"])
        raw_regular = sum(start <= pd.Timestamp(bar["t"]) < stop
                          for bar in page_bars[day["date"]])
        day["raw_api_regular_rows"] = raw_regular
        day["missing_regular_minutes"] = day["expected"] - day["alpaca_bars"]
        day["regular_coverage"] = day["alpaca_bars"] / day["expected"]
        day["coverage_passed"] = day["regular_coverage"] >= calendar["threshold"]
        if day["alpaca_bars"] != raw_regular:
            raise ValueError(f"CSV and SIP page regular-session count differ: {day['date']}")

    local_dates = np.asarray(local.strftime("%Y-%m-%d"))
    close = selected.close.to_numpy(dtype=np.float64)
    prev = np.roll(close, 1)
    same_session = np.r_[False, local_dates[1:] == local_dates[:-1]]
    adjacent = np.r_[False, np.diff(selected.index.as_unit("ns").asi8) == 60 * 10**9]
    valid = same_session & adjacent
    all_returns = np.r_[np.nan, np.diff(np.log(close))]
    returns = np.where(valid, all_returns, np.nan)
    jumps = np.flatnonzero(np.isfinite(returns) & (np.abs(returns) >= np.log(1.05)))
    largest = np.flatnonzero(np.isfinite(returns))
    largest = largest[np.argsort(np.abs(returns[largest]))[-10:][::-1]]
    jump_rows = [{"timestamp": selected.index[i].isoformat(), "log_return": float(returns[i]),
                  "previous_close": float(prev[i]), "close": float(close[i])} for i in largest]
    offsets = {d["date"]: int(selected.loc[
        (selected.index.tz_convert("America/New_York").strftime("%Y-%m-%d") == d["date"])].index[0]
        .tz_convert("America/New_York").utcoffset().total_seconds() // 60) for d in daily}
    expected_dst = offsets.get("2024-03-08") != offsets.get("2024-03-11") and \
        offsets.get("2024-10-31") != offsets.get("2024-11-22")
    early = next((x for x in daily if x["date"] == "2024-11-29"), None)
    if early is None or early["expected"] != 210:
        raise ValueError("XNYS calendar did not identify the 2024-11-29 13:00 ET early close")
    passed = (calendar["coverage_passed"] and expected_dst and
              early["alpaca_bars"] == 210 and early["coverage_passed"] and
              all(x["coverage_passed"] for x in daily))
    result = {"status": "provider_access_and_structure_audit_complete" if passed else "audit_failed",
              "formal_eligible": False, "source_verified": False,
              "receipt_sha256": digest(root / "receipt.json"),
              "csv_sha256": digest(csv_path), "raw_page_count": len(pages),
              "raw_pages_integrity_verified": True, "request": {k: receipt[k] for k in
                  ("symbol", "feed", "adjustment", "timeframe", "purpose")},
              "raw_to_csv_verification": verification,
              "calendar": "XNYS", "timezone": "America/New_York", "timestamp_label":
                  "bar_start as documented by Alpaca FAQ; historical point-in-time publication remains unproven",
              "calendar_coverage": calendar, "dst_switch_verified": expected_dst,
              "early_close_2024_11_29": early, "large_return_threshold": float(np.log(1.05)),
              "large_return_count": len(jumps), "largest_adjacent_regular_returns": jump_rows,
              "limitations": ["A high coverage rate is not proof of price accuracy or trade eligibility.",
                              "SIP/raw settings do not establish point-in-time publication or correction policy.",
                              "The early-close day yields no 240-minute-history plus 12-minute-target windows.",
                              "Historical sample is for data-source evaluation and has been inspected; it is not a sealed test."]}
    write_json(root / "quality_audit.json", result)
    if not passed:
        raise SystemExit("Audit gates failed; see quality_audit.json")
    print(json.dumps({k: v for k, v in result.items() if k != "calendar_coverage"}, indent=2))
    print(json.dumps({"daily": daily, "calendar_coverage": calendar["regular_coverage"],
                      "minimum_daily_coverage": calendar["minimum_daily_coverage"]}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    audit_run(args.run)


if __name__ == "__main__":
    main()
