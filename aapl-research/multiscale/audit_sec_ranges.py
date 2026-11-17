"""Compare a candidate's adjusted extrema with Apple's 2000 SEC filing.

This is a source-quality diagnostic only. It neither establishes data rights nor
validates daily close values, all adjustment events, or a predictive model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .audit_kaggle_candidates import candidate, kpatel_splits
from .bars import file_digest
from .spec import OUT


SEC_URL = "https://www.sec.gov/Archives/edgar/data/320193/000091205700053623/a2032880z10-k.txt"
QUARTERS = (
    (1999, 1, "1998-09-27", "1998-12-26"), (1999, 2, "1998-12-27", "1999-03-27"),
    (1999, 3, "1999-03-28", "1999-06-26"), (1999, 4, "1999-06-27", "1999-09-25"),
    (2000, 1, "1999-09-26", "2000-01-01"), (2000, 2, "2000-01-02", "2000-04-01"),
    (2000, 3, "2000-04-02", "2000-07-01"), (2000, 4, "2000-07-02", "2000-09-30"),
)


def filing_ranges(text: str):
    if "On June 21, 2000, the Company effected a two-for-one stock split" not in text:
        raise ValueError("Expected historical split statement is absent")
    result = {}
    for year in (1999, 2000):
        match = re.search(rf"Fiscal {year} price range per common\s+share\.+\s+([^\n]+)", text)
        if not match:
            raise ValueError(f"Fiscal {year} table is missing")
        pairs = re.findall(r"\$([0-9]+\.[0-9]+)-\$([0-9]+\.[0-9]+)", match.group(1))
        if len(pairs) != 4:
            raise ValueError(f"Fiscal {year} must contain four reported ranges")
        # Filing orders the quarters fourth through first.
        result[year] = [(float(hi), float(lo)) for hi, lo in reversed(pairs)]
    return result


def audit_ranges(prices: pd.DataFrame, events, filing: str):
    reported = filing_ranges(filing)
    later_splits = [x for x in events if x["date"] > "2000-06-21"]
    if [(x["date"], x["ratio"]) for x in later_splits] != [
            ("2005-02-28", 2.0), ("2014-06-09", 7.0), ("2020-08-31", 4.0)]:
        raise ValueError("Unexpected candidate split chain after the filing's 2000 adjustment")
    factor = math.prod(x["ratio"] for x in later_splits)
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["Date"], format="%d-%m-%Y", errors="raise").dt.strftime("%Y-%m-%d")
    for name in ("High", "Low"):
        prices[name] = pd.to_numeric(prices[name], errors="raise")
    if (not np.isfinite(prices[["High", "Low"]].to_numpy()).all()
            or (prices[["High", "Low"]] <= 0).any().any() or (prices.High < prices.Low).any()):
        raise ValueError("Invalid candidate extrema")
    rows = []
    for year, quarter, start, end in QUARTERS:
        group = prices.loc[prices.date.between(start, end)]
        if group.empty:
            raise ValueError(f"Candidate is missing fiscal {year} quarter {quarter}")
        expected_high, expected_low = reported[year][quarter - 1]
        high_difference = float(group.High.max() * factor - expected_high)
        low_difference = float(group.Low.min() * factor - expected_low)
        rows.append({"fiscal_year": year, "quarter": quarter, "sessions": len(group),
                     "high_difference_usd": high_difference, "low_difference_usd": low_difference,
                     "both_within_one_cent": max(abs(high_difference), abs(low_difference)) <= .01 + 1e-9})
    return {"candidate_to_2000_filing_share_units_factor": factor, "quarters": rows,
            "all_within_one_cent": all(row["both_within_one_cent"] for row in rows),
            "maximum_absolute_difference_usd": max(max(abs(row["high_difference_usd"]),
                                                       abs(row["low_difference_usd"])) for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kpatel", type=Path, required=True)
    args = parser.parse_args()
    directory, receipt, _ = candidate(args.kpatel, "kpatel")
    source_dir = OUT / "source_docs"
    source_path = source_dir / "apple_sec_2000_10k.txt"
    source_receipt = json.loads((source_dir / "apple_sec_2000_10k.receipt.json").read_text(encoding="utf8"))
    if source_receipt.get("url") != SEC_URL or file_digest(source_path) != source_receipt.get("sha256"):
        raise ValueError("SEC source identity differs from its receipt")
    price_entry = next(x for x in receipt["files"] if x["original_name"] == "AppleStockPrice.csv")
    prices = pd.read_csv(directory / price_entry["file"])
    result = {"status": "diagnostic_not_admitted", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "candidate_receipt_sha256": file_digest(directory / "receipt.json"),
              "sec_source": {"url": SEC_URL, "sha256": source_receipt["sha256"]},
              "comparison": audit_ranges(prices, kpatel_splits(directory, receipt),
                                         source_path.read_text(encoding="utf8")),
              "Q3_values_read": False,
              "limitations": ["Compares 1999-2000 fiscal-quarter highs and lows, not each daily close.",
                              "The adjustment chain comes from the candidate's split table.",
                              "This does not verify upstream data rights or earlier historical prices.",
                              "No prediction model was trained or evaluated."]}
    out = OUT / "kaggle_review"
    out.mkdir(parents=True, exist_ok=True)
    path = out / ("sec_ranges_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    with path.open("x", encoding="utf8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"path": str(path), "status": result["status"], **result["comparison"]}))


if __name__ == "__main__":
    main()
