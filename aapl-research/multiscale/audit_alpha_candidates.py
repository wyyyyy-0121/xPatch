"""Audit two Alpha-Vantage-attributed AAPL snapshots, without admitting them.

All source material must already have passed the pre-Q3 acquisition checks.
Provider attribution is the uploaders' claim, and licence rights remain open.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import pandas as pd

from .acquire_kaggle_candidate import DATASETS, LICENSE_LABELS, csv_metadata, parse_day
from .admission import checked_source, verify_page_rows
from .audit_kaggle_candidates import calendar_audit, candidate, compare, kpatel_splits
from .audit_sec_ranges import QUARTERS, SEC_URL, filing_ranges
from .bars import file_digest
from .spec import DEV_END, DEV_START, OUT


def read_candidate(path: Path, kind: str):
    directory = path.resolve()
    if kind not in ("pratham", "ayesha") or not directory.is_relative_to((OUT / "kaggle_candidates").resolve()):
        raise ValueError("Unexpected Alpha-attributed candidate location or kind")
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf8"))
    if (receipt.get("status") != "candidate_not_admitted" or receipt.get("Q3_values_read") is not False
            or receipt.get("ref") != DATASETS[kind] or receipt.get("license_label") != LICENSE_LABELS[kind]):
        raise ValueError("Unexpected Alpha-attributed candidate receipt")
    expected = ({"AAPL_stock_dividend.csv", "AAPL_stock_price.csv", "AAPL_stock_split.csv"}
                if kind == "pratham" else {"aapl_stock_data.csv"})
    entries = receipt["files"]
    if len(entries) != len(expected) or {item["original_name"] for item in entries} != expected:
        raise ValueError("Unexpected Alpha-attributed candidate files")
    for item in entries:
        path = (directory / item["file"]).resolve()
        if not path.is_relative_to(directory) or file_digest(path) != item["sha256"]:
            raise ValueError("Candidate CSV identity changed")
        info = csv_metadata(path.read_bytes())
        if any(info[key] != item[key] for key in ("columns", "rows", "first", "last")):
            raise ValueError("Candidate CSV metadata changed")
    for name, key in (("download.zip", "archive_sha256"), ("dataset_metadata.json", "metadata_sha256")):
        if file_digest(directory / name) != receipt[key]:
            raise ValueError("Candidate archive or metadata identity changed")
    entry = next(item for item in entries if item["original_name"] in ("AAPL_stock_price.csv", "aapl_stock_data.csv"))
    columns = (["Date", "High Price", "Low Price", "Close Price"] if kind == "pratham"
               else ["Date", "Close"])
    raw = pd.read_csv(directory / entry["file"], usecols=columns)
    raw["date"] = pd.to_datetime(raw.pop("Date"), format="%Y-%m-%d", errors="raise").dt.strftime("%Y-%m-%d")
    names = {"High Price": "high", "Low Price": "low", "Close Price": "close", "Close": "close"}
    raw = raw.rename(columns=names).sort_values("date", ignore_index=True)
    for column in raw.columns.difference(["date"]):
        raw[column] = pd.to_numeric(raw[column], errors="raise")
        if not np.isfinite(raw[column].to_numpy()).all() or (raw[column] <= 0).any():
            raise ValueError("Invalid candidate price")
    if (len(raw) != entry["rows"] or raw.date.duplicated().any()
            or not raw.date.between(DEV_START, DEV_END).all()):
        raise ValueError("Candidate dates or row count differ from the receipt")
    if kind == "pratham" and (raw.high < raw.low).any():
        raise ValueError("A candidate high is below its low")
    return directory, receipt, raw


def pratham_splits(directory: Path, receipt: dict):
    entry = next(item for item in receipt["files"] if item["original_name"] == "AAPL_stock_split.csv")
    raw = pd.read_csv(directory / entry["file"], dtype=str)
    events = []
    for _, row in raw.iterrows():
        parts = row["Stock Splits"].split(":")
        if len(parts) != 2 or any(not part.isdecimal() or int(part) <= 0 for part in parts):
            raise ValueError("Invalid Alpha-attributed split ratio")
        events.append({"date": parse_day(row.Date).isoformat(), "ratio": int(parts[0]) / int(parts[1])})
    return sorted(events, key=lambda event: event["date"])


def compare_raw_to_adjusted(raw: pd.DataFrame, adjusted: pd.DataFrame, events):
    merged = raw[["date", "close"]].merge(adjusted[["date", "close"]], on="date",
                                          suffixes=("_raw", "_adjusted"), validate="one_to_one")
    if merged.empty:
        raise ValueError("No overlap between candidate price series")
    factors = np.ones(len(merged), dtype=float)
    for event in events:
        factors[merged.date.to_numpy() < event["date"]] *= event["ratio"]
    difference = np.abs(merged.close_adjusted.to_numpy(float) * factors - merged.close_raw.to_numpy(float))
    relative = difference / merged.close_raw.to_numpy(float)
    first_month = pd.Timestamp(merged.date.iloc[0]).to_period("M")
    last_month = pd.Timestamp(merged.date.iloc[-1]).to_period("M")
    calendar = xc.get_calendar("XNYS", start=first_month.start_time - pd.Timedelta(days=7),
                               end=last_month.end_time.normalize() + pd.Timedelta(days=7)).schedule
    dates = calendar.index.strftime("%Y-%m-%d")
    month_ends = pd.Series(dates, index=calendar.index.to_period("M")).groupby(level=0).last()
    on_month_end = np.isin(merged.date.to_numpy(), month_ends.to_numpy())
    month_end_relative = relative[on_month_end]
    month_end_dates = merged.date.to_numpy()[on_month_end]
    return {"overlap": len(merged), "first": str(merged.date.iloc[0]), "last": str(merged.date.iloc[-1]),
            "median_relative_difference": float(np.median(relative)),
            "p95_relative_difference": float(np.quantile(relative, .95)),
            "max_relative_difference": float(relative.max()),
            "above_0_1_percent_count": int(np.sum(relative > .001)),
            "above_1_percent_count": int(np.sum(relative > .01)),
            "over_0_1_percent_dates": [{"date": str(merged.date.iloc[i]), "relative_difference": float(relative[i])}
                                        for i in np.flatnonzero(relative > .001)],
            "completed_month_end_overlap": int(on_month_end.sum()),
            "month_end_max_relative_difference": float(month_end_relative.max()) if len(month_end_relative) else None,
            "month_end_over_0_1_percent_dates": [{"date": str(day), "relative_difference": float(value)}
                                                for day, value in zip(month_end_dates, month_end_relative, strict=True)
                                                if value > .001],
            "candidate_reported_split_chain_used": True}


def compare_filing_quarters(prices: pd.DataFrame, filing: str):
    reported = filing_ranges(filing)
    calendar = xc.get_calendar("XNYS", start="1998-09-01", end="2000-10-05").schedule
    rows = []
    for year, quarter, start, end in QUARTERS:
        group = prices.loc[prices.date.between(start, end)]
        expected = set(calendar.loc[start:end].index.strftime("%Y-%m-%d"))
        actual = set(group.date)
        if expected != actual:
            rows.append({"fiscal_year": year, "quarter": quarter, "status": "incomplete_candidate_quarter",
                         "expected_sessions": len(expected), "observed_sessions": len(actual)})
            continue
        factors = np.where(group.date.to_numpy() < "2000-06-21", .5, 1.0)
        expected_high, expected_low = reported[year][quarter - 1]
        high_difference = float(np.max(group.high.to_numpy() * factors) - expected_high)
        low_difference = float(np.min(group.low.to_numpy() * factors) - expected_low)
        rows.append({"fiscal_year": year, "quarter": quarter, "status": "compared",
                     "sessions": len(group), "high_difference_usd": high_difference,
                     "low_difference_usd": low_difference,
                     "both_within_one_cent": max(abs(high_difference), abs(low_difference)) <= .01 + 1e-9})
    completed = [row for row in rows if row["status"] == "compared"]
    return {"quarter_policy": "compare only when candidate dates match all XNYS proxy sessions",
            "filing_price_units": "pre-2000-06-21 candidate prices divided by two for filing's 2000 split",
            "quarters": rows, "complete_quarters": len(completed),
            "all_compared_within_one_cent": bool(completed) and all(row["both_within_one_cent"] for row in completed)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("pratham", "ayesha", "kpatel", "alpaca"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--twelve", type=Path, help="Optional existing development-only diagnostic source")
    args = parser.parse_args()
    first_dir, first_receipt, first = read_candidate(args.pratham, "pratham")
    second_dir, second_receipt, second = read_candidate(args.ayesha, "ayesha")
    third_dir, third_receipt, third = candidate(args.kpatel, "kpatel")
    alpaca_dir, alpaca_receipt, alpaca = checked_source(args.alpaca, "alpaca_daily_raw.csv")
    verify_page_rows(alpaca_dir, alpaca_receipt, alpaca, "alpaca")
    actions = kpatel_splits(third_dir, third_receipt)
    claimed_actions = pratham_splits(first_dir, first_receipt)
    if actions != claimed_actions:
        raise ValueError("Candidate split tables disagree")
    source_dir = OUT / "source_docs"
    sec_path = source_dir / "apple_sec_2000_10k.txt"
    sec_receipt = json.loads((source_dir / "apple_sec_2000_10k.receipt.json").read_text(encoding="utf8"))
    if sec_receipt.get("url") != SEC_URL or file_digest(sec_path) != sec_receipt.get("sha256"):
        raise ValueError("SEC source identity differs from its receipt")
    result = {"status": "candidate_not_admitted", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "source diagnostics only; no model training or forecast scores", "Q3_values_read": False,
              "source_receipt_sha256": {"pratham": file_digest(first_dir / "receipt.json"),
                                         "ayesha": file_digest(second_dir / "receipt.json"),
                                         "kpatel": file_digest(third_dir / "receipt.json"),
                                         "alpaca": file_digest(alpaca_dir / "receipt.json")},
              "uploader_provider_claim": "Alpha Vantage", "upstream_rights_independently_verified": False,
              "calendars": {"pratham": calendar_audit(first), "ayesha": calendar_audit(second)},
              "comparisons": {"pratham_vs_ayesha": compare(first, second),
                              "pratham_vs_raw_alpaca": compare(first, alpaca),
                              "ayesha_vs_raw_alpaca": compare(second, alpaca),
                              "pratham_raw_vs_kpatel_split_adjusted": compare_raw_to_adjusted(first, third, actions)},
              "reported_split_tables_match": True, "reported_splits": actions,
              "sec_2000_filing_sha256": sec_receipt["sha256"],
              "sec_fiscal_comparison": compare_filing_quarters(first, sec_path.read_text(encoding="utf8")),
              "limitations": ["Both Alpha-attributed candidates may be copies of the same upstream series.",
                              "Uploader licence labels do not establish the original provider's rights.",
                              "Historical final files do not establish point-in-time publication or revision history.",
                              "The SEC comparison covers fiscal-quarter highs/lows, not every daily close.",
                              "No source splice or formal data admission is performed."]}
    if args.twelve:
        twelve_dir, twelve_receipt, twelve = checked_source(args.twelve, "twelve_daily.csv")
        verify_page_rows(twelve_dir, twelve_receipt, twelve, "twelve")
        result["source_receipt_sha256"]["twelve_diagnostic"] = file_digest(twelve_dir / "receipt.json")
        result["comparisons"]["pratham_raw_vs_twelve_assumed_split_adjusted"] = compare_raw_to_adjusted(first, twelve, actions)
        result["comparisons"]["twelve_vs_kpatel"] = compare(twelve, third)
        result["limitations"].append("Twelve Data is an existing source diagnostic, not admitted for training or publication.")
    out = OUT / "kaggle_review"
    out.mkdir(parents=True, exist_ok=True)
    path = out / ("alpha_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    with path.open("x", encoding="utf8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"path": str(path), "status": result["status"], "calendars": result["calendars"],
                      "comparisons": result["comparisons"], "sec_fiscal_comparison": result["sec_fiscal_comparison"]}))


if __name__ == "__main__":
    main()
