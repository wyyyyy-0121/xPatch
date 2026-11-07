"""Audit public-labeled candidates against each other and account data, without admitting them."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import pandas as pd

from .acquire_kaggle_candidate import csv_metadata
from .admission import checked_source, verify_page_rows
from .admission_alpaca import official_split
from .bars import file_digest
from .spec import DEV_END, DEV_START, OUT


def candidate(path: Path, kind: str):
    directory = path.resolve()
    if not directory.is_relative_to((OUT / "kaggle_candidates").resolve()):
        raise ValueError("Candidate path outside isolated data directory")
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf8"))
    if (receipt.get("status") != "candidate_not_admitted" or receipt.get("Q3_values_read") is not False
            or receipt.get("license_label") != "CC0: Public Domain"):
        raise ValueError("Unexpected candidate receipt")
    entries = receipt["files"]
    if kind == "kpatel":
        expected = {"AppleStockPrice.csv", "AppleStockDividend.csv", "AppleStockSplit.csv"}
        if len(entries) != len(expected) or {entry["original_name"] for entry in entries} != expected:
            raise ValueError("Expected the three documented kpatel CSV files")
        entry = next(entry for entry in entries if entry["original_name"] == "AppleStockPrice.csv")
    else:
        if len(entries) != 1:
            raise ValueError("Expected one AAPL CSV in the candidate archive")
        entry = entries[0]
    for item in entries:
        csv_path = (directory / item["file"]).resolve()
        if not csv_path.is_relative_to(directory) or file_digest(csv_path) != item["sha256"]:
            raise ValueError("Candidate CSV identity changed")
        actual = csv_metadata(csv_path.read_bytes())
        if any(actual[key] != item[key] for key in ("columns", "rows", "first", "last")):
            raise ValueError("Candidate CSV metadata changed")
    if file_digest(directory / "download.zip") != receipt["archive_sha256"]:
        raise ValueError("Candidate archive bytes changed")
    if file_digest(directory / "dataset_metadata.json") != receipt["metadata_sha256"]:
        raise ValueError("Candidate metadata bytes changed")
    raw = pd.read_csv(directory / entry["file"])
    source_column = {"rishabh": "Close", "shivam": "Price", "kpatel": "Close"}[kind]
    date_format = "%d-%m-%Y" if kind == "kpatel" else "mixed"
    dates = pd.to_datetime(raw["Date"], format=date_format, errors="raise").dt.strftime("%Y-%m-%d")
    values = pd.to_numeric(raw[source_column].astype(str).str.replace(",", "", regex=False)
                           .str.replace("$", "", regex=False), errors="raise")
    frame = pd.DataFrame({"date": dates, "close": values}).sort_values("date", ignore_index=True)
    if len(frame) != entry["rows"] or frame.date.duplicated().any() or not frame.date.between(DEV_START, DEV_END).all():
        raise ValueError("Candidate dates differ from receipt or cross the boundary")
    if not np.isfinite(frame.close.to_numpy()).all() or (frame.close <= 0).any():
        raise ValueError("Candidate has invalid close prices")
    return directory, receipt, frame


def kpatel_splits(directory: Path, receipt: dict):
    """Read the five uploader-reported actions; this is not provider verification."""
    entry = next(item for item in receipt["files"] if item["original_name"] == "AppleStockSplit.csv")
    frame = pd.read_csv(directory / entry["file"], dtype=str)
    if list(frame.columns) != ["Date", "Stock Splits"]:
        raise ValueError("Unexpected split table columns")
    dates = pd.to_datetime(frame["Date"], format="%Y-%m-%d", errors="raise").dt.strftime("%Y-%m-%d")
    events = []
    for day, ratio in zip(dates, frame["Stock Splits"], strict=True):
        parts = ratio.split(":")
        if len(parts) != 2 or any(not part.isdecimal() or int(part) <= 0 for part in parts):
            raise ValueError("Invalid reported stock split ratio")
        events.append({"date": day, "ratio": int(parts[0]) / int(parts[1])})
    return sorted(events, key=lambda event: event["date"])


def compare(left: pd.DataFrame, right: pd.DataFrame, *, raw_alpaca_to_2020_split_units=False):
    data = left[["date", "close"]].merge(right[["date", "close"]], on="date", suffixes=("_left", "_right"),
                                         validate="one_to_one")
    if data.empty:
        raise ValueError("No overlap between compared sources")
    right_price = data.close_right.to_numpy(dtype=float).copy()
    if raw_alpaca_to_2020_split_units:
        right_price[data.date.to_numpy() < "2020-08-31"] /= 4.0
    relative = np.abs(data.close_left.to_numpy(dtype=float) - right_price) / right_price
    return {"overlap": len(data), "first": str(data.date.iloc[0]), "last": str(data.date.iloc[-1]),
            "median_relative_difference": float(np.median(relative)),
            "p95_relative_difference": float(np.quantile(relative, .95)),
            "max_relative_difference": float(relative.max()),
            "above_0_1_percent_count": int(np.sum(relative > .001)),
            "above_1_percent_count": int(np.sum(relative > .01)),
            "largest_difference_dates": data.iloc[np.argsort(relative)[-8:]].date.tolist(),
            "alpaca_raw_split_conversion_applied": raw_alpaca_to_2020_split_units}


def calendar_audit(frame):
    calendar = xc.get_calendar("XNYS", start="1980-12-01", end="2026-07-07").schedule
    first, last = frame.date.iloc[0], frame.date.iloc[-1]
    expected = set(calendar.loc[first:last].index.strftime("%Y-%m-%d"))
    actual = set(frame.date)
    return {"calendar": "XNYS proxy for Nasdaq", "expected_sessions": len(expected),
            "missing_sessions": sorted(expected - actual), "source_dates_not_in_calendar": sorted(actual - expected)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("rishabh", "shivam", "alpaca", "nasdaq"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--kpatel", type=Path, help="Optional three-file CC0-labeled candidate")
    args = parser.parse_args()
    first_dir, first_receipt, first = candidate(args.rishabh, "rishabh")
    second_dir, second_receipt, second = candidate(args.shivam, "shivam")
    alpaca_dir, alpaca_receipt, alpaca = checked_source(args.alpaca, "alpaca_daily_raw.csv")
    nasdaq_dir, nasdaq_receipt, nasdaq = checked_source(args.nasdaq, "nasdaq_daily.csv")
    verify_page_rows(alpaca_dir, alpaca_receipt, alpaca, "alpaca")
    verify_page_rows(nasdaq_dir, nasdaq_receipt, nasdaq, "nasdaq")
    _, action_source = official_split()
    comparisons = {"candidate1_vs_candidate2": compare(first, second),
                   "candidate1_vs_nasdaq": compare(first, nasdaq),
                   "candidate2_vs_nasdaq": compare(second, nasdaq),
                   "candidate1_vs_alpaca": compare(first, alpaca, raw_alpaca_to_2020_split_units=True),
                   "candidate2_vs_alpaca": compare(second, alpaca, raw_alpaca_to_2020_split_units=True)}
    calendars = {"candidate1": calendar_audit(first), "candidate2": calendar_audit(second)}
    result = {"status": "candidate_not_admitted", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "source quality diagnostics; no model training or predictive scores", "Q3_values_read": False,
              "source_receipt_sha256": {"candidate1": file_digest(first_dir / "receipt.json"),
                                         "candidate2": file_digest(second_dir / "receipt.json"),
                                         "alpaca": file_digest(alpaca_dir / "receipt.json"),
                                         "nasdaq": file_digest(nasdaq_dir / "receipt.json")},
              "license_claims": [{"ref": first_receipt["ref"], "license_label": first_receipt["license_label"]},
                                 {"ref": second_receipt["ref"], "license_label": second_receipt["license_label"]}],
              "upstream_rights_independently_verified": False, "source_adjustment_policy_independently_verified": False,
              "apple_2020_split_source": action_source, "calendars": calendars, "comparisons": comparisons,
              "limitations": ["A CC0 label is an uploader claim; neither candidate establishes its original provider or rights.",
                              "Close columns appear split-adjusted in the 2016-2020 overlap, but the full historical adjustment chain is not established.",
                              "Two dataset uploaders may share the same underlying price source.",
                              "A raw nominal-price reconstruction and splice to 2026 are not authorized by this diagnostic."]}
    if args.kpatel:
        third_dir, third_receipt, third = candidate(args.kpatel, "kpatel")
        result["source_receipt_sha256"]["candidate3"] = file_digest(third_dir / "receipt.json")
        result["license_claims"].append({"ref": third_receipt["ref"],
                                         "license_label": third_receipt["license_label"]})
        result["calendars"]["candidate3"] = calendar_audit(third)
        result["comparisons"].update({"candidate3_vs_candidate1": compare(third, first),
                                      "candidate3_vs_nasdaq": compare(third, nasdaq),
                                      "candidate3_vs_alpaca": compare(third, alpaca,
                                                                        raw_alpaca_to_2020_split_units=True)})
        result["candidate3_reported_splits"] = kpatel_splits(third_dir, third_receipt)
        result["limitations"].append("Candidate 3 describes educational/research use but does not identify its upstream provider.")
        result["limitations"].append("Candidate 3's split table is an uploader claim, not independent verification of every event.")
    directory = OUT / "kaggle_review" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "review.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")
    print(json.dumps({"directory": str(directory), "status": result["status"], "calendars": calendars,
                      "comparisons": comparisons}, ensure_ascii=False))


if __name__ == "__main__":
    main()
