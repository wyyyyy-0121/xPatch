"""Shared source-audit, calendar, and chronological-fold helpers.

Long-history commercial vendor admission is paused pending academic rights.
The existing-account Alpaca fallback uses the helpers without the vendor feed.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import exchange_calendars as xc
import numpy as np
import pandas as pd

from .bars import Bars, file_digest
from .spec import DEV_END, DEV_START, NY, OUT
from .splits import rolling_folds


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def checked_source(directory, csv_filename=None):
    directory = Path(directory).resolve()
    if not directory.is_relative_to((OUT / "daily_sources").resolve()):
        raise ValueError("Source path outside isolated multiscale data directory")
    receipt = read_json(directory / "receipt.json")
    if receipt.get("status") != "acquired_not_admitted" or receipt.get("upper_date_inclusive_new_york") != DEV_END:
        raise ValueError("Source receipt is incomplete or has an unexpected cutoff")
    for page in receipt["pages"]:
        path = (directory / page["file"]).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError("Invalid source page path")
        if file_digest(path) != page["sha256"] or path.stat().st_size != page["bytes"]:
            raise ValueError("Raw provider page identity differs from receipt")
    table = None
    if csv_filename:
        csv_path = directory / csv_filename
        if file_digest(csv_path) != receipt["csv_sha256"]:
            raise ValueError("Normalized CSV identity differs from receipt")
        table = pd.read_csv(csv_path, dtype={"date": str}).sort_values("date", ignore_index=True)
        dates = table["date"]
        if dates.duplicated().any() or not dates.between(DEV_START, DEV_END).all():
            raise ValueError("Source CSV has duplicate or out-of-scope dates")
        if not np.isfinite(table["close"].to_numpy(dtype=float)).all() or (table["close"] <= 0).any():
            raise ValueError("Source close is not finite and positive")
        if len(table) != receipt["rows"] or dates.iloc[0] != receipt["first"] or dates.iloc[-1] != receipt["last"]:
            raise ValueError("Source CSV range/count differs from receipt")
    return directory, receipt, table


def verify_page_rows(directory: Path, receipt: dict, table: pd.DataFrame, provider: str):
    """Independently rebuild every normalized close from saved provider pages."""
    rows = []
    for page in receipt["pages"]:
        if page["file"].endswith("splits.json"):
            continue
        payload = read_json(directory / page["file"])
        if provider == "eodhd":
            source = payload
            rows.extend((x["date"], float(x["close"])) for x in source)
        elif provider == "twelve":
            source = payload["values"]
            rows.extend((x["datetime"], float(x["close"])) for x in source)
        elif provider == "alpaca":
            source = payload["bars"]["AAPL"]
            rows.extend((datetime.fromisoformat(x["t"].replace("Z", "+00:00"))
                         .astimezone(ZoneInfo(NY)).date().isoformat(), float(x["c"])) for x in source)
        elif provider == "nasdaq":
            source = payload["data"]["tradesTable"]["rows"]
            rows.extend((datetime.strptime(x["date"], "%m/%d/%Y").date().isoformat(),
                         float(x["close"].replace("$", "").replace(",", ""))) for x in source)
        else:
            raise ValueError("Unknown provider")
        if len(source) != page["rows"]:
            raise ValueError("Provider page row count differs from receipt")
    reconstructed = pd.DataFrame(rows, columns=["date", "close"]).sort_values("date", ignore_index=True)
    if not reconstructed["date"].equals(table["date"]):
        raise ValueError("Provider page dates differ from normalized CSV")
    if not np.allclose(reconstructed["close"], table["close"], rtol=1e-12, atol=1e-12):
        raise ValueError("Provider page close prices differ from normalized CSV")


def split_events(eod_dir: Path, twelve_dir: Path):
    first = []
    for item in read_json(eod_dir / "eodhd_splits.json"):
        n, sep, d = item["split"].partition("/")
        if not sep:
            raise ValueError("Invalid EODHD split")
        first.append((item["date"], Decimal(n) / Decimal(d)))
    second = []
    payload = read_json(twelve_dir / "twelve_splits.json")
    if payload.get("meta", {}).get("symbol") != "AAPL":
        raise ValueError("Twelve Data split symbol mismatch")
    for item in payload["splits"]:
        second.append((item["date"], Decimal(str(item["from_factor"])) / Decimal(str(item["to_factor"]))))
    first.sort(); second.sort()
    if first != second or len({x[0] for x in first}) != len(first):
        raise ValueError("The two provider split-event histories disagree")
    if any(not DEV_START <= day <= DEV_END for day, _ in first):
        raise ValueError("Out-of-scope split event")
    return [{"date": day, "ratio": float(ratio)} for day, ratio in first]


def comparison(left: pd.DataFrame, right: pd.DataFrame, actions, *, right_is_split_adjusted: bool):
    merged = left[["date", "close"]].merge(right[["date", "close"]], on="date", suffixes=("_eod", "_other"),
                                            validate="one_to_one")
    if merged.empty:
        raise ValueError("No overlapping provider dates")
    factor = np.ones(len(merged), dtype=float)
    if right_is_split_adjusted:
        for event in actions:
            factor[merged["date"].to_numpy() < event["date"]] *= event["ratio"]
    diff = np.abs(merged["close_eod"].to_numpy(float) - merged["close_other"].to_numpy(float) * factor)
    ratio = diff / merged["close_eod"].to_numpy(float)
    return {"overlap": len(merged), "first": merged["date"].iloc[0], "last": merged["date"].iloc[-1],
            "right_split_adjustment_reversed_only_for_audit": right_is_split_adjusted,
            "median_absolute_difference_usd": float(np.median(diff)),
            "p95_absolute_difference_usd": float(np.quantile(diff, 0.95)),
            "max_absolute_difference_usd": float(diff.max()),
            "max_relative_difference": float(ratio.max()),
            "within_one_cent_fraction": float(np.mean(diff <= 0.01 + 1e-9)),
            "over_one_cent_dates_count": int(np.sum(diff > 0.01 + 1e-9)),
            "largest_difference_dates": merged.loc[np.argsort(diff)[-5:], "date"].tolist()}


def daily_bars(eod: pd.DataFrame):
    """Place calendar-day final bars at the next local midnight conservatively.

    Alpaca documents aggregation by New York calendar day. Historical final
    values can still be corrected later, so midnight is a lower-bound assumption,
    not proof of point-in-time publication or receipt.
    """
    calendar = xc.get_calendar("XNYS", start="1980-12-01", end="2026-07-07")
    first = max(DEV_START, str(eod["date"].min()))
    schedule = calendar.schedule.loc[first:DEV_END]
    dates = schedule.index.strftime("%Y-%m-%d").to_numpy(dtype="U10")
    source_dates = set(eod["date"])
    expected_dates = set(dates)
    missing = sorted(expected_dates - source_dates)
    extra = sorted(source_dates - expected_dates)
    aligned = eod.set_index("date").reindex(dates)
    source_observed = aligned["close"].notna().to_numpy(dtype=bool)
    local_midnights = pd.DatetimeIndex(schedule.index).tz_localize(NY)
    local_next_midnights = (pd.DatetimeIndex(schedule.index) + pd.Timedelta(days=1)).tz_localize(NY)
    starts = local_midnights.tz_convert("UTC").as_unit("ns").asi8
    ends = local_next_midnights.tz_convert("UTC").as_unit("ns").asi8
    cutoff = pd.Timestamp("2026-07-01", tz=NY).tz_convert("UTC").as_unit("ns").value
    within_cutoff = ends < cutoff
    observed = source_observed & within_cutoff
    bars = Bars("day", starts, ends, aligned["close"].to_numpy(dtype=float), observed, dates)
    audit = {"calendar": "XNYS proxy for Nasdaq, including historical-calendar limitations",
             "expected_sessions": len(dates), "source_observed_sessions": int(source_observed.sum()),
             "usable_before_cutoff": int(observed.sum()),
             "missing_sessions": missing, "source_dates_not_in_calendar": extra,
             "cutoff_excluded_sessions": list(dates[source_observed & ~within_cutoff]),
             "decision_time_policy": "next 00:00 America/New_York after the source date; final revisions can arrive later",
             "missing_policy": "mark unobserved and exclude any affected context or complete label; no filling",
             "extra_policy": "exclude from XNYS-grid analysis and retain raw source; historical exchange difference requires review"}
    return bars, audit


def stamp(ns):
    return pd.Timestamp(int(ns), tz="UTC").tz_convert(NY).isoformat()


def fold_summary(bars, origins, horizon):
    splits = rolling_folds(bars, origins, horizon, folds=5)
    rows = []
    for entry in splits:
        train, validation, inner = entry["train"], entry["validation"], entry["inner"]
        row = {"fold": entry["fold"], "status": entry["status"], "train_origins": len(train),
               "validation_origins": len(validation),
               "validation_first_decision": stamp(bars.end_ns[validation[0]]) if len(validation) else None,
               "validation_last_decision": stamp(bars.end_ns[validation[-1]]) if len(validation) else None,
               "validation_last_target_completion": stamp(bars.end_ns[validation[-1] + horizon]) if len(validation) else None,
               "train_last_target_completion": stamp(bars.end_ns[train[-1] + horizon]) if len(train) else None,
               "inner_fit_origins": len(inner["fit"]) if inner else 0,
               "inner_stop_origins": len(inner["stop"]) if inner else 0}
        if len(train) and bars.end_ns[train[-1] + horizon] >= bars.end_ns[validation[0]]:
            raise ValueError("Outer label purge violation")
        if inner and len(inner["fit"]) and bars.end_ns[inner["fit"][-1] + horizon] >= bars.end_ns[inner["stop"][0]]:
            raise ValueError("Inner label purge violation")
        rows.append(row)
    return rows


def save_bars(path, bars):
    np.savez_compressed(path, start_ns=bars.start_ns, end_ns=bars.end_ns, close=bars.close,
                        observed=bars.observed, session_date=bars.session_date)



if __name__ == "__main__":
    raise SystemExit("Long-history vendor admission is paused until academic usage rights are verified. Use multiscale.admission_alpaca for the existing-account development source.")
