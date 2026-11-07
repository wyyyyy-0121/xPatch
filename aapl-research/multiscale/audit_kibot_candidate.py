"""Cross-check the bounded Kibot guest candidate without admitting or publishing it.

All comparisons and coverage planning remain in ignored local output. No model
is trained, no prediction is scored, and no post-cutoff data is requested.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import exchange_calendars as xc

from .admission import checked_source, daily_bars, verify_page_rows
from .audit_alpha_candidates import compare_filing_quarters, compare_raw_to_adjusted, read_candidate
from .audit_kaggle_candidates import calendar_audit, candidate, compare, kpatel_splits
from .audit_sec_ranges import SEC_URL
from .bars import aggregate_months, eligible_origins, file_digest
from .spec import DEV_END, DEV_START, OUT
from .splits import rolling_folds, warmup_folds


def read_kibot(directory: Path):
    directory = directory.resolve()
    if not directory.is_relative_to((OUT / "daily_sources").resolve()):
        raise ValueError("Kibot candidate path is outside the isolated workspace")
    receipt_path = directory / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf8"))
    if (receipt.get("status") != "candidate_internal_diagnostic_only" or
            receipt.get("publication_rights_verified") is not False or
            receipt.get("training_authorized") is not False or receipt.get("Q3_values_read") is not False):
        raise ValueError("Unexpected Kibot diagnostic state")
    path = directory / "kibot_daily_raw.csv"
    if not path.resolve().is_relative_to(directory) or path.is_symlink():
        raise ValueError("Kibot CSV resolves outside the candidate directory")
    if file_digest(path) != receipt["csv_sha256"]:
        raise ValueError("Kibot CSV identity differs from the acquisition receipt")
    frame = pd.read_csv(path, dtype={"date": str})
    if list(frame.columns) != ["date", "open", "high", "low", "close", "volume"]:
        raise ValueError("Unexpected Kibot daily columns")
    dates = frame.date
    if (len(frame) != receipt["rows"] or dates.duplicated().any() or not dates.is_monotonic_increasing or
            not dates.between(DEV_START, DEV_END).all() or dates.iloc[0] != receipt["first"] or
            dates.iloc[-1] != receipt["last"]):
        raise ValueError("Kibot candidate date range or count differs from the receipt")
    prices = frame[["open", "high", "low", "close"]].to_numpy(dtype=float)
    if (not np.isfinite(prices).all() or (prices <= 0).any() or
            (frame.high < frame[["open", "low", "close"]].max(axis=1)).any() or
            (frame.low > frame[["open", "high", "close"]].min(axis=1)).any()):
        raise ValueError("Kibot candidate has invalid OHLC values")
    return directory, receipt, frame


def date_only_coverage(frame):
    # Keep all real prices out of the planning calculation.
    artificial = pd.DataFrame({"date": frame.date, "close": np.full(len(frame), 100.0)})
    day, _ = daily_bars(artificial)
    month = aggregate_months(day)
    result = {}
    for name, lookback, horizon in (("annual", 60, 12), ("quarterly", 36, 3)):
        origins = eligible_origins(month, lookback, horizon, same_session=False)
        ordinary = rolling_folds(month, origins, horizon)
        warmed = warmup_folds(month, origins, horizon, min_inner_fit=24, min_inner_stop=4,
                              min_validation_groups=6)
        def summarize(folds):
            return [{"fold": fold["fold"], "fit_origins": len(fold["inner"]["fit"]) if fold["inner"] else 0,
                     "stop_origins": len(fold["inner"]["stop"]) if fold["inner"] else 0,
                     "validation_origins": len(fold["validation"])} for fold in folds]
        result[name] = {"valid_origins": len(origins), "ordinary_five_folds": summarize(ordinary),
                        "warmup_five_folds_not_frozen": summarize(warmed)}
    return result


def historical_unit_diagnostics(kibot, adjusted, actions):
    """Compare units by year, and flag very large adjacent raw-close jumps."""
    merged = kibot[["date", "close"]].merge(adjusted[["date", "close"]], on="date",
                                              suffixes=("_kibot", "_adjusted"), validate="one_to_one")
    factors = np.ones(len(merged), dtype=float)
    for action in actions:
        factors[merged.date.to_numpy() < action["date"]] *= action["ratio"]
    merged["unit_ratio"] = merged.close_kibot.to_numpy() / (merged.close_adjusted.to_numpy() * factors)
    by_year = []
    for year, rows in merged.groupby(merged.date.str[:4]):
        ratios = rows.unit_ratio.to_numpy()
        by_year.append({"year": int(year), "overlap": len(rows), "median_ratio": float(np.median(ratios)),
                        "p05_ratio": float(np.quantile(ratios, .05)), "p95_ratio": float(np.quantile(ratios, .95))})
    ratios = kibot.close.to_numpy()[1:] / kibot.close.to_numpy()[:-1]
    unusual = np.flatnonzero((ratios > 10.0) | (ratios < .1))
    return {"annual_ratios_vs_candidate_nominal_reconstruction": by_year,
            "adjacent_close_changes_above_tenfold": [{"date": str(kibot.date.iloc[i + 1]),
                                                       "ratio_to_previous_observation": float(ratios[i])}
                                                      for i in unusual],
            "qualification": "A tenfold jump is a diagnostic flag, not by itself proof of the correct source."}


def month_end_comparison(left, right):
    merged = left[["date", "close"]].merge(right[["date", "close"]], on="date",
                                            suffixes=("_left", "_right"), validate="one_to_one")
    if merged.empty:
        raise ValueError("No overlapping dates for month-end comparison")
    first, last = merged.date.iloc[0], merged.date.iloc[-1]
    first_month, last_month = pd.Timestamp(first).to_period("M"), pd.Timestamp(last).to_period("M")
    calendar = xc.get_calendar("XNYS", start=first_month.start_time - pd.Timedelta(days=7),
                               end=last_month.end_time.normalize() + pd.Timedelta(days=7)).schedule
    month_ends = calendar.index.to_series().groupby(calendar.index.to_period("M")).max().dt.strftime("%Y-%m-%d")
    monthly = merged[merged.date.isin(set(month_ends))]
    relative = np.abs(monthly.close_left.to_numpy() - monthly.close_right.to_numpy()) / monthly.close_right.to_numpy()
    return {"overlapping_proxy_month_ends": len(monthly), "above_0_1_percent_count": int((relative > .001).sum()),
            "above_0_1_percent_dates": monthly.date.to_numpy()[relative > .001].tolist(),
            "qualification": "XNYS is a proxy; disagreement does not identify which vendor is correct."}


def audit(kibot_dir: Path, alpha_dir: Path, kpatel_dir: Path, alpaca_dir: Path):
    kb_dir, kb_receipt, kb = read_kibot(kibot_dir)
    alpha_dir, _, alpha = read_candidate(alpha_dir, "pratham")
    kpatel_dir, k_receipt, kpatel = candidate(kpatel_dir, "kpatel")
    alpaca_dir, alpaca_receipt, alpaca = checked_source(alpaca_dir, "alpaca_daily_raw.csv")
    verify_page_rows(alpaca_dir, alpaca_receipt, alpaca, "alpaca")
    actions = kpatel_splits(kpatel_dir, k_receipt)
    sec = OUT / "source_docs" / "apple_sec_2000_10k.txt"
    sec_receipt = json.loads((sec.parent / "apple_sec_2000_10k.receipt.json").read_text(encoding="utf8"))
    if sec_receipt.get("url") != SEC_URL or file_digest(sec) != sec_receipt.get("sha256"):
        raise ValueError("SEC filing identity differs from the saved receipt")
    day = "2015-09-30"
    ka = kb.loc[kb.date == day, "close"]
    aa = alpha.loc[alpha.date == day, "close"]
    kk = kpatel.loc[kpatel.date == day, "close"]
    if not (len(ka) == len(aa) == len(kk) == 1):
        raise ValueError("A source is missing the disputed month-end")
    factor = float(np.prod([entry["ratio"] for entry in actions if entry["date"] > day]))
    kibot_price, alpha_price, adjusted_price = float(ka.iloc[0]), float(aa.iloc[0]), float(kk.iloc[0]) * factor
    disputed = {"date": day,
                "relative_difference_from_alpha_attributed": abs(kibot_price - alpha_price) / alpha_price,
                "relative_difference_from_kpatel_after_candidate_split_conversion":
                    abs(kibot_price - adjusted_price) / adjusted_price,
                "candidate_split_factor_used": factor}
    month_end = month_end_comparison(kb, alpha)
    after_1998 = kb[kb.date >= "1998-01-01"]
    masked = after_1998[~after_1998.date.isin(month_end["above_0_1_percent_dates"])]
    result = {"status": "internal_source_diagnostic_not_admitted",
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "experiment_scored": False,
              "Q3_values_read": False, "training_authorized": False, "publication_rights_verified": False,
              "source_receipt_sha256": {"kibot": file_digest(kb_dir / "receipt.json"),
                                         "alpha_attributed": file_digest(alpha_dir / "receipt.json"),
                                         "kpatel": file_digest(kpatel_dir / "receipt.json"),
                                         "alpaca": file_digest(alpaca_dir / "receipt.json")},
              "kibot_calendar": calendar_audit(kb),
              "comparisons": {"kibot_vs_alpaca_raw": compare(kb, alpaca),
                              "kibot_vs_alpha_attributed_raw": compare(kb, alpha),
                              "kibot_raw_vs_kpatel_adjusted": compare_raw_to_adjusted(kb, kpatel, actions)},
              "disputed_month_end": disputed,
              "historical_unit_diagnostics": historical_unit_diagnostics(kb, kpatel, actions),
              "month_end_kibot_vs_alpha_attributed": month_end,
              "sec_1999_2000_fiscal_ranges": compare_filing_quarters(kb, sec.read_text(encoding="utf8")),
              "date_only_constant_price_coverage": {"full_candidate": date_only_coverage(kb),
                                                    "candidate_from_1998_not_admitted":
                                                        date_only_coverage(after_1998),
                                                    "exploratory_mask_month_end_disagreements_not_frozen":
                                                        date_only_coverage(masked)},
              "limitations": ["This is internal diagnostics, not permission to train or publish provider data or derived values.",
                              "Providers may share upstream market data; agreement does not prove independence or correctness.",
                              "XNYS is a proxy for Nasdaq's historical session calendar.",
                              "SEC ranges validate only selected quarterly extremes, not each close or all corporate actions.",
                              "Final history may include revisions unavailable at the historical decision time.",
                              "The discrepancy-masking scenario is exploratory after a source comparison, not a frozen selection rule.",
                              "The dates-only folds and warm-up rule are not frozen, and no model is evaluated."]}
    out = OUT / "kibot_review"
    out.mkdir(parents=True, exist_ok=True)
    path = out / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf8")
    return path, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("kibot", "alpha", "kpatel", "alpaca"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    path, result = audit(args.kibot, args.alpha, args.kpatel, args.alpaca)
    print(json.dumps({"path": str(path), "status": result["status"], "experiment_scored": False,
                      "publication_rights_verified": False}))


if __name__ == "__main__":
    main()
