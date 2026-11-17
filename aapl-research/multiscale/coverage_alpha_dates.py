"""Plan long-horizon sample counts from dates and synthetic constant prices.

The union of dates is hypothetical: this does not splice provider prices,
admit a source, freeze a split, train models or calculate forecast scores.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .admission import checked_source, daily_bars, verify_page_rows
from .audit_alpha_candidates import read_candidate
from .bars import aggregate_months, eligible_origins, file_digest
from .spec import OUT
from .splits import rolling_folds, warmup_folds


def fold_counts(months, folds):
    output = []
    for fold in folds:
        inner = fold.get("inner")
        validation = fold["validation"]
        output.append({"fold": fold["fold"], "status": fold["status"],
                       "fit_origins": len(inner["fit"]) if inner else 0,
                       "stop_origins": len(inner["stop"]) if inner else 0,
                       "validation_origins": len(validation),
                       "validation_first": str(months.session_date[validation[0]]) if len(validation) else None,
                       "validation_last": str(months.session_date[validation[-1]]) if len(validation) else None})
    return output


def scenario(dates, excluded_dates):
    selected = sorted(set(dates) - set(excluded_dates))
    # Only dates affect these counts. Ones are deliberately synthetic, not prices.
    artificial = pd.DataFrame({"date": selected, "close": np.ones(len(selected))})
    daily, calendar_report = daily_bars(artificial)
    months = aggregate_months(daily)
    tasks = {}
    for profile, lookback, horizon in (("annual", 60, 12), ("quarterly", 36, 3)):
        origins = eligible_origins(months, lookback, horizon, same_session=False)
        equal = rolling_folds(months, origins, horizon)
        warmup = warmup_folds(months, origins, horizon, min_inner_fit=24, min_inner_stop=4,
                              min_validation_groups=6)
        tasks[profile] = {"lookback_months": lookback, "horizon_months": horizon,
                          "eligible_origins": len(origins),
                          "first_origin": str(months.session_date[origins[0]]) if len(origins) else None,
                          "last_origin": str(months.session_date[origins[-1]]) if len(origins) else None,
                          "equal_folds": fold_counts(months, equal),
                          "proposed_warmup_folds": fold_counts(months, warmup)}
    return {"excluded_dates": sorted(excluded_dates), "full_observed_months": int(months.observed.sum()),
            "missing_expected_dates": calendar_report["missing_sessions"],
            "cutoff_excluded_dates": calendar_report["cutoff_excluded_sessions"], "tasks": tasks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pratham", type=Path, required=True)
    parser.add_argument("--alpaca", type=Path, required=True)
    args = parser.parse_args()
    alpha_dir, _, alpha = read_candidate(args.pratham, "pratham")
    alpaca_dir, receipt, alpaca = checked_source(args.alpaca, "alpaca_daily_raw.csv")
    verify_page_rows(alpaca_dir, receipt, alpaca, "alpaca")
    dates = sorted(set(alpha.date).union(alpaca.date))
    if "2015-09-30" not in dates:
        raise ValueError("The prespecified source-disagreement sensitivity date is absent")
    result = {"status": "hypothetical_coverage_not_admitted", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "date union with synthetic constant prices; no source splice, model training or scoring",
              "Q3_values_read": False, "experiment_scored": False, "protocol_frozen": False,
              "source_receipt_sha256": {"alpha_attributed": file_digest(alpha_dir / "receipt.json"),
                                         "alpaca": file_digest(alpaca_dir / "receipt.json")},
              "earliest_date": dates[0], "latest_date": dates[-1],
              "scenarios": {"all_dates": scenario(dates, []),
                            "exclude_disputed_month_end": scenario(dates, ["2015-09-30"])},
              "limitations": ["The date union does not choose or merge actual prices from providers.",
                              "Source permissions, source precedence and point-in-time versions remain unresolved.",
                              "Warm-up folds are proposals, not a frozen protocol or proof of adequate training.",
                              "Exposure history and independent confirmation require separate review.",
                              "Exclusion of the disputed month is a sensitivity scenario, not a cleaning decision."]}
    out = OUT / "coverage_planning"
    out.mkdir(parents=True, exist_ok=True)
    path = out / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    with path.open("x", encoding="utf8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    compact = {name: {profile: {"origins": task["eligible_origins"],
                                "equal_first_fit": task["equal_folds"][0]["fit_origins"] if task["equal_folds"] else 0,
                                "warmup_first_fit": task["proposed_warmup_folds"][0]["fit_origins"]
                                if task["proposed_warmup_folds"] else 0}
                       for profile, task in data["tasks"].items()}
               for name, data in result["scenarios"].items()}
    print(json.dumps({"path": str(path), "status": result["status"], "scenarios": compact}))


if __name__ == "__main__":
    main()
