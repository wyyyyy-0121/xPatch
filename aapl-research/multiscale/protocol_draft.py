"""Make a non-runnable twenty-task protocol and transparent update inventory."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from .bars import file_digest
from .spec import MODELS, OUT, PROFILES, TARGETS


def make_draft(admission: dict, admission_sha256: str):
    if admission.get("Q3_values_read") is not False or admission.get("experiment_scored") is not False:
        raise ValueError("Unexpected source admission state")
    rows = admission.get("tasks", [])
    expected = {(p.name, h, target) for p in PROFILES for h in p.horizons for target in TARGETS}
    identities = [(r["profile"], r["horizon"], r["target"]) for r in rows]
    if set(identities) != expected or len(identities) != len(expected):
        raise ValueError("Admission does not contain the exact twenty unique tasks")
    periods = {"minute": 390, "hour": 6, "day": 252, "month": 12}
    tasks, planned_updates = [], 0
    for row in rows:
        if row["frequency"] not in periods:
            raise ValueError("Unexpected task frequency")
        updates = sum(math.ceil(fold["inner_fit_origins"] / 128) * 20
                      for fold in row["folds"]) * 4
        planned_updates += updates
        tasks.append({"profile": row["profile"], "frequency": row["frequency"],
                      "horizon": row["horizon"], "target": row["target"],
                      "primary_endpoint": row["primary_endpoint"], "common_max_lookback": row["common_max_lookback"],
                      "lookback_candidates": row["lookback_candidates"],
                      "seasonal_period_trading_bars": periods[row["frequency"]],
                      "origins": row["origins"], "candidate_status": row["status"],
                      "minimum_inner_fit_origins": row["minimum_inner_fit_origins"],
                      "known_legacy_overlap_origins": row["known_legacy_overlap_origins"],
                      "folds": row["folds"], "proposed_warmup_folds": row["proposed_warmup_folds"],
                      "neural_update_upper_bound_one_seed": updates})
    return {"status": "draft_not_runnable", "phase": "development_only", "frozen": False,
            "admission_sha256": admission_sha256, "development_cutoff_new_york": admission["development_cutoff_new_york"],
            "models": list(MODELS), "targets": list(TARGETS), "tasks": tasks,
            "proposed_training": {"fixed_seed": 2026, "max_epochs": 20, "early_stopping_patience": 3,
                                  "batch_size": 128, "learning_rate": 0.001,
                                  "ridge_alpha_grid": [0.1, 1.0, 10.0],
                                  "lookback_policy": "common maximum per task; configurations remain proposals",
                                  "selection": "inner stopping partition only; outer scores after all models fitted"},
            "neural_update_upper_bound_one_seed_all_tasks": planned_updates,
            "update_count_limitations": ["Assumes every fold runs all 20 epochs; no early stopping savings.",
                                         "Includes blocked or inadequate-data tasks and is not permission to run them.",
                                         "Does not include validation, input preparation, checkpoints, Ridge or three-seed extensions.",
                                         "Not a runtime or 5090 throughput estimate."],
            "blocking_items": ["Academic data usage and publication rights need verification.",
                               "Annual history is insufficient in the Alpaca fallback.",
                               "Quarterly equal-fold early fitting is too small; proposed warm-up folds are not frozen.",
                               "Full source, exposure, hyperparameter and time-budget protocol must be frozen before scoring.",
                               "The existing cloud job is separate and must not be modified or competed with."],
            "independent_test": False, "Q3_values_read": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    args = parser.parse_args()
    directory = args.admission.resolve()
    if not directory.is_relative_to((OUT / "admission").resolve()):
        raise ValueError("Admission path outside the multiscale workspace")
    path = directory / "admission.json"
    with path.open(encoding="utf-8") as stream:
        receipt = json.load(stream)
    for name, digest in receipt["files"].items():
        if file_digest(directory / name) != digest:
            raise ValueError(f"Admission artifact identity changed: {name}")
    draft = make_draft(receipt, file_digest(path))
    out = OUT / "protocol_drafts"
    out.mkdir(parents=True, exist_ok=True)
    target = out / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    with target.open("x", encoding="utf-8") as stream:
        json.dump(draft, stream, indent=2, allow_nan=False)
    print(json.dumps({"path": str(target), "status": draft["status"], "tasks": len(draft["tasks"]),
                      "neural_update_upper_bound_one_seed_all_tasks": draft["neural_update_upper_bound_one_seed_all_tasks"]}))


if __name__ == "__main__":
    main()
