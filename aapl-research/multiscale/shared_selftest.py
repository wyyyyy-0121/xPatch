"""Exercise two shared-horizon queue folds using fictional synthetic inputs only.

The sparse generated timestamps, fictional reviews and one-epoch runs are
engineering fixtures. They do not provide market results, data rights, a real
compute authorization, or a representative time projection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket

import numpy as np
import pandas as pd
import torch

from .admission import save_bars
from .bars import Bars, MINUTE_NS, eligible_origins, file_digest
from .corporate_actions import SPLIT_SENSITIVITY_RULE
from .matrix_executor import freeze as freeze_base
from .shared_executor import execute, freeze
from .spec import DEV_END, NY, OUT, PROFILES, TARGETS
from .verify_shared_development import check_shared_fold


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf8")


def _item(path, quote=None):
    result = {"path": str(path.resolve()), "sha256": file_digest(path)}
    if quote is not None:
        result["grant_quote"] = quote
    return result


def _synthetic_minutes():
    # Widely spaced fictional sessions keep this a small engineering test.
    dates = pd.date_range("2022-01-04", periods=48, freq="14D")
    count = 100
    starts_local = (dates + pd.Timedelta(hours=9, minutes=30)).tz_localize(NY)
    starts = np.repeat(starts_local.tz_convert("UTC").as_unit("ns").asi8, count)
    starts += np.tile(np.arange(count, dtype=np.int64), len(dates)) * MINUTE_NS
    labels = np.repeat(dates.strftime("%Y-%m-%d").to_numpy(dtype="U10"), count)
    pos = np.arange(len(starts), dtype=np.float64)
    close = 100.0 + 0.0001 * pos + np.sin(pos / 31.0)
    return Bars("minute", starts, starts + MINUTE_NS, close, np.ones(len(starts), dtype=bool), labels)


def run(benchmark: Path):
    if not torch.cuda.is_available():
        raise RuntimeError("Synthetic shared-queue self-test requires a local CUDA device")
    benchmark = Path(benchmark).resolve()
    if not benchmark.is_relative_to((OUT / "synthetic_matrix_benchmark").resolve()) or not benchmark.is_file():
        raise ValueError("A previously recorded synthetic benchmark is required")
    torch.set_num_threads(2)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    work = OUT / "synthetic_mechanism_validation" / stamp
    source = OUT / "admission" / ("synthetic_test_shared_" + stamp)
    work.mkdir(parents=True, exist_ok=False)
    source.mkdir(parents=True, exist_ok=False)
    marker = "SYNTHETIC TEST ONLY: prices, reviews, permissions, budgets and scores are fictional.\n"
    (work / "SYNTHETIC_ONLY.txt").write_text(marker, encoding="utf8")
    (source / "SYNTHETIC_ONLY.txt").write_text(marker, encoding="utf8")
    bars = _synthetic_minutes()
    bars_path = source / "bars_minute.npz"
    save_bars(bars_path, bars)
    files = {bars_path.name: file_digest(bars_path)}
    minute = next(profile for profile in PROFILES if profile.name == "minute")
    for horizon in minute.horizons:
        path = source / f"origins_minute_h{horizon}.npy"
        np.save(path, eligible_origins(bars, max(minute.lookbacks), horizon, same_session=True), allow_pickle=False)
        files[path.name] = file_digest(path)
    rows = [{"profile": profile.name, "frequency": profile.frequency, "horizon": horizon,
             "target": target, "minimum_history_met": profile.name == "minute"}
            for profile in PROFILES for horizon in profile.horizons for target in TARGETS]
    admission = {"schema_version": 1, "data_kind": "synthetic_test", "status": "synthetic_test_only",
                 "scope": "Fictional sparse minute fixture; not market data or research results",
                 "development_cutoff_new_york": DEV_END, "Q3_values_read": False, "experiment_scored": False,
                 "legacy_test_exposure": {"sha256": hashlib.sha256(b"fictional exposure ledger").hexdigest()},
                 "split_actions": [], "tasks": rows, "files": files}
    admission_path = source / "admission.json"
    _write(admission_path, admission)
    admission_hash = file_digest(admission_path)
    person = {"reviewed_by": "fictional synthetic self-test reviewer; not a provider or user approval",
              "reviewed_at_utc": datetime.now(timezone.utc).isoformat()}
    quote = "SYNTHETIC TEST ONLY: fictional prices and aggregate test results may be used in this fixture."
    rights_path = work / "fictional_rights.txt"
    rights_path.write_text(quote, encoding="utf8")
    io_path, base_projection_path = work / "fictional_io.json", work / "fictional_base_projection.json"
    _write(io_path, {"scope": "synthetic fixture only", "rows": len(bars.close)})
    _write(base_projection_path, {"scope": "synthetic fixture only", "maximum_seconds": 180})
    selected = ["minute"]
    review = {"data_kind": "synthetic_test", "admission_sha256": admission_hash, "selected_profiles": selected,
              "rights": {**person, "decision": "permitted", "training_allowed": True,
                         "publication_of_nonreconstructable_aggregate_results_allowed": True,
                         "admission_sha256": admission_hash, "selected_profiles": selected,
                         "minute_evidence": _item(rights_path, quote)},
              "quality": {**person, "decision": "approved", "admission_sha256": admission_hash,
                          "selected_profiles": selected, "nominal_price_and_split_chain_checked": True,
                          "material_discrepancies_disposition_fixed": True,
                          "review_notes": "SYNTHETIC TEST ONLY: no external price source or corporate action exists."},
              "exposure": {**person, "decision": "reviewed_for_development_only", "admission_sha256": admission_hash,
                           "independent_test": False, "Q3_sealed": True,
                           "legacy_ledger_sha256": admission["legacy_test_exposure"]["sha256"],
                           "post_cutoff_note_sha256": file_digest(OUT / "POST_CUTOFF_EXPOSURE.md")},
              "compute": {**person, "decision": "approved", "admission_sha256": admission_hash,
                          "device": "cuda", "device_name": torch.cuda.get_device_name(0), "location": "local",
                          "host": socket.gethostname(), "maximum_wall_seconds": 180, "projected_wall_seconds": 120,
                          "real_source_io_reviewed": True, "benchmark_evidence": _item(benchmark),
                          "real_source_io_evidence": _item(io_path), "projection_evidence": _item(base_projection_path)}}
    review_path = work / "fictional_base_review.json"
    _write(review_path, review)
    recipe = {"status": "reviewed_synthetic_test", "data_kind": "synthetic_test", "admission_sha256": admission_hash,
              "selected_profiles": selected, "omitted_profiles": {p.name: "SYNTHETIC TEST: this scale is not generated."
                                                                    for p in PROFILES if p.name not in selected},
              **person, "fold_strategies": {p.name: "rolling" for p in PROFILES},
              "warmup": {"min_inner_fit": 24, "min_inner_stop": 4, "min_validation_groups": 6},
              "analysis": {"block_lengths_by_profile": {"annual": [12, 24], "quarterly": [3, 6],
                                                         "monthly": [21, 42], "hourly": [2, 5], "minute": [1, 5]},
                           "resamples": 100, "seed": 2026, "confidence": .95, "development_p_values": False,
                           "annual_nonoverlap": True, "split_event_sensitivity": SPLIT_SENSITIVITY_RULE},
              "training": {"max_epochs": 1, "patience": 1, "batch_size": 128, "learning_rate": .001,
                           "seed": 2026, "ridge_alphas": [.1, 1.0, 10.0]}}
    recipe_path = work / "fictional_recipe.json"
    _write(recipe_path, recipe)
    base_protocol = freeze_base(source, review_path, recipe_path)
    base_hash, review_hash = file_digest(base_protocol), file_digest(review_path)
    projection = {**person, "status": "synthetic_mechanism_projection_only", "data_kind": "synthetic_test",
                  "admission_sha256": admission_hash, "base_protocol_sha256": base_hash,
                  "benchmark_sha256": file_digest(benchmark), "seed_count": 3, "fold_count": 5,
                  "models_per_fold": 5, "Q3_values_read": False, "includes_io_scoring_and_verification": True,
                  "projected_wall_seconds": 120}
    projection_path = work / "fictional_mechanism_projection.json"
    _write(projection_path, projection)
    mechanism_review = {**person, "schema_version": 1, "status": "reviewed_synthetic_test",
                        "data_kind": "synthetic_test", "admission_sha256": admission_hash,
                        "base_protocol_sha256": base_hash, "base_review_sha256": review_hash,
                        "base_matrix_state_sha256": None, "development_only": True, "independent_test": False,
                        "Q3_sealed": True, "selection_before_mechanism_scoring": True,
                        "prior_core_exposure_acknowledged": True, "synthetic_fixture_only": True,
                        "seeds": [2026, 2027, 2028],
                        "compute": {"host": socket.gethostname(), "device": "cuda",
                                    "device_name": torch.cuda.get_device_name(0), "location": "local",
                                    "maximum_wall_seconds": 180, "projected_wall_seconds": 120,
                                    "projection_evidence": _item(projection_path)}}
    mechanism_review_path = work / "fictional_mechanism_review.json"
    _write(mechanism_review_path, mechanism_review)
    protocol = freeze(source, review_path, base_protocol, mechanism_review_path)
    first = execute(source, review_path, base_protocol, mechanism_review_path, protocol, max_jobs=1)
    second = execute(source, review_path, base_protocol, mechanism_review_path, protocol, max_jobs=1)
    if (first["status"] != "paused_after_job_limit" or second["status"] != "paused_after_job_limit" or
            len(first["jobs"]) != 1 or len(second["jobs"]) != 2 or
            not second["consumed_wall_seconds"] > first["consumed_wall_seconds"] > 0 or
            any(record["status"] != "completed" for record in second["jobs"].values())):
        raise ValueError("Synthetic queue did not pause and resume with cumulative budget")
    run_dir = OUT / "synthetic_mechanism_validation" / "runs" / second["budget_identity"][:24]
    for record in second["jobs"].values():
        checked = check_shared_fold(run_dir / record["directory"], expected_data_identity="synthetic_test:" + admission_hash,
                                    expected_protocol_identity=file_digest(protocol))
        if checked["models_verified"] != 5 or checked["fold_receipt_sha256"] != record["receipt_sha256"]:
            raise ValueError("Synthetic shared-fold independent verification failed")
    state_path = run_dir / "mechanism_state.json"
    before = file_digest(state_path)
    refrozen = freeze(source, review_path, base_protocol, mechanism_review_path)
    try:
        execute(source, review_path, base_protocol, mechanism_review_path, refrozen, max_jobs=1)
    except ValueError as exc:
        if "budget cannot be reset" not in str(exc):
            raise
    else:
        raise ValueError("Synthetic budget was reset by refreezing")
    if file_digest(state_path) != before:
        raise ValueError("Refreezing altered the existing synthetic budget state")
    receipt = {"status": "passed_synthetic_test_only", "market_data_read": False, "formal_experiment": False,
               "independent_test": False, "synthetic_source": str(source), "base_protocol": str(base_protocol),
               "mechanism_protocol": str(protocol), "run_directory": str(run_dir), "frozen_folds": 15,
               "completed_synthetic_folds": 2, "models_verified_per_fold": 5,
               "consumed_wall_seconds_first_call": first["consumed_wall_seconds"],
               "consumed_wall_seconds_after_resume": second["consumed_wall_seconds"],
               "refreeze_did_not_reset_budget": True,
               "limitations": ["Fictional reviews do not grant rights or user authorization.",
                               "Only two of 15 sparse synthetic folds ran; no core market matrix was executed.",
                               "Partial-fold checkpoint recovery and a full OS crash were not exercised.",
                               "This fixture cannot estimate real training time, convergence, or predictive value."]}
    target = work / "selftest_receipt.json"
    _write(target, receipt)
    return target, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    args = parser.parse_args()
    target, receipt = run(args.benchmark)
    print(json.dumps({"status": receipt["status"], "receipt": str(target),
                      "completed_synthetic_folds": receipt["completed_synthetic_folds"], "market_data_read": False}))


if __name__ == "__main__":
    main()
