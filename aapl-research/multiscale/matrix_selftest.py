"""Exercise matrix freezing, one-fold execution and resume on synthetic data.

Every price and every permission/review record created by this program is
fictional test data, kept in a separately labeled local directory. It does not
admit a provider, read market data, or produce research results.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import time

import numpy as np
import torch

from .admission import save_bars
from .aggregate_development import aggregate
from .bars import eligible_origins, file_digest
from .benchmark_matrix_synthetic import synthetic_bars
from .matrix_executor import execute, freeze
from .spec import DEV_END, OUT, PROFILES, TARGETS


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf8")


def item(path, quote=None):
    result = {"path": str(path.resolve()), "sha256": file_digest(path)}
    if quote is not None:
        result["grant_quote"] = quote
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True,
                        help="completed local synthetic shape benchmark")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Synthetic CUDA self-test requires a local GPU")
    benchmark = args.benchmark.resolve()
    if not benchmark.is_relative_to((OUT / "synthetic_matrix_benchmark").resolve()):
        raise ValueError("Benchmark is outside the synthetic benchmark directory")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    work = OUT / "synthetic_matrix_validation" / stamp
    source = OUT / "admission" / ("synthetic_test_" + stamp)
    work.mkdir(parents=True, exist_ok=False)
    source.mkdir(parents=True, exist_ok=False)
    marker = "SYNTHETIC TEST ONLY: all prices, reviews, permissions, and scores are fictional; not research evidence.\n"
    (work / "SYNTHETIC_ONLY.txt").write_text(marker, encoding="utf8")
    (source / "SYNTHETIC_ONLY.txt").write_text(marker, encoding="utf8")
    bars_by_frequency = {frequency: synthetic_bars(frequency) for frequency in ("minute", "hour", "day", "month")}
    files, rows = {}, []
    for frequency, bars in bars_by_frequency.items():
        target = source / f"bars_{frequency}.npz"
        save_bars(target, bars)
        files[target.name] = file_digest(target)
    for profile in PROFILES:
        bars = bars_by_frequency[profile.frequency]
        for horizon in profile.horizons:
            origins = eligible_origins(bars, max(profile.lookbacks), horizon,
                                       same_session=(profile.frequency == "minute"))
            target = source / f"origins_{profile.name}_h{horizon}.npy"
            np.save(target, origins, allow_pickle=False)
            files[target.name] = file_digest(target)
            for mode in TARGETS:
                rows.append({"profile": profile.name, "frequency": profile.frequency,
                             "horizon": horizon, "target": mode, "minimum_history_met": True})
    admission = {"schema_version": 1, "data_kind": "synthetic_test", "status": "synthetic_test_only",
                 "scope": "SYNTHETIC PRICES; no provider data and no research results",
                 "development_cutoff_new_york": DEV_END, "Q3_values_read": False, "experiment_scored": False,
                 "legacy_test_exposure": {"sha256": hashlib.sha256(b"synthetic ledger only").hexdigest()},
                 "split_actions": [], "tasks": rows, "files": files}
    admission_path = source / "admission.json"
    write_json(admission_path, admission)
    admission_hash = file_digest(admission_path)
    quote = "SYNTHETIC TEST ONLY: fictional generated prices may be used and aggregate test results may be saved."
    evidence_file = work / "synthetic_rights.txt"
    evidence_file.write_text(quote, encoding="utf8")
    io_started = time.monotonic()
    with np.load(source / "bars_minute.npz", allow_pickle=False) as data:
        count = len(data["close"])
    io_file = work / "synthetic_io.json"
    write_json(io_file, {"scope": "synthetic test source only", "synthetic_rows": count,
                         "elapsed_seconds": time.monotonic() - io_started})
    projection_file = work / "synthetic_projection.json"
    write_json(projection_file, {"scope": "synthetic self-test only", "maximum_seconds": 180,
                                 "projected_seconds": 120})
    person = {"reviewed_by": "automated synthetic self-test; not a provider approval",
              "reviewed_at_utc": datetime.now(timezone.utc).isoformat()}
    selected = [p.name for p in PROFILES]
    exposure_note = OUT / "POST_CUTOFF_EXPOSURE.md"
    review = {"data_kind": "synthetic_test", "admission_sha256": admission_hash, "selected_profiles": selected,
              "rights": {**person, "decision": "permitted", "training_allowed": True,
                         "publication_of_nonreconstructable_aggregate_results_allowed": True,
                         "admission_sha256": admission_hash, "selected_profiles": selected,
                         "daily_evidence": item(evidence_file, quote),
                         "minute_evidence": item(evidence_file, quote)},
              "quality": {**person, "decision": "approved", "admission_sha256": admission_hash,
                          "selected_profiles": selected,
                          "nominal_price_and_split_chain_checked": True, "material_discrepancies_disposition_fixed": True,
                          "review_notes": "SYNTHETIC TEST ONLY: generated series have no external market-data source."},
              "exposure": {**person, "decision": "reviewed_for_development_only", "admission_sha256": admission_hash,
                           "independent_test": False, "Q3_sealed": True,
                           "legacy_ledger_sha256": admission["legacy_test_exposure"]["sha256"],
                           "post_cutoff_note_sha256": file_digest(exposure_note)},
              "compute": {**person, "decision": "approved", "admission_sha256": admission_hash,
                          "device": "cuda", "device_name": torch.cuda.get_device_name(0),
                          "location": "local", "host": socket.gethostname(),
                          "maximum_wall_seconds": 180, "projected_wall_seconds": 120,
                          "real_source_io_reviewed": True, "benchmark_evidence": item(benchmark),
                          "real_source_io_evidence": item(io_file), "projection_evidence": item(projection_file)}}
    review_path = work / "synthetic_review.json"
    write_json(review_path, review)
    recipe = {"status": "reviewed_synthetic_test", "data_kind": "synthetic_test", "admission_sha256": admission_hash,
              "selected_profiles": selected, "omitted_profiles": {},
              **person, "fold_strategies": {p.name: "rolling" for p in PROFILES},
              "warmup": {"min_inner_fit": 24, "min_inner_stop": 4, "min_validation_groups": 6},
              "analysis": {"block_lengths_by_profile": {"annual": [12, 24], "quarterly": [3, 6],
                                                         "monthly": [21, 42], "hourly": [2, 5], "minute": [1, 5]},
                           "resamples": 100, "seed": 2026, "confidence": .95,
                           "development_p_values": False, "annual_nonoverlap": True},
              "training": {"max_epochs": 1, "patience": 1, "batch_size": 128, "learning_rate": .001,
                           "seed": 2026, "ridge_alphas": [.1, 1.0, 10.0]}}
    recipe_path = work / "synthetic_recipe.json"
    write_json(recipe_path, recipe)
    protocol_path = freeze(source, review_path, recipe_path)
    first = execute(source, review_path, protocol_path, max_jobs=1)
    if first["status"] != "paused_after_job_limit" or sum(j["status"] == "completed" for j in first["jobs"].values()) != 1:
        raise ValueError("First synthetic queue execution did not pause after one fold")
    second = execute(source, review_path, protocol_path, max_jobs=1)
    count = sum(j["status"] == "completed" for j in second["jobs"].values())
    if second["status"] != "paused_after_job_limit" or count != 2 or second["data_kind"] != "synthetic_test":
        raise ValueError("Synthetic queue did not verify and resume the next fold")
    run_dir = OUT / "synthetic_matrix_validation" / "runs" / file_digest(protocol_path)[:24]
    summary = aggregate(source, review_path, protocol_path, run_dir, "annual", 12, "future_close", allow_partial=True)
    if (summary["status"] != "synthetic_test_results_only" or not summary["partial"] or
            summary["completed_folds"] != [1, 2] or len(summary["models"]) != 8 or
            summary["development_p_values_reported"] is not False):
        raise ValueError("Synthetic partial-fold summary is incorrectly labeled")
    aggregate_path = work / "synthetic_aggregate.json"
    write_json(aggregate_path, summary)
    partial_selected = ["hourly", "minute"]
    partial_review = deepcopy(review)
    partial_review["selected_profiles"] = partial_selected
    partial_review["rights"]["selected_profiles"] = partial_selected
    partial_review["rights"]["daily_evidence"] = {}
    partial_review["quality"]["selected_profiles"] = partial_selected
    partial_review_path = work / "synthetic_partial_review.json"
    write_json(partial_review_path, partial_review)
    partial_recipe = deepcopy(recipe)
    partial_recipe["selected_profiles"] = partial_selected
    partial_recipe["omitted_profiles"] = {name: "SYNTHETIC TEST: this scale is not included in this batch."
                                           for name in ("annual", "quarterly", "monthly")}
    partial_recipe_path = work / "synthetic_partial_recipe.json"
    write_json(partial_recipe_path, partial_recipe)
    partial_protocol_path = freeze(source, partial_review_path, partial_recipe_path)
    partial_protocol = json.loads(partial_protocol_path.read_text(encoding="utf8"))
    partial_state = execute(source, partial_review_path, partial_protocol_path, max_jobs=1)
    if (partial_protocol["matrix_scope"] != "partial" or len(partial_protocol["jobs"]) != 70 or
            len(partial_protocol["task_inventory"]) != 20 or partial_state["matrix_scope"] != "partial" or
            partial_state["status"] != "paused_after_job_limit" or
            sum(job["status"] == "completed" for job in partial_state["jobs"].values()) != 1):
        raise ValueError("Synthetic partial-scale queue did not preserve its omitted-task scope")
    receipt = {"status": "passed_synthetic_test_only", "market_data_read": False, "formal_experiment": False,
               "scope": "Full and partial synthetic matrices; three synthetic fold executions in separate queue calls",
               "synthetic_admission": str(source), "synthetic_protocol": str(protocol_path),
               "jobs_frozen": len(json.loads(protocol_path.read_text(encoding="utf8"))["jobs"]),
               "completed_synthetic_folds": count, "queue_resumed_after_job_limit": True,
               "synthetic_aggregate": str(aggregate_path), "partial_aggregate_correctly_labeled": True,
               "partial_scale_protocol": str(partial_protocol_path), "partial_scale_jobs_frozen": len(partial_protocol["jobs"]),
               "partial_scale_folds_executed": 1, "omitted_scale_status_preserved": True,
               "consumed_wall_seconds": second["consumed_wall_seconds"],
               "limitations": ["Synthetic reviews are fictional test fixtures and do not grant market-data rights.",
                               "Only three synthetic folds were executed across both matrices, not the formal study.",
                               "An OS crash during a full queue has not been tested by this self-test."]}
    target = work / "selftest_receipt.json"
    write_json(target, receipt)
    print(json.dumps({"status": receipt["status"], "receipt": str(target),
                      "completed_synthetic_folds": count, "market_data_read": False}))


if __name__ == "__main__":
    main()
