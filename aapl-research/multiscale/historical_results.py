"""Compile every historical-development task from verified saved predictions.

No fitting, model selection, holdout reads, or budget changes are performed.
Incomplete studies require --allow-partial; all missing cells remain visible.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np

from .aggregate_development import session_labels
from .evaluation import paired_block_bootstrap
from .historical_coverage import (CONTROLS, NEURAL, OUT, TASKS, StateChanged,
                                  digest, in_tree, read_json, require, snapshot)
from .historical_minute_check import MODELS, calculate, fold_sources, pool, score_fold


BLOCKS = {"annual": (12, 24), "quarterly": (3, 6), "monthly": (21, 42),
          "hourly": (2, 5), "minute": (1, 5)}
FREQUENCIES = {"annual": "month", "quarterly": "month", "monthly": "day",
               "hourly": "hour", "minute": "minute"}
NEURAL_MODELS = ("mlp", "gru", "lstm", "xpatch")
ADMISSION = OUT / "admission/alpaca_20261107T052228573242Z"
PERMISSION = OUT / "permission_confirmations/20261108_ALPACA_KIBOT_USER_CONFIRMED.json"
SOURCE_MANIFEST = OUT / "data_provenance/20261108_CURRENT_DEVELOPMENT_SOURCE_MANIFEST.json"
NIGHT = OUT / "private_low_load/20261107T153438623158Z"
SCALARS = ("endpoint_mse", "endpoint_rmse", "endpoint_mae", "path_mse", "path_mae",
           "relative_improvement", "direction_coverage", "direction_accuracy_when_active",
           "active_count", "correct_active_count")


def task_key(row):
    return row["profile"], int(row["horizon"]), row["target"]


def checked_coverage(coverage, current, *, allow_partial):
    expected = {(*task, fold) for task in TASKS for fold in range(1, 6)}
    rows = coverage.get("folds", [])
    keys = [(*task_key(row), int(row["fold"])) for row in rows]
    require(coverage.get("status") == "private_verified_coverage_snapshot" and
            coverage.get("script_sha256") == current.get("script_sha256") and
            len(keys) == len(expected) and set(keys) == expected,
            "Coverage must identify all 100 planned task-folds exactly once")
    live = {(*task_key(row), int(row["fold"])): row for row in current["folds"]}
    complete = [row for row in rows if row["status"] == "completed"]
    require(len(complete) == coverage.get("completed_folds"), "Coverage count differs")
    for row in complete:
        other = live[(*task_key(row), int(row["fold"]))]
        require(other["status"] == "completed" and row.get("evidence") == other.get("evidence"),
                "A completed fold no longer matches independently verified source evidence")
    require(allow_partial or len(complete) == len(expected),
            "The study is incomplete; partial reporting requires --allow-partial")
    return sorted(rows, key=lambda row: (*task_key(row), row["fold"]))


def pooled_paths(folds):
    """Pool unequal fold sizes by origins, retaining every prediction step."""
    result = pool(folds)
    for model in MODELS:
        for metric in ("per_step_mse", "per_step_mae"):
            arrays = [np.asarray(fold["models"][model][metric], dtype=np.float64) for fold in folds]
            require(all(a.ndim == 1 and a.shape == arrays[0].shape for a in arrays),
                    "Fold path dimensions differ")
            value = sum(fold["origins"] * a for fold, a in zip(folds, arrays)) / result["origins"]
            result["models"][model][metric] = value.tolist()
        require(math.isclose(result["models"][model]["per_step_mse"][-1],
                             result["models"][model]["endpoint_mse"], rel_tol=1e-12),
                "Pooled endpoint and final path step differ")
    return result


def descriptive_interval(difference, groups, grid, length):
    diff = np.asarray(difference, dtype=np.float64)
    observed = len(np.unique(groups))
    if length >= len(grid) or observed < 2:
        return {"status": "insufficient_groups", "block_length": length, "groups": len(grid),
                "observed_groups": observed, "origins": len(diff),
                "mean_paired_loss_difference": float(diff.mean()), "confidence_interval": None,
                "resamples": 0, "seed": 2026, "confidence": .95}
    value = paired_block_bootstrap(diff, groups, block_length=length, resamples=5000,
                                  seed=2026, confidence=.95, group_order=grid)
    value.pop("centered_one_sided_p_improvement", None)
    value["status"] = ("descriptive_development_only" if value["bootstrap_status"] == "ok"
                       else "degenerate_bootstrap")
    value["interpretation"] = "Post-exposure description; negative favors model; no independent confirmation."
    return value


def task_uncertainty(rows, labels, pooled):
    """Use the full source session grid, including gaps between observed groups."""
    differences = {model: [] for model in MODELS}
    subsets = {model: [] for model in MODELS}
    groups, selected_dates = [], []
    previous_selected = None
    profile, horizon, target = task_key(rows[0])
    for row in rows:
        target_dir, folders, saved, _, target_hash = fold_sources(row)
        path = target_dir / "targets.npz"
        require(digest(path) == target_hash, "Target identity changed during compilation")
        with np.load(path, allow_pickle=False) as archive:
            actual, anchor, origins, decision = (archive[key].copy() for key in
                                                ("actual", "anchor", "origins", "decision_ns"))
        require(origins.min() >= 0 and origins.max() + horizon < len(labels),
                "Origin or target session is outside admitted metadata")
        groups.append(labels[origins])
        predictions = {}
        selected = []
        if profile == "annual":
            for index, origin in enumerate(origins):
                if previous_selected is None or int(origin) >= previous_selected + horizon:
                    previous_selected = int(origin)
                    selected.append(index)
                    selected_dates.append({"origin": str(labels[origin]),
                                           "target_start": str(labels[origin + 1]),
                                           "target_end": str(labels[origin + horizon])})
        for model in MODELS:
            path = folders[model] / f"prediction_{model}.npz"
            require(digest(path) == saved[model]["prediction_sha256"],
                    "Prediction identity changed during compilation")
            with np.load(path, allow_pickle=False) as archive:
                require(np.array_equal(archive["origins"], origins) and
                        np.array_equal(archive["decision_ns"], decision), "Paired origins differ")
                predictions[model] = archive["prediction"].copy()
            if selected:
                subsets[model].append(calculate(actual[selected], anchor[selected],
                                                predictions[model][selected], target))
        base_loss = np.square(predictions["last_close"][:, -1] - actual[:, -1])
        for model in MODELS:
            differences[model].append(np.square(predictions[model][:, -1] - actual[:, -1]) - base_loss)
    groups = np.concatenate(groups)
    require(np.all(groups[1:] >= groups[:-1]), "Pooled session groups are unordered")
    grid = np.unique(labels)
    grid = grid[(grid >= groups[0]) & (grid <= groups[-1])]
    result = {"models": {}, "session_grid_groups": len(grid),
              "observed_groups": len(np.unique(groups)), "annual_nonoverlap": None}
    for model in MODELS:
        diff = np.concatenate(differences[model])
        expected = pooled["models"][model]["endpoint_mse"] - pooled["models"]["last_close"]["endpoint_mse"]
        require(math.isclose(float(diff.mean()), expected, rel_tol=1e-8, abs_tol=1e-12),
                "Pooled and paired losses disagree")
        result["models"][model] = [descriptive_interval(diff, groups, grid, n) for n in BLOCKS[profile]]
    if profile == "annual":
        nonoverlap = {"origins": len(selected_dates), "sessions": selected_dates, "models": {},
                      "interpretation": "Nonoverlapping labels do not establish independent observations."}
        for model in MODELS:
            values = subsets[model]
            n = sum(value["origins"] for value in values)
            nonoverlap["models"][model] = {
                key: sum(value["origins"] * value[key] for value in values) / n
                for key in ("endpoint_mse", "endpoint_mae", "path_mse", "path_mae")}
        baseline = nonoverlap["models"]["last_close"]["endpoint_mse"]
        for model in MODELS:
            nonoverlap["models"][model]["relative_improvement"] = (
                1 - nonoverlap["models"][model]["endpoint_mse"] / baseline if baseline else None)
        result["annual_nonoverlap"] = nonoverlap
    return result


def elapsed_record(raw, pause=None):
    require(math.isfinite(raw) and raw >= 0, "Invalid recorded elapsed time")
    estimate = 0.0 if pause is None else float(pause)
    require(math.isfinite(estimate) and estimate >= 0, "Invalid pause estimate")
    return {"raw_wall_seconds": raw, "pause_envelope_estimate_seconds": estimate,
            "suspension_adjusted_elapsed_estimate_seconds": max(0.0, raw - estimate),
            "timing_basis": ("raw_monotonic_envelope_less_external_pause_estimate" if pause is not None
                             else "recorded_elapsed_no_known_night_pause"),
            "exact_cpu_seconds": None}


def training_records(row, pauses, admission_hash):
    common = dict(zip(("profile", "horizon", "target"), task_key(row))) | {"fold": row["fold"]}
    if row["status"] != "completed":
        return ([common | {"model": model, "status": row["status"]} for model in NEURAL_MODELS],
                [common | {"stage": "all_models", "status": row["status"]}])
    if "fold_directory" in row["evidence"]:
        folder = in_tree(row["evidence"]["fold_directory"])
        receipt_path = folder / "fold_receipt.json"
        receipt = read_json(receipt_path)
        config = receipt["config"]
        require(receipt["data_identity"].split(":", 1)[1] == admission_hash,
                "Training data identity differs from the admitted source")
        raw = float(receipt["wall_seconds"])
        costs = [common | {"stage": "all_models", "status": "completed",
                          "receipt": str(receipt_path), "receipt_sha256": digest(receipt_path)} |
                 elapsed_record(raw, pauses.get(str(folder)))]
        saved, counts = receipt["models"], receipt
    else:
        folder = NEURAL
        config = read_json(NEURAL / "PRIVATE_PROTOCOL.json")
        require(config["admission_sha256"] == admission_hash, "First-fold admission differs")
        neural_result = read_json(NEURAL / "PRIVATE_RESULT.json")
        saved, counts = neural_result["models"], config
        costs = []
        for stage, directory in (("controls", CONTROLS), ("neural", NEURAL)):
            path = directory / "PRIVATE_RESULT.json"
            data = read_json(path)
            costs.append(common | {"stage": stage, "status": "completed", "receipt": str(path),
                                   "receipt_sha256": digest(path)} |
                         elapsed_record(float(data["elapsed_seconds"])))
    models = []
    for model in NEURAL_MODELS:
        record = saved[model]
        summary = folder / record.get("checkpoint_dir", model) / "training_summary.json"
        require(digest(summary) == record["training_summary_sha256"], "Training summary identity differs")
        models.append(common | {"model": model, "status": "completed", "lookback": config["lookback"],
                      "seed": config["seed"], "max_epochs": config["max_epochs"],
                      "patience": config["patience"], "batch_size": config["batch_size"],
                      "learning_rate": config["learning_rate"], "fit_origins": counts["fit_origins"],
                      "stop_origins": counts["stop_origins"], "validation_origins": counts["validation_origins"],
                      "epochs": record["epochs"], "best_epoch": record["best_epoch"],
                      "steps": record["steps"], "stopped_early": record["stopped_early"],
                      "epoch_cap_reached": record["epochs"] >= config["max_epochs"],
                      "model_elapsed_seconds_when_measured": record.get("model_elapsed_seconds"),
                      "training_identity": record["training_identity"],
                      "training_summary": str(summary), "training_summary_sha256": digest(summary)})
    return models, costs


def write_json(path, data):
    with path.open("x", encoding="utf8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compile_results(coverage_path, output, *, allow_partial=False):
    coverage_path, output = in_tree(coverage_path), in_tree(output)
    require(not output.exists(), "A result directory already exists; use a new destination")
    coverage = read_json(coverage_path)
    for attempt in range(3):
        try:
            current = snapshot()
            break
        except StateChanged:
            if attempt == 2:
                raise
            time.sleep(.2)
    rows = checked_coverage(coverage, current, allow_partial=allow_partial)
    admission_path = ADMISSION / "admission.json"
    admission, admission_hash = read_json(admission_path), digest(admission_path)
    require(admission.get("Q3_values_read") is False, "Admission includes protected data")
    permission = read_json(PERMISSION)
    require(permission.get("status") == "permitted_as_confirmed_by_user" and
            permission.get("evidence_type") == "direct_user_confirmation",
            "Expected the current direct-user permission record")
    recovery = read_json(NIGHT / "ACTUAL_RESUME_VERIFICATION.json")
    saved_night = NIGHT / "CHECKPOINT_AND_BUDGET_SNAPSHOT.json"
    night = read_json(saved_night)
    require(recovery["original_remaining_budgets_preserved"] is True,
            "Night recovery budget evidence differs")
    pauses = {str(in_tree(worker["fold_dir"])): recovery["pause_envelope_seconds"]
              for worker in night["workers"]}
    labels = {frequency: session_labels(ADMISSION, admission, frequency)
              for frequency in set(FREQUENCIES.values())}
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    protocol = {"status": "historical_development_result_compilation_plan",
                "coverage": str(coverage_path), "coverage_sha256": digest(coverage_path),
                "admission_sha256": admission_hash, "permission_sha256": digest(PERMISSION),
                "source_manifest_sha256": digest(SOURCE_MANIFEST),
                "night_snapshot_sha256": digest(saved_night),
                "actual_recovery_sha256": digest(NIGHT / "ACTUAL_RESUME_VERIFICATION.json"),
                "analysis": {"block_lengths_by_profile": BLOCKS, "resamples": 5000, "seed": 2026,
                             "confidence": .95, "group_grid": "full original ordered source sessions",
                             "paired_difference": "model minus persistence/zero endpoint squared loss"},
                "source_sha256": {str(Path(__file__).relative_to(OUT.parents[1])): digest(Path(__file__)),
                                   "multiscale/evaluation.py": digest(Path(__file__).with_name("evaluation.py")),
                                   "multiscale/historical_minute_check.py": digest(Path(__file__).with_name("historical_minute_check.py"))},
                "allow_partial": allow_partial, "independent_test": False,
                "p_values_reported": False, "Q3_values_read": False,
                "created_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "COMPILATION_PROTOCOL.json", protocol)
    grouped = defaultdict(list)
    for row in rows:
        grouped[task_key(row)].append(row)
    tasks, pooled_rows, fold_rows, step_rows, interval_rows, train_rows, cost_rows = [], [], [], [], [], [], []
    for task, task_rows in sorted(grouped.items()):
        common = dict(zip(("profile", "horizon", "target"), task))
        complete = [row for row in task_rows if row["status"] == "completed"]
        folds = [score_fold(row) for row in complete]
        missing = [row["fold"] for row in task_rows if row["status"] != "completed"]
        scope = "complete_five_fold_development" if not missing else "partial_development"
        pooled = pooled_paths(folds) if folds else None
        uncertainty = task_uncertainty(complete, labels[FREQUENCIES[task[0]]], pooled) if complete else None
        tasks.append(common | {"scope": scope, "completed_folds": [row["fold"] for row in complete],
                     "missing_folds": missing, "pooled": pooled, "folds": folds, "uncertainty": uncertainty})
        for row in task_rows:
            training, costs = training_records(row, pauses, admission_hash)
            train_rows.extend(training)
            cost_rows.extend(costs)
        for model in MODELS:
            pooled_rows.append(common | {"model": model, "scope": scope,
                               "completed_folds": len(complete), "missing_folds": ",".join(map(str, missing)),
                               "origins": pooled["origins"] if pooled else 0} |
                               ({key: pooled["models"][model][key] for key in SCALARS} if pooled else {}))
            for row in task_rows:
                match = next((fold for fold in folds if fold["fold"] == row["fold"]), None)
                fold_rows.append(common | {"model": model, "fold": row["fold"], "status": row["status"],
                                          "origins": match["origins"] if match else None} |
                                 ({key: match["models"][model][key] for key in SCALARS} if match else {}))
            for step in range(task[1]):
                step_rows.append(common | {"model": model, "scope": scope, "step": step + 1,
                                 "completed_folds": len(complete), "origins": pooled["origins"] if pooled else 0,
                                 "mse": pooled["models"][model]["per_step_mse"][step] if pooled else None,
                                 "mae": pooled["models"][model]["per_step_mae"][step] if pooled else None})
            for length in BLOCKS[task[0]]:
                item = next((value for value in uncertainty["models"][model]
                             if value["block_length"] == length), None) if uncertainty else None
                bounds = item.get("confidence_interval") if item else None
                interval_rows.append(common | {"model": model, "scope": scope, "block_length": length,
                                     "origins": pooled["origins"] if pooled else 0,
                                     "status": item["status"] if item else "no_completed_folds",
                                     "groups": item["groups"] if item else None,
                                     "observed_groups": item["observed_groups"] if item else None,
                                     "mean_model_minus_baseline_loss": item["mean_paired_loss_difference"] if item else None,
                                     "lower": bounds[0] if bounds else None, "upper": bounds[1] if bounds else None,
                                     "resamples": item["resamples"] if item else 0, "seed": 2026})
        print(json.dumps({"stage": "compiled_task", **common, "completed_folds": len(complete),
                          "tasks": len(tasks), "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
    require((len(pooled_rows), len(fold_rows), len(interval_rows), len(train_rows)) == (160, 800, 320, 400),
            "The complete planned reporting grid is not represented")
    files = {"METRICS_POOLED.csv": pooled_rows, "METRICS_BY_FOLD.csv": fold_rows,
             "METRICS_PER_STEP.csv": step_rows, "DESCRIPTIVE_INTERVALS.csv": interval_rows,
             "TRAINING_MODELS.csv": train_rows, "FOLD_ELAPSED.csv": cost_rows,
             "COVERAGE.csv": [{k: row[k] for k in ("profile", "horizon", "target", "fold", "status")} for row in rows]}
    for name, records in files.items():
        write_csv(output / name, records)
    costs = [row for row in cost_rows if row["status"] == "completed"]
    result = {"status": ("complete_historical_development_results" if all(not t["missing_folds"] for t in tasks)
                         else "partial_historical_development_results"),
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "protocol_sha256": digest(output / "COMPILATION_PROTOCOL.json"),
              "completed_folds": sum(len(t["completed_folds"]) for t in tasks),
              "complete_tasks": sum(not t["missing_folds"] for t in tasks), "tasks": tasks,
              "csv_sha256": {name: digest(output / name) for name in files},
              "interval_status_counts": dict(Counter(row["status"] for row in interval_rows)),
              "elapsed_accounting": {"unique_completed_stage_records": len(costs),
                                     "sum_recorded_wall_envelopes_seconds": sum(r["raw_wall_seconds"] for r in costs),
                                     "sum_suspension_adjusted_elapsed_estimates_seconds": sum(r["suspension_adjusted_elapsed_estimate_seconds"] for r in costs),
                                     "pause_adjusted_stage_records": sum(r["pause_envelope_estimate_seconds"] > 0 for r in costs),
                                     "exact_cpu_seconds": None,
                                     "note": "Stage sums are not study wall-clock duration. Parallel stages overlap; per-model CPU time was not measured. Suspension subtraction is an external estimate, not exact active computation. Queue deadlines are accounted separately; unfinished work is excluded."},
              "independent_test": False, "p_values_reported": False, "Q3_values_read": False,
              "compiler_elapsed_seconds": time.monotonic() - started,
              "limitations": ["All results concern previously exposed historical development periods.",
                              "Annual short-history variant has few overlapping market periods.",
                              "Finite epoch caps and early stopping do not establish convergence.",
                              "Missing cells remain missing; partial task losses use different period coverage.",
                              "Permission is recorded from direct user confirmation, not independently inspected contracts."]}
    write_json(output / "RESULTS.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = compile_results(args.coverage, args.output, allow_partial=args.allow_partial)
    print(json.dumps({key: result[key] for key in ("status", "completed_folds", "complete_tasks",
                                                  "interval_status_counts", "compiler_elapsed_seconds")} |
                     {"output": str(args.output.resolve()), "sha256": digest(args.output / "RESULTS.json")}))


if __name__ == "__main__":
    main()
