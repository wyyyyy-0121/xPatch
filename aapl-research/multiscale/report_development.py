"""Build a complete task-coverage table from verified development folds.

The existing admission and frozen-protocol checks remain mandatory. Partial
folds are never presented as complete; fictional self-tests are labeled on
every output. This program does not select models or perform confirmation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import date, datetime, timezone
import csv
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from .aggregate_development import aggregate
from .bars import file_digest
from .matrix_executor import _job_key, _json_inside, load_reviewed
from .matrix_gate import locked_protocol_issues, profile_selection
from .spec import MODELS, OUT, ROOT, tasks
from .training import digest_indices


METRICS = ("origins", "endpoint_mse", "endpoint_rmse", "endpoint_mae", "path_mse", "path_mae",
           "relative_improvement_vs_persistence", "direction_coverage", "direction_accuracy_when_active")
CSV_FIELDS = ("data_kind", "profile", "frequency", "horizon", "target", "primary_endpoint", "model",
              "task_status", "completed_folds", "missing_folds", "omission_reason", "seed",
              "squared_error_unit", "absolute_error_unit", *METRICS,
              "primary_block_length", "primary_loss_difference_lower", "primary_loss_difference_upper",
              "sensitivity_block_length", "sensitivity_loss_difference_lower", "sensitivity_loss_difference_upper",
              "annual_nonoverlapping_origins", "annual_first_origin_session", "annual_last_origin_session",
              "annual_first_target_session", "annual_last_target_session", "split_events_in_evaluated_paths",
              "cross_split_origins", "cross_split_endpoint_mse", "cross_split_endpoint_mae", "cross_split_path_mse",
              "cross_split_relative_improvement", "no_split_origins", "no_split_endpoint_mse",
              "no_split_endpoint_mae", "no_split_path_mse", "no_split_relative_improvement")
TRAINING_FIELDS = ("data_kind", "profile", "frequency", "horizon", "target", "fold", "model", "fit",
                   "fit_origins", "stop_origins", "validation_origins", "ridge_alpha", "max_epochs", "epochs",
                   "best_epoch", "steps", "stopped_early", "budget_cap_reached", "training_status",
                   "fold_wall_seconds", "fold_wall_time_policy", "resume_count")


@contextmanager
def stable_matrix(run_dir: Path):
    """Do not produce a report while the executor is mutating its state."""
    lock_path = run_dir / "matrix.lock"
    if not lock_path.is_file():
        raise ValueError("Matrix lock file is missing")
    with lock_path.open("r") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Matrix is running; wait for a stable fold boundary") from None
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _contains_p_value(value):
    if isinstance(value, dict):
        return any((key not in ("development_p_values_reported", "development_p_values") or item is not False) and
                   (re.search(r"(^|_)p($|_)|p[_-]?value", str(key), re.I) or _contains_p_value(item))
                   for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_p_value(item) for item in value)
    return False


def _validate_summary(summary, task, completed, protocol_hash, admission_hash, data_kind, analysis):
    expected_status = "synthetic_test_results_only" if data_kind == "synthetic_test" else "historical_development_only"
    expected_task = {key: task[key] for key in ("profile", "frequency", "horizon", "target")}
    if (not isinstance(summary, dict) or summary.get("status") != expected_status or
            summary.get("data_kind") != data_kind or summary.get("independent_test") is not False or
            summary.get("development_p_values_reported") is not False or
            summary.get("protocol_sha256") != protocol_hash or summary.get("admission_sha256") != admission_hash or
            summary.get("task") != expected_task or summary.get("completed_folds") != completed or
            summary.get("missing_folds") != [i for i in range(1, 6) if i not in completed] or
            summary.get("partial") != (len(completed) != 5) or summary.get("analysis") != analysis or
            not isinstance(summary.get("models"), dict) or set(summary["models"]) != set(MODELS)):
        raise ValueError("Aggregate identity, scope, folds, or model inventory differs")
    if _contains_p_value(summary):
        raise ValueError("Development reports may not contain inferential p-values")
    baseline_values = summary["models"]["last_close"]
    if not isinstance(baseline_values, dict):
        raise ValueError("Missing baseline metrics")
    baseline = baseline_values.get("endpoint_mse")
    reference_count = baseline_values.get("origins")
    if isinstance(baseline, bool) or not isinstance(baseline, (float, int)) or not math.isfinite(baseline) or baseline < 0:
        raise ValueError("Invalid baseline error")
    split_reference = baseline_values.get("split_event_sensitivity")
    event_count = summary.get("split_events_in_evaluated_paths")
    if (not isinstance(split_reference, dict) or set(split_reference) != {"crosses_split", "no_split"} or
            any(not isinstance(split_reference[name], dict) or type(split_reference[name].get("origins")) is not int or
                split_reference[name]["origins"] < 0 for name in split_reference) or
            sum(row["origins"] for row in split_reference.values()) != reference_count or
            type(event_count) is not int or event_count < 0 or
            (event_count == 0) != (split_reference["crosses_split"]["origins"] == 0)):
        raise ValueError("Invalid split-event subgroup counts")
    annual_reference = baseline_values.get("annual_nonoverlapping_origins") if task["profile"] == "annual" else None
    date_fields = ("first_origin_session", "last_origin_session", "first_target_session", "last_target_session")
    if task["profile"] == "annual":
        if (not isinstance(annual_reference, dict) or type(annual_reference.get("origins")) is not int or
                not 1 <= annual_reference["origins"] <= reference_count):
            raise ValueError("Annual nonoverlapping origin count is invalid")
        try:
            dates = [date.fromisoformat(annual_reference[key]) for key in date_fields]
            canonical = all(isinstance(annual_reference[key], str) and
                            value.isoformat() == annual_reference[key] for key, value in zip(date_fields, dates))
        except (KeyError, ValueError, TypeError):
            raise ValueError("Annual nonoverlapping date coverage is invalid") from None
        if (not canonical or dates[0] > dates[1] or dates[0] >= dates[2] or
                dates[1] >= dates[3] or dates[2] > dates[3]):
            raise ValueError("Annual nonoverlapping date coverage is invalid")
    for model, values in summary["models"].items():
        if (not isinstance(values, dict) or type(values.get("origins")) is not int or values["origins"] < 1 or
                values["origins"] != reference_count):
            raise ValueError(f"Invalid aggregate origin count for {model}")
        for name in METRICS[1:]:
            value = values.get(name)
            optional = name in ("relative_improvement_vs_persistence", "direction_accuracy_when_active")
            if value is None and optional:
                continue
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"Invalid {name} for {model}")
        if any(values[name] < 0 for name in ("endpoint_mse", "endpoint_rmse", "endpoint_mae", "path_mse", "path_mae")):
            raise ValueError(f"Negative error metric for {model}")
        if not 0 <= values["direction_coverage"] <= 1 or (
                values["direction_accuracy_when_active"] is not None and
                not 0 <= values["direction_accuracy_when_active"] <= 1):
            raise ValueError(f"Invalid direction metric for {model}")
        if ((values["direction_coverage"] == 0) != (values["direction_accuracy_when_active"] is None)):
            raise ValueError(f"Direction abstention and accuracy disagree for {model}")
        for step_key, endpoint_key, path_key in (("per_step_mse", "endpoint_mse", "path_mse"),
                                                 ("per_step_mae", "endpoint_mae", "path_mae")):
            step = values.get(step_key)
            if (not isinstance(step, list) or len(step) != task["horizon"] or
                    any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) or x < 0 for x in step) or
                    not math.isclose(step[-1], values[endpoint_key], rel_tol=1e-10, abs_tol=1e-15) or
                    not math.isclose(sum(step) / len(step), values[path_key], rel_tol=1e-10, abs_tol=1e-15)):
                raise ValueError(f"Step, endpoint, and path metrics disagree for {model}")
        if not math.isclose(math.sqrt(values["endpoint_mse"]), values["endpoint_rmse"], rel_tol=1e-10, abs_tol=1e-15):
            raise ValueError(f"Endpoint MSE and RMSE disagree for {model}")
        folds = values.get("per_fold_endpoint_mse")
        if (not isinstance(folds, list) or any(not isinstance(x, dict) for x in folds) or
                [x.get("fold") for x in folds] != completed or
                any(type(x.get("origins")) is not int or x["origins"] < 1 or
                    isinstance(x.get("endpoint_mse"), bool) or not isinstance(x.get("endpoint_mse"), (int, float)) or
                    not math.isfinite(x["endpoint_mse"]) or x["endpoint_mse"] < 0 for x in folds) or
                sum(x["origins"] for x in folds) != values["origins"] or
                not math.isclose(sum(x["origins"] * x["endpoint_mse"] for x in folds) / values["origins"],
                                 values["endpoint_mse"], rel_tol=1e-10, abs_tol=1e-15)):
            raise ValueError(f"Fold and pooled endpoint errors disagree for {model}")
        improvement = values["relative_improvement_vs_persistence"]
        expected = 1.0 - values["endpoint_mse"] / baseline if baseline and baseline > 0 else None
        if ((expected is None) != (improvement is None) or
                (expected is not None and not math.isclose(improvement, expected, rel_tol=1e-10, abs_tol=1e-15))):
            raise ValueError(f"Relative baseline improvement disagrees for {model}")
        if annual_reference is not None:
            annual = values.get("annual_nonoverlapping_origins")
            if (not isinstance(annual, dict) or
                    any(annual.get(key) != annual_reference.get(key) for key in ("origins", *date_fields)) or
                    annual.get("selection") != "earliest eligible origin, then at least H completed monthly bars apart"):
                raise ValueError(f"Annual nonoverlapping coverage differs for {model}")
            for metric in ("endpoint_mse", "endpoint_mae"):
                value = annual.get(metric)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"Annual nonoverlapping {metric} is invalid for {model}")
            base_annual = annual_reference.get("endpoint_mse")
            expected_annual = (1 - annual["endpoint_mse"] / base_annual
                               if isinstance(base_annual, (float, int)) and not isinstance(base_annual, bool) and
                               math.isfinite(base_annual) and base_annual > 0 else None)
            observed_annual = annual.get("relative_improvement_vs_persistence")
            if ((expected_annual is None) != (observed_annual is None) or
                    (expected_annual is not None and
                     (isinstance(observed_annual, bool) or not isinstance(observed_annual, (float, int)) or
                      not math.isfinite(observed_annual) or
                      not math.isclose(observed_annual, expected_annual, rel_tol=1e-10, abs_tol=1e-15)))):
                raise ValueError(f"Annual nonoverlapping relative improvement disagrees for {model}")
        split_groups = values.get("split_event_sensitivity")
        if not isinstance(split_groups, dict) or set(split_groups) != set(split_reference):
            raise ValueError(f"Missing split-event subgroups for {model}")
        weighted = {name: 0.0 for name in ("endpoint_mse", "endpoint_mae", "path_mse")}
        for group_name, reference in split_reference.items():
            subgroup = split_groups.get(group_name)
            n_group = reference["origins"]
            if (not isinstance(subgroup, dict) or type(subgroup.get("origins")) is not int or
                    subgroup["origins"] != n_group or
                    subgroup.get("status") != ("descriptive_development_only" if n_group else "no_origins")):
                raise ValueError(f"Invalid split-event subgroup for {model}")
            for metric in weighted:
                value = subgroup.get(metric)
                if n_group:
                    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                        raise ValueError(f"Invalid split-event {metric} for {model}")
                    weighted[metric] += n_group * value
                elif value is not None:
                    raise ValueError(f"An empty split-event subgroup has a {metric} for {model}")
            base_group_mse = reference.get("endpoint_mse")
            expected_group = (1 - subgroup["endpoint_mse"] / base_group_mse
                              if n_group and isinstance(base_group_mse, (int, float)) and base_group_mse > 0 else None)
            observed_group = subgroup.get("relative_improvement_vs_persistence")
            if ((expected_group is None) != (observed_group is None) or
                    (expected_group is not None and
                     (isinstance(observed_group, bool) or not isinstance(observed_group, (int, float)) or
                      not math.isfinite(observed_group) or
                      not math.isclose(observed_group, expected_group, rel_tol=1e-10, abs_tol=1e-15)))):
                raise ValueError(f"Split-event relative improvement disagrees for {model}")
        if any(not math.isclose(weighted[name] / values["origins"], values[name], rel_tol=1e-10, abs_tol=1e-15)
               for name in weighted):
            raise ValueError(f"Split-event and pooled errors disagree for {model}")
        intervals = values.get("descriptive_block_intervals")
        expected_lengths = analysis["block_lengths_by_profile"][task["profile"]]
        if (not isinstance(intervals, list) or any(not isinstance(x, dict) for x in intervals) or
                [x.get("block_length") for x in intervals] != expected_lengths):
            raise ValueError(f"Missing predeclared block intervals for {model}")
        for interval in intervals:
            if interval.get("status") == "descriptive_development_only":
                limits = interval.get("confidence_interval")
                if (not isinstance(limits, list) or len(limits) != 2 or
                        any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
                            for v in limits) or limits[0] > limits[1]):
                    raise ValueError(f"Invalid descriptive interval for {model}")
            elif interval.get("status") != "insufficient_groups":
                raise ValueError(f"Unexpected block interval status for {model}")


def assemble(protocol, state, protocol_hash, admission_hash, review_hash, *, allow_partial, aggregate_task,
             training_task=None):
    """Assemble all twenty tasks without inventing missing scores."""
    data_kind = protocol.get("data_kind")
    if data_kind not in ("market", "synthetic_test") or type(allow_partial) is not bool:
        raise ValueError("Invalid report scope")
    selected = profile_selection(protocol.get("selected_profiles"))
    inventory = protocol.get("task_inventory")
    if not isinstance(inventory, list) or len(inventory) != 20:
        raise ValueError("Frozen task inventory must contain twenty tasks")
    task_list = tasks()
    expected_keys = {(item["profile"], item["horizon"], item["target"]) for item in task_list}
    inventory_by_key = {(item.get("profile"), item.get("horizon"), item.get("target")): item for item in inventory}
    if len(inventory_by_key) != 20 or set(inventory_by_key) != expected_keys:
        raise ValueError("Frozen task inventory differs from the study scope")
    jobs = protocol.get("jobs")
    records = state.get("jobs")
    if not isinstance(jobs, list) or not isinstance(records, dict):
        raise ValueError("Missing matrix jobs or state")
    job_keys = [_job_key(job) for job in jobs]
    if len(set(job_keys)) != len(job_keys) or set(records) - set(job_keys):
        raise ValueError("Matrix state contains unknown or repeated jobs")
    omitted = protocol.get("omitted_profiles")
    if not isinstance(omitted, dict):
        raise ValueError("Missing omitted-profile reasons")
    seed = protocol.get("training", {}).get("seed")
    if type(seed) is not int:
        raise ValueError("A recorded training seed is required")
    assembled = []
    allowed_states = {"not_started", "running", "completed", "failed", "interrupted_at_budget"}
    for task in task_list:
        key = (task["profile"], task["horizon"], task["target"])
        declaration = inventory_by_key[key]
        included = task["profile"] in selected
        if included:
            if declaration.get("status") != "included" or declaration.get("reason") is not None:
                raise ValueError("Included task differs from its frozen declaration")
            task_jobs = [job for job in jobs if (job["profile"], job["horizon"], job["target"]) == key]
            if sorted(job["fold"] for job in task_jobs) != list(range(1, 6)):
                raise ValueError("Each included task must have exactly five frozen folds")
            task_jobs.sort(key=lambda item: item["fold"])
            fold_states = []
            for job in task_jobs:
                record = records.get(_job_key(job), {})
                status = record.get("status", "not_started") if isinstance(record, dict) else None
                if status not in allowed_states:
                    raise ValueError("Unknown fold state")
                fold_states.append({"fold": job["fold"], "status": status})
            completed = [row["fold"] for row in fold_states if row["status"] == "completed"]
            failed = any(row["status"] in ("failed", "interrupted_at_budget") for row in fold_states)
            if len(completed) == 5:
                status = "complete"
            elif completed and allow_partial:
                status = "partial_descriptive_with_failure" if failed else "partial_descriptive"
            elif completed:
                status = "partial_unreported_with_failure" if failed else "partial_unreported"
            elif failed:
                status = "failed_or_interrupted"
            elif any(row["status"] == "running" for row in fold_states):
                status = "running_or_interrupted"
            else:
                status = "not_started"
            summary = None
            training_folds = None
            if completed and (len(completed) == 5 or allow_partial):
                summary = aggregate_task(task, len(completed) != 5)
                _validate_summary(summary, task, completed, protocol_hash, admission_hash, data_kind, protocol["analysis"])
                if training_task is not None:
                    training_folds = [training_task(job, records[_job_key(job)])
                                      for job in task_jobs if job["fold"] in completed]
                    if [row.get("fold") for row in training_folds] != completed:
                        raise ValueError("Training metadata does not match completed folds")
            reason = None
        else:
            if (declaration.get("status") != "omitted_before_scoring" or
                    declaration.get("reason") != omitted.get(task["profile"]) or
                    not isinstance(declaration.get("reason"), str) or not declaration["reason"].strip()):
                raise ValueError("Omitted task lacks its frozen reason")
            if any((job["profile"], job["horizon"], job["target"]) == key for job in jobs):
                raise ValueError("Omitted task unexpectedly has matrix jobs")
            fold_states, completed, summary, training_folds = [], [], None, None
            status, reason = "omitted_before_scoring", declaration["reason"]
        assembled.append({"profile": task["profile"], "frequency": task["frequency"],
                          "horizon": task["horizon"], "target": task["target"],
                          "primary_endpoint": task["primary"], "status": status, "omission_reason": reason,
                          "fold_states": fold_states, "completed_folds": completed,
                          "missing_folds": [i for i in range(1, 6) if i not in completed],
                          "summary": summary, "training_folds": training_folds})
    return {"schema_version": 1,
            "status": "synthetic_test_report_only" if data_kind == "synthetic_test" else "historical_development_report",
            "data_kind": data_kind, "independent_test": False, "development_p_values_reported": False,
            "protocol_sha256": protocol_hash, "admission_sha256": admission_hash, "review_sha256": review_hash,
            "matrix_scope": protocol["matrix_scope"], "selected_profiles": selected, "omitted_profiles": omitted,
            "seed": seed, "seed_count": 1, "allow_partial": allow_partial,
            "complete_twenty_task_matrix": all(row["status"] == "complete" for row in assembled),
            "task_status_counts": dict(Counter(row["status"] for row in assembled)), "tasks": assembled,
            "limitations": ["Synthetic reports contain only fictional test data; market reports are historical development.",
                            "Partial folds are descriptive, not a completed five-fold comparison.",
                            "No cross-task averaging, independent confirmation, ensemble, or multi-seed analysis is performed.",
                            "Fold wall time is not per-model time and may undercount abrupt interruptions.",
                            "Data rights and permission to publish derived results require separate review."]}


def _checked_local_file(root: Path, relative: str, expected: str):
    if not isinstance(relative, str) or not relative or not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Missing saved training artifact identity")
    root = root.resolve(); path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file() or file_digest(path) != expected:
        raise ValueError("Saved training artifact is missing, outside its fold, or changed")
    return path


def _training_record(run_dir, job, state_record, protocol_hash, admission_hash, protocol):
    data_kind = protocol["data_kind"]
    relative = state_record.get("directory")
    if not isinstance(relative, str) or not relative:
        raise ValueError("Completed fold directory is missing")
    root = run_dir.resolve(); folder = (root / relative).resolve()
    if not folder.is_relative_to(root):
        raise ValueError("Completed fold escapes the matrix run")
    receipt_path = _checked_local_file(folder, "fold_receipt.json", state_record.get("receipt_sha256"))
    receipt = json.loads(receipt_path.read_text(encoding="utf8"))
    cfg = receipt.get("config", {})
    training = protocol.get("training", {})
    training_keys = ("ridge_alphas", "max_epochs", "patience", "batch_size", "learning_rate", "seed")
    if (receipt.get("status") != "completed" or receipt.get("phase") != "development" or
            receipt.get("independent_test") is not False or receipt.get("Q3_values_read") is not False or
            receipt.get("data_identity") != data_kind + ":" + admission_hash or
            cfg.get("protocol_identity") != protocol_hash or
            any(cfg.get(key) != job.get(key) for key in ("profile", "horizon", "target", "lookback", "seasonal_period"))):
        raise ValueError("Saved fold training identity differs from its matrix job")
    if any(key not in training or cfg.get(key) != training[key] for key in training_keys):
        raise ValueError("Saved training configuration differs from the frozen configuration")
    counts = [receipt.get(key) for key in ("fit_origins", "stop_origins", "validation_origins")]
    wall, resumes = receipt.get("wall_seconds"), receipt.get("resume_count")
    if (any(type(value) is not int or value < 1 for value in counts) or
            counts != [job.get(key) for key in ("fit_origins", "stop_origins", "validation_origins")] or
            isinstance(wall, bool) or not isinstance(wall, (float, int)) or not math.isfinite(wall) or wall < 0 or
            type(resumes) is not int or resumes < 0 or not isinstance(receipt.get("wall_time_policy"), str)):
        raise ValueError("Invalid saved fold timing or origin counts")
    identity = receipt.get("identity_fields")
    sources, frozen_sources = receipt.get("source_sha256"), protocol.get("source_code_sha256")
    split_keys = ("train_indices_sha256", "fit_indices_sha256", "stop_indices_sha256", "validation_indices_sha256")
    if (not isinstance(identity, dict) or identity.get("config") != cfg or
            identity.get("data_identity") != receipt["data_identity"] or identity.get("source_sha256") != sources or
            not isinstance(sources, dict) or not sources or not isinstance(frozen_sources, dict) or
            any(frozen_sources.get(name) != digest for name, digest in sources.items()) or
            any(identity.get(key) != job.get(key) or not isinstance(job.get(key), str) or len(job[key]) != 64
                for key in split_keys)):
        raise ValueError("Saved fold sources or split identities differ from the frozen job")
    try:
        identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                                                   allow_nan=False).encode("utf8")).hexdigest()
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid saved fold identity") from exc
    if receipt.get("fold_identity") != identity_hash:
        raise ValueError("Saved fold identity hash differs from its fields")
    targets_path = _checked_local_file(folder, "targets.npz", receipt.get("targets_sha256"))
    with np.load(targets_path, allow_pickle=False) as saved:
        origins = saved["origins"]
    if (origins.ndim != 1 or origins.dtype.kind not in "iu" or len(origins) != counts[2] or
            digest_indices(origins) != job["validation_indices_sha256"]):
        raise ValueError("Saved validation origins differ from the frozen split")
    models = receipt.get("models")
    if not isinstance(models, dict) or set(models) != set(MODELS):
        raise ValueError("Fold training metadata lacks a model")
    collected = {}
    for name in MODELS:
        details = models[name]
        if name in ("last_close", "drift", "seasonal"):
            if details.get("fit") != "none":
                raise ValueError("Unexpected training status for an untrained baseline")
            collected[name] = {"fit": "none", "training_status": "untrained_baseline"}
        elif name == "ridge":
            alpha = details.get("alpha")
            if (details.get("fit") != "inner_only" or alpha not in cfg.get("ridge_alphas", []) or
                    details.get("training_origins") != counts[0]):
                raise ValueError("Ridge training metadata differs from the fold")
            collected[name] = {"fit": "inner_only", "ridge_alpha": alpha,
                               "training_status": "inner_fit_with_stop_selection"}
        else:
            rel = details.get("checkpoint_dir", name)
            if (not isinstance(rel, str) or Path(rel).name != rel or
                    (rel != name and not (rel.startswith(name + "-retry-") and rel[len(name) + 7:].isdigit()))):
                raise ValueError("Invalid neural checkpoint directory")
            summary_path = _checked_local_file(folder, rel + "/training_summary.json", details.get("training_summary_sha256"))
            summary = json.loads(summary_path.read_text(encoding="utf8"))
            if (summary.get("status") != "completed" or summary.get("identity") != details.get("training_identity") or
                    any(summary.get(key) != details.get(key) for key in ("epochs", "best_epoch", "steps", "stopped_early"))):
                raise ValueError("Neural training summary differs from the fold receipt")
            epochs, best, steps, stopped = (summary.get(key) for key in ("epochs", "best_epoch", "steps", "stopped_early"))
            history = summary.get("history")
            if (type(epochs) is not int or not 1 <= epochs <= cfg["max_epochs"] or type(best) is not int or
                    not 1 <= best <= epochs or type(steps) is not int or steps < 1 or type(stopped) is not bool or
                    not isinstance(history, list) or len(history) != epochs or
                    any(not isinstance(row, dict) for row in history) or
                    [row.get("epoch") for row in history] != list(range(1, epochs + 1))):
                raise ValueError("Invalid neural training history dimensions")
            previous_steps = 0
            for row in history:
                current_steps = row.get("steps")
                losses = (row.get("training_scaled_path_loss"), row.get("inner_stop_metric"))
                if (type(current_steps) is not int or current_steps <= previous_steps or
                        any(isinstance(v, bool) or not isinstance(v, (float, int)) or
                            not math.isfinite(v) or v < 0 for v in losses)):
                    raise ValueError("Invalid neural training history values")
                previous_steps = current_steps
            if previous_steps != steps or (not stopped and epochs != cfg["max_epochs"]):
                raise ValueError("Neural training stop state is inconsistent")
            best_loss = min(row["inner_stop_metric"] for row in history)
            if (not math.isclose(summary.get("inner_stop_metric", float("nan")), best_loss, rel_tol=1e-10, abs_tol=1e-15) or
                    history[best - 1]["inner_stop_metric"] != best_loss):
                raise ValueError("Best checkpoint does not match inner stopping history")
            cap_reached = epochs == cfg["max_epochs"]
            training_status = ("early_stop_fired_at_max_epochs_convergence_unverified" if stopped and cap_reached else
                               "early_stop_fired_convergence_unverified" if stopped else
                               "max_epochs_reached_convergence_unverified")
            collected[name] = {"fit": "inner_only", "max_epochs": cfg["max_epochs"], "epochs": epochs,
                               "best_epoch": best, "steps": steps, "stopped_early": stopped,
                               "budget_cap_reached": cap_reached, "history": history,
                               "training_status": training_status}
    return {"fold": job["fold"], "fit_origins": counts[0], "stop_origins": counts[1],
            "validation_origins": counts[2], "fold_wall_seconds": wall,
            "fold_wall_time_policy": receipt["wall_time_policy"], "resume_count": resumes, "models": collected}


def build_report(admission_dir: Path, review_path: Path, protocol_path: Path, run_dir: Path, *, allow_partial=False):
    protocol, protocol_hash, verified_protocol_path = _json_inside(Path(protocol_path), OUT)
    selected = profile_selection(protocol.get("selected_profiles"))
    admission, admission_hash, source, review, review_hash, verified_review_path = load_reviewed(
        Path(admission_dir), Path(review_path), selected_profiles=selected)
    issues = locked_protocol_issues(admission, admission_hash, protocol, review_hash)
    if issues:
        raise ValueError("Frozen report blocked: " + ", ".join(issues))
    data_kind = protocol["data_kind"]
    root = (OUT / "synthetic_matrix_validation" / "runs" if data_kind == "synthetic_test" else OUT / "matrix_runs").resolve()
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_relative_to(root) or run_dir.name != protocol_hash[:24]:
        raise ValueError("Matrix run does not match the frozen protocol")
    with stable_matrix(run_dir):
        state_path = run_dir / "matrix_state.json"
        state_hash = file_digest(state_path)
        state = json.loads(state_path.read_text(encoding="utf8"))
        if (state.get("protocol_sha256") != protocol_hash or state.get("review_sha256") != review_hash or
                state.get("data_kind") != data_kind or state.get("matrix_scope") != protocol["matrix_scope"] or
                state.get("selected_profiles") != selected):
            raise ValueError("Matrix state differs from the reviewed protocol")
        def aggregate_task(task, partial):
            return aggregate(Path(admission_dir), Path(review_path), Path(protocol_path), run_dir,
                             task["profile"], task["horizon"], task["target"], allow_partial=partial)
        def training_task(job, record):
            return _training_record(run_dir, job, record, protocol_hash, admission_hash, protocol)
        report = assemble(protocol, state, protocol_hash, admission_hash, review_hash,
                          allow_partial=allow_partial, aggregate_task=aggregate_task, training_task=training_task)
        exposure_hash = review["exposure"]["post_cutoff_note_sha256"]
        identities = ((state_path, state_hash), (source / "admission.json", admission_hash),
                      (verified_review_path, review_hash), (verified_protocol_path, protocol_hash),
                      (OUT / "POST_CUTOFF_EXPOSURE.md", exposure_hash))
        if any(file_digest(path) != expected for path, expected in identities):
            raise ValueError("Matrix, admission, review, protocol, or exposure identity changed during reporting")
    report.update(matrix_state_sha256=state_hash, matrix_status=state.get("status"),
                  matrix_consumed_wall_seconds=state.get("consumed_wall_seconds"),
                  exposure_note_sha256=exposure_hash,
                  reporting_source_sha256=file_digest(Path(__file__)))
    return report


def rows(report):
    result = []
    for task in report["tasks"]:
        summary = task["summary"]
        for model in MODELS:
            row = {"data_kind": report["data_kind"], "profile": task["profile"], "frequency": task["frequency"],
                   "horizon": task["horizon"], "target": task["target"], "primary_endpoint": task["primary_endpoint"],
                   "model": model, "task_status": task["status"],
                   "completed_folds": ",".join(map(str, task["completed_folds"])),
                   "missing_folds": ",".join(map(str, task["missing_folds"])),
                   "omission_reason": task["omission_reason"], "seed": report["seed"],
                   "squared_error_unit": "USD^2/share^2" if task["target"] == "future_close" else "log_return^2",
                   "absolute_error_unit": "USD/share" if task["target"] == "future_close" else "log_return"}
            values = summary["models"][model] if summary is not None else {}
            row.update({name: values.get(name) for name in METRICS})
            intervals = values.get("descriptive_block_intervals", [])
            for index, label in enumerate(("primary", "sensitivity")):
                interval = intervals[index] if len(intervals) > index else {}
                limits = interval.get("confidence_interval") if interval.get("status") == "descriptive_development_only" else None
                row[label + "_block_length"] = interval.get("block_length")
                row[label + "_loss_difference_lower"] = limits[0] if limits else None
                row[label + "_loss_difference_upper"] = limits[1] if limits else None
            nonoverlap = values.get("annual_nonoverlapping_origins") or {}
            row["annual_nonoverlapping_origins"] = nonoverlap.get("origins")
            for name in ("first_origin_session", "last_origin_session", "first_target_session", "last_target_session"):
                row["annual_" + name] = nonoverlap.get(name)
            row["split_events_in_evaluated_paths"] = summary.get("split_events_in_evaluated_paths") if summary else None
            split_groups = values.get("split_event_sensitivity") or {}
            for name, label in (("crosses_split", "cross_split"), ("no_split", "no_split")):
                subgroup = split_groups.get(name) or {}
                row[label + "_origins"] = subgroup.get("origins")
                for metric in ("endpoint_mse", "endpoint_mae", "path_mse"):
                    row[label + "_" + metric] = subgroup.get(metric)
                row[label + "_relative_improvement"] = subgroup.get("relative_improvement_vs_persistence")
            result.append(row)
    return result


def training_rows(report):
    result = []
    for task in report["tasks"]:
        for fold in task.get("training_folds") or []:
            for model in MODELS:
                values = fold["models"][model]
                result.append({"data_kind": report["data_kind"], "profile": task["profile"],
                               "frequency": task["frequency"], "horizon": task["horizon"], "target": task["target"],
                               "fold": fold["fold"], "model": model, "fit": values.get("fit"),
                               "fit_origins": fold["fit_origins"], "stop_origins": fold["stop_origins"],
                               "validation_origins": fold["validation_origins"], "ridge_alpha": values.get("ridge_alpha"),
                               "max_epochs": values.get("max_epochs"), "epochs": values.get("epochs"),
                               "best_epoch": values.get("best_epoch"), "steps": values.get("steps"),
                               "stopped_early": values.get("stopped_early"),
                               "budget_cap_reached": values.get("budget_cap_reached"),
                               "training_status": values.get("training_status"), "fold_wall_seconds": fold["fold_wall_seconds"],
                               "fold_wall_time_policy": fold["fold_wall_time_policy"], "resume_count": fold["resume_count"]})
    return result


def write_report(report):
    root = (OUT / "synthetic_matrix_validation" / "reports" if report["data_kind"] == "synthetic_test"
            else OUT / "development_reports")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = root / stamp
    directory.mkdir(parents=True, exist_ok=False)
    prefix = "SYNTHETIC_ONLY_" if report["data_kind"] == "synthetic_test" else "DEVELOPMENT_ONLY_"
    json_path, csv_path, training_path, md_path = (directory / (prefix + name) for name in
                                                   ("report.json", "model_rows.csv", "training_rows.csv", "coverage.md"))
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8")
    with csv_path.open("x", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="raise")
        writer.writeheader(); writer.writerows(rows(report))
    with training_path.open("x", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TRAINING_FIELDS, extrasaction="raise")
        writer.writeheader(); writer.writerows(training_rows(report))
    title = "SYNTHETIC TEST ONLY — NOT MARKET RESULTS" if report["data_kind"] == "synthetic_test" else "HISTORICAL DEVELOPMENT ONLY — NOT INDEPENDENT CONFIRMATION"
    lines = ["# " + title, "", f"Protocol: `{report['protocol_sha256']}`; seed: `{report['seed']}`.",
             "", "Missing values have not been calculated; no tasks are ranked or averaged across units.",
             "Training rows cover only verified completed folds; fold wall time is not per-model time.", "",
             "| Scale | Horizon | Target | Primary | Status | Completed folds | Missing folds | Reason |",
             "| --- | ---: | --- | --- | --- | --- | --- | --- |"]
    for task in report["tasks"]:
        reason = (task["omission_reason"] or "").replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {task['profile']} | {task['horizon']} | {task['target']} | {task['primary_endpoint']} | "
                     f"{task['status']} | {','.join(map(str, task['completed_folds'])) or '—'} | "
                     f"{','.join(map(str, task['missing_folds'])) or '—'} | {reason} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf8")
    manifest = {"status": report["status"], "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "data_kind": report["data_kind"], "independent_test": False,
                "protocol_sha256": report["protocol_sha256"], "matrix_state_sha256": report["matrix_state_sha256"],
                "files_sha256": {path.name: file_digest(path) for path in (json_path, csv_path, training_path, md_path)},
                "reporting_source_sha256": file_digest(Path(__file__))}
    manifest_path = directory / (prefix + "manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf8")
    return {"status": report["status"], "directory": str(directory), "manifest": str(manifest_path),
            "task_status_counts": report["task_status_counts"], "model_rows": len(rows(report)),
            "training_rows": len(training_rows(report))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("admission", "review", "protocol", "run"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = build_report(args.admission, args.review, args.protocol, args.run, allow_partial=args.allow_partial)
    print(json.dumps(write_report(result), allow_nan=False))


if __name__ == "__main__":
    main()
