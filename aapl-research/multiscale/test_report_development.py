"""Tests use fictional metrics and never assert a real market result."""
from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import math

import numpy as np
import pytest

from . import report_development as report_module
from .bars import file_digest
from .corporate_actions import SPLIT_SENSITIVITY_RULE
from .matrix_executor import _job_key
from .spec import MODELS, PROFILES, tasks
from .training import digest_indices


PROTOCOL_HASH = "a" * 64
ADMISSION_HASH = "b" * 64
REVIEW_HASH = "c" * 64


def fictional_protocol():
    selected = ["hourly", "minute"]
    omitted = {name: "SYNTHETIC TEST ONLY: omitted before any scoring." for name in ("annual", "quarterly", "monthly")}
    inventory = []
    jobs = []
    for task in tasks():
        profile = task["profile"]
        included = profile in selected
        inventory.append({"profile": profile, "horizon": task["horizon"], "target": task["target"],
                          "status": "included" if included else "omitted_before_scoring",
                          "reason": None if included else omitted[profile]})
        if included:
            jobs.extend({"profile": profile, "horizon": task["horizon"], "target": task["target"], "fold": fold}
                        for fold in range(1, 6))
    return {"data_kind": "synthetic_test", "selected_profiles": selected, "omitted_profiles": omitted,
            "matrix_scope": "partial", "task_inventory": inventory, "jobs": jobs,
            "training": {"seed": 2026},
            "analysis": {"block_lengths_by_profile": {p.name: [1, 2] for p in PROFILES},
                         "resamples": 100, "seed": 2026, "confidence": 0.95,
                         "development_p_values": False, "annual_nonoverlap": True,
                         "split_event_sensitivity": SPLIT_SENSITIVITY_RULE}}


def fictional_summary(task, completed, analysis):
    model_data = {}
    for index, model in enumerate(MODELS):
        mse = 4.0 if model == "last_close" else float(index + 1)
        rmse = math.sqrt(mse)
        model_data[model] = {"origins": 3 * len(completed), "endpoint_mse": mse, "endpoint_rmse": rmse,
                             "endpoint_mae": rmse, "path_mse": mse, "path_mae": rmse,
                             "per_step_mse": [mse] * task["horizon"], "per_step_mae": [rmse] * task["horizon"],
                             "relative_improvement_vs_persistence": 1 - mse / 4.0,
                             "direction_coverage": 0.0 if model == "last_close" else 0.5,
                             "direction_accuracy_when_active": None if model == "last_close" else 0.5,
                             "split_event_sensitivity": {
                                 name: {"status": "descriptive_development_only", "origins": count * len(completed),
                                        "endpoint_mse": mse, "endpoint_mae": rmse, "path_mse": mse,
                                        "relative_improvement_vs_persistence": 1 - mse / 4.0}
                                 for name, count in (("crosses_split", 1), ("no_split", 2))},
                             "per_fold_endpoint_mse": [{"fold": fold, "endpoint_mse": mse, "origins": 3}
                                                       for fold in completed],
                             "descriptive_block_intervals": [
                                 {"block_length": length, "status": "descriptive_development_only",
                                  "confidence_interval": [-1.0, 1.0]} for length in
                                 analysis["block_lengths_by_profile"][task["profile"]]]}
    return {"status": "synthetic_test_results_only", "data_kind": "synthetic_test", "independent_test": False,
            "development_p_values_reported": False, "protocol_sha256": PROTOCOL_HASH,
            "admission_sha256": ADMISSION_HASH,
            "task": {key: task[key] for key in ("profile", "frequency", "horizon", "target")},
            "completed_folds": completed, "missing_folds": [x for x in range(1, 6) if x not in completed],
            "partial": len(completed) != 5, "analysis": analysis, "split_events_in_evaluated_paths": 1,
            "models": model_data}


def one_partial_state(protocol):
    selected = [job for job in protocol["jobs"] if job["profile"] == "hourly" and
                job["horizon"] == 1 and job["target"] == "future_close"]
    return {"jobs": {_job_key(job): {"status": "completed" if job["fold"] <= 2 else "failed"}
                     for job in selected[:3]}}


def test_missing_and_omitted_tasks_remain_visible_and_partial_is_opt_in():
    protocol = fictional_protocol(); state = one_partial_state(protocol); calls = []
    def aggregate_task(task, partial):
        calls.append((task["profile"], task["horizon"], task["target"], partial))
        return fictional_summary(task, [1, 2], protocol["analysis"])
    default = report_module.assemble(protocol, state, PROTOCOL_HASH, ADMISSION_HASH, REVIEW_HASH,
                                     allow_partial=False, aggregate_task=aggregate_task)
    assert not calls and default["status"] == "synthetic_test_report_only"
    assert default["task_status_counts"] == {"omitted_before_scoring": 6,
                                              "partial_unreported_with_failure": 1, "not_started": 13}
    assert not default["complete_twenty_task_matrix"]
    assert len(report_module.rows(default)) == 160
    assert all(row["endpoint_mse"] is None for row in report_module.rows(default))
    partial = report_module.assemble(protocol, state, PROTOCOL_HASH, ADMISSION_HASH, REVIEW_HASH,
                                     allow_partial=True, aggregate_task=aggregate_task)
    assert calls == [("hourly", 1, "future_close", True)]
    assert partial["task_status_counts"]["partial_descriptive_with_failure"] == 1
    computed = [row for row in report_module.rows(partial) if row["endpoint_mse"] is not None]
    assert len(computed) == 8 and all(row["data_kind"] == "synthetic_test" for row in computed)
    assert all(row["completed_folds"] == "1,2" and row["missing_folds"] == "3,4,5" for row in computed)
    assert {row["squared_error_unit"] for row in computed} == {"USD^2/share^2"}
    assert all(row["cross_split_origins"] == 2 and row["no_split_origins"] == 4 for row in computed)


def test_report_rejects_inconsistent_metrics_and_p_values():
    protocol = fictional_protocol(); state = one_partial_state(protocol)
    def bad_steps(task, partial):
        summary = fictional_summary(task, [1, 2], protocol["analysis"])
        summary["models"]["mlp"]["per_step_mse"][-1] += 1.0
        return summary
    with pytest.raises(ValueError, match="metrics disagree"):
        report_module.assemble(protocol, state, PROTOCOL_HASH, ADMISSION_HASH, REVIEW_HASH,
                               allow_partial=True, aggregate_task=bad_steps)
    def bad_p(task, partial):
        summary = fictional_summary(task, [1, 2], protocol["analysis"])
        summary["models"]["mlp"]["descriptive_block_intervals"][0]["centered_one_sided_p_improvement"] = 0.01
        return summary
    with pytest.raises(ValueError, match="p-values"):
        report_module.assemble(protocol, state, PROTOCOL_HASH, ADMISSION_HASH, REVIEW_HASH,
                               allow_partial=True, aggregate_task=bad_p)
    def bad_subgroup(task, partial):
        summary = fictional_summary(task, [1, 2], protocol["analysis"])
        summary["models"]["mlp"]["split_event_sensitivity"]["crosses_split"]["endpoint_mse"] += 1.0
        return summary
    with pytest.raises(ValueError, match="Split-event"):
        report_module.assemble(protocol, state, PROTOCOL_HASH, ADMISSION_HASH, REVIEW_HASH,
                               allow_partial=True, aggregate_task=bad_subgroup)


def test_complete_task_is_not_misrepresented_as_complete_study():
    protocol = fictional_protocol()
    jobs = [job for job in protocol["jobs"] if job["profile"] == "hourly" and
            job["horizon"] == 1 and job["target"] == "future_close"]
    state = {"jobs": {_job_key(job): {"status": "completed"} for job in jobs}}
    def aggregate_task(task, partial):
        assert not partial
        return fictional_summary(task, [1, 2, 3, 4, 5], protocol["analysis"])
    report = report_module.assemble(protocol, state, PROTOCOL_HASH, ADMISSION_HASH, REVIEW_HASH,
                                    allow_partial=False, aggregate_task=aggregate_task)
    assert report["task_status_counts"]["complete"] == 1
    assert not report["complete_twenty_task_matrix"] and report["seed_count"] == 1


def test_empty_split_subgroup_stays_empty_in_report():
    protocol = fictional_protocol()
    jobs = [job for job in protocol["jobs"] if job["profile"] == "hourly" and
            job["horizon"] == 1 and job["target"] == "future_close"]
    state = {"jobs": {_job_key(job): {"status": "completed"} for job in jobs}}
    def aggregate_task(task, partial):
        summary = fictional_summary(task, [1, 2, 3, 4, 5], protocol["analysis"])
        summary["split_events_in_evaluated_paths"] = 0
        for values in summary["models"].values():
            groups = values["split_event_sensitivity"]
            groups["crosses_split"] = {"status": "no_origins", "origins": 0, "endpoint_mse": None,
                                       "endpoint_mae": None, "path_mse": None,
                                       "relative_improvement_vs_persistence": None}
            groups["no_split"]["origins"] = values["origins"]
        return summary
    report = report_module.assemble(protocol, state, PROTOCOL_HASH, ADMISSION_HASH, REVIEW_HASH,
                                    allow_partial=False, aggregate_task=aggregate_task)
    row = next(row for row in report_module.rows(report) if row["profile"] == "hourly" and
               row["horizon"] == 1 and row["target"] == "future_close" and row["model"] == "mlp")
    assert row["cross_split_origins"] == 0 and row["cross_split_endpoint_mse"] is None
    assert row["no_split_origins"] == 15 and row["split_events_in_evaluated_paths"] == 0


def test_export_labels_synthetic_rows_and_leaves_missing_values_empty(tmp_path, monkeypatch):
    protocol = fictional_protocol(); state = one_partial_state(protocol)
    report = report_module.assemble(protocol, state, PROTOCOL_HASH, ADMISSION_HASH, REVIEW_HASH,
        allow_partial=True, aggregate_task=lambda task, partial: fictional_summary(task, [1, 2], protocol["analysis"]))
    report["matrix_state_sha256"] = "d" * 64
    monkeypatch.setattr(report_module, "OUT", tmp_path)
    result = report_module.write_report(report)
    directory = tmp_path / "synthetic_matrix_validation" / "reports" / result["directory"].split("/")[-1]
    assert result["model_rows"] == 160
    assert "SYNTHETIC TEST ONLY" in (directory / "SYNTHETIC_ONLY_coverage.md").read_text(encoding="utf8")
    with (directory / "SYNTHETIC_ONLY_model_rows.csv").open(encoding="utf8", newline="") as stream:
        data = list(csv.DictReader(stream))
    assert len(data) == 160 and {row["data_kind"] for row in data} == {"synthetic_test"}
    assert all(not row["endpoint_mse"] for row in data if row["task_status"] == "not_started")
    loaded = json.loads((directory / "SYNTHETIC_ONLY_report.json").read_text(encoding="utf8"))
    assert loaded["independent_test"] is False and loaded["development_p_values_reported"] is False


def test_report_will_not_read_a_matrix_with_an_active_writer(tmp_path):
    lock = tmp_path / "matrix.lock"; lock.touch()
    with lock.open("r+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="Matrix is running"):
            with report_module.stable_matrix(tmp_path):
                pass
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    with report_module.stable_matrix(tmp_path):
        pass


def test_fictional_training_history_is_verified_and_budget_limit_is_disclosed(tmp_path):
    job = {"profile": "minute", "horizon": 5, "target": "future_log_return", "fold": 1,
           "lookback": 30, "seasonal_period": 12, "fit_origins": 10, "stop_origins": 3,
           "validation_origins": 4, "train_indices_sha256": "1" * 64, "fit_indices_sha256": "2" * 64,
           "stop_indices_sha256": "3" * 64,
           "validation_indices_sha256": digest_indices(np.array([1, 2, 3, 4], dtype=np.int64))}
    training = {"max_epochs": 3, "ridge_alphas": [0.1, 1.0], "patience": 1, "batch_size": 8,
                "learning_rate": 0.001, "seed": 2026}
    sources = {"multiscale/training.py": "f" * 64}
    protocol = {"data_kind": "synthetic_test", "training": training, "source_code_sha256": sources}
    cfg = {**{key: job[key] for key in ("profile", "horizon", "target", "lookback", "seasonal_period")},
           "protocol_identity": PROTOCOL_HASH, **training}
    folder = tmp_path / "fictional-fold"; folder.mkdir()
    targets_path = folder / "targets.npz"
    np.savez(targets_path, origins=np.array([1, 2, 3, 4], dtype=np.int64))
    history = [{"epoch": epoch, "steps": 3 * epoch, "training_scaled_path_loss": 4.0 / epoch,
                "inner_stop_metric": value} for epoch, value in enumerate((4.0, 2.0, 3.0), start=1)]
    models = {name: {"fit": "none"} for name in ("last_close", "drift", "seasonal")}
    models["ridge"] = {"fit": "inner_only", "alpha": 0.1, "training_origins": 10}
    for name in ("mlp", "gru", "lstm", "xpatch"):
        model_dir = folder / name; model_dir.mkdir()
        summary = {"status": "completed", "identity": "e" * 64, "epochs": 3, "best_epoch": 2, "steps": 9,
                   "stopped_early": False, "inner_stop_metric": 2.0, "history": history}
        path = model_dir / "training_summary.json"; path.write_text(json.dumps(summary), encoding="utf8")
        models[name] = {"fit": "inner_only", "checkpoint_dir": name, "training_identity": "e" * 64,
                        "epochs": 3, "best_epoch": 2, "steps": 9, "stopped_early": False,
                        "training_summary_sha256": file_digest(path)}
    data_identity = "synthetic_test:" + ADMISSION_HASH
    identity_fields = {"config": cfg, "data_identity": data_identity, "source_sha256": sources,
                       **{key: job[key] for key in ("train_indices_sha256", "fit_indices_sha256",
                                                    "stop_indices_sha256", "validation_indices_sha256")}}
    identity_hash = hashlib.sha256(json.dumps(identity_fields, sort_keys=True, separators=(",", ":"),
                                               allow_nan=False).encode("utf8")).hexdigest()
    receipt = {"status": "completed", "phase": "development", "independent_test": False,
               "Q3_values_read": False, "data_identity": "synthetic_test:" + ADMISSION_HASH,
               "config": cfg, "fit_origins": 10, "stop_origins": 3, "validation_origins": 4,
               "wall_seconds": 2.5, "wall_time_policy": "fictional test timing", "resume_count": 0,
               "models": models, "identity_fields": identity_fields, "fold_identity": identity_hash,
               "source_sha256": sources, "targets_sha256": file_digest(targets_path)}
    path = folder / "fold_receipt.json"; path.write_text(json.dumps(receipt), encoding="utf8")
    state = {"directory": "fictional-fold", "receipt_sha256": file_digest(path)}
    result = report_module._training_record(tmp_path, job, state, PROTOCOL_HASH, ADMISSION_HASH, protocol)
    assert result["models"]["mlp"]["training_status"] == "max_epochs_reached_convergence_unverified"
    assert result["models"]["mlp"]["history"] == history
    report = {"data_kind": "synthetic_test", "tasks": [{**job, "frequency": "minute", "training_folds": [result]}]}
    mlp = next(row for row in report_module.training_rows(report) if row["model"] == "mlp")
    assert mlp["max_epochs"] == 3 and mlp["budget_cap_reached"] is True
    mismatch = {**protocol, "training": {**training, "max_epochs": 4}}
    with pytest.raises(ValueError, match="frozen configuration"):
        report_module._training_record(tmp_path, job, state, PROTOCOL_HASH, ADMISSION_HASH, mismatch)
    with pytest.raises(ValueError, match="origin counts"):
        report_module._training_record(tmp_path, {**job, "fit_origins": 11}, state,
                                       PROTOCOL_HASH, ADMISSION_HASH, protocol)
    np.savez(targets_path, origins=np.array([1, 2, 3, 5], dtype=np.int64))
    receipt["targets_sha256"] = file_digest(targets_path)
    path.write_text(json.dumps(receipt), encoding="utf8")
    state["receipt_sha256"] = file_digest(path)
    with pytest.raises(ValueError, match="validation origins"):
        report_module._training_record(tmp_path, job, state, PROTOCOL_HASH, ADMISSION_HASH, protocol)
    np.savez(targets_path, origins=np.array([1, 2, 3, 4], dtype=np.int64))
    receipt["targets_sha256"] = file_digest(targets_path)
    path.write_text(json.dumps(receipt), encoding="utf8")
    state["receipt_sha256"] = file_digest(path)
    summary_path = folder / "mlp" / "training_summary.json"
    summary_path.write_text(summary_path.read_text(encoding="utf8") + " ", encoding="utf8")
    with pytest.raises(ValueError, match="artifact.*changed"):
        report_module._training_record(tmp_path, job, state, PROTOCOL_HASH, ADMISSION_HASH, protocol)
