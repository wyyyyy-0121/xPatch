"""Independently verify fictional split groups; never open market observations."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np


out = Path("outputs/multiscale").resolve()
root = out / "synthetic_matrix_validation"
source = out / "admission/synthetic_test_20261107T105553738329Z"
protocol = root / "locked_protocols/20261107T105556523892Z.json"
partial_protocol = root / "locked_protocols/20261107T105606538004Z.json"
run = root / "runs/be0325dcdd9b1b323a2291e2"
report_dir = root / "reports/20261107T105755281409Z"
partial_dir = root / "reports/20261107T105755136838Z"
models = ("last_close", "drift", "seasonal", "ridge", "mlp", "gru", "lstm", "xpatch")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal_number(actual, expected, label):
    if expected is None:
        require(actual is None, f"{label} should be empty")
    else:
        require(actual is not None and math.isclose(float(actual), float(expected),
                                                     rel_tol=1e-10, abs_tol=1e-16),
                f"{label} does not match independently calculated values")


def read_report(directory, expected_protocol):
    manifest = json.loads((directory / "SYNTHETIC_ONLY_manifest.json").read_text())
    report = json.loads((directory / "SYNTHETIC_ONLY_report.json").read_text())
    require(manifest.get("data_kind") == report.get("data_kind") == "synthetic_test", "Not synthetic")
    require(manifest["protocol_sha256"] == report["protocol_sha256"] == sha(expected_protocol),
            "Protocol identity differs")
    require(report["independent_test"] is False and report["complete_twenty_task_matrix"] is False,
            "Synthetic scope is mislabeled")
    for filename, digest in manifest["files_sha256"].items():
        require(Path(filename).name == filename and sha(directory / filename) == digest,
                "Report output hash mismatch")
    with (directory / "SYNTHETIC_ONLY_model_rows.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (directory / "SYNTHETIC_ONLY_training_rows.csv").open(newline="") as stream:
        training = list(csv.DictReader(stream))
    require(len(rows) == 160, "Expected 160 model slots")
    return report, rows, training


require(source.name.startswith("synthetic_test_") and (source / "SYNTHETIC_ONLY.txt").is_file(),
        "Synthetic admission marker is missing")
admission = json.loads((source / "admission.json").read_text())
proto = json.loads(protocol.read_text())
state = json.loads((run / "matrix_state.json").read_text())
require(all(x.get("data_kind") == "synthetic_test" for x in (admission, proto, state)), "Not synthetic")
require(admission["Q3_values_read"] is False and admission["experiment_scored"] is False,
        "Unexpected synthetic admission identity")
require(admission["split_actions"] == [{"date": "2015-01-01", "ratio": 2.0}], "Unexpected fictional event")
require(sha(source / "bars_month.npz") == admission["files"]["bars_month.npz"], "Synthetic bar hash mismatch")
require(state["protocol_sha256"] == sha(protocol), "Matrix protocol identity differs")
events = [item["date"] for item in admission["split_actions"]]
with np.load(source / "bars_month.npz", allow_pickle=False) as data:
    session_dates = data["session_date"].copy()

report, rows, training = read_report(report_dir, protocol)
require(report["task_status_counts"] == {"complete": 2, "not_started": 18}, "Unexpected task coverage")
require(len(training) == 80, "Unexpected training row count")
require(not any(row["endpoint_mse"] for row in rows if row["task_status"] == "not_started"),
        "An unstarted task has a score")
require(all(row["max_epochs"] == "1" and row["budget_cap_reached"] == "True" for row in training
            if row["model"] in ("mlp", "gru", "lstm", "xpatch")), "Synthetic training cap is missing")

counts = {}
for target in ("future_close", "future_log_return"):
    task = next(t for t in report["tasks"] if t["profile"] == "annual" and
                t["horizon"] == 12 and t["target"] == target)
    require(task["status"] == "complete" and task["completed_folds"] == [1, 2, 3, 4, 5] and
            not task["missing_folds"], "Five-fold task coverage differs")
    jobs = sorted((j for j in proto["jobs"] if j["profile"] == "annual" and
                   j["horizon"] == 12 and j["target"] == target), key=lambda j: j["fold"])
    require([job["fold"] for job in jobs] == [1, 2, 3, 4, 5], "Protocol folds differ")
    actual_parts, origin_parts = [], []
    prediction_parts = {model: [] for model in models}
    for job in jobs:
        key = f"annual/h12/{target}/fold{job['fold']:02d}"
        record = state["jobs"][key]
        folder = (run / record["directory"]).resolve()
        require(folder.is_relative_to(run.resolve()) and record["status"] == "completed", "Invalid fold path")
        receipt_path = folder / "fold_receipt.json"
        require(sha(receipt_path) == record["receipt_sha256"], "Fold receipt hash mismatch")
        receipt = json.loads(receipt_path.read_text())
        require(sha(folder / "targets.npz") == receipt["targets_sha256"], "Synthetic target hash mismatch")
        with np.load(folder / "targets.npz", allow_pickle=False) as data:
            origin, actual = data["origins"].copy(), data["actual"].copy()
            decisions = data["decision_ns"].copy()
        require(np.all(origin + 12 < len(session_dates)), "Synthetic target index out of range")
        origin_parts.append(origin)
        actual_parts.append(actual)
        for model in models:
            prediction_path = folder / f"prediction_{model}.npz"
            require(sha(prediction_path) == receipt["models"][model]["prediction_sha256"],
                    "Synthetic prediction hash mismatch")
            with np.load(prediction_path, allow_pickle=False) as data:
                require(np.array_equal(data["origins"], origin) and
                        np.array_equal(data["decision_ns"], decisions), "Prediction origins differ")
                prediction_parts[model].append(data["prediction"].copy())
    origin = np.concatenate(origin_parts)
    actual = np.concatenate(actual_parts)
    origin_dates, target_dates = session_dates[origin], session_dates[origin + 12]
    # Use direct calendar comparisons rather than the production grouping function.
    mask = np.array([any(str(start) < event <= str(end) for event in events)
                     for start, end in zip(origin_dates, target_dates, strict=True)])
    covered = {event for event in events if any(str(start) < event <= str(end)
                                                 for start, end in zip(origin_dates, target_dates, strict=True))}
    summary = task["summary"]
    require(summary["split_events_in_evaluated_paths"] == len(covered) == 1, "Event count differs")
    require(mask.any() and (~mask).any(), "Both synthetic groups must be exercised")
    errors = {model: np.concatenate(prediction_parts[model]) - actual for model in models}
    require(set(summary["models"]) == set(models), "Model coverage differs")
    counts[target] = {"crosses_split": int(mask.sum()), "no_split": int((~mask).sum()),
                      "distinct_events": len(covered)}
    for model, error in errors.items():
        metrics = summary["models"][model]
        row = next(r for r in rows if r["profile"] == "annual" and r["horizon"] == "12" and
                   r["target"] == target and r["model"] == model)
        unit = "USD^2/share^2" if target == "future_close" else "log_return^2"
        absolute_unit = "USD/share" if target == "future_close" else "log_return"
        require(row["squared_error_unit"] == unit and row["absolute_error_unit"] == absolute_unit and
                row["completed_folds"] == "1,2,3,4,5" and not row["missing_folds"], "Units or fold labels differ")
        equal_number(metrics["endpoint_mse"], np.mean(error[:, -1] ** 2), "pooled endpoint MSE")
        equal_number(metrics["endpoint_mae"], np.mean(np.abs(error[:, -1])), "pooled endpoint MAE")
        equal_number(metrics["path_mse"], np.mean(error ** 2), "pooled path MSE")
        for group, label, selected in (("crosses_split", "cross_split", mask), ("no_split", "no_split", ~mask)):
            values = metrics["split_event_sensitivity"][group]
            group_error = error[selected]
            baseline_mse = float(np.mean(errors["last_close"][selected, -1] ** 2))
            expected = {"endpoint_mse": float(np.mean(group_error[:, -1] ** 2)),
                        "endpoint_mae": float(np.mean(np.abs(group_error[:, -1]))),
                        "path_mse": float(np.mean(group_error ** 2))}
            require(values["status"] == "descriptive_development_only" and
                    values["origins"] == int(selected.sum()) == int(row[label + "_origins"]),
                    "Synthetic subgroup count differs")
            for metric, value in expected.items():
                equal_number(values[metric], value, f"{group} {metric}")
                equal_number(row[label + "_" + metric], value, f"CSV {group} {metric}")
            improvement = 1 - expected["endpoint_mse"] / baseline_mse if baseline_mse > 0 else None
            equal_number(values["relative_improvement_vs_persistence"], improvement, f"{group} improvement")
            equal_number(row[label + "_relative_improvement"] or None, improvement, f"CSV {group} improvement")

partial, partial_rows, partial_training = read_report(partial_dir, partial_protocol)
require(partial["task_status_counts"] == {"omitted_before_scoring": 6, "partial_descriptive": 1,
                                          "not_started": 13} and len(partial_training) == 8,
        "Partial-scale coverage differs")
partial_task = next(t for t in partial["tasks"] if t["status"] == "partial_descriptive")
require(partial_task["summary"]["split_events_in_evaluated_paths"] == 0, "Empty event group differs")
for model in models:
    groups = partial_task["summary"]["models"][model]["split_event_sensitivity"]
    empty = groups["crosses_split"]
    require(empty["status"] == "no_origins" and empty["origins"] == 0 and
            all(empty[name] is None for name in ("endpoint_mse", "endpoint_mae", "path_mse",
                                                  "relative_improvement_vs_persistence")) and
            groups["no_split"]["origins"] > 0, "Empty synthetic subgroup is not empty")
    row = next(r for r in partial_rows if r["profile"] == partial_task["profile"] and
               r["horizon"] == str(partial_task["horizon"]) and r["target"] == partial_task["target"] and
               r["model"] == model)
    require(row["cross_split_origins"] == "0" and
            not any(row["cross_split_" + key] for key in ("endpoint_mse", "endpoint_mae", "path_mse",
                                                            "relative_improvement")),
            "Empty synthetic CSV subgroup is not empty")

receipt = {"status": "passed_synthetic_test_only", "created_at_utc": datetime.now(timezone.utc).isoformat(),
           "market_data_read": False, "formal_experiment": False, "five_fold_tasks_verified": 2,
           "models_per_task": len(models), "synthetic_group_counts": counts, "empty_group_verified": True,
           "checks": ["independent calendar grouping and distinct event count", "subgroup and pooled errors",
                      "within-group baseline improvements", "CSV values and price/return units",
                      "empty subgroup values", "fold, target, prediction and report hashes"],
           "protocol_sha256": sha(protocol), "matrix_state_sha256": sha(run / "matrix_state.json"),
           "report_manifest_sha256": sha(report_dir / "SYNTHETIC_ONLY_manifest.json"),
           "partial_manifest_sha256": sha(partial_dir / "SYNTHETIC_ONLY_manifest.json"),
           "verification_source_sha256": sha(Path(__file__)),
           "limitations": ["The event and smooth price series are fictional; the series contains no actual split jump.",
                           "This does not establish market data rights, price quality, or predictive performance.",
                           "Overlapping origins are not independent events; bootstrap intervals were not rechecked here."]}
destination = Path(__file__).with_name("SYNTHETIC_ONLY_receipt.json")
with destination.open("x", encoding="utf8") as stream:
    json.dump(receipt, stream, indent=2, allow_nan=False)
    stream.write("\n")
print(json.dumps({"status": receipt["status"], "receipt": str(destination),
                  "synthetic_group_counts": counts, "empty_group_verified": True}))
