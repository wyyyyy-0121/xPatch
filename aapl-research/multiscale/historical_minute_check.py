"""Recompute completed minute-fold metrics from an identity-checked coverage snapshot."""
from collections import defaultdict
from datetime import datetime, timezone
import argparse
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from .historical_coverage import (CONTROLS, NEURAL, OUT, EvidenceError, digest,
                                        in_tree, read_json, require, snapshot)


MODELS = ("last_close", "drift", "seasonal", "ridge", "mlp", "gru", "lstm", "xpatch")
NY = ZoneInfo("America/New_York")
CUTOFF_NS = int(datetime(2026, 7, 1, tzinfo=NY).timestamp()) * 1_000_000_000


def verify_times(decision, completion):
    decision = np.asarray(decision, dtype=np.int64)
    completion = np.asarray(completion, dtype=np.int64)
    require(decision.ndim == completion.ndim == 1 and len(decision) > 0 and
            len(decision) == len(completion) and np.all(np.diff(decision) > 0) and
            np.all(completion > decision) and int(completion.max()) < CUTOFF_NS,
            "Saved timestamps are empty, unordered or cross the protected cutoff")
    return decision, completion


def calculate(actual, anchor, prediction, target):
    actual = np.asarray(actual, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    require(actual.ndim == 2 and actual.shape == prediction.shape and actual.shape[0] > 0 and
            anchor.shape == (actual.shape[0],) and np.isfinite(actual).all() and
            np.isfinite(prediction).all() and np.isfinite(anchor).all() and np.all(anchor > 0),
            "Prediction, target or anchor values are invalid")
    error = prediction - actual
    squared, absolute = np.square(error), np.abs(error)
    if target == "future_close":
        predicted_move, actual_move = prediction[:, -1] - anchor, actual[:, -1] - anchor
    elif target == "future_log_return":
        predicted_move, actual_move = prediction[:, -1], actual[:, -1]
    else:
        raise EvidenceError("Unknown target")
    active = predicted_move != 0
    active_count = int(active.sum())
    correct_count = int((np.sign(predicted_move[active]) == np.sign(actual_move[active])).sum())
    n, h = actual.shape
    endpoint_mse = float(squared[:, -1].mean())
    return {"origins": n, "horizon": h, "endpoint_mse": endpoint_mse,
            "endpoint_rmse": math.sqrt(endpoint_mse), "endpoint_mae": float(absolute[:, -1].mean()),
            "path_mse": float(squared.mean()), "path_mae": float(absolute.mean()),
            "per_step_mse": squared.mean(axis=0).tolist(), "per_step_mae": absolute.mean(axis=0).tolist(),
            "direction_coverage": active_count / n,
            "direction_accuracy_when_active": correct_count / active_count if active_count else None,
            "active_count": active_count, "correct_active_count": correct_count}


def check_metrics(computed, saved):
    for key in ("endpoint_mse", "endpoint_rmse", "endpoint_mae", "path_mse", "path_mae",
                "direction_coverage", "direction_accuracy_when_active"):
        a, b = computed[key], saved[key]
        require((a is None and b is None) or (a is not None and b is not None and
                math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-16)),
                "Saved fold metric differs: " + key)
    for key in ("per_step_mse", "per_step_mae"):
        require(np.allclose(computed[key], saved[key], rtol=1e-10, atol=1e-16),
                "Saved per-step metric differs: " + key)
    require(computed["origins"] == saved["origins"] and computed["horizon"] == saved["horizon"],
            "Saved metric dimensions differ")


def local_time(ns):
    return datetime.fromtimestamp(int(ns) / 1_000_000_000, timezone.utc).astimezone(NY).isoformat()


def fold_sources(row):
    evidence = row["evidence"]
    if "fold_directory" in evidence:
        folder = in_tree(evidence["fold_directory"])
        receipt_path = folder / "fold_receipt.json"
        require(digest(receipt_path) == evidence["receipt_sha256"], "Fold receipt changed")
        receipt = read_json(receipt_path)
        verification_path = in_tree(evidence["verification"])
        require(digest(verification_path) == evidence["verification_sha256"], "Fold verification changed")
        verification = read_json(verification_path)
        return folder, {model: folder for model in MODELS}, receipt["models"], verification["models"], receipt["targets_sha256"]
    require((row["horizon"], row["target"], row["fold"]) == (60, "future_log_return", 1),
            "Unexpected split-file first fold")
    controls, neural = read_json(CONTROLS / "PRIVATE_RESULT.json"), read_json(NEURAL / "PRIVATE_RESULT.json")
    cv, nv = read_json(CONTROLS / "PRIVATE_INDEPENDENT_CHECK.json"), read_json(NEURAL / "PRIVATE_INDEPENDENT_CHECK.json")
    require(digest(CONTROLS / "PRIVATE_INDEPENDENT_CHECK.json") == evidence["controls_check_sha256"] and
            digest(NEURAL / "PRIVATE_INDEPENDENT_CHECK.json") == evidence["neural_check_sha256"] and
            digest(CONTROLS / "PRIVATE_RESULT.json") == cv["result_sha256"] and
            digest(NEURAL / "PRIVATE_RESULT.json") == nv["result_sha256"], "First-fold sources changed")
    models = {**controls["models"], **neural["models"]}
    checks = {**cv["models"], **nv["models"]}
    folders = {model: CONTROLS if model in controls["models"] else NEURAL for model in MODELS}
    return CONTROLS, folders, models, checks, controls["targets_sha256"]


def score_fold(row):
    target_dir, folders, saved_models, checks, target_hash = fold_sources(row)
    require(set(saved_models) == set(checks) == set(MODELS), "A completed fold lacks model evidence")
    target_path = target_dir / "targets.npz"
    with np.load(target_path, allow_pickle=False) as z:
        require(set(z.files) == {"origins", "actual", "anchor", "decision_ns", "target_completion_ns"},
                "Unexpected target archive fields")
        decision, completion = verify_times(z["decision_ns"], z["target_completion_ns"])
        require(digest(target_path) == target_hash, "Target archive identity differs")
        origins = np.asarray(z["origins"], dtype=np.int64)
        actual = np.asarray(z["actual"], dtype=np.float64)
        anchor = np.asarray(z["anchor"], dtype=np.float64)
    require(origins.ndim == 1 and len(origins) == len(decision) == row["evidence"]["validation_origins"] and
            np.all(np.diff(origins) > 0) and actual.shape == (len(origins), row["horizon"]),
            "Saved origin or target dimensions differ")
    result = {}
    for model in MODELS:
        record = saved_models[model]
        path = folders[model] / f"prediction_{model}.npz"
        require(digest(path) == record["prediction_sha256"], "Prediction archive identity differs")
        with np.load(path, allow_pickle=False) as z:
            require(set(z.files) == {"origins", "decision_ns", "prediction"} and
                    np.array_equal(z["origins"], origins) and np.array_equal(z["decision_ns"], decision),
                    "Prediction and target origins differ")
            calculated = calculate(actual, anchor, z["prediction"], row["target"])
        check_metrics(calculated, record["metrics"])
        independently_checked = checks[model].get("metrics", checks[model])
        require(math.isclose(calculated["endpoint_mse"], independently_checked["endpoint_mse"],
                             rel_tol=1e-10, abs_tol=1e-16), "Prior independent endpoint check differs")
        result[model] = calculated
    baseline = result["last_close"]["endpoint_mse"]
    for model in MODELS:
        result[model]["relative_improvement"] = 1 - result[model]["endpoint_mse"] / baseline if baseline else None
    return {"fold": row["fold"], "origins": len(origins), "first_decision_ns": int(decision[0]),
            "last_decision_ns": int(decision[-1]), "last_completion_ns": int(completion.max()),
            "first_decision_ny": local_time(decision[0]), "last_decision_ny": local_time(decision[-1]),
            "last_completion_ny": local_time(completion.max()), "target_sha256": target_hash,
            "models": result}


def pool(folds):
    require(folds and all(folds[i]["last_decision_ns"] < folds[i + 1]["first_decision_ns"]
                          for i in range(len(folds) - 1)), "Fold decisions overlap or are unordered")
    n = sum(f["origins"] for f in folds)
    pooled = {}
    for model in MODELS:
        rows = [(f["origins"], f["models"][model]) for f in folds]
        weighted = lambda name: sum(count * values[name] for count, values in rows) / n
        mse = weighted("endpoint_mse")
        active = sum(values["active_count"] for _, values in rows)
        correct = sum(values["correct_active_count"] for _, values in rows)
        pooled[model] = {"endpoint_mse": mse, "endpoint_rmse": math.sqrt(mse),
                         "endpoint_mae": weighted("endpoint_mae"), "path_mse": weighted("path_mse"),
                         "path_mae": weighted("path_mae"), "direction_coverage": active / n,
                         "direction_accuracy_when_active": correct / active if active else None,
                         "active_count": active, "correct_active_count": correct}
    baseline = pooled["last_close"]["endpoint_mse"]
    for model in MODELS:
        pooled[model]["relative_improvement"] = 1 - pooled[model]["endpoint_mse"] / baseline if baseline else None
    return {"origins": n, "models": pooled}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task", help="Optional HORIZON:TARGET filter for a bounded validation run")
    args = parser.parse_args()
    coverage_path = in_tree(args.coverage)
    coverage = read_json(coverage_path)
    current = snapshot()
    require(coverage.get("status") == current["status"] and
            coverage.get("script_sha256") == current["script_sha256"] and
            coverage.get("queue_state_sha256") == current["queue_state_sha256"] and
            coverage.get("folds") == current["folds"], "Coverage snapshot is stale or altered")
    rows = [r for r in coverage["folds"] if r["profile"] == "minute" and r["status"] == "completed"]
    if args.task:
        horizon, target = args.task.split(":", 1)
        rows = [r for r in rows if r["horizon"] == int(horizon) and r["target"] == target]
        require(rows, "The selected task has no verified completed folds")
    grouped = defaultdict(list)
    for row in rows:
        result = score_fold(row)
        grouped[(row["horizon"], row["target"])].append(result)
        print(json.dumps({"stage": "recomputed_fold", "horizon": row["horizon"],
                          "target": row["target"], "fold": row["fold"]}), flush=True)
    tasks = []
    for (horizon, target), folds in sorted(grouped.items()):
        folds.sort(key=lambda x: x["fold"])
        numbers = [f["fold"] for f in folds]
        require(len(numbers) == len(set(numbers)), "Duplicated completed fold")
        tasks.append({"profile": "minute", "horizon": horizon, "target": target,
                      "completed_folds": numbers, "missing_folds": [n for n in range(1, 6) if n not in numbers],
                      "full_five_fold_task": numbers == list(range(1, 6)), "folds": folds, "pooled": pool(folds)})
    out = in_tree(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"status": "private_minute_metrics_recomputed_from_verified_folds",
               "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "coverage": str(coverage_path), "coverage_sha256": digest(coverage_path),
               "script_sha256": digest(Path(__file__)), "task_filter": args.task,
               "completed_folds_recomputed": len(rows), "tasks": tasks,
               "independent_test": False, "Q3_values_read": False,
               "publication_rights": "outside_arithmetic_check_scope_see_current_permission_record",
               "limitations": ["Previously exposed historical development, not independent confirmation.",
                               "Partial fold sets are reported explicitly and may not be representative.",
                               "No claim of model convergence, publication permission or trading profitability."]}
    with out.open("x", encoding="utf8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print(json.dumps({"status": payload["status"], "path": str(out), "sha256": digest(out),
                      "completed_folds_recomputed": len(rows), "tasks": len(tasks)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
