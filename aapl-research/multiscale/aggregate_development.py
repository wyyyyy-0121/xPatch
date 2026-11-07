"""Independently pool verified development folds without claiming confirmation.

The analysis uses the pre-frozen block settings and source session labels. It
never reports development p-values or treats synthetic tests as market results.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .bars import file_digest
from .evaluation import paired_block_bootstrap
from .matrix_executor import _job_key, _json_inside, load_reviewed
from .matrix_gate import locked_protocol_issues
from .spec import DEV_END, MODELS, OUT, PROFILES, TARGETS
from .verify_development import check_fold


def session_labels(directory: Path, admission: dict, frequency: str):
    """Read only session-date metadata, after verifying the saved bar identity."""
    name = f"bars_{frequency}.npz"
    path = (directory / name).resolve()
    if (not path.is_relative_to(directory.resolve()) or
            file_digest(path) != admission.get("files", {}).get(name)):
        raise ValueError("Candidate session metadata identity changed")
    with np.load(path, allow_pickle=False) as data:
        labels = data["session_date"]
    if (labels.dtype.kind != "U" or labels.ndim != 1 or not len(labels) or
            (labels > DEV_END).any() or (labels[1:] < labels[:-1]).any()):
        raise ValueError("Invalid or post-cutoff session labels")
    return labels


def aggregate(admission_dir: Path, review_path: Path, protocol_path: Path, run_dir: Path,
              profile: str, horizon: int, target: str, *, allow_partial=False):
    admission, admission_hash, source, _, review_hash, _ = load_reviewed(admission_dir, review_path)
    protocol, protocol_hash, _ = _json_inside(protocol_path, OUT)
    issues = locked_protocol_issues(admission, admission_hash, protocol, review_hash)
    if issues:
        raise ValueError("Frozen analysis blocked: " + ", ".join(issues))
    data_kind = protocol["data_kind"]
    permitted_root = (OUT / "synthetic_matrix_validation" / "runs" if data_kind == "synthetic_test"
                      else OUT / "matrix_runs").resolve()
    run_dir = run_dir.resolve()
    if not run_dir.is_relative_to(permitted_root) or run_dir.name != protocol_hash[:24]:
        raise ValueError("Matrix output path does not match the frozen protocol")
    state = json.loads((run_dir / "matrix_state.json").read_text(encoding="utf8"))
    if (state.get("protocol_sha256") != protocol_hash or state.get("review_sha256") != review_hash or
            state.get("data_kind") != data_kind):
        raise ValueError("Matrix state identity differs from the frozen protocol")
    requested = [job for job in protocol["jobs"] if job["profile"] == profile and
                 job["horizon"] == horizon and job["target"] == target]
    if len(requested) != 5 or target not in TARGETS:
        raise ValueError("Expected one of the twenty requested tasks")
    requested.sort(key=lambda job: job["fold"])
    complete, missing = [], []
    for job in requested:
        key = _job_key(job)
        record = state["jobs"].get(key, {})
        if record.get("status") != "completed":
            missing.append(job["fold"])
            continue
        folder = (run_dir / record["directory"]).resolve()
        if not folder.is_relative_to(run_dir):
            raise ValueError("Fold output escapes the matrix run")
        verified = check_fold(folder)
        if (verified["fold_receipt_sha256"] != record.get("receipt_sha256") or
                verified["protocol_identity"] != protocol_hash or
                verified["data_identity"] != data_kind + ":" + admission_hash or
                not all(verified["current_source_matches_recorded"].values())):
            raise ValueError("Saved fold differs from the frozen matrix state")
        complete.append((job, folder, verified))
    if not complete or (missing and not allow_partial):
        raise ValueError("All five folds must be complete unless partial reporting is explicitly requested")
    frequency = requested[0]["frequency"]
    labels = session_labels(source, admission, frequency)
    accum = {model: {"n": 0, "step_squared": np.zeros(horizon), "step_absolute": np.zeros(horizon),
                     "direction_active": 0, "direction_correct": 0, "endpoint_differences": [],
                     "fold_endpoint_mse": [], "nonoverlap_squared": 0.0,
                     "nonoverlap_absolute": 0.0} for model in MODELS}
    groups, prior_origin, nonoverlap_count = [], None, 0
    prior_decision = None
    for job, folder, verified in complete:
        with np.load(folder / "targets.npz", allow_pickle=False) as saved:
            actual, anchor, origins, decisions = (saved[key].copy() for key in ("actual", "anchor", "origins", "decision_ns"))
        if (origins[-1] >= len(labels) or (prior_decision is not None and decisions[0] <= prior_decision)):
            raise ValueError("Saved folds overlap or contain out-of-range origins")
        prior_decision = int(decisions[-1])
        groups.append(labels[origins])
        nonoverlap = []
        if profile == "annual":
            for position, origin in enumerate(origins):
                if prior_origin is None or int(origin) >= prior_origin + horizon:
                    nonoverlap.append(position)
                    prior_origin = int(origin)
            nonoverlap_count += len(nonoverlap)
        with np.load(folder / "prediction_last_close.npz", allow_pickle=False) as saved:
            baseline = np.asarray(saved["prediction"], dtype=np.float64)
        baseline_endpoint_loss = np.square(baseline[:, -1] - actual[:, -1])
        for model in MODELS:
            with np.load(folder / f"prediction_{model}.npz", allow_pickle=False) as saved:
                prediction = np.asarray(saved["prediction"], dtype=np.float64)
            error = prediction - actual
            squared, absolute = np.square(error), np.abs(error)
            data = accum[model]
            data["n"] += len(actual)
            data["step_squared"] += squared.sum(axis=0)
            data["step_absolute"] += absolute.sum(axis=0)
            actual_move = actual[:, -1] - anchor if target == "future_close" else actual[:, -1]
            predicted_move = prediction[:, -1] - anchor if target == "future_close" else prediction[:, -1]
            active = predicted_move != 0
            data["direction_active"] += int(active.sum())
            data["direction_correct"] += int((np.sign(predicted_move[active]) == np.sign(actual_move[active])).sum())
            data["endpoint_differences"].append(squared[:, -1] - baseline_endpoint_loss)
            data["fold_endpoint_mse"].append({"fold": job["fold"],
                                               "endpoint_mse": verified["models"][model]["metrics"]["endpoint_mse"],
                                               "origins": len(actual)})
            if profile == "annual":
                data["nonoverlap_squared"] += float(squared[nonoverlap, -1].sum())
                data["nonoverlap_absolute"] += float(absolute[nonoverlap, -1].sum())
    group = np.concatenate(groups)
    full_grid = np.unique(labels)
    full_grid = full_grid[(full_grid >= group[0]) & (full_grid <= group[-1])]
    analysis = protocol["analysis"]
    base_mse = float(accum["last_close"]["step_squared"][-1] / accum["last_close"]["n"])
    baseline_nonoverlap = (accum["last_close"]["nonoverlap_squared"] / nonoverlap_count
                           if nonoverlap_count else None)
    models = {}
    for model, data in accum.items():
        n = data["n"]
        per_step_mse, per_step_mae = data["step_squared"] / n, data["step_absolute"] / n
        endpoint_mse = float(per_step_mse[-1])
        summary = {"origins": n, "endpoint_mse": endpoint_mse, "endpoint_rmse": float(np.sqrt(endpoint_mse)),
                   "endpoint_mae": float(per_step_mae[-1]), "path_mse": float(per_step_mse.mean()),
                   "path_mae": float(per_step_mae.mean()), "per_step_mse": per_step_mse.tolist(),
                   "per_step_mae": per_step_mae.tolist(), "direction_coverage": data["direction_active"] / n,
                   "direction_accuracy_when_active": (data["direction_correct"] / data["direction_active"]
                                                      if data["direction_active"] else None),
                   "relative_improvement_vs_persistence": 1 - endpoint_mse / base_mse if base_mse > 0 else None,
                   "per_fold_endpoint_mse": data["fold_endpoint_mse"], "descriptive_block_intervals": []}
        difference = np.concatenate(data["endpoint_differences"])
        for length in analysis["block_lengths_by_profile"][profile]:
            if length > len(full_grid):
                summary["descriptive_block_intervals"].append({"block_length": length, "status": "insufficient_groups"})
                continue
            interval = paired_block_bootstrap(difference, group, block_length=length,
                                              resamples=analysis["resamples"], seed=analysis["seed"],
                                              confidence=analysis["confidence"], group_order=full_grid)
            interval.pop("centered_one_sided_p_improvement")
            interval["status"] = "descriptive_development_only"
            interval["interpretation"] = "Negative paired loss difference favors the model; this is not independent confirmation."
            summary["descriptive_block_intervals"].append(interval)
        if profile == "annual" and nonoverlap_count:
            mse = data["nonoverlap_squared"] / nonoverlap_count
            summary["annual_nonoverlapping_origins"] = {
                "origins": nonoverlap_count, "endpoint_mse": mse,
                "endpoint_mae": data["nonoverlap_absolute"] / nonoverlap_count,
                "relative_improvement_vs_persistence": 1 - mse / baseline_nonoverlap if baseline_nonoverlap else None,
                "selection": "earliest eligible origin, then at least H completed monthly bars apart"}
        models[model] = summary
    return {"status": "synthetic_test_results_only" if data_kind == "synthetic_test" else "historical_development_only",
            "data_kind": data_kind, "independent_test": False, "development_p_values_reported": False,
            "task": {"profile": profile, "frequency": frequency, "horizon": horizon, "target": target},
            "protocol_sha256": protocol_hash, "admission_sha256": admission_hash,
            "completed_folds": [job["fold"] for job, _, _ in complete], "missing_folds": missing,
            "partial": bool(missing), "group_grid": "all candidate source sessions between first and last evaluated origin",
            "analysis": analysis, "models": models,
            "limitations": ["This aggregates historical development folds, not independent confirmation.",
                            "Block intervals are descriptive and do not establish model superiority or trading profitability.",
                            "A complete result requires all five frozen folds and admitted data rights."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("admission", "review", "protocol", "run"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--profile", choices=[p.name for p in PROFILES], required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = aggregate(args.admission, args.review, args.protocol, args.run,
                       args.profile, args.horizon, args.target, allow_partial=args.allow_partial)
    root = (OUT / "synthetic_matrix_validation" / "aggregates" if result["data_kind"] == "synthetic_test"
            else OUT / "development_aggregates")
    root.mkdir(parents=True, exist_ok=True)
    path = root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf8")
    print(json.dumps({"status": result["status"], "path": str(path), "partial": result["partial"],
                      "completed_folds": result["completed_folds"]}))


if __name__ == "__main__":
    main()
