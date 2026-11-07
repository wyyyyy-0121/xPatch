"""Independently recalculate saved development-fold metrics and verify artifacts.

This checks one completed fold's arrays, timing metadata, reported statistics and
model files. It cannot establish source rights, point-in-time data quality,
protocol admission, forecast novelty or independent statistical confirmation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .bars import file_digest
from .spec import DEV_END, MODELS, NY, OUT, ROOT


def require_digest(path: Path, expected: str):
    if not isinstance(expected, str) or len(expected) != 64 or not path.is_file():
        raise ValueError(f"Missing artifact or identity: {path.name}")
    if file_digest(path) != expected:
        raise ValueError(f"Artifact identity mismatch: {path.name}")


def assert_match(observed, expected, label: str):
    """Compare reported JSON metrics with independently calculated values."""
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(observed) != set(expected):
            raise ValueError(f"Reported metric keys differ: {label}")
        for key in expected:
            assert_match(observed[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise ValueError(f"Reported metric length differs: {label}")
        for index, (actual_item, expected_item) in enumerate(zip(observed, expected, strict=True)):
            assert_match(actual_item, expected_item, f"{label}[{index}]")
    elif expected is None:
        if observed is not None:
            raise ValueError(f"Reported metric differs: {label}")
    elif isinstance(expected, (float, int)):
        tolerance = 1e-14 if label.endswith("relative_improvement") else 1e-18
        if isinstance(observed, bool) or not isinstance(observed, (float, int)) or not np.isclose(
                observed, expected, rtol=1e-12, atol=tolerance):
            raise ValueError(f"Reported metric differs: {label}")
    elif observed != expected:
        raise ValueError(f"Reported value differs: {label}")


def independent_metrics(actual, prediction, target, anchor):
    """Recalculate, without using the scoring functions from evaluation.py."""
    error = prediction - actual
    with np.errstate(over="raise", invalid="raise"):
        squared = error * error
        endpoint = squared[:, -1]
        step_mse = np.mean(squared, axis=0)
        step_mae = np.mean(np.abs(error), axis=0)
    if target == "future_close":
        move_actual, move_predicted = actual[:, -1] - anchor, prediction[:, -1] - anchor
    elif target == "future_log_return":
        move_actual, move_predicted = actual[:, -1], prediction[:, -1]
    else:
        raise ValueError("Unknown target in fold receipt")
    active = move_predicted != 0
    metrics = {"origins": len(actual), "horizon": actual.shape[1],
               "endpoint_mse": float(step_mse[-1]), "endpoint_rmse": float(np.sqrt(step_mse[-1])),
               "endpoint_mae": float(step_mae[-1]), "path_mse": float(np.mean(squared)),
               "path_mae": float(np.mean(np.abs(error))), "per_step_mse": step_mse.tolist(),
               "per_step_mae": step_mae.tolist(), "direction_coverage": float(np.mean(active)),
               "direction_accuracy_when_active": float(np.mean(np.sign(move_predicted[active]) ==
                                                               np.sign(move_actual[active]))) if active.any() else None}
    return metrics, float(np.mean(endpoint))


def check_fold(directory: Path):
    directory = Path(directory).resolve()
    receipt_path = directory / "fold_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf8"))
    if (receipt.get("schema_version") != 2 or receipt.get("status") != "completed"
            or receipt.get("phase") != "development" or receipt.get("independent_test") is not False
            or receipt.get("Q3_values_read") is not False):
        raise ValueError("Expected a completed schema-2 development fold")
    cfg = receipt["config"]
    if not receipt.get("data_identity") or not cfg.get("protocol_identity"):
        raise ValueError("Source or protocol identity is missing")
    if set(receipt["models"]) != set(MODELS):
        raise ValueError("Fold does not contain the eight required models")
    target_file = directory / "targets.npz"
    require_digest(target_file, receipt["targets_sha256"])
    with np.load(target_file, allow_pickle=False) as stored:
        if set(stored.files) != {"origins", "actual", "anchor", "decision_ns", "target_completion_ns"}:
            raise ValueError("Unexpected target artifact fields")
        origin = stored["origins"]
        actual = np.asarray(stored["actual"], dtype=np.float64)
        anchor = np.asarray(stored["anchor"], dtype=np.float64)
        decisions, completions = stored["decision_ns"], stored["target_completion_ns"]
    count = receipt["validation_origins"]
    horizon = cfg["horizon"]
    vectors = (origin, anchor, decisions, completions)
    if not isinstance(count, int) or count < 1 or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("Invalid fold dimensions")
    if any(value.shape != (count,) for value in vectors) or actual.shape != (count, horizon):
        raise ValueError("Target artifact dimensions do not match the receipt")
    if any(not np.issubdtype(value.dtype, np.integer) for value in (origin, decisions, completions)):
        raise ValueError("Origins and timestamps must be integer arrays")
    if ((origin < 0).any() or (np.diff(origin) <= 0).any() or (np.diff(decisions) <= 0).any()
            or (completions <= decisions).any()):
        raise ValueError("Invalid target or decision chronology")
    cutoff = pd.Timestamp("2026-07-01", tz=NY).tz_convert("UTC").as_unit("ns").value
    if (completions >= cutoff).any():
        raise ValueError("Saved target completion crosses the protected cutoff")
    if cfg["profile"] == "minute":
        origin_dates = pd.to_datetime(decisions, utc=True).tz_convert(NY).date
        end_dates = pd.to_datetime(completions, utc=True).tz_convert(NY).date
        if not np.array_equal(origin_dates, end_dates):
            raise ValueError("A minute target crosses a calendar session")
    if not np.isfinite(actual).all() or not np.isfinite(anchor).all() or (anchor <= 0).any():
        raise ValueError("Invalid actual values or origin prices")
    if cfg["target"] == "future_close" and (actual <= 0).any():
        raise ValueError("Actual close prices must be positive")
    predictions = {}
    for name in MODELS:
        details = receipt["models"][name]
        path = directory / f"prediction_{name}.npz"
        require_digest(path, details["prediction_sha256"])
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {"prediction", "origins", "decision_ns"}:
                raise ValueError(f"Unexpected prediction fields for {name}")
            values = np.asarray(data["prediction"], dtype=np.float64)
            if not np.array_equal(data["origins"], origin) or not np.array_equal(data["decision_ns"], decisions):
                raise ValueError(f"Prediction origins or decision times differ for {name}")
        if values.shape != actual.shape or not np.isfinite(values).all():
            raise ValueError(f"Invalid prediction array for {name}")
        predictions[name] = values
        if name == "ridge":
            require_digest(directory / "ridge_coefficients.npz", details["coefficients_sha256"])
            require_digest(directory / "ridge_scaler.json", details["scaler_sha256"])
        elif name in ("mlp", "gru", "lstm", "xpatch"):
            for filename, key in (("best.pt", "best_checkpoint_sha256"),
                                  ("latest.pt", "latest_checkpoint_sha256"),
                                  ("training_summary.json", "training_summary_sha256")):
                require_digest(directory / name / filename, details[key])
    baseline = predictions["last_close"]
    _, baseline_loss = independent_metrics(actual, baseline, cfg["target"], anchor)
    calculated = {}
    for name in MODELS:
        metrics, loss = independent_metrics(actual, predictions[name], cfg["target"], anchor)
        against = {"model_mse": loss, "baseline_mse": baseline_loss,
                   "relative_improvement": (1.0 - loss / baseline_loss) if baseline_loss > 0 else None}
        assert_match(receipt["models"][name]["metrics"], metrics, f"{name}.metrics")
        assert_match(receipt["models"][name]["against_last_close_or_zero"], against, f"{name}.baseline")
        calculated[name] = {"metrics": metrics, "against_last_close_or_zero": against}
    current_code_matches = {}
    for name, digest in receipt.get("source_sha256", {}).items():
        source = (ROOT / name).resolve()
        if not source.is_relative_to(ROOT.resolve()):
            raise ValueError("Recorded source path is outside the project")
        current_code_matches[name] = source.is_file() and file_digest(source) == digest
    if not current_code_matches:
        raise ValueError("Fold has no recorded source identities")
    return {"status": "verified_saved_fold", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "fold_directory": str(directory), "fold_receipt_sha256": file_digest(receipt_path),
            "targets_sha256": receipt["targets_sha256"], "data_identity": receipt["data_identity"],
            "protocol_identity": cfg["protocol_identity"], "development_cutoff": DEV_END,
            "phase": "development", "independent_test": False, "models": calculated,
            "verification_code_sha256": file_digest(Path(__file__)),
            "verification_runtime": {"python": sys.version.split()[0], "numpy": np.__version__,
                                     "pandas": pd.__version__},
            "current_source_matches_recorded": current_code_matches,
            "limitations": ["Checks saved arrays, timings, summaries and artifact digests only.",
                            "Does not verify source rights, historical point-in-time data or actual protocol admission.",
                            "A development fold does not constitute independent confirmation or predictive success."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fold", type=Path, help="Completed development fold inside outputs/multiscale")
    args = parser.parse_args()
    fold = args.fold.resolve()
    if not fold.is_relative_to(OUT.resolve()):
        raise ValueError("Fold must be inside the isolated multiscale output directory")
    result = check_fold(fold)
    out = OUT / "verification"
    out.mkdir(parents=True, exist_ok=True)
    path = out / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    with path.open("x", encoding="utf8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"path": str(path), "status": result["status"], "models": len(result["models"]),
                      "current_source_matches": all(result["current_source_matches_recorded"].values())}))


if __name__ == "__main__":
    main()
