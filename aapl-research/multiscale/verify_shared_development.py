"""Recalculate a matched shared-horizon fold from saved development artifacts.

This checks artifact identities and metrics; it cannot establish data rights,
historical price quality, a reviewed resource budget, or independent evidence.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .bars import file_digest
from .shared_development import SHARED_SOURCE_FILES
from .spec import NY, ROOT
from .training import digest_indices
from .verify_development import assert_match, independent_metrics, require_digest


HORIZONS = (5, 15, 30, 60)
MODEL_NAMES = tuple(f"independent_h{h}" for h in HORIZONS) + ("shared_h60",)
SPLIT_KEYS = ("train_indices_sha256", "fit_indices_sha256", "stop_indices_sha256", "validation_indices_sha256")


def _inside(directory: Path, relative: str):
    path = directory / relative
    if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Shared-fold artifact escapes its directory")
    return path


def check_shared_fold(directory: Path, *, expected_data_identity=None, expected_protocol_identity=None,
                      expected_split_hashes=None):
    directory = Path(directory).resolve()
    receipt_path = _inside(directory, "fold_receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf8"))
    cfg = receipt.get("config", {})
    if (not isinstance(cfg, dict) or receipt.get("schema_version") != 1 or receipt.get("status") != "completed" or
            receipt.get("phase") != "development" or receipt.get("independent_test") is not False or
            receipt.get("Q3_values_read") is not False or receipt.get("purge_horizon_for_every_model") != 60 or
            cfg.get("horizons") != list(HORIZONS) or not isinstance(cfg.get("protocol_identity"), str) or
            type(cfg.get("max_epochs")) is not int or cfg["max_epochs"] < 1 or
            type(receipt.get("fit_origins")) is not int or receipt["fit_origins"] < 8 or
            type(receipt.get("stop_origins")) is not int or receipt["stop_origins"] < 1 or
            not isinstance(receipt.get("data_identity"), str) or not receipt["data_identity"] or
            (receipt["data_identity"].startswith("market:") and
             (receipt.get("data_kind") != "market" or receipt.get("scope") != "historical_development_only")) or
            (receipt["data_identity"].startswith(("synthetic-", "synthetic_test:")) and
             (receipt.get("data_kind") != "synthetic_test" or receipt.get("scope") != "synthetic_test_only")) or
            not receipt["data_identity"].startswith(("market:", "synthetic-", "synthetic_test:")) or
            set(receipt.get("models", {})) != set(MODEL_NAMES) or
            set(receipt.get("comparisons", {})) != {str(h) for h in HORIZONS}):
        raise ValueError("Expected a completed matched development fold")
    if ((expected_data_identity is not None and receipt["data_identity"] != expected_data_identity) or
            (expected_protocol_identity is not None and cfg["protocol_identity"] != expected_protocol_identity) or
            (expected_split_hashes is not None and any(receipt.get(key) != expected_split_hashes.get(key)
                                                        for key in SPLIT_KEYS))):
        raise ValueError("Shared-fold data, protocol, or split identity differs")
    for key in SPLIT_KEYS:
        value = receipt.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("Shared-fold split identity is missing")
    sources = receipt.get("source_sha256")
    if not isinstance(sources, dict) or set(sources) != set(SHARED_SOURCE_FILES):
        raise ValueError("Shared-fold source identities are incomplete")
    try:
        source_matches = {name: file_digest(ROOT / name) == digest for name, digest in sources.items()}
    except (OSError, TypeError) as exc:
        raise ValueError("Shared-fold source identity cannot be checked") from exc
    if not all(source_matches.values()):
        raise ValueError("Shared-fold source code no longer matches")

    target_path = _inside(directory, "targets.npz")
    require_digest(target_path, receipt.get("targets_sha256"))
    with np.load(target_path, allow_pickle=False) as stored:
        if set(stored.files) != {"origins", "actual", "decision_ns", "target_completion_ns"}:
            raise ValueError("Unexpected shared-fold target fields")
        origins, decisions, completions = (stored[name].copy() for name in ("origins", "decision_ns", "target_completion_ns"))
        count = receipt.get("validation_origins")
        if (type(count) is not int or count < 1 or any(x.shape != (count,) for x in (origins, decisions, completions)) or
                any(not np.issubdtype(x.dtype, np.integer) for x in (origins, decisions, completions)) or
                (origins < 0).any() or (np.diff(origins) <= 0).any() or (np.diff(decisions) <= 0).any() or
                (completions <= decisions).any() or digest_indices(origins) != receipt["validation_indices_sha256"]):
            raise ValueError("Invalid shared-fold origin or time identity")
        cutoff = pd.Timestamp("2026-07-01", tz=NY).tz_convert("UTC").as_unit("ns").value
        if (completions >= cutoff).any():
            raise ValueError("Shared-fold targets cross the protected cutoff")
        if not np.array_equal(pd.to_datetime(decisions, utc=True).tz_convert(NY).date,
                              pd.to_datetime(completions, utc=True).tz_convert(NY).date):
            raise ValueError("A shared-fold target crosses a session")
        actual = np.asarray(stored["actual"], dtype=np.float64)
    if actual.shape != (count, 60) or not np.isfinite(actual).all():
        raise ValueError("Invalid shared-fold targets")

    predictions = {}
    for name in MODEL_NAMES:
        details = receipt["models"][name]
        horizon = 60 if name == "shared_h60" else int(name.removeprefix("independent_h"))
        if (not isinstance(details, dict) or details.get("horizon") != horizon or
                details.get("shared") is not (name == "shared_h60") or
                type(details.get("parameters")) is not int or details["parameters"] < 1 or
                type(details.get("epochs")) is not int or not 1 <= details["epochs"] <= cfg.get("max_epochs", 0) or
                type(details.get("best_epoch")) is not int or not 1 <= details["best_epoch"] <= details["epochs"] or
                type(details.get("steps")) is not int or details["steps"] < 1 or
                type(details.get("stopped_early")) is not bool or
                not isinstance(details.get("training_identity"), str) or len(details["training_identity"]) != 64 or
                any(char not in "0123456789abcdef" for char in details["training_identity"]) or
                isinstance(details.get("training_and_inference_seconds"), bool) or
                not isinstance(details.get("training_and_inference_seconds"), (int, float)) or
                not math.isfinite(details["training_and_inference_seconds"]) or details["training_and_inference_seconds"] < 0):
            raise ValueError("Invalid shared-fold training metadata")
        prediction_path = _inside(directory, f"prediction_{name}.npz")
        require_digest(prediction_path, details.get("prediction_sha256"))
        with np.load(prediction_path, allow_pickle=False) as stored:
            if (set(stored.files) != {"prediction", "origins", "decision_ns"} or
                    not np.array_equal(stored["origins"], origins) or not np.array_equal(stored["decision_ns"], decisions)):
                raise ValueError("Shared-fold predictions do not use the same origins")
            prediction = np.asarray(stored["prediction"], dtype=np.float64)
        if prediction.shape != (count, horizon) or not np.isfinite(prediction).all():
            raise ValueError("Invalid shared-fold prediction shape")
        predictions[name] = prediction
        for filename, key in (("best.pt", "best_checkpoint_sha256"), ("latest.pt", "latest_checkpoint_sha256"),
                              ("training_summary.json", "training_summary_sha256")):
            require_digest(_inside(directory, f"{name}/{filename}"), details.get(key))
        training = json.loads(_inside(directory, f"{name}/training_summary.json").read_text(encoding="utf8"))
        if (training.get("status") != "completed" or training.get("identity") != details["training_identity"] or
                any(training.get(key) != details.get(key) for key in ("epochs", "best_epoch", "steps", "stopped_early"))):
            raise ValueError("Shared-fold training summary differs from its receipt")

    for horizon in HORIZONS:
        truth = actual[:, :horizon]
        shared = predictions["shared_h60"][:, :horizon]
        independent = predictions[f"independent_h{horizon}"]
        shared_metrics, shared_loss = independent_metrics(truth, shared, "future_log_return", None)
        individual_metrics, individual_loss = independent_metrics(truth, independent, "future_log_return", None)
        comparison = receipt["comparisons"][str(horizon)]
        assert_match(comparison.get("shared"), shared_metrics, f"h{horizon}.shared")
        assert_match(comparison.get("independent"), individual_metrics, f"h{horizon}.independent")
        expected = {"model_mse": shared_loss, "baseline_mse": individual_loss,
                    "relative_improvement": 1.0 - shared_loss / individual_loss if individual_loss > 0 else None}
        assert_match(comparison.get("shared_vs_independent"), expected, f"h{horizon}.relative_improvement")
    return {"status": "verified_saved_shared_fold", "fold_receipt_sha256": file_digest(receipt_path),
            "data_identity": receipt["data_identity"], "protocol_identity": cfg["protocol_identity"],
            "validation_indices_sha256": receipt["validation_indices_sha256"], "models_verified": len(MODEL_NAMES),
            "source_matches": source_matches,
            "limitations": ["Checks saved artifacts and metrics, not data rights or budget authorization.",
                            "This development fold is not independent confirmation."]}
