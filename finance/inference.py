"""Freeze an explicitly unqualified reference baseline and emit honest RV outputs."""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import load_data, prepare_data
from .calibration import Calibration
from .data import RVData
from .features import FeatureStore, FitState
from .io import ROOT, read_json, sha, write_json
from .jobs import receipt
from .models import load_model, raw_predictions
from .scheduler import Scheduler
from .splits import partition


def freeze_baseline(config, run_directory):
    from .provenance import verify_freeze
    freeze = verify_freeze(config)
    directory = Path(run_directory)
    evaluation = read_json(directory / "evaluation.json")
    if evaluation["passed_candidates"]:
        confirmation = read_json(directory / "confirmation/evaluation.json")
        if confirmation["passed_candidates"] or confirmation.get("errors"):
            raise ValueError("A confirmed candidate requires prospective acceptance before final delivery")
    if evaluation["errors"] or evaluation["complete_model_panel_results"] != 12:
        raise ValueError("Incomplete search must not be labeled a conclusive negative result")
    selection = read_json(directory / "2025Q2_aapl_har/selection.json")
    cache, identity = prepare_data(config)
    data = load_data(cache, identity)
    split = partition("delivery_baseline", data.days[:-1].tolist(), data.days[-1:].tolist(),
        config["splits"]["calibration_sessions"], config["splits"]["purge_sessions"])
    model_directory = directory / "delivery_baseline"
    request = {"job_id": "delivery_baseline", "directory": str(model_directory.relative_to(ROOT)), "signature": freeze["signature"],
        "phase": "delivery", "partition": split.dictionary(), "model": "har", "panel": "aapl",
        "config_id": selection["config_id"], "params": selection["params"], "seed": 2026}
    with Scheduler(config, cache, identity, directory / "delivery_fit_resources") as scheduler:
        result = scheduler.map([request])[0]
    if result["status"] != "complete":
        raise RuntimeError("Reference baseline final fitting failed")
    manifest = {"schema_version": 1, "signature": freeze["signature"], "outcome": "completed_no_predictive_edge",
        "role": "reference_har_baseline", "active_predictive_edge_established": False, "prospective_qualified": False,
        "model": "har", "panel": "aapl", "params": selection["params"], "config_id": selection["config_id"],
        "source_directory": request["directory"], "source_result_sha256": sha(model_directory / "result.json"),
        "trained_through": split.fit_dates[-1], "calibration_through": split.calibration_dates[-1],
        "input_data_last_session": str(data.days[-1]), "selection_policy": "last_development_fold_inner_selected_aapl_HAR",
        "direction_or_profit_claim": False, "default_mode": "historical_replay"}
    write_json(ROOT / "delivery/model.json", manifest)
    return manifest


def infer(config, model_manifest="delivery/model.json", as_of=None, mode="historical", source_manifest=None):
    from .provenance import verify_freeze
    freeze = verify_freeze(config)
    manifest = read_json(ROOT / model_manifest)
    if manifest["signature"] != freeze["signature"]:
        raise ValueError("Inference manifest belongs to another protocol")
    now = datetime.now(timezone.utc).isoformat()
    base = {"schema_version": 1, "issued_at": now, "model": manifest["model"], "panel": manifest["panel"],
        "model_config_id": manifest["config_id"], "signature": manifest["signature"], "mode": mode,
        "active_predictive_edge_established": manifest["active_predictive_edge_established"],
        "prospective_qualified": manifest["prospective_qualified"], "eligible_signal": False,
        "unit": "sum_of_squared_log_returns_over_30_minutes", "horizon_minutes": 30,
        "direction_or_profit_prediction": False}
    if mode == "live":
        # Current verified entitlement does not admit timely SIP. A historical
        # replay or user-supplied timestamp cannot substitute for received data.
        return {**base, "status": "live_unavailable", "reason": "timely_SIP_and_received_at_capture_not_admitted",
            "external_dependency_record": "delivery/external_dependencies.json", "rv_mean": None}
    source = ROOT / manifest["source_directory"]
    evidence = receipt(source)
    if evidence is None or sha(source / "result.json") != manifest["source_result_sha256"]:
        raise ValueError("Frozen inference model artifact changed")
    data = RVData.from_manifest(source_manifest or config["data_manifest"])
    eligible = data.origins(require_target=False)
    if as_of is None:
        if not len(eligible):
            return {**base, "status": "insufficient_input_history", "rv_mean": None}
        origin = eligible[-1]
    else:
        stamp = pd.Timestamp(as_of)
        if stamp.tzinfo is None:
            raise ValueError("as-of must include an explicit timezone")
        stamp = stamp.tz_convert("UTC")
        location = data.timestamps.get_indexer([stamp])[0]
        if location < 0 or location not in eligible:
            return {**base, "status": "outside_session_or_insufficient_history", "as_of": str(stamp), "rv_mean": None}
        origin = location
    state = FitState(**read_json(source / "fit_state.json"))
    calibration = Calibration(**read_json(source / "calibration.json"))
    as_of_day = data.dates[origin]
    if as_of_day <= calibration.dates[-1] or as_of_day <= state.fit_dates[-1]:
        return {**base, "status": "before_frozen_model_availability", "as_of": str(data.timestamps[origin]),
            "calibration_through": calibration.dates[-1], "rv_mean": None}
    store = FeatureStore(data, config)
    device = "cuda" if manifest["model"] in ("xpatch", "chronos2") else "cpu"
    if device == "cuda":
        from .runtime import configure_torch
        configure_torch()
    model = load_model(manifest["model"], source, manifest["params"], config, device)
    raw = raw_predictions(manifest["model"], model, data, store, state, [origin], manifest["params"], manifest["panel"], config, device)
    mean, quantiles = calibration.predict(raw)
    stamp = data.timestamps[origin]
    return {**base, "status": "historical_replay_baseline", "as_of": str(stamp),
        "data_cutoff_at": str(stamp - pd.Timedelta(minutes=1)), "received_at": None,
        "target_window_start": str(stamp), "target_first_return_end": str(stamp + pd.Timedelta(minutes=1)),
        "target_window_end": str(stamp + pd.Timedelta(minutes=30)), "rv_mean": float(mean[0]),
        "quantiles": {str(level): float(value) for level, value in zip(calibration.levels, quantiles[0])},
        "interval_claim": "historical_empirical_no_coverage_guarantee"}
