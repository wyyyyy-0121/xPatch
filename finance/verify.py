"""Independent CSV label/metric recomputation and saved-model prediction replay."""
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import Calibration
from .features import FitState
from .io import ROOT, read_json, sha, stable_hash, write_json
from .jobs import receipt
from .models import ChronosModel, load_model, raw_predictions


def reference_targets(closes, origin_timestamps):
    """31 actual source closes give 30 adjacent returns, independent of RVData."""
    if isinstance(origin_timestamps, pd.DatetimeIndex):
        if origin_timestamps.tz is None:
            raise ValueError("Independent target timestamps must be timezone aware")
        timestamps = origin_timestamps.as_unit("ns").asi8
    else:
        timestamps = np.asarray(origin_timestamps, dtype=np.int64)
    if np.any(timestamps < pd.Timestamp("2000-01-01", tz="UTC").value):
        raise ValueError("Integer target timestamps must use nanoseconds since the epoch")
    parts = []
    for start in range(0, len(timestamps), 4000):
        batch = timestamps[start:start + 4000]
        starts = batch[:, None] + np.arange(-1, 30) * 60 * 10**9
        prices = closes.reindex(pd.DatetimeIndex(starts.reshape(-1), tz="UTC")).to_numpy().reshape(-1, 31)
        if not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError("Independent target has missing/nonpositive source closes")
        parts.append(np.sum(np.diff(np.log(prices), axis=1) ** 2, axis=1))
    return np.concatenate(parts)


def independent_scores(y, p, dates):
    y = np.maximum(y, 1e-12)
    p = np.maximum(p, 1e-12)
    frame = pd.DataFrame({"date": dates, "qlike": y / p + np.log(p) - np.log(y) - 1,
        "log_rv_mse": (np.log(y) - np.log(p)) ** 2})
    grouped = frame.groupby("date", sort=True)[["qlike", "log_rv_mse"]].mean()
    return {key: float(grouped[key].mean()) for key in grouped}


def replay_indices(size, batch, pretrained):
    if not pretrained:
        return np.arange(size)
    # Preserve original batch boundaries, including the last partial batch.
    last = ((size - 1) // batch) * batch
    return np.unique(np.r_[np.arange(min(size, batch)), np.arange(last, size)])


def verify_one(context, request, directory):
    source = ROOT / request["source_directory"]
    original = receipt(source)
    if original is None or original["status"] != "complete":
        raise ValueError("Replay source is not a verified completed job")
    for key in ("signature", "model", "panel", "params", "config_id", "seed"):
        if stable_hash(request[key]) != stable_hash(original[key]):
            raise ValueError("Replay request differs from source model identity")
    config, data, store = context["config"], context["data"], context["store"]
    original_request = read_json(source / "request.json")
    if original_request["phase"] not in ("outer", "confirmation", "smoke"):
        raise ValueError("Replay only completed evaluation jobs")
    state_saved = FitState(**read_json(source / "fit_state.json"))
    state = FitState.fit(data, original_request["partition"]["fit_dates"], config["seasonality"])
    if state_saved.dictionary() != state.dictionary():
        raise ValueError("Fit-only transforms could not be rebuilt")
    if "reference_closes" not in context:
        entry = read_json(ROOT / config["data_manifest"])["assets"]["AAPL"]
        if sha(ROOT / entry["csv"]) != entry["csv_sha256"]:
            raise ValueError("Independent target source changed")
        frame = pd.read_csv(ROOT / entry["csv"], usecols=["timestamp", "close"])
        context["reference_closes"] = pd.Series(frame.close.to_numpy(), index=pd.DatetimeIndex(pd.to_datetime(frame.timestamp, utc=True)))
    with np.load(source / "predictions.npz", allow_pickle=False) as saved:
        values = {key: saved[key] for key in saved.files}
    origins, cal_origins = values["origins"], values["calibration_origins"]
    rebuilt_y = reference_targets(context["reference_closes"], values["timestamps"])
    rebuilt_cal_y = reference_targets(context["reference_closes"], data.timestamps[cal_origins].asi8)
    np.testing.assert_allclose(rebuilt_y, values["y"], rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(rebuilt_cal_y, values["calibration_y"], rtol=1e-8, atol=1e-14)
    recomputed = independent_scores(rebuilt_y, values["mean"], values["dates"])
    for key in recomputed:
        np.testing.assert_allclose(recomputed[key], original["score"][key], rtol=1e-8, atol=1e-10)
    name, params, panel = request["model"], request["params"], request["panel"]
    if name == "chronos2" and context["chronos"] is None:
        context["chronos"] = ChronosModel(config, context["device"])
    model = load_model(name, source, params, config, context["device"], context["chronos"])
    batch = config["search"]["chronos2"]["batch_size"] if name == "chronos2" else params.get("batch_size", 512)
    indices = replay_indices(len(origins), batch, name == "chronos2")
    cal_indices = replay_indices(len(cal_origins), batch, name == "chronos2")
    # First and last pretrained batches are called separately, as in original prediction.
    def replay(selected):
        if name != "chronos2":
            return raw_predictions(name, model, data, store, state, selected, params, panel, config, context["device"])
        return np.concatenate([raw_predictions(name, model, data, store, state, selected[start:start + batch], params, panel,
            config, context["device"]) for start in range(0, len(selected), batch)])
    raw, cal_raw = replay(origins[indices]), replay(cal_origins[cal_indices])
    np.testing.assert_allclose(raw, values["raw"][indices], rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(cal_raw, values["calibration_raw"][cal_indices], rtol=1e-6, atol=1e-12)
    calibration = Calibration(**read_json(source / "calibration.json"))
    rebuilt_cal = Calibration.fit(rebuilt_cal_y, values["calibration_raw"], data.dates[cal_origins], calibration.levels)
    np.testing.assert_allclose(rebuilt_cal.log_factor, calibration.log_factor, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(rebuilt_cal.residual_quantiles, calibration.residual_quantiles, rtol=1e-7, atol=1e-8)
    mean, quantiles = calibration.predict(raw)
    np.testing.assert_allclose(mean, values["mean"][indices], rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(quantiles, values["quantiles"][indices], rtol=1e-6, atol=1e-12)
    result = {"passed": True, "source_directory": request["source_directory"], "source_result_sha256": sha(source / "result.json"),
        "independent_labels": len(origins), "independent_calibration_labels": len(cal_origins),
        "replayed_origins": len(indices), "replayed_calibration_origins": len(cal_indices),
        "scope": "first_and_last_fixed_batches" if name == "chronos2" else "full",
        "max_raw_absolute_error": float(np.max(np.abs(raw - values["raw"][indices]))), "independent_scores": recomputed}
    write_json(Path(directory) / "verification.json", result)
    if name == "xpatch":
        import gc
        import torch
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def verify_run(config, run_directory, round_number):
    from .cache import prepare_data
    from .provenance import verify_freeze
    from .scheduler import Scheduler
    freeze = verify_freeze(config)
    directory = Path(run_directory)
    setup = read_json(directory / "setup.json")
    if setup["signature"] != freeze["signature"]:
        raise ValueError("Verification run belongs to another freeze")
    destination = directory / f"verify_{round_number}"
    cache, identity = prepare_data(config)
    requests = []
    for split in setup["partitions"]:
        for panel in config["feature_sets"]:
            for name in config["models"]:
                group = read_json(directory / f"{split['name']}_{panel}_{name}" / "experiment.json")
                if group["status"] != "complete":
                    raise ValueError("Cannot verify incomplete outer model groups")
                for source_directory in group["outer_jobs"]:
                    source = read_json(ROOT / source_directory / "request.json")
                    job_id = f"verify{round_number}/{source['job_id']}"
                    request = {**{key: source[key] for key in ("signature", "model", "panel", "config_id", "params", "seed", "partition")},
                        "job_id": job_id, "directory": str((destination / source["job_id"]).relative_to(ROOT)),
                        "phase": "verify", "source_directory": source_directory}
                    requests.append(request)
    confirmation_path = directory / "confirmation/evaluation.json"
    if confirmation_path.exists():
        for source_directory in read_json(confirmation_path).get("job_directories", []):
            original = receipt(ROOT / source_directory)
            if original is None or original["status"] != "complete":
                raise ValueError("Cannot verify failed conditional confirmation jobs")
            source = read_json(ROOT / source_directory / "request.json")
            requests.append({**{key: source[key] for key in ("signature", "model", "panel", "config_id", "params", "seed", "partition")},
                "job_id": f"verify{round_number}/{source['job_id']}",
                "directory": str((destination / source["job_id"]).relative_to(ROOT)),
                "phase": "verify", "source_directory": source_directory})
    with Scheduler(config, cache, identity, destination) as scheduler:
        results = scheduler.map(requests)
    summary = {"signature": freeze["signature"], "round": round_number, "passed": all(r["status"] == "complete" for r in results),
        "jobs": len(results), "job_directories": [r["directory"] for r in results],
        "pretrained_scope": "first_and_last_fixed_batches_of_each_calibration_and_evaluation_job",
        "supervised_scope": "full_calibration_and_evaluation_predictions"}
    write_json(destination / "summary.json", summary)
    return summary
