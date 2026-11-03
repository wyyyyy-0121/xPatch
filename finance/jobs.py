"""Persistent, resource-limited workers and fingerprinted resumable job receipts."""
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time
import traceback

from .io import ROOT, read_json, sha, stable_hash, write_json
from .runtime import configure, configure_torch, snapshot


_CONTEXT = None


def initialize(config, cache_directory, cache_identity, allocations, use_gpu):
    global _CONTEXT
    ids = allocations.get() if hasattr(allocations, "get") else allocations
    configure(len(ids), ids, config["runtime"]["nice_increment"])
    if use_gpu:
        configure_torch()
    from .cache import load_data
    from .features import FeatureStore
    data = load_data(cache_directory, cache_identity)
    _CONTEXT = {"config": config, "data": data, "store": FeatureStore(data, config),
        "states": OrderedDict(), "chronos": None, "device": "cuda" if use_gpu else "cpu"}


def receipt(directory, request_hash=None):
    directory = Path(directory)
    path = directory / "result.json"
    if not path.exists():
        return None
    result = read_json(path)
    if request_hash is not None and request_hash != result["request_hash"]:
        raise ValueError("Existing job request changed; preserve it in another run")
    for filename, digest in result.get("files", {}).items():
        if sha(directory / filename) != digest:
            raise ValueError(f"Job artifact changed: {directory.name}/{filename}")
    return result


def finish_result(result, directory, started, before):
    result.update(wall_seconds=time.monotonic() - started, finished_utc=datetime.now(timezone.utc).isoformat(), resource_after=snapshot())
    result["cpu_seconds"] = result["resource_after"]["cpu_seconds"] - before["cpu_seconds"]
    result["files"] = {p.name: sha(p) for p in sorted(directory.iterdir()) if p.is_file() and p.name != "result.json"}
    write_json(directory / "result.json", result)
    return result


def run_job(request):
    """No selective row drops or quiet exception-to-success conversions."""
    import numpy as np
    from .calibration import Calibration
    from .features import FitState
    from .metrics import score
    from .models import ChronosModel, fit_neural, fit_tabular, raw_predictions
    from .splits import Partition
    context = _CONTEXT
    if context is None:
        raise RuntimeError("Numerical worker was not initialized")
    started = time.monotonic()
    config, data, store = context["config"], context["data"], context["store"]
    directory = ROOT / request["directory"]
    request_hash = stable_hash(request)
    previous = receipt(directory, request_hash)
    if previous is not None:
        return {**previous, "reused": True}
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "request.json").exists() and stable_hash(read_json(directory / "request.json")) != request_hash:
        raise ValueError("Interrupted job has a different request identity")
    write_json(directory / "request.json", request)
    before = snapshot()
    name = request["model"]
    result = {"job_id": request["job_id"], "request_hash": request_hash, "signature": request["signature"],
        "directory": str(directory.relative_to(ROOT)), "model": request["model"], "panel": request["panel"],
        "config_id": request["config_id"], "params": request["params"], "seed": request["seed"],
        "phase": request["phase"], "fold": request["partition"]["name"], "stage_epochs": request.get("max_epochs"),
        "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running", "resource_before": before}
    if request["phase"] == "verify":
        try:
            from .verify import verify_one
            result.update(status="complete", verification=verify_one(context, request, directory))
        except Exception as error:
            result.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        return finish_result(result, directory, started, before)
    try:
        partition = Partition(**{k: tuple(v) if k != "name" else v for k, v in request["partition"].items()})
        if request["phase"] == "smoke" and any(day >= "2023-01-01" for day in partition.evaluation_dates):
            raise ValueError("Resource smoke must stay before the first outer evaluation")
        if request["phase"] == "outer" and config["status"] != "frozen":
            raise ValueError("Formal outer jobs require a frozen protocol")
        state_key = stable_hash(partition.fit_dates)
        if state_key not in context["states"]:
            # Drop an old state before allocating the next one to bound peak memory.
            if len(context["states"]) >= 4:
                context["states"].popitem(last=False)
            context["states"][state_key] = FitState.fit(data, partition.fit_dates, config["seasonality"])
        state = context["states"][state_key]
        context["states"].move_to_end(state_key)
        write_json(directory / "fit_state.json", state.dictionary())
        name, params, panel, seed = request["model"], request["params"], request["panel"], request["seed"]
        if request["phase"] in ("outer", "confirmation", "delivery"):
            if name == "lightgbm" and request.get("fixed_iterations") is None:
                raise ValueError("Outer LightGBM iterations must come from inner selection")
            if name == "xpatch" and request.get("fixed_epochs") is None:
                raise ValueError("Outer xPatch epochs must come from inner selection")
        training = data.origins(partition.fit_dates, stride=config["sampling"]["training_stride_minutes"])
        calibration = data.origins(partition.calibration_dates, stride=config["sampling"]["calibration_stride_minutes"])
        evaluation = data.origins(partition.evaluation_dates, stride=(config["sampling"]["outer_stride_minutes"]
            if request["phase"] in ("outer", "confirmation", "delivery") else config["sampling"]["inner_validation_stride_minutes"]))
        if not len(training) or not len(calibration) or not len(evaluation):
            raise ValueError("Empty qualified training, calibration or evaluation panel")
        if request.get("smoke_limit_training") is not None:
            if request["phase"] != "smoke":
                raise ValueError("Training limits are reserved for unscored resource smoke")
            training = training[-request["smoke_limit_training"]:]
        if request.get("smoke_limit_prediction") is not None:
            if request["phase"] != "smoke":
                raise ValueError("Prediction limits are reserved for resource smoke")
            calibration = calibration[:request["smoke_limit_prediction"]]
            evaluation = evaluation[:request["smoke_limit_prediction"]]
        if name in ("har", "ridge", "lightgbm"):
            model, fitting = fit_tabular(name, data, store, state, partition, training, params, panel, config, seed,
                directory, before["worker_threads"], request.get("fixed_iterations"))
        elif name == "xpatch":
            if request.get("resume_directory"):
                source = ROOT / request["resume_directory"]
                evidence = receipt(source)
                if evidence is None or evidence["status"] != "complete":
                    raise ValueError("Halving resume requires a verified completed predecessor")
                predecessor = read_json(source / "request.json")
                for key in ("signature", "model", "panel", "config_id", "params", "seed", "partition"):
                    if stable_hash(predecessor[key]) != stable_hash(request[key]):
                        raise ValueError("Halving predecessor differs from the requested trial")
                shutil.copyfile(source / "model.pt", directory / "model.pt")
            model, fitting = fit_neural(data, store, state, partition, training, params, panel, config, seed,
                directory, request["max_epochs"], request.get("fixed_epochs"), context["device"],
                resume=bool(request.get("resume_directory") or (directory / "model.pt").exists()))
        elif name == "chronos2":
            if context["chronos"] is None:
                context["chronos"] = ChronosModel(config, context["device"])
            model, fitting = context["chronos"], {"training_origins": 0, "fixed_weights": True}
        elif name == "last_rv":
            model, fitting = None, {"training_origins": 0, "fixed_baseline": True}
        else:
            raise ValueError(name)
        raw_cal = raw_predictions(name, model, data, store, state, calibration, params, panel, config, context["device"])
        calibrator = Calibration.fit(data.targets(calibration), raw_cal, data.dates[calibration], config["calibration"]["quantiles"])
        write_json(directory / "calibration.json", calibrator.dictionary())
        raw = raw_predictions(name, model, data, store, state, evaluation, params, panel, config, context["device"])
        mean, quantiles = calibrator.predict(raw)
        y, dates = data.targets(evaluation), data.dates[evaluation]
        temporary = directory / "predictions.tmp.npz"
        np.savez_compressed(temporary, origins=evaluation, timestamps=data.timestamps[evaluation].asi8,
            dates=np.asarray(dates, dtype="U10"), y=y, raw=raw, mean=mean, quantiles=quantiles,
            calibration_origins=calibration, calibration_y=data.targets(calibration), calibration_raw=raw_cal)
        temporary.replace(directory / "predictions.npz")
        result.update(status="complete", fitting=fitting, score=score(y, mean, dates, quantiles, calibrator.levels),
            evaluation_origins=len(evaluation), evaluation_sessions=len(set(dates)),
            evaluation_dates_planned=len(partition.evaluation_dates), calibration_origins=len(calibration))
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finish_result(result, directory, started, before)
    if name == "xpatch":
        # Do not retain trained models across trials; only the fixed pipeline is shared.
        import gc
        import torch
        if "model" in locals():
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result
