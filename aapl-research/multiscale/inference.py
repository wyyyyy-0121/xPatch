"""Export and run a historical-development forecast bundle.

The bundle contains model parameters and metadata, not saved target or forecast
arrays.  It is local engineering infrastructure: neither exporting nor loading
it establishes data rights, authorizes publishing weights, or unlocks 2026Q3.
Inputs must be supplied by the caller and end at the requested decision time.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import shutil

import exchange_calendars as xc
import numpy as np
import pandas as pd
import torch

from .bars import Bars, MINUTE_NS, file_digest
from .baselines import predict as baseline_predict
from .development import bars_identity
from .features import CloseWindows, Scaler
from .models import make_model
from .ridge import RidgeFit
from .spec import DEV_END, MODELS, NY, OUT, PROFILES, ROOT, TARGETS
from .training import digest_indices, predict_best


NEURAL = ("mlp", "gru", "lstm", "xpatch")
PROFILE_BY_NAME = {profile.name: profile for profile in PROFILES}
HOUR_NS = 60 * MINUTE_NS


def _sha256(value):
    return (isinstance(value, str) and len(value) == 64 and
            all(char in "0123456789abcdef" for char in value))


def _checked_file(directory: Path, relative: str, expected: str):
    if not isinstance(relative, str) or not relative or not _sha256(expected):
        raise ValueError("Missing artifact path or SHA-256")
    root = directory.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file() or file_digest(path) != expected:
        raise ValueError(f"Artifact missing, outside bundle, or changed: {relative}")
    return path


def _check_sources(sources):
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Missing recorded training source identities")
    for name, expected in sources.items():
        _checked_file(ROOT, name, expected)


def _task(config):
    profile = PROFILE_BY_NAME.get(config.get("profile"))
    horizon, target = config.get("horizon"), config.get("target")
    if (profile is None or type(horizon) is not int or horizon not in profile.horizons or
            target not in TARGETS or type(config.get("lookback")) is not int or
            config["lookback"] < 2 or type(config.get("seasonal_period")) is not int or
            not 1 <= config["seasonal_period"] <= config["lookback"]):
        raise ValueError("Fold does not specify a supported multiscale task")
    return profile


def _actions(items):
    if not isinstance(items, list):
        raise ValueError("Split-action list is missing")
    result, previous = [], None
    for item in items:
        if not isinstance(item, dict) or set(item) != {"date", "ratio"}:
            raise ValueError("Invalid split-action record")
        try:
            day = date.fromisoformat(item["date"]).isoformat()
            if isinstance(item["ratio"], (bool, np.bool_)):
                raise ValueError("Boolean split ratio")
            ratio = float(item["ratio"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid split-action date or ratio") from exc
        if day != item["date"] or day > DEV_END or not np.isfinite(ratio) or ratio <= 0:
            raise ValueError("Split-action record is invalid or beyond the protected cutoff")
        if previous is not None and day <= previous:
            raise ValueError("Split-action dates must be strictly increasing")
        result.append({"date": day, "ratio": ratio})
        previous = day
    return result


def _json_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")
    return hashlib.sha256(payload).hexdigest()


def _data_kind(data_identity):
    if data_identity.startswith(("synthetic_test:", "synthetic-")):
        return "synthetic_test"
    if data_identity.startswith("market:") and _sha256(data_identity[len("market:"):]):
        return "market"
    raise ValueError("Fold data identity is neither explicitly synthetic nor an admitted market identity")


def _check_market_provenance(fold_directory, receipt, receipt_sha256, admission_dir, review_path, protocol_path):
    """Reapply the existing market-data review gate without opening price arrays."""
    if admission_dir is None or review_path is None or protocol_path is None:
        raise ValueError("Market bundles require the reviewed admission, review and frozen protocol")
    from .matrix_executor import _job_key, _json_inside, load_reviewed
    from .matrix_gate import locked_protocol_issues, profile_selection

    protocol, protocol_hash, _ = _json_inside(Path(protocol_path), OUT)
    selected = profile_selection(protocol.get("selected_profiles"))
    admission, admission_hash, _, _, review_hash, _ = load_reviewed(
        Path(admission_dir), Path(review_path), selected_profiles=selected)
    issues = locked_protocol_issues(admission, admission_hash, protocol, review_hash)
    if (issues or protocol.get("data_kind") != "market" or
            receipt["data_identity"] != "market:" + admission_hash or
            receipt["config"].get("protocol_identity") != protocol_hash):
        raise ValueError("Market bundle is not bound to the approved frozen protocol")
    root = (OUT / "matrix_runs" / protocol_hash[:24]).resolve()
    if not root.is_relative_to((OUT / "matrix_runs").resolve()) or not fold_directory.is_relative_to(root):
        raise ValueError("Market fold is outside its frozen matrix run")
    state = json.loads((root / "matrix_state.json").read_text(encoding="utf8"))
    if (state.get("protocol_sha256") != protocol_hash or state.get("review_sha256") != review_hash or
            state.get("data_kind") != "market"):
        raise ValueError("Market matrix state does not match its reviewed protocol")
    cfg = receipt["config"]
    candidates = [job for job in protocol.get("jobs", [])
                  if all(job.get(key) == cfg.get(key) for key in
                         ("profile", "horizon", "target", "lookback", "seasonal_period"))]
    records = [state.get("jobs", {}).get(_job_key(job), {}) for job in candidates]
    matches = [entry for entry in records if entry.get("status") == "completed" and
               entry.get("receipt_sha256") == receipt_sha256 and isinstance(entry.get("directory"), str) and
               (root / entry["directory"]).resolve() == fold_directory]
    if len(matches) != 1:
        raise ValueError("Market fold is not a completed entry in the reviewed matrix state")
    return {"admission_sha256": admission_hash, "review_sha256": review_hash, "protocol_sha256": protocol_hash}


def export_development_bundle(fold_directory: Path, *, expected_receipt_sha256: str,
                              output_directory: Path, admission_dir: Path | None = None,
                              review_path: Path | None = None, protocol_path: Path | None = None):
    """Copy only eight models' inference artifacts from a completed local fold.

    The receipt digest must come from a trusted matrix/verification record. The
    first saved validation decision is used as a conservative lower bound: the
    fold runner purges all fit and stopping labels before that time. This is not
    proof that the source or use of weights is licensed.
    """
    fold_directory = Path(fold_directory).resolve()
    receipt_path = _checked_file(fold_directory, "fold_receipt.json", expected_receipt_sha256)
    receipt = json.loads(receipt_path.read_text(encoding="utf8"))
    if (receipt.get("schema_version") != 2 or receipt.get("status") != "completed" or
            receipt.get("phase") != "development" or receipt.get("independent_test") is not False or
            receipt.get("Q3_values_read") is not False or not isinstance(receipt.get("data_identity"), str) or
            not receipt["data_identity"]):
        raise ValueError("Expected a completed historical-development fold")
    config, fields = receipt.get("config"), receipt.get("identity_fields")
    if not isinstance(config, dict) or not isinstance(fields, dict):
        raise ValueError("Fold configuration and identity fields are required")
    profile = _task(config)
    if (fields.get("config") != config or fields.get("data_identity") != receipt["data_identity"] or
            fields.get("source_sha256") != receipt.get("source_sha256") or
            _json_hash(fields) != receipt.get("fold_identity")):
        raise ValueError("Fold identity fields do not match the receipt")
    _check_sources(receipt["source_sha256"])
    data_kind = _data_kind(receipt["data_identity"])
    provenance = (_check_market_provenance(fold_directory, receipt, expected_receipt_sha256,
                                           admission_dir, review_path, protocol_path)
                  if data_kind == "market" else None)
    actions = _actions(fields.get("actions"))
    models = receipt.get("models")
    if not isinstance(models, dict) or set(models) != set(MODELS):
        raise ValueError("Fold must contain all eight models")
    last_close = models["last_close"]
    prediction_file = _checked_file(fold_directory, "prediction_last_close.npz",
                                    last_close.get("prediction_sha256"))
    # Do not read the stored forecast values or target labels to establish time.
    with np.load(prediction_file, allow_pickle=False) as stored:
        if set(stored.files) != {"prediction", "origins", "decision_ns"}:
            raise ValueError("Unexpected saved validation timing fields")
        decisions = stored["decision_ns"]
        origins = stored["origins"]
    if (decisions.ndim != 1 or not len(decisions) or
            len(decisions) != receipt.get("validation_origins") or
            not np.issubdtype(decisions.dtype, np.integer) or
            origins.shape != decisions.shape or not np.issubdtype(origins.dtype, np.integer) or
            (np.diff(origins) <= 0).any() or digest_indices(origins) != fields.get("validation_indices_sha256") or
            (np.diff(decisions) <= 0).any() or int(decisions[-1]) >= _cutoff_ns()):
        raise ValueError("Invalid saved validation decision times")

    files = {}
    source_files = {}
    ridge = models["ridge"]
    for src, target, field in (("ridge_coefficients.npz", "ridge_coefficients.npz", "coefficients_sha256"),
                               ("ridge_scaler.json", "ridge_scaler.json", "scaler_sha256")):
        source_files[target] = _checked_file(fold_directory, src, ridge.get(field))
        files[target] = ridge[field]
    model_meta = {"ridge": {"alpha": ridge.get("alpha"), "observations": ridge.get("training_origins")}}
    for name in NEURAL:
        details = models[name]
        checkpoint_dir = details.get("checkpoint_dir", name)
        if (not isinstance(checkpoint_dir, str) or Path(checkpoint_dir).name != checkpoint_dir or
                (checkpoint_dir != name and not (checkpoint_dir.startswith(name + "-retry-") and
                 checkpoint_dir[len(name) + 7:].isdigit())) or not _sha256(details.get("training_identity"))):
            raise ValueError(f"Invalid saved checkpoint identity for {name}")
        target = name + "_best.pt"
        source_files[target] = _checked_file(fold_directory, checkpoint_dir + "/best.pt",
                                             details.get("best_checkpoint_sha256"))
        files[target] = details["best_checkpoint_sha256"]
        model_meta[name] = {"training_identity": details["training_identity"], "epochs": details.get("epochs"),
                            "best_epoch": details.get("best_epoch"), "steps": details.get("steps"),
                            "stopped_early": details.get("stopped_early")}

    out = Path(output_directory)
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name, source in source_files.items():
        with (out / name).open("xb") as destination, source.open("rb") as stream:
            shutil.copyfileobj(stream, destination)
        _checked_file(out, name, files[name])
    bundle_status = ("synthetic_test_inference_bundle" if data_kind == "synthetic_test"
                     else "historical_development_inference_bundle")
    manifest = {"schema_version": 1, "status": bundle_status, "data_kind": data_kind,
                "independent_test": False, "Q3_values_read": False,
                "rights_status": "export does not authorize use or publication of model weights",
                "fold_receipt_sha256": expected_receipt_sha256, "fold_identity": receipt["fold_identity"],
                "market_review_provenance": provenance,
                "protocol_identity": config.get("protocol_identity"),
                "training_data_identity": receipt["data_identity"],
                "profile": profile.name, "frequency": profile.frequency,
                "horizon": config["horizon"], "target": config["target"],
                "lookback": config["lookback"], "seasonal_period": config["seasonal_period"],
                "training_recipe": {key: config.get(key) for key in
                                    ("ridge_alphas", "max_epochs", "patience", "batch_size", "learning_rate", "seed")},
                "training_runtime": fields.get("runtime"),
                "minimum_decision_ns": int(decisions[0]), "split_actions": actions,
                "models": list(MODELS), "model_metadata": model_meta, "files_sha256": files,
                "training_source_sha256": receipt["source_sha256"],
                "inference_source_sha256": file_digest(Path(__file__)),
                "limits": ["Forecasts and target schedules are development-only, not confirmation.",
                           "No target arrays or validation prediction arrays are exported.",
                           "Weights and actions may themselves be subject to supplier terms."]}
    manifest_path = out / "manifest.json"
    with manifest_path.open("x", encoding="utf8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    manifest_sha256 = file_digest(manifest_path)
    try:
        load_development_bundle(out, expected_manifest_sha256=manifest_sha256)
    except Exception as exc:
        manifest["status"] = "failed_bundle_verification"
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf8")
        raise
    return {"bundle": str(out.resolve()), "manifest": str(manifest_path.resolve()),
            "manifest_sha256": manifest_sha256, "models": len(MODELS),
            "raw_market_price_or_target_arrays_exported": False,
            "status": manifest["status"]}


def _cutoff_ns():
    return int(pd.Timestamp("2026-07-01", tz=NY).tz_convert("UTC").as_unit("ns").value)


def _calendar(labels, frequency, horizon):
    try:
        parsed = [date.fromisoformat(str(value)) for value in labels]
    except ValueError as exc:
        raise ValueError("Session labels must be ISO dates") from exc
    if any(day.isoformat() != str(label) for day, label in zip(parsed, labels, strict=True)):
        raise ValueError("Session labels must be canonical ISO dates")
    first = parsed[0].replace(day=1) - timedelta(days=7)
    extension = {"minute": 14, "hour": 45, "day": 3 * horizon + 45,
                 "month": 35 * horizon + 65}[frequency]
    last = parsed[-1] + timedelta(days=extension)
    return xc.get_calendar("XNYS", start=str(first), end=str(last)).schedule


def _sessions(schedule):
    return [(str(day.date()), int(row["open"].value), int(row["close"].value))
            for day, row in schedule.iterrows()]


def _intraday_slots(sessions, width):
    slots = []
    for label, opened, closed in sessions:
        for start in range(opened, closed - width + 1, width):
            slots.append((label, start, start + width))
    return slots


def _history_and_schedule(bars: Bars, lookback: int, horizon: int):
    if len(bars.close) < lookback:
        raise ValueError("Not enough completed historical bars")
    if any(not np.issubdtype(a.dtype, np.integer) for a in (bars.start_ns, bars.end_ns)):
        raise ValueError("Bar timestamps must be integer nanoseconds")
    part = slice(len(bars.close) - lookback, None)
    history = Bars(bars.frequency, bars.start_ns[part], bars.end_ns[part], bars.close[part],
                   bars.observed[part], bars.session_date[part])
    if not history.observed.all():
        raise ValueError("The complete input window must be observed")
    schedule = _calendar(history.session_date, history.frequency, horizon)
    sessions = _sessions(schedule)
    by_date = {day: (opened, closed) for day, opened, closed in sessions}
    labels = history.session_date.tolist()
    if history.frequency in ("minute", "hour"):
        width = MINUTE_NS if history.frequency == "minute" else HOUR_NS
        slots = _intraday_slots(sessions, width)
        expected = [slot for slot in slots if int(history.start_ns[0]) <= slot[1] <= int(history.start_ns[-1])]
        if (len(expected) != lookback or [x[0] for x in expected] != labels or
                not np.array_equal(np.asarray([x[1] for x in expected], dtype=np.int64), history.start_ns) or
                not np.array_equal(np.asarray([x[2] for x in expected], dtype=np.int64), history.end_ns)):
            raise ValueError("Input window is not a complete regular-session bar grid")
    elif history.frequency == "day":
        expected = [day for day, _, _ in sessions if labels[0] <= day <= labels[-1]]
        if expected != labels or any(int(start) > by_date[day][0] or int(end) < by_date[day][1]
                                      for day, start, end in zip(labels, history.start_ns, history.end_ns, strict=True)):
            raise ValueError("Daily input is incomplete or ends before the regular close")
    elif history.frequency == "month":
        months = pd.period_range(labels[0][:7], labels[-1][:7], freq="M")
        grouped = {}
        for day, opened, closed in sessions:
            month = day[:7]
            if month not in grouped:
                grouped[month] = [day, opened, day, closed]
            else:
                grouped[month][2:] = [day, closed]
        expected = [grouped[str(month)] for month in months]
        if (len(expected) != lookback or [x[2] for x in expected] != labels or
                any(int(start) > item[1] or int(end) < item[3]
                    for item, start, end in zip(expected, history.start_ns, history.end_ns, strict=True))):
            raise ValueError("Monthly input is incomplete or not a completed trading month")
    else:
        raise ValueError("Unsupported bar frequency")

    origin, decision = labels[-1], int(history.end_ns[-1])
    if history.frequency == "minute":
        end = by_date[origin][1]
        if decision + horizon * MINUTE_NS > end:
            raise ValueError("Minute forecast would cross the regular session")
        future = [(origin, decision + step * MINUTE_NS) for step in range(1, horizon + 1)]
    elif history.frequency == "hour":
        future = [(day, end) for day, start, end in slots if start >= decision][:horizon]
    elif history.frequency == "day":
        future = [(day, close) for day, _, close in sessions if day > origin][:horizon]
    else:
        future = [(value[2], value[3]) for month, value in sorted(grouped.items())
                  if month > origin[:7]][:horizon]
    if len(future) != horizon or any(end <= decision or end >= _cutoff_ns() for _, end in future):
        raise ValueError("Forecast targets are unavailable or cross the protected cutoff")
    result = [{"step": index, "session_date": day, "scheduled_session_or_bar_end_ns": end,
               "target_completion_ns": None,
               "elapsed_seconds_from_decision": (end - decision) / 1e9,
               "observed": None} for index, (day, end) in enumerate(future, start=1)]
    return history, result


def load_development_bundle(directory: Path, *, expected_manifest_sha256: str, device="cpu"):
    """Verify a bundle and return an object able to forecast from past-only bars."""
    directory = Path(directory).resolve()
    path = _checked_file(directory, "manifest.json", expected_manifest_sha256)
    manifest = json.loads(path.read_text(encoding="utf8"))
    kind = manifest.get("data_kind")
    expected_status = ("synthetic_test_inference_bundle" if kind == "synthetic_test" else
                       "historical_development_inference_bundle" if kind == "market" else None)
    if (manifest.get("schema_version") != 1 or expected_status is None or manifest.get("status") != expected_status or
            manifest.get("independent_test") is not False or manifest.get("Q3_values_read") is not False or
            manifest.get("models") != list(MODELS) or not _sha256(manifest.get("fold_receipt_sha256")) or
            not _sha256(manifest.get("fold_identity")) or not isinstance(manifest.get("training_data_identity"), str) or
            not manifest["training_data_identity"] or type(manifest.get("minimum_decision_ns")) is not int or
            manifest["minimum_decision_ns"] >= _cutoff_ns()):
        raise ValueError("Invalid inference manifest or protected scope")
    if _data_kind(manifest["training_data_identity"]) != kind:
        raise ValueError("Bundle and training-data identities have different kinds")
    if kind == "market":
        provenance = manifest.get("market_review_provenance")
        if (not isinstance(provenance, dict) or
                set(provenance) != {"admission_sha256", "review_sha256", "protocol_sha256"} or
                any(not _sha256(value) for value in provenance.values()) or
                manifest["training_data_identity"] != "market:" + provenance["admission_sha256"] or
                manifest.get("protocol_identity") != provenance["protocol_sha256"]):
            raise ValueError("Market bundle lacks reviewed provenance")
    elif manifest.get("market_review_provenance") is not None:
        raise ValueError("Synthetic bundle cannot claim market review provenance")
    profile = _task(manifest)
    if manifest.get("frequency") != profile.frequency:
        raise ValueError("Bundle frequency differs from its study profile")
    _actions(manifest.get("split_actions"))
    _check_sources(manifest.get("training_source_sha256"))
    if file_digest(Path(__file__)) != manifest.get("inference_source_sha256"):
        raise ValueError("Inference source differs from the exported bundle")
    expected_files = {"ridge_coefficients.npz", "ridge_scaler.json"} | {name + "_best.pt" for name in NEURAL}
    files = manifest.get("files_sha256")
    if not isinstance(files, dict) or set(files) != expected_files:
        raise ValueError("Bundle artifacts are incomplete")
    for name, digest in files.items():
        _checked_file(directory, name, digest)
    metadata = manifest.get("model_metadata")
    if not isinstance(metadata, dict) or set(metadata) != {"ridge", *NEURAL}:
        raise ValueError("Bundle model metadata is incomplete")
    recipe = manifest.get("training_recipe")
    if (not isinstance(recipe, dict) or set(recipe) !=
            {"ridge_alphas", "max_epochs", "patience", "batch_size", "learning_rate", "seed"} or
            not isinstance(recipe["ridge_alphas"], list) or not recipe["ridge_alphas"]):
        raise ValueError("Bundle training recipe is incomplete")
    ridge = metadata["ridge"]
    if (not isinstance(ridge, dict) or not isinstance(ridge.get("alpha"), (float, int)) or
            isinstance(ridge["alpha"], bool) or not np.isfinite(ridge["alpha"]) or ridge["alpha"] <= 0 or
            type(ridge.get("observations")) is not int or ridge["observations"] < 1):
        raise ValueError("Invalid Ridge metadata")
    if ridge["alpha"] not in recipe["ridge_alphas"]:
        raise ValueError("Saved Ridge alpha is outside the training recipe")
    ridge_scaler = _validate_scaler(json.loads((directory / "ridge_scaler.json").read_text(encoding="utf8")),
                                    manifest["horizon"])
    if ridge_scaler.fitted_origin_count != ridge["observations"]:
        raise ValueError("Ridge scaler and training-origin counts differ")
    with np.load(directory / "ridge_coefficients.npz", allow_pickle=False) as saved:
        if set(saved.files) != {"coefficients"}:
            raise ValueError("Unexpected Ridge coefficient fields")
        coefficients = np.asarray(saved["coefficients"], dtype=np.float64)
    if (coefficients.shape != (manifest["lookback"] + 2, manifest["horizon"]) or
            not np.isfinite(coefficients).all()):
        raise ValueError("Invalid Ridge coefficient dimensions or values")
    for name in NEURAL:
        if not isinstance(metadata[name], dict) or not _sha256(metadata[name].get("training_identity")):
            raise ValueError(f"Invalid training identity for {name}")
        detail = metadata[name]
        if (type(detail.get("epochs")) is not int or detail["epochs"] < 1 or
                type(detail.get("best_epoch")) is not int or not 1 <= detail["best_epoch"] <= detail["epochs"] or
                type(detail.get("steps")) is not int or detail["steps"] < 1 or
                type(detail.get("stopped_early")) is not bool):
            raise ValueError(f"Invalid training summary for {name}")
        saved = torch.load(directory / (name + "_best.pt"), map_location="cpu", weights_only=True)
        config = saved.get("config", {})
        if (saved.get("identity") != metadata[name]["training_identity"] or config.get("model") != name or
                config.get("lookback") != manifest["lookback"] or config.get("horizon") != manifest["horizon"] or
                config.get("target") != manifest["target"] or config.get("shared_horizons") not in ([], ())):
            raise ValueError(f"Checkpoint task or training identity differs for {name}")
        if (any(config.get(key) != recipe[key] for key in
                ("max_epochs", "patience", "batch_size", "learning_rate", "seed")) or
                detail["epochs"] > config["max_epochs"]):
            raise ValueError(f"Checkpoint training recipe differs for {name}")
        _validate_scaler(saved.get("scaler"), manifest["horizon"])
        state = saved.get("model")
        if not isinstance(state, dict) or any(not isinstance(value, torch.Tensor) or
                                              not torch.isfinite(value).all() for value in state.values()):
            raise ValueError(f"Invalid checkpoint model state for {name}")
        with torch.random.fork_rng(devices=[]):
            make_model(name, manifest["lookback"], manifest["horizon"]).load_state_dict(state, strict=True)
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Unsupported or unavailable inference device")
    return DevelopmentPredictor(directory, manifest, expected_manifest_sha256, device)


def _validate_scaler(value, horizon):
    if not isinstance(value, dict):
        raise ValueError("Missing scaler")
    try:
        scaler = Scaler(**value)
        scalars = np.asarray([scaler.x_mean, scaler.x_std, scaler.anchor_mean, scaler.anchor_std], dtype=float)
        means, scales = np.asarray(scaler.y_mean, dtype=float), np.asarray(scaler.y_std, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid scaler fields") from exc
    if (not np.isfinite(scalars).all() or scalars[1] <= 0 or scalars[3] <= 0 or
            means.shape != (horizon,) or scales.shape != (horizon,) or not np.isfinite(means).all() or
            not np.isfinite(scales).all() or (scales <= 0).any() or
            type(scaler.fitted_origin_count) is not int or scaler.fitted_origin_count < 1):
        raise ValueError("Invalid scaler values or dimensions")
    return scaler


class DevelopmentPredictor:
    def __init__(self, directory, manifest, manifest_sha256, device):
        self.directory, self.manifest, self.manifest_sha256, self.device = directory, manifest, manifest_sha256, device
        self._manifest_content_sha256 = _json_hash(manifest)

    def predict(self, bars: Bars, *, profile: str, horizon: int, target: str, model: str,
                as_of_ns: int, history_identity: str):
        m = self.manifest
        _checked_file(self.directory, "manifest.json", self.manifest_sha256)
        if _json_hash(m) != self._manifest_content_sha256 or file_digest(Path(__file__)) != m["inference_source_sha256"]:
            raise ValueError("Loaded inference identity changed")
        _check_sources(m["training_source_sha256"])
        if (profile != m["profile"] or type(horizon) is not int or horizon != m["horizon"] or target != m["target"] or
                model not in MODELS or bars.frequency != m["frequency"]):
            raise ValueError("Requested task, frequency, or model differs from the bundle")
        if (not isinstance(as_of_ns, (int, np.integer)) or isinstance(as_of_ns, (bool, np.bool_)) or
                int(as_of_ns) != int(bars.end_ns[-1]) or int(as_of_ns) < m["minimum_decision_ns"] or
                int(as_of_ns) >= _cutoff_ns() or bars.session_date[-1] > DEV_END):
            raise ValueError("Decision time is not an eligible completed historical bar")
        if not isinstance(history_identity, str) or not history_identity.strip():
            raise ValueError("A caller-supplied history identity is required")
        history, future = _history_and_schedule(bars, m["lookback"], horizon)
        actions = [action for action in m["split_actions"] if action["date"] <= history.session_date[-1]]
        windows = CloseWindows(history, m["lookback"], actions)
        origin = np.asarray([len(history.close) - 1], dtype=np.int64)
        if model in ("last_close", "drift", "seasonal"):
            prediction = baseline_predict(model, windows, origin, horizon, target,
                                          period=m["seasonal_period"] if model == "seasonal" else None)
        elif model == "ridge":
            _checked_file(self.directory, "ridge_coefficients.npz", m["files_sha256"]["ridge_coefficients.npz"])
            _checked_file(self.directory, "ridge_scaler.json", m["files_sha256"]["ridge_scaler.json"])
            with np.load(self.directory / "ridge_coefficients.npz", allow_pickle=False) as saved:
                if set(saved.files) != {"coefficients"}:
                    raise ValueError("Unexpected Ridge coefficient fields")
                coefficients = np.asarray(saved["coefficients"], dtype=np.float64)
            scaler_data = json.loads((self.directory / "ridge_scaler.json").read_text(encoding="utf8"))
            scaler = _validate_scaler(scaler_data, horizon)
            meta = m["model_metadata"]["ridge"]
            if coefficients.shape != (m["lookback"] + 2, horizon) or not np.isfinite(coefficients).all():
                raise ValueError("Invalid Ridge coefficient dimensions")
            prediction = RidgeFit(coefficients, float(meta["alpha"]), int(meta["observations"])).predict(
                windows, scaler, origin)
        else:
            _checked_file(self.directory, model + "_best.pt", m["files_sha256"][model + "_best.pt"])
            devices = [torch.cuda.current_device()] if self.device == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                prediction = predict_best(self.directory / (model + "_best.pt"), windows, origin,
                                          m["model_metadata"][model]["training_identity"], device=self.device)
        if prediction.shape != (1, horizon) or not np.isfinite(prediction).all():
            raise ValueError("Invalid forecast path")
        status = "synthetic_test_forecast_only" if m["data_kind"] == "synthetic_test" else "historical_development_forecast"
        return {"status": status, "data_kind": m["data_kind"], "independent_test": False,
                "bundle_manifest_sha256": self.manifest_sha256, "fold_identity": m["fold_identity"],
                "training_data_identity": m["training_data_identity"], "history_identity": history_identity,
                "input_window_sha256": bars_identity(history), "profile": profile, "frequency": history.frequency,
                "horizon": horizon, "target": target, "model": model,
                "origin_start_ns": int(history.start_ns[-1]), "decision_ns": int(as_of_ns),
                "origin_session_date": str(history.session_date[-1]), "prediction": prediction[0].tolist(),
                "future_targets": future, "calendar": "XNYS proxy", "calendar_version": xc.__version__,
                "limits": ["Scheduled target times are calendar expectations, not observed labels or vendor availability.",
                           "Source rights, historical revisions, and monthly source completeness require separate review.",
                           "A development forecast is not independent confirmation or a trading result."]}


def _read_bars(path: Path):
    with np.load(path, allow_pickle=False) as data:
        required = {"frequency", "start_ns", "end_ns", "close", "observed", "session_date"}
        if set(data.files) != required:
            raise ValueError("Input bars must contain exactly the documented arrays")
        labels = data["session_date"]
        if labels.dtype.kind != "U" or labels.ndim != 1 or not len(labels) or (labels > DEV_END).any():
            raise ValueError("Input session metadata crosses the protected cutoff")
        frequency = str(data["frequency"].item())
        return Bars(frequency, data["start_ns"], data["end_ns"], data["close"], data["observed"], labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--fold", type=Path, required=True)
    export.add_argument("--receipt-sha256", required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--admission", type=Path)
    export.add_argument("--review", type=Path)
    export.add_argument("--protocol", type=Path)
    forecast = commands.add_parser("forecast")
    forecast.add_argument("--bundle", type=Path, required=True)
    forecast.add_argument("--manifest-sha256", required=True)
    forecast.add_argument("--bars", type=Path, required=True)
    forecast.add_argument("--profile", choices=list(PROFILE_BY_NAME), required=True)
    forecast.add_argument("--horizon", type=int, required=True)
    forecast.add_argument("--target", choices=TARGETS, required=True)
    forecast.add_argument("--model", choices=MODELS, required=True)
    forecast.add_argument("--as-of-ns", type=int, required=True)
    forecast.add_argument("--history-identity", required=True)
    forecast.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.command == "export":
        result = export_development_bundle(args.fold, expected_receipt_sha256=args.receipt_sha256,
                                           output_directory=args.output, admission_dir=args.admission,
                                           review_path=args.review, protocol_path=args.protocol)
    else:
        predictor = load_development_bundle(args.bundle, expected_manifest_sha256=args.manifest_sha256,
                                            device=args.device)
        result = predictor.predict(_read_bars(args.bars), profile=args.profile, horizon=args.horizon,
                                   target=args.target, model=args.model, as_of_ns=args.as_of_ns,
                                   history_identity=args.history_identity)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
