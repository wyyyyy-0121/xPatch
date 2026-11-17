"""Execute one chronological development fold, never an independent test.

The caller must separately freeze an admissible protocol, source identity and
resource budget. This component does not select data sources or rent resources.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import torch

from .bars import Bars, file_digest, targets
from .baselines import predict as baseline_predict
from .evaluation import endpoint_improvement, path_metrics
from .features import CloseWindows, fit_scaler
from .ridge import fit_many
from .spec import DEV_END, MODELS, NY
from .training import TrainConfig, digest_indices, identity as training_identity, predict_best, run_fit


@dataclass(frozen=True)
class FoldConfig:
    profile: str
    horizon: int
    target: str
    lookback: int
    seasonal_period: int
    ridge_alphas: tuple[float, ...]
    max_epochs: int
    patience: int
    batch_size: int
    learning_rate: float
    seed: int
    protocol_identity: str

    def validate(self):
        if self.horizon < 1 or self.lookback < 2 or not 1 <= self.seasonal_period <= self.lookback:
            raise ValueError("Invalid task dimensions or seasonal period")
        if self.target not in ("future_close", "future_log_return"):
            raise ValueError("Invalid target")
        if not self.ridge_alphas or any(not np.isfinite(a) or a <= 0 for a in self.ridge_alphas):
            raise ValueError("Invalid Ridge search")
        if len(set(self.ridge_alphas)) != len(self.ridge_alphas) or not self.protocol_identity:
            raise ValueError("Repeated Ridge settings or missing protocol identity")
        TrainConfig("mlp", self.lookback, self.horizon, self.target, self.max_epochs,
                    self.patience, self.batch_size, self.learning_rate, self.seed).validate()


def validate_split(bars: Bars, fold: dict, horizon: int):
    train = np.asarray(fold["train"], dtype=np.int64)
    validation = np.asarray(fold["validation"], dtype=np.int64)
    inner = fold.get("inner")
    if inner is None:
        raise ValueError("Missing purged inner partition")
    fit, stop = (np.asarray(inner[key], dtype=np.int64) for key in ("fit", "stop"))
    for name, values in (("train", train), ("validation", validation), ("fit", fit), ("stop", stop)):
        if not len(values) or (np.diff(values) <= 0).any() or values[0] < 0 or values[-1] + horizon >= len(bars.close):
            raise ValueError(f"Invalid {name} origins")
    if len(fit) < 8 or not np.isin(fit, train).all() or not np.isin(stop, train).all():
        raise ValueError("Inner partitions are outside train or have fewer than eight fit examples")
    if bars.end_ns[train[-1] + horizon] >= bars.end_ns[validation[0]]:
        raise ValueError("Outer training labels overlap validation")
    if bars.end_ns[fit[-1] + horizon] >= bars.end_ns[stop[0]]:
        raise ValueError("Inner training labels overlap stopping")
    cutoff = pd.Timestamp("2026-07-01", tz=NY).tz_convert("UTC").as_unit("ns").value
    if bars.session_date[-1] > DEV_END or bars.end_ns[validation[-1] + horizon] >= cutoff:
        raise ValueError("Development fold crosses the protected data cutoff")
    return train, fit, stop, validation


def batched_baseline(name, windows, origins, horizon, target, period, batch_size=4096):
    out = []
    for start in range(0, len(origins), batch_size):
        out.append(baseline_predict(name, windows, origins[start:start + batch_size], horizon, target,
                                    period=period if name == "seasonal" else None))
    return np.concatenate(out)


def atomic_json(path: Path, value):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf8", prefix=".fold-", dir=path.parent,
                                     delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_npz(path: Path, **arrays):
    with tempfile.NamedTemporaryFile(mode="w+b", prefix=".fold-", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def bars_identity(bars: Bars):
    digest = hashlib.sha256(bars.frequency.encode("utf8"))
    for name in ("start_ns", "end_ns", "close", "observed", "session_date"):
        array = np.ascontiguousarray(getattr(bars, name))
        digest.update(name.encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def run_development_fold(bars: Bars, fold: dict, actions, cfg: FoldConfig, directory: Path,
                         data_identity: str, *, device="cpu", resume=False):
    """Run or resume one fold while holding its exclusive process lock."""
    cfg.validate()
    if not isinstance(data_identity, str) or not data_identity:
        raise ValueError("Verified data identity is required")
    directory = Path(directory)
    if resume:
        if not directory.is_dir():
            raise ValueError("No resumable fold directory")
    else:
        directory.mkdir(parents=True, exist_ok=False)
    with (directory / "fold.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Fold is already active in another process") from None
        try:
            return _run_development_fold(bars, fold, actions, cfg, directory, data_identity,
                                         device=device, resume=resume)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _run_development_fold(bars: Bars, fold: dict, actions, cfg: FoldConfig, directory: Path,
                          data_identity: str, *, device="cpu", resume=False):
    """Fit and score a development fold; resume only its identical saved state."""
    cfg.validate()
    if not isinstance(data_identity, str) or not data_identity:
        raise ValueError("Verified data identity is required")
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested device is unavailable")
    train, fit, stop, validation = validate_split(bars, fold, cfg.horizon)
    actions = [{"date": str(item["date"]), "ratio": float(item["ratio"])} for item in actions]
    windows = CloseWindows(bars, cfg.lookback, actions)
    future_idx = validation[:, None] + np.arange(1, cfg.horizon + 1)
    if not bars.observed[validation].all() or not bars.observed[future_idx].all():
        raise ValueError("Validation path has an unobserved bar")
    if bars.frequency == "minute" and not np.all(bars.session_date[validation] == bars.session_date[validation + cfg.horizon]):
        raise ValueError("Minute validation path crosses sessions")
    anchor = bars.close[validation]
    directory = Path(directory)
    local = Path(__file__).parent
    source_names = ("development.py", "bars.py", "baselines.py", "evaluation.py", "features.py",
                    "models.py", "ridge.py", "splits.py", "spec.py", "training.py", "mechanism.py")
    source_paths = [local / name for name in source_names] + [local.parent / "aapl" / "models.py",
                                                              local.parent / "aapl" / "vendor" / "xpatch_network.py"]
    sources = {str(path.relative_to(local.parent)): file_digest(path) for path in source_paths}
    identity_fields = {"config": asdict(cfg), "data_identity": data_identity, "bars_sha256": bars_identity(bars),
                       "train_indices_sha256": digest_indices(train),
                       "fit_indices_sha256": digest_indices(fit), "stop_indices_sha256": digest_indices(stop),
                       "validation_indices_sha256": digest_indices(validation), "actions": actions,
                       "source_sha256": sources,
                       "runtime": {"device": device, "python": sys.version.split()[0], "numpy": np.__version__,
                                   "pandas": pd.__version__, "torch": str(torch.__version__),
                                   "cuda": torch.version.cuda,
                                   "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None}}
    identity = hashlib.sha256(json.dumps(identity_fields, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode("utf8")).hexdigest()
    state_path, receipt_path = directory / "fold_state.json", directory / "fold_receipt.json"
    if resume:
        if not state_path.is_file():
            raise ValueError("No resumable fold state")
        manifest = json.loads(state_path.read_text(encoding="utf8"))
        if manifest.get("fold_identity") != identity:
            raise ValueError("Fold data, split, source, runtime or configuration changed")
        if receipt_path.is_file():
            saved_receipt = json.loads(receipt_path.read_text(encoding="utf8"))
            if saved_receipt.get("status") == "completed":
                if saved_receipt.get("fold_identity") != identity:
                    raise ValueError("Completed fold identity changed")
                from .verify_development import check_fold
                check_fold(directory)
                atomic_json(state_path, saved_receipt)
                return saved_receipt
        if manifest.get("status") not in ("running", "failed"):
            raise ValueError("Fold state cannot be resumed")
        manifest.pop("failure", None)
        manifest["status"] = "running"
        manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
    else:
        manifest = {"schema_version": 2, "status": "running", "phase": "development", "independent_test": False,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(), "config": asdict(cfg),
                    "data_identity": data_identity, "fold_identity": identity, "identity_fields": identity_fields,
                    "outer_train_origins": len(train), "fit_origins": len(fit), "stop_origins": len(stop),
                    "validation_origins": len(validation),
                    "Q3_values_read": False, "source_sha256": sources, "resume_count": 0,
                    "wall_time_policy": "persisted active time only; abrupt exit may undercount unfinished work",
                    "models": {}}
        atomic_json(state_path, manifest)
    started = time.monotonic()
    prior_wall = float(manifest.get("wall_seconds", 0.0))

    def persist_state():
        manifest["wall_seconds"] = prior_wall + time.monotonic() - started
        atomic_json(state_path, manifest)

    def artifact(path, expected):
        if not path.is_file() or file_digest(path) != expected:
            raise ValueError(f"Completed fold artifact identity changed: {path.name}")

    def checkpoint_dir(name, details):
        rel = details.get("checkpoint_dir", name)
        if not isinstance(rel, str) or Path(rel).name != rel:
            raise ValueError("Invalid checkpoint directory")
        if rel != name and not (rel.startswith(name + "-retry-") and rel[len(name) + 7:].isdigit()):
            raise ValueError("Unexpected checkpoint directory")
        return directory / rel

    def load_prediction(name):
        details = manifest["models"][name]
        path = directory / f"prediction_{name}.npz"
        artifact(path, details["prediction_sha256"])
        with np.load(path, allow_pickle=False) as stored:
            if set(stored.files) != {"prediction", "origins", "decision_ns"}:
                raise ValueError(f"Unexpected prediction fields for {name}")
            prediction = stored["prediction"]
            if (not np.array_equal(stored["origins"], validation)
                    or not np.array_equal(stored["decision_ns"], bars.end_ns[validation])):
                raise ValueError(f"Prediction origins differ for {name}")
        if prediction.shape != (len(validation), cfg.horizon) or not np.isfinite(prediction).all():
            raise ValueError(f"Invalid prediction output for {name}")
        if name == "ridge":
            artifact(directory / "ridge_coefficients.npz", details["coefficients_sha256"])
            artifact(directory / "ridge_scaler.json", details["scaler_sha256"])
        elif name in ("mlp", "gru", "lstm", "xpatch"):
            model_dir = checkpoint_dir(name, details)
            for filename, key in (("best.pt", "best_checkpoint_sha256"),
                                  ("latest.pt", "latest_checkpoint_sha256"),
                                  ("training_summary.json", "training_summary_sha256")):
                artifact(model_dir / filename, details[key])
        return prediction

    def save_prediction(name, prediction, details):
        if prediction.shape != (len(validation), cfg.horizon) or not np.isfinite(prediction).all():
            raise ValueError(f"Invalid prediction output for {name}")
        path = directory / f"prediction_{name}.npz"
        atomic_npz(path, prediction=prediction, origins=validation, decision_ns=bars.end_ns[validation])
        manifest["models"][name] = {"prediction_sha256": file_digest(path), **details}
        persist_state()

    def next_attempt(name):
        for index in range(1000):
            path = directory / (name if index == 0 else f"{name}-retry-{index:03d}")
            if not path.exists():
                return path, False
            if (path / "latest.pt").is_file():
                return path, True
        raise ValueError(f"Too many incomplete attempts for {name}")

    try:
        # A completed stage is reused only after its saved identity is verified.
        for name in manifest["models"]:
            if name not in MODELS:
                raise ValueError("Unexpected model in fold state")
            load_prediction(name)
        if "last_close" not in manifest["models"]:
            baseline = batched_baseline("last_close", windows, validation, cfg.horizon, cfg.target, cfg.seasonal_period)
            save_prediction("last_close", baseline, {"fit": "none"})
        baseline = load_prediction("last_close")
        for name in ("drift", "seasonal"):
            if name not in manifest["models"]:
                prediction = batched_baseline(name, windows, validation, cfg.horizon, cfg.target, cfg.seasonal_period)
                save_prediction(name, prediction, {"fit": "none", "period": cfg.seasonal_period if name == "seasonal" else None})
        if "ridge" not in manifest["models"]:
            scaler = fit_scaler(windows, fit, cfg.horizon, cfg.target)
            ridge_models = fit_many(windows, scaler, fit, cfg.horizon, cfg.target, cfg.ridge_alphas)
            stop_truth = targets(bars, stop, cfg.horizon, cfg.target)
            stop_losses = {}
            for alpha, model in ridge_models.items():
                stop_pred = model.predict(windows, scaler, stop)
                stop_losses[alpha] = float(np.square(stop_pred[:, -1] - stop_truth[:, -1]).mean())
            chosen = min(stop_losses, key=lambda alpha: (stop_losses[alpha], alpha))
            selected = ridge_models[chosen]
            atomic_npz(directory / "ridge_coefficients.npz", coefficients=selected.coefficients)
            atomic_json(directory / "ridge_scaler.json", asdict(scaler))
            save_prediction("ridge", selected.predict(windows, scaler, validation),
                            {"fit": "inner_only", "alpha": chosen, "inner_stop_mse_by_alpha": stop_losses,
                             "training_origins": selected.observations,
                             "coefficients_sha256": file_digest(directory / "ridge_coefficients.npz"),
                             "scaler_sha256": file_digest(directory / "ridge_scaler.json")})
        base = TrainConfig("mlp", cfg.lookback, cfg.horizon, cfg.target, cfg.max_epochs,
                           cfg.patience, cfg.batch_size, cfg.learning_rate, cfg.seed)
        training_data_identity = data_identity + ":" + cfg.protocol_identity + ":" + identity
        for name in ("mlp", "gru", "lstm", "xpatch"):
            if name in manifest["models"]:
                continue
            model_cfg = replace(base, model=name)
            model_dir, model_resume = next_attempt(name)
            expected_identity = training_identity(model_cfg, training_data_identity, fit, stop, device)
            summary_path = model_dir / "training_summary.json"
            trained = None
            if model_resume and summary_path.is_file():
                try:
                    trained = json.loads(summary_path.read_text(encoding="utf8"))
                except (ValueError, UnicodeError):
                    trained = None  # A torn summary can be rebuilt from the saved checkpoint.
                if trained is not None and (trained.get("status") != "completed" or trained.get("identity") != expected_identity):
                    raise ValueError(f"Training summary identity differs for {name}")
                if trained is not None:
                    latest = torch.load(model_dir / "latest.pt", map_location="cpu", weights_only=True)
                    state = latest.get("state", {})
                    if (latest.get("identity") != expected_identity or
                            any(trained.get(key) != state.get(state_key) for key, state_key in
                                (("epochs", "epoch"), ("steps", "steps"), ("best_epoch", "best_epoch"),
                                 ("history", "history"))) or "stale_epochs" not in state or
                            trained.get("stopped_early") != (state["stale_epochs"] >= model_cfg.patience)):
                        raise ValueError(f"Training checkpoint and summary disagree for {name}")
                    best = torch.load(model_dir / "best.pt", map_location="cpu", weights_only=True)
                    if best.get("identity") != expected_identity or best.get("epoch") != trained.get("best_epoch"):
                        raise ValueError(f"Best checkpoint and summary disagree for {name}")
            if trained is None:
                trained = run_fit(windows, fit, stop, model_cfg, model_dir, training_data_identity,
                                  device=device, resume=model_resume)
            if trained.get("status") != "completed" or trained.get("identity") != expected_identity:
                raise ValueError(f"Training did not complete with the expected identity for {name}")
            prediction = predict_best(model_dir / "best.pt", windows, validation, trained["identity"], device=device)
            save_prediction(name, prediction, {"fit": "inner_only", "epochs": trained["epochs"],
                                              "best_epoch": trained["best_epoch"], "steps": trained["steps"],
                                              "stopped_early": trained["stopped_early"],
                                              "training_identity": trained["identity"],
                                              "checkpoint_dir": model_dir.name,
                                              "best_checkpoint_sha256": file_digest(model_dir / "best.pt"),
                                              "latest_checkpoint_sha256": file_digest(model_dir / "latest.pt"),
                                              "training_summary_sha256": file_digest(summary_path)})
        if tuple(manifest["models"]) != MODELS:
            raise ValueError("Development fold did not produce all eight requested models")
        # Extract and score outer labels only after every model and inner selection is fixed.
        # Source-integrity hashing covers all bars but does not influence model updates or selection.
        truth = targets(bars, validation, cfg.horizon, cfg.target)
        atomic_npz(directory / "targets.npz", origins=validation, actual=truth, anchor=anchor,
                   decision_ns=bars.end_ns[validation], target_completion_ns=bars.end_ns[validation + cfg.horizon])
        for name in MODELS:
            prediction = load_prediction(name)
            comparison = endpoint_improvement(truth, prediction, baseline)
            comparison.pop("paired_loss_difference")
            manifest["models"][name].update(metrics=path_metrics(truth, prediction, cfg.target, anchor=anchor),
                                            against_last_close_or_zero=comparison)
        manifest.update(status="completed", wall_seconds=prior_wall + time.monotonic() - started,
                        targets_sha256=file_digest(directory / "targets.npz"))
        atomic_json(receipt_path, manifest)
        atomic_json(state_path, manifest)
        return manifest
    except Exception as exc:
        manifest.update(status="failed", failure=f"{type(exc).__name__}: {exc}",
                        wall_seconds=prior_wall + time.monotonic() - started)
        atomic_json(state_path, manifest)
        atomic_json(receipt_path, manifest)
        raise
