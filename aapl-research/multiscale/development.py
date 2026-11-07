"""Execute one chronological development fold, never an independent test.

The caller must separately freeze an admissible protocol, source identity and
resource budget. This component does not select data sources or rent resources.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .bars import Bars, file_digest, targets
from .baselines import predict as baseline_predict
from .evaluation import endpoint_improvement, path_metrics
from .features import CloseWindows, fit_scaler
from .ridge import fit_many
from .spec import DEV_END, MODELS, NY
from .training import TrainConfig, predict_best, run_fit


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


def run_development_fold(bars: Bars, fold: dict, actions, cfg: FoldConfig, directory: Path,
                         data_identity: str, *, device="cpu"):
    """Fit and score exactly one development fold with inner-only selection."""
    cfg.validate()
    if not data_identity:
        raise ValueError("Verified data identity is required")
    _, fit, stop, validation = validate_split(bars, fold, cfg.horizon)
    windows = CloseWindows(bars, cfg.lookback, actions)
    future_idx = validation[:, None] + np.arange(1, cfg.horizon + 1)
    if not bars.observed[validation].all() or not bars.observed[future_idx].all():
        raise ValueError("Validation path has an unobserved bar")
    if bars.frequency == "minute" and not np.all(bars.session_date[validation] == bars.session_date[validation + cfg.horizon]):
        raise ValueError("Minute validation path crosses sessions")
    anchor = bars.close[validation]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    local = Path(__file__).parent
    source_names = ("development.py", "bars.py", "baselines.py", "evaluation.py", "features.py",
                    "models.py", "ridge.py", "splits.py", "spec.py", "training.py", "mechanism.py")
    source_paths = [local / name for name in source_names] + [local.parent / "aapl" / "models.py",
                                                              local.parent / "aapl" / "vendor" / "xpatch_network.py"]
    manifest = {"schema_version": 2, "status": "running", "phase": "development", "independent_test": False,
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "config": asdict(cfg),
                "data_identity": data_identity, "fit_origins": len(fit), "stop_origins": len(stop),
                "validation_origins": len(validation), "Q3_values_read": False,
                "source_sha256": {str(path.relative_to(local.parent)): file_digest(path) for path in source_paths},
                "models": {}}
    try:
        baseline = batched_baseline("last_close", windows, validation, cfg.horizon, cfg.target, cfg.seasonal_period)

        def save_prediction(name, prediction, details):
            if prediction.shape != (len(validation), cfg.horizon) or not np.isfinite(prediction).all():
                raise ValueError(f"Invalid prediction output for {name}")
            path = directory / f"prediction_{name}.npz"
            np.savez_compressed(path, prediction=prediction, origins=validation,
                                decision_ns=bars.end_ns[validation])
            manifest["models"][name] = {"prediction_sha256": file_digest(path), **details}

        save_prediction("last_close", baseline, {"fit": "none"})
        for name in ("drift", "seasonal"):
            prediction = batched_baseline(name, windows, validation, cfg.horizon, cfg.target, cfg.seasonal_period)
            save_prediction(name, prediction, {"fit": "none", "period": cfg.seasonal_period if name == "seasonal" else None})
        scaler = fit_scaler(windows, fit, cfg.horizon, cfg.target)
        ridge_models = fit_many(windows, scaler, fit, cfg.horizon, cfg.target, cfg.ridge_alphas)
        stop_truth = targets(bars, stop, cfg.horizon, cfg.target)
        stop_losses = {}
        for alpha, model in ridge_models.items():
            stop_pred = model.predict(windows, scaler, stop)
            stop_losses[alpha] = float(np.square(stop_pred[:, -1] - stop_truth[:, -1]).mean())
        chosen = min(stop_losses, key=lambda alpha: (stop_losses[alpha], alpha))
        selected = ridge_models[chosen]
        np.savez_compressed(directory / "ridge_coefficients.npz", coefficients=selected.coefficients)
        (directory / "ridge_scaler.json").write_text(json.dumps(asdict(scaler), indent=2), encoding="utf-8")
        save_prediction("ridge", selected.predict(windows, scaler, validation),
                        {"fit": "inner_only", "alpha": chosen, "inner_stop_mse_by_alpha": stop_losses,
                         "training_origins": selected.observations,
                         "coefficients_sha256": file_digest(directory / "ridge_coefficients.npz"),
                         "scaler_sha256": file_digest(directory / "ridge_scaler.json")})
        base = TrainConfig("mlp", cfg.lookback, cfg.horizon, cfg.target, cfg.max_epochs,
                           cfg.patience, cfg.batch_size, cfg.learning_rate, cfg.seed)
        for name in ("mlp", "gru", "lstm", "xpatch"):
            trained = run_fit(windows, fit, stop, replace(base, model=name), directory / name,
                              data_identity + ":" + cfg.protocol_identity, device=device)
            prediction = predict_best(directory / name / "best.pt", windows, validation,
                                      trained["identity"], device=device)
            save_prediction(name, prediction, {"fit": "inner_only", "epochs": trained["epochs"],
                                              "best_epoch": trained["best_epoch"], "steps": trained["steps"],
                                              "stopped_early": trained["stopped_early"],
                                              "training_identity": trained["identity"],
                                              "best_checkpoint_sha256": file_digest(directory / name / "best.pt"),
                                              "latest_checkpoint_sha256": file_digest(directory / name / "latest.pt"),
                                              "training_summary_sha256": file_digest(directory / name / "training_summary.json")})
        if tuple(manifest["models"]) != MODELS:
            raise ValueError("Development fold did not produce all eight requested models")
        # Only after every model and inner selection is fixed are outer labels read and scored.
        truth = targets(bars, validation, cfg.horizon, cfg.target)
        np.savez_compressed(directory / "targets.npz", origins=validation, actual=truth, anchor=anchor,
                            decision_ns=bars.end_ns[validation], target_completion_ns=bars.end_ns[validation + cfg.horizon])
        for name in MODELS:
            with np.load(directory / f"prediction_{name}.npz", allow_pickle=False) as saved:
                prediction = saved["prediction"]
            comparison = endpoint_improvement(truth, prediction, baseline)
            comparison.pop("paired_loss_difference")
            manifest["models"][name].update(metrics=path_metrics(truth, prediction, cfg.target, anchor=anchor),
                                            against_last_close_or_zero=comparison)
        manifest.update(status="completed", wall_seconds=time.monotonic() - started,
                        targets_sha256=file_digest(directory / "targets.npz"))
        (directory / "fold_receipt.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
        return manifest
    except Exception as exc:
        manifest.update(status="failed", failure=f"{type(exc).__name__}: {exc}",
                        wall_seconds=time.monotonic() - started)
        (directory / "fold_receipt.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
        raise
