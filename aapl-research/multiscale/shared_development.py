"""Matched development comparison for minute shared versus separate horizons.

This does not create a protocol or start a matrix. Every model uses the same
H60-purged origins; results must remain development evidence until independently
confirmed. GPU time and parameter counts are recorded rather than assumed equal.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from .bars import Bars, file_digest, targets
from .development import validate_split
from .evaluation import endpoint_improvement, path_metrics
from .features import CloseWindows
from .models import make_model
from .training import TrainConfig, predict_best, run_fit


@dataclass(frozen=True)
class SharedConfig:
    lookback: int
    max_epochs: int
    patience: int
    batch_size: int
    learning_rate: float
    seed: int
    protocol_identity: str
    horizons: tuple[int, ...] = (5, 15, 30, 60)

    def validate(self):
        if self.horizons != (5, 15, 30, 60) or not self.protocol_identity:
            raise ValueError("The predefined H5/H15/H30/H60 mechanism and protocol identity are required")
        TrainConfig("xpatch", self.lookback, 60, "future_log_return", self.max_epochs,
                    self.patience, self.batch_size, self.learning_rate, self.seed,
                    shared_horizons=self.horizons).validate()


def run_shared_fold(bars: Bars, fold: dict, actions, cfg: SharedConfig, directory: Path,
                    data_identity: str, *, device="cpu"):
    cfg.validate()
    if bars.frequency != "minute" or not data_identity:
        raise ValueError("A minute series and verified data identity are required")
    _, fit, stop, validation = validate_split(bars, fold, 60)
    if not np.all(bars.session_date[validation] == bars.session_date[validation + 60]):
        raise ValueError("Minute mechanism paths cannot cross sessions")
    windows = CloseWindows(bars, cfg.lookback, actions)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    result = {"status": "running", "phase": "development", "independent_test": False,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "config": asdict(cfg),
              "data_identity": data_identity, "fit_origins": len(fit), "stop_origins": len(stop),
              "validation_origins": len(validation), "purge_horizon_for_every_model": 60,
              "Q3_values_read": False, "models": {}, "comparisons": {}}
    started = time.monotonic()
    try:
        for name, horizon, shared in [("independent_h5", 5, False), ("independent_h15", 15, False),
                                      ("independent_h30", 30, False), ("independent_h60", 60, False),
                                      ("shared_h60", 60, True)]:
            config = TrainConfig("xpatch", cfg.lookback, horizon, "future_log_return", cfg.max_epochs,
                                 cfg.patience, cfg.batch_size, cfg.learning_rate, cfg.seed,
                                 shared_horizons=cfg.horizons if shared else ())
            parameter_count = sum(p.numel() for p in make_model("xpatch", cfg.lookback, horizon).parameters())
            before = time.monotonic()
            trained = run_fit(windows, fit, stop, config, directory / name,
                              data_identity + ":" + cfg.protocol_identity, device=device)
            prediction = predict_best(directory / name / "best.pt", windows, validation,
                                      trained["identity"], device=device)
            path = directory / f"prediction_{name}.npz"
            np.savez_compressed(path, prediction=prediction)
            result["models"][name] = {"horizon": horizon, "shared": shared,
                                      "parameters": parameter_count, "epochs": trained["epochs"],
                                      "best_epoch": trained["best_epoch"], "steps": trained["steps"],
                                      "stopped_early": trained["stopped_early"],
                                      "training_and_inference_seconds": time.monotonic() - before,
                                      "prediction_sha256": file_digest(path),
                                      "training_identity": trained["identity"]}
        # Development labels are not read until every matched model is fixed.
        truth = targets(bars, validation, 60, "future_log_return")
        np.savez_compressed(directory / "targets.npz", origins=validation, actual=truth,
                            decision_ns=bars.end_ns[validation], target_completion_ns=bars.end_ns[validation + 60])
        with np.load(directory / "prediction_shared_h60.npz", allow_pickle=False) as source:
            shared_prediction = source["prediction"]
        for horizon in cfg.horizons:
            with np.load(directory / f"prediction_independent_h{horizon}.npz", allow_pickle=False) as source:
                individual = source["prediction"]
            shared = shared_prediction[:, :horizon]
            actual = truth[:, :horizon]
            paired = endpoint_improvement(actual, shared, individual)
            paired.pop("paired_loss_difference")
            result["comparisons"][str(horizon)] = {"shared": path_metrics(actual, shared, "future_log_return"),
                                                   "independent": path_metrics(actual, individual, "future_log_return"),
                                                   "shared_vs_independent": paired}
        result.update(status="completed", total_seconds=time.monotonic() - started,
                      targets_sha256=file_digest(directory / "targets.npz"))
        (directory / "fold_receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        return result
    except Exception as exc:
        result.update(status="failed", failure=f"{type(exc).__name__}: {exc}",
                      total_seconds=time.monotonic() - started)
        (directory / "fold_receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        raise
