"""Inner-early-stopped neural training with resumable batch order and RNG.

This reusable component does not select a dataset or start an experiment. The
caller must supply admitted bars, an already-purged inner fit/stop split, a data
identity and a separately approved training configuration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

from .bars import file_digest, targets
from .features import CloseWindows, Scaler, fit_scaler
from .mechanism import shared_path_loss
from .models import make_model


@dataclass(frozen=True)
class TrainConfig:
    model: str
    lookback: int
    horizon: int
    target: str
    max_epochs: int
    patience: int
    batch_size: int
    learning_rate: float
    seed: int
    checkpoint_batches: int = 100
    shared_horizons: tuple[int, ...] = ()

    def validate(self):
        if self.model not in ("mlp", "gru", "lstm", "xpatch"):
            raise ValueError("Unknown model")
        if self.target not in ("future_close", "future_log_return"):
            raise ValueError("Unknown target")
        if self.lookback < 2 or self.horizon < 1 or self.max_epochs < 1 or self.patience < 1:
            raise ValueError("Invalid training dimensions or stopping settings")
        if self.batch_size < 2 or self.learning_rate <= 0 or self.checkpoint_batches < 1:
            raise ValueError("Invalid training optimizer or batch settings")
        if self.shared_horizons and (self.model != "xpatch" or self.target != "future_log_return" or
                tuple(sorted(set(self.shared_horizons))) != self.shared_horizons or
                self.shared_horizons[-1] != self.horizon or self.shared_horizons[0] < 1):
            raise ValueError("Invalid shared-horizon xPatch configuration")


def digest_indices(values):
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def identity(config, data_identity, fit, stop, device):
    if not isinstance(data_identity, str) or not data_identity:
        raise ValueError("A verified data identity is required")
    local = Path(__file__).parent
    source_files = [Path(__file__), local / "models.py", local / "features.py", local / "bars.py", local / "mechanism.py",
                    local.parent / "aapl" / "models.py", local.parent / "aapl" / "vendor" / "xpatch_network.py"]
    value = {"config": asdict(config), "data_identity": data_identity,
             "fit_indices_sha256": digest_indices(fit), "stop_indices_sha256": digest_indices(stop),
             "runtime": {"device": device, "python": sys.version.split()[0], "numpy": np.__version__,
                         "torch": str(torch.__version__), "cuda": torch.version.cuda,
                         "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None},
             "source_sha256": {str(path.relative_to(local.parent)): file_digest(path) for path in source_files}}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_torch_save(value, path: Path):
    with tempfile.NamedTemporaryFile(prefix=".checkpoint-", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(value, stream)
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


def atomic_json_save(value, path: Path):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf8", prefix=".summary-", dir=path.parent,
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


def standardized_batch(windows, scaler, indices, horizon, target, device):
    x, anchor = windows.arrays(indices)
    x, anchor = scaler.transform_x(x, anchor)
    y = scaler.transform_y(targets(windows.bars, indices, horizon, target))
    return (torch.as_tensor(x[:, :, None], device=device),
            torch.as_tensor(anchor, device=device), torch.as_tensor(y, device=device))


def stopping_mse(model, windows, scaler, origins, config, device):
    """Select epochs using the inner stop tail's raw endpoint units only."""
    model.eval()
    loss_sum = 0.0
    with torch.inference_mode():
        for start in range(0, len(origins), config.batch_size):
            idx = origins[start:start + config.batch_size]
            x, anchor = windows.arrays(idx)
            x, anchor = scaler.transform_x(x, anchor)
            prediction = model(torch.as_tensor(x[:, :, None], device=device),
                               torch.as_tensor(anchor, device=device)).detach().cpu().numpy()
            raw = scaler.invert_y(prediction)
            truth = targets(windows.bars, idx, config.horizon, config.target)
            if raw.shape != truth.shape or not np.isfinite(raw).all():
                raise ValueError("Invalid stop prediction")
            if config.shared_horizons:
                hidx = np.asarray(config.shared_horizons) - 1
                scaled_error = (raw[:, hidx] - truth[:, hidx]) / np.asarray(scaler.y_std)[hidx]
                loss_sum += float(np.square(scaled_error).mean(axis=1).sum())
            else:
                loss_sum += float(np.square(raw[:, -1] - truth[:, -1]).sum())
    return loss_sum / len(origins)


def run_fit(windows: CloseWindows, fit_origins, stop_origins, config: TrainConfig,
            directory: Path, data_identity: str, *, device="cpu", resume=False,
            stop_after_steps=None):
    config.validate()
    if windows.lookback != config.lookback:
        raise ValueError("Feature window does not match model lookback")
    fit = np.asarray(fit_origins, dtype=np.int64)
    stop = np.asarray(stop_origins, dtype=np.int64)
    if len(fit) < 2 or not len(stop) or (np.diff(fit) <= 0).any() or (np.diff(stop) <= 0).any():
        raise ValueError("Fit/stop origins must be nonempty, unique and sorted")
    if (fit[0] < config.lookback - 1 or stop[0] < config.lookback - 1 or
            fit[-1] + config.horizon >= len(windows.bars.close) or
            stop[-1] + config.horizon >= len(windows.bars.close)):
        raise ValueError("Fit/stop windows exceed available history or target")
    if windows.bars.end_ns[fit[-1] + config.horizon] >= windows.bars.end_ns[stop[0]]:
        raise ValueError("Inner fit labels overlap the stopping period")
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested device is unavailable")
    if stop_after_steps is not None and stop_after_steps < 1:
        raise ValueError("stop_after_steps must be positive")
    digest = identity(config, data_identity, fit, stop, device)
    directory = Path(directory)
    if resume:
        if not (directory / "latest.pt").is_file():
            raise ValueError("No resumable checkpoint")
    else:
        directory.mkdir(parents=True, exist_ok=False)
    scaler = fit_scaler(windows, fit, config.horizon, config.target)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    model = make_model(config.model, config.lookback, config.horizon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    rng = np.random.default_rng(config.seed)
    state = {"epoch": 0, "cursor": 0, "permutation": None, "steps": 0,
             "best_loss": float("inf"), "best_epoch": None, "best_state": None,
             "stale_epochs": 0, "epoch_loss_sum": 0.0, "epoch_seen": 0, "history": []}
    if resume:
        saved = torch.load(directory / "latest.pt", map_location=device, weights_only=True)
        if saved["identity"] != digest or saved["scaler"] != asdict(scaler):
            raise ValueError("Checkpoint data, split, scaler or configuration changed")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        state = saved["state"]
        if state["permutation"] is not None:
            state["permutation"] = state["permutation"].cpu().numpy()
        rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"].cpu())

    def checkpoint():
        serial = dict(state)
        if serial["permutation"] is not None:
            serial["permutation"] = torch.as_tensor(serial["permutation"].copy(), dtype=torch.int64)
        atomic_torch_save({"schema": 1, "identity": digest, "scaler": asdict(scaler),
                           "state": serial, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                           "cuda_rng": torch.cuda.get_rng_state() if device == "cuda" else None},
                          directory / "latest.pt")

    # Persist the initial state before the first batch so an interrupted fold
    # can resume even if no periodic checkpoint has yet been reached.
    if not resume:
        checkpoint()

    steps_in_call = 0
    while state["epoch"] < config.max_epochs and state["stale_epochs"] < config.patience:
        if state["permutation"] is None:
            state["permutation"] = rng.permutation(len(fit))
            state["cursor"] = 0
        while state["cursor"] < len(fit):
            cursor = state["cursor"]
            end = min(len(fit), cursor + config.batch_size)
            if len(fit) - end == 1:
                end = len(fit)  # xPatch's batch norm cannot train on a lone residual sample.
            indices = fit[state["permutation"][cursor:end]]
            x, anchor, y = standardized_batch(windows, scaler, indices, config.horizon, config.target, device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, anchor)
            if pred.shape != y.shape:
                raise ValueError("Invalid training output shape")
            loss = shared_path_loss(pred, y, config.shared_horizons) if config.shared_horizons else (pred - y).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("Missing or nonfinite model gradients")
            optimizer.step()
            state["epoch_loss_sum"] += float(loss.detach().cpu()) * len(indices)
            state["epoch_seen"] += len(indices)
            state["cursor"] = end
            state["steps"] += 1
            steps_in_call += 1
            if state["steps"] % config.checkpoint_batches == 0:
                checkpoint()
            if stop_after_steps is not None and steps_in_call >= stop_after_steps:
                checkpoint()
                return {"status": "interrupted_for_test", "steps": state["steps"], "identity": digest}
        score = stopping_mse(model, windows, scaler, stop, config, device)
        if not np.isfinite(score):
            raise ValueError("Nonfinite stopping loss")
        state["history"].append({"epoch": state["epoch"] + 1, "training_scaled_path_loss":
                                 state["epoch_loss_sum"] / state["epoch_seen"],
                                 "inner_stop_metric": score, "steps": state["steps"]})
        if score < state["best_loss"]:
            state["best_loss"], state["best_epoch"], state["stale_epochs"] = score, state["epoch"] + 1, 0
            state["best_state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            atomic_torch_save({"identity": digest, "epoch": state["best_epoch"],
                               "model": state["best_state"], "scaler": asdict(scaler),
                               "config": asdict(config)}, directory / "best.pt")
        else:
            state["stale_epochs"] += 1
        state["epoch"] += 1
        state["cursor"], state["permutation"] = 0, None
        state["epoch_loss_sum"], state["epoch_seen"] = 0.0, 0
        checkpoint()
    result = {"status": "completed", "identity": digest, "epochs": state["epoch"],
              "steps": state["steps"], "best_epoch": state["best_epoch"],
              "inner_stop_metric": state["best_loss"],
              "inner_stop_definition": ("equal mean standardized endpoint MSE across shared horizons" if config.shared_horizons
                                        else "raw terminal endpoint MSE"), "history": state["history"],
              "stopped_early": state["stale_epochs"] >= config.patience}
    atomic_json_save(result, directory / "training_summary.json")
    return result


def predict_best(checkpoint: Path, windows: CloseWindows, origins, expected_identity: str,
                 *, device="cpu", batch_size=512):
    """Predict only from completed historical windows; future prices are not read."""
    if batch_size < 1:
        raise ValueError("Prediction batch size must be positive")
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    if saved["identity"] != expected_identity:
        raise ValueError("Checkpoint identity mismatch")
    config = TrainConfig(**saved["config"])
    config.validate()
    if windows.lookback != config.lookback:
        raise ValueError("Feature window does not match checkpoint")
    model = make_model(config.model, config.lookback, config.horizon).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    scaler = Scaler(**saved["scaler"])
    indices = np.asarray(origins, dtype=np.int64)
    out = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            x, anchor = windows.arrays(indices[start:start + batch_size])
            x, anchor = scaler.transform_x(x, anchor)
            prediction = model(torch.as_tensor(x[:, :, None], device=device),
                               torch.as_tensor(anchor, device=device)).detach().cpu().numpy()
            out.append(scaler.invert_y(prediction))
    result = np.concatenate(out) if out else np.empty((0, config.horizon))
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite model prediction")
    return result
