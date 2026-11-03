"""Six RV adapters; fitted state, model checkpoints and calibration stay separate."""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .data import ASSETS
from .features import FLOOR
from .io import ROOT, read_json, sha, write_json


def seed_torch(seed):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@dataclass
class LinearModel:
    center: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float
    feature_names: list

    @classmethod
    def fit(cls, x, y, alpha, names):
        from sklearn.linear_model import Ridge
        center, scale = x.mean(axis=0), x.std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.)
        model = Ridge(alpha=alpha, solver="svd").fit((x - center) / scale, y)
        return cls(center, scale, model.coef_, float(model.intercept_), list(names))

    def predict(self, x, names):
        if list(names) != self.feature_names:
            raise ValueError("Feature identity/order differs from model fitting")
        return (x - self.center) / self.scale @ self.coefficient + self.intercept

    def save(self, path):
        np.savez_compressed(path, center=self.center, scale=self.scale, coefficient=self.coefficient,
            intercept=self.intercept, feature_names=np.asarray(self.feature_names))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as saved:
            return cls(saved["center"], saved["scale"], saved["coefficient"], float(saved["intercept"]), saved["feature_names"].tolist())


def sequence_inputs(data, state, origins, length, representation, panel, config):
    assets = [ASSETS.index(asset) for asset in config["feature_sets"][panel]]
    values = state.series(data, representation, standardized=True)
    return data.context(origins, length, values)[:, :, assets].astype(np.float32)


def sequence_labels(data, state, origins, representation, panel, config):
    """31 steps from u-1 through u+30; the final step is the RV30 target."""
    origins = np.asarray(origins, dtype=np.int64)
    indices = origins[:, None] + np.arange(31)
    if not len(origins) or (indices >= len(data.timestamps)).any():
        raise ValueError("Sequence labels outside data")
    if not np.all(data.timestamps[indices[:, -1]] == data.timestamps[origins] + pd.Timedelta(minutes=30)):
        raise ValueError("Sequence label crosses a nontrading gap")
    if not np.all(data.dates[indices] == data.dates[origins, None]):
        raise ValueError("Sequence training target crosses a session")
    assets = [ASSETS.index(asset) for asset in config["feature_sets"][panel]]
    values = state.series(data, representation, standardized=True)[indices][:, :, assets]
    return values.astype(np.float32)


def native_xpatch(params, channels, device):
    import torch
    from models.xPatch import Model
    arguments = {key: params[key] for key in ("seq_len", "pred_len", "patch_len", "stride", "ma_type", "revin")}
    # The signed original DEMA constructor calls scalar.to('cuda') directly.
    # Wrap only its scalar parameters; the original recurrence is unchanged.
    if params["ma_type"] == "dema":
        if not torch.cuda.is_available():
            raise RuntimeError("The preserved native DEMA constructor requires CUDA")
        arguments.update(alpha=torch.tensor(params["alpha"], device="cuda"), beta=torch.tensor(params["beta"], device="cuda"))
    else:
        arguments.update(alpha=params["alpha"], beta=params["beta"])
    model = Model(SimpleNamespace(**arguments, enc_in=channels, padding_patch="end")).to(device)
    if params["ma_type"] == "dema":
        # Scalars in the original DEMA are attributes rather than buffers.
        model.decomp.ma.alpha = model.decomp.ma.alpha.to(device)
        model.decomp.ma.beta = model.decomp.ma.beta.to(device)
    if not params["revin"]:
        # Unused parameters remain in the original module/state for source fidelity.
        for parameter in model.revin_layer.parameters():
            parameter.requires_grad_(False)
    return model


def neural_values(model, inputs, batch_size, device):
    import torch
    from .cuda_graph import ForwardCache
    outputs = []
    model.eval()
    graph_forward = None
    if model.ma_type == "dema" and device.startswith("cuda"):
        if not hasattr(model, "_inference_graphs"):
            model._inference_graphs = ForwardCache(model)
        graph_forward = model._inference_graphs
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            value = torch.as_tensor(inputs[start:start + batch_size], device=device)
            output = model(value) if graph_forward is None else graph_forward.run(value)
            outputs.append(output.cpu().numpy())
    values = np.concatenate(outputs).astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite xPatch output; no forecast rows may be deleted")
    return values


def fit_neural(data, store, state, partition, origins, params, panel, config, seed, directory,
        max_epochs, fixed_epochs=None, device="cuda", resume=False):
    """Halving resumes the exact optimizer/RNG; outer fitting uses fixed epochs."""
    import torch
    from torch import nn
    seed_torch(seed)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "model.pt"
    representation, batch = params["representation"], params["batch_size"]
    x = sequence_inputs(data, state, origins, params["seq_len"], representation, panel, config)
    y = sequence_labels(data, state, origins, representation, panel, config)
    valid = np.isfinite(y).all(axis=(1, 2))
    excluded = int((~valid).sum())
    # Auxiliary future missingness affects training only, never evaluation origins.
    x, y = x[valid], y[valid]
    if len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Insufficient finite sequence training data")
    channels = x.shape[-1]
    model = native_xpatch(params, channels, device)
    from .cuda_graph import StepCache
    graph_steps = StepCache(model) if params["ma_type"] == "dema" and device.startswith("cuda") else None
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
        lr=params["learning_rate"], weight_decay=params["weight_decay"])
    if fixed_epochs is None:
        validation = data.origins(partition.evaluation_dates, stride=config["sampling"]["inner_validation_stride_minutes"])
        vx = sequence_inputs(data, state, validation, params["seq_len"], representation, panel, config)
        vy = sequence_labels(data, state, validation, representation, panel, config)[:, -1, 0]
        if not np.isfinite(vy).all():
            raise ValueError("AAPL validation target must be complete")
    generator = np.random.default_rng(seed)
    completed, best_epoch, bad, best_loss, history = 0, 0, 0, float("inf"), []
    best_weights = None
    if resume and checkpoint.exists():
        previous = torch.load(checkpoint, map_location=device, weights_only=False)
        if previous["params"] != params or previous["seed"] != seed or previous["fit_dates"] != list(partition.fit_dates):
            raise ValueError("Halving checkpoint identity changed")
        model.load_state_dict(previous["last_state"])
        optimizer.load_state_dict(previous["optimizer"])
        completed, best_epoch, bad = previous["completed_epochs"], previous["best_epoch"], previous["bad_epochs"]
        best_loss, history, best_weights = previous["best_loss"], previous["history"], previous["state_dict"]
        generator.bit_generator.state = previous["numpy_generator"]
        torch.set_rng_state(previous["torch_rng"].cpu())
        if device.startswith("cuda"):
            torch.cuda.set_rng_state_all([v.cpu() for v in previous["cuda_rng"]])
    limit = fixed_epochs if fixed_epochs is not None else max_epochs
    patience = config["search"]["xpatch"]["patience"]
    for epoch in range(completed + 1, limit + 1):
        if fixed_epochs is None and bad >= patience:
            break
        model.train()
        order = generator.permutation(len(x))
        total, count = 0., 0
        for start in range(0, len(x), batch):
            indices = order[start:start + batch]
            if len(indices) < 2:
                # Keep every sample even when BatchNorm would see a singleton.
                model.eval()
            optimizer.zero_grad(set_to_none=True)
            inputs = torch.as_tensor(x[indices], device=device)
            targets = torch.as_tensor(y[indices], device=device)
            if graph_steps is None:
                prediction = model(inputs)
                loss = nn.functional.mse_loss(prediction, targets)
            else:
                prediction, loss = graph_steps.run(inputs, targets)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite neural training loss")
            if graph_steps is None:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
            count += len(indices)
            model.train()
        validation_loss = None if fixed_epochs is not None else float(np.mean(
            (neural_values(model, vx, batch, device)[:, -1, 0] - vy) ** 2))
        history.append({"epoch": epoch, "train_mse": total / count, "validation_mse": validation_loss})
        completed = epoch
        improved = fixed_epochs is not None or validation_loss < best_loss - 1e-8
        if improved:
            best_loss = total / count if fixed_epochs is not None else validation_loss
            best_epoch, bad = epoch, 0
            best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            bad += 1
        payload = {"params": params, "seed": seed, "channels": channels, "fit_dates": list(partition.fit_dates),
            "completed_epochs": completed, "best_epoch": best_epoch, "bad_epochs": bad, "best_loss": best_loss,
            "history": history, "state_dict": best_weights,
            "last_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "optimizer": optimizer.state_dict(), "numpy_generator": generator.bit_generator.state,
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else []}
        temporary = checkpoint.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(checkpoint)
    if best_weights is None:
        raise RuntimeError("No trained checkpoint was produced")
    model.load_state_dict(best_weights)
    return model, {"completed_epochs": completed, "best_epoch": best_epoch, "early_stopped": bad >= patience,
        "excluded_auxiliary_training_origins": excluded, "training_origins": len(x), "checkpoint": "model.pt"}


class ChronosModel:
    def __init__(self, config, device="cuda"):
        from chronos import Chronos2Pipeline
        entry = read_json(ROOT / config["model_lock"])["models"]["chronos2"]
        for filename, evidence in entry["files"].items():
            if sha(ROOT / entry["directory"] / filename) != evidence["sha256"]:
                raise ValueError("Pinned Chronos file changed")
        self.pipeline = Chronos2Pipeline.from_pretrained(str(ROOT / entry["directory"]), device_map=device)
        self.config = config

    def predict(self, data, store, state, origins, params, panel):
        import torch
        length, representation = params["context_length"], params["representation"]
        batch_size = self.config["search"]["chronos2"]["batch_size"]
        series = state.series(data, representation, standardized=True)
        means = []
        for start in range(0, len(origins), batch_size):
            batch = origins[start:start + batch_size]
            context = data.context(batch, length, series).astype(np.float32)
            inputs = []
            # Every origin is one independent task, with covariates confined to it.
            for row in context:
                entry = {"target": row[:, 0]}
                if panel == "market":
                    entry["past_covariates"] = {asset: row[:, ASSETS.index(asset)] for asset in ("SPY", "QQQ")}
                inputs.append(entry)
            with torch.inference_mode():
                # This pinned API returns its median in the variable named "mean".
                # Treat it as a log-scale location; the disjoint smear yields RV mean.
                _, point = self.pipeline.predict_quantiles(inputs, prediction_length=31, quantile_levels=[.1, .5, .9],
                    context_length=length, batch_size=batch_size, cross_learning=False)
            means.extend(float(item[0, -1].cpu()) for item in point)
        transformed = np.asarray(means) * state.series_std[representation][0] + state.series_center[representation][0]
        return state.inverse(data, origins, transformed, representation)


def fit_tabular(name, data, store, state, partition, origins, params, panel, config, seed,
        directory, threads, fixed_iterations=None):
    x, names = store.matrix(origins, state, panel, name, params)
    y = state.labels(data, origins, params["representation"])
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if name in ("har", "ridge"):
        model = LinearModel.fit(x, y, params["alpha"], names)
        model.save(directory / "model.npz")
        return model, {"training_origins": len(x)}
    import lightgbm as lgb
    settings = config["search"]["lightgbm"]
    options = {k: params[k] for k in ("num_leaves", "max_depth", "min_child_samples", "learning_rate",
        "feature_fraction", "bagging_fraction", "lambda_l1", "lambda_l2")}
    model = lgb.LGBMRegressor(**options, n_estimators=fixed_iterations or params["max_estimators"],
        bagging_freq=1, random_state=seed, n_jobs=threads, verbosity=-1, deterministic=True, force_col_wise=True)
    fit_options = {}
    if fixed_iterations is None:
        validation = data.origins(partition.evaluation_dates, stride=config["sampling"]["inner_validation_stride_minutes"])
        vx, vn = store.matrix(validation, state, panel, name, params)
        if names != vn:
            raise ValueError("Validation feature identity differs")
        fit_options = {"eval_set": [(vx, state.labels(data, validation, params["representation"]))],
            "callbacks": [lgb.early_stopping(settings["early_stopping_rounds"], verbose=False)]}
    model.fit(x, y, **fit_options)
    model.booster_.save_model(str(directory / "model.txt"))
    write_json(directory / "features.json", names)
    model.booster_._rv_feature_order = names
    return model.booster_, {"best_iteration": int(model.best_iteration_ or model.n_estimators), "training_origins": len(x)}


def raw_predictions(name, model, data, store, state, origins, params, panel, config, device="cuda"):
    origins = np.asarray(origins, dtype=np.int64)
    representation = params["representation"]
    if name == "last_rv":
        values = data.rv(30)[origins - 1, 0]
        return np.maximum(values / state.expected(data, 30)[origins - 1, 0] * state.future_season(data, origins), FLOOR)
    if name == "chronos2":
        return model.predict(data, store, state, origins, params, panel)
    if name == "xpatch":
        x = sequence_inputs(data, state, origins, params["seq_len"], representation, panel, config)
        values = neural_values(model, x, params["batch_size"], device)[:, -1, 0]
        values = values * state.series_std[representation][0] + state.series_center[representation][0]
    else:
        x, names = store.matrix(origins, state, panel, name, params)
        if name == "lightgbm" and getattr(model, "_rv_feature_order", None) != names:
            raise ValueError("LightGBM feature order differs from fitting")
        values = model.predict(x, names) if isinstance(model, LinearModel) else model.predict(x)
    return state.inverse(data, origins, values, representation)


def load_model(name, directory, params, config, device="cuda", chronos=None):
    directory = Path(directory)
    if name == "last_rv":
        return None
    if name in ("har", "ridge"):
        return LinearModel.load(directory / "model.npz")
    if name == "lightgbm":
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(directory / "model.txt"))
        model._rv_feature_order = read_json(directory / "features.json")
        return model
    if name == "chronos2":
        return chronos or ChronosModel(config, device)
    if name == "xpatch":
        import torch
        saved = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
        model = native_xpatch(params, saved["channels"], device)
        model.load_state_dict(saved["state_dict"])
        return model.eval()
    raise ValueError(name)
