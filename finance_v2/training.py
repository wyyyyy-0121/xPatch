"""Train fixed baselines and native xPatch; labels never enter pretrained inference."""
import random
import time

import numpy as np
import torch

from .io import write_json
from .models import XPatch


def setup_runtime(config):
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(config["runtime"]["torch_threads"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return "cuda" if torch.cuda.is_available() and config["runtime"]["device"] != "cpu" else "cpu"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


@torch.inference_mode()
def neural_predictions(model, x, batch_size, target_std):
    model.eval()
    out = []
    device = next(model.parameters()).device
    for start in range(0, len(x), batch_size):
        batch = torch.as_tensor(x[start:start + batch_size], device=device)
        out.append(model(batch).cpu().numpy().astype(np.float64) * target_std)
    return np.concatenate(out)


def train_neural(name, data, fit_origins, selection_origins, scale, config, seed, directory,
                 fixed_epochs=None, phase="selection"):
    directory.mkdir(parents=True, exist_ok=True)
    device = setup_runtime(config)
    seed_everything(seed)
    x = torch.from_numpy(data.inputs(fit_origins, scale)).to(device)
    y = torch.from_numpy((data.step_targets(fit_origins) / scale["target_std"]).astype(np.float32)).to(device)
    selection_x = data.inputs(selection_origins, scale) if selection_origins is not None else None
    selection_y = data.targets(selection_origins)[:, -1] if selection_origins is not None else None
    model = XPatch(name, data.features.shape[1], max(data.horizons)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    batch_size = config["training"]["batch_size"]
    max_epochs = fixed_epochs or config["training"]["max_epochs"]
    best, best_epoch, bad = float("inf"), 0, 0
    history = []
    started = time.perf_counter()
    checkpoint = directory / f"{phase}.pt"
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(x), device=device)
        total = 0.0
        for start in range(0, len(x), batch_size):
            ids = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            pred = model(x[ids])
            loss = ((pred - y[ids]) ** 2).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss: {name} epoch {epoch}")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        item = {"epoch": epoch, "train_scaled_mse": total / len(x)}
        if fixed_epochs is None:
            pred = neural_predictions(model, selection_x, config["runtime"]["batch_size_inference"],
                                      scale["target_std"]).sum(axis=1)
            score = float(np.mean((pred - selection_y) ** 2))
            item["inner_selection_cumulative12_mse"] = score
            improved = score < best
            if improved:
                best, best_epoch, bad = score, epoch, 0
            else:
                bad += 1
        else:
            improved = epoch == max_epochs
            best_epoch = epoch
        if improved:
            torch.save({"model": name, "channels": data.features.shape[1], "horizon": max(data.horizons),
                        "seed": seed, "epoch": best_epoch, "scaler": scale,
                        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}, checkpoint)
        history.append(item)
        write_json(directory / f"{phase}_history.json", history)
        print(f"{directory} {phase} epoch={epoch} {item}", flush=True)
        if fixed_epochs is None and bad >= config["training"]["patience"]:
            break
    synchronize(device)
    elapsed = time.perf_counter() - started
    del model, optimizer, x, y
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"epoch": best_epoch, "epochs_run": len(history), "training_seconds": elapsed,
            "checkpoint": checkpoint.name}


def load_neural(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = XPatch(payload["model"], payload["channels"], payload["horizon"])
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), payload


def fit_baseline(name, data, origins, scale, config, directory):
    x = data.inputs(origins, scale).reshape(len(origins), -1).astype(np.float64)
    y = data.step_targets(origins) / scale["target_std"]
    started = time.perf_counter()
    if name == "ridge":
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=config["training"]["ridge_alpha"])
        model.fit(x, y)
        np.savez_compressed(directory / "ridge.npz", coef=model.coef_, intercept=model.intercept_)
        metadata = {"parameters": model.coef_.size + model.intercept_.size}
    elif name == "lightgbm":
        import lightgbm as lgb
        settings = config["training"]["lightgbm"]
        for j in range(y.shape[1]):
            model = lgb.LGBMRegressor(**settings, n_jobs=config["runtime"]["torch_threads"],
                                      random_state=2026, deterministic=True, force_col_wise=True)
            model.fit(x, y[:, j])
            model.booster_.save_model(str(directory / f"lightgbm_{j}.txt"))
        metadata = {"models": y.shape[1], "n_estimators": settings["n_estimators"]}
    else:
        raise ValueError(name)
    metadata["training_seconds"] = time.perf_counter() - started
    return metadata


def predict_baseline(name, data, origins, scale, directory):
    if name == "zero":
        return np.zeros((len(origins), len(data.horizons)))
    x = data.inputs(origins, scale).reshape(len(origins), -1).astype(np.float64)
    if name == "ridge":
        with np.load(directory / "ridge.npz") as state:
            steps = x @ state["coef"].T + state["intercept"]
    elif name == "lightgbm":
        import lightgbm as lgb
        steps = np.column_stack([lgb.Booster(model_file=str(directory / f"lightgbm_{j}.txt"))
                                .predict(x, num_threads=4) for j in range(max(data.horizons))])
    else:
        raise ValueError(name)
    return np.cumsum(steps * scale["target_std"], axis=1)[:, np.asarray(data.horizons) - 1]
