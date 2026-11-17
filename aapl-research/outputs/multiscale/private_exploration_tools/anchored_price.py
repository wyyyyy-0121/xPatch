"""Post-hoc private experiment: initialize a price model at persistence."""
from dataclasses import asdict

import numpy as np
import torch
from torch import nn

from multiscale.bars import targets
from multiscale.models import make_model
from multiscale.training import stopping_mse


class AnchoredPrice(nn.Module):
    """Predict standardized price as standardized anchor plus a learned residual."""
    def __init__(self, name, lookback, horizon, scaler):
        super().__init__()
        self.base = make_model(name, lookback, horizon)
        self.residual_head = nn.Linear(horizon, horizon)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        y_mean = np.asarray(scaler.y_mean, dtype=np.float64)
        y_std = np.asarray(scaler.y_std, dtype=np.float64)
        self.register_buffer("anchor_scale", torch.as_tensor(scaler.anchor_std / y_std, dtype=torch.float32))
        self.register_buffer("anchor_offset", torch.as_tensor((scaler.anchor_mean - y_mean) / y_std,
                                                               dtype=torch.float32))

    def forward(self, x, anchor):
        persistence = anchor * self.anchor_scale + self.anchor_offset
        return persistence + self.residual_head(self.base(x, anchor))


def predict(model, windows, scaler, indices, *, batch_size=128):
    model.eval(); values = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            part = indices[start:start + batch_size]
            x, a = windows.arrays(part); x, a = scaler.transform_x(x, a)
            pred = model(torch.as_tensor(x[:, :, None]), torch.as_tensor(a)).cpu().numpy()
            raw = scaler.invert_y(pred)
            if not np.isfinite(raw).all():
                raise ValueError("Anchored model produced nonfinite predictions")
            values.append(raw)
    return np.concatenate(values)


def fit(model_name, windows, scaler, fit_indices, stop_indices, config):
    """Train the same price objective, including persistence as epoch zero."""
    torch.manual_seed(config.seed)
    model = AnchoredPrice(model_name, config.lookback, config.horizon, scaler)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    rng = np.random.default_rng(config.seed)
    initial = predict(model, windows, scaler, stop_indices[:min(16, len(stop_indices))])
    anchor = windows.bars.close[stop_indices[:len(initial)]]
    if not np.allclose(initial, anchor[:, None], rtol=2e-5, atol=2e-5):
        raise ValueError("The zero-initialized head does not reproduce persistence")
    best_loss = float(stopping_mse(model, windows, scaler, stop_indices, config, "cpu"))
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history = [{"epoch":0,"inner_stop_endpoint_mse":best_loss}]
    stale = 0; steps = 0
    for epoch in range(1, config.max_epochs + 1):
        order = rng.permutation(len(fit_indices)); cursor = 0; sum_loss = 0.; seen = 0
        while cursor < len(order):
            end = min(len(order), cursor + config.batch_size)
            if len(order) - end == 1:
                end = len(order)
            indices = fit_indices[order[cursor:end]]
            x, a = windows.arrays(indices); x, a = scaler.transform_x(x, a)
            y = scaler.transform_y(targets(windows.bars, indices, config.horizon, config.target))
            model.train(); optimizer.zero_grad(set_to_none=True)
            predicted = model(torch.as_tensor(x[:, :, None]), torch.as_tensor(a))
            loss = (predicted - torch.as_tensor(y)).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite anchored training loss")
            loss.backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("Missing or nonfinite anchored gradients")
            optimizer.step(); steps += 1
            sum_loss += float(loss.detach()) * len(indices); seen += len(indices); cursor = end
        score = float(stopping_mse(model, windows, scaler, stop_indices, config, "cpu"))
        if not np.isfinite(score):
            raise ValueError("Nonfinite anchored stopping loss")
        history.append({"epoch":epoch,"training_scaled_path_mse":sum_loss/seen,"inner_stop_endpoint_mse":score})
        if score < best_loss:
            best_loss = score; best_epoch = epoch; stale = 0
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    return model, {"epochs":len(history)-1,"best_epoch":best_epoch,"steps":steps,
                   "stopped_early":stale>=config.patience,"history":history,
                   "parameter_count":sum(p.numel() for p in model.parameters()),
                   "extra_residual_head_parameters":sum(p.numel() for p in model.residual_head.parameters()),
                   "price_scaler":asdict(scaler)}
