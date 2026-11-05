"""Regularized stacking and foundation-conditioned mixtures, fitted on OOF data."""
import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn


def design(data, basis):
    experts = data['experts']
    if basis == 'static':
        return experts
    columns = [experts, data['features'],
               experts.std(axis=1, keepdims=True), np.abs(experts).mean(axis=1, keepdims=True)]
    if basis == 'foundation':
        columns.append(data['embedding'])
    elif basis != 'without_foundation':
        raise ValueError(basis)
    return np.concatenate(columns, axis=1)


class StateGate(nn.Module):
    def __init__(self, inputs, experts, hidden, residual_cap):
        super().__init__()
        self.context = nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh())
        self.weights = nn.Linear(hidden, experts + 1)
        self.residual = nn.Linear(hidden, 1)
        self.residual_cap = residual_cap
        nn.init.zeros_(self.weights.weight)
        nn.init.zeros_(self.weights.bias)
        with torch.no_grad():
            self.weights.bias[-1] = 1.5
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, x, experts, volatility):
        h = self.context(x)
        weights = self.weights(h).softmax(dim=1)
        choices = torch.cat([experts, torch.zeros_like(experts[:, :1])], dim=1)
        pred = (weights * choices).sum(dim=1)
        residual = self.residual_cap * volatility * torch.tanh(self.residual(h)[:, 0])
        return pred + residual


def fit(config, data, seed=2026):
    x = design(data, config['basis']).astype(np.float64)
    center, std = x.mean(axis=0), x.std(axis=0)
    std[std < 1e-9] = 1.
    x = np.clip((x - center) / std, -10, 10).astype(np.float32)
    target_scale = max(float(np.sqrt(np.mean(data['y'] ** 2))), 1e-6)
    state = {'config': config, 'seed': seed, 'center': center, 'std': std,
             'target_scale': target_scale, 'input_columns': x.shape[1],
             'expert_columns': data['experts'].shape[1]}
    if config['kind'] == 'ridge':
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=config['alpha'], fit_intercept=False, solver='cholesky')
        model.fit(x, data['y'] / target_scale)
        state['coef'] = model.coef_.astype(np.float64)
        return state
    torch.manual_seed(seed)
    net = StateGate(x.shape[1], data['experts'].shape[1], config['hidden'], config['residual_cap'])
    optimizer = torch.optim.AdamW(net.parameters(), lr=config['lr'], weight_decay=config['decay'])
    tensor = torch.from_numpy(x)
    experts = torch.tensor(data['experts'] / target_scale, dtype=torch.float32)
    volatility = torch.tensor(data['volatility'] / target_scale, dtype=torch.float32)
    y = torch.tensor(data['y'] / target_scale, dtype=torch.float32)
    # Whole fixed epochs; no confirmation labels or evaluation-driven stopping.
    generator = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(config['epochs']):
        order = torch.randperm(len(x), generator=generator)
        total = 0.
        for batch in order.split(1024):
            optimizer.zero_grad(set_to_none=True)
            pred = net(tensor[batch], experts[batch], volatility[batch])
            loss = (pred - y[batch]).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Fusion training diverged')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        history.append(total / len(x))
    state['weights'] = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    state['training_loss'] = history
    return state


def predict(state, data):
    x = design(data, state['config']['basis']).astype(np.float64)
    x = np.clip((x - state['center']) / state['std'], -10, 10).astype(np.float32)
    if x.shape[1] != state['input_columns']:
        raise ValueError('Fusion input schema changed')
    if state['config']['kind'] == 'ridge':
        return (x @ state['coef']) * state['target_scale']
    net = StateGate(state['input_columns'], state['expert_columns'],
                    state['config']['hidden'], state['config']['residual_cap'])
    net.load_state_dict(state['weights'])
    net.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(x), 4096):
            end = start + 4096
            pred = net(torch.from_numpy(x[start:end]),
                       torch.tensor(data['experts'][start:end] / state['target_scale'], dtype=torch.float32),
                       torch.tensor(data['volatility'][start:end] / state['target_scale'], dtype=torch.float32))
            predictions.append(pred.numpy().astype(np.float64) * state['target_scale'])
    return np.concatenate(predictions)


def save(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(state, tmp)
    tmp.replace(path)


def load(path):
    # Only load locally generated, manifest-verified fusion artifacts.
    return torch.load(path, map_location='cpu', weights_only=False)
