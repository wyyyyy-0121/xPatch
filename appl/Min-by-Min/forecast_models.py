"""Serializable model factory; all estimators share fold preprocessing."""
from __future__ import annotations

import numpy as np

FEATURE_ORDER = ['log_open_close', 'log_high_close', 'log_low_close',
                 'one_bar_log_return', 'log1p_volume']
BASELINES = {'last_close', 'drift', 'seasonal'}
MODEL_ORDER = ['last_close', 'drift', 'seasonal', 'ridge', 'mlp', 'gru', 'lstm', 'xpatch']


def compute_manifest():
    import torch
    cuda = torch.cuda.is_available()
    return dict(device=torch.cuda.get_device_name(0) if cuda else 'cpu',
                cuda_available=cuda, cpu_threads=24, cpu_dataloader_workers=24,
                cuda_dataloader_workers=0, torch_version=torch.__version__)


def model_factory(name, lookback, horizon):
    import torch.nn as nn
    if name == 'mlp':
        return nn.Sequential(nn.Flatten(), nn.Linear(lookback*5, 64), nn.ReLU(),
                             nn.Dropout(.1), nn.Linear(64, horizon))
    if name in ('gru', 'lstm'):
        # State dict is exported; this wrapper is reconstructed, never pickled.
        class Recurrent(nn.Module):
            def __init__(self):
                super().__init__()
                layer = nn.GRU if name == 'gru' else nn.LSTM
                self.r = layer(5, 32, batch_first=True)
                self.h = nn.Linear(32, horizon)
            def forward(self, x):
                return self.h(self.r(x)[0][:, -1])
        return Recurrent()
    if name == 'xpatch':
        from config import Config
        from models.xpatch import XPatch
        cfg = Config()
        cfg.EMA_ALPHA, cfg.HIDDEN_SIZE, cfg.NUM_LAYERS = .1, 32, 1
        cfg.NUM_HEADS, cfg.D_FF, cfg.DROPOUT = 4, 64, .1
        cfg.PATCH_SIZES = [min(5, lookback), min(10, lookback)]
        cfg.PREDICTION_LENGTH = horizon
        return XPatch(5, cfg)
    raise ValueError(f'Unknown neural model {name}')


def forward(model, x):
    out = model(x)
    return out['regression_pred'] if isinstance(out, dict) else out


def train_model(name, X, y, horizon):
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    feature = StandardScaler().fit(X.reshape(-1, 5))
    target = StandardScaler().fit(y)
    x = feature.transform(X.reshape(-1, 5)).reshape(X.shape).astype('float32')
    yt = target.transform(y).astype('float32')
    bundle = dict(name=name, lookback=X.shape[1], horizon=horizon,
                  feature_scaler=feature, target_scaler=target)
    if name == 'ridge':
        from sklearn.linear_model import Ridge
        with threadpool_limits(limits=24):
            bundle['estimator'] = Ridge(alpha=1.).fit(x.reshape(len(x), -1), yt)
        return bundle
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    torch.set_num_threads(24)
    torch.manual_seed(2026)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.set_device(0)
        torch.cuda.manual_seed_all(2026)
    model = model_factory(name, X.shape[1], horizon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    # Early stopping uses a purged tail of the outer TRAIN partition only.
    cut = int(len(x)*.85)
    stop_start = cut + horizon
    stopping = stop_start < len(x) and cut >= 8
    fit_end = cut if stopping else len(x)
    loader = DataLoader(TensorDataset(torch.from_numpy(x[:fit_end]), torch.from_numpy(yt[:fit_end])),
                        batch_size=128, shuffle=True, num_workers=0 if device == 'cuda' else 24)
    best, best_loss, stale = None, float('inf'), 0
    epochs = 0
    for epoch in range(20):
        epochs = epoch + 1
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = (forward(model, bx)-by).square().mean()
            if not torch.isfinite(loss):
                raise ValueError('non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        model.eval()
        if stopping:
            losses = []
            with torch.no_grad():
                for start in range(stop_start, len(x), 256):
                    bx = torch.from_numpy(x[start:start+256]).to(device)
                    by = torch.from_numpy(yt[start:start+256]).to(device)
                    losses.append(((forward(model, bx)-by).square().sum().item(), by.numel()))
            score = sum(a for a,b in losses)/sum(b for a,b in losses)
            if score < best_loss:
                best_loss, stale = score, 0
                best = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:
                stale += 1
            if stale >= 3:
                break
    if best is not None:
        model.load_state_dict(best)
    bundle.update(state_dict={k: v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()},
                  epochs=epochs, trained_device=device,
                  architecture=dict(width=32, layers=1, mlp_width=64, dropout=.1),
                  early_stopping='purged chronological tail of training only' if stopping else 'disabled: small training partition')
    return bundle


def predict_model(bundle, X):
    x = bundle['feature_scaler'].transform(X.reshape(-1, 5)).reshape(X.shape).astype('float32')
    if bundle['name'] == 'ridge':
        prediction = bundle['estimator'].predict(x.reshape(len(x), -1))
    else:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model_factory(bundle['name'], bundle['lookback'], bundle['horizon'])
        model.load_state_dict({k: torch.from_numpy(v.copy()) for k,v in bundle['state_dict'].items()})
        model.to(device).eval()
        parts = []
        with torch.no_grad():
            for start in range(0, len(x), 256):
                parts.append(forward(model, torch.from_numpy(x[start:start+256]).to(device)).cpu().numpy())
        prediction = np.concatenate(parts)
    return bundle['target_scaler'].inverse_transform(np.asarray(prediction).reshape(len(X), bundle['horizon']))


def baseline(name, closes, origins, horizon, lookback, interval, target_mode):
    base = closes[origins]
    steps = np.arange(1, horizon+1)[None, :]
    if name == 'last_close':
        return np.repeat(base[:, None], horizon, axis=1)
    if name == 'drift':
        prior = closes[origins-lookback+1]
        if target_mode == 'future_close':
            return np.maximum(1e-12, base[:, None] + steps*(base-prior)[:, None]/max(lookback-1, 1))
        return base[:, None]*np.exp(np.clip(steps*np.log(base/prior)[:, None]/max(lookback-1,1), -20, 20))
    period = {'1mo':12, '1d':252, '1h':6, '1min':390, '5min':78}[interval]
    # Repeat the last complete seasonal cycle; never index a future target.
    ids = origins[:, None] - period + 1 + (steps-1) % period
    pred = np.repeat(base[:, None], horizon, axis=1)
    valid = ids >= 0
    pred[valid] = closes[ids[valid]]
    return pred
