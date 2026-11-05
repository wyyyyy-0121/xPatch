"""Strictly past-only features and real Chronos-2 encoder representations."""
import numpy as np
import torch


def price_features(market, origins):
    origins = np.asarray(origins, dtype=np.int64)
    bars = market.bars
    close = bars.close.to_numpy()
    high, low = bars.high.to_numpy(), bars.low.to_numpy()
    volume = bars.volume.to_numpy()
    values, names = [], []
    rv12 = None
    for n in (5, 12, 30, 60, 120, 240):
        idx = origins[:, None] + np.arange(1 - n, 1)
        ret = market.returns[idx]
        if not np.isfinite(ret).all():
            raise ValueError('Window crosses a session boundary')
        std = np.sqrt(np.mean(ret ** 2, axis=1) + 1e-14)
        if n == 60:
            rv12 = std * np.sqrt(12)
        vol = volume[idx]
        features = [ret.sum(axis=1) / (std * np.sqrt(n)),
                    np.log(std), np.mean(np.abs(ret), axis=1) / std,
                    np.mean((high[idx] - low[idx]) / close[idx], axis=1) / std,
                    np.log1p(vol).mean(axis=1),
                    vol[:, -1] / (vol.mean(axis=1) + 1)]
        labels = ['momentum', 'log_volatility', 'absolute_return', 'range', 'log_volume', 'relative_volume']
        values.extend(features)
        names.extend([f'{label}_{n}' for label in labels])
    # Local clock has no target information and handles DST explicitly.
    clock = bars.index[origins].tz_convert('America/New_York')
    phase = (clock.hour.to_numpy() * 60 + clock.minute.to_numpy() - 570) / 390
    candle = (bars.close.to_numpy()[origins] - bars.open.to_numpy()[origins])
    candle /= high[origins] - low[origins] + 1e-8
    values.extend([np.sin(2*np.pi*phase), np.cos(2*np.pi*phase), candle])
    names.extend(['clock_sin', 'clock_cos', 'current_candle_body'])
    out = np.column_stack(values).astype(np.float32)
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite causal market features')
    return out, rv12.astype(np.float32), names


def context_tensor(market, origins):
    origins = np.asarray(origins, dtype=np.int64)
    idx = origins[:, None] + np.arange(-239, 1)
    if not np.all(market.dates[idx] == market.dates[origins, None]):
        raise ValueError('Foundation context crosses session')
    close = market.bars.close.to_numpy()
    span = np.log(market.bars.high.to_numpy()[idx] / market.bars.low.to_numpy()[idx])
    x = np.stack([market.returns[idx], span,
                  np.log1p(market.bars.volume.to_numpy()[idx])], axis=1).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite foundation context')
    return x


@torch.inference_mode()
def encode_context(pipe, contexts, projection=None):
    """Three variates communicate within an origin; origins cannot communicate."""
    contexts = np.asarray(contexts, dtype=np.float32)
    n, channels, lookback = contexts.shape
    if channels != 3 or lookback != 240:
        raise ValueError('Expected [origins, 3, 240]')
    pipe.model.eval()
    tensor = torch.as_tensor(contexts.reshape(n*channels, lookback), device=pipe.model.device)
    group_ids = torch.arange(n, device=tensor.device).repeat_interleave(channels)
    encoded, *_ = pipe.model.encode(context=tensor, group_ids=group_ids, num_output_patches=1)
    # The masked output token is contextualized entirely by observed history.
    latent = encoded[0][:, -1, :].reshape(n, -1).float().cpu().numpy()
    if projection is None:
        rng = np.random.default_rng(20261105)
        projection = (rng.normal(size=(latent.shape[1], 48)) / np.sqrt(latent.shape[1])).astype(np.float32)
    if projection.shape != (latent.shape[1], 48):
        raise ValueError('Projection identity/shape mismatch')
    embedding = latent @ projection
    if not np.isfinite(embedding).all():
        raise ValueError('Nonfinite foundation embedding')
    return embedding, projection


def representation(pipe, market, origins, batch_size=16, projection=None):
    out = []
    for start in range(0, len(origins), batch_size):
        z, projection = encode_context(pipe, context_tensor(market, origins[start:start+batch_size]), projection)
        out.append(z)
    return np.concatenate(out), projection
