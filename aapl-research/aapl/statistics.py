"""Paired day-block inference; independent minute observations are never assumed."""
import numpy as np


def paired_blocks(delta, dates, repeats=5000, block=5, seed=20261104):
    delta = np.asarray(delta, dtype=np.float64)
    dates = np.asarray(dates, dtype=str)
    if len(delta) != len(dates) or not np.isfinite(delta).all():
        raise ValueError("Invalid paired losses")
    days, idx = np.unique(dates, return_inverse=True)
    n = len(days)
    if n < 2 * block:
        raise ValueError("Insufficient complete day blocks")
    sums = np.bincount(idx, weights=delta)
    counts = np.bincount(idx)
    observed = delta.mean()
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(repeats)
    blocks = int(np.ceil(n / block))
    for i in range(0, repeats, 128):
        size = min(128, repeats - i)
        starts = rng.integers(n, size=(size, blocks))
        draw = ((starts[..., None] + np.arange(block)) % n).reshape(size, -1)[:, :n]
        bootstrap[i:i + size] = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    # Center each per-origin loss at the null: reweighting works even with unequal days.
    centered = bootstrap - observed
    p = (1 + int((centered <= observed).sum())) / (repeats + 1)
    return {"loss_difference": float(observed), "ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "one_sided_p": float(p), "block_days": block, "days": n,
            "repeats": repeats, "method": "null_centered_circular_moving_day_block_bootstrap"}


def holm(values):
    ordered = sorted(values, key=values.get)
    result, last = {}, 0.0
    for j, key in enumerate(ordered):
        last = max(last, min(1.0, (len(ordered) - j) * values[key]))
        result[key] = last
    return result
