"""Path errors and dependence-aware comparisons in each task's original units."""
from __future__ import annotations

import numpy as np


def path_metrics(actual, predicted, target: str, *, anchor=None):
    y = np.asarray(actual, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    if y.ndim != 2 or y.shape != p.shape or not y.size:
        raise ValueError("Actual and predicted arrays must be nonempty matching paths")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Path contains nonfinite values")
    if target == "future_close":
        a = np.asarray(anchor, dtype=np.float64)
        if a.shape != (len(y),) or not np.isfinite(a).all() or (a <= 0).any() or (y <= 0).any():
            raise ValueError("Positive origin and actual prices are required")
        actual_move, predicted_move = y[:, -1] - a, p[:, -1] - a
    elif target == "future_log_return":
        actual_move, predicted_move = y[:, -1], p[:, -1]
    else:
        raise ValueError("Unknown target")
    error = p - y
    per_step_mse = np.square(error).mean(axis=0)
    per_step_mae = np.abs(error).mean(axis=0)
    active = predicted_move != 0
    return {"origins": len(y), "horizon": y.shape[1],
            "endpoint_mse": float(per_step_mse[-1]), "endpoint_rmse": float(np.sqrt(per_step_mse[-1])),
            "endpoint_mae": float(per_step_mae[-1]), "path_mse": float(np.square(error).mean()),
            "path_mae": float(np.abs(error).mean()),
            "per_step_mse": per_step_mse.tolist(), "per_step_mae": per_step_mae.tolist(),
            "direction_coverage": float(active.mean()),
            "direction_accuracy_when_active": float((np.sign(predicted_move[active]) ==
                                                       np.sign(actual_move[active])).mean()) if active.any() else None}


def endpoint_improvement(actual, predicted, baseline):
    y, model, control = (np.asarray(x, dtype=np.float64) for x in (actual, predicted, baseline))
    if y.ndim != 2 or y.shape != model.shape or y.shape != control.shape or not y.size:
        raise ValueError("Compared paths must be nonempty and aligned")
    if not np.isfinite(y).all() or not np.isfinite(model).all() or not np.isfinite(control).all():
        raise ValueError("Compared paths must be finite")
    model_loss = np.square(model[:, -1] - y[:, -1])
    control_loss = np.square(control[:, -1] - y[:, -1])
    control_mse = float(control_loss.mean())
    return {"model_mse": float(model_loss.mean()), "baseline_mse": control_mse,
            "relative_improvement": float(1.0 - model_loss.mean() / control_mse) if control_mse > 0 else None,
            "paired_loss_difference": model_loss - control_loss}


def paired_block_bootstrap(loss_difference, groups, *, block_length: int, resamples: int = 5000,
                           seed: int = 2026, confidence: float = 0.95):
    """Non-circular contiguous-group bootstrap, weighted by origin count.

    Negative differences favor the model. The centered one-sided p-value tests
    improvement relative to zero; independence and suitability of the chosen
    grouping/block length remain the caller's responsibility.
    """
    diff = np.asarray(loss_difference, dtype=np.float64)
    group = np.asarray(groups)
    if diff.ndim != 1 or group.ndim != 1 or len(diff) != len(group) or not len(diff):
        raise ValueError("Loss differences and groups must be equal-length vectors")
    if not np.isfinite(diff).all() or not 0 < confidence < 1 or resamples < 100:
        raise ValueError("Invalid loss differences, confidence or resample count")
    unique, inverse = np.unique(group, return_inverse=True)
    if not 1 <= block_length <= len(unique):
        raise ValueError("Block length must fit the number of observed groups")
    sums = np.bincount(inverse, weights=diff, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique))
    observed = float(diff.mean())
    rng = np.random.default_rng(seed)
    block_count = int(np.ceil(len(unique) / block_length))
    draws = np.empty(resamples, dtype=np.float64)
    offsets = np.arange(block_length)
    for i in range(resamples):
        starts = rng.integers(0, len(unique) - block_length + 1, size=block_count)
        sample = (starts[:, None] + offsets).ravel()[:len(unique)]
        draws[i] = sums[sample].sum() / counts[sample].sum()
    tail = (1.0 - confidence) / 2
    lo, hi = np.quantile(draws, [tail, 1 - tail])
    p = (1 + np.count_nonzero(draws - observed <= observed)) / (resamples + 1)
    return {"groups": len(unique), "origins": len(diff), "block_length": block_length,
            "resamples": resamples, "seed": seed, "confidence": confidence,
            "mean_paired_loss_difference": observed, "confidence_interval": [float(lo), float(hi)],
            "centered_one_sided_p_improvement": float(p),
            "interpretation": "Negative loss difference favors the model; apply only to prespecified independent comparisons."}


def holm(p_values):
    vals = np.asarray(p_values, dtype=np.float64)
    if vals.ndim != 1 or not len(vals) or not np.isfinite(vals).all() or (vals < 0).any() or (vals > 1).any():
        raise ValueError("p-values must be a nonempty vector in [0,1]")
    order = np.argsort(vals)
    adjusted = np.empty(len(vals), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = min(1.0, max(running, (len(vals) - rank) * vals[index]))
        adjusted[index] = running
    return adjusted
