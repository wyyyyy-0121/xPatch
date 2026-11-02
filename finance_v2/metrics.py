"""Cumulative-return scores, disjoint-tail calibration, and paired day/block inference."""
import numpy as np


def calibrate(y, pred, levels=(0.1, 0.5, 0.9)):
    y, pred = _matched(y, pred)
    levels = np.asarray(levels, dtype=np.float64)
    if not len(levels) or np.any(np.diff(levels) <= 0) or levels[0] <= 0 or levels[-1] >= 1:
        raise ValueError("Strictly increasing quantiles inside (0,1) required")
    return {"quantile_levels": levels.tolist(),
            "residual_quantiles": np.quantile(y - pred, levels, axis=0).T.tolist(),
            "samples": len(y), "exchangeable_coverage_guarantee": False}


def calibrated(pred, calibration):
    return np.asarray(pred, dtype=np.float64)[..., None] + np.asarray(calibration["residual_quantiles"])


def _matched(y, pred):
    y, pred = np.asarray(y, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    if y.ndim != 2 or y.shape != pred.shape or not len(y):
        raise ValueError("Matching nonempty [sample,cumulative horizon] arrays required")
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Nonfinite values in scores")
    return y, pred


def forecast_metrics(y, pred, horizons=(1, 5, 12), quantiles=None, levels=(0.1, 0.5, 0.9)):
    y, pred = _matched(y, pred)
    if y.shape[1] != len(horizons):
        raise ValueError("Cumulative horizon shape mismatch")
    if quantiles is not None:
        quantiles = np.asarray(quantiles, dtype=np.float64)
        if quantiles.shape != (*y.shape, len(levels)) or not np.isfinite(quantiles).all():
            raise ValueError("Invalid quantiles")
        if (np.diff(quantiles, axis=-1) < 0).any():
            raise ValueError("Quantile crossing")
    result = {}
    for j, h in enumerate(horizons):
        error = pred[:, j] - y[:, j]
        mse = float(np.mean(error ** 2))
        zero = float(np.mean(y[:, j] ** 2))
        item = {"mse": mse, "mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(mse)),
                "skill_vs_zero": 1 - mse / zero if zero else None,
                "direction_accuracy": float(np.mean(np.sign(pred[:, j]) == np.sign(y[:, j])))}
        if quantiles is not None:
            residual = y[:, j, None] - quantiles[:, j]
            item["pinball"] = {str(q): float(np.mean(np.maximum(q * residual[:, k], (q - 1) * residual[:, k])))
                               for k, q in enumerate(levels)}
            item["cdf_coverage"] = {str(q): float(np.mean(y[:, j] <= quantiles[:, j, k]))
                                    for k, q in enumerate(levels)}
            item["central_interval_coverage"] = float(np.mean((y[:, j] >= quantiles[:, j, 0]) &
                                                            (y[:, j] <= quantiles[:, j, -1])))
            item["mean_interval_width"] = float(np.mean(quantiles[:, j, -1] - quantiles[:, j, 0]))
        result[str(h)] = item
    return result


def paired_block_bootstrap(loss_difference, dates, repeats=5000, seed=2026, block_days=1):
    """Sample contiguous date blocks, retaining every intraday window and seed-mean loss.

    The one-sided p value is the upper-tail probability at zero because a
    negative loss difference favors the candidate. It is a bootstrap
    approximation, not an iid minute test.
    """
    delta, dates = np.asarray(loss_difference, dtype=np.float64), np.asarray(dates).astype(str)
    if delta.ndim != 1 or len(delta) != len(dates) or not len(delta) or not np.isfinite(delta).all():
        raise ValueError("Finite aligned paired loss differences required")
    days, groups = np.unique(dates, return_inverse=True)
    if block_days < 1 or len(days) < block_days or repeats < 100:
        raise ValueError("Insufficient date blocks or resamples")
    counts, sums = np.bincount(groups), np.bincount(groups, weights=delta)
    rng = np.random.default_rng(seed)
    n, estimates = len(days), np.empty(repeats)
    blocks = int(np.ceil(n / block_days))
    offsets = np.arange(block_days)
    # Circular moving blocks preserve order within each block. Same RNG draws
    # for every paired comparison aid reproducibility; no per-row bootstrap.
    for start in range(0, repeats, 256):
        size = min(256, repeats - start)
        starts = rng.integers(0, n, size=(size, blocks))
        draw = ((starts[..., None] + offsets) % n).reshape(size, -1)[:, :n]
        estimates[start:start + size] = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    observed = float(delta.mean())
    # For H1 candidate loss < baseline loss, the null boundary is zero; use the
    # bootstrap distribution of the paired difference and its upper tail.
    p = (1 + int(np.sum(estimates >= 0.0))) / (repeats + 1)
    return {"mse_difference": observed, "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
            "p_value_one_sided_candidate_better": p, "days": n, "block_days": block_days,
            "repeats": repeats, "seed": seed,
            "interpretation": "negative favors candidate; bootstrap uncertainty conditional on historical dates and fitted models"}


def holm(p_values):
    """Return monotone Holm adjusted p values keyed by predeclared family member."""
    ordered = sorted(p_values, key=p_values.get)
    adjusted, previous = {}, 0.0
    for i, key in enumerate(ordered):
        value = float(p_values[key])
        if not 0 <= value <= 1:
            raise ValueError("Invalid p value")
        previous = max(previous, min(1.0, (len(ordered) - i) * value))
        adjusted[key] = previous
    return adjusted
