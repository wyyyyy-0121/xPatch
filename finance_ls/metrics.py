"""Metrics in original log-return units and paired day-block uncertainty."""
import numpy as np


def forecast_metrics(y, pred):
    y, pred = np.asarray(y, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    if y.ndim != 2 or pred.shape != y.shape or not len(y):
        raise ValueError("Expected matching nonempty [samples, horizon] arrays")
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Nonfinite forecasts or labels")
    result = {}
    for h in (1, 5, 12):
        if h > y.shape[1]:
            continue
        actual = y[:, :h].sum(axis=1)
        forecast = pred[:, :h].sum(axis=1)
        error = forecast - actual
        mse = float(np.mean(error ** 2))
        zero_mse = float(np.mean(actual ** 2))
        result[str(h)] = {"mse": mse, "mae": float(np.mean(np.abs(error))),
                          "rmse": float(np.sqrt(mse)),
                          "skill_vs_zero": 1.0 - mse / zero_mse if zero_mse > 0 else None,
                          "direction_accuracy": float(np.mean(np.sign(forecast) == np.sign(actual))),
                          "direction_policy": "sign equality; zeros count only when both zero"}
    return result


def paired_day_bootstrap(y, candidate, baseline, dates, horizon=12, repeats=5000, seed=2026):
    """CI for candidate-minus-baseline MSE, resampling whole dates, not rows."""
    y, candidate, baseline = (np.asarray(x, dtype=np.float64) for x in (y, candidate, baseline))
    if y.shape != candidate.shape or y.shape != baseline.shape or len(dates) != len(y):
        raise ValueError("Paired predictions and dates must align")
    actual = y[:, :horizon].sum(axis=1)
    delta = ((candidate[:, :horizon].sum(axis=1) - actual) ** 2 -
             (baseline[:, :horizon].sum(axis=1) - actual) ** 2)
    _, groups = np.unique(dates, return_inverse=True)
    counts = np.bincount(groups)
    sums = np.bincount(groups, weights=delta)
    rng = np.random.default_rng(seed)
    estimates = np.empty(repeats)
    for i in range(repeats):
        draw = rng.integers(0, len(counts), len(counts))
        estimates[i] = sums[draw].sum() / counts[draw].sum()
    return {"mse_difference": float(delta.mean()),
            "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
            "days": len(counts), "repeats": repeats, "seed": seed,
            "interpretation": "negative favors candidate; days treated as exchangeable blocks"}
