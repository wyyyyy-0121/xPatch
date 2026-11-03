"""Equal-day RV scores, serial dependence and the fixed multiple-testing family."""
import numpy as np
from scipy.stats import norm


def qlike(y, p):
    y, p = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    if y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all() or (y < 0).any() or (p <= 0).any():
        raise ValueError("QLIKE needs aligned finite nonnegative targets and positive predictions")
    ratio = np.maximum(y, 1e-12) / np.maximum(p, 1e-12)
    return ratio - np.log(ratio) - 1


def daily_mean(values, dates):
    values, dates = np.asarray(values), np.asarray(dates)
    if values.ndim != 1 or len(values) != len(dates) or not np.isfinite(values).all():
        raise ValueError("Daily loss arrays must be aligned and finite")
    days, inverse = np.unique(dates, return_inverse=True)
    if not len(days):
        raise ValueError("Empty daily score")
    return days, np.bincount(inverse, weights=values) / np.bincount(inverse)


def score(y, p, dates, quantiles=None, levels=None):
    _, losses = daily_mean(qlike(y, p), dates)
    _, mse = daily_mean((np.log(np.maximum(y, 1e-12)) - np.log(p)) ** 2, dates)
    result = {"qlike": float(losses.mean()), "log_rv_mse": float(mse.mean()),
        "sessions": len(losses), "origins": len(y)}
    if quantiles is not None:
        levels = list(levels)
        for low, high, name in ((0.1, 0.9, "80"), (0.05, 0.95, "90")):
            lower, upper = quantiles[:, levels.index(low)], quantiles[:, levels.index(high)]
            _, covered = daily_mean(((y >= lower) & (y <= upper)).astype(float), dates)
            _, widths = daily_mean(upper - lower, dates)
            result[f"coverage_{name}"] = float(covered.mean())
            result[f"width_{name}"] = float(widths.mean())
    return result


def hac_meat(scores, lags=5):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        scores = scores[:, None]
    meat = scores.T @ scores
    for lag in range(1, min(lags, len(scores) - 1) + 1):
        covariance = scores[lag:].T @ scores[:-lag]
        meat += (1 - lag / (lags + 1)) * (covariance + covariance.T)
    return meat


def dm_test(difference, lags=5):
    d = np.asarray(difference, dtype=float)
    if d.ndim != 1 or len(d) < 2 or not np.isfinite(d).all():
        raise ValueError("DM requires at least two finite daily loss differences")
    mean = float(d.mean())
    variance = max(float(hac_meat(d - mean, lags)[0, 0] / len(d) ** 2), 0)
    if variance <= 1e-30:
        return {"mean_difference": mean, "statistic": None, "p_less": 0.0 if mean < 0 else 1.0,
            "hac_lags": lags, "zero_variance": True}
    statistic = mean / np.sqrt(variance)
    return {"mean_difference": mean, "statistic": float(statistic),
        "p_less": float(norm.cdf(statistic)), "hac_lags": lags, "zero_variance": False}


def block_comparison(difference, block=1, repeats=5000, seed=2026):
    d = np.asarray(difference, dtype=float)
    if d.ndim != 1 or len(d) < 2 or not np.isfinite(d).all() or block < 1 or repeats < 2:
        raise ValueError("Invalid bootstrap inputs")
    rng, means = np.random.default_rng(seed), []
    blocks = (len(d) + block - 1) // block
    for start in range(0, repeats, 500):
        count = min(500, repeats - start)
        starts = rng.integers(0, len(d), size=(count, blocks))
        indices = (starts[:, :, None] + np.arange(block)) % len(d)
        means.extend(d[indices.reshape(count, -1)[:, :len(d)]].mean(axis=1))
    means = np.asarray(means)
    centered = means - d.mean()
    return {"block_sessions": block, "repeats": repeats, "seed": seed,
        "mean_difference": float(d.mean()), "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "p_less_centered_null": float((1 + np.sum(centered <= d.mean())) / (repeats + 1))}


def holm(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    if not np.isfinite(pvalues).all() or ((pvalues < 0) | (pvalues > 1)).any():
        raise ValueError("Invalid p values")
    order = np.argsort(pvalues, kind="stable")
    adjusted = np.empty_like(pvalues)
    adjusted[order] = np.minimum(1, np.maximum.accumulate((len(pvalues) - np.arange(len(pvalues))) * pvalues[order]))
    return adjusted


def mincer_zarnowitz(y, p, dates, lags=5):
    days, observed = daily_mean(y, dates)
    check, predicted = daily_mean(p, dates)
    if not np.array_equal(days, check) or len(days) < 3:
        raise ValueError("MZ requires at least three aligned daily means")
    scale = max(float(observed.mean()), 1e-12)
    x = np.column_stack([np.ones(len(days)), predicted / scale])
    if np.linalg.matrix_rank(x) < 2:
        return {"unit": "daily_mean_of_overlapping_intraday_RV_forecasts", "identified": False,
            "reason": "constant_or_collinear_daily_forecast", "intercept": None, "slope": None,
            "hac_covariance": None, "hac_lags": lags, "sessions": len(days)}
    beta = np.linalg.lstsq(x, observed / scale, rcond=None)[0]
    residual = observed / scale - x @ beta
    bread = np.linalg.pinv(x.T @ x)
    covariance = bread @ hac_meat(x * residual[:, None], lags) @ bread
    covariance *= len(days) / (len(days) - 2)
    conversion = np.diag([scale, 1.0])
    return {"unit": "daily_mean_of_overlapping_intraday_RV_forecasts", "identified": True, "intercept": float(beta[0] * scale),
        "slope": float(beta[1]), "hac_covariance": (conversion @ covariance @ conversion).tolist(),
        "hac_lags": lags, "sessions": len(days)}
