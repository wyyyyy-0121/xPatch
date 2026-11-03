"""Calibration uses a disjoint historical tail, never evaluation labels."""
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


@dataclass
class Calibration:
    dates: list
    log_factor: float
    levels: list
    residual_quantiles: list
    samples: int

    @classmethod
    def fit(cls, y, raw, dates, levels=(0.05, 0.1, 0.5, 0.9, 0.95)):
        y, raw = np.asarray(y), np.asarray(raw)
        dates = np.asarray(dates)
        if y.ndim != 1 or y.shape != raw.shape or len(dates) != len(y) or not len(y):
            raise ValueError("Calibration arrays must be nonempty and aligned")
        if not np.isfinite(y).all() or not np.isfinite(raw).all() or (y < 0).any() or (raw <= 0).any():
            raise ValueError("Calibration predictions and targets must be finite nonnegative variances")
        residual = np.log(np.maximum(y, 1e-12)) - np.log(raw)
        return cls(sorted(np.unique(dates).tolist()), float(logsumexp(residual) - np.log(len(y))),
            list(levels), np.quantile(residual, levels).tolist(), len(y))

    def predict(self, raw):
        raw = np.asarray(raw, dtype=np.float64)
        if not np.isfinite(raw).all() or (raw <= 0).any():
            raise ValueError("Invalid uncalibrated variance forecast")
        mean = raw * np.exp(self.log_factor)
        quantiles = raw[:, None] * np.exp(self.residual_quantiles)
        if not np.isfinite(mean).all() or not np.isfinite(quantiles).all():
            raise ValueError("Nonfinite calibrated forecast")
        return np.maximum(mean, 1e-12), np.maximum(quantiles, 1e-12)

    def dictionary(self):
        return vars(self).copy()
