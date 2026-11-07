"""Bounded-memory multi-output ridge regression with an unpenalized intercept."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .bars import targets
from .features import CloseWindows, Scaler


@dataclass(frozen=True)
class RidgeFit:
    coefficients: np.ndarray
    alpha: float
    observations: int

    def predict(self, windows: CloseWindows, scaler: Scaler, origins, batch_size=4096):
        indices = np.asarray(origins, dtype=np.int64)
        outputs = []
        for start in range(0, len(indices), batch_size):
            x, anchor = windows.arrays(indices[start:start + batch_size])
            x, anchor = scaler.transform_x(x, anchor)
            design = np.column_stack((x.astype(np.float64), anchor.astype(np.float64), np.ones(len(x))))
            outputs.append(scaler.invert_y(design @ self.coefficients))
        return np.concatenate(outputs) if outputs else np.empty((0, self.coefficients.shape[1]))


def fit_many(windows: CloseWindows, scaler: Scaler, fit_origins, horizon: int, mode: str,
             alphas, batch_size: int = 4096):
    indices = np.asarray(fit_origins, dtype=np.int64)
    alphas = tuple(float(a) for a in alphas)
    if not len(indices) or not alphas or any(not np.isfinite(a) or a <= 0 for a in alphas) or batch_size < 1:
        raise ValueError("Positive alphas, batch size and nonempty fit origins required")
    if len(set(alphas)) != len(alphas):
        raise ValueError("Ridge alphas must be unique")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Ridge fit origins must be distinct")
    width = windows.lookback + 2  # window, anchor, intercept
    gram = np.zeros((width, width), dtype=np.float64)
    cross = np.zeros((width, horizon), dtype=np.float64)
    for start in range(0, len(indices), batch_size):
        batch = indices[start:start + batch_size]
        x, anchor = windows.arrays(batch)
        x, anchor = scaler.transform_x(x, anchor)
        y = scaler.transform_y(targets(windows.bars, batch, horizon, mode))
        design = np.column_stack((x.astype(np.float64), anchor.astype(np.float64), np.ones(len(x))))
        gram += design.T @ design
        cross += design.T @ y.astype(np.float64)
    result = {}
    for alpha in alphas:
        penalty = np.eye(width, dtype=np.float64) * alpha
        penalty[-1, -1] = 0.0
        coefficients = np.linalg.solve(gram + penalty, cross)
        if not np.isfinite(coefficients).all():
            raise ValueError("Nonfinite ridge coefficients")
        result[alpha] = RidgeFit(coefficients, alpha, len(indices))
    return result


def fit(windows: CloseWindows, scaler: Scaler, fit_origins, horizon: int, mode: str,
        alpha: float, batch_size: int = 4096):
    return fit_many(windows, scaler, fit_origins, horizon, mode, (alpha,), batch_size)[float(alpha)]
