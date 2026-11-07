"""Past-only close windows and training-only scaling for path predictions."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .bars import Bars, targets


class CloseWindows:
    """Log-relative history adjusted only for splits already effective at origin.

    Target prices remain actual nominal provider closes. The split adjustment of
    history removes a known past share-count change; a future event cannot alter
    an earlier feature window. Ex-post vendor adjusted_close is never used.
    """
    def __init__(self, bars: Bars, lookback: int, actions=()):
        if lookback < 1:
            raise ValueError("lookback must be positive")
        self.bars = bars
        self.lookback = int(lookback)
        self.log_close = np.full(len(bars.close), np.nan, dtype=np.float64)
        self.log_close[bars.observed] = np.log(bars.close[bars.observed])
        self.log_split = np.zeros(len(bars.close), dtype=np.float64)
        previous = None
        for action in actions:
            date, ratio = str(action["date"]), float(action["ratio"])
            if previous is not None and date <= previous:
                raise ValueError("Split dates must be ordered and distinct")
            if not np.isfinite(ratio) or ratio <= 0:
                raise ValueError("Invalid split ratio")
            self.log_split[bars.session_date >= date] += np.log(ratio)
            previous = date

    def arrays(self, origins):
        idx = np.asarray(origins, dtype=np.int64)
        if idx.ndim != 1 or (idx < self.lookback - 1).any() or (idx >= len(self.bars.close)).any():
            raise ValueError("Origin lacks the requested history")
        indices = idx[:, None] + np.arange(1 - self.lookback, 1)
        if not self.bars.observed[indices].all():
            raise ValueError("History must consist of observed bars")
        log_history = self.log_close[indices] + self.log_split[indices] - self.log_split[idx, None]
        x = log_history - self.log_close[idx, None]
        anchor = self.bars.close[idx, None]
        if not np.isfinite(x).all() or not np.isfinite(anchor).all():
            raise ValueError("Nonfinite historical feature")
        return x, anchor


@dataclass(frozen=True)
class Scaler:
    x_mean: float
    x_std: float
    anchor_mean: float
    anchor_std: float
    y_mean: tuple[float, ...]
    y_std: tuple[float, ...]
    fitted_origin_count: int

    def transform_x(self, x, anchor):
        return ((x - self.x_mean) / self.x_std).astype(np.float32), \
               ((anchor - self.anchor_mean) / self.anchor_std).astype(np.float32)

    def transform_y(self, y):
        return ((y - np.asarray(self.y_mean)) / np.asarray(self.y_std)).astype(np.float32)

    def invert_y(self, y):
        return np.asarray(y) * np.asarray(self.y_std) + np.asarray(self.y_mean)


def fit_scaler(windows: CloseWindows, fit_origins, horizon: int, mode: str, batch_size: int = 4096):
    """Fit only the supplied inner-fit origins, in bounded-memory batches."""
    indices = np.asarray(fit_origins, dtype=np.int64)
    if not len(indices) or horizon < 1 or batch_size < 1:
        raise ValueError("A nonempty training partition, horizon and batch size are required")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Scaler fit origins must be distinct")
    x_sum = x_sq = a_sum = a_sq = 0.0
    y_sum, y_sq = np.zeros(horizon), np.zeros(horizon)
    for start in range(0, len(indices), batch_size):
        chunk = indices[start:start + batch_size]
        x, anchor = windows.arrays(chunk)
        y = targets(windows.bars, chunk, horizon, mode)
        x_sum += float(x.sum(dtype=np.float64)); x_sq += float(np.square(x).sum(dtype=np.float64))
        a_sum += float(anchor.sum(dtype=np.float64)); a_sq += float(np.square(anchor).sum(dtype=np.float64))
        y_sum += y.sum(axis=0, dtype=np.float64); y_sq += np.square(y).sum(axis=0, dtype=np.float64)
    n = len(indices)
    x_mean, anchor_mean, y_mean = x_sum / (n * windows.lookback), a_sum / n, y_sum / n
    x_std = np.sqrt(max(x_sq / (n * windows.lookback) - x_mean ** 2, 0))
    anchor_std = np.sqrt(max(a_sq / n - anchor_mean ** 2, 0))
    y_std = np.sqrt(np.maximum(y_sq / n - y_mean ** 2, 0))
    x_std = float(x_std if x_std > 1e-12 else 1.0)
    anchor_std = float(anchor_std if anchor_std > 1e-12 else 1.0)
    y_std = np.where(y_std > 1e-12, y_std, 1.0)
    return Scaler(float(x_mean), x_std, float(anchor_mean), anchor_std,
                  tuple(map(float, y_mean)), tuple(map(float, y_std)), n)
