"""Naive, drift and fixed-cycle seasonal baselines in the same target units."""
import numpy as np

from .features import CloseWindows


def predict(name: str, windows: CloseWindows, origins, horizon: int, mode: str, period: int = None):
    idx = np.asarray(origins, dtype=np.int64)
    x, anchor = windows.arrays(idx)
    steps = np.arange(1, horizon + 1, dtype=np.float64)[None, :]
    if mode not in ("future_close", "future_log_return"):
        raise ValueError("Unknown target")
    if name == "last_close":
        return np.repeat(anchor if mode == "future_close" else np.zeros_like(anchor), horizon, axis=1)
    if name == "drift":
        if windows.lookback < 2:
            raise ValueError("Drift needs two historical bars")
        if mode == "future_close":
            past = anchor * np.exp(x[:, :1])
            return np.maximum(anchor + steps * (anchor - past) / (windows.lookback - 1), 1e-12)
        return -steps * x[:, :1] / (windows.lookback - 1)
    if name == "seasonal":
        if period is None or period < 1 or period > windows.lookback:
            raise ValueError("A complete observed seasonal cycle is required")
        # Repeat the last period of known close levels, relative to the origin.
        # period is in trading bars; early-close phase effects must be disclosed.
        seasonal = x[:, -period:][:, np.arange(horizon) % period]
        return anchor * np.exp(seasonal) if mode == "future_close" else seasonal
    raise ValueError(f"Unknown baseline {name}")
