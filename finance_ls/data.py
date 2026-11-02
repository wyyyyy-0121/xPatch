"""Price-to-return conversion and chronological sampling, without imputation."""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class MarketData:
    timestamps: np.ndarray
    dates: np.ndarray
    returns: np.ndarray
    origins: np.ndarray
    audit: dict
    lookback: int
    horizon: int

    def samples(self, dates):
        return self.origins[np.isin(self.dates[self.origins], dates)]

    def scale(self, train_dates):
        values = self.returns[np.isin(self.dates, train_dates)]
        values = values[np.isfinite(values)]
        if not len(values):
            raise ValueError("No valid training returns")
        scale = float(values.std(dtype=np.float64))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Training return standard deviation must be positive")
        return {"std": scale, "center": 0.0, "fit_dates": list(train_dates),
                "n_returns": len(values), "method": "train_only_std_ddof0"}

    def arrays(self, origins, scale):
        origins = np.asarray(origins, dtype=np.int64)
        x_idx = origins[:, None] + np.arange(1 - self.lookback, 1)
        y_idx = origins[:, None] + np.arange(1, self.horizon + 1)
        x = (self.returns[x_idx] / scale).astype(np.float32)[..., None]
        y = (self.returns[y_idx] / scale).astype(np.float32)[..., None]
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Invalid return entered a sample")
        return x, y


def load_market_data(path, lookback=240, horizon=12):
    path = Path(path)
    if lookback < 1 or horizon < 1:
        raise ValueError("Positive lookback and horizon required")
    df = pd.read_csv(path)
    if not {"timestamp", "close"}.issubset(df.columns):
        raise ValueError("CSV must contain timestamp and close")
    ts = pd.to_datetime(df["timestamp"], errors="raise")
    if ts.isna().any():
        raise ValueError("Missing timestamp")
    if ts.duplicated().any():
        raise ValueError("Duplicate timestamps; resolve at source")
    sorted_input = bool(ts.is_monotonic_increasing)
    order = np.argsort(ts.to_numpy())
    ts = ts.iloc[order].reset_index(drop=True)
    if ((ts.dt.second != 0) | (ts.dt.microsecond != 0) | (ts.dt.nanosecond != 0)).any():
        raise ValueError("Timestamps must be on exact minute boundaries")
    close = pd.to_numeric(df["close"].iloc[order], errors="raise").to_numpy(np.float64)
    if not (np.isfinite(close) & (close > 0)).all():
        raise ValueError("Close prices must be finite and positive")
    dates = ts.dt.strftime("%Y-%m-%d").to_numpy()
    # Absolute instants for continuity; supplied local/calendar dates for sessions.
    timestamps = ts.to_numpy(dtype="datetime64[ns]")
    n = len(ts)
    if n < lookback + horizon + 1:
        raise ValueError("Insufficient price rows")
    adjacent = np.zeros(n, dtype=bool)
    adjacent[1:] = ((np.diff(timestamps) == np.timedelta64(1, "m")) &
                    (dates[1:] == dates[:-1]))
    returns = np.full(n, np.nan, dtype=np.float64)
    returns[1:] = np.diff(np.log(close))
    returns[~adjacent] = np.nan
    starts = np.maximum.accumulate(np.where(~adjacent, np.arange(n), 0))
    ends = np.r_[np.flatnonzero(~adjacent)[1:] - 1, n - 1]
    run_end = np.repeat(ends, np.diff(np.r_[0, ends + 1]))
    idx = np.arange(n)
    eligible = (idx - starts >= lookback) & (run_end - idx >= horizon)
    origins = idx[eligible]
    all_dates = np.unique(dates)
    coverage = []
    for date in all_dates:
        rows = np.flatnonzero(dates == date)
        candidates = max(0, len(rows) - lookback - horizon)
        actual = int(np.sum(dates[origins] == date))
        coverage.append({"date": str(date), "price_rows": len(rows),
                         "row_count_possible_origins": candidates,
                         "eligible_origins": actual,
                         "excluded_for_gaps": candidates - actual})
    audit = {"source": str(path.resolve()), "sha256": sha256(path.read_bytes()).hexdigest(),
             "rows": n, "sorted_input": sorted_input, "dates": len(all_dates),
             "first_timestamp": str(ts.iloc[0]), "last_timestamp": str(ts.iloc[-1]),
             "timezone": str(ts.dt.tz) if ts.dt.tz else "unspecified",
             "source_and_adjustment": "not established by CSV; no market-hours filtering",
             "session_policy": "same supplied calendar date, exact consecutive minutes",
             "lookback_returns": lookback, "required_history_prices": lookback + 1,
             "horizon_minutes": horizon, "eligible_origins": len(origins),
             "within_date_gaps": int(np.sum((dates[1:] == dates[:-1]) & ~adjacent[1:])),
             "coverage_by_date": coverage}
    valid_returns = returns[np.isfinite(returns)]
    largest = np.flatnonzero(np.isfinite(returns))
    largest = largest[np.argsort(np.abs(returns[largest]))[-5:][::-1]]
    audit["return_quality"] = {
        "zero_fraction": float(np.mean(valid_returns == 0)),
        "abs_return_quantiles": {str(q): float(np.quantile(np.abs(valid_returns), q))
                                 for q in (0.5, 0.95, 0.99, 1.0)},
        "largest_absolute_returns": [{"timestamp": str(ts.iloc[i]), "log_return": float(returns[i]),
                                      "previous_close": float(close[i - 1]), "close": float(close[i])}
                                     for i in largest],
        "policy": "reported without clipping; provenance and corporate actions require source evidence"}
    if not len(origins):
        raise ValueError("No continuous intraday windows for requested lookback/horizon")
    return MarketData(timestamps, dates, returns, origins, audit, lookback, horizon)


def chronological_splits(data, test_fraction=0.2, folds=3):
    if not 0 < test_fraction < 1 or folds < 1:
        raise ValueError("Invalid split settings")
    dates = np.unique(data.dates)
    test_count = max(1, int(np.ceil(len(dates) * test_fraction)))
    development, test = dates[:-test_count], dates[-test_count:]
    # Initial 40% of development dates trains fold 1; remaining 60% validates.
    edges = np.linspace(int(np.floor(len(development) * 0.4)), len(development), folds + 1).astype(int)
    if edges[0] < 1 or np.any(np.diff(edges) < 1):
        raise ValueError("Too few dates for requested expanding folds")
    splits = []
    for i in range(folds):
        train_dates = development[:edges[i]]
        val_dates = development[edges[i]:edges[i + 1]]
        splits.append({"name": f"fold_{i + 1}", "train_dates": train_dates.tolist(),
                       "eval_dates": val_dates.tolist()})
    splits.append({"name": "test", "train_dates": development.tolist(), "eval_dates": test.tolist()})
    for split in splits:
        if not len(data.samples(split["train_dates"])) or not len(data.samples(split["eval_dates"])):
            raise ValueError(f"No eligible samples in {split['name']}")
        # Same-date label restriction implies disjoint labels across date splits.
        if max(split["train_dates"]) >= min(split["eval_dates"]):
            raise AssertionError("Non-chronological split")
    return splits
