"""Exchange-calendar audit, causal windows, and session-purged date splits."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .io import digest


OHLCV = ["open", "high", "low", "close", "volume"]
FEATURE_NAMES = {"close": ["close_return"],
                 "ohlcv": ["open_to_prev_close", "high_to_prev_close", "low_to_prev_close",
                           "close_return", "log1p_volume"]}


def read_bars(path, naive_timezone=None):
    frame = pd.read_csv(path)
    if not {"timestamp", *OHLCV}.issubset(frame):
        raise ValueError("CSV requires timestamp/open/high/low/close/volume")
    # Parsing in UTC handles EST/EDT mixed offsets. Explicit awareness is checked
    # first so a naive provider file cannot silently be interpreted as UTC.
    if frame.timestamp.isna().any() or not len(frame):
        raise ValueError("Empty input or missing timestamps")
    awareness = frame.timestamp.astype(str).str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True)
    if awareness.all():
        stamps = pd.DatetimeIndex(pd.to_datetime(frame.timestamp, utc=True, errors="raise"))
    elif not awareness.any() and naive_timezone:
        stamps = pd.DatetimeIndex(pd.to_datetime(frame.timestamp, errors="raise")).tz_localize(
            naive_timezone, ambiguous="raise", nonexistent="raise").tz_convert("UTC")
    else:
        raise ValueError("Explicit timezone required; mixed aware/naive timestamps rejected")
    if stamps.isna().any() or stamps.duplicated().any():
        raise ValueError("Missing or duplicate timestamps")
    if (stamps != stamps.floor("min")).any():
        raise ValueError("Timestamps must be exact minute boundaries")
    frame = frame[OHLCV].apply(pd.to_numeric, errors="raise").astype(np.float64)
    values = frame.to_numpy()
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
        raise ValueError("Nonfinite/invalid price or negative volume")
    if ((frame.high < frame[["open", "low", "close"]].max(axis=1)) |
            (frame.low > frame[["open", "high", "close"]].min(axis=1))).any():
        raise ValueError("OHLC bounds invalid")
    frame.index = stamps
    return frame.sort_index()


def calendar_grid(start, end):
    import exchange_calendars as xc
    start, end = str(start), str(end)
    if start > end:
        raise ValueError("Calendar start follows end")
    # The library requires a nonempty range with start < end, even when the
    # caller audits just one session or a holiday.
    lower = (pd.Timestamp(start) - pd.Timedelta(days=7)).date()
    upper = (pd.Timestamp(end) + pd.Timedelta(days=7)).date()
    calendar = xc.get_calendar("XNYS", start=str(lower), end=str(upper))
    # Explicit [open, close) grids match Alpaca's bar-start convention and avoid
    # library-specific inclusion of a 16:00 minute or a half-day closing minute.
    schedule = calendar.schedule.loc[str(start):str(end)]
    grids = {str(day.date()): pd.date_range(row.open, row.close, freq="min", inclusive="left")
             for day, row in schedule.iterrows()}
    return grids


def audit_bars(frame, start=None, end=None, requested_dates=None, minimum_coverage=0.98):
    local = frame.index.tz_convert("America/New_York")
    start = start or str(local.min().date())
    end = end or str(local.max().date())
    if requested_dates is not None:
        start, end = min(start, min(requested_dates)), max(end, max(requested_dates))
    grids = calendar_grid(start, end)
    if requested_dates is not None:
        missing_sessions = set(requested_dates) - set(grids)
        if missing_sessions:
            raise ValueError(f"Requested non-session dates: {sorted(missing_sessions)}")
        grids = {date: grids[date] for date in requested_dates}
    if not grids:
        raise ValueError("No exchange sessions in requested range")
    expected_ns = np.concatenate([g.as_unit("ns").asi8 for g in grids.values()])
    expected = pd.DatetimeIndex(expected_ns, tz="UTC")
    regular = frame.index.isin(expected)
    daily = []
    for date, grid in grids.items():
        observed = grid.intersection(frame.index)
        daily.append({"date": date, "expected": len(grid), "observed": len(observed),
                      "coverage": len(observed) / len(grid),
                      "missing": len(grid.difference(observed)),
                      "early_close": len(grid) < 390,
                      "open_utc": grid[0].isoformat(),
                      "close_exclusive_utc": (grid[-1] + pd.Timedelta(minutes=1)).isoformat()})
    regular_frame = frame.loc[regular]
    coverage = len(regular_frame) / len(expected)
    report = {"calendar": "XNYS", "timezone": "America/New_York", "timestamp_label": "bar_start",
              "rows": len(frame), "regular_rows": len(regular_frame), "expected_rows": len(expected),
              "outside_requested_regular_hours": int((~regular).sum()),
              "regular_coverage": coverage, "minimum_daily_coverage": min(d["coverage"] for d in daily),
              "coverage_passed": coverage >= minimum_coverage and all(
                  d["coverage"] >= minimum_coverage for d in daily),
              "threshold": minimum_coverage, "daily": daily,
              "source_verified": False, "formal_eligible": False,
              "limitations": ["Structural coverage alone does not verify provenance, corrections or point-in-time availability.",
                              "No interpolation, outlier deletion or corporate-action adjustment applied."]}
    return regular_frame, report


@dataclass
class MarketData:
    bars: pd.DataFrame
    dates: np.ndarray
    returns: np.ndarray
    features: np.ndarray
    origins: np.ndarray
    lookback: int
    horizons: tuple
    feature_set: str
    audit: dict

    @property
    def timestamps(self):
        return self.bars.index.tz_localize(None).to_numpy(dtype="datetime64[ns]")

    def samples(self, dates, stride=1):
        if stride < 1:
            raise ValueError("Positive origin stride required")
        candidates = self.origins[np.isin(self.dates[self.origins], dates)]
        # Anchor to the exchange minute-of-day, never to missing-row positions.
        local = self.bars.index[candidates].tz_convert("America/New_York")
        offset = np.asarray(local.hour * 60 + local.minute - (9 * 60 + 30))
        return candidates[offset % stride == 0]

    def scaler(self, dates):
        values = self.features[np.isin(self.dates, dates)]
        values = values[np.isfinite(values).all(axis=1)]
        r = self.returns[np.isin(self.dates, dates)]
        r = r[np.isfinite(r)]
        if not len(values) or not len(r) or np.std(r) <= 0:
            raise ValueError("Training scaler has no positive finite return variance")
        center = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        std[std <= 0] = 1.0
        return {"center": center.tolist(), "std": std.tolist(), "target_std": float(np.std(r)),
                "fit_dates": list(dates), "feature_set": self.feature_set, "n_rows": len(values)}

    def inputs(self, origins, scaler):
        origins = np.asarray(origins, dtype=np.int64)
        idx = origins[:, None] + np.arange(1 - self.lookback, 1)
        x = ((self.features[idx] - scaler["center"]) / scaler["std"]).astype(np.float32)
        if not np.isfinite(x).all():
            raise ValueError("Invalid input window")
        return x

    def targets(self, origins):
        origins = np.asarray(origins, dtype=np.int64)
        steps = self.returns[origins[:, None] + np.arange(1, max(self.horizons) + 1)]
        if not np.isfinite(steps).all():
            raise ValueError("Invalid future target")
        return np.cumsum(steps, axis=1)[:, np.array(self.horizons) - 1]


def load_market_data(path, lookback=240, horizons=(1, 5, 12), feature_set="close", start=None, end=None):
    if feature_set not in FEATURE_NAMES or lookback < 1 or not horizons or min(horizons) < 1:
        raise ValueError("Invalid feature set or window lengths")
    frame = read_bars(path)
    bars, audit = audit_bars(frame, start, end)
    n = len(bars)
    if not n:
        raise ValueError("No regular-session bars")
    dates = bars.index.tz_convert("America/New_York").strftime("%Y-%m-%d").to_numpy()
    minute_ns = bars.index.as_unit("ns").asi8
    adjacent = np.r_[False, (np.diff(minute_ns) == 60 * 10**9) & (dates[1:] == dates[:-1])]
    close = bars.close.to_numpy()
    previous = np.roll(close, 1)
    previous[~adjacent] = np.nan
    returns = np.log(close / previous)
    if feature_set == "close":
        features = returns[:, None]
    else:
        features = np.column_stack([np.log(bars[c].to_numpy() / previous) for c in OHLCV[:4]] +
                                   [np.log1p(bars.volume.to_numpy())])
    idx = np.arange(n)
    starts = np.maximum.accumulate(np.where(~adjacent, idx, 0))
    ends = np.r_[np.flatnonzero(~adjacent)[1:] - 1, n - 1]
    run_end = np.repeat(ends, np.diff(np.r_[0, ends + 1]))
    origins = idx[(idx - starts >= lookback) & (run_end - idx >= max(horizons))]
    audit.update(source=str(Path(path).resolve()), sha256=digest(path), lookback_returns=lookback,
                 horizons=list(horizons), feature_set=feature_set, eligible_origins=len(origins),
                 within_session_gaps=int(np.sum((dates[1:] == dates[:-1]) & ~adjacent[1:])))
    # A short/half-day or sparse audited sample can legitimately have no windows.
    return MarketData(bars, dates, returns, features, origins, lookback, tuple(horizons), feature_set, audit)


def quarterly_splits(config):
    """Plan dates using the exchange schedule without reading any test prices."""
    p = config["periods"]
    if not (p["development_start"] <= p["development_end"] < p["confirmation_start"] <= p["confirmation_end"]):
        raise ValueError("Overlapping/reversed development and confirmation ranges")
    dates = np.array(sorted(calendar_grid(p["development_start"], p["confirmation_end"])))
    development = dates[(dates >= p["development_start"]) & (dates <= p["development_end"])]
    splits = []
    quarters = pd.period_range(p["validation_start"], p["development_end"], freq="Q")
    for q in quarters:
        evaluation = development[(development >= str(q.start_time.date())) &
                                 (development <= str(q.end_time.date()))]
        if len(evaluation):
            train = development[development < evaluation[0]]
            splits.append(_split(str(q), train, evaluation, config["purge_sessions"], "validation"))
    evaluation = dates[(dates >= p["confirmation_start"]) & (dates <= p["confirmation_end"])]
    splits.append(_split("historical_confirmation", development, evaluation,
                         config["purge_sessions"], "historical_confirmation"))
    return splits


def _split(name, train, evaluation, purge, kind):
    if purge < 1 or len(train) <= purge or not len(evaluation):
        raise ValueError("Insufficient dates or invalid session purge")
    return {"name": name, "kind": kind, "train_dates": train[:-purge].tolist(),
            "purged_dates": train[-purge:].tolist(), "eval_dates": evaluation.tolist()}


def fit_calibration_dates(train_dates, calibration_sessions=20, purge_sessions=1):
    if len(train_dates) <= calibration_sessions + purge_sessions:
        raise ValueError("Insufficient training dates for disjoint calibration tail")
    return (train_dates[:-(calibration_sessions + purge_sessions)],
            train_dates[-calibration_sessions:])
