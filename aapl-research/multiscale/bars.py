"""Build observed, completed bars and path labels without compressing missing data."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .spec import DEV_END, NY, ROOT

MINUTE_NS = 60_000_000_000
DATA = ROOT / "outputs" / "fusion_next" / "data"


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


@dataclass(frozen=True)
class Bars:
    """A bar is available only at ``end_ns``; session labels are local dates."""
    frequency: str
    start_ns: np.ndarray
    end_ns: np.ndarray
    close: np.ndarray
    observed: np.ndarray
    session_date: np.ndarray

    def __post_init__(self):
        arrays = (self.start_ns, self.end_ns, self.close, self.observed, self.session_date)
        n = len(self.start_ns)
        if n == 0 or any(len(a) != n or a.ndim != 1 for a in arrays):
            raise ValueError("Bars must be nonempty equal-length one-dimensional arrays")
        if (np.diff(self.start_ns) <= 0).any() or (np.diff(self.end_ns) <= 0).any():
            raise ValueError("Bar timestamps must be strictly increasing")
        if (self.start_ns >= self.end_ns).any() or (self.start_ns[1:] < self.end_ns[:-1]).any():
            raise ValueError("Bar intervals must be positive and nonoverlapping")
        if self.observed.dtype != np.bool_:
            raise ValueError("observed must be a boolean mask")
        if self.session_date.dtype.kind != "U":
            raise ValueError("Session dates must be fixed-width Unicode, not pickled objects")
        if not np.isfinite(self.close[self.observed]).all() or (self.close[self.observed] <= 0).any():
            raise ValueError("Observed close values must be finite and positive")
        if (self.session_date[1:] < self.session_date[:-1]).any():
            raise ValueError("Session dates must be ordered")


def session_bounds(labels: np.ndarray):
    if not len(labels):
        raise ValueError("No session labels")
    start = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    end = np.r_[start[1:] - 1, len(labels) - 1]
    if len(np.unique(labels[start])) != len(start):
        raise ValueError("A session may not reappear after a later session")
    return start, end


def load_admitted_minutes():
    """Recheck the exact prior AAPL shard and calendar; never open sealed inputs."""
    report = read_json(ROOT / "outputs" / "fusion_next" / "data_admission.json")
    audit = read_json(ROOT / "outputs" / "fusion_next" / "data_audit.json")
    manifest = read_json(DATA / "AAPL" / "manifest.json")
    protocol = read_json(ROOT / "protocol_locked.json")
    if not (report.get("passed") and audit.get("complete") and audit.get("passed")
            and manifest.get("admitted") and report.get("all_symbols_Q3_blocked")
            and audit.get("Q3_values_read") is False and manifest.get("Q3_values_read") is False):
        raise ValueError("Prior AAPL development admission is not in the expected state")
    source = Path(manifest["source"]["source_csv_read_only"]).resolve()
    if source.is_relative_to(ROOT / "holdout_sealed"):
        raise ValueError("Sealed source is not allowed")
    if source != Path(protocol["source_csv_read_only"]).resolve():
        raise ValueError("Source path differs from original development protocol")
    source_hash = file_digest(source)
    if source_hash != manifest["source"]["sha256"] or source_hash != protocol["source_csv_expected_sha256"]:
        raise ValueError("Source bytes differ from the prior admissible AAPL data")
    for filename in ("timestamps_ns.npy", "dates.npy", "session_start.npy", "session_end.npy"):
        if file_digest(DATA / filename) != audit["global_files_sha256"][filename]:
            raise ValueError(f"Global calendar array identity changed: {filename}")
    for filename in ("close.npy", "observed.npy"):
        if file_digest(DATA / "AAPL" / filename) != manifest["files"][filename]:
            raise ValueError(f"AAPL prepared array identity changed: {filename}")
    values = {name: np.load(DATA / name, allow_pickle=False)
              for name in ("timestamps_ns.npy", "dates.npy", "session_start.npy", "session_end.npy")}
    close = np.load(DATA / "AAPL" / "close.npy", allow_pickle=False)
    observed = np.load(DATA / "AAPL" / "observed.npy", allow_pickle=False)
    starts = values["timestamps_ns.npy"]
    dates = values["dates.npy"]
    session_start, session_end = session_bounds(dates)
    flags = np.zeros(len(starts), dtype=bool)
    flags[session_start] = True
    if not np.array_equal(flags, values["session_start.npy"]):
        raise ValueError("Session-start flags do not match dates")
    expected_ends = np.repeat(session_end, session_end - session_start + 1)
    if not np.array_equal(expected_ends, values["session_end.npy"]):
        raise ValueError("Session-end indices do not match dates")
    local_dates = pd.to_datetime(starts, utc=True).tz_convert(NY).strftime("%Y-%m-%d").to_numpy()
    if not np.array_equal(local_dates, dates) or dates[-1] > DEV_END:
        raise ValueError("Calendar timezone or Q3 cutoff mismatch")
    from aapl.data import sessions
    grids = sessions(str(dates[0]), DEV_END)
    expected = np.concatenate([g.as_unit("ns").asi8 for g in grids.values()])
    if not np.array_equal(starts, expected):
        raise ValueError("Prepared minutes do not match the independent XNYS session grid")
    if int(observed.sum()) != manifest["observed_rows"] or len(starts) != manifest["regular_rows"]:
        raise ValueError("AAPL observed count differs from admitted manifest")
    bars = Bars("minute", starts, starts + MINUTE_NS, close, observed, dates)
    evidence = {"source_sha256": source_hash, "source_rows": manifest["source"]["rows"],
                "admission_sha256": file_digest(ROOT / "outputs" / "fusion_next" / "data_admission.json"),
                "audit_sha256": file_digest(ROOT / "outputs" / "fusion_next" / "data_audit.json"),
                "array_sha256": {name: audit["global_files_sha256"][name]
                                 for name in ("timestamps_ns.npy", "dates.npy", "session_start.npy", "session_end.npy")},
                "close_sha256": manifest["files"]["close.npy"],
                "observed_sha256": manifest["files"]["observed.npy"],
                "Q3_values_read": False}
    return bars, evidence


def aggregate_intraday(minutes: Bars):
    """Use complete 60-minute blocks; discard residual minutes from hourly bars."""
    if minutes.frequency != "minute":
        raise ValueError("Expected minute bars")
    start, end = session_bounds(minutes.session_date)
    bad_clock = np.flatnonzero((np.diff(minutes.start_ns) != MINUTE_NS)
                               & (minutes.session_date[1:] == minutes.session_date[:-1]))
    if len(bad_clock) or not np.all(minutes.end_ns - minutes.start_ns == MINUTE_NS):
        raise ValueError("A regular-session minute grid has a gap or wrong duration")
    missing_prefix = np.r_[0, np.cumsum(~minutes.observed, dtype=np.int64)]
    full_counts = (end - start + 1) // 60
    hour_start = np.concatenate([first + 60 * np.arange(count) for first, count in zip(start, full_counts)])
    hour_end = hour_start + 59
    hour_valid = missing_prefix[hour_end + 1] == missing_prefix[hour_start]
    hour = Bars("hour", minutes.start_ns[hour_start], minutes.end_ns[hour_end],
                minutes.close[hour_end], hour_valid, minutes.session_date[hour_start])
    day_valid = missing_prefix[end + 1] == missing_prefix[start]
    day = Bars("day", minutes.start_ns[start], minutes.end_ns[end],
               minutes.close[end], day_valid, minutes.session_date[start])
    metadata = {"sessions": len(start), "full_hour_bars": len(hour_start),
                "residual_minutes_total": int(np.sum((end - start + 1) % 60)),
                "hours_per_session_counts": {str(k): int(np.sum(full_counts == k))
                                             for k in np.unique(full_counts)},
                "hour_bar_policy": "60 consecutive regular-session minutes from open; residual excluded",
                "hour_label_policy": "subsequent complete bars may cross sessions",
                "day_close_policy": "last regular-session minute close, not official auction close"}
    return hour, day, metadata


def aggregate_months(days: Bars):
    """Include a month only as observed when all of its XNYS sessions are present."""
    if days.frequency != "day":
        raise ValueError("Expected daily bars")
    periods = pd.PeriodIndex(days.session_date, freq="M")
    expected_periods = pd.period_range(periods[0], periods[-1], freq="M")
    if not periods.unique().equals(expected_periods):
        raise ValueError("Missing whole calendar month in daily data")
    import exchange_calendars as xc
    first_calendar = str(expected_periods[0].start_time.date())
    last_calendar = str(expected_periods[-1].end_time.date())
    calendar = xc.get_calendar("XNYS", start=expected_periods[0].start_time - pd.Timedelta(days=7),
                               end=expected_periods[-1].end_time.normalize() + pd.Timedelta(days=7))
    calendar_dates = set(calendar.schedule.loc[first_calendar:last_calendar].index.strftime("%Y-%m-%d"))
    first, last, valid = [], [], []
    for period in expected_periods:
        idx = np.flatnonzero(periods == period)
        expected_dates = sorted(d for d in calendar_dates if d.startswith(str(period)))
        actual_dates = list(days.session_date[idx])
        first.append(int(idx[0])); last.append(int(idx[-1]))
        valid.append(actual_dates == expected_dates and bool(days.observed[idx].all()))
    first = np.asarray(first, dtype=np.int64)
    last = np.asarray(last, dtype=np.int64)
    return Bars("month", days.start_ns[first], days.end_ns[last], days.close[last],
                np.asarray(valid, dtype=bool), days.session_date[last])


def eligible_origins(bars: Bars, lookback: int, horizon: int, *, same_session: bool):
    """All history and every path label must be observed; no future filling."""
    if lookback < 1 or horizon < 1:
        raise ValueError("Lookback and horizon must be positive")
    if len(bars.close) < lookback + horizon:
        return np.array([], dtype=np.int64)
    idx = np.arange(lookback - 1, len(bars.close) - horizon, dtype=np.int64)
    missing = np.r_[0, np.cumsum(~bars.observed, dtype=np.int64)]
    valid = missing[idx + horizon + 1] == missing[idx - lookback + 1]
    if same_session:
        valid &= bars.session_date[idx] == bars.session_date[idx + horizon]
    return idx[valid]


def targets(bars: Bars, origins, horizon: int, mode: str):
    if horizon < 1:
        raise ValueError("Horizon must be positive")
    idx = np.asarray(origins, dtype=np.int64)
    if idx.ndim != 1 or (idx < 0).any() or (idx + horizon >= len(bars.close)).any():
        raise ValueError("Origin outside series")
    future_idx = idx[:, None] + np.arange(1, horizon + 1)
    if not bars.observed[idx].all() or not bars.observed[future_idx].all():
        raise ValueError("Target requires observed origin and every future bar")
    if bars.frequency == "minute" and not np.all(bars.session_date[idx] == bars.session_date[idx + horizon]):
        raise ValueError("Minute targets must remain in the same session")
    close = bars.close[future_idx]
    if mode == "future_close":
        return close.copy()
    if mode == "future_log_return":
        return np.log(close / bars.close[idx, None])
    raise ValueError("Unknown target mode")
