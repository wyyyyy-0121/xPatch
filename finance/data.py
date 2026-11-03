"""Session-aware RV labels and strictly historical, paired model inputs."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from finance_v2.data import calendar_grid, read_bars
from .io import ROOT, read_json, sha


ASSETS = ("AAPL", "SPY", "QQQ")


def rolling_sum(values, length):
    """A missing ordinary observation invalidates the entire rolling window."""
    values = np.asarray(values, dtype=np.float64)
    if length < 1:
        raise ValueError("Positive rolling length required")
    valid = np.isfinite(values)
    total = np.concatenate([np.zeros_like(values[:1]), np.cumsum(np.where(valid, values, 0), axis=0)], axis=0)
    counts = np.concatenate([np.zeros_like(values[:1], dtype=np.int64), np.cumsum(valid, axis=0)], axis=0)
    result = np.full_like(values, np.nan)
    if len(values) >= length:
        sums = total[length:] - total[:-length]
        complete = counts[length:] - counts[:-length] == length
        result[length - 1:] = np.where(complete, sums, np.nan)
    return result


@dataclass
class RVData:
    timestamps: pd.DatetimeIndex  # minute ends, with structural first bars removed
    dates: np.ndarray
    days: np.ndarray
    day_index: np.ndarray
    returns: np.ndarray
    volume: np.ndarray
    log_range: np.ndarray
    close_by_date: dict
    regular_audits: dict = field(default_factory=dict)
    _rolls: dict = field(default_factory=dict)
    _daily: np.ndarray | None = None

    @classmethod
    def from_frames(cls, frames, start, end):
        if set(frames) != set(ASSETS):
            raise ValueError("Paired comparison requires all three registered assets")
        grids = calendar_grid(start, end)
        if not grids:
            raise ValueError("No sessions in requested range")
        days = np.array(list(grids))
        grid = pd.DatetimeIndex(np.concatenate([g.as_unit("ns").asi8 for g in grids.values()]), tz="UTC")
        ends = grid + pd.Timedelta(minutes=1)
        dates = np.asarray(ends.tz_convert("America/New_York").strftime("%Y-%m-%d"))
        same_day = np.r_[False, dates[1:] == dates[:-1]]
        adjacent = np.r_[False, np.diff(grid.asi8) == 60 * 10**9]
        # Every first bar has no same-session previous close, regardless of its price.
        keep = same_day & adjacent
        returns, volume, ranges, audits = [], [], [], {}
        for asset in ASSETS:
            frame = frames[asset]
            if frame.index.tz is None or frame.index.duplicated().any():
                raise ValueError("Asset timestamps must be aware and unique")
            aligned = frame.reindex(grid)
            prices = aligned.close.to_numpy(dtype=np.float64)
            observed = np.isfinite(prices)
            ret = np.r_[np.nan, np.diff(np.log(prices))]
            ret[~keep] = np.nan
            returns.append(ret[keep])
            volume.append(aligned.volume.to_numpy(dtype=np.float64)[keep])
            ranges.append(np.log(aligned.high.to_numpy(dtype=np.float64) / aligned.low.to_numpy(dtype=np.float64))[keep])
            audits[asset] = {"regular_rows": int(observed.sum()), "expected_rows": len(grid),
                "missing_rows": int((~observed).sum())}
        dates = dates[keep]
        mapping = {day: i for i, day in enumerate(days)}
        return cls(ends[keep], dates, days, np.array([mapping[day] for day in dates]),
            np.column_stack(returns), np.column_stack(volume), np.column_stack(ranges),
            {day: grid_day[-1] + pd.Timedelta(minutes=1) for day, grid_day in grids.items()}, audits)

    @classmethod
    def from_manifest(cls, manifest="delivery/data.json"):
        receipt = read_json(ROOT / manifest)
        if not receipt["passed_for_retrospective_research"]:
            raise ValueError("Input audit has not passed")
        frames = {}
        for asset in ASSETS:
            entry = receipt["assets"][asset]
            path = ROOT / entry["csv"]
            if sha(path) != entry["csv_sha256"]:
                raise ValueError(f"Input fingerprint changed: {asset}")
            frames[asset] = read_bars(path)
        return cls.from_frames(frames, receipt["start"], receipt["end"])

    def rv(self, length=30):
        if length not in self._rolls:
            self._rolls[length] = rolling_sum(self.returns ** 2, length)
        return self._rolls[length]

    def daily_rv(self):
        if self._daily is None:
            valid = np.isfinite(self.returns)
            totals = np.zeros((len(self.days), len(ASSETS)), dtype=np.float64)
            counts = np.zeros_like(totals, dtype=np.int64)
            np.add.at(totals, self.day_index, np.where(valid, self.returns ** 2, 0))
            np.add.at(counts, self.day_index, valid)
            expected = np.bincount(self.day_index, minlength=len(self.days))[:, None]
            self._daily = np.where((counts == expected) & (expected > 0), totals, np.nan)
        return self._daily

    def previous_sessions_rv(self, sessions):
        daily = rolling_sum(self.daily_rv(), sessions)
        values = np.full_like(daily, np.nan)
        values[1:] = daily[:-1]
        return values[self.day_index]

    def origins(self, dates=None, stride=1, context=512, history_sessions=22, require_target=True):
        if stride < 1 or context < 1 or history_sessions < 0:
            raise ValueError("Invalid origin qualification settings")
        local = self.timestamps.tz_convert("America/New_York")
        clock = np.asarray(local.hour * 60 + local.minute)
        first = 9 * 60 + 45
        # Origin u indexes a future, not yet observed return. Last observed is u-1.
        close_ns = np.array([self.close_by_date[day].value for day in self.days])
        before_close = self.timestamps.asi8 + 30 * 60 * 10**9 <= close_ns[self.day_index]
        eligible = (clock >= first) & before_close & ((clock - first) % stride == 0)
        idx = np.arange(len(self.timestamps))
        previous = idx - 1
        safe_previous = np.maximum(previous, 0)
        history = rolling_sum(self.returns ** 2, context + 29)
        eligible &= (previous >= context + 28) & np.isfinite(history[safe_previous]).all(axis=1)
        # Check the information lag against actual minute stamps, not packed row numbers.
        eligible &= self.timestamps[safe_previous] == self.timestamps - pd.Timedelta(minutes=1)
        if history_sessions:
            eligible &= np.isfinite(self.previous_sessions_rv(history_sessions)).all(axis=1)
        if dates is not None:
            eligible &= np.isin(self.dates, dates)
        if require_target:
            future = idx + 30
            safe_future = np.minimum(future, len(idx) - 1)
            eligible &= (future < len(idx)) & (self.dates[safe_future] == self.dates)
            eligible &= self.timestamps[safe_future] == self.timestamps + pd.Timedelta(minutes=30)
            eligible &= np.isfinite(self.rv(30)[safe_future, 0])
        return np.flatnonzero(eligible)

    def targets(self, origins):
        origins = np.asarray(origins, dtype=np.int64)
        terminal = origins + 30
        if (terminal >= len(self.timestamps)).any() or (origins < 0).any():
            raise ValueError("Target outside data grid")
        if not np.all(self.timestamps[terminal] == self.timestamps[origins] + pd.Timedelta(minutes=30)):
            raise ValueError("Target crosses a nontrading gap")
        if not np.all(self.dates[terminal] == self.dates[origins]):
            raise ValueError("Overnight RV target prohibited")
        result = self.rv(30)[terminal, 0]
        if not np.isfinite(result).all():
            raise ValueError("Target contains missing minutes")
        return result

    def context(self, origins, length, series):
        origins = np.asarray(origins, dtype=np.int64)
        indices = origins[:, None] - 1 + np.arange(1 - length, 1)
        if (indices < 0).any() or (indices >= len(self.timestamps)).any():
            raise ValueError("Context outside available grid")
        if not np.all(self.timestamps[indices[:, -1]] == self.timestamps[origins] - pd.Timedelta(minutes=1)):
            raise ValueError("Information cutoff is not exactly one minute before origin")
        result = series[indices]
        if not np.isfinite(result).all():
            raise ValueError("Context includes an ordinary missing minute")
        return result
