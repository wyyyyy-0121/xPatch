"""Causal feature windows and fit-only seasonal/scaling state for RV30."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import ASSETS, rolling_sum


FLOOR = 1e-12
REPRESENTATIONS = {"level_rv", "log_rv", "log_seasonal_ratio"}
CLOCKS = ("clock_sin1", "clock_cos1", "clock_sin2", "clock_cos2", "minutes_to_close")


def daily_totals(data, values):
    valid = np.isfinite(values)
    totals = np.zeros((len(data.days), len(ASSETS)))
    counts = np.zeros_like(totals, dtype=np.int64)
    np.add.at(totals, data.day_index, np.where(valid, values, 0))
    np.add.at(counts, data.day_index, valid)
    expected = np.bincount(data.day_index, minlength=len(data.days))[:, None]
    return np.where((counts == expected) & (expected > 0), totals, np.nan)


def clock_bucket(stamps, bucket_minutes=15):
    local = stamps.tz_convert("America/New_York")
    return np.asarray((local.hour * 60 + local.minute - 570) // bucket_minutes, dtype=np.int64)


@dataclass
class FitState:
    fit_dates: list
    seasonal_profile: list
    log_volume_center: list
    series_center: dict
    series_std: dict
    target_level_scale: float
    bucket_minutes: int
    floor: float = FLOOR

    @classmethod
    def fit(cls, data, dates, settings):
        dates = list(dates)
        mask = np.isin(data.dates, dates)
        if not mask.any() or dates != sorted(set(dates)):
            raise ValueError("Fit dates must be nonempty, ordered and unique")
        r2 = data.returns ** 2
        global_mean = np.nanmean(r2[mask], axis=0)
        if not np.isfinite(global_mean).all() or (global_mean <= 0).any():
            raise ValueError("Fit data has no positive finite variance")
        bucket_minutes = settings["bucket_minutes"]
        buckets = clock_bucket(data.timestamps, bucket_minutes)
        shrink = settings["global_mean_shrinkage"]
        floor = settings.get("variance_floor", FLOOR)
        profile = []
        for bucket in range(390 // bucket_minutes + 1):
            selected = mask & (buckets == bucket)
            mean = np.nanmean(r2[selected], axis=0) if selected.any() else global_mean
            mean = np.where(np.isfinite(mean), mean, global_mean)
            profile.append(np.maximum((1 - shrink) * mean + shrink * global_mean, floor))
        volume_center = np.nanmean(np.log1p(data.volume[mask]), axis=0)
        state = cls(dates, np.asarray(profile).tolist(), volume_center.tolist(), {}, {},
            float(np.nanmean(data.rv(30)[mask, 0])), bucket_minutes, floor)
        for representation in ("log_rv", "log_seasonal_ratio"):
            values = state.series(data, representation)[mask]
            center = np.nanmean(values, axis=0)
            std = np.nanstd(values, axis=0)
            if not np.isfinite(center).all() or not np.isfinite(std).all():
                raise ValueError("Fit-only series scale is nonfinite")
            state.series_center[representation] = center.tolist()
            state.series_std[representation] = np.where(std > 1e-8, std, 1.0).tolist()
        if not np.isfinite(state.target_level_scale) or state.target_level_scale <= 0:
            raise ValueError("Invalid training target scale")
        return state

    def dictionary(self):
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}

    def _bind(self, data):
        if getattr(self, "_bound_data", None) is not data:
            for key in list(vars(self)):
                if key.startswith("_"):
                    delattr(self, key)
            self._bound_data = data
            self._expected = {}

    def seasonal_minutes(self, data):
        self._bind(data)
        if not hasattr(self, "_minutes"):
            self._minutes = np.asarray(self.seasonal_profile)[clock_bucket(data.timestamps, self.bucket_minutes)]
        return self._minutes

    def expected(self, data, window, sessions=False):
        self._bind(data)
        key = (window, sessions)
        if key not in self._expected:
            if sessions:
                totals = daily_totals(data, self.seasonal_minutes(data))
                aggregate = rolling_sum(totals, window) / window
                prior = np.full_like(aggregate, np.nan)
                prior[1:] = aggregate[:-1]
                values = prior[data.day_index]
            else:
                values = rolling_sum(self.seasonal_minutes(data), window)
            self._expected[key] = values
        return self._expected[key]

    def future_season(self, data, origins, asset=0):
        # Compute future calendar positions without consulting future bars/labels.
        stamps = pd.DatetimeIndex((data.timestamps[np.asarray(origins)].asi8[:, None] +
            np.arange(1, 31) * 60 * 10**9).reshape(-1), tz="UTC")
        lookup = np.asarray(self.seasonal_profile)[clock_bucket(stamps, self.bucket_minutes), asset]
        return lookup.reshape(-1, 30).sum(axis=1)

    def series(self, data, representation, standardized=False):
        self._bind(data)
        if representation not in ("log_rv", "log_seasonal_ratio"):
            raise ValueError("Sequence inputs require a log representation")
        cache_name = "_series_" + representation
        if not hasattr(self, cache_name):
            rv = data.rv(30)
            if representation == "log_seasonal_ratio":
                rv = rv / self.expected(data, 30)
            setattr(self, cache_name, np.log(np.maximum(rv, self.floor)))
        values = getattr(self, cache_name)
        if standardized:
            return (values - self.series_center[representation]) / self.series_std[representation]
        return values

    def labels(self, data, origins, representation):
        y = data.targets(origins)
        if representation == "level_rv":
            return y / self.target_level_scale
        if representation == "log_seasonal_ratio":
            y = y / self.future_season(data, origins)
        if representation not in REPRESENTATIONS:
            raise ValueError("Unknown target representation")
        return np.log(np.maximum(y, self.floor))

    def inverse(self, data, origins, values, representation):
        values = np.asarray(values, dtype=np.float64)
        if representation == "level_rv":
            result = values * self.target_level_scale
        elif representation in ("log_rv", "log_seasonal_ratio"):
            with np.errstate(over="raise", invalid="raise"):
                result = np.exp(values)
            if representation == "log_seasonal_ratio":
                result *= self.future_season(data, origins)
        else:
            raise ValueError("Unknown target representation")
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite forecast; evaluation rows must not be dropped")
        return np.maximum(result, self.floor)


class FeatureStore:
    """Stateless past windows are shared across fit states; no fitted full-data stats."""
    def __init__(self, data, config):
        self.data, self.config = data, config
        self.cache, self.daily_cache = {}, {}

    def daily(self, key, values, sessions=1):
        cache_key = (key, sessions)
        if cache_key not in self.daily_cache:
            data = self.data
            if (key, 0) not in self.daily_cache:
                self.daily_cache[(key, 0)] = daily_totals(data, values)
            aggregate = rolling_sum(self.daily_cache[(key, 0)], sessions) / sessions
            previous = np.full_like(aggregate, np.nan)
            previous[1:] = aggregate[:-1]
            self.daily_cache[cache_key] = previous[data.day_index]
        return self.daily_cache[cache_key]

    def raw(self, name):
        if name in self.cache:
            return self.cache[name]
        data = self.data
        if name.startswith("log_rv_day"):
            result = self.daily("r2", data.returns ** 2, int(name.split("day")[1]))
        elif name.startswith("log_rv_") and "lag" not in name:
            result = data.rv(int(name.rsplit("_", 1)[1]))
        elif name.startswith("semivariance_"):
            negative = "negative" in name
            values = np.where(np.isfinite(data.returns),
                np.where(data.returns < 0 if negative else data.returns >= 0, data.returns ** 2, 0), np.nan)
            result = self.daily("neg" if negative else "pos", values) if name.endswith("day1") else rolling_sum(values, 30)
        elif name.startswith("return_"):
            result = self.daily("return", data.returns) if name.endswith("day1") else rolling_sum(data.returns, int(name.rsplit("_", 1)[1]))
        elif name.startswith("relative_log_volume_"):
            window = int(name.rsplit("_", 1)[1])
            result = np.log1p(rolling_sum(data.volume, window) / window)
        elif name.startswith("log_high_low_range_"):
            # Sum of intrabar log(high/low); no overnight price level is involved.
            result = rolling_sum(data.log_range, int(name.rsplit("_", 1)[1]))
        elif name.startswith("log_rv30_lag"):
            lag = int(name.split("lag")[1])
            result = np.full_like(data.rv(), np.nan)
            result[lag:] = data.rv()[:-lag]
        else:
            raise ValueError(f"Unknown registered feature: {name}")
        self.cache[name] = result
        return result

    def names(self, model, params):
        if model == "har":
            names = list(self.config["features"]["har_base"])
            if params["semivariance"]:
                names += ["semivariance_negative_30", "semivariance_positive_30"]
        else:
            names = list(self.config["features"]["compact"])
            if params.get("feature_profile", "compact") == "rich":
                for name in self.config["features"]["rich_additions"]:
                    names.extend([f"log_rv30_lag{lag}" for lag in (1, 5, 15, 30, 60)]
                        if name == "log_rv30_lags_1_5_15_30_60" else [name])
        return names

    def matrix(self, origins, state, panel, model, params):
        data, representation = self.data, params["representation"]
        origins = np.asarray(origins, dtype=np.int64)
        last = origins - 1
        if (last < 0).any() or not np.all(data.timestamps[last] == data.timestamps[origins] - pd.Timedelta(minutes=1)):
            raise ValueError("Invalid information cutoff")
        names = self.names(model, params)
        channels = [ASSETS.index(asset) for asset in self.config["feature_sets"][panel]]
        columns, labels = [], []
        for asset in channels:
            for name in names:
                if name in CLOCKS:
                    continue
                values = self.raw(name)[last, asset].copy()
                if name.startswith("log_rv") or name.startswith("semivariance"):
                    if representation == "log_seasonal_ratio":
                        if "day" in name:
                            sessions = int(name.split("day")[1])
                            denominator = state.expected(data, sessions, sessions=True)[last, asset]
                        else:
                            window = int(name.rsplit("_", 1)[1]) if name.startswith("log_rv_") else 30
                            if "lag" in name:
                                lag = int(name.split("lag")[1])
                                denominator = state.expected(data, 30)[last - lag, asset]
                            else:
                                denominator = state.expected(data, window)[last, asset]
                        values /= denominator
                    if not (model == "har" and representation == "level_rv"):
                        values = np.log(np.maximum(values, state.floor))
                elif name.startswith("relative_log_volume"):
                    values -= state.log_volume_center[asset]
                columns.append(values)
                labels.append(f"{ASSETS[asset]}:{name}")
        local = data.timestamps[origins].tz_convert("America/New_York")
        minute = np.asarray(local.hour * 60 + local.minute) - 570
        for name in names:
            if name not in CLOCKS:
                continue
            if name == "minutes_to_close":
                closes = np.array([data.close_by_date[day].value for day in data.days])
                values = (closes[data.day_index[origins]] - data.timestamps[origins].asi8) / (60 * 10**9)
            else:
                harmonic = int(name[-1])
                values = (np.sin if "sin" in name else np.cos)(2 * np.pi * harmonic * minute / 390)
            columns.append(values)
            labels.append(name)
        matrix = np.column_stack(columns)
        if not np.isfinite(matrix).all():
            raise ValueError("Nonfinite registered input; no selective forecast-row deletion")
        return matrix, labels
