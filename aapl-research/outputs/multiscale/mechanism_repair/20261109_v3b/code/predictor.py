"""Origin-only deterministic context/history Ridge predictor; no target access."""
from pathlib import Path
import numpy as np
import pandas as pd


def feature_names(horizon, lookback):
    lags = sorted({1, max(2, horizon // 4), max(3, horizon // 2), horizon, 2 * horizon, 4 * horizon, lookback - 1})
    assert len(lags) == 7 and max(4 * horizon, max(lags)) <= lookback - 1
    return ([f'return_lag_{k}' for k in lags] + [f'rms_{horizon}', f'rms_{4*horizon}'] +
            ['clock_sin1', 'clock_cos1', 'clock_sin2', 'clock_cos2', 'Monday', 'Tuesday', 'Wednesday', 'Thursday'])


def features(history, completion_ns, horizon):
    x = np.asarray(history, dtype=np.float64)
    stamps = np.asarray(completion_ns, dtype=np.int64)
    if x.ndim != 2 or stamps.shape != (len(x),) or not np.isfinite(x).all():
        raise ValueError('Supply finite completed log-relative histories and one origin completion timestamp per row')
    if not np.allclose(x[:, -1], 0., rtol=0., atol=1e-12):
        raise ValueError('Each input history must end at zero log-relative close')
    lags = sorted({1, max(2, horizon // 4), max(3, horizon // 2), horizon, 2*horizon, 4*horizon, x.shape[1]-1})
    if len(lags) != 7 or max(4*horizon, max(lags)) > x.shape[1]-1:
        raise ValueError('Frozen horizon lag set exceeds completed input window')
    values = [-x[:, -1-k] for k in lags]
    for k in (horizon, 4*horizon):
        increments = np.diff(x[:, -(k+1):], axis=1)
        values.append(np.sqrt(np.mean(increments**2, axis=1)))
    ny = pd.to_datetime(stamps, utc=True).tz_convert('America/New_York')
    minute = ny.hour.to_numpy() * 60 + ny.minute.to_numpy() + ny.second.to_numpy()/60
    u = (minute - 570.)/390.
    if np.any((ny.dayofweek.to_numpy() > 4) | (u <= 0) | (u > 1)):
        raise ValueError('Origin completion must be within the frozen weekday regular-session clock')
    values += [np.sin(2*np.pi*u), np.cos(2*np.pi*u), np.sin(4*np.pi*u), np.cos(4*np.pi*u)]
    values += [(ny.dayofweek.to_numpy() == day).astype(float) for day in range(4)]
    out = np.column_stack(values)
    assert out.shape == (len(x), 17) and np.isfinite(out).all()
    return out


class Predictor:
    def __init__(self, model_file):
        with np.load(Path(model_file), allow_pickle=False) as a:
            self.coef = a['coef'].copy(); self.x_mean = a['x_mean'].copy()
            self.x_std = a['x_std'].copy(); self.y_mean = a['y_mean'].copy()
            self.columns = a['columns'].copy(); self.horizon = int(a['horizon'])
            self.alpha = float(a['alpha'])

    def predict(self, completed_history, origin_completion_ns, *, shrunk=False):
        x = features(completed_history, origin_completion_ns, self.horizon)
        z = (x[:, self.columns] - self.x_mean)/self.x_std
        result = self.y_mean + z @ self.coef
        if shrunk:
            result *= self.alpha
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite forecast')
        return result
