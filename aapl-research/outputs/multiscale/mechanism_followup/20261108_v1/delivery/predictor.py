"""Standalone CPU inference from 480 completed, consistently split-adjusted closes."""
from pathlib import Path
import json
import numpy as np
import torch
from scipy.signal import lfilter


def features(log_history):
    x = np.asarray(log_history, dtype='float64')
    if x.ndim != 2 or x.shape[1] != 480 or not np.isfinite(x).all():
        raise ValueError('Expected finite log-relative histories with shape (batch, 480)')
    def ema(span):
        a = 2/(span+1)
        v, _ = lfilter([a], [1, -(1-a)], x, axis=1, zi=((1-a)*x[:, 0])[:, None])
        return v
    e8, e32 = ema(8), ema(32)
    r = np.diff(x, axis=1)
    s60 = np.maximum(r[:, -60:].std(axis=1), 1e-6)
    trend = np.column_stack((x[:, -1]-e8[:, -1], e8[:, -1]-e32[:, -1],
        e8[:, -1]-e8[:, -9], e32[:, -1]-e32[:, -33]))/s60[:, None]
    gate = np.column_stack((r[:, -8:].std(axis=1)/s60,
        r[:, -32:].std(axis=1)/np.maximum(r[:, -120:].std(axis=1), 1e-6),
        np.abs(e8[:, -1]-e32[:, -1])/s60, np.abs(x[:, -1]-e8[:, -1])/s60))
    residual = (x-e32)/s60[:, None]
    return trend.astype('float32'), residual.astype('float32'), gate.astype('float32')


class Predictor:
    """An archived fold/seed predictor; not retrained on all available history."""
    def __init__(self, folder):
        self.folder = Path(folder)
        self.info = json.loads((self.folder/'preprocess.json').read_text())
        self.network = torch.jit.load(str(self.folder/'network.pt'), map_location='cpu').eval()

    def predict_log_relative(self, history):
        t, r, g = features(history)
        values = []
        for key, value in zip(('trend', 'residual', 'gate'), (t, r, g)):
            if key in self.info['feature_stats']:
                st = self.info['feature_stats'][key]
                value = ((value-np.asarray(st['mean']))/np.asarray(st['std'])).astype('float32')
            values.append(torch.from_numpy(value))
        with torch.inference_mode():
            p = self.network(*values).numpy()
        scaler = self.info['scaler']
        return p*np.asarray(scaler['y_std'])+np.asarray(scaler['y_mean'])

    def predict(self, closes):
        x = np.asarray(closes, dtype='float64')
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != 480 or x.shape[0] == 0:
            raise ValueError('Supply exactly 480 completed historical closes per sample')
        if not np.isfinite(x).all() or (x <= 0).any():
            raise ValueError('Closes must be finite and strictly positive')
        log = np.log(x)
        return self.predict_log_relative(log-log[:, -1:])
