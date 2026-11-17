"""Past-input-only inference for a registered mechanism-study checkpoint."""
from pathlib import Path
import sys
import numpy as np
import torch

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE/'reference_source'))
from multiscale.models import make_model
from tiny_architecture import TinyPath


class Predictor:
    def __init__(self, checkpoint):
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.cfg = saved['cfg']; self.scaler = saved['scaler']
        self.ym = np.asarray(self.scaler['y_mean']); self.ys = np.asarray(self.scaler['y_std'])
        if self.cfg['model'] == 'tiny':
            self.net = TinyPath(self.cfg['horizon'], 'long', -self.ym/self.ys)
        else:
            self.net = make_model(self.cfg['model'], self.cfg['lookback'], self.cfg['horizon'])
        self.net.load_state_dict(saved['model']); self.net.eval()

    def predict_prepared(self, history, current_close, *, shrink_alpha=1., baseline_only=False):
        """History is log-relative, or the frozen demeaned float32 input for tiny.

        No timestamp, future label, scaler fitting, stopping choice or calibration
        fitting takes place here. Alpha must have been fixed from inner data.
        """
        x = np.asarray(history); anchor = np.asarray(current_close, dtype='float64').reshape(-1, 1)
        if x.ndim != 2 or x.shape != (len(anchor), self.cfg['lookback']):
            raise ValueError('History must have shape batch by configured lookback')
        if not np.isfinite(x).all() or not np.isfinite(anchor).all() or not (anchor > 0).all():
            raise ValueError('Inputs must be finite and current closes positive')
        if not np.isfinite(shrink_alpha) or not 0 <= shrink_alpha <= 1:
            raise ValueError('The frozen shrinkage scalar must be in [0,1]')
        base = np.zeros((len(anchor), self.cfg['horizon'])) if self.cfg['target'] == 'future_log_return' else \
            np.broadcast_to(anchor, (len(anchor), self.cfg['horizon']))
        if baseline_only:
            return base.copy()
        xx = ((x-self.scaler['x_mean'])/self.scaler['x_std']).astype('float32')
        if self.cfg['model'] == 'tiny':
            args = (torch.from_numpy(xx),)
        else:
            aa = ((anchor-self.scaler['anchor_mean'])/self.scaler['anchor_std']).astype('float32')
            if self.cfg['anchor'] == 'constant':
                aa = np.zeros_like(aa)
            args = torch.from_numpy(xx[:, :, None]), torch.from_numpy(aa)
        with torch.inference_mode():
            raw = self.net(*args).numpy()*self.ys+self.ym
        if self.cfg['output'] == 'residual':
            raw += anchor
        if shrink_alpha != 1.:
            raw = base+shrink_alpha*(raw-base)
        if not np.isfinite(raw).all():
            raise ValueError('Nonfinite forecast')
        return raw

    def predict(self, completed_closes, **kwargs):
        """Closes must be chronological and on the share basis known at origin."""
        close = np.asarray(completed_closes, dtype='float64')
        if close.ndim != 2 or close.shape[1] != self.cfg['lookback'] or \
                not np.isfinite(close).all() or not (close > 0).all():
            raise ValueError('Supply a positive finite batch by lookback close array')
        logs = np.log(close); history = logs-logs[:, -1, None]
        if self.cfg['model'] == 'tiny':
            history = (history-history.mean(1, keepdims=True)).astype('float32')
        return self.predict_prepared(history, close[:, -1], **kwargs)
