import torch
from torch import nn

class EMA(nn.Module):
    """Causal EMA initialized at x[0], preserving input device and dtype.

    Cached triangular weights avoid division by exponentially small powers.
    The nonpersistent buffer preserves compatibility with existing checkpoints.
    """

    def __init__(self, alpha):
        super().__init__()
        if not 0 < alpha <= 1:
            raise ValueError("EMA alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self.register_buffer("_weights", torch.empty(0), persistent=False)
        self._cache_key = None

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] == 0 or not x.is_floating_point():
            raise ValueError("EMA expects nonempty floating [batch, time, channel]")
        t = x.shape[1]
        key = (t, x.device, x.dtype, self.alpha)
        if self._cache_key != key:
            position = torch.arange(t, device=x.device)
            lag = position[:, None] - position[None, :]
            weights = self.alpha * torch.pow(1 - self.alpha, lag.clamp_min(0).to(x.dtype))
            weights = weights.masked_fill(lag < 0, 0)
            weights[:, 0] = torch.pow(1 - self.alpha, position.to(x.dtype))
            self._weights = weights
            self._cache_key = key
        return torch.matmul(x.transpose(1, 2), self._weights.T).transpose(1, 2)
