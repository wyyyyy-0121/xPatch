"""Matched native xPatch bundles differing only in nonlinear-encoder sharing.

Four real H-specific native decoders retain the trend branch. No forecast path
is sliced from H60. Imports resolve to the caller's hash-frozen native sources.
"""
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from multiscale.models import NativeXPatch

HORIZONS = (5, 15, 30, 60)


class Encoder(nn.Module):
    """H-independent EMA and nonlinear patch encoder, through native flatten1."""
    NAMES = ('padding_patch_layer', 'fc1', 'gelu1', 'bn1', 'conv1', 'gelu2',
             'bn2', 'fc2', 'conv2', 'gelu3', 'bn3', 'flatten1')

    def __init__(self, native):
        super().__init__()
        backbone = native.backbone
        self.ema = deepcopy(native.ema)
        self.patch_len = backbone.patch_len
        self.stride = backbone.stride
        self.padding_patch = backbone.padding_patch
        for name in self.NAMES:
            if hasattr(backbone, name):
                setattr(self, name, deepcopy(getattr(backbone, name)))

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != 1:
            raise ValueError('Expected batch by completed history by one close channel')
        trend = self.ema(x)
        seasonal = (x-trend).permute(0, 2, 1).reshape(len(x), x.shape[1])
        trend = trend.permute(0, 2, 1).reshape(len(x), x.shape[1])
        if self.padding_patch == 'end':
            seasonal = self.padding_patch_layer(seasonal)
        seasonal = seasonal.unfold(-1, self.patch_len, self.stride)
        seasonal = self.bn1(self.gelu1(self.fc1(seasonal)))
        residual = seasonal
        seasonal = self.bn2(self.gelu2(self.conv1(seasonal)))
        seasonal = seasonal+self.fc2(residual)
        seasonal = self.bn3(self.gelu3(self.conv2(seasonal)))
        return self.flatten1(seasonal), trend


class Decoder(nn.Module):
    """One native H-specific nonlinear head, trend MLP, fusion and anchor head."""
    NAMES = ('fc3', 'gelu4', 'fc4', 'fc5', 'avgpool1', 'ln1', 'fc6',
             'avgpool2', 'ln2', 'fc7', 'fc8')

    def __init__(self, native):
        super().__init__()
        self.horizon = native.horizon
        for name in self.NAMES:
            setattr(self, name, deepcopy(getattr(native.backbone, name)))
        self.anchor_head = deepcopy(native.anchor_head)

    def forward(self, encoded, anchor):
        seasonal, trend = encoded
        seasonal = self.fc4(self.gelu4(self.fc3(seasonal)))
        trend = self.ln1(self.avgpool1(self.fc5(trend)))
        trend = self.ln2(self.avgpool2(self.fc6(trend)))
        trend = self.fc7(trend)
        value = self.fc8(torch.cat((seasonal, trend), dim=1))
        if anchor.ndim != 2 or anchor.shape != (len(value), 1):
            raise ValueError('Expected one standardized current-price anchor per origin')
        return value+self.anchor_head(anchor)


def native_reference(lookback, horizon, seed, scaler=None):
    """Construct a native function without consuming caller RNG state.

    scaler=None preserves its original random initialization for restoration
    tests. A scaler selects the prespecified exact-zero-return initialization.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        net = NativeXPatch(lookback, horizon)
    if scaler is not None:
        means = np.asarray(scaler['y_mean'], dtype=np.float64)[:horizon]
        scales = np.asarray(scaler['y_std'], dtype=np.float64)[:horizon]
        if means.shape != (horizon,) or scales.shape != (horizon,) or not np.isfinite(means).all() or not np.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError('Supply the common H60 fit-only cumulative-return target means/scales')
        with torch.no_grad():
            net.backbone.fc8.weight.zero_()
            net.backbone.fc8.bias.zero_()
            net.anchor_head.weight.zero_()
            net.anchor_head.bias.copy_(torch.as_tensor(-means/scales, dtype=torch.float32))
    return net


class Bundle(nn.Module):
    """Four matched native functions, with tied or untied nonlinear encoders.

    forward(x, anchor) returns {5: Bx5, 15: Bx15, 30: Bx30, 60: Bx60}.
    scaler is the original flat H60 fit-only scaler dictionary. Baseline
    initialization predicts raw zero up to the recorded float32 inversion
    tolerance, exactly matching the original v3 initialized native functions.
    """
    def __init__(self, lookback, scaler, seed, tied: bool):
        super().__init__()
        if not isinstance(tied, bool):
            raise ValueError('tied must be explicitly True or False')
        if lookback < 2 or len(scaler['y_mean']) != 60 or len(scaler['y_std']) != 60:
            raise ValueError('The matched minute experiment requires an H60 fit scaler')
        self.lookback = int(lookback)
        self.seed = int(seed)
        self.tied = tied
        prototypes = {h: native_reference(lookback, h, seed, scaler) for h in HORIZONS}
        template = Encoder(prototypes[HORIZONS[0]])
        initial = template.state_dict()
        for horizon in HORIZONS[1:]:
            other = Encoder(prototypes[horizon]).state_dict()
            if initial.keys() != other.keys() or any(not torch.equal(initial[name], other[name]) for name in initial):
                raise RuntimeError('Native encoder is not H-independent at matched initialization')
        if tied:
            self.encoder = template
        else:
            self.encoders = nn.ModuleDict({str(h): deepcopy(template) for h in HORIZONS})
        self.heads = nn.ModuleDict({str(h): Decoder(prototypes[h]) for h in HORIZONS})

    def forward(self, x, anchor):
        if x.ndim != 3 or x.shape[1:] != (self.lookback, 1):
            raise ValueError('Completed-history shape differs from the frozen lookback')
        if self.tied:
            encoded = self.encoder(x)
            return {h: self.heads[str(h)](encoded, anchor) for h in HORIZONS}
        return {h: self.heads[str(h)](self.encoders[str(h)](x), anchor) for h in HORIZONS}

    def parameter_counts(self):
        encoder = self.encoder if self.tied else self.encoders[str(HORIZONS[0])]
        one_encoder = sum(p.numel() for p in encoder.parameters())
        head_counts = {h: sum(p.numel() for p in self.heads[str(h)].parameters()) for h in HORIZONS}
        return {'total': sum(p.numel() for p in self.parameters()),
            'one_nonlinear_encoder': one_encoder, 'encoder_copies': 1 if self.tied else 4,
            'decoder_parameters': head_counts,
            'shared_scope': 'EMA and nonlinear patch encoder; native trend branches remain H-specific'}
