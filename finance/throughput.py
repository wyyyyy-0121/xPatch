"""Exact CUDA DEMA recurrence/adjoint and bounded, layout-preserving input reuse.

The recurrence and gradient additions retain native PyTorch float32 order.
Do not enable FMA or change the gradient layout: either changes RevIN updates.
"""
import numpy as np
import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _forward(X, A, B, Y, N, C: tl.constexpr, L: tl.constexpr,
        SB: tl.constexpr, ST: tl.constexpr, SC: tl.constexpr, BLOCK: tl.constexpr):
    ids = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = ids < N * C
    row, channel = ids // C, ids % C
    offset = row * SB + channel * SC
    alpha, beta = tl.load(A), tl.load(B)
    one_a, one_b = 1.0 - alpha, 1.0 - beta
    previous = tl.load(X + offset, mask=mask, other=0.0)
    trend = tl.load(X + offset + ST, mask=mask, other=0.0) - previous
    tl.store(Y + row * L * C + channel, previous, mask=mask)
    for t in range(1, L):
        value = tl.load(X + offset + t * ST, mask=mask, other=0.0)
        current = alpha * value + one_a * (previous + trend)
        trend = beta * (current - previous) + one_b * trend
        previous = current
        tl.store(Y + row * L * C + t * C + channel, current, mask=mask)


@triton.jit
def _backward(GS, GT, A, B, DX, N, C: tl.constexpr, L: tl.constexpr,
        S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
        T0: tl.constexpr, T1: tl.constexpr, T2: tl.constexpr,
        D0: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr, BLOCK: tl.constexpr):
    ids = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = ids < N * C
    row, channel = ids // C, ids % C
    ps, pt = row * S0 + channel * S2, row * T0 + channel * T2
    out = row * D0 + channel * D2
    alpha, beta = tl.load(A), tl.load(B)
    one_a, one_b = 1.0 - alpha, 1.0 - beta
    gs_next = tl.full((BLOCK,), 0.0, tl.float32)
    gb_next = tl.full((BLOCK,), 0.0, tl.float32)
    first = True
    for reverse in range(L - 1):
        t = L - 1 - reverse
        residual = tl.load(GS + ps + t * S1, mask=mask, other=0.0)
        trend = tl.load(GT + pt + t * T1, mask=mask, other=0.0)
        gy = trend - residual
        if first:
            gb = tl.full((BLOCK,), 0.0, tl.float32)
            gs = gy
        else:
            gb = one_b * gb_next + one_a * gs_next
            gs = ((-beta * gb_next + one_a * gs_next) + gy) + beta * gb
        first = False
        tl.store(DX + out + t * D1, residual + alpha * gs, mask=mask)
        gs_next, gb_next = gs, gb
    residual = tl.load(GS + ps, mask=mask, other=0.0)
    trend = tl.load(GT + pt, mask=mask, other=0.0)
    gb0 = one_b * gb_next + one_a * gs_next
    g0 = ((-beta * gb_next + one_a * gs_next) + (trend - residual)) - gb0
    tl.store(DX + out, residual + g0, mask=mask)
    g1 = tl.load(DX + out + D1, mask=mask, other=0.0)
    tl.store(DX + out + D1, g1 + gb0, mask=mask)


class _Decompose(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, original):
        trend = torch.empty(value.shape, device=value.device, dtype=value.dtype)
        n, length, channels = value.shape
        _forward[(triton.cdiv(n * channels, 32),)](value, original.ma.alpha,
            original.ma.beta, trend, n, channels, length, *value.stride(),
            32, num_warps=1, enable_fp_fusion=False)
        ctx.original, ctx.shape = original, value.shape
        return value - trend, trend

    @staticmethod
    def backward(ctx, seasonal_gradient, trend_gradient):
        n, length, channels = ctx.shape
        # Native autograd borrows this layout; RevIN reductions depend on it.
        gradient = torch.empty_like(seasonal_gradient)
        _backward[(triton.cdiv(n * channels, 32),)](seasonal_gradient, trend_gradient,
            ctx.original.ma.alpha, ctx.original.ma.beta, gradient, n, channels, length,
            *seasonal_gradient.stride(), *trend_gradient.stride(), *gradient.stride(),
            32, num_warps=1, enable_fp_fusion=False)
        return gradient, None


class FusedDecomp(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original

    def forward(self, value):
        if value.is_cuda and value.dtype == torch.float32:
            return _Decompose.apply(value, self.original)
        return self.original(value)


def resident_admitted(required_bytes, free_bytes, maximum_bytes=1024**3, reserve_bytes=1024**3):
    """Keep at least 1 GiB free and cap the optional data copy at 1 GiB."""
    return 0 < required_bytes <= maximum_bytes and required_bytes + reserve_bytes <= free_bytes


class ResidentTraining:
    @classmethod
    def maybe(cls, x, y, batch, device):
        if not str(device).startswith("cuda"):
            return None
        free_bytes, _ = torch.cuda.mem_get_info(device)
        if not resident_admitted(x.nbytes + y.nbytes, free_bytes):
            return None
        try:
            return cls(x, y, batch, device)
        except torch.OutOfMemoryError:
            # External allocations can race admission; use the original path.
            return None

    def __init__(self, x, y, batch, device):
        indices = np.arange(min(batch, len(x)))
        a = torch.as_tensor(x[indices], device=device)
        b = torch.as_tensor(y[indices], device=device)
        self.strides = (a.stride(), b.stride())
        self.x = torch.as_tensor(x, device=device)
        self.y = torch.as_tensor(y, device=device)
        self.buffers = {}

    def batch(self, indices):
        size = len(indices)
        if size not in self.buffers:
            self.buffers[size] = tuple(torch.empty_strided((size, *source.shape[1:]), stride,
                dtype=source.dtype, device=source.device)
                for source, stride in zip((self.x, self.y), self.strides))
        selected = torch.as_tensor(indices, device=self.x.device)
        for source, target in zip((self.x, self.y), self.buffers[size]):
            torch.index_select(source, 0, selected, out=target)
        return self.buffers[size]
