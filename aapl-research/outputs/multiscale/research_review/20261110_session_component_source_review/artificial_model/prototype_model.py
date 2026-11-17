"""Artificial-only matched two-state GRU prototype; no data-module imports."""
from __future__ import annotations
import copy
import torch
from torch import nn


class TwoStateGRU(nn.Module):
    def __init__(self, horizon: int, zero_scaled: torch.Tensor):
        super().__init__()
        if type(horizon) is not int or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if (not isinstance(zero_scaled, torch.Tensor) or zero_scaled.shape != (horizon,)
                or zero_scaled.dtype not in (torch.float32, torch.float64)
                or zero_scaled.device.type != "cpu" or not torch.isfinite(zero_scaled).all()):
            raise ValueError("zero_scaled must be a finite CPU floating [H] tensor")
        self.horizon = horizon
        self.cell1 = nn.GRUCell(2, 32)
        self.cell2 = nn.GRUCell(2, 32)
        self.head = nn.Linear(82, horizon)
        self.register_buffer("zero_scaled", zero_scaled.detach().clone(), persistent=True)
        self.to(dtype=zero_scaled.dtype, device="cpu")
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.zero_()

    def _validate(self, x, boundary, anchor, context, reset_flag):
        if type(reset_flag) is not int or reset_flag not in (0, 1):
            raise ValueError("reset_flag must be integer 0 or 1")
        if (not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[2] != 1
                or x.shape[0] < 1 or x.shape[1] < 1):
            raise ValueError("x must have positive [B,L,1] shape")
        B, L, _ = x.shape
        if (not isinstance(boundary, torch.Tensor) or boundary.dtype != torch.bool
                or boundary.shape != (B, L) or boundary.device.type != "cpu"
                or bool(boundary[:, 0].any())):
            raise ValueError("boundary must be CPU bool [B,L] with first slot false")
        for name, value, shape in (("x", x, (B, L, 1)),
                                   ("anchor", anchor, (B, 1)),
                                   ("context", context, (B, 17))):
            if (not isinstance(value, torch.Tensor) or value.shape != shape
                    or value.dtype != self.zero_scaled.dtype or value.device.type != "cpu"
                    or not torch.isfinite(value).all()):
                raise ValueError(name + " must have model dtype, valid shape and finite CPU values")
        if self.cell1.weight_ih.dtype != x.dtype or self.cell1.weight_ih.device.type != "cpu":
            raise ValueError("model/input dtype or CPU placement mismatch")

    def forward(self, x, boundary, anchor, context, *, reset_flag: int):
        self._validate(x, boundary, anchor, context, reset_flag)
        B, L, _ = x.shape
        h1 = x.new_zeros((B, 32))
        h2 = x.new_zeros((B, 32))
        for j in range(L):
            g = boundary[:, j:j + 1].to(dtype=x.dtype)
            cell_input = torch.cat((x[:, j], g), dim=1)
            h1 = self.cell1(cell_input, h1)
            # Both arms execute this exact operation and identical GRUCell loop.
            h2 = h2 * (1 - reset_flag * g)
            h2 = self.cell2(cell_input, h2)
        output = self.zero_scaled + self.head(torch.cat((h1, h2, anchor, context), dim=1))
        return {"output": output, "h1": h1, "h2": h2}


def paired_models(horizon: int, zero_scaled: torch.Tensor):
    """Independent cell draws within one base; paired copies across C/R."""
    base = TwoStateGRU(horizon, zero_scaled)
    return {"C": copy.deepcopy(base), "R": copy.deepcopy(base)}
