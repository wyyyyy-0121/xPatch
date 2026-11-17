"""Native supervised architectures and local, immutable Chronos-2 adapters."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from .common import ROOT, digest, read_json
from .vendor.revin import RevIN
from .vendor.xpatch_network import Network
from .vendor.patchtst_backbone import PatchTST_backbone


class CausalEMA(nn.Module):
    """Paper recurrence in a stable triangular form; no fixed device/power division."""
    def __init__(self, alpha):
        super().__init__()
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0,1]")
        self.alpha = float(alpha)
        self.register_buffer("weights", torch.empty(0), persistent=False)
        self.key = None

    def forward(self, x):
        n = x.shape[1]
        key = (n, x.device, x.dtype)
        if key != self.key:
            pos = torch.arange(n, device=x.device)
            lag = pos[:, None] - pos[None, :]
            w = self.alpha * (1 - self.alpha) ** lag.clamp_min(0).to(x.dtype)
            w[lag < 0] = 0
            w[:, 0] = (1 - self.alpha) ** pos.to(x.dtype)
            self.weights, self.key = w, key
        return (x.transpose(1, 2) @ self.weights.T).transpose(1, 2)


class XPatch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_revin = config.get("revin", False)
        if self.use_revin:
            self.revin = RevIN(1)
        self.ema = CausalEMA(config["alpha"])
        self.net = Network(config["lookback"], 12, config["patch"], config["stride"], "end")

    def forward(self, x):
        if self.use_revin:
            x = self.revin(x, "norm")
        trend = self.ema(x)
        y = self.net(x - trend, trend)
        if self.use_revin:
            y = self.revin(y, "denorm")
        return y[:, :, 0]


class PatchTST(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = PatchTST_backbone(c_in=1, context_window=config["lookback"], target_window=12,
            patch_len=config["patch"], stride=config["stride"], n_layers=config["layers"],
            d_model=config["d_model"], n_heads=4, d_ff=2 * config["d_model"],
            dropout=config["dropout"], fc_dropout=0.0, head_dropout=0.0,
            padding_patch="end", revin=config.get("revin", False), affine=True,
            subtract_last=False, individual=False)

    def forward(self, x):
        return self.net(x.transpose(1, 2))[:, 0, :]


def neural(config):
    if config["model"] == "xpatch":
        return XPatch(config)
    if config["model"] == "patchtst":
        return PatchTST(config)
    raise ValueError(config["model"])


def verify_chronos_weight():
    old_root = Path("/home/ayaka/xPatch")
    lock = read_json(old_root / "configs/aapl_v2_model_lock.json")["models"]["chronos2"]
    path = old_root / lock["directory"]
    for name, row in lock["files"].items():
        if digest(path / name) != row["sha256"]:
            raise ValueError(f"Chronos checkpoint changed: {name}")
    return path, {"repository": lock["repository"], "revision": lock["revision"],
                  "license": lock["license"], "files": lock["files"], "path_read_only": str(path)}


def chronos_pipeline(lora=False, rank=8, device="cuda"):
    from chronos import Chronos2Pipeline
    path, receipt = verify_chronos_weight()
    pipe = Chronos2Pipeline.from_pretrained(str(path), device_map=device, local_files_only=True,
                                          torch_dtype=torch.float32)
    if lora:
        # Fail closed: never permit API fallback to full fine-tuning.
        from peft import LoraConfig, get_peft_model
        targets = ["self_attention.q", "self_attention.v", "self_attention.k", "self_attention.o",
                   "output_patch_embedding.output_layer"]
        pipe.model = get_peft_model(pipe.model, LoraConfig(r=rank, lora_alpha=2 * rank,
                                                        lora_dropout=0.0, target_modules=targets))
        n_trainable = sum(p.numel() for p in pipe.model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in pipe.model.parameters())
        if not 0 < n_trainable < n_all / 10:
            raise ValueError("Unexpected LoRA trainable-parameter count")
        receipt.update(trainable_parameters=n_trainable, all_parameters=n_all,
                       trainable_names=[n for n, p in pipe.model.named_parameters() if p.requires_grad],
                       lora_rank=rank, lora_targets=targets)
    pipe.model.eval()
    return pipe, receipt


def quantile_point(q, levels, method):
    """q is [..., quantile, horizon] in original units."""
    levels = np.asarray(levels, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if method == "native_median":
        return q[..., int(np.flatnonzero(np.isclose(levels, 0.5))[0]), :]
    if method != "quantile_integral_tailflat":
        raise ValueError(method)
    grid = np.r_[0.0, levels, 1.0]
    values = np.concatenate([q[..., :1, :], q, q[..., -1:, :]], axis=-2)
    return np.trapezoid(values, x=grid, axis=-2)


@torch.inference_mode()
def chronos_predict(pipe, market, origins, config, batch_size=32):
    pipe.model.eval()
    out = []
    for start in range(0, len(origins), batch_size):
        items = market.chronos_inputs(origins[start:start + batch_size], config["lookback"],
                                      config.get("representation", "minute_return"), config.get("panel", "close"))
        q = pipe.predict(items, prediction_length=12, context_length=config["lookback"],
                         batch_size=batch_size, cross_learning=False)
        q = torch.stack([t[0] for t in q]).float().cpu().numpy()
        steps = quantile_point(q, pipe.quantiles, config.get("point", "native_median"))[:, :12]
        value = steps.sum(axis=1) if config.get("representation", "minute_return") == "minute_return" else steps[:, 11]
        out.append(value)
    result = np.concatenate(out)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite Chronos prediction")
    return result


def chronos_training_batch(market, origins, lookback, device):
    idx = np.asarray(origins)[:, None] + np.arange(1 - lookback, 1)
    context = torch.as_tensor(market.returns[idx].astype(np.float32), device=device)
    future = torch.as_tensor(market.y_steps(origins).astype(np.float32), device=device)
    # Unique group per origin prevents any cross-origin information sharing.
    return dict(context=context, future_target=future,
                group_ids=torch.arange(len(origins), device=device), num_output_patches=1)
