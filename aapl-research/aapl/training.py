"""Streamed training and identity-checked, deterministic resumable checkpoints."""
import copy
import os
from pathlib import Path
import random
import time
from contextlib import nullcontext

import numpy as np
import torch

from .common import ROOT, digest, identity, sync, write_json, precision_policy
from .models import chronos_predict, chronos_training_batch


def source_identity():
    # Gate/report scripts do not alter a model checkpoint's training identity.
    files = [ROOT / "aapl" / n for n in ("common.py", "data.py", "models.py", "training.py", "baselines.py", "resources.py")]
    files += sorted((ROOT / "aapl/vendor").glob("*.py"))
    return identity({str(p.relative_to(ROOT)): digest(p) for p in files})


class Trainer:
    def __init__(self, model, config, market, origins, scale, device, seed, max_budget,
                 lora=False):
        self.model, self.config, self.market = model, config, market
        self.origins = np.asarray(origins)
        self.scale, self.device, self.seed, self.max_budget, self.lora = scale, device, seed, max_budget, lora
        self.model.to(device)
        self.precision = precision_policy(config)
        self.forward_model = torch.compile(model, mode='reduce-overhead', dynamic=False) if config.get('compile', False) else model
        self.optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                          lr=config.get("lr", 0.001), weight_decay=config.get("weight_decay", 0.0),
                                          fused=config.get('optimizer_fused', False))
        self.cached_x = self.cached_y = self.cached_returns = None
        if config.get('device_cache', False):
            # One source sequence, never an N x lookback x C window expansion.
            values = market.features[:, 3:4] if scale['panel'] == 'close' else market.features
            self.cached_x = torch.as_tensor(((values - scale['center']) / scale['std']).astype(np.float32), device=device)
            self.cached_y = torch.as_tensor((market.returns / scale['target_std']).astype(np.float32), device=device)
            self.cached_returns = torch.as_tensor(market.returns.astype(np.float32), device=device)
        self.sample_fraction = config.get('sample_fraction', 1.0)
        if not 0 < self.sample_fraction <= 1:
            raise ValueError('Invalid training-origin fraction')
        self.epoch, self.position, self.updates = 0, 0, 0
        self.order = None
        self.history = []
        self.batch_seconds = []
        self.signature = identity({"config": config, "seed": seed, "max_budget": max_budget,
                                   "origins": identity(self.origins.tolist()), "scale": scale,
                                   "source": source_identity(), "lora": lora})

    def _next_ids(self, batch_size):
        if self.order is None or self.position >= len(self.order):
            self.order = np.random.permutation(len(self.origins))[:int(np.ceil(len(self.origins) * self.sample_fraction))]
            self.position = 0
        take = self.order[self.position:self.position + batch_size]
        self.position += len(take)
        return self.origins[take]

    def update(self):
        from .resources import check_resources
        check_resources()
        self.model.train()
        ids = self._next_ids(self.config["batch_size"])
        sync()
        started = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        precision_policy(self.config)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.precision == 'bf16'):
            if self.lora:
                if self.cached_returns is None:
                    batch = chronos_training_batch(self.market, ids, self.config['lookback'], self.device)
                else:
                    origins = torch.as_tensor(ids, device=self.device)
                    ix = origins[:, None] + torch.arange(1 - self.config['lookback'], 1, device=self.device)
                    iy = origins[:, None] + torch.arange(1, 13, device=self.device)
                    batch = dict(context=self.cached_returns[ix], future_target=self.cached_returns[iy],
                                 group_ids=torch.arange(len(ids), device=self.device), num_output_patches=1)
                loss = self.forward_model(**batch).loss
            else:
                if self.cached_x is None:
                    x = torch.as_tensor(self.market.x(ids, self.config['lookback'], self.scale), device=self.device)
                    y = torch.as_tensor((self.market.y_steps(ids) / self.scale['target_std']).astype(np.float32), device=self.device)
                else:
                    origins = torch.as_tensor(ids, device=self.device)
                    ix = origins[:, None] + torch.arange(1 - self.config['lookback'], 1, device=self.device)
                    iy = origins[:, None] + torch.arange(1, 13, device=self.device)
                    x, y = self.cached_x[ix], self.cached_y[iy]
                pred = self.forward_model(x).float()
                if self.config['loss'] == 'step_mse':
                    loss = ((pred - y) ** 2).mean()
                else:
                    loss = ((pred.sum(axis=1) - y.sum(axis=1)) ** 2).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 1.0)
        self.optimizer.step()
        self.updates += 1
        if self.lora:
            # Fixed 900-step schedule at every rung; extending a rung cannot alter its prefix.
            factor = max(0.0, 1 - self.updates / self.max_budget)
            for group in self.optimizer.param_groups:
                group["lr"] = self.config["lr"] * factor
        sync()
        self.batch_seconds.append(time.perf_counter() - started)
        return float(loss.detach()), len(ids)

    def train_updates(self, stop):
        losses = []
        while self.updates < stop:
            loss, n = self.update()
            losses.append(loss)
        return losses

    def train_epoch(self):
        if self.order is None or self.position >= len(self.order):
            self.order = np.random.permutation(len(self.origins))[:int(np.ceil(len(self.origins) * self.sample_fraction))]
            self.position = 0
        losses, count = 0.0, 0
        while self.position < len(self.order):
            loss, n = self.update()
            losses += loss * n
            count += n
        self.epoch += 1
        return losses / count

    def save(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Checkpoint path outside new research")
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.lora:
            from peft import get_peft_model_state_dict
            state = get_peft_model_state_dict(self.model)
        else:
            state = self.model.state_dict()
        payload = {"signature": self.signature, "model": {k: v.detach().cpu() for k, v in state.items()},
                   "optimizer": self.optimizer.state_dict(), "epoch": self.epoch, "position": self.position,
                   "updates": self.updates, "order": self.order, "history": self.history,
                   "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
                   "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        write_json(path.with_suffix(".receipt.json"), {"signature": self.signature, "sha256": digest(path),
                                                      "updates": self.updates, "epoch": self.epoch, "position": self.position})

    def restore(self, path):
        from .common import read_json
        path = Path(path)
        receipt = read_json(path.with_suffix(".receipt.json"))
        if receipt["signature"] != self.signature or digest(path) != receipt["sha256"]:
            raise ValueError("Checkpoint identity/hash mismatch")
        # Trusted self-generated local checkpoints only; receipt/hash verified first.
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state["signature"] != self.signature:
            raise ValueError("Checkpoint payload identity mismatch")
        if self.lora:
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(self.model, state["model"])
        else:
            self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.epoch, self.position, self.updates = state["epoch"], state["position"], state["updates"]
        self.order, self.history = state["order"], state["history"]
        random.setstate(state["rng_python"])
        np.random.set_state(state["rng_numpy"])
        torch.set_rng_state(state["rng_torch"].cpu())
        if state["rng_cuda"] is not None:
            torch.cuda.set_rng_state_all([x.cpu() for x in state["rng_cuda"]])


@torch.inference_mode()
def neural_predict(model, config, market, origins, scale, device, batch_size=256):
    model.eval()
    precision = precision_policy(config)
    result = []
    for start in range(0, len(origins), batch_size):
        x = torch.as_tensor(market.x(origins[start:start + batch_size], config["lookback"], scale), device=device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=precision == 'bf16'):
            value = model(x)
        result.append(value.float().cpu().numpy().astype(np.float64).sum(axis=1) * scale['target_std'])
    prediction = np.concatenate(result)
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite neural prediction")
    return prediction
