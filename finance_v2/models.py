"""Fixed model adapters. Native xPatch structure and input information are explicit."""
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from models.xPatch import Model as OriginalXPatch
from .io import ROOT, digest, read_json


NEURAL = {"xpatch48", "xpatch240"}
STOCHASTIC = NEURAL | {"kronos_small"}
PRETRAINED = {"kronos_small", "chronos2"}
ALL_MODELS = {"zero", "ridge", "lightgbm"} | NEURAL | PRETRAINED


class XPatch(nn.Module):
    def __init__(self, name, channels, horizon=12):
        super().__init__()
        self.length = {"xpatch48": 48, "xpatch240": 240}[name]
        self.close_channel = 0 if channels == 1 else 3
        self.model = OriginalXPatch(SimpleNamespace(
            seq_len=self.length, pred_len=horizon, enc_in=channels,
            patch_len=12, stride=6, padding_patch="end", revin=False,
            ma_type="ema", alpha=0.1, beta=0.1))
        del self.model.revin_layer

    def forward(self, x):
        return self.model(x[:, -self.length:])[:, :, self.close_channel]


def verify_model_lock(config):
    lock = read_json(ROOT / config["pretrained"]["lock"])
    for name, entry in lock["models"].items():
        for filename, evidence in entry["files"].items():
            if digest(ROOT / entry["directory"] / filename) != evidence["sha256"]:
                raise ValueError(f"Checkpoint changed: {name}/{filename}")
    return lock


class Pretrained:
    def __init__(self, name, config, device):
        self.name, self.config, self.device = name, config, device
        lock = verify_model_lock(config)
        if name == "kronos_small":
            from .vendor.kronos import Kronos, KronosTokenizer, KronosPredictor
            model = Kronos.from_pretrained(str(ROOT / lock["models"][name]["directory"]))
            tokenizer = KronosTokenizer.from_pretrained(str(ROOT / lock["models"]["kronos_tokenizer"]["directory"]))
            self.pipeline = KronosPredictor(model.eval(), tokenizer.eval(), device=device,
                                           max_context=config["pretrained"]["kronos"]["max_context"])
        elif name == "chronos2":
            from chronos import Chronos2Pipeline
            self.pipeline = Chronos2Pipeline.from_pretrained(
                str(ROOT / lock["models"][name]["directory"]), device_map=device)
        else:
            raise ValueError(name)

    @torch.inference_mode()
    def predict(self, data, origins, seed):
        import pandas as pd
        origins = np.asarray(origins, dtype=np.int64)
        horizon = max(data.horizons)
        pieces = []
        if self.name == "chronos2":
            settings = self.config["pretrained"]["chronos2"]
            batch_size = settings["batch_size"]
        else:
            settings = self.config["pretrained"]["kronos"]
            batch_size = settings["batch_size"]
        for start in range(0, len(origins), batch_size):
            batch = origins[start:start + batch_size]
            # Fixed batches/order/seed are part of the signed protocol. Calibration
            # and evaluation use separate calls, so they cannot share context.
            torch.manual_seed(int(seed) + start)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed) + start)
            if self.name == "chronos2":
                inputs = []
                for origin in batch:
                    idx = np.arange(origin - data.lookback + 1, origin + 1)
                    entry = {"target": data.returns[idx].astype(np.float32)}
                    if data.feature_set == "ohlcv":
                        entry["past_covariates"] = {f"feature_{j}": data.features[idx, j].astype(np.float32)
                                                    for j in (0, 1, 2, 4)}
                    inputs.append(entry)
                _, mean = self.pipeline.predict_quantiles(
                    inputs, prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9],
                    context_length=data.lookback, batch_size=settings["batch_size"],
                    cross_learning=settings["cross_learning"])
                cumulative = np.cumsum(np.stack([v.cpu().numpy()[0] for v in mean]).astype(np.float64), axis=1)
            else:
                frames, x_times, y_times = [], [], []
                last = []
                for origin in batch:
                    idx = np.arange(origin - data.lookback, origin + 1)
                    close = data.bars.close.iloc[idx].to_numpy() / data.bars.close.iloc[origin] * 100.0
                    values = np.column_stack([close] * 4 + [np.zeros(len(idx)), np.zeros(len(idx))])
                    if data.feature_set == "ohlcv":
                        observed = data.bars.iloc[idx[1:]]
                        values[1:, :4] = observed[["open", "high", "low", "close"]].to_numpy() / data.bars.close.iloc[origin] * 100.0
                        values[1:, 4] = observed.volume.to_numpy()
                        values[1:, 5] = values[1:, :4].mean(axis=1) * values[1:, 4]
                    frames.append(pd.DataFrame(values, columns=["open", "high", "low", "close", "volume", "amount"]))
                    stamps = data.bars.index[idx].tz_convert("America/New_York")
                    x_times.append(pd.Series(stamps))
                    # Generate future calendar stamps from the origin, never
                    # use labels or the presence/absence of actual future bars.
                    y_times.append(pd.Series(pd.date_range(stamps[-1] + pd.Timedelta(minutes=1),
                                                           periods=horizon, freq="min")))
                    last.append(close[-1])
                outputs = self.pipeline.predict_batch(
                    frames, x_times, y_times, pred_len=horizon, T=settings["temperature"],
                    top_p=settings["top_p"], sample_count=settings["sample_count"], verbose=False)
                closes = np.stack([v.close.to_numpy() for v in outputs]).astype(np.float64)
                if (closes <= 0).any():
                    raise ValueError("Kronos generated nonpositive close; no selective row deletion allowed")
                cumulative = np.log(closes / np.asarray(last)[:, None])
            if not np.isfinite(cumulative).all():
                raise ValueError("Nonfinite pretrained forecast")
            pieces.append(cumulative[:, np.asarray(data.horizons) - 1])
        return np.concatenate(pieces)
