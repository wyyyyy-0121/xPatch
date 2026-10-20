"""Leakage-aware AAPL multi-timescale forecast experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROFILE_PATH = HERE / "profiles.json"
OHLCV = ("open", "high", "low", "close", "volume")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ohlcv(path: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path)
    cols = {str(c).strip().lower(): c for c in raw.columns}
    time_col = next((cols[c] for c in ("timestamp", "datetime", "date") if c in cols), None)
    missing = [c for c in OHLCV if c not in cols]
    if time_col is None or missing:
        raise ValueError(f"CSV needs timestamp and {', '.join(OHLCV)} columns; missing: {missing}")
    df = raw[[time_col, *(cols[c] for c in OHLCV)]].copy()
    df.columns = ["timestamp", *OHLCV]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)
    if df.timestamp.isna().any():
        raise ValueError(f"CSV contains {int(df.timestamp.isna().sum())} invalid timestamps")
    for col in OHLCV:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[list(OHLCV)].isna().any().any():
        raise ValueError("CSV contains missing or non-numeric OHLCV values")
    if not np.isfinite(df[list(OHLCV)].to_numpy(dtype=float)).all():
        raise ValueError("CSV contains non-finite OHLCV values")
    if (df[["open", "high", "low", "close"]] <= 0).any().any() or (df.volume < 0).any():
        raise ValueError("Prices must be positive and volume must be non-negative")
    if (df.high < df[["open", "low", "close"]].max(axis=1)).any() or (df.low > df[["open", "high", "close"]].min(axis=1)).any():
        raise ValueError("OHLC rows violate high/low bounds")
    original_rows = len(df)
    duplicates = int(df.timestamp.duplicated(keep="last").sum())
    df = df.drop_duplicates("timestamp", keep="last").sort_values("timestamp").set_index("timestamp")
    source = "local CSV; provenance not independently verified"
    sidecar = path.with_suffix(path.suffix + ".manifest.json")
    declared = {}
    if sidecar.exists():
        declared = json.loads(sidecar.read_text())
        if declared.get("source_verified") is True and declared.get("source") and declared.get("timezone"):
            source = str(declared["source"])
            if df.index.tz is None:
                df.index = df.index.tz_localize(declared["timezone"])
            elif str(df.index.tz) != str(declared["timezone"]):
                raise ValueError("CSV timezone does not match its sidecar manifest")
        elif declared.get("provider"):
            source = f"{declared['provider']}; manifest provenance not verified"
    minute_coverages = []
    for _, day in df.groupby(df.index.normalize()):
        if len(day) > 1:
            expected = int((day.index[-1] - day.index[0]).total_seconds() // 60) + 1
            minute_coverages.append(min(1.0, len(day) / max(expected, 1)))
    meta = {"rows_input": original_rows, "duplicate_timestamps_removed": duplicates,
            "timestamp_timezone": str(df.index.tz) if df.index.tz else "naive / unspecified",
            "start": df.index.min().isoformat(), "end": df.index.max().isoformat(),
            "sessions": int(df.index.normalize().nunique()),
            "median_within_day_minute_coverage": float(np.median(minute_coverages)) if minute_coverages else None,
            "source": source, "sidecar_manifest": str(sidecar) if sidecar.exists() else None,
            "provider_url": declared.get("provider_url"), "adjustment": declared.get("adjustment")}
    return df, meta


def aggregate_bars(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    if interval in {"1min", "1d"}:
        if interval == "1d":
            return df.groupby(df.index.normalize()).agg(
                open=("open", "first"), high=("high", "max"), low=("low", "min"),
                close=("close", "last"), volume=("volume", "sum"))
        return df.copy()
    if interval == "1mo":
        return df.resample("MS").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    if interval == "1h":
        parts = []
        for _, day in df.groupby(df.index.normalize()):
            x = day.resample("1h", origin=day.index[0]).agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
            parts.append(x)
        return pd.concat(parts) if parts else df.iloc[:0]
    if interval in {"5min", "1min"}:
        freq = "5min" if interval == "5min" else "1min"
        parts = []
        for _, day in df.groupby(df.index.normalize()):
            x = day.resample(freq, origin=day.index[0]).agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
            parts.append(x)
        return pd.concat(parts) if parts else df.iloc[:0]
    raise ValueError(f"Unsupported interval: {interval}")


def years_covered(index: pd.Index) -> float:
    if not len(index):
        return 0.0
    return (index[-1] - index[0]).total_seconds() / (365.2425 * 86400)


def data_gate(profile: str, bars: pd.DataFrame, meta: dict, requirements: dict) -> list[str]:
    issues = []
    req = requirements[profile]
    years = years_covered(bars.index)
    if years < req.get("minimum_years", 0):
        issues.append(f"history {years:.2f} years < required {req['minimum_years']} years")
    if years * 12 < req.get("minimum_months", 0):
        issues.append(f"history {years * 12:.1f} months < required {req['minimum_months']} months")
    if profile in {"hourly", "minute"}:
        if meta.get("timestamp_timezone", "").startswith("naive"):
            issues.append("intraday timezone is unspecified; trading-session continuity cannot be verified")
        if "provenance not independently verified" in meta.get("source", ""):
            issues.append("intraday source/provenance is not independently verified")
        coverage = meta.get("median_within_day_minute_coverage")
        minimum = req.get("minimum_session_coverage", 0)
        if coverage is not None and coverage < minimum:
            issues.append(f"median within-day minute coverage {coverage:.3f} < required {minimum:.3f}")
    return issues


def make_samples(bars: pd.DataFrame, lookback: int, horizon: int, interval: str, target_mode: str):
    close = bars.close.to_numpy(dtype=np.float64)
    timestamps = bars.index
    features = np.column_stack([
        np.log(bars.open.to_numpy() / close), np.log(bars.high.to_numpy() / close),
        np.log(bars.low.to_numpy() / close), np.log(close / np.r_[close[0], close[:-1]]),
        np.log1p(bars.volume.to_numpy())
    ])
    intraday = interval in {"1min", "5min", "1h"}
    groups = pd.Series(np.arange(len(bars)), index=timestamps).groupby(timestamps.normalize()).apply(list).tolist() if intraday else [list(range(len(bars)))]
    xs, ys, origins = [], [], []
    for ids in groups:
        for end in range(lookback - 1, len(ids) - horizon):
            i = ids[end]
            future_ids = ids[end + 1:end + horizon + 1]
            if len(future_ids) != horizon:
                continue
            xids = ids[end - lookback + 1:end + 1]
            step = {"1min": pd.Timedelta(minutes=1), "5min": pd.Timedelta(minutes=5),
                    "1h": pd.Timedelta(hours=1)}.get(interval)
            if step is not None:
                full_ids = ids[end - lookback + 1:end + horizon + 1]
                if not np.all(np.diff(timestamps[full_ids]) == step):
                    continue
            xs.append(features[xids])
            future = close[future_ids]
            if target_mode == "future_close":
                ys.append(future)
            else:
                ys.append(np.log(future / close[i]))
            origins.append(i)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float64), np.asarray(origins, dtype=np.int64), close


def rolling_splits(origins: np.ndarray, dates: np.ndarray, folds=5, purge=1):
    order = np.argsort(origins)
    chunks = np.array_split(order, folds + 1)
    splits = []
    for fold in range(1, len(chunks)):
        train = np.concatenate(chunks[:fold])
        valid = chunks[fold]
        if not len(train) or not len(valid):
            continue
        cutoff = origins[valid].min()
        train = train[origins[train] + purge < cutoff]
        if len(train):
            splits.append((train, valid))
    return splits


def metrics(actual, predicted, last_close, mase_scale):
    err = predicted - actual
    denom = np.maximum((np.abs(actual) + np.abs(predicted)) / 2, 1e-8)
    actual_returns = np.log(actual / last_close[:, None])
    predicted_returns = np.log(np.maximum(predicted, 1e-12) / last_close[:, None])
    per_step = []
    for step in range(actual.shape[1]):
        step_err = err[:, step]
        per_step.append({"step": step + 1, "MAE": float(np.mean(np.abs(step_err))),
                         "RMSE": float(np.sqrt(np.mean(step_err ** 2)))})
    return {"MAE": float(np.mean(np.abs(err))), "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "MASE": float(np.mean(np.abs(err)) / max(mase_scale, 1e-12)),
            "sMAPE": float(np.mean(np.abs(err) / denom) * 100),
            "direction_accuracy": float(np.mean(np.sign(actual - last_close[:, None]) == np.sign(predicted - last_close[:, None]))),
            "log_return_RMSE": float(np.sqrt(np.mean((predicted_returns - actual_returns) ** 2))),
            "per_step": per_step}


def price_space(y, target_mode, last_close):
    return y if target_mode == "future_close" else last_close[:, None] * np.exp(y)


def seasonal_forecast(closes, origins, horizon, interval):
    period = {"1mo": 12, "1d": 252}.get(interval)
    if period is None:
        return np.repeat(closes[origins, None], horizon, axis=1)
    steps = np.arange(1, horizon + 1)
    indices = origins[:, None] + steps[None, :] - period
    pred = np.repeat(closes[origins, None], horizon, axis=1)
    valid = indices >= 0
    pred[valid] = closes[indices[valid]]
    return pred


def run_profile(name: str, csv_path: Path, out_dir: Path, lookbacks=None, requested_models=None, bar_interval=None):
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    profiles = json.loads(PROFILE_PATH.read_text())
    if name not in profiles["profiles"]:
        raise ValueError(f"Unknown profile {name}; choose {', '.join(profiles['profiles'])}")
    raw, meta = load_ohlcv(csv_path)
    existing_report = out_dir / "report.json"
    if existing_report.exists() and json.loads(existing_report.read_text()).get("final_test"):
        raise FileExistsError(f"Final test already exists in {existing_report}; use a new output directory")
    p = profiles["profiles"][name]
    if bar_interval:
        allowed = {"annual": {"1mo"}, "quarterly": {"1mo"}, "monthly": {"1d"},
                   "hourly": {"1h"}, "minute": {"1min", "5min"}}
        if bar_interval not in allowed[name]:
            raise ValueError(f"{name} profile does not support {bar_interval}")
        p = {**p, "bar_interval": bar_interval}
        if name == "minute" and bar_interval == "5min":
            p = {**p, "lookbacks": [6, 12, 24, 48, 96], "horizons": [1, 3, 6, 12]}
    bars = aggregate_bars(raw, p["bar_interval"])
    issues = data_gate(name, bars, meta, profiles["requirements"])
    out_dir.mkdir(parents=True, exist_ok=True)
    report_base = {"symbol": "AAPL", "profile": name, "bar_interval": p["bar_interval"], "data": meta,
                   "bars": len(bars), "history_years": round(years_covered(bars.index), 4),
                   "source_sha256": sha256(csv_path), "generated_utc": datetime.now(timezone.utc).isoformat(),
                   "data_limitations": (["Local CSV adjustment policy has not been independently confirmed."]
                                         if meta.get("sidecar_manifest") else
                                         (["Local CSV provenance and adjustment policy are unverified."]
                                          if "local CSV" in meta.get("source", "") else [])),
                   "compute": _compute_manifest()}
    if issues:
        result = {**report_base, "status": "insufficient_data", "reasons": issues, "rankings": []}
        (out_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        (out_dir / "REPORT.md").write_text(f"# {name} experiment\n\nStatus: `insufficient_data`\n\n" + "\n".join(f"- {x}" for x in issues) + "\n")
        return result
    models = requested_models or ["last_close", "drift", "seasonal", "ridge", "mlp", "gru", "lstm", "xpatch"]
    records = []
    test_jobs = []
    for target_mode in ("future_close", "future_log_return"):
      for horizon in p["horizons"]:
        for lookback in lookbacks or p["lookbacks"]:
            X, y, origins, closes = make_samples(bars, lookback, horizon, p["bar_interval"], target_mode)
            final_cut = int(np.floor(len(origins) * .8))
            if final_cut <= 0 or final_cut >= len(origins):
                continue
            test_start_origin = origins[final_cut]
            final_train = np.flatnonzero(origins + horizon < test_start_origin)
            final_test = np.arange(final_cut, len(origins))
            splits = rolling_splits(origins, bars.index[origins].normalize().to_numpy(), folds=5, purge=horizon)
            if len(splits) < 3:
                continue
            for model_name in models:
                fold_scores = []
                for fold_id, (tr, va) in enumerate(splits):
                    from sklearn.preprocessing import StandardScaler
                    feature_scaler = StandardScaler().fit(X[tr].reshape(-1, X.shape[-1]))
                    train_scaled = feature_scaler.transform(X[tr].reshape(-1, X.shape[-1])).reshape(X[tr].shape)
                    valid_scaled = feature_scaler.transform(X[va].reshape(-1, X.shape[-1])).reshape(X[va].shape)
                    xtr, xva = train_scaled.reshape(len(tr), -1), valid_scaled.reshape(len(va), -1)
                    y_scaler = StandardScaler().fit(y[tr])
                    ytr = y_scaler.transform(y[tr])
                    if model_name == "last_close":
                        pred_price = np.repeat(closes[origins[va], None], horizon, axis=1)
                    elif model_name in {"drift", "seasonal"}:
                        base = closes[origins[va]]
                        prior = closes[np.maximum(origins[va] - lookback + 1, 0)]
                        if model_name == "seasonal":
                            pred_price = seasonal_forecast(closes, origins[va], horizon, p["bar_interval"])
                        else:
                            rate = np.log(base / prior) / max(lookback - 1, 1)
                            pred_price = base[:, None] * np.exp(np.arange(1, horizon + 1)[None, :] * rate[:, None])
                            if target_mode == "future_close":
                                pred_price = base[:, None] + np.arange(1, horizon + 1)[None, :] * (base - prior)[:, None] / max(lookback - 1, 1)
                    elif model_name == "ridge":
                        model = Ridge(alpha=1.0).fit(xtr, ytr)
                        pred = y_scaler.inverse_transform(np.asarray(model.predict(xva)).reshape(len(va), horizon))
                        pred_price = np.asarray(pred).reshape(len(va), horizon) if target_mode == "future_close" else price_space(np.asarray(pred).reshape(len(va), horizon), target_mode, closes[origins[va]])
                    elif model_name == "mlp":
                        model = MLPRegressor(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=100, random_state=2026,
                                             early_stopping=False).fit(xtr, ytr)
                        pred = y_scaler.inverse_transform(np.asarray(model.predict(xva)).reshape(len(va), horizon))
                        pred_price = pred if target_mode == "future_close" else price_space(pred, target_mode, closes[origins[va]])
                    else:
                        pred_price = _fit_torch(model_name, train_scaled, ytr, valid_scaled, horizon, lookback,
                                                 closes[origins[va]], target_mode, y_scaler.mean_, y_scaler.scale_)
                    actual = closes[origins[va, None] + np.arange(1, horizon + 1)]
                    train_origin_set = origins[tr]
                    adjacent = np.diff(train_origin_set) == 1
                    adjacent_origins = train_origin_set[1:][adjacent]
                    if p["bar_interval"] in {"1min", "5min", "1h"}:
                        same_session = bars.index[adjacent_origins].normalize() == bars.index[adjacent_origins - 1].normalize()
                        adjacent_origins = adjacent_origins[same_session]
                    mase_scale = np.mean(np.abs(closes[adjacent_origins] - closes[adjacent_origins - 1])) if len(adjacent_origins) else 0.0
                    score = metrics(actual, pred_price, closes[origins[va]], mase_scale)
                    fold_scores.append({"fold": fold_id + 1, "metrics": score, "train_samples": len(tr), "validation_samples": len(va)})
                if fold_scores:
                    baseline = next((r for r in records if r["model"] == "last_close" and r["target_mode"] == target_mode
                                     and r["lookback"] == lookback and r["horizon"] == horizon), None)
                    improvements = []
                    if model_name == "last_close":
                        improvements = [False] * len(fold_scores)
                    elif baseline:
                        improvements = [f["metrics"]["RMSE"] < b["metrics"]["RMSE"]
                                        for f, b in zip(fold_scores, baseline["folds"])]
                    records.append({"model": model_name, "target_mode": target_mode, "lookback": lookback,
                                    "horizon": horizon, "folds": fold_scores,
                                    "median_MASE": float(np.median([f["metrics"]["MASE"] for f in fold_scores])),
                                    "median_RMSE": float(np.median([f["metrics"]["RMSE"] for f in fold_scores])),
                                    "baseline_improvement_fraction": float(np.mean(improvements)) if improvements else 0.0,
                                    "median_baseline_RMSE_gain": float(np.median([
                                        1 - f["metrics"]["RMSE"] / max(b["metrics"]["RMSE"], 1e-12)
                                        for f, b in zip(fold_scores, baseline["folds"])])) if baseline and model_name != "last_close" else 0.0})
                    test_jobs.append((target_mode, horizon, lookback, model_name, X, y, origins, closes,
                                      final_train, final_test))
    records.sort(key=lambda r: (r["median_MASE"], r["median_RMSE"]))
    accepted = [r for r in records if r["model"] != "last_close" and r["median_MASE"] < 1
                and r["median_baseline_RMSE_gain"] >= .02 and r["baseline_improvement_fraction"] >= .6]
    selected_keys = set()
    for target_mode in ("future_close", "future_log_return"):
        for horizon in p["horizons"]:
            options = [r for r in records if r["target_mode"] == target_mode and r["horizon"] == horizon]
            if options:
                selected = min(options, key=lambda r: (r["median_MASE"], r["median_RMSE"]))
                selected_keys.add((target_mode, horizon, selected["lookback"], selected["model"]))
    final_results = []
    for job in test_jobs:
        target_mode, horizon, lookback, model_name, X, y, origins, closes, train_ids, test_ids = job
        if (target_mode, horizon, lookback, model_name) not in selected_keys or not len(train_ids) or not len(test_ids):
            continue
        test_close = closes[origins[test_ids]]
        if model_name == "last_close":
            pred_price = np.repeat(test_close[:, None], horizon, axis=1)
        elif model_name in {"drift", "seasonal"}:
            if model_name == "seasonal":
                pred_price = seasonal_forecast(closes, origins[test_ids], horizon, p["bar_interval"])
            else:
                prior = closes[np.maximum(origins[test_ids] - lookback + 1, 0)]
                rate = np.log(test_close / prior) / max(lookback - 1, 1)
                pred_price = test_close[:, None] * np.exp(np.arange(1, horizon + 1)[None, :] * rate[:, None])
                if target_mode == "future_close":
                    pred_price = test_close[:, None] + np.arange(1, horizon + 1)[None, :] * (test_close - prior)[:, None] / max(lookback - 1, 1)
        else:
            from sklearn.preprocessing import StandardScaler
            x_train, x_test = X[train_ids], X[test_ids]
            feature_scaler = StandardScaler().fit(x_train.reshape(-1, X.shape[-1]))
            x_train = feature_scaler.transform(x_train.reshape(-1, X.shape[-1])).reshape(x_train.shape)
            x_test = feature_scaler.transform(x_test.reshape(-1, X.shape[-1])).reshape(x_test.shape)
            target_scaler = StandardScaler().fit(y[train_ids])
            y_train = target_scaler.transform(y[train_ids])
            x_train_flat, x_test_flat = x_train.reshape(len(train_ids), -1), x_test.reshape(len(test_ids), -1)
            if model_name == "ridge":
                model = Ridge(alpha=1.0).fit(x_train_flat, y_train)
                prediction = target_scaler.inverse_transform(np.asarray(model.predict(x_test_flat)).reshape(len(test_ids), horizon))
            elif model_name == "mlp":
                model = MLPRegressor(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=100,
                                     random_state=2026, early_stopping=False).fit(x_train_flat, y_train)
                prediction = target_scaler.inverse_transform(np.asarray(model.predict(x_test_flat)).reshape(len(test_ids), horizon))
            else:
                prediction = _fit_torch(model_name, x_train, y_train, x_test, horizon, lookback,
                                        test_close, target_mode, target_scaler.mean_, target_scaler.scale_)
            pred_price = prediction if target_mode == "future_close" else test_close[:, None] * np.exp(prediction)
        actual = closes[origins[test_ids, None] + np.arange(1, horizon + 1)]
        train_origins = origins[train_ids]
        adj = train_origins[1:][np.diff(train_origins) == 1]
        if p["bar_interval"] in {"1min", "5min", "1h"}:
            adj = adj[bars.index[adj].normalize() == bars.index[adj - 1].normalize()]
        scale = np.mean(np.abs(closes[adj] - closes[adj - 1])) if len(adj) else 0.0
        final_results.append({"model": model_name, "target_mode": target_mode, "lookback": lookback,
                              "horizon": horizon, "train_samples": len(train_ids), "test_samples": len(test_ids),
                              "test_start": bars.index[origins[test_ids[0]]].isoformat(),
                              "test_end": bars.index[origins[test_ids[-1]]].isoformat(),
                              "metrics": metrics(actual, pred_price, test_close, scale)})
    result = {**report_base, "status": "complete" if records else "insufficient_data", "rankings": records,
              "acceptance_criteria": {"accepted_run_exists": bool(accepted),
                                      "median_MASE_below_1": bool(accepted),
                                      "baseline_improvement_at_least_2_percent": bool(accepted),
                                      "at_least_3_of_5_folds_improve": bool(accepted),
                                      "accepted_runs": [{"model": r["model"], "target_mode": r["target_mode"],
                                                         "lookback": r["lookback"], "horizon": r["horizon"]}
                                                        for r in accepted]},
              "hybrid": {"status": "not_enabled", "reason": "out-of-fold error complementarity has not been established"},
              "final_test": {"status": "evaluated_once_after_validation_selection", "results": final_results},
              "note": "Final test results are reported separately and are not used to select models or windows."}
    (out_dir / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (out_dir / "REPORT.md").write_text(render_report(result))
    return result


def _compute_manifest():
    import torch
    cuda = torch.cuda.is_available()
    return {"device": torch.cuda.get_device_name(0) if cuda else "cpu",
            "cuda_available": cuda, "cpu_threads": 24, "cpu_dataloader_workers": 24,
            "cuda_dataloader_workers": 0 if cuda else None}


def _fit_torch(name, xtrain, ytrain, xvalid, horizon, lookback, last_close, target_mode, y_mean, y_scale):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    else:
        torch.set_num_threads(24)
    print(f"Training {name} on {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU (24 threads, 24 workers)'}", flush=True)
    if name == "xpatch":
        from config import Config
        from models.xpatch import XPatch
        cfg = Config()
        cfg.EMA_ALPHA, cfg.HIDDEN_SIZE, cfg.NUM_LAYERS = 0.1, 32, 1
        cfg.NUM_HEADS, cfg.D_FF, cfg.DROPOUT = 4, 64, 0.1
        cfg.PATCH_SIZES, cfg.PREDICTION_LENGTH = [min(5, lookback), min(10, lookback)], horizon
        model = XPatch(5, cfg)
        forward = lambda z: model(z)["regression_pred"]
    elif name in {"gru", "lstm"}:
        layer = nn.GRU if name == "gru" else nn.LSTM
        recurrent = layer(5, 32, batch_first=True)
        head = nn.Linear(32, horizon)
        class Recurrent(nn.Module):
            def __init__(self): super().__init__(); self.r, self.h = recurrent, head
            def forward(self, z):
                out = self.r(z)[0][:, -1]
                return self.h(out)
        model, forward = Recurrent(), lambda z: model(z)
    else:
        raise ValueError(f"Unknown model: {name}")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    tx = torch.tensor(xtrain, dtype=torch.float32)
    ty = torch.tensor(ytrain, dtype=torch.float32)
    vx = torch.tensor(xvalid, dtype=torch.float32, device=device)
    ys = torch.tensor(y_scale, dtype=torch.float32, device=device)
    ym = torch.tensor(y_mean, dtype=torch.float32, device=device)
    ty = ty.to(device)
    train_loader = DataLoader(TensorDataset(tx, ty), batch_size=128, shuffle=True,
                              num_workers=24 if device.type == "cpu" else 0,
                              persistent_workers=False)
    model.train()
    for _ in range(5):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = forward(bx)
            if pred.ndim == 3: pred = pred.squeeze(-1)
            loss = ((pred - by) ** 2).mean()
            loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        pred = forward(vx)
        if pred.ndim == 3: pred = pred.squeeze(-1)
        raw = pred.cpu().numpy() * ys.cpu().numpy() + ym.cpu().numpy()
    return raw if target_mode == "future_close" else last_close[:, None] * np.exp(raw)


def render_report(result):
    if result["status"] == "insufficient_data":
        return "# Experiment report\n\nStatus: `insufficient_data`\n\n" + "\n".join(f"- {x}" for x in result["reasons"]) + "\n"
    lines = [f"# {result['profile'].title()} AAPL experiment", "", f"Status: `{result['status']}`", "",
             f"Bars: {result['bars']} at {result['bar_interval']}; coverage: {result['history_years']:.2f} years.", "",
             "## Final test", "", "Selected by validation, then evaluated on the final chronological 20%:", "",
             "| Target | Model | Window | Horizon | Test RMSE | Test MASE |", "|---|---|---:|---:|---:|---:|"]
    for row in result.get("final_test", {}).get("results", []):
        lines.append(f"| {row['target_mode']} | {row['model']} | {row['lookback']} | {row['horizon']} | {row['metrics']['RMSE']:.6g} | {row['metrics']['MASE']:.4f} |")
    lines.extend(["", "## Validation", "", "Fold-level metrics, baseline comparisons, window rankings, and the acceptance decision are in `report.json`.",
                  "Final test scores are reported separately and do not affect selection.", ""])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=["annual", "quarterly", "monthly", "hourly", "minute"])
    parser.add_argument("--csv", type=Path, default=HERE / "data" / "AAPL_1min.csv")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--lookbacks", type=int, nargs="+")
    parser.add_argument("--bar-interval", choices=["1min", "5min", "1h", "1d", "1mo"])
    parser.add_argument("--models", nargs="+", choices=["last_close", "drift", "seasonal", "ridge", "mlp", "gru", "lstm", "xpatch"])
    args = parser.parse_args(argv)
    result = run_profile(args.profile, args.csv, args.output / args.profile, args.lookbacks, args.models, args.bar_interval)
    print(json.dumps({"status": result["status"], "reasons": result.get("reasons", []), "report": str(args.output / args.profile / "REPORT.md")}, indent=2))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
