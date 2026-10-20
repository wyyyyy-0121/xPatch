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
    source_verified = False
    if sidecar.exists():
        declared = json.loads(sidecar.read_text())
        source_verified = (declared.get("source_verified") is True and declared.get("source")
                           and declared.get("timezone") and declared.get("sha256") == sha256(path))
        if declared.get("source_verified") is True and declared.get("sha256") != sha256(path):
            raise ValueError("CSV SHA-256 does not match its sidecar manifest")
        if source_verified:
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
            "source_verified": bool(source_verified),
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
    intraday = interval in {"1min", "5min"}
    hourly = interval == "1h"
    groups = (pd.Series(np.arange(len(bars)), index=timestamps).groupby(timestamps.normalize()).apply(list).tolist()
              if intraday else [list(range(len(bars)))])
    xs, ys, origins = [], [], []
    for ids in groups:
        for end in range(lookback - 1, len(ids) - horizon):
            i = ids[end]
            future_ids = ids[end + 1:end + horizon + 1]
            if len(future_ids) != horizon:
                continue
            if hourly and not np.all(timestamps[future_ids].normalize() == timestamps[ids[end]].normalize()):
                continue
            xids = ids[end - lookback + 1:end + 1]
            step = {"1min": pd.Timedelta(minutes=1), "5min": pd.Timedelta(minutes=5), "1h": pd.Timedelta(hours=1)}.get(interval)
            if step is not None:
                full_ids = ids[end - lookback + 1:end + horizon + 1]
                differences = np.diff(timestamps[full_ids])
                same_session = timestamps[full_ids[:-1]].normalize() == timestamps[full_ids[1:]].normalize()
                if not np.all(differences[same_session] == step):
                    continue
            xs.append(features[xids])
            future = close[future_ids]
            if target_mode == "future_close":
                ys.append(future)
            else:
                ys.append(np.log(future / close[i]))
            origins.append(i)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float64), np.asarray(origins, dtype=np.int64), close


def rolling_splits(origins: np.ndarray, dates: np.ndarray, folds=5, purge=1,
                   validation_ranges=None):
    order = np.argsort(origins)
    if validation_ranges is not None:
        splits = []
        for start, end in validation_ranges:
            valid = order[(origins[order] >= start) & (origins[order] < end)]
            train = order[origins[order] + purge < start]
            if len(train) and len(valid):
                splits.append((train, valid))
        return splits
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


def holdout_indices(origins: np.ndarray, horizon: int, test_fraction=.2,
                    cutoff_origin=None):
    if cutoff_origin is None:
        cut = int(np.floor(len(origins) * (1 - test_fraction)))
        if cut <= 0 or cut >= len(origins):
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        cutoff_origin = origins[cut]
    if cutoff_origin <= 0 or cutoff_origin >= origins.max():
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    test = np.flatnonzero(origins >= cutoff_origin)
    train = np.flatnonzero(origins + horizon < cutoff_origin)
    return train, test


def scale_fold(X_train, X_eval, y_train):
    from sklearn.preprocessing import StandardScaler
    feature_scaler = StandardScaler().fit(X_train.reshape(-1, X_train.shape[-1]))
    train_scaled = feature_scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    eval_scaled = feature_scaler.transform(X_eval.reshape(-1, X_eval.shape[-1])).reshape(X_eval.shape)
    target_scaler = StandardScaler().fit(y_train)
    target_scaled = target_scaler.transform(y_train)
    return train_scaled, eval_scaled, target_scaled, target_scaler


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


def run_profile(name: str, csv_path: Path, out_dir: Path, lookbacks=None, requested_models=None,
                bar_interval=None, horizons=None):
    from sklearn.linear_model import Ridge
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
    if horizons:
        if any(value <= 0 for value in horizons):
            raise ValueError("Horizons must be positive bar counts")
        p = {**p, "horizons": horizons}
    if lookbacks and any(value <= 0 for value in lookbacks):
        raise ValueError("Lookbacks must be positive bar counts")
    bars = aggregate_bars(raw, p["bar_interval"])
    issues = data_gate(name, bars, meta, profiles["requirements"])
    active_lookbacks = lookbacks or p["lookbacks"]
    holdout_cutoff = int(np.floor(len(bars) * .8))
    cv_edges = np.linspace(max(active_lookbacks) - 1, holdout_cutoff, 7, dtype=int)
    cv_ranges = [(int(cv_edges[i]), int(cv_edges[i + 1])) for i in range(1, 6)]
    models = requested_models or ["last_close", "drift", "seasonal", "ridge", "mlp", "gru", "lstm", "xpatch"]
    if "last_close" not in models:
        models = ["last_close", *models]
    out_dir.mkdir(parents=True, exist_ok=True)
    report_base = {"symbol": "AAPL", "profile": name, "bar_interval": p["bar_interval"], "data": meta,
                   "bars": len(bars), "history_years": round(years_covered(bars.index), 4),
                   "source_sha256": sha256(csv_path), "generated_utc": datetime.now(timezone.utc).isoformat(),
                   "code_sha256": sha256(Path(__file__)), "profile_config_sha256": sha256(PROFILE_PATH),
                   "data_limitations": (["Input source is recorded in a manifest but has not been independently verified."]
                                         if meta.get("sidecar_manifest") and not meta.get("source_verified") else
                                         (["Local CSV provenance and adjustment policy are unverified."]
                                          if "local CSV" in meta.get("source", "") else [])),
                   "compute": _compute_manifest(),
                   "experiment_config": {"lookbacks": lookbacks or p["lookbacks"], "horizons": p["horizons"],
                                         "target_modes": ["future_close", "future_log_return"],
                                         "models": models, "rolling_folds": 5,
                                         "validation": "expanding-window CV over the first 80 percent; model selection uses validation only",
                                         "purge_gap_bars": "equal to forecast horizon",
                                         "final_test_fraction": 0.2,
                                         "holdout_boundary": bars.index[holdout_cutoff].isoformat() if len(bars) > holdout_cutoff else None,
                                         "validation_boundaries": [[bars.index[start].isoformat(), bars.index[end].isoformat()]
                                                                    for start, end in cv_ranges if end < len(bars)],
                                         "ridge_alpha": 1.0, "mlp": {"hidden_layers": [64], "dropout": 0.1},
                                         "neural": {"architecture_width": 32, "epochs": 5, "batch_size": 128,
                                                     "optimizer": "Adam", "learning_rate": 0.001,
                                                     "weight_decay": 0.0001, "seed": 2026}}}
    if issues:
        result = {**report_base, "status": "insufficient_data", "reasons": issues, "rankings": []}
        (out_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        (out_dir / "REPORT.md").write_text(f"# {name} experiment\n\nStatus: `insufficient_data`\n\n" + "\n".join(f"- {x}" for x in issues) + "\n")
        return result
    records = []
    best_test_jobs = {}
    for target_mode in ("future_close", "future_log_return"):
      for horizon in p["horizons"]:
        for lookback in lookbacks or p["lookbacks"]:
            X, y, origins, closes = make_samples(bars, lookback, horizon, p["bar_interval"], target_mode)
            final_train, final_test = holdout_indices(origins, horizon, cutoff_origin=holdout_cutoff)
            if not len(final_train) or not len(final_test):
                continue
            development_ids = final_train
            local_splits = rolling_splits(origins[development_ids],
                                          bars.index[origins[development_ids]].normalize().to_numpy(),
                                          folds=5, purge=horizon, validation_ranges=cv_ranges)
            splits = [(development_ids[tr], development_ids[va]) for tr, va in local_splits]
            if len(splits) != 5:
                continue
            for model_name in models:
                fold_scores = []
                for fold_id, (tr, va) in enumerate(splits):
                    train_scaled, valid_scaled, ytr, y_scaler = scale_fold(X[tr], X[va], y[tr])
                    xtr, xva = train_scaled.reshape(len(tr), -1), valid_scaled.reshape(len(va), -1)
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
                    else:
                        pred = _fit_torch(model_name, train_scaled, ytr, valid_scaled, horizon, lookback,
                                          y_scaler.mean_, y_scaler.scale_)
                        pred_price = pred if target_mode == "future_close" else price_space(pred, target_mode, closes[origins[va]])
                    actual = closes[origins[va, None] + np.arange(1, horizon + 1)]
                    train_origin_set = origins[tr]
                    adjacent = np.diff(train_origin_set) == 1
                    adjacent_origins = train_origin_set[1:][adjacent]
                    if p["bar_interval"] in {"1min", "5min", "1h"}:
                        same_session = bars.index[adjacent_origins].normalize() == bars.index[adjacent_origins - 1].normalize()
                        adjacent_origins = adjacent_origins[same_session]
                    mase_scale = np.mean(np.abs(closes[adjacent_origins] - closes[adjacent_origins - 1])) if len(adjacent_origins) else 0.0
                    score = metrics(actual, pred_price, closes[origins[va]], mase_scale)
                    start_origin = origins[va[0]]
                    end_origin = origins[va[-1]] + horizon
                    validation_return = closes[end_origin] / closes[start_origin] - 1
                    period_returns = np.diff(np.log(closes[start_origin:end_origin + 1]))
                    if p["bar_interval"] in {"1min", "5min", "1h"}:
                        pair_ids = np.arange(start_origin + 1, end_origin + 1)
                        same_session = bars.index[pair_ids].normalize() == bars.index[pair_ids - 1].normalize()
                        period_returns = period_returns[same_session]
                    phase = "up" if validation_return > .02 else ("down" if validation_return < -.02 else "flat")
                    fold_scores.append({"fold": fold_id + 1, "metrics": score, "train_samples": len(tr),
                                        "validation_samples": len(va),
                                        "validation_start": bars.index[start_origin].isoformat(),
                                        "validation_end": bars.index[end_origin].isoformat(),
                                        "realized_return": float(validation_return), "market_phase": phase,
                                        "realized_volatility": float(np.std(period_returns)) if len(period_returns) else 0.0})
                if fold_scores:
                    baseline = next((r for r in records if r["model"] == "last_close" and r["target_mode"] == target_mode
                                     and r["lookback"] == lookback and r["horizon"] == horizon), None)
                    improvements = []
                    if model_name == "last_close":
                        improvements = [False] * len(fold_scores)
                    elif baseline:
                        gains = [1 - f["metrics"]["RMSE"] / max(b["metrics"]["RMSE"], 1e-12)
                                 for f, b in zip(fold_scores, baseline["folds"])]
                        improvements = [gain > 0 for gain in gains]
                        for fold, gain in zip(fold_scores, gains):
                            fold["baseline_rmse_gain"] = float(gain)
                    else:
                        gains = []
                    record = {"model": model_name, "target_mode": target_mode, "lookback": lookback,
                                    "horizon": horizon, "folds": fold_scores,
                                    "median_MASE": float(np.median([f["metrics"]["MASE"] for f in fold_scores])),
                                    "median_RMSE": float(np.median([f["metrics"]["RMSE"] for f in fold_scores])),
                                    "baseline_improvement_fraction": float(np.mean(improvements)) if improvements else 0.0,
                                    "median_baseline_RMSE_gain": float(np.median(gains)) if baseline and model_name != "last_close" else 0.0,
                                    "underperforming_folds": [{"fold": f["fold"], "start": f["validation_start"],
                                                               "end": f["validation_end"], "market_phase": f["market_phase"],
                                                               "realized_return": f["realized_return"],
                                                               "realized_volatility": f["realized_volatility"]}
                                                              for f in fold_scores if f.get("baseline_rmse_gain", 0) < 0]}
                    records.append(record)
                    selection_key = (record["median_MASE"], record["median_RMSE"])
                    current = best_test_jobs.get((target_mode, horizon))
                    if current is None or selection_key < current["selection_key"]:
                        best_test_jobs[(target_mode, horizon)] = {"selection_key": selection_key,
                            "target_mode": target_mode, "horizon": horizon, "lookback": lookback,
                            "model": model_name}
    records.sort(key=lambda r: (r["median_MASE"], r["median_RMSE"]))
    accepted = [r for r in records if r["model"] != "last_close" and r["median_MASE"] < 1
                and r["median_baseline_RMSE_gain"] >= .02 and r["baseline_improvement_fraction"] >= .6]
    final_results = []
    for selected in best_test_jobs.values():
        target_mode, horizon = selected["target_mode"], selected["horizon"]
        lookback, model_name = selected["lookback"], selected["model"]
        X, y, origins, closes = make_samples(bars, lookback, horizon, p["bar_interval"], target_mode)
        train_ids, test_ids = holdout_indices(origins, horizon, cutoff_origin=holdout_cutoff)
        if not len(train_ids) or not len(test_ids):
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
            x_train, x_test = X[train_ids], X[test_ids]
            x_train, x_test, y_train, target_scaler = scale_fold(x_train, x_test, y[train_ids])
            x_train_flat, x_test_flat = x_train.reshape(len(train_ids), -1), x_test.reshape(len(test_ids), -1)
            if model_name == "ridge":
                model = Ridge(alpha=1.0).fit(x_train_flat, y_train)
                prediction = target_scaler.inverse_transform(np.asarray(model.predict(x_test_flat)).reshape(len(test_ids), horizon))
            else:
                prediction = _fit_torch(model_name, x_train, y_train, x_test, horizon, lookback,
                                        target_scaler.mean_, target_scaler.scale_)
            pred_price = prediction if target_mode == "future_close" else price_space(prediction, target_mode, test_close)
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
        del X, y, origins, closes
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
              "validation_diagnostic": {"status": "stable_advantage_found" if accepted else "no_candidate_met_acceptance_thresholds",
                                        "underperforming_windows": [{"model": r["model"], "target_mode": r["target_mode"],
                                                                     "lookback": r["lookback"], "horizon": r["horizon"],
                                                                     "fold": f["fold"], "market_phase": f["market_phase"],
                                                                     "start": f["start"], "end": f["end"]}
                                                                    for r in records for f in r["underperforming_folds"]]},
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


def _fit_torch(name, xtrain, ytrain, xvalid, horizon, lookback, y_mean, y_scale):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    else:
        torch.set_num_threads(24)
    torch.manual_seed(2026)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(2026)
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
    elif name == "mlp":
        model = nn.Sequential(nn.Flatten(), nn.Linear(lookback * 5, 64), nn.ReLU(),
                              nn.Dropout(.1), nn.Linear(64, horizon))
        forward = lambda z: model(z)
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
    return raw


def render_report(result):
    if result["status"] == "insufficient_data":
        return "# Experiment report\n\nStatus: `insufficient_data`\n\n" + "\n".join(f"- {x}" for x in result["reasons"]) + "\n"
    lines = [f"# {result['profile'].title()} AAPL experiment", "", f"Status: `{result['status']}`", "",
             f"Bars: {result['bars']} at {result['bar_interval']}; coverage: {result['history_years']:.2f} years.", "",
             "## Final test", "", "Selected by validation, then evaluated on the final chronological 20%:", "",
             "| Target | Model | Window | Horizon | Test RMSE | Test MASE |", "|---|---|---:|---:|---:|---:|"]
    for row in result.get("final_test", {}).get("results", []):
        lines.append(f"| {row['target_mode']} | {row['model']} | {row['lookback']} | {row['horizon']} | {row['metrics']['RMSE']:.6g} | {row['metrics']['MASE']:.4f} |")
    lines.extend(["", "## Validation comparison", "",
                  "| Target | Model | Window | Horizon | Median MASE | Median RMSE | Baseline RMSE gain | Folds improved |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in result.get("rankings", [])[:20]:
        lines.append(f"| {row['target_mode']} | {row['model']} | {row['lookback']} | {row['horizon']} | "
                     f"{row['median_MASE']:.4f} | {row['median_RMSE']:.6g} | "
                     f"{row['median_baseline_RMSE_gain']:.2%} | {row['baseline_improvement_fraction']:.0%} |")
    accepted = result.get("acceptance_criteria", {}).get("accepted_run_exists", False)
    lines.extend(["", f"Acceptance threshold met: `{str(accepted).lower()}`.", "",
                  "## Weak validation periods", ""])
    weak = result.get("validation_diagnostic", {}).get("underperforming_windows", [])
    if weak:
        for row in weak[:12]:
            lines.append(f"- {row['model']} / {row['target_mode']} (window {row['lookback']}, horizon {row['horizon']}): "
                         f"fold {row['fold']} {row['market_phase']} period, {row['start']} to {row['end']}.")
        if len(weak) > 12:
            lines.append(f"- {len(weak) - 12} additional underperforming folds are listed in `report.json`.")
    else:
        lines.append("No validation fold was worse than the last-close baseline.")
    if not accepted:
        lines.extend(["", "No tested configuration met all acceptance criteria; the validation results do not support a stable predictive advantage."])
    if result.get("data_limitations"):
        lines.extend(["", "## Data limitations", "", *[f"- {limitation}" for limitation in result["data_limitations"]]])
    lines.extend(["", "Per-fold metrics, per-step errors, data manifest, and the complete experiment configuration are in `report.json`.",
                  "Final test scores are reported separately and do not affect selection.", ""])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=["annual", "quarterly", "monthly", "hourly", "minute"])
    parser.add_argument("--csv", type=Path, default=HERE / "data" / "AAPL_1min.csv")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--lookbacks", type=int, nargs="+")
    parser.add_argument("--horizons", type=int, nargs="+")
    parser.add_argument("--bar-interval", choices=["1min", "5min", "1h", "1d", "1mo"])
    parser.add_argument("--models", nargs="+", choices=["last_close", "drift", "seasonal", "ridge", "mlp", "gru", "lstm", "xpatch"])
    args = parser.parse_args(argv)
    result = run_profile(args.profile, args.csv, args.output / args.profile, args.lookbacks,
                         args.models, args.bar_interval, args.horizons)
    print(json.dumps({"status": result["status"], "reasons": result.get("reasons", []), "report": str(args.output / args.profile / "REPORT.md")}, indent=2))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
