"""Read-only SIP/raw data, independent labels, shared origins and nested dates."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ROOT, digest, read_json

OHLCV = ["open", "high", "low", "close", "volume"]


def sessions(start, end):
    import exchange_calendars as xc
    lower = str((pd.Timestamp(start) - pd.Timedelta(days=7)).date())
    upper = str((pd.Timestamp(end) + pd.Timedelta(days=7)).date())
    schedule = xc.get_calendar("XNYS", start=lower, end=upper).schedule.loc[start:end]
    return {str(d.date()): pd.date_range(s.open, s.close, freq="min", inclusive="left")
            for d, s in schedule.iterrows()}


def read_source(path, _sealed_admission=False, authorization=None):
    if Path(path).resolve().is_relative_to(ROOT / 'holdout_sealed') and not _sealed_admission:
        from .sealed import assert_access, event
        _, manifest = assert_access(authorization)
        if digest(path) != manifest['csv_sha256']:
            raise ValueError('Sealed data identity differs')
        event('authorized_read', {'csv_sha256': manifest['csv_sha256'], 'purpose': 'locked_final_test'})
    frame = pd.read_csv(path)
    required = ["timestamp", *OHLCV, "trade_count", "vwap"]
    if list(frame) != required or not len(frame):
        raise ValueError("Unexpected SIP/raw CSV schema")
    if not frame.timestamp.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True).all():
        raise ValueError("Every timestamp must have an explicit timezone")
    stamps = pd.DatetimeIndex(pd.to_datetime(frame.pop("timestamp"), utc=True, errors="raise"))
    if stamps.isna().any() or stamps.duplicated().any() or not stamps.is_monotonic_increasing:
        raise ValueError("Invalid/duplicate/unordered timestamps")
    if (stamps != stamps.floor("min")).any():
        raise ValueError("Not minute-boundary timestamps")
    frame = frame.apply(pd.to_numeric, errors="raise").astype(np.float64)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Nonfinite source values")
    if (frame[OHLCV[:4]] <= 0).any().any() or (frame.volume < 0).any():
        raise ValueError("Invalid source prices/volume")
    if ((frame.high < frame[["open", "low", "close"]].max(axis=1)) |
            (frame.low > frame[["open", "high", "close"]].min(axis=1))).any():
        raise ValueError("Invalid OHLC relationships")
    frame.index = stamps
    return frame


def verify_receipt(frame, directory, expected_hash):
    """Verify all saved page bytes, pagination and CSV fields; write nothing to legacy."""
    directory = Path(directory).resolve()
    receipt = read_json(directory / "receipt.json")
    if not receipt.get("download_complete") or receipt.get("csv_sha256") != expected_hash:
        raise ValueError("Source receipt incomplete or hash differs")
    if (receipt["symbol"], receipt["feed"], receipt["adjustment"], receipt["timeframe"]) != (
            "AAPL", "sip", "raw", "1Min"):
        raise ValueError("Unexpected request identity")
    if receipt["endpoint"] != "https://data.alpaca.markets/v2/stocks/bars":
        raise ValueError("Unexpected source endpoint")
    groups = {tuple(x): [] for x in receipt["ranges"]}
    if len(groups) != len(receipt["ranges"]):
        raise ValueError("Duplicate ranges")
    for row in receipt["pages"]:
        groups[(row["range_start"], row["range_end"])].append(row)
    seen = np.zeros(len(frame), dtype=bool)
    paths = set()
    for (start, end), pages in groups.items():
        pages.sort(key=lambda r: r["page"])
        if [x["page"] for x in pages] != list(range(len(pages))) or not pages:
            raise ValueError("Missing/non-contiguous pages")
        low = pd.Timestamp(start, tz="America/New_York").tz_convert("UTC")
        high = (pd.Timestamp(end) + pd.Timedelta(days=1)).tz_localize("America/New_York").tz_convert("UTC")
        tokens = set()
        for j, row in enumerate(pages):
            raw = (directory / row["file"]).resolve()
            if not raw.is_relative_to(directory / "raw") or raw in paths:
                raise ValueError("Unsafe/duplicate page path")
            paths.add(raw)
            if digest(raw) != row["sha256"] or raw.stat().st_size != row["bytes"]:
                raise ValueError("Raw page byte identity mismatch")
            payload = read_json(raw)
            bars = payload.get("bars", {}).get("AAPL", []) or []
            token = payload.get("next_page_token")
            if len(bars) != row["rows_returned"] or bool(token) != row["has_next_page"]:
                raise ValueError("Page row count/token mismatch")
            if bool(token) != (j < len(pages) - 1) or (token and token in tokens):
                raise ValueError("Incomplete/cyclic pagination")
            if token:
                tokens.add(token)
            rows = [(pd.Timestamp(b["t"]), b["o"], b["h"], b["l"], b["c"], b["v"], b["n"], b["vw"])
                    for b in bars if low <= pd.Timestamp(b["t"]) < high]
            if not rows:
                continue
            stamps = pd.DatetimeIndex([r[0] for r in rows]).tz_convert("UTC")
            idx = frame.index.get_indexer(stamps)
            if (idx < 0).any() or seen[idx].any() or len(np.unique(idx)) != len(idx):
                raise ValueError("Raw/CSV timestamp mismatch/overlap")
            if not np.allclose(frame.iloc[idx].to_numpy(), np.asarray([r[1:] for r in rows]),
                               rtol=1e-14, atol=1e-12):
                raise ValueError("Raw-to-CSV value mismatch")
            seen[idx] = True
    if not seen.all() or int(seen.sum()) != receipt["rows"]:
        raise ValueError("Unverified CSV rows")
    return {"pages_verified": len(paths), "ranges_verified": len(groups), "rows_verified": int(seen.sum()),
            "fields_verified": list(frame), "receipt_sha256": digest(directory / "receipt.json")}


@dataclass
class Market:
    bars: pd.DataFrame
    dates: np.ndarray
    returns: np.ndarray
    features: np.ndarray
    origins: np.ndarray
    audit: dict

    @classmethod
    def load(cls, config=None, full_receipt=False):
        config = config or read_json(ROOT / "protocol_draft.json")
        path = Path(config["source_csv_read_only"])
        checksum = digest(path)
        if checksum != config["source_csv_expected_sha256"]:
            raise ValueError("Source CSV hash differs from planned identity")
        all_bars = read_source(path)
        market = build_market(all_bars, "2021-01-04", "2026-06-30")
        market.audit.update(source_read_only=str(path), source_sha256=checksum)
        if full_receipt:
            market.audit['raw_receipt_verification'] = verify_receipt(all_bars, path.parent, checksum)
        return market

    @classmethod
    def load_holdout(cls, authorization):
        from .sealed import SEALED, MANIFEST
        manifest = read_json(MANIFEST)
        return build_market(read_source(SEALED / manifest['csv'], authorization=authorization),
                            manifest['start'], manifest['end'])

    def select(self, dates):
        return self.origins[np.isin(self.dates[self.origins], dates)]

    def scale(self, dates, panel="close"):
        values = self.features[np.isin(self.dates, dates)]
        values = values[np.isfinite(values).all(axis=1)]
        if panel == "close":
            values = values[:, 3:4]
        if not len(values):
            raise ValueError("Empty training scaler")
        std = values.std(axis=0)
        std[std == 0] = 1.0
        return {"center": values.mean(axis=0).tolist(), "std": std.tolist(),
                "target_std": float(values[:, 0 if panel == "close" else 3].std()),
                "dates": list(map(str, dates)), "panel": panel}

    def x(self, origins, lookback, scale):
        if lookback not in (48, 96, 240):
            raise ValueError("Unplanned lookback")
        idx = np.asarray(origins)[:, None] + np.arange(1 - lookback, 1)
        features = self.features[:, 3:4] if scale["panel"] == "close" else self.features
        x = ((features[idx] - scale["center"]) / scale["std"]).astype(np.float32)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite historical window")
        return x

    def y_steps(self, origins):
        return self.returns[np.asarray(origins)[:, None] + np.arange(1, 13)]

    def y(self, origins):
        o = np.asarray(origins)
        return np.log(self.bars.close.to_numpy()[o + 12] / self.bars.close.to_numpy()[o])

    def chronos_inputs(self, origins, lookback, representation="minute_return", panel="close"):
        items = []
        for origin in origins:
            idx = np.arange(origin - lookback + 1, origin + 1)
            if representation == "minute_return":
                target = self.returns[idx]
            elif representation == "anchored_log_price":
                target = np.log(self.bars.close.to_numpy()[idx] / self.bars.close.iloc[origin])
            else:
                raise ValueError("Unplanned representation")
            item = {"target": target.astype(np.float32)}
            if panel == "ohlcv":
                item["past_covariates"] = {f"feature_{c}": self.features[idx, c].astype(np.float32)
                                           for c in (0, 1, 2, 4)}
            items.append(item)
        return items


def build_market(all_bars, start, end):
        """Calendar/causality transform shared by development and authorized final-test loaders."""
        grids = sessions(start, end)
        expected = pd.DatetimeIndex(np.concatenate([x.as_unit("ns").asi8 for x in grids.values()]), tz="UTC")
        in_regular = all_bars.index.isin(expected)
        bars = all_bars.loc[in_regular].copy()
        if not bars.index.equals(expected):
            raise ValueError("Regular-session data has missing calendar minutes")
        dates = bars.index.tz_convert("America/New_York").strftime("%Y-%m-%d").to_numpy(dtype="U10")
        ns = bars.index.as_unit("ns").asi8
        adjacent = np.r_[False, (np.diff(ns) == 60_000_000_000) & (dates[1:] == dates[:-1])]
        close = bars.close.to_numpy()
        prev = np.roll(close, 1)
        prev[~adjacent] = np.nan
        returns = np.log(close / prev)
        features = np.column_stack([np.log(bars[c].to_numpy() / prev) for c in OHLCV[:4]] +
                                   [np.log1p(bars.volume.to_numpy())])
        idx = np.arange(len(bars))
        starts = np.maximum.accumulate(np.where(~adjacent, idx, 0))
        ends = np.r_[np.flatnonzero(~adjacent)[1:] - 1, len(bars) - 1]
        run_end = np.repeat(ends, np.diff(np.r_[0, ends + 1]))
        origins = idx[(idx - starts >= 240) & (run_end - idx >= 12)]
        daily = [{"date": d, "bars": len(g), "early_close": len(g) < 390,
                  "eligible_origins": int(np.sum(dates[origins] == d))} for d, g in grids.items()]
        audit = {"all_rows": len(all_bars),
                 "regular_rows": len(bars), "sessions": len(grids), "calendar": "XNYS",
                 "timestamp_label": "bar_start", "complete_calendar_coverage": True,
                 "eligible_origins": len(origins), "daily": daily}
        return Market(bars, dates, returns, features, origins, audit)


def development_splits(market):
    dates = np.asarray(sorted(set(market.dates)), dtype="U10")
    inner = []
    for name, quarter in (('V1', '2024Q4'), ('V2', '2025Q2'), ('V3', '2025Q4')):
        q = pd.Period(quarter, freq='Q')
        val = dates[(dates >= str(q.start_time.date())) & (dates <= str(q.end_time.date()))]
        past = dates[dates < val[0]]
        inner.append({'name': name, 'quarter': quarter, 'fit_dates': past[:-1].tolist(),
                      'purge_dates': past[-1:].tolist(), 'validation_dates': val.tolist()})
    confirmation = dates[(dates >= '2026-01-01') & (dates <= '2026-03-31')]
    past = dates[dates < confirmation[0]]
    calibration = dates[(dates >= '2026-06-01') & (dates <= '2026-06-30')][-20:]
    return {'schema_version': 2, 'fold': 'development', 'inner': inner,
            'confirmation': {'fit_dates': past[:-22].tolist(),
                             'purge_dates': np.r_[past[-22:-21], past[-1:]].tolist(),
                             'calibration_dates': past[-21:-1].tolist(),
                             'evaluation_dates': confirmation.tolist()},
            'final_fit_dates': dates[dates <= '2026-05-31'].tolist(),
            'calibration_dates': calibration.tolist(),
            'holdout_dates_calendar_only': list(sessions('2026-07-01', '2026-09-30'))}
