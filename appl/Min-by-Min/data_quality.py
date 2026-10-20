"""Fail-closed provenance and XNYS calendar validation. Bars are start-labelled."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

OHLCV = ('open', 'high', 'low', 'close', 'volume')
INTRADAY = {'1min': 1, '5min': 5, '1h': 60}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=32)
def schedule(start, end):
    import exchange_calendars as xc
    calendar = xc.get_calendar('XNYS', start=pd.Timestamp(start)-pd.Timedelta(days=7),
                               end=pd.Timestamp(end)+pd.Timedelta(days=7))
    return calendar.schedule.loc[start:end]


def as_market(index):
    index = pd.DatetimeIndex(index)
    return (index.tz_localize('America/New_York', ambiguous='raise', nonexistent='raise')
            if index.tz is None else index.tz_convert('America/New_York'))


def expected_index(index, interval):
    if not len(index):
        return pd.DatetimeIndex([], tz='America/New_York')
    local = as_market(index)
    days = schedule(str(local.min().date()), str(local.max().date()))
    parts = []
    width = INTRADAY[interval]
    for row in days.itertuples():
        op, cl = row.open, row.close
        count = int((cl - op).total_seconds() // (60 * width))
        if count:
            parts.append(pd.date_range(op, periods=count, freq=f'{width}min'))
    if not parts:
        return pd.DatetimeIndex([], tz='America/New_York')
    result = parts[0]
    for part in parts[1:]:
        result = result.append(part)
    return result.tz_convert('America/New_York')


def provenance(path, declared):
    """A boolean/source string is never evidence. Require a hash-bound receipt."""
    record = declared.get('provenance', {})
    evidence = record.get('evidence_path')
    if not evidence or not declared.get('timezone') or declared.get('symbol') != 'AAPL':
        return False
    evidence = (path.parent / evidence).resolve()
    if not evidence.is_file() or sha256(evidence) != record.get('evidence_sha256'):
        return False
    receipt = json.loads(evidence.read_text())
    if record.get('method') == 'provider_download':
        return (receipt.get('method') == 'provider_download'
                and receipt.get('provider') == 'Yahoo Finance'
                and receipt.get('url', '').startswith('https://query1.finance.yahoo.com/')
                and receipt.get('csv_sha256') == sha256(path)
                and receipt.get('downloaded_utc') == declared.get('downloaded_utc'))
    # Manual reviews must name evidence and reviewer; self-asserted source_verified is ignored.
    return (record.get('method') == 'manual_review'
            and receipt.get('decision') == 'verified'
            and bool(receipt.get('reviewer')) and bool(receipt.get('source_url'))
            and bool(receipt.get('review_evidence'))
            and receipt.get('csv_sha256') == sha256(path)
            and receipt.get('timezone') == declared.get('timezone'))


def load_ohlcv(path):
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + '.manifest.json')
    declared = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    digest = sha256(path)
    if declared.get('sha256') and declared['sha256'] != digest:
        raise ValueError('CSV SHA-256 does not match its sidecar manifest')
    raw = pd.read_csv(path)
    cols = {str(c).strip().lower(): c for c in raw.columns}
    time = next((cols[c] for c in ('timestamp', 'datetime', 'date') if c in cols), None)
    if time is None or any(c not in cols for c in OHLCV):
        raise ValueError('CSV needs timestamp and open, high, low, close, volume columns')
    if raw.empty:
        raise ValueError('CSV contains no rows')
    df = raw[[time, *(cols[c] for c in OHLCV)]].copy()
    df.columns = ['timestamp', *OHLCV]
    # Mixed UTC offsets (DST) require UTC parsing; naive timestamps require a declaration.
    stamps = df.pop('timestamp').astype(str)
    has_offset = stamps.str.contains(r'(?:Z|[+-]\d\d:\d\d)$').all()
    index = pd.DatetimeIndex(pd.to_datetime(stamps, errors='raise', utc=bool(has_offset)))
    if index.tz is None and declared.get('timezone') not in (None, 'None', 'daily date, timezone not applicable'):
        index = index.tz_localize(declared['timezone'], ambiguous='raise', nonexistent='raise')
    df.index = index
    for c in OHLCV:
        df[c] = pd.to_numeric(df[c], errors='raise')
    values = df.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('CSV contains missing or non-finite OHLCV values')
    if (values[:, :4] <= 0).any() or (df.volume < 0).any():
        raise ValueError('Prices must be positive and volume non-negative')
    if (df.high < df[['open', 'low', 'close']].max(axis=1)).any() or (df.low > df[['open', 'high', 'close']].min(axis=1)).any():
        raise ValueError('OHLC rows violate high/low bounds')
    duplicates = int(df.index.duplicated(keep='last').sum())
    df = df[~df.index.duplicated(keep='last')].sort_index()
    verified = provenance(path, declared)
    meta = dict(rows_input=len(raw), duplicate_timestamps_removed=duplicates,
                timestamp_timezone=str(df.index.tz) if df.index.tz else 'naive / unspecified',
                start=df.index.min().isoformat(), end=df.index.max().isoformat(),
                source=declared.get('source', 'local CSV; provenance unverified'),
                source_verified=bool(verified), sha256=digest,
                sidecar_manifest=str(sidecar) if sidecar.exists() else None,
                provider_url=declared.get('provider_url'), adjustment=declared.get('adjustment'),
                declared_interval=declared.get('interval'), timestamp_label=declared.get('timestamp_label'),
                synthetic=declared.get('synthetic', False))
    # Detect continuous interpolation even when a misleading manifest is supplied.
    if len(df) >= 100:
        delta2 = np.diff(df.to_numpy(dtype=float), n=2, axis=0)
        linear = np.all(np.abs(delta2) <= np.array([1e-9]*4 + [1e-4]), axis=1)
        meta['linear_second_difference_fraction'] = float(linear.mean())
        # Constant quotes are common; require changing prices AND fractional volumes.
        meta['suspected_interpolation'] = bool(linear.mean() > .95
            and (np.abs(df.volume % 1) > 1e-5).mean() > .5
            and np.ptp(df.close.to_numpy()) > 1e-6)
    is_intraday = bool((df.index != df.index.normalize()).any())
    if is_intraday:
        expected = expected_index(df.index, '1min')
        observed = as_market(df.index)
        observed = observed[observed.isin(expected)]
        expected_counts = pd.Series(1, index=expected).groupby(expected.normalize()).sum()
        observed_counts = pd.Series(1, index=observed).groupby(observed.normalize()).sum()
        coverage = observed_counts.reindex(expected_counts.index, fill_value=0) / expected_counts
        meta.update(expected_regular_minutes=len(expected), observed_regular_minutes=len(observed),
                    calendar_minute_coverage=float(len(observed)/len(expected)) if len(expected) else 0.,
                    minimum_session_coverage=float(coverage.min()) if len(coverage) else 0.,
                    median_within_day_minute_coverage=float(coverage.median()) if len(coverage) else 0.,
                    missing_sessions=int((coverage == 0).sum()),
                    sessions=int(len(expected_counts)), outside_regular_rows=len(df)-len(observed))
    return df, meta


def aggregate_bars(df, interval):
    agg = dict(open='first', high='max', low='min', close='last', volume='sum')
    if interval in INTRADAY:
        local = df.copy()
        local.index = as_market(local.index)
        minute_grid = expected_index(local.index, '1min')
        local = local.loc[local.index.isin(minute_grid)]
        grid = expected_index(df.index, interval)
        if interval == '1min':
            result = local.copy()
        else:
            pieces = []
            width = INTRADAY[interval]
            # Calendar bins start at the exchange open, never at the first observed row.
            for day, group in local.groupby(local.index.normalize()):
                day_grid = grid[grid.normalize() == day]
                if not len(day_grid):
                    continue
                origin = day_grid[0]
                grouped = group.resample(f'{width}min', origin=origin)
                bars = grouped.agg(agg)
                counts = grouped.close.count()
                pieces.append(bars.loc[(counts == width) & bars.index.isin(day_grid)])
            result = pd.concat(pieces) if pieces else local.iloc[:0].copy()
        result.attrs['sequence_ids'] = grid.get_indexer(result.index).tolist()
        result.attrs['bar_interval'] = interval
        return result
    if interval == '1d':
        result = df.groupby(df.index.normalize()).agg(agg)
    elif interval == '1mo':
        result = df.resample('MS').agg(agg).dropna()
        # Native month bars are allowed; daily inputs must contain every session
        # in a calendar month, including both boundary months.
        if len(df) > 1 and pd.Series(df.index).diff().median() < pd.Timedelta(days=20):
            keep = []
            all_sessions = schedule(str(result.index[0].date()),
                                    str((result.index[-1]+pd.offsets.MonthEnd(0)).date())).index
            for stamp in result.index:
                required = all_sessions[(all_sessions.year == stamp.year) & (all_sessions.month == stamp.month)]
                actual = df.index[(df.index.year == stamp.year) & (df.index.month == stamp.month)]
                actual = actual.tz_localize(None).normalize() if actual.tz else actual.normalize()
                required = required.tz_localize(None) if required.tz else required
                keep.append(len(required) > 0 and required.isin(actual).all())
            result = result.loc[keep]
    else:
        raise ValueError(f'Unsupported interval {interval}')
    return result


def data_gate(profile, bars, meta, requirements):
    req = requirements[profile]
    years = ((bars.index[-1] - bars.index[0]).total_seconds()/31556952) if len(bars) else 0.
    issues = []
    if years < req.get('minimum_years', 0):
        issues.append(f"history {years:.2f} years < required {req['minimum_years']} years")
    if years*12 < req.get('minimum_months', 0):
        issues.append(f"history {years*12:.1f} months < required {req['minimum_months']} months")
    if not len(bars):
        issues.append('no complete eligible bars')
    if profile in ('hourly', 'minute'):
        if meta.get('timestamp_timezone', 'naive').startswith('naive'):
            issues.append('intraday timezone is unspecified')
        if meta.get('source_verified') is not True:
            issues.append('intraday source/provenance lacks a hash-bound verification receipt')
        if meta.get('declared_interval') != '1min' or meta.get('timestamp_label') != 'bar_start':
            issues.append('intraday source must declare one-minute bar_start timestamps')
        if not meta.get('adjustment'):
            issues.append('intraday corporate-action adjustment policy is unspecified')
        minimum = req.get('minimum_session_coverage', .98)
        for field in ('calendar_minute_coverage', 'minimum_session_coverage'):
            coverage = meta.get(field, 0.)
            if coverage < minimum:
                issues.append(f'{field} {coverage:.4f} < required {minimum:.4f}; full XNYS sessions including missing days')
        if meta.get('synthetic') or meta.get('suspected_interpolation'):
            issues.append('synthetic/interpolated input is ineligible for real intraday experiments')
    return issues
