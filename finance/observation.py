"""Actual-receipt SIP data, immutable event chains, and exchange session grids."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .io import ROOT, read_json, stable_hash


UTC = timezone.utc
ASSETS = ('AAPL', 'SPY', 'QQQ')


def utc_now():
    return datetime.now(UTC).isoformat()


def stamp(value):
    value = pd.Timestamp(value)
    if value.tzinfo is None:
        raise ValueError('A prospective timestamp must include its timezone')
    return value.tz_convert('UTC')


def session_origins(day):
    from finance_v2.data import calendar_grid
    grid = calendar_grid(str(day), str(day))
    if str(day) not in grid:
        return pd.DatetimeIndex([], tz='UTC')
    minute_starts = grid[str(day)]
    return pd.date_range(minute_starts[0] + pd.Timedelta(minutes=15),
        minute_starts[-1] + pd.Timedelta(minutes=1 - 30), freq='min')


def immutable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Publish a complete, fsynced file atomically without replacing a prior one.
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.pending-', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, path)
            descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            temporary.unlink(missing_ok=True)


class Journal:
    """Callers hold the observer/driver lock. Existing records cannot be replaced."""
    def __init__(self, directory, signature):
        self.directory, self.signature = Path(directory), signature
        self.events = list(self.read())
        self.last_hash = self.events[-1]['hash'] if self.events else None
        self.last_time = stamp(self.events[-1]['recorded_at']) if self.events else None

    def read(self):
        previous = None
        last_time = None
        for number, path in enumerate(sorted((self.directory/'events').glob('*.json')), 1):
            entry = read_json(path)
            payload = {k:v for k,v in entry.items() if k != 'hash'}
            at = stamp(entry['recorded_at'])
            if (path.name != f'{number:08d}.json' or entry['sequence'] != number or
                    entry['plan_signature'] != self.signature or entry['previous_hash'] != previous or
                    stable_hash(payload) != entry['hash'] or (last_time is not None and at < last_time)):
                raise ValueError('Observation chain has a missing, altered or reordered event')
            previous, last_time = entry['hash'], at
            yield entry

    def append(self, kind, payload, recorded_at=None):
        at = stamp(recorded_at or utc_now())
        if self.last_time is not None and at < self.last_time:
            raise ValueError('Wall clock moved backwards; observation stopped')
        value = {'sequence':len(self.events)+1, 'plan_signature':self.signature,
            'previous_hash':self.last_hash, 'recorded_at':at.isoformat(), 'kind':kind, 'payload':payload}
        value['hash'] = stable_hash(value)
        immutable_json(self.directory/'events'/f"{value['sequence']:08d}.json", value)
        self.events.append(value)
        self.last_hash, self.last_time = value['hash'], at
        return value


class BarStore:
    def __init__(self):
        self.bars = {name:{} for name in ASSETS}
        self.first_live_aapl_closes = {}
        self.last_received_at = None
        self.last_event_hash = None

    def apply(self, event):
        if event['kind'] != 'bars':
            return
        data = event['payload']
        if data['feed'] != 'sip' or data['adjustment'] != 'raw' or data['http_status'] != 200:
            raise ValueError('Only actual SIP/raw success responses can supply observation bars')
        received = stamp(data['received_at'])
        if received > stamp(event['recorded_at']):
            raise ValueError('Response was recorded before it was actually received')
        cutoff = stamp(data['end'])
        for asset, bars in data['response'].get('bars', {}).items():
            if asset not in ASSETS:
                raise ValueError('Unexpected asset in observation data')
            for bar in bars:
                t = stamp(bar['t'])
                if t > cutoff or t + pd.Timedelta(minutes=1) > received:
                    raise ValueError('A partial or future bar cannot supply a forecast')
                row = {key:float(bar[short]) for key,short in (('open','o'),('high','h'),('low','l'),('close','c'),('volume','v'))}
                if not np.isfinite(list(row.values())).all() or min(row[k] for k in ('open','high','low','close')) <= 0 or row['volume'] < 0:
                    raise ValueError('Invalid received OHLCV observation')
                if row['high'] < max(row['open'],row['low'],row['close']) or row['low'] > min(row['open'],row['high'],row['close']):
                    raise ValueError('Received price range is inconsistent')
                self.bars[asset][t] = row
                if asset == 'AAPL' and data['purpose'] == 'live':
                    self.first_live_aapl_closes.setdefault(t, row['close'])
        self.last_received_at, self.last_event_hash = received, event['hash']

    def frames(self, cutoff=None):
        result = {}
        limit = stamp(cutoff) if cutoff is not None else None
        for asset, bars in self.bars.items():
            rows = {t:r for t,r in bars.items() if limit is None or t <= limit}
            if not rows:
                raise ValueError('Received history is insufficient')
            result[asset] = pd.DataFrame.from_dict(rows, orient='index').sort_index()
            result[asset].index = pd.DatetimeIndex(result[asset].index)
        return result

    def target(self, origin):
        times = stamp(origin) + pd.to_timedelta(np.arange(-1, 30), unit='min')
        values = np.asarray([self.first_live_aapl_closes.get(t, np.nan) for t in times])
        if not np.isfinite(values).all():
            raise ValueError('Actual live target minutes are incomplete')
        return float(np.sum(np.diff(np.log(values))**2))


class DataUnavailable(PermissionError):
    def __init__(self, status):
        self.http_status = status
        super().__init__(f'SIP data unavailable (HTTP {status}); no feed fallback or purchase performed')


class Receiver:
    """Market-data-only client; API credentials never enter events or output."""
    def __init__(self, journal):
        from data_acquisition.delivery_data import Client
        self.client, self.journal = Client(), journal

    def fetch(self, start, end, purpose):
        if purpose not in ('warmup','live','admission'):
            raise ValueError('Unknown receipt purpose')
        token = None
        events = []
        while True:
            params = {'symbols':','.join(ASSETS), 'timeframe':'1Min','feed':'sip','adjustment':'raw',
                'start':stamp(start).isoformat(),'end':stamp(end).isoformat(),'limit':10000,'sort':'asc'}
            if token:
                params['page_token'] = token
            try:
                status, body, headers = self.client.get('/v2/stocks/bars', params)
            except RuntimeError:
                self.journal.append('data_unavailable', {'http_status':None,'received_at':utc_now(),
                    'feed':'sip','purpose':purpose,'reason':'market_data_network_unavailable'})
                raise DataUnavailable(None) from None
            received = utc_now()
            if status != 200:
                self.journal.append('data_unavailable', {'http_status':status,'received_at':received,
                    'feed':'sip','purpose':purpose,'reason':'permission_denied' if status == 403 else 'market_data_request_failed'})
                raise DataUnavailable(status)
            response = json.loads(body)
            event = self.journal.append('bars', {'http_status':status, 'feed':'sip','adjustment':'raw',
                'purpose':purpose,'start':params['start'],'end':params['end'],'received_at':received,
                'server_headers':headers, 'response':response})
            events.append(event)
            token = response.get('next_page_token')
            if not token:
                return events


def complete_days(events, config):
    """Quality alone selects days. No loss or model ranking is calculated here."""
    store, predictions, seen, commits = BarStore(), {}, set(), {}
    for event in events:
        if event['kind'] == 'bars':
            store.apply(event)
        elif event['kind'] == 'prediction':
            p = event['payload'];origin = stamp(p['origin']);key=origin.isoformat()
            if key in seen:
                raise ValueError('Duplicate prospective origin, including an attempted replacement')
            seen.add(key)
            predictions[key] = {**p, '_event_hash':event['hash']}
        elif event['kind'] == 'issuance_commit':
            reference = event['payload']['prediction_hash']
            if reference in commits:
                raise ValueError('Duplicate prediction commit acknowledgement')
            commits[reference] = stamp(event['recorded_at'])
    days = {stamp(p['origin']).tz_convert('America/New_York').strftime('%Y-%m-%d') for p in predictions.values()}
    days.update(stamp(e['payload']['received_at']).tz_convert('America/New_York').strftime('%Y-%m-%d')
        for e in events if e['kind']=='bars' and e['payload']['purpose']=='live')
    days = sorted(days)
    models = set(config['candidate_keys']) | set(config['reference_keys'])
    quality, qualified = {}, []
    for day in days:
        expected = session_origins(day)
        good, missing, late, missing_targets = 0, 0, 0, 0
        for origin in expected:
            p = predictions.get(origin.isoformat())
            if p is None or p['status'] != 'issued' or set(p.get('predictions',{})) != models:
                missing += 1
                continue
            committed = commits.get(p['_event_hash'])
            if (committed is None or not stamp(p['received_at']) <= stamp(p['issued_at']) <= committed < origin or
                    stamp(p['data_cutoff_at']) != origin-pd.Timedelta(minutes=1)):
                late += 1
                continue
            try:
                store.target(origin)
            except ValueError:
                missing_targets += 1
                continue
            good += 1
        complete = len(expected) > 0 and good == len(expected)
        quality[day] = {'complete':complete,'expected_origins':len(expected),'complete_origins':good,
            'missing_or_unavailable_origins':missing,'late_origins':late,'incomplete_targets':missing_targets}
        if complete:
            qualified.append(day)
    return {'days':quality,'complete_dates':qualified,'complete_sessions':len(qualified),
        'minimum_complete_sessions':config['minimum_complete_sessions'],
        'ready_for_final_evaluation':len(qualified)>=config['minimum_complete_sessions'],
        'cumulative_predictive_scores_disclosed':False}
