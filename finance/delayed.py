"""Replay the frozen market-time forecast 15 minutes later, with honest wall times."""
from contextlib import contextmanager
from functools import lru_cache
from types import FunctionType

import numpy as np
import pandas as pd

from . import prospective as original
from .delayed_validation import MODE, DELAY, complete_days, session_origins
from .io import ROOT, read_json, sha, stable_hash, write_json
from .observation import BarStore, DataUnavailable, Journal, Receiver, immutable_json, stamp, utc_now


PLAN = 'delivery/delayed_sip_lock.json'
SETTINGS = 'configs/rv30_delayed_sip.json'
EXECUTION = 'delivery/rv30_delayed_execution_lock.json'
OBSERVATION = 'artifacts/rv30/delayed_replay'
_read_original = original.read_plan


@lru_cache(maxsize=4)
def _calendar(start, end):
    """Cache calendar-only arrays; never cache received prices or fitted features."""
    from finance_v2.data import calendar_grid
    grids = calendar_grid(start, end)
    if not grids:
        raise ValueError('No sessions in requested range')
    days = np.array(list(grids))
    grid = pd.DatetimeIndex(np.concatenate([g.as_unit('ns').asi8 for g in grids.values()]), tz='UTC')
    ends = grid+pd.Timedelta(minutes=1)
    dates = np.asarray(ends.tz_convert('America/New_York').strftime('%Y-%m-%d'))
    keep = np.r_[False, dates[1:] == dates[:-1]] & np.r_[False, np.diff(grid.asi8) == 60*10**9]
    dates = dates[keep]; mapping = {day:i for i,day in enumerate(days)}
    index = np.array([mapping[day] for day in dates])
    closes = {day:g[-1]+pd.Timedelta(minutes=1) for day,g in grids.items()}
    for array in (days, dates, keep, index):
        array.setflags(write=False)
    return grid, ends[keep], days, dates, keep, index, closes


def causal_data(frames, start, end):
    """Same operations as RVData.from_frames; share only immutable session metadata."""
    from .data import ASSETS, RVData
    if set(frames) != set(ASSETS):
        raise ValueError('Paired comparison requires all three registered assets')
    grid, ends, days, dates, keep, index, closes = _calendar(start, end)
    returns, volume, ranges, audits = [], [], [], {}
    for asset in ASSETS:
        frame = frames[asset]
        if frame.index.tz is None or frame.index.duplicated().any():
            raise ValueError('Asset timestamps must be aware and unique')
        aligned = frame.reindex(grid)
        prices = aligned.close.to_numpy(dtype=np.float64)
        observed = np.isfinite(prices)
        ret = np.r_[np.nan, np.diff(np.log(prices))]; ret[~keep] = np.nan
        returns.append(ret[keep])
        volume.append(aligned.volume.to_numpy(dtype=np.float64)[keep])
        ranges.append(np.log(aligned.high.to_numpy(dtype=np.float64)/aligned.low.to_numpy(dtype=np.float64))[keep])
        audits[asset] = {'regular_rows':int(observed.sum()), 'expected_rows':len(grid),
                         'missing_rows':int((~observed).sum())}
    return RVData(ends, dates, days, index, np.column_stack(returns), np.column_stack(volume),
                  np.column_stack(ranges), closes.copy(), audits)


class Predictor(original.Predictor):
    """Original model loading, features and prediction math, with calendar reuse."""
    def predict(self, store, origin):
        from .features import FeatureStore
        from .models import raw_predictions
        origin = stamp(origin)
        frames = store.frames(origin-pd.Timedelta(minutes=2))
        history_start = origin.normalize()-pd.Timedelta(days=80)
        frames = {asset:frame.loc[frame.index>=history_start] for asset,frame in frames.items()}
        start = min(f.index.min() for f in frames.values()).tz_convert('America/New_York').strftime('%Y-%m-%d')
        day = origin.tz_convert('America/New_York').strftime('%Y-%m-%d')
        data = causal_data(frames, start, day)
        location = data.timestamps.get_indexer([origin])[0]
        if location<0 or location not in data.origins(require_target=False):
            raise ValueError('Missing minutes or insufficient causal warmup history')
        features = FeatureStore(data, self.config); output = {}
        for key,(model,state,calibration) in self.models.items():
            entry = self.lock['payload']['models'][key]
            if day<=entry['calibration_through']:
                raise ValueError('Prospective origin predates the frozen model availability')
            raw = raw_predictions(entry['model'], model, data, features, state, [location],
                                  entry['params'], entry['panel'], self.config, 'cpu')
            mean, quantiles = calibration.predict(raw)
            output[key] = {'raw':float(raw[0]), 'rv_mean':float(mean[0]),
                'quantiles':{str(k):float(v) for k,v in zip(calibration.levels,quantiles[0])},
                'source_result_sha256':entry['source_result_sha256']}
        return output


class ReplayStore(BarStore):
    """Keep every receipt rule; avoid pandas' expensive nested Timestamp dict path."""
    def frames(self, cutoff=None):
        limit = stamp(cutoff) if cutoff is not None else None
        frames = {}
        for asset, bars in self.bars.items():
            times = [t for t in bars if limit is None or t <= limit]
            if not times:
                raise ValueError('Received history is insufficient')
            frames[asset] = pd.DataFrame.from_records([bars[t] for t in times],
                index=pd.DatetimeIndex(times), columns=['open','high','low','close','volume']).sort_index()
        return frames


def observation_directory(lock):
    return ROOT / OBSERVATION / lock['signature']


def next_origin(now):
    return stamp(now).floor('min') + pd.Timedelta(minutes=1) - DELAY


def read_plan(config, full=True):
    base = _read_original(config, full=full)
    lock = read_json(ROOT / PLAN); p = lock['payload']
    if (stable_hash(p) != lock['signature'] or p['original_plan_signature'] != base['signature'] or
            p['original_plan_sha256'] != sha(ROOT / original.PLAN) or
            p['execution_lock_sha256'] != sha(ROOT / EXECUTION) or
            p['settings'] != read_json(ROOT / SETTINGS) or p['models'] != base['payload']['models'] or
            p['historical_prerequisites'] != base['payload']['historical_prerequisites'] or
            p['base_signature'] != base['payload']['base_signature'] or
            p['mode'] != MODE or p['prospective_qualified'] is not False):
        raise ValueError('Delayed replay identity, original models or prerequisites changed')
    for path, digest in p['source_files'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('Delayed replay source changed: ' + path)
    return lock


def freeze_plan(config):
    if (ROOT / PLAN).exists():
        return read_plan(config)
    base = _read_original(config)
    settings = read_json(ROOT / SETTINGS)
    expected = {**base['payload']['settings'], 'scope':MODE, 'publication_delay_minutes':15,
        'prospective_qualified':False, 'complete_session_policy':'original_grid_all_four_models_published_15_minutes_later'}
    if settings != expected:
        raise ValueError('Only the publication clock and replay qualification may differ')
    sources = ('finance/delayed.py', 'finance/delayed_validation.py', 'finance/delayed_service.py',
               SETTINGS, 'delivery/DELAYED_REPLAY.md')
    payload = {**base['payload'], 'created_utc':utc_now(), 'mode':MODE,
        'prospective_qualified':False, 'timely_forward_validation':False,
        'original_plan_signature':base['signature'], 'original_plan_sha256':sha(ROOT / original.PLAN),
        'execution_lock_sha256':sha(ROOT / EXECUTION), 'settings':settings,
        'source_files':{name:sha(ROOT / name) for name in sources},
        'no_replay_observations_before_freeze':True}
    lock = {'signature':stable_hash(payload), 'payload':payload}
    immutable_json(ROOT / PLAN, lock)
    return lock


def admission(config, lock=None, journal=None):
    lock = lock or read_plan(config)
    directory = observation_directory(lock)
    journal = journal or Journal(directory, lock['signature'])
    now = stamp(utc_now()); market_now = now - DELAY
    day = market_now.tz_convert('America/New_York').strftime('%Y-%m-%d')
    origins = session_origins(day)
    in_session = bool(len(origins) and origins[0]-pd.Timedelta(minutes=15) <= market_now <=
                      origins[-1]+pd.Timedelta(minutes=32))
    result = {'passed':False, 'mode':MODE, 'prospective_qualified':False,
        'status':'waiting_regular_session_delayed_replay', 'checked_utc':utc_now(),
        'plan_signature':lock['signature'], 'regular_session':in_session}
    if in_session:
        end = now.floor('min') - pd.Timedelta(minutes=16)
        try:
            store = BarStore()
            for event in Receiver(journal).fetch(end-pd.Timedelta(minutes=5), end, 'admission'):
                store.apply(event)
            recent = all(store.bars[a] and max(store.bars[a]) >= end-pd.Timedelta(minutes=1)
                         for a in store.bars)
            result.update(passed=bool(recent), http_status=200, recent_all_three_assets=bool(recent),
                status='free_delayed_sip_admitted' if recent else 'waiting_delayed_regular_session_bars')
        except DataUnavailable as error:
            result.update(http_status=error.http_status, status='free_delayed_sip_unavailable')
    result['last_event_hash'] = journal.last_hash
    # Keep each attempt and preserve the first successful admission permanently.
    attempt = directory / 'admissions' / (result['checked_utc'].replace(':','_') + '.json')
    immutable_json(attempt, result)
    path = directory / 'admission.json'
    if not path.exists() or not read_json(path)['passed']:
        write_json(path, result)
    write_json(directory / 'latest_admission.json', result)
    return result


def issue(journal, predictor, store, origin):
    origin = stamp(origin); deadline = origin + DELAY
    common = {'origin':origin.isoformat(), 'market_origin':origin.isoformat(), 'mode':MODE,
        'scheduled_publication_at':deadline.isoformat(), 'prospective_qualified':False,
        'eligible_signal':False, 'direction_or_profit_prediction':False,
        'received_at':store.last_received_at.isoformat() if store.last_received_at is not None else None,
        'data_cutoff_at':(origin-pd.Timedelta(minutes=1)).isoformat(), 'data_event_hash':store.last_event_hash,
        'target_window_start':origin.isoformat(), 'target_window_end':(origin+pd.Timedelta(minutes=30)).isoformat(),
        'unit':'sum_of_squared_log_returns_over_30_minutes'}
    at = stamp(utc_now())
    if at < origin or at >= deadline:
        return journal.append('prediction', {**common, 'status':'not_issued_deadline_passed'})
    try:
        forecasts = predictor.predict(store, origin)
        issued = stamp(utc_now())
        if store.last_received_at is None or not origin <= store.last_received_at <= issued < deadline:
            raise ValueError('Actual receipt or generation missed the delayed publication deadline')
        event = journal.append('prediction', {**common, 'status':'issued',
            'issued_at':issued.isoformat(), 'predictions':forecasts})
        journal.append('issuance_commit', {'prediction_hash':event['hash'], 'mode':MODE})
        return event
    except (ValueError, KeyError, IndexError) as error:
        return journal.append('prediction', {**common, 'status':'unavailable', 'reason':str(error)})


def verify(config, round_number):
    # Reuse the original complete numerical verifier in its own namespace.
    # Predictor, labels, metrics, thresholds and original module globals stay intact.
    namespace = {**original.__dict__, 'read_plan':read_plan,
        'observation_directory':observation_directory, 'complete_days':complete_days}
    def replay_proof(path, result):
        immutable_json(path, {**result, 'mode':MODE, 'prospective_qualified':False,
                            'timely_forward_validation':False})
    namespace['immutable_json'] = replay_proof
    function = FunctionType(original.verify_observation.__code__, namespace)
    result = function(config, round_number)
    return {**result, 'mode':MODE, 'prospective_qualified':False, 'timely_forward_validation':False}


@contextmanager
def resource_context():
    """Only the dedicated, driver-locked replay process uses this directory binding."""
    previous = original.observation_directory
    original.observation_directory = observation_directory
    try:
        yield
    finally:
        original.observation_directory = previous


def infer(config, as_of):
    from finance_v2.data import read_bars
    lock = read_plan(config)
    origin = stamp(as_of)
    # Use the same audited OHLCV files as the existing historical inference.
    manifest = read_json(ROOT / config['data_manifest'])
    store = ReplayStore()
    for asset, entry in manifest['assets'].items():
        path = ROOT / entry['csv']
        if sha(path) != entry['csv_sha256']:
            raise ValueError('Audited historical input changed: ' + asset)
        frame = read_bars(path)
        store.bars[asset] = frame.loc[(frame.index <= origin-pd.Timedelta(minutes=2)) &
            (frame.index >= origin.normalize()-pd.Timedelta(days=80))].to_dict('index')
    output = Predictor(config, lock).predict(store, origin)
    return {'mode':MODE, 'status':'historical_smoke_not_an_observation',
        'market_origin':origin.isoformat(), 'actual_generated_at':utc_now(),
        'scheduled_publication_at':(origin+DELAY).isoformat(), 'prospective_qualified':False,
        'eligible_signal':False, 'target_window_start':origin.isoformat(),
        'target_window_end':(origin+pd.Timedelta(minutes=30)).isoformat(), 'predictions':output}
