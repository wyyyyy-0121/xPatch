"""Quality checks for a 15-minute delayed replay of the original market clock."""
from functools import lru_cache

import pandas as pd

from .observation import BarStore, session_origins as _session_origins, stamp


MODE = 'delayed_shadow_replay'
DELAY = pd.Timedelta(minutes=15)


@lru_cache(maxsize=256)
def session_origins(day):
    return _session_origins(day)


def complete_days(events, settings):
    """Select whole replay sessions by timing/integrity, without scoring models."""
    store, predictions, commits = BarStore(), {}, {}
    for event in events:
        p = event['payload']
        if event['kind'] == 'bars':
            if stamp(p['end']) > stamp(p['received_at']) - DELAY:
                raise ValueError('Free SIP request crosses the 15-minute availability boundary')
            store.apply(event)
        elif event['kind'] == 'prediction':
            origin = stamp(p['origin']); key = origin.isoformat()
            day = origin.tz_convert('America/New_York').strftime('%Y-%m-%d')
            if key in predictions:
                raise ValueError('Duplicate delayed replay origin; replacements are forbidden')
            if origin not in session_origins(day):
                raise ValueError('Delayed replay changed the original market origin grid')
            prefix_ok = (p.get('data_event_hash') == store.last_event_hash and
                store.last_received_at is not None and p.get('received_at') is not None and
                stamp(p['received_at']) == store.last_received_at)
            predictions[key] = {**p, '_hash':event['hash'], '_recorded':event['recorded_at'],
                '_prefix_ok':prefix_ok}
        elif event['kind'] == 'issuance_commit':
            reference = p['prediction_hash']
            if reference in commits:
                raise ValueError('Duplicate delayed prediction commit')
            commits[reference] = stamp(event['recorded_at'])
    days = sorted({stamp(p['origin']).tz_convert('America/New_York').strftime('%Y-%m-%d')
                   for p in predictions.values()})
    models = set(settings['candidate_keys'] + settings['reference_keys'])
    quality, qualified = {}, []
    for day in days:
        expected = session_origins(day)
        good = missing = late = missing_targets = 0
        for origin in expected:
            p = predictions.get(origin.isoformat())
            if p is None or p['status'] != 'issued' or set(p.get('predictions', {})) != models:
                missing += 1
                continue
            committed = commits.get(p['_hash'])
            deadline = origin + DELAY
            valid = (p['_prefix_ok'] and committed is not None and
                p.get('mode') == MODE and p.get('prospective_qualified') is False and
                p.get('eligible_signal') is False and
                origin <= stamp(p['received_at']) <= stamp(p['issued_at']) <=
                stamp(p['_recorded']) <= committed < deadline and
                stamp(p['market_origin']) == origin and
                stamp(p['scheduled_publication_at']) == deadline and
                stamp(p['data_cutoff_at']) == origin - pd.Timedelta(minutes=1) and
                stamp(p['target_window_start']) == origin and
                stamp(p['target_window_end']) == origin + pd.Timedelta(minutes=30))
            if not valid:
                late += 1
                continue
            try:
                store.target(origin)
            except ValueError:
                missing_targets += 1
                continue
            good += 1
        complete = bool(len(expected) and good == len(expected))
        quality[day] = {'complete':complete, 'expected_origins':len(expected),
            'complete_origins':good, 'missing_or_unavailable_origins':missing,
            'late_or_invalid_origins':late, 'incomplete_targets':missing_targets}
        if complete:
            qualified.append(day)
    return {'mode':MODE, 'prospective_qualified':False, 'timely_prospective_complete_sessions':0,
        'days':quality, 'complete_dates':qualified, 'complete_sessions':len(qualified),
        'minimum_complete_sessions':settings['minimum_complete_sessions'],
        'ready_for_final_evaluation':len(qualified) >= settings['minimum_complete_sessions'],
        'cumulative_predictive_scores_disclosed':False}
