"""Run the free delayed SIP replay locally; the agent only checks it hourly."""
import argparse
import json
import os
from pathlib import Path
import signal

from .io import ROOT, read_json, sha, write_json


STOP = False


def save_status(lock, stage, quality, error=None):
    from .delayed import MODE, observation_directory
    from .observation import utc_now
    identity = {'pid':os.getpid(), 'kernel_start_ticks':Path('/proc/self/stat').read_text().split()[21]}
    value = {'stage':stage, 'mode':MODE, 'updated_utc':utc_now(), 'plan_signature':lock['signature'],
        **identity, 'cpu_ids':sorted(os.sched_getaffinity(0)), 'last_error':error,
        'delayed_replay_complete_sessions':quality['complete_sessions'],
        'prospective_complete_sessions':0, 'prospective_qualified':False, 'quality':quality}
    directory = observation_directory(lock)
    write_json(directory / 'status.json', value)
    for name in ('status', 'run'):
        path = ROOT / f'delivery/{name}.json'
        previous = read_json(path)
        previous.update({k:v for k,v in value.items() if k != 'quality'})
        previous.update(goal_status='active', final_acceptance=False, goal_can_complete=False,
            monitoring='active_hourly_minimal_state_checks', monitor_interval_minutes=60,
            delayed_replay_plan='delivery/delayed_sip_lock.json',
            delayed_replay_directory=str(directory.relative_to(ROOT)))
        if name == 'run':
            previous.update(state=stage, command='python -m finance.delayed_service run',
                log=str((directory/'service.log').relative_to(ROOT)))
        write_json(path, previous)


def sync_charges(lock):
    """Mirror immutable replay receipts into the original cumulative budget."""
    from . import forward_runtime as runtime
    rows, _ = runtime.resource_records(lock)
    path = ROOT / 'artifacts/rv30' / lock['payload']['base_signature'] / 'interrupted_resource_charges.json'
    charges = read_json(path)
    for row in rows:
        start = ROOT / row['path']; end = start.with_name('end.json')
        charges['delayed_replay_resource:' + row['path']] = {
            'allocated_cpu_seconds':row['charged_cpu_seconds'], 'gpu_wall_seconds':0.,
            'start':row['path'], 'start_sha256':sha(start),
            'end':str(end.relative_to(ROOT)) if row['end'] else None,
            'end_sha256':sha(end) if row['end'] else None,
            'accounting':'actual active allocation after clean exit; full reservation while unresolved'}
    write_json(path, charges)


def mirrored_cpu(lock):
    from . import forward_runtime as runtime
    rows, _ = runtime.resource_records(lock)
    path = ROOT / 'artifacts/rv30' / lock['payload']['base_signature'] / 'interrupted_resource_charges.json'
    charges = read_json(path); total = 0.
    for row in rows:
        entry = charges.get('delayed_replay_resource:' + row['path'])
        if entry is None:
            continue
        end_path = (ROOT / row['path']).with_name('end.json')
        if (entry['start_sha256'] != sha(ROOT / row['path']) or
                entry['allocated_cpu_seconds'] != row['charged_cpu_seconds'] or
                entry['end_sha256'] != (sha(end_path) if row['end'] else None)):
            raise ValueError('Replay resource mirror differs from its original immutable receipts')
        total += entry['allocated_cpu_seconds']
    return total


def execute(config, lock, action, function):
    from . import forward_runtime as runtime
    from .delayed import resource_context
    with resource_context():
        runtime.recover_interrupted(lock)
        sync_charges(lock)
        historical = runtime.historical_usage
        # Meter itself adds the replay receipts. Subtract only the validated
        # mirrors here, so the original budget counts every action exactly once.
        def excluding_mirrors(signature):
            cpu, gpu = historical(signature)
            return cpu - mirrored_cpu(lock), gpu
        runtime.historical_usage = excluding_mirrors
        try:
            try:
                with runtime.metered(config, lock, action):
                    sync_charges(lock)  # Reserve before performing work.
                    return function(runtime.ACTIVE)
            finally:
                sync_charges(lock)
        finally:
            runtime.historical_usage = historical


def observe(config, lock, meter):
    import pandas as pd
    from .delayed import (DELAY, Predictor, ReplayStore, admission, complete_days, issue, next_origin,
                          observation_directory, session_origins)
    from .observation import BarStore, DataUnavailable, Journal, Receiver, stamp, utc_now
    settings = lock['payload']['settings']; directory = observation_directory(lock)
    journal = Journal(directory, lock['signature']); store = ReplayStore()
    for event in journal.events:
        if event['kind'] == 'bars':
            store.apply(event)
    seen = {e['payload']['origin'] for e in journal.events if e['kind'] == 'prediction'}
    quality = complete_days(journal.events, settings)
    previous_wall = stamp(utc_now()); warmed_day = quality_day = None
    predictor = receiver = None
    retry_after = previous_wall
    while not STOP and not quality['ready_for_final_evaluation']:
        now = stamp(utc_now())
        if now < previous_wall:
            raise RuntimeError('Wall clock moved backwards; delayed replay stopped')
        previous_wall = now
        market_now = now-DELAY
        day = market_now.tz_convert('America/New_York').strftime('%Y-%m-%d')
        origins = session_origins(day)
        in_session = bool(len(origins) and origins[0]-pd.Timedelta(minutes=15) <= market_now <=
                          origins[-1]+pd.Timedelta(minutes=32))
        if not in_session:
            save_status(lock, 'waiting_regular_session_delayed_replay', quality)
            meter.sleep(30)
            continue
        if warmed_day != day:
            if now < retry_after:
                meter.sleep(min(30, (retry_after-now).total_seconds()))
                continue
            admitted = admission(config, lock, journal)
            if not admitted['passed']:
                save_status(lock, admitted['status'], quality, admitted.get('http_status'))
                retry_after = now+pd.Timedelta(hours=1)
                meter.sleep(30)
                continue
            receiver = Receiver(journal)
            end = stamp(utc_now()).floor('min')-pd.Timedelta(minutes=16)
            start = (min(max(bars) for bars in store.bars.values())-pd.Timedelta(minutes=3)
                     if warmed_day is not None and all(store.bars.values()) else end-pd.Timedelta(days=80))
            try:
                for event in receiver.fetch(start, end, 'warmup'):
                    store.apply(event)
            except DataUnavailable as error:
                save_status(lock, 'delayed_replay_data_unavailable', quality, error.http_status)
                retry_after = now+pd.Timedelta(hours=1)
                continue
            predictor = predictor or Predictor(config, lock)
            warmed_day = day
            save_status(lock, 'observing_delayed_shadow_replay', quality)
            continue
        origin = next_origin(now); deadline = origin+DELAY
        wake = deadline-pd.Timedelta(seconds=settings['forecast_lead_seconds'])
        if now < wake:
            meter.sleep(min((wake-now).total_seconds(), 30))
            continue
        end = origin-pd.Timedelta(minutes=2)
        try:
            for event in receiver.fetch(end-pd.Timedelta(minutes=3), end, 'live'):
                store.apply(event)
            if origin in origins and origin.isoformat() not in seen:
                issue(journal, predictor, store, origin)
                seen.add(origin.isoformat())
            # All target bars are available about 16 minutes after the close.
            if market_now >= origins[-1]+pd.Timedelta(minutes=31) and quality_day != day:
                quality = complete_days(journal.events, settings)
                quality_day = day
            save_status(lock, 'observing_delayed_shadow_replay', quality)
        except DataUnavailable as error:
            save_status(lock, 'delayed_replay_data_unavailable', quality, error.http_status)
        delay = (deadline+pd.Timedelta(seconds=1)-stamp(utc_now())).total_seconds()
        if delay > 0:
            meter.sleep(min(delay, 30))
    return quality


def run(config, lock):
    from .delayed import MODE, observation_directory, verify
    from . import forward_runtime as runtime
    from .observation import Journal
    quality = execute(config, lock, 'observe', lambda meter:observe(config, lock, meter))
    if STOP:
        save_status(lock, 'delayed_replay_service_stopped', quality)
        return {'passed':False, 'status':'service_stopped'}
    for number in (1, 2):
        save_status(lock, f'verifying_delayed_replay_round_{number}', quality)
        result = execute(config, lock, f'verify_{number}', lambda meter:verify(config, number))
        if not result['passed']:
            save_status(lock, 'delayed_replay_verification_failed', quality, result.get('error'))
            return result
    proofs = {str(n):{'path':str((observation_directory(lock)/f'verify_{n}.json').relative_to(ROOT)),
        'sha256':sha(observation_directory(lock)/f'verify_{n}.json')} for n in (1,2)}
    resources = execute(config, lock, 'accept', lambda meter:runtime.resource_evidence(
        config, lock, Journal(observation_directory(lock), lock['signature']), proofs))
    result = {'passed':True, 'mode':MODE, 'prospective_qualified':False,
        'timely_forward_validation':False, 'goal_can_complete':False,
        'stage':'delayed_shadow_evaluated_pending_timely_forward',
        'qualified_replay_candidates':result['qualified_candidates'],
        'verifications':proofs, 'resources':resources}
    write_json(ROOT/'delivery/delayed_replay_results.json', result)
    save_status(lock, result['stage'], quality)
    return result


def main():
    from .runtime import configure, exclusive_run
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze','admit','run','status','verify','infer'))
    parser.add_argument('--round', type=int, choices=(1,2))
    parser.add_argument('--as-of')
    args = parser.parse_args()
    configure(4, cpu_ids=sorted(os.sched_getaffinity(0))[:4])
    from . import delayed
    config = read_json(ROOT/'configs/rv30.json')
    def stop(signum, frame):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    if args.action == 'status':
        lock = delayed.read_plan(config, full=False)
        path = delayed.observation_directory(lock)/'status.json'
        result = read_json(path) if path.exists() else {'status':'not_started', 'mode':delayed.MODE}
    else:
        with exclusive_run(ROOT/'artifacts/rv30/driver.lock'):
            if args.action == 'freeze':
                result = delayed.freeze_plan(config)
            else:
                lock = delayed.read_plan(config)
                if args.action == 'admit':
                    result = delayed.admission(config, lock)
                elif args.action == 'verify':
                    if args.round is None:
                        parser.error('verify requires --round')
                    result = execute(config, lock, f'verify_{args.round}', lambda meter:delayed.verify(config, args.round))
                elif args.action == 'infer':
                    if args.as_of is None:
                        parser.error('infer requires --as-of')
                    result = delayed.infer(config, args.as_of)
                else:
                    try:
                        result = run(config, lock)
                    except Exception as error:
                        path = delayed.observation_directory(lock)/'status.json'
                        quality = (read_json(path)['quality'] if path.exists() else
                                   delayed.complete_days([], lock['payload']['settings']))
                        save_status(lock, 'delayed_replay_service_failed', quality,
                                    f'{type(error).__name__}: {error}')
                        raise
    print(json.dumps(result, ensure_ascii=False))
    return 2 if result.get('passed') is False else 0


if __name__ == '__main__':
    raise SystemExit(main())
