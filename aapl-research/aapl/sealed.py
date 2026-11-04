"""Sealed acquisition and audited, gate-controlled access for the implementation plan."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path

import pandas as pd

from .common import ROOT, digest, identity, read_json, utc_now, write_json
from .exposure import audit

SEALED = ROOT / 'holdout_sealed'
MANIFEST = ROOT / 'holdout_manifest.json'


def event(kind, details):
    import fcntl
    directory = ROOT / 'outputs/holdout_access'
    directory.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'outputs/holdout_access.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = sorted(directory.glob('*.json'))
        record = {'kind': kind, 'recorded_at_utc': utc_now(), 'details': details,
                  'previous_sha256': digest(previous[-1]) if previous else None}
        write_json(directory / f'{len(previous) + 1:06d}.json', record)


def assert_access(authorization):
    """No time-based or filename-only bypass: the actual development decision is required."""
    if authorization is None:
        event('denied', {'reason': 'explicit unseal authorization missing'})
        raise PermissionError('Holdout is sealed; explicit gate authorization is required')
    from .runner import require_lock
    protocol = require_lock()
    decision_path = ROOT / 'outputs/development_decision.json'
    decision = read_json(decision_path)
    manifest = read_json(MANIFEST)
    expected = identity({'protocol_sha256': digest(ROOT / 'protocol_locked.json'),
                         'decision_sha256': digest(decision_path),
                         'holdout_csv_sha256': manifest['csv_sha256']})
    if not decision['enter_holdout'] or authorization != expected:
        event('denied', {'reason': 'development gate or authorization mismatch'})
        raise PermissionError('Holdout entry gate has not passed')
    if decision['protocol_sha256'] != digest(ROOT / 'protocol_locked.json'):
        raise PermissionError('Development decision refers to a different protocol')
    return protocol, manifest


def acquire():
    if MANIFEST.exists():
        row = read_json(MANIFEST)
        if digest(SEALED / row['csv']) != row['csv_sha256']:
            raise ValueError('Existing sealed file changed')
        print({'sealed': True, 'reused_verified_manifest': True}, flush=True)
        return
    exposure = audit()
    source = Path('/home/ayaka/xPatch/data_acquisition/download_alpaca_sip.py')
    # Reuse only the read-only authenticated request helpers; never run legacy main().
    spec = importlib.util.spec_from_file_location('readonly_alpaca_request_helpers', source)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    key, secret = helper.read_credentials()
    if not key or not secret:
        raise RuntimeError('Local Alpaca credentials unavailable; no secret values logged')
    SEALED.mkdir(exist_ok=False)
    (SEALED / 'raw').mkdir()
    pages, bars = [], []
    ranges = [('2026-07-01', '2026-07-30'), ('2026-07-31', '2026-08-29'),
              ('2026-08-30', '2026-09-28'), ('2026-09-29', '2026-09-30')]
    receipt = {'symbol': 'AAPL', 'feed': 'sip', 'adjustment': 'raw', 'timeframe': '1Min',
               'endpoint': helper.ENDPOINT, 'ranges': ranges, 'pages': pages,
               'purpose': 'sealed_final_test_acquisition', 'download_complete': False,
               'created_utc': utc_now(), 'request_helper_read_only': str(source),
               'request_helper_sha256': digest(source)}
    write_json(SEALED / 'receipt.json', receipt)
    try:
        for ri, (start, end) in enumerate(ranges):
            low = pd.Timestamp(start, tz='America/New_York').tz_convert('UTC')
            high = (pd.Timestamp(end) + pd.Timedelta(days=1)).tz_localize('America/New_York').tz_convert('UTC')
            if high.to_pydatetime() > datetime.now(timezone.utc) - timedelta(minutes=15):
                raise ValueError('Requested end is too recent')
            params = {'symbols': 'AAPL', 'timeframe': '1Min', 'feed': 'sip', 'adjustment': 'raw',
                      'start': low.isoformat(), 'end': high.isoformat(), 'limit': 10000, 'sort': 'asc'}
            tokens = set()
            page = 0
            while True:
                body, headers = helper.request_body(params, key, secret)
                path = SEALED / 'raw' / f'range_{ri:03d}_page_{page:04d}.json'
                path.write_bytes(body)
                payload = read_json(path)
                group = payload.get('bars', {}).get('AAPL', []) or []
                token = payload.get('next_page_token')
                bars.extend(b for b in group if low <= pd.Timestamp(b['t']) < high)
                pages.append({'file': str(path.relative_to(SEALED)), 'sha256': digest(path),
                              'bytes': len(body), 'rows_returned': len(group),
                              'range_start': start, 'range_end': end, 'page': page,
                              'received_utc': utc_now(), 'headers': headers, 'has_next_page': bool(token)})
                write_json(SEALED / 'receipt.json', receipt)
                if not token:
                    break
                if token in tokens or page >= 500:
                    raise ValueError('Invalid/cyclic pagination')
                tokens.add(token)
                params['page_token'] = token
                page += 1
        bars.sort(key=lambda b: pd.Timestamp(b['t']))
        csvpath = SEALED / 'AAPL_sip_1min_raw.csv'
        with csvpath.open('x', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap'])
            for b in bars:
                writer.writerow([pd.Timestamp(b['t']).tz_convert('America/New_York').isoformat(),
                                 b['o'], b['h'], b['l'], b['c'], b['v'], b['n'], b['vw']])
        receipt.update(download_complete=True, rows=len(bars), csv=csvpath.name, csv_sha256=digest(csvpath))
        write_json(SEALED / 'receipt.json', receipt)
        from .data import read_source, verify_receipt, build_market
        frame = read_source(csvpath, _sealed_admission=True)
        verified = verify_receipt(frame, SEALED, digest(csvpath))
        market = build_market(frame, '2026-07-01', '2026-09-30')
        manifest = {'sealed': True, 'created_at_utc': utc_now(), 'start': '2026-07-01', 'end': '2026-09-30',
                    'csv': csvpath.name, 'csv_sha256': digest(csvpath), 'rows': len(frame),
                    'receipt_sha256': digest(SEALED / 'receipt.json'), 'receipt_verification': verified,
                    'regular_rows': len(market.bars), 'calendar_sessions': market.audit['sessions'],
                    'eligible_origins': len(market.origins), 'effective_sessions': len(set(market.dates[market.origins])),
                    'exposure_audit_sha256': digest(ROOT / 'outputs/holdout_exposure_audit.json'),
                    'exposure_audit_time': exposure['checked_at_utc'], 'predictive_scores_computed': False,
                    'access_policy': 'Admission inspection only; model access requires locked development decision and explicit unseal token'}
        write_json(MANIFEST, manifest)
        for path in SEALED.rglob('*'):
            if path.is_file():
                path.chmod(0o444)
        for path in sorted((p for p in SEALED.rglob('*') if p.is_dir()), reverse=True):
            path.chmod(0o555)
        SEALED.chmod(0o555)
        event('acquired_and_sealed', {'manifest_sha256': digest(MANIFEST), 'model_access': False})
        print({'sealed': True, 'calendar_sessions': manifest['calendar_sessions'],
               'effective_sessions': manifest['effective_sessions'], 'eligible_origins': manifest['eligible_origins']}, flush=True)
    except Exception as exc:
        write_json(ROOT / 'outputs/holdout_acquisition_failure.json', {'error_type': type(exc).__name__,
                   'failed_at_utc': utc_now(), 'partial_files_preserved': True})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--acquire', action='store_true', required=True)
    parser.parse_args()
    acquire()
