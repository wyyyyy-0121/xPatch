"""Audit legacy data-access dates without reading any cumulative model scores."""
from pathlib import Path

import pandas as pd

from .common import ROOT, digest, read_json, utc_now, write_json


LEGACY = Path('/home/ayaka/xPatch')


def overlaps(start, end):
    return str(start)[:10] <= '2026-09-30' and str(end)[:10] >= '2026-07-01'


def audit():
    receipts = []
    for p in sorted((LEGACY / 'data_acquisition/runs').glob('*/receipt.json')):
        row = read_json(p)
        ranges = row.get('ranges', [])
        start = row.get('start') or (min(r[0] for r in ranges) if ranges else None)
        end = row.get('end') or (max(r[1] for r in ranges) if ranges else None)
        receipts.append({'path': str(p), 'sha256': digest(p), 'start': start, 'end': end,
                         'overlaps_holdout': overlaps(start, end) if start and end else None})
    observation = []
    for directory in ('delayed_replay', 'prospective'):
        base = LEGACY / 'artifacts/rv30' / directory
        for folder in sorted(base.glob('*')):
            if not folder.is_dir():
                continue
            status = folder / 'status.json'
            state = read_json(status) if status.exists() else {}
            # Allowlist metadata only; quality and score fields are never surfaced or used.
            item = {'directory': str(folder), 'status_present': status.exists(),
                    'status': {k: state.get(k) for k in ('stage', 'updated_utc', 'mode')},
                    'bars_event_count': 0, 'bar_access_ranges': [], 'overlaps_holdout': False}
            for event_path in sorted((folder / 'events').glob('*.json')):
                event = read_json(event_path)
                if event.get('kind') != 'bars':
                    continue
                payload = event['payload']
                start, end = payload['start'], payload['end']
                item['bars_event_count'] += 1
                item['bar_access_ranges'].append({'start': start, 'end': end,
                    'purpose': payload.get('purpose'), 'event_sha256': digest(event_path)})
                item['overlaps_holdout'] |= overlaps(start, end)
            observation.append(item)
    passed = bool(receipts) and all(r['overlaps_holdout'] is False for r in receipts)
    passed = passed and all(not r['overlaps_holdout'] for r in observation)
    result = {'passed': passed, 'checked_at_utc': utc_now(), 'holdout': ['2026-07-01', '2026-09-30'],
              'legacy_acquisition_receipts': receipts, 'observation_metadata': observation,
              'cumulative_model_scores_read': False,
              'scope': 'Listed legacy acquisition receipts and observer access journals at this timestamp; no claim about unknown external use'}
    write_json(ROOT / 'outputs/holdout_exposure_audit.json', result)
    if not passed:
        raise RuntimeError('Legacy holdout overlap or incomplete receipt bounds; preserve evidence and stop acquisition')
    return result


if __name__ == '__main__':
    record = audit()
    print({'passed': record['passed'], 'receipts': len(record['legacy_acquisition_receipts']),
           'observation_directories': len(record['observation_metadata'])}, flush=True)
