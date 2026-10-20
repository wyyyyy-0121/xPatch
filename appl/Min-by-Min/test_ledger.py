"""Project-wide, locked test exposure ledger; output folders cannot reset it."""
import fcntl
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd


def timestamp(value):
    ts = pd.Timestamp(value)
    return ts.tz_localize('UTC') if ts.tz is None else ts.tz_convert('UTC')


def migrate_history(root):
    entries = []
    for path in Path(root).rglob('report.json'):
        try:
            report = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        for row in report.get('final_test', {}).get('results', []):
            if row.get('test_start') and row.get('test_end'):
                # Older reports stored the last origin rather than label end.
                # Conservatively reserve through the source end as well.
                source_end = report.get('data', {}).get('end', row['test_end'])
                end = max((row['test_end'], source_end), key=timestamp)
                entries.append(dict(symbol=report.get('symbol','AAPL'), start=row['test_start'],
                                    end=end, output=str(path.resolve()),
                                    status='historically_exposed_exploratory_regression'))
    return entries


def claim_period(path, symbol, start, end, output, frozen_sha256):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        entries = json.loads(path.read_text()) if path.exists() else []
        known = {(e['symbol'], e['start'], e['end'], e['output']) for e in entries}
        entries += [e for e in migrate_history(path.parent)
                    if (e['symbol'],e['start'],e['end'],e['output']) not in known]
        path.write_text(json.dumps(entries, indent=2)+'\n')
        for entry in entries:
            if entry['symbol'] == symbol and timestamp(start) <= timestamp(entry['end']) and timestamp(end) >= timestamp(entry['start']):
                raise ValueError('Final test period overlaps prior exposure in shared ledger: '+entry['output'])
        entries.append(dict(symbol=symbol, start=start, end=end, output=output,
                            frozen_sha256=frozen_sha256, status='claimed_before_evaluation',
                            claimed_utc=datetime.now(timezone.utc).isoformat()))
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(entries, indent=2)+'\n')
        temporary.replace(path)
