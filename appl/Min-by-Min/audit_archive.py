"""Stream a supplied archive; report interpolation without extracting or training."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from data_quality import sha256


def audit(path, output):
    totals = dict(rows=0, weekend_rows=0, fractional_volume_rows=0, linear_triplets=0, triplets=0)
    start, end, tail = None, None, None
    first_day = None
    with zipfile.ZipFile(path) as archive:
        members = [m for m in archive.infolist() if m.filename.lower().endswith('.csv')]
        if len(members) != 1:
            raise ValueError('Expected one CSV; choose/inspect archive explicitly')
        member = members[0]
        with archive.open(member) as stream:
            for chunk in pd.read_csv(stream, chunksize=200000):
                dates = pd.to_datetime(chunk.iloc[:,0])
                numbers = chunk.iloc[:,1:].to_numpy(dtype=float)
                if start is None:
                    start = dates.iloc[0].isoformat()
                    first_day = numbers[:1441].copy()
                end = dates.iloc[-1].isoformat()
                totals['rows'] += len(chunk)
                totals['weekend_rows'] += int((dates.dt.dayofweek >= 5).sum())
                volume = chunk['Volume'].to_numpy(dtype=float)
                totals['fractional_volume_rows'] += int((np.abs(volume % 1) > 1e-5).sum())
                all_values = np.concatenate([tail,numbers]) if tail is not None else numbers
                tolerance = np.full(all_values.shape[1],1e-9)
                tolerance[-1] = 1e-4
                linear = np.all(np.abs(np.diff(all_values,n=2,axis=0)) <= tolerance,axis=1)
                totals['linear_triplets'] += int(linear.sum())
                totals['triplets'] += len(linear)
                tail = all_values[-2:].copy()
    interpolated = np.linspace(first_day[0], first_day[-1],len(first_day))
    residual = np.max(np.abs(first_day-interpolated),axis=0).tolist()
    fraction = totals['linear_triplets']/totals['triplets']
    rejected = fraction > .95 and totals['weekend_rows'] > 0 and totals['fractional_volume_rows'] > totals['rows']/2
    result = dict(archive=str(Path(path).resolve()), archive_sha256=sha256(path),
                  member=member.filename, uncompressed_bytes=member.file_size,
                  scanned_utc=datetime.now(timezone.utc).isoformat(),start=start,end=end,
                  **totals, linear_second_difference_fraction=fraction,
                  first_1441_rows_linear_interpolation_max_error=residual,
                  decision='rejected_synthetic_interpolation' if rejected else 'requires_provenance_review',
                  eligible_for_intraday_training=False,
                  reason='Nearly all OHLCV triplets are linear; weekend/off-session timestamps and fractional volumes indicate daily-to-minute interpolation. Future daily endpoints contaminate intermediate timestamps.',
                  source_url='https://www.kaggle.com/datasets/deltatrup/aapl-1-minute-historical-stock-data-2006-2024',
                  source_url_status='matching metadata only; local download URL is absent')
    output.mkdir(parents=True,exist_ok=True)
    (output/'archive_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'ARCHIVE_AUDIT.md').write_text(
        '# Supplied archive audit\n\nDecision: `'+result['decision']+'`\n\n'+result['reason']+
        '\n\nRows: '+str(totals['rows'])+'; range: '+start+' to '+end+
        '\n\nLinear triplet fraction: '+str(fraction)+'; weekend rows: '+str(totals['weekend_rows'])+
        '\n\nThe original ZIP is preserved. Filtering to market hours cannot remove future endpoint interpolation. Do not use it for minute forecasting. Full counts, SHA-256 and tolerances are in archive_audit.json.\n')
    print(json.dumps(result,indent=2))
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'data'/'audits')
    args=parser.parse_args()
    audit(args.archive,args.output)
