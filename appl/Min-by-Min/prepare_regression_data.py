"""Recreate the explicitly stale, unverified regression cache from a pinned mirror."""
import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from data_quality import aggregate_bars, load_ohlcv, sha256

COMMIT='08ba60b568c300a635da1b6291a0f54eb9ef3f13'
URL=f'https://raw.githubusercontent.com/Hetal-Mehta/AAPL-Dataset/{COMMIT}/AAPL.csv'
RAW_SHA='9cfa57d3b9a2b0776d6275cb1b86f0ae08048ad2e61958c1dca9913a0cd03576'
CSV_SHA='1600769c640b352de6bd4b5cf40b6836abd5fc303406d3362c204f7f953703a9'


def prepare(raw=None, output=None):
    data=Path(output or Path(__file__).parent/'data')
    data.mkdir(parents=True,exist_ok=True)
    if raw:
        body=Path(raw).read_bytes()
    else:
        with urlopen(URL,timeout=120) as stream:
            body=stream.read()
    if hashlib.sha256(body).hexdigest()!=RAW_SHA:
        raise ValueError('Pinned mirror raw hash mismatch (possibly truncated download)')
    frame=pd.read_csv(BytesIO(body))
    frame=frame.rename(columns={c:c.lower() for c in frame.columns}).rename(columns={'date':'timestamp'})
    canonical=frame[['timestamp','open','high','low','close','volume']].to_csv(index=False).encode()
    if hashlib.sha256(canonical).hexdigest()!=CSV_SHA:
        raise ValueError('Canonical CSV hash mismatch')
    target=data/'AAPL_1d.csv'
    if target.exists() and sha256(target)!=CSV_SHA:
        raise FileExistsError('Refusing to overwrite a different daily cache: '+str(target))
    target.write_bytes(canonical)
    manifest=dict(symbol='AAPL',interval='1d',source='Legacy Kaggle mirror; upstream quotes/adjustment not independently verified',
                  provider='Hetal-Mehta/AAPL-Dataset',provider_url=URL,mirror_commit=COMMIT,
                  downloaded_utc=None if raw else datetime.now(timezone.utc).isoformat(),
                  prepared_utc=datetime.now(timezone.utc).isoformat(),
                  timezone='daily date, timezone not applicable',source_verified=False,
                  start='1980-12-12',end='2022-06-17',sha256=CSV_SHA,raw_sha256=RAW_SHA,
                  adjustment='Legacy mirror OHLC, Adj Close excluded; corporate action semantics unverified',
                  intended_use='stale unverified engineering regression only, not current AAPL forecasts')
    target.with_suffix('.csv.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    bars,_=load_ohlcv(target)
    monthly=aggregate_bars(bars,'1mo')
    monthly_path=data/'AAPL_1mo.csv'
    monthly.rename_axis('timestamp').to_csv(monthly_path)
    monthly_manifest={**manifest,'interval':'1mo','sha256':sha256(monthly_path),
                      'derived_from_sha256':CSV_SHA,'derivation':'XNYS complete calendar months only',
                      'start':monthly.index[0].isoformat(),'end':monthly.index[-1].isoformat()}
    monthly_path.with_suffix('.csv.manifest.json').write_text(json.dumps(monthly_manifest,indent=2)+'\n')
    inventory=dict(generated_utc=datetime.now(timezone.utc).isoformat(),datasets=[])
    for path in sorted(data.glob('AAPL_*.csv')):
        _,meta=load_ohlcv(path)
        inventory['datasets'].append(dict(path=path.name,sha256=sha256(path),metadata=meta))
    (data/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps(dict(status='stale_unverified_regression_cache_prepared',daily_rows=len(frame),monthly_rows=len(monthly)),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,help='Optional already-downloaded pinned mirror file')
    args=parser.parse_args()
    prepare(args.raw)
