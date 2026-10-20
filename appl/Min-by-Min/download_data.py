"""Cache all AAPL daily/monthly history available from the public Yahoo chart API."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
from data_quality import load_ohlcv, sha256


def download(symbol, interval, output, refresh=False):
    if symbol != 'AAPL':
        raise ValueError('This experiment suite is restricted to AAPL')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not refresh:
        _, meta = load_ohlcv(output)
        print(json.dumps({'status':'cached','csv':str(output),'metadata':meta},indent=2))
        return meta
    now = datetime.now(timezone.utc)
    url = 'https://query1.finance.yahoo.com/v8/finance/chart/'+symbol+'?'+urlencode(
        dict(period1=0, period2=int(now.timestamp()), interval=interval, events='history'))
    body = None
    for attempt in range(3):
        try:
            with urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=30) as response:
                body = response.read()
            break
        except Exception as exc:
            if attempt == 2:
                failure = dict(status='download_failed',url=url,attempted_utc=now.isoformat(),error=str(exc),
                               existing_cache_preserved=output.exists())
                output.with_suffix('.download_failure.json').write_text(json.dumps(failure,indent=2)+'\n')
                raise RuntimeError('Download failed; cache preserved. '+str(exc)) from exc
            time.sleep(2**attempt)
    result = json.loads(body)['chart']['result'][0]
    quote = result['indicators']['quote'][0]
    stamps = pd.to_datetime(result['timestamp'],unit='s',utc=True).tz_convert(result['meta']['exchangeTimezoneName'])
    frame = pd.DataFrame({c:quote[c] for c in ('open','high','low','close','volume')},index=stamps)
    missing = int(frame.isna().any(axis=1).sum())
    frame = frame.dropna()
    # Use session dates/month starts for long-term bars, consistently with aggregation.
    frame.index = frame.index.tz_localize(None).normalize()
    if interval == '1mo':
        frame.index = frame.index.to_period('M').to_timestamp()
        # The API returns an unfinished current month; exclude it.
        frame = frame.loc[frame.index < pd.Timestamp(now.date()).replace(day=1)]
    elif len(frame) and frame.index[-1].date() == now.date():
        frame = frame.iloc[:-1]  # conservative: never cache a possibly unfinished daily bar
    if frame.empty:
        raise ValueError('Provider returned no complete bars')
    frame.index.name = 'timestamp'
    temporary = output.with_suffix('.download.csv')
    frame.to_csv(temporary)
    load_ohlcv(temporary)  # validate before replacing the persistent cache
    temporary.replace(output)
    receipt_path = output.with_suffix('.receipt.json')
    receipt = dict(method='provider_download',provider='Yahoo Finance',url=url,
                   downloaded_utc=now.isoformat(),csv_sha256=sha256(output),
                   response_sha256=hashlib.sha256(body).hexdigest(),null_rows_dropped=missing)
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    manifest = dict(symbol=symbol, interval=interval,provider='Yahoo Finance',source='Yahoo Finance chart API',
                    provider_url=url,downloaded_utc=now.isoformat(),start=frame.index.min().isoformat(),
                    end=frame.index.max().isoformat(),timezone='America/New_York', rows=len(frame),
                    adjustment='Yahoo returned OHLC; no dividend adjustment applied; inspect split regime changes',
                    sha256=sha256(output),provenance=dict(method='provider_download',
                        evidence_path=receipt_path.name,evidence_sha256=sha256(receipt_path)))
    output.with_suffix('.csv.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'status':'downloaded','csv':str(output),'manifest':manifest},indent=2))
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbol',default='AAPL',choices=['AAPL'])
    parser.add_argument('--interval',required=True,choices=['1d','1mo'])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--refresh',action='store_true')
    args=parser.parse_args()
    output=args.output or Path(__file__).parent/'data'/f'{args.symbol}_{args.interval}.csv'
    try:
        download(args.symbol,args.interval,output,args.refresh)
    except (RuntimeError,ValueError) as exc:
        raise SystemExit(str(exc))


if __name__=='__main__':
    main()
