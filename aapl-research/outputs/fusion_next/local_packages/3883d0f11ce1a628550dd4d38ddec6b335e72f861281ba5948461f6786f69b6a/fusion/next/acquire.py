"""Quota-limited, resumable SIP/raw development-only acquisition; no orders."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import gzip
import importlib.util
import json
from pathlib import Path
import threading
import time
from urllib.request import Request,build_opener,HTTPRedirectHandler
from urllib.parse import urlencode
from urllib.error import HTTPError,URLError

import pandas as pd

from .common import OUT,ROOT,PLAN,STOCKS,BACKGROUND,DEV_START,DEV_END,digest,identity,read_json,write_json,utc_now,status

ENDPOINT='https://data.alpaca.markets/v2/stocks/bars'
_RATE_LOCK=threading.Lock()
_NEXT_REQUEST=0.


def credentials():
    # Existing literal parser reads local credentials without sourcing shell code.
    path=Path('/home/ayaka/xPatch/data_acquisition/download_alpaca_sip.py')
    spec=importlib.util.spec_from_file_location('_read_only_alpaca_helper',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    key,secret=mod.read_credentials()
    if not key or not secret:raise RuntimeError('Local Alpaca credentials unavailable; values suppressed')
    return key,secret


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None


def request(params,key,secret):
    global _NEXT_REQUEST
    # Hard calendar boundary for every symbol, including market background.
    if pd.Timestamp(params['start'])<pd.Timestamp('2021-01-04T00:00Z') or pd.Timestamp(params['end'])>=pd.Timestamp('2026-07-01T04:00Z'):
        raise ValueError('Request crosses declared development boundary')
    opener=build_opener(NoRedirect)
    for attempt in range(5):
        with _RATE_LOCK:
            delay=max(0.,_NEXT_REQUEST-time.monotonic())
            if delay:time.sleep(delay)
            _NEXT_REQUEST=time.monotonic()+.45  # <= 134 requests/minute in this process
        req=Request(ENDPOINT+'?'+urlencode(params),headers={'APCA-API-KEY-ID':key,'APCA-API-SECRET-KEY':secret,
                    'User-Agent':'Local-foundation-fusion-research/1.0'})
        try:
            with opener.open(req,timeout=45) as response:
                if response.url.split('?')[0]!=ENDPOINT:raise RuntimeError('Unexpected data endpoint')
                body=response.read(30_000_001)
                if len(body)>30_000_000:raise RuntimeError('Data page exceeds size guard')
                headers={k:v for k,v in response.headers.items() if k.lower() in
                         ('date','x-request-id','x-ratelimit-limit','x-ratelimit-remaining','x-ratelimit-reset')}
                return body,headers
        except HTTPError as exc:
            if (exc.code==429 or exc.code>=500) and attempt<4:
                time.sleep(min(45,3*(attempt+1)));continue
            raise RuntimeError(f'Historical data access rejected HTTP {exc.code}; no feed fallback; body suppressed') from None
        except (URLError,TimeoutError):
            if attempt<4:time.sleep(min(20,2**attempt));continue
            raise RuntimeError('Historical data network failure after bounded retries') from None


def bounds(start,end):
    low=pd.Timestamp(start,tz='America/New_York').tz_convert('UTC')
    high=(pd.Timestamp(end)+pd.Timedelta(days=1)).tz_localize('America/New_York').tz_convert('UTC')-pd.Timedelta(nanoseconds=1)
    return low.isoformat(),high.isoformat()


def probe():
    key,secret=credentials();low,high=bounds('2021-01-04','2021-01-04')
    body,headers=request({'symbols':'SPY,QQQ,MSFT','timeframe':'1Min','start':low,'end':high,
                          'feed':'sip','adjustment':'raw','limit':10000,'sort':'asc'},key,secret)
    payload=json.loads(body)
    report={'checked_at_utc':utc_now(),'endpoint':ENDPOINT,'feed':'sip','adjustment':'raw',
            'requested_date':'2021-01-04','rows_by_symbol':{s:len(v) for s,v in payload['bars'].items()},
            'pagination_needed':bool(payload.get('next_page_token')),'response_headers':headers,
            'credentials_available':True,'new_data_stays_before_Q3':True}
    OUT.mkdir(parents=True,exist_ok=True)
    with gzip.open(OUT/'access_probe_20210104.json.gz','wb') as f:f.write(body)
    write_json(OUT/'data_access_probe.json',report)
    return report


def download(symbol):
    if symbol not in STOCKS+BACKGROUND or symbol=='AAPL':raise ValueError('Symbol outside new download scope')
    directory=OUT/'raw'/symbol;directory.mkdir(parents=True,exist_ok=True)
    receipt_path=directory/'receipt.json'
    scope={'symbol':symbol,'start':DEV_START,'end':DEV_END,'feed':'sip','adjustment':'raw','timeframe':'1Min'}
    receipt=read_json(receipt_path) if receipt_path.exists() else {
        'scope':scope,'scope_sha256':identity(scope),'plan_sha256':digest(PLAN),'pages':[],
        'download_complete':False,'created_at_utc':utc_now()}
    if receipt['scope']!=scope:raise ValueError('Existing data request scope changed')
    if receipt['download_complete']:return receipt
    key,secret=credentials();low,high=bounds(DEV_START,DEV_END)
    params={'symbols':symbol,'timeframe':'1Min','start':low,'end':high,'feed':'sip','adjustment':'raw','limit':10000,'sort':'asc'}
    seen=set();last_stamp=None;row_count=0
    for i,entry in enumerate(receipt['pages']):
        p=directory/entry['file']
        if i!=entry['page'] or digest(p)!=entry['sha256']:raise ValueError('Corrupt cached SIP response')
        with gzip.open(p,'rb') as f:payload=json.load(f)
        bars=payload['bars'].get(symbol,[]) or [];row_count+=len(bars)
        if bars:last_stamp=pd.Timestamp(bars[-1]['t'])
        token=payload.get('next_page_token')
        if not token:
            receipt.update(download_complete=True,rows=row_count,finished_at_utc=utc_now());write_json(receipt_path,receipt);return receipt
        if token in seen:raise ValueError('Repeated cached pagination token')
        seen.add(token);params['page_token']=token
    for page in range(len(receipt['pages']),1000):
        body,headers=request(params,key,secret);payload=json.loads(body)
        bars=payload['bars'].get(symbol,[]) or []
        for bar in bars:
            stamp=pd.Timestamp(bar['t'])
            if not pd.Timestamp(low)<=stamp<=pd.Timestamp(high):raise ValueError('Provider returned timestamp outside request')
            if last_stamp is not None and stamp<=last_stamp:raise ValueError('Unordered or duplicate provider minute')
            last_stamp=stamp
        p=directory/f'page_{page:04d}.json.gz'
        with gzip.open(p,'wb',compresslevel=3) as f:f.write(body)
        token=payload.get('next_page_token');row_count+=len(bars)
        receipt['pages'].append({'page':page,'file':p.name,'sha256':digest(p),'rows':len(bars),
                                 'bytes':p.stat().st_size,'has_next_page':bool(token),
                                 'received_at_utc':utc_now(),'headers':headers})
        receipt['rows']=row_count;write_json(receipt_path,receipt)
        write_json(OUT/'acquisition_progress'/f'{symbol}.json',{'symbol':symbol,'pages':page+1,'rows':row_count,'last_timestamp':str(last_stamp),'at_utc':utc_now()})
        if not token:
            receipt.update(download_complete=True,finished_at_utc=utc_now());write_json(receipt_path,receipt);return receipt
        if token in seen:raise ValueError('Repeated provider pagination token')
        seen.add(token);params['page_token']=token
    raise RuntimeError('SIP pagination page guard exceeded')


def all_development(workers=6):
    status('F1_data_acquisition',symbols=list(STOCKS+BACKGROUND),holdout_accessed=False)
    probe()
    results={}
    if not 1<=workers<=8:raise ValueError('Download concurrency must stay within the admitted I/O range')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs={pool.submit(download,s):s for s in STOCKS+BACKGROUND if s!='AAPL'}
        for future in as_completed(jobs):
            symbol=jobs[future]
            try:r=future.result();results[symbol]={'passed':True,'rows':r['rows'],'pages':len(r['pages'])}
            except Exception as exc:
                results[symbol]={'passed':False,'error':repr(exc)}
                write_json(OUT/'data_failures'/f'{symbol}.json',results[symbol])
            write_json(OUT/'download_summary.json',{'updated_at_utc':utc_now(),'symbols':results,'Q3_accessed':False})
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--probe',action='store_true');parser.add_argument('--workers',type=int,default=6);args=parser.parse_args()
    if args.probe:print(json.dumps(probe(),ensure_ascii=False,indent=2))
    else:print(json.dumps(all_development(args.workers),ensure_ascii=False,indent=2))
