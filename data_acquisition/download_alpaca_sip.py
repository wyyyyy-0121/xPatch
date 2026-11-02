"""Fetch authenticated AAPL SIP minute bars; no trading endpoints or paid purchase.
Use APCA_API_KEY_ID/APCA_API_SECRET_KEY environment variables, never command args.
Without --start/--end, requests the preselected 20 calendar-audit sessions.
"""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import getpass
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import time as clock
import threading
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, build_opener, HTTPRedirectHandler
from zoneinfo import ZoneInfo

SAMPLE_DATES = ['2024-03-04','2024-03-05','2024-03-06','2024-03-07','2024-03-08',
 '2024-03-11','2024-03-12','2024-03-13','2024-03-14','2024-03-15',
 '2024-10-25','2024-10-28','2024-10-29','2024-10-30','2024-10-31',
 '2024-11-22','2024-11-25','2024-11-26','2024-11-27','2024-11-29']
ENDPOINT='https://data.alpaca.markets/v2/stocks/bars'
NY=ZoneInfo('America/New_York')

class RejectRedirect(HTTPRedirectHandler):
 def redirect_request(self, req, fp, code, msg, headers, newurl):
  return None

OPENER=build_opener(RejectRedirect)

def digest(path):
 return hashlib.sha256(path.read_bytes()).hexdigest()

def read_credentials(path=None):
 """Read only the two literal credential assignments; never source shell code."""
 names=('APCA_API_KEY_ID','APCA_API_SECRET_KEY')
 values={name:os.environ.get(name,'') for name in names}
 candidate=Path(path).expanduser() if path else Path.home()/'.config/alpaca/credentials.env'
 if not all(values.values()) and candidate.is_file():
  for line in candidate.read_text().splitlines():
   line=line.strip()
   if not line or line.startswith('#'):continue
   if line.startswith('export '):line=line[7:].strip()
   name,sep,value=line.partition('=')
   if not sep or name.strip() not in names:continue
   try:parts=shlex.split(value,comments=True)
   except ValueError:raise ValueError('Invalid literal credential assignment; values suppressed') from None
   if len(parts)!=1:raise ValueError('Credential assignments require one literal value; values suppressed')
   if not values[name.strip()]:values[name.strip()]=parts[0]
 return tuple(values[name] for name in names)

def request_body(params,key,secret):
 request=Request(ENDPOINT+'?'+urlencode(params),headers={
  'APCA-API-KEY-ID':key,'APCA-API-SECRET-KEY':secret,
  'User-Agent':'AAPL-research-data-evaluation/1.0'})
 for attempt in range(3):
  try:
   with OPENER.open(request,timeout=45) as response:
    if response.url.split('?')[0]!=ENDPOINT:
     raise RuntimeError('Unexpected response host/path')
    body=response.read(30_000_001)
    if len(body)>30_000_000:raise RuntimeError('Response exceeds size guard')
    return body,{k:v for k,v in response.headers.items()
     if k.lower() in ('date','x-request-id','content-type')}
  except HTTPError as exc:
   if (exc.code==429 or exc.code>=500) and attempt<2:
    clock.sleep(min(10,2**attempt));continue
   raise RuntimeError('Data request rejected: HTTP '+str(exc.code)+'. Check credentials/permissions; no feed fallback.') from None
  except (URLError,TimeoutError):
   if attempt<2:clock.sleep(2**attempt);continue
   raise RuntimeError('Data request failed after three attempts') from None

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--start',type=date.fromisoformat)
 p.add_argument('--end',type=date.fromisoformat,help='inclusive local date; historical development only')
 p.add_argument('--output',type=Path)
 p.add_argument('--dry-run',action='store_true')
 p.add_argument('--prompt-credentials',action='store_true',help='Read keys in your local terminal without echo or persistence')
 p.add_argument('--credentials-file',type=Path,help='Literal local env file; defaults to ~/.config/alpaca/credentials.env when environment keys are absent')
 p.add_argument('--resume',action='store_true',help='Verify existing raw pages and continue the same requested date ranges')
 p.add_argument('--workers',type=int,default=4,help='Concurrent date ranges (1-4; API quota friendly)')
 args=p.parse_args()
 if not 1<=args.workers<=4:p.error('--workers must be between 1 and 4')
 if (args.start is None)!=(args.end is None):p.error('Need both --start and --end')
 if args.start is not None:
  if args.end<args.start:p.error('End precedes start')
  ranges=[];a=args.start
  while a<=args.end:
   b=min(args.end,a+timedelta(days=29));ranges.append((a,b));a=b+timedelta(days=1)
 else:ranges=[(date.fromisoformat(x),date.fromisoformat(x)) for x in SAMPLE_DATES]
 key,secret=read_credentials(args.credentials_file)
 plan={'symbol':'AAPL','feed':'sip','adjustment':'raw','timeframe':'1Min',
       'purpose':'development_data_acquisition_only','independent_confirmation':False,
       'ranges':[(str(a),str(b)) for a,b in ranges],
       'credentials_available':bool(key and secret)}
 if args.dry_run:
  print(json.dumps(plan,indent=2));return
 if args.prompt_credentials and (not key or not secret):
  key=getpass.getpass('Alpaca API key (hidden): ')
  secret=getpass.getpass('Alpaca secret (hidden): ')
 if not key or not secret:
  p.exit(2,'Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY in this Linux process. No request sent; no output created. Do not paste secrets into chat.\n')
 if datetime.combine(ranges[-1][1]+timedelta(days=1),time(),NY).astimezone(timezone.utc)>datetime.now(timezone.utc)-timedelta(minutes=20):
  p.error('Use complete historical dates ending at least 20 minutes ago')
 root=args.output or Path(__file__).resolve().parent/'runs'/('alpaca_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
 if args.resume:
  if not args.output or not (root/'receipt.json').is_file():p.error('--resume requires --output pointing to an existing receipt')
  manifest=json.loads((root/'receipt.json').read_text())
  for field in ('symbol','feed','adjustment','timeframe','purpose','ranges'):
   if json.dumps(manifest.get(field))!=json.dumps(plan[field]):p.error('Resume request differs from saved receipt: '+field)
  manifest['resume_script_sha256']=digest(Path(__file__))
  manifest['resumed_utc']=datetime.now(timezone.utc).isoformat()
 else:
  root.mkdir(parents=True,exist_ok=False)
  (root/'raw').mkdir()
  manifest={**plan,'created_utc':datetime.now(timezone.utc).isoformat(),
   'endpoint':ENDPOINT,'download_complete':False,'formal_eligible':False,
   'script_sha256':digest(Path(__file__)),'pages':[]}
 def save():
  with manifest_lock:
   temp=root/'receipt.tmp';temp.write_text(json.dumps(manifest,indent=2)+chr(10));temp.replace(root/'receipt.json')
 manifest_lock=threading.Lock()
 save();bars=[]
 try:
  def fetch_range(ri,a,b):
   local_bars=[]
   start=datetime.combine(a,time(),NY).astimezone(timezone.utc)
   stop=datetime.combine(b+timedelta(days=1),time(),NY).astimezone(timezone.utc)
   params={'symbols':'AAPL','timeframe':'1Min','feed':'sip','adjustment':'raw',
      'start':start.isoformat(),'end':stop.isoformat(),'limit':10000,'sort':'asc'}
   seen=set();page=0
   cached=[entry for entry in manifest['pages'] if entry['range_start']==str(a) and entry['range_end']==str(b)]
   for entry in cached:
    if entry['page']!=page:raise RuntimeError('Non-contiguous cached page numbers')
    path=root/entry['file']
    if digest(path)!=entry['sha256'] or path.stat().st_size!=entry['bytes']:raise RuntimeError('Cached raw page integrity failure')
    payload=json.loads(path.read_bytes())
    group=payload.get('bars',{}).get('AAPL',[]) or []
    if len(group)!=entry['rows_returned']:raise RuntimeError('Cached raw page row count mismatch')
    for bar in group:
     stamp=datetime.fromisoformat(bar['t'].replace('Z','+00:00'))
     if start<=stamp<stop:local_bars.append(bar)
    token=payload.get('next_page_token')
    if bool(token)!=entry['has_next_page']:raise RuntimeError('Cached pagination metadata mismatch')
    page+=1
    if token:
     if token in seen:raise RuntimeError('Repeated cached pagination token')
     seen.add(token);params['page_token']=token
    elif page!=len(cached):raise RuntimeError('Cached pages continue after final page')
   if cached and not cached[-1]['has_next_page']:
    print(json.dumps({'range_start':str(a),'range_end':str(b),'pages':page,'cached':True}),flush=True)
    return local_bars
   while True:
    body,headers=request_body(params,key,secret)
    path=root/'raw'/('range_%03d_page_%04d.json'%(ri,page))
    with path.open('xb') as f:f.write(body)
    payload=json.loads(body)
    if not isinstance(payload.get('bars'),dict):raise RuntimeError('Unexpected bars payload')
    group=payload['bars'].get('AAPL',[])
    if group is None:group=[]
    for bar in group:
     stamp=datetime.fromisoformat(bar['t'].replace('Z','+00:00'))
     if stamp.tzinfo is None:raise RuntimeError('Response timestamp lacks timezone')
     if start<=stamp<stop:local_bars.append(bar)
    token=payload.get('next_page_token')
    entry={'file':str(path.relative_to(root)),
       'sha256':digest(path),'bytes':len(body),'rows_returned':len(group),
       'range_start':str(a),'range_end':str(b),'page':page,
       'received_utc':datetime.now(timezone.utc).isoformat(),'headers':headers,
       'has_next_page':bool(token)}
    with manifest_lock:
     manifest['pages'].append(entry)
    save()
    if not token:break
    if token in seen:raise RuntimeError('Repeated pagination token')
    seen.add(token);params['page_token']=token;page+=1
    if page>500:raise RuntimeError('Pagination size guard exceeded')
   print(json.dumps({'range_start':str(a),'range_end':str(b),'pages':page+1}),flush=True)
   return local_bars
  with ThreadPoolExecutor(max_workers=args.workers,thread_name_prefix='alpaca-range') as pool:
   futures=[pool.submit(fetch_range,i,a,b) for i,(a,b) in enumerate(ranges)]
   for future in as_completed(futures):bars.extend(future.result())
  if not bars:raise RuntimeError('No AAPL bars returned; no dataset accepted')
  stamps=[datetime.fromisoformat(x['t'].replace('Z','+00:00')) for x in bars]
  if len(stamps)!=len(set(stamps)):raise RuntimeError('Duplicate timestamps across pages')
  ordered=sorted(zip(stamps,bars),key=lambda x:x[0])
  csvpath=root/'AAPL_sip_1min_raw.csv'
  candidate=root/'AAPL_sip_1min_raw.csv.partial'
  with candidate.open('w',newline='') as f:
   writer=csv.writer(f);writer.writerow(['timestamp','open','high','low','close','volume','trade_count','vwap'])
   for stamp,bar in ordered:
    writer.writerow([stamp.astimezone(NY).isoformat(),bar['o'],bar['h'],bar['l'],bar['c'],bar['v'],bar.get('n'),bar.get('vw')])
  if csvpath.exists():
   if digest(csvpath)!=digest(candidate):raise RuntimeError('Existing CSV differs from verified cached raw pages; original preserved')
   candidate.unlink()
  else:candidate.replace(csvpath)
  manifest.pop('failure',None)
  manifest.update(download_complete=True,rows=len(ordered),start=ordered[0][0].isoformat(),
   end=ordered[-1][0].isoformat(),csv=csvpath.name,csv_sha256=digest(csvpath),
   timezone='America/New_York',timestamp_label='bar_start (must be checked against provider aggregation documentation)',
   source_verified=False,notes=['Requested SIP explicitly; never fallback to IEX.',
    'raw specifies corporate-action adjustment, not tick-level or point-in-time availability.',
    'RTH coverage, independent cross-check and corporate-action audit still required.',
    'This acquisition is developmental; inspected historical data is not a sealed confirmation set.'])
  save();print(json.dumps({'output':str(root),'rows':len(ordered),'download_complete':True}))
 except (RuntimeError,ValueError,KeyError,TypeError) as error:
  manifest['failure']=str(error);save()
  raise SystemExit(str(error)) from None

if __name__=='__main__':main()
