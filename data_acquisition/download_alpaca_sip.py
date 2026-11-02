"""Fetch authenticated AAPL SIP minute bars; no trading endpoints or paid purchase.
Use APCA_API_KEY_ID/APCA_API_SECRET_KEY environment variables, never command args.
Without --start/--end, requests the preselected 20 calendar-audit sessions.
"""
import argparse
import csv
import getpass
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import time as clock
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
 args=p.parse_args()
 if (args.start is None)!=(args.end is None):p.error('Need both --start and --end')
 if args.start is not None:
  if args.end<args.start:p.error('End precedes start')
  ranges=[];a=args.start
  while a<=args.end:
   b=min(args.end,a+timedelta(days=29));ranges.append((a,b));a=b+timedelta(days=1)
 else:ranges=[(date.fromisoformat(x),date.fromisoformat(x)) for x in SAMPLE_DATES]
 key=os.environ.get('APCA_API_KEY_ID');secret=os.environ.get('APCA_API_SECRET_KEY')
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
 root.mkdir(parents=True,exist_ok=False)
 (root/'raw').mkdir()
 manifest={**plan,'created_utc':datetime.now(timezone.utc).isoformat(),
  'endpoint':ENDPOINT,'download_complete':False,'formal_eligible':False,
  'script_sha256':digest(Path(__file__)),'pages':[]}
 def save():
  temp=root/'receipt.tmp';temp.write_text(json.dumps(manifest,indent=2)+chr(10));temp.replace(root/'receipt.json')
 save();bars=[]
 try:
  for ri,(a,b) in enumerate(ranges):
   start=datetime.combine(a,time(),NY).astimezone(timezone.utc)
   stop=datetime.combine(b+timedelta(days=1),time(),NY).astimezone(timezone.utc)
   params={'symbols':'AAPL','timeframe':'1Min','feed':'sip','adjustment':'raw',
      'start':start.isoformat(),'end':stop.isoformat(),'limit':10000,'sort':'asc'}
   seen=set();page=0
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
     if start<=stamp<stop:bars.append(bar)
    token=payload.get('next_page_token')
    manifest['pages'].append({'file':str(path.relative_to(root)),
       'sha256':digest(path),'bytes':len(body),'rows_returned':len(group),
       'range_start':str(a),'range_end':str(b),'page':page,
       'received_utc':datetime.now(timezone.utc).isoformat(),'headers':headers,
       'has_next_page':bool(token)})
    save()
    if not token:break
    if token in seen:raise RuntimeError('Repeated pagination token')
    seen.add(token);params['page_token']=token;page+=1
    if page>500:raise RuntimeError('Pagination size guard exceeded')
   print(json.dumps({'range_start':str(a),'range_end':str(b),'pages':page+1}),flush=True)
  if not bars:raise RuntimeError('No AAPL bars returned; no dataset accepted')
  stamps=[datetime.fromisoformat(x['t'].replace('Z','+00:00')) for x in bars]
  if len(stamps)!=len(set(stamps)):raise RuntimeError('Duplicate timestamps across pages')
  ordered=sorted(zip(stamps,bars),key=lambda x:x[0])
  csvpath=root/'AAPL_sip_1min_raw.csv'
  with csvpath.open('x',newline='') as f:
   writer=csv.writer(f);writer.writerow(['timestamp','open','high','low','close','volume','trade_count','vwap'])
   for stamp,bar in ordered:
    writer.writerow([stamp.astimezone(NY).isoformat(),bar['o'],bar['h'],bar['l'],bar['c'],bar['v'],bar.get('n'),bar.get('vw')])
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
