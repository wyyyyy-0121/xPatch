"""Fetch only public AAPL evaluation samples/evidence; preserve old experiments."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile
import io

ROOT = Path(__file__).resolve().parent
SESSION = ROOT / 'runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
SESSION.mkdir(parents=True, exist_ok=False)
SOURCES = {
 'frd_sample_page.html': 'https://firstratedata.com/free-intraday-data',
 'frd_aapl_product.html': 'https://firstratedata.com/i/stock/AAPL',
 'frd_faq.html': 'https://firstratedata.com/about/FAQ',
 'frd_quality.html': 'https://firstratedata.com/about/dataquality',
 'frd_license.html': 'https://firstratedata.com/about/license',
 'alpaca_access.md': 'https://docs.alpaca.markets/us/docs/about-market-data-api.md',
 'alpaca_faq.md': 'https://docs.alpaca.markets/us/docs/market-data-faq.md',
 'alpaca_bars.md': 'https://docs.alpaca.markets/us/reference/stockbars.md',
 'frd_sample_stock_AAPL.zip': 'https://frd001.s3.us-east-2.amazonaws.com/frd_sample_stock_AAPL.zip',
}
records = []
for name, url in SOURCES.items():
 record = {'url': url, 'attempted_utc': datetime.now(timezone.utc).isoformat()}
 try:
  with urlopen(Request(url, headers={'User-Agent': 'AAPL-research-data-evaluation/1.0'}), timeout=35) as response:
   body = response.read(20_000_001)
   if len(body) > 20_000_000:
    raise ValueError('Public sample/evidence exceeds 20 MB limit')
   record.update(status=response.status, final_url=response.url,
    headers={k: v for k, v in response.headers.items() if k.lower() in ('content-type', 'content-length', 'last-modified', 'etag', 'date')})
  (SESSION / name).write_bytes(body)
  record.update(path=name, bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
  if name.endswith('.zip'):
   with zipfile.ZipFile(io.BytesIO(body)) as archive:
    record['members'] = [{'name': x.filename, 'bytes': x.file_size, 'compressed_bytes': x.compress_size} for x in archive.infolist()]
 except (HTTPError, URLError, TimeoutError, ValueError, zipfile.BadZipFile) as error:
  record['error'] = str(error)
  record['status'] = getattr(error, 'code', None)
 records.append(record)
 print(json.dumps({k: v for k, v in record.items() if k not in ('headers', 'members')}, ensure_ascii=False), flush=True)
 (SESSION / 'download_receipts.json').write_text(json.dumps({'scope': 'public_provider_evaluation_only', 'formal_eligible': False, 'records': records}, indent=2) + chr(10))
print(json.dumps({'run_directory': str(SESSION), 'sample_members': [r.get('members') for r in records if r.get('path', '').endswith('.zip')]}, indent=2))
