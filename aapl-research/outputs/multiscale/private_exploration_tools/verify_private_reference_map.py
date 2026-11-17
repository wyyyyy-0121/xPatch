"""Independently verify the local private manuscript reference evidence map."""
from datetime import datetime,timezone
from hashlib import sha256
from html.parser import HTMLParser
from html import unescape
import json
from pathlib import Path
import re
import sys
import unicodedata

d=Path(sys.argv[1]).resolve()
audit_path=d/'REFERENCE_EVIDENCE_AUDIT.json'
map_path=d/'REFERENCE_EVIDENCE_MAP_EN.md'
audit=json.loads(audit_path.read_text(encoding='utf8'))
md=Path(audit['manuscript'])
def hash_file(p):return sha256(Path(p).read_bytes()).hexdigest()
def require(ok,msg):
 if not ok:raise ValueError(msg)
def norm(s):
 return ''.join(c for c in unicodedata.normalize('NFKD',unescape(str(s))).casefold()
                if c.isalnum() and not unicodedata.combining(c))
require(hash_file(md)==audit['manuscript_sha256'] and hash_file(map_path)==audit['map_sha256'],'Draft or map changed')
text=md.read_text(encoding='utf8')
refs_text=text.split('## References\n',1)[1].split('## Appendix A.',1)[0]
refs={int(m.group(1)):m.group(2) for m in re.finditer(r'^(\d+)\.\s+(.+)$',refs_text,re.MULTILINE)}
require(set(refs)==set(range(1,18)) and set(audit['sources'])=={str(i) for i in range(1,18)},'Reference inventory differs')
for n,ref in refs.items():
 item=audit['sources'][str(n)]
 require(hash_file(item['source'])==item['source_sha256'],'Evidence source changed: '+str(n))
 require(item['identifier'] in ref and all(norm(a) in norm(ref) for a in item['authors']),
         'Reference identifier or author differs: '+str(n))
 if n!=16:require(norm(item['title']) in norm(ref),'Reference title differs: '+str(n))
class Page(HTMLParser):
 def __init__(self):super().__init__();self.skip=0;self.parts=[]
 def handle_starttag(self,t,attrs):
  if t in ('script','style','code','pre'):self.skip+=1
 def handle_endtag(self,t):
  if t in ('script','style','code','pre') and self.skip:self.skip-=1
 def handle_data(self,s):
  if not self.skip and s.strip():self.parts.append(re.sub(r'\s+',' ',s.strip()))
faq_source=Path('/tmp/aapl_vendor_faq_20261108.html')
page=Page();page.feed(faq_source.read_text(encoding='utf8'))
stored=json.loads((d/'ALPACA_FAQ_EXCERPT.json').read_text(encoding='utf8'))
require(any('Market Data FAQ' in x for x in page.parts) and 'How are bars aggregated?' in page.parts and
        stored['selected_text'] in page.parts and hash_file(faq_source)==stored['raw_html_sha256'],
        'Vendor title, section or selected excerpt differs')
holm=json.loads((d/'HOLM_OPENALEX.json').read_text(encoding='utf8'))
require(holm['biblio']['volume']=='6' and holm['biblio']['first_page']=='65' and
        holm['biblio']['last_page']=='70' and holm['biblio']['issue'] in (None,''),
        'Holm volume, pages or missing issue state differs')
out=d/'REFERENCE_EVIDENCE_VERIFICATION.json'
require(not out.exists(),'Reference verification already exists')
result={'status':'private_reference_map_independently_checked',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'source_audit_sha256':hash_file(audit_path),'map_sha256':hash_file(map_path),
        'manuscript_sha256':hash_file(md),'script_sha256':hash_file(Path(__file__).resolve()),
        'checks':{'entries':17,'source_hashes_match':True,'identifiers_titles_and_author_names_match':True,
                  'vendor_title_section_and_excerpt_match':True,'holm_issue_in_source_is_missing':True},
        'limitations':['Issue 2 is not verified by the captured Holm record.',
                       'Metadata and selected excerpts are not a systematic full-text review.',
                       'Vendor documentation does not resolve publication permission.']}
out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
print(json.dumps({'status':result['status'],'path':str(out),'checks':result['checks']},ensure_ascii=False))
