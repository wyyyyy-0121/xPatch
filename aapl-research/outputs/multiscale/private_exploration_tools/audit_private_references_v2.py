"""Audit the private draft's references against stored bibliographic evidence."""
from datetime import datetime, timezone
import hashlib
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import sys
import unicodedata

ROOT=Path('/home/ayaka/wslcode/aapl-research')
OUT=ROOT/'outputs/multiscale'
LIT=OUT/'literature'
FUSION=ROOT/'outputs/fusion_followup/literature'
PARENT=Path(sys.argv[1]).resolve()
FAQ=Path('/tmp/aapl_vendor_faq_20261108.html')
HOLM=Path('/tmp/aapl_holm_openalex_20261108.json')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text(encoding='utf8'))
def require(ok,msg):
    if not ok:raise ValueError(msg)
def norm(value):
    value=unicodedata.normalize('NFKD',unescape(str(value))).casefold()
    return ''.join(c for c in value if c.isalnum() and not unicodedata.combining(c))
def find(seq,**tests):
    matches=[x for x in seq if all(x.get(k)==v for k,v in tests.items())]
    require(len(matches)==1,'Metadata record is missing or ambiguous: '+str(tests))
    return matches[0]
def family_from_citation(values):
    return [v.split(',')[0].strip() for v in values]
def family_from_full(values):
    return [v.rsplit(' ',1)[-1] for v in values]
def contains_title(ref,title):return norm(title) in norm(ref)

md=PARENT/'PRIVATE_WORKING_MANUSCRIPT_EN.md'
provenance=load(PARENT/'PRIVATE_ANNUAL_MANUSCRIPT_AUDIT.json')
require(sha(md)==provenance['manuscript_sha256'], 'Manuscript changed')
text=md.read_text(encoding='utf8')
body,refs_part=text.split('## References\n',1)
refs_part=refs_part.split('## Appendix A.',1)[0]
refs={int(m.group(1)):m.group(2) for m in re.finditer(r'^(\d+)\.\s+(.+)$',refs_part,re.MULTILINE)}
require(set(refs)==set(range(1,18)), 'Reference numbering differs')
order=[]
for match in re.finditer(r'\[(\d+)\]',body):
    n=int(match.group(1))
    if n not in order:order.append(n)
require(order==list(range(1,18)), 'References are not in order of first citation')

meta_path=LIT/'reference_metadata_20261107T111218898884Z/receipt.json'
follow_path=LIT/'reference_metadata_20261107T111218898884Z/followup.json'
cross=load(meta_path)['records'];follow=load(follow_path)['records']
xp=find(follow,label='xpatch')['data']
def crossref(doi):
    x=find(cross,doi_requested=doi)
    require(x.get('http_status')==200 and x.get('doi_returned')==doi, 'Crossref DOI did not resolve')
    return x
arxiv_path=LIT/'arxiv_20261107T084026815902Z/receipt.json'
ballarin_path=LIT/'ballarin_v1_20261107T084158007672Z/receipt.json'
kroujiline_path=LIT/'kroujiline_v1_20261107T092802084282Z/receipt.json'
target_path=LIT/'target_representation_metadata_20261107T102218128779Z/receipt.json'
p2=find(load(arxiv_path)['results'],label='patchtst_v2')['metadata']
p6=find(load(ballarin_path)['results'],label='abstract')['metadata']
p7=find(load(kroujiline_path)['items'],label='metadata')['metadata']
p10=find(load(target_path)['results'],name='hu_v1')['metadata']
tft_path=FUSION/'arxiv_metadata.json'
tft=next(x for x in load(tft_path)['entries'] if '1912.09363v3' in x.get('id',''))
lob_path=FUSION/'financial/arxiv_metadata.json'
lob=next(x for x in load(lob_path)['records'] if '1808.03668v6' in x.get('id',''))
kronos_path=FUSION/'minute_returns_20261106/kronos_v1_metadata.json'
kronos=load(kronos_path)
fin_path=LIT/'finmultitime_v1_20261107T102614Z/article-v1.txt'
fin_text=fin_path.read_text(encoding='utf8')
fin_names=['Wenyan Xu','Dawei Xiang','Yue Liu','Xiyu Wang','Yanxiang Ma','Liang Zhang','Chang Xu','Jiaheng Zhang']
require(all(x in fin_text[:20000] for x in fin_names), 'FinMultiTime authors differ from stored version')
holm=load(HOLM)
require(holm.get('doi')=='https://doi.org/10.2307/4615733' and holm.get('publication_year')==1979 and
        holm.get('biblio',{}).get('volume')=='6' and holm['biblio'].get('first_page')=='65' and
        holm['biblio'].get('last_page')=='70', 'Holm metadata differs')
require(holm['biblio'].get('issue') in (None,''), 'Update issue-status handling if source metadata changes')

class PageText(HTMLParser):
    def __init__(self):super().__init__();self.skip=0;self.parts=[]
    def handle_starttag(self,tag,attrs):
        if tag in ('script','style','code','pre'):self.skip+=1
    def handle_endtag(self,tag):
        if tag in ('script','style','code','pre') and self.skip:self.skip-=1
    def handle_data(self,value):
        if not self.skip and value.strip():self.parts.append(re.sub(r'\s+',' ',value.strip()))
faq=PageText();faq.feed(FAQ.read_text(encoding='utf8'))
require(any('Market Data FAQ' in x for x in faq.parts) and 'How are bars aggregated?' in faq.parts, 'Vendor FAQ title or section missing')
faq_match=[x for x in faq.parts if x.startswith('Minute and daily bars are aggregated from trades.')]
require(len(faq_match)==1 and 'to the day (in New York) for daily bars' in faq_match[0],
        'Vendor daily-bar aggregation text not found')

records={
 1:(xp['title'][0],[a['family'] for a in xp['authors']],follow_path,'Crossref title, authors, 2025, volume 39, issue 19, pages 20601–20609; prior selected xPatch methods reading.','No full re-review or final venue style check.'),
 2:(p2['citation_title'][0],family_from_citation(p2['citation_author']),arxiv_path,'arXiv v2 title/authors and abstract; supports temporal patches and channel independence.','No complete review of all experiments.'),
 3:(tft['title'],family_from_full(tft['authors']),tft_path,'arXiv v3 title/authors; prior selected gating and variable-selection reading.','This model was not run in the current comparison.'),
 4:(crossref('10.1016/j.eswa.2022.116659')['title'][0],crossref('10.1016/j.eswa.2022.116659')['authors'],meta_path,'Crossref title/authors and journal metadata; previously checked abstract.','Survey full text was not reviewed.'),
 5:(lob['title'],family_from_full(lob['authors']),lob_path,'arXiv v6 title/authors; prior selected order-book and classification passages.','Different information set and task; not a direct model comparison.'),
 6:(p6['citation_title'][0],family_from_citation(p6['citation_author']),ballarin_path,'arXiv v1 title/authors and selected intraday method/data passages.','Different stock universe and information set.'),
 7:(p7['citation_title'][0],family_from_citation(p7['citation_author']),kroujiline_path,'arXiv v1 metadata; selected news-input and backtest passages checked in local PDF.','Journal-version details and full text not fully checked.'),
 8:(kronos['title'],family_from_full(kronos['authors']),kronos_path,'arXiv v1 title/authors; prior selected candlestick-model passages.','Kronos was not run here.'),
 9:(crossref('10.3844/jcssp.2021.251.264')['title'][0],crossref('10.3844/jcssp.2021.251.264')['authors'],meta_path,'Crossref metadata and selected publisher PDF passages on price/return input features.','Input representations and classification differ from this study’s regression targets.'),
 10:(p10['citation_title'][0],family_from_citation(p10['citation_author']),target_path,'arXiv v1 metadata and selected candlestick input/target passages.','No complete paper-by-paper or direct like-for-like comparison.'),
 11:('FinMultiTime: A Four-Modal Bilingual Dataset for Financial Time-Series Analysis',family_from_full(fin_names),fin_path,'Title/authors and selected modalities/source passages in stored arXiv v1 text.','AAPL history and rights of the underlying source remain unverified.'),
 12:(crossref('10.1080/14697688.2024.2412687')['title'][0],crossref('10.1080/14697688.2024.2412687')['authors'],meta_path,'Crossref metadata and selected AAPL/horizon passages in the author-hosted manuscript.','Published journal text not checked line-by-line against the author manuscript.'),
 13:(crossref('10.1080/07350015.1995.10524599')['title'][0],crossref('10.1080/07350015.1995.10524599')['authors'],meta_path,'Crossref bibliographic metadata; manuscript does not claim this test was applied.','No new full-text or implementation review.'),
 14:(crossref('10.1016/j.ins.2011.12.028')['title'][0],crossref('10.1016/j.ins.2011.12.028')['authors'],meta_path,'Crossref bibliographic metadata; chronological evaluation claim is application-specific.','No full-text re-review.'),
 15:(crossref('10.1214/aos/1176347265')['title'][0],crossref('10.1214/aos/1176347265')['authors'],meta_path,'Crossref bibliographic metadata; block-resampling assumptions remain explicit.','No claim that bootstrap assumptions or stationarity are established.'),
 16:('Market Data FAQ: How are bars aggregated?',['Alpaca'],FAQ,'Official documentation says daily SIP trade timestamps are truncated to New York calendar days.','The excerpt does not establish final-revision timing or account-specific publication rights.'),
 17:(holm['display_name'],family_from_full([x['author']['display_name'] for x in holm['authorships']]),HOLM,'OpenAlex confirms title, author, year 1979, volume 6 and pages 65–70.','The issue is omitted from the draft pending confirmation; the captured issue field is empty.')
}
identifier={1:'10.1609/aaai.v39i19.34270',2:'2211.14730v2',3:'1912.09363v3',
            4:'10.1016/j.eswa.2022.116659',5:'1808.03668v6',6:'2504.19623v1',
            7:'1508.04332v1',8:'2508.02739v1',9:'10.3844/jcssp.2021.251.264',
            10:'2604.12659v1',11:'2506.05019v1',12:'10.1080/14697688.2024.2412687',
            13:'10.1080/07350015.1995.10524599',14:'10.1016/j.ins.2011.12.028',
            15:'10.1214/aos/1176347265',16:'market-data-faq',17:'4615733'}
for n,(title,authors,source,summary,open_item) in records.items():
    require(source.is_file(), 'Evidence source missing for reference '+str(n))
    require(identifier[n] in refs[n], 'Identifier mismatch for reference '+str(n))
    if n!=16:
        require(contains_title(refs[n],title), 'Title mismatch for reference '+str(n))
    require(all(norm(a) in norm(refs[n]) for a in authors), 'Author mismatch for reference '+str(n))
require(xp['DOI']=='10.1609/aaai.v39i19.34270' and xp['volume']=='39' and
        xp['issue']=='19' and xp['page']=='20601-20609', 'xPatch publication details differ')

destination=PARENT/'reference_evidence_20261108'
require(not destination.exists(), 'Reference evidence destination exists')
destination.mkdir()
holm_saved=destination/'HOLM_OPENALEX.json'
shutil.copy2(HOLM,holm_saved)
faq_saved=destination/'ALPACA_FAQ_EXCERPT.json'
faq_record={'url':'https://docs.alpaca.markets/us/docs/market-data-faq',
            'captured_at_utc':datetime.now(timezone.utc).isoformat(),
            'raw_html_sha256':sha(FAQ),'raw_html_archived':False,
            'selected_section':'How are bars aggregated?','selected_text':faq_match[0],
            'scope':'Vendor bar aggregation text only; not publication rights or proof of final revision timing.',
            'market_data_requested':False}
faq_saved.write_text(json.dumps(faq_record,indent=2,ensure_ascii=False)+'\n',encoding='utf8')

lines=['# Reference evidence map — private working manuscript','',
       'This checks identifiers, titles and author names against recorded metadata or selected original passages. '
       'It is not a systematic literature review, a complete full-text audit, or a final journal-format check.','',
       '| Ref. | Identifier | Checked scope | Remaining limitation |',
       '| ---: | --- | --- | --- |']
for n in range(1,18):
    title,authors,source,summary,open_item=records[n]
    lines.append(f'| {n} | {identifier[n]} | {summary} | {open_item} |')
lines += ['', 'The captured OpenAlex record does not supply Holm’s issue number. '
          'The JSTOR request returned an HTML client challenge and the Semantic Scholar request returned HTTP 429; '
          'neither was treated as a successful bibliographic verification. OpenCitations returned a conflicting book '
          'classification with no issue/volume/pages and was not used as authority.', '',
          'Evidence is local: previous receipt paths and SHA-256 digests are listed in the JSON audit. '
          'The vendor excerpt and the Holm OpenAlex response are archived alongside this map. '
          'No market-price endpoint was requested for this check.', '']
map_path=destination/'REFERENCE_EVIDENCE_MAP_EN.md'
map_path.write_text('\n'.join(lines),encoding='utf8')
sources={}
for n,(title,authors,source,summary,open_item) in records.items():
    actual=faq_saved if n==16 else holm_saved if n==17 else source
    sources[str(n)]={'identifier':identifier[n],'title':title,'authors':authors,'source':str(actual),
                     'source_sha256':sha(actual),'checked_scope':summary,'limitation':open_item}
audit={'status':'private_reference_metadata_mapped_not_a_full_literature_review',
       'created_at_utc':datetime.now(timezone.utc).isoformat(),'manuscript':str(md),'manuscript_sha256':sha(md),
       'script_sha256':sha(Path(__file__).resolve()),'map_sha256':sha(map_path),
       'checks':{'references':len(refs),'first_citation_order':order,'identifiers_titles_and_author_names_checked':True,
                 'holm_issue_2_verified':False,'vendor_new_york_day_definition_checked':True},
       'sources':sources,'market_data_requested':False,'Q3_market_data_requested':False,
       'limitations':['Metadata and selected-passages checks do not constitute a systematic or complete literature review.',
                      'Holm issue number and venue-specific citation formatting remain unverified.',
                      'Vendor documentation does not resolve the account-specific publication permission.']}
audit_path=destination/'REFERENCE_EVIDENCE_AUDIT.json'
audit_path.write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
print(json.dumps({'status':audit['status'],'directory':str(destination),'map':str(map_path),
                  'audit':str(audit_path),'checks':audit['checks']},ensure_ascii=False))
