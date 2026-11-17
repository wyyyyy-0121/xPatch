"""Quarterly reported source facts; no network, numerical library or market input."""
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import signal
import sys
from decimal import Decimal

BASE=Path(__file__).resolve().parent
CACHE=BASE.parent/'20261110_earnings_release_cache'

def require(ok,message):
    if not ok:raise RuntimeError(message)
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    temporary=path.with_suffix('.partial')
    with temporary.open('w') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)

class Paragraphs(HTMLParser):
    def __init__(self):
        super().__init__();self.depth=0;self.skip=0;self.active=[];self.blocks=[];self.ordinal=0;self.article_depth=None
    def handle_starttag(self,tag,attrs):
        self.depth+=1
        if tag in ['script','style']:self.skip+=1
        cls=dict(attrs).get('class','').split()
        if tag=='article' and 'article' in cls:self.article_depth=self.depth
        if self.article_depth is not None and (tag=='p' or (tag=='div' and 'pagebody-copy' in cls)):
            self.ordinal+=1;self.active.append({'depth':self.depth,'tag':tag,'ordinal':self.ordinal,'parts':[]})
    def handle_startendtag(self,tag,attrs):pass
    def handle_endtag(self,tag):
        # HTML void elements do not receive end tags; handle_starttag compensates below.
        for item in list(self.active):
            if item['tag']==tag and item['depth']==self.depth:
                text=re.sub(r'\s+',' ',' '.join(item['parts'])).strip()
                if text:self.blocks.append({'DOM_block_ordinal':item['ordinal'],'tag':tag,'text':text})
                self.active.remove(item)
        if tag in ['script','style']:self.skip=max(0,self.skip-1)
        if tag=='article' and self.depth==self.article_depth:self.article_depth=None
        self.depth=max(0,self.depth-1)
    def handle_data(self,data):
        if not self.skip:
            for item in self.active:item['parts'].append(data)

# A small second parser avoids dependence on browser error recovery for void elements.
class SourceParagraphs(Paragraphs):
    def handle_starttag(self,tag,attrs):
        if tag in ['area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr']:
            if tag=='br':
                for item in self.active:item['parts'].append(' ')
            return
        super().handle_starttag(tag,attrs)

EPS=re.compile(r'\b(?:quarterly\s+)?(?:earnings\s+per\s+diluted\s+share|diluted\s+earnings\s+per\s+share)\s+(?:of|was|were)\s*\$\s*(\d+\.\d+)',re.I)
DIRECT_EPS=re.compile(r'(?:earnings\s+per\s+diluted\s+share|diluted\s+earnings\s+per\s+share)\s+(?:of|was|were)\s*\$\s*(\d+\.\d+)',re.I)
DISALLOWED=re.compile(r'\b(?:adjusted|non[- ]GAAP|excluding|excludes|exclude|full[- ]year|annual)\b',re.I)
ADJUSTED=re.compile(r'\b(?:adjusted|non[- ]GAAP|excluding|excludes|exclude)\b',re.I)
def attribution_prefix(text,start):
    sentence=re.split(r'(?<=[.!?])\s+',text[:start])[-1]
    prefix=re.split(r'[,;]',sentence)[-1]
    explicit_prior_assignment=re.search(r'\b(?:in|for|during)\s+(?:the\s+)?(?:year[- ]ago|prior|previous|last(?:\s+year[\u2019\x27]s)?)\s+(?:quarter|period|year)\b',sentence,re.I)
    return bool(explicit_prior_assignment or re.search(r'\b(?:last\s+year|prior|previous|year[- ]ago|comparative|comparable|compared|versus)\b',prefix,re.I))

def extract_fact(html,event,source):
    parser=SourceParagraphs();parser.feed(html)
    ordinal={1:'first',2:'second',3:'third',4:'fourth'}[event['expected_quarter_for_URL_only']]
    from datetime import date
    end=date.fromisoformat(event['financial_filing']['reportDate'])
    end_text=end.strftime('%B')+' '+str(end.day)+', '+str(end.year)
    blocks=[x for x in parser.blocks if re.search(r'Apple\s*(?:®\s*)?today\s+announced',x['text'],re.I)
            and 'financial results' in x['text'].lower() and ordinal+' quarter' in x['text'].lower()
            and end_text.lower() in x['text'].lower()]
    # Nested modern p/div blocks often contain identical text. Require one unique
    # introductory disclosure paragraph, rather than choosing from several values.
    unique={x['text']:x for x in blocks}
    qualified_other_disclosures=[x for x in parser.blocks if ADJUSTED.search(x['text']) and
          not (re.search(r'\b(?:annual|full[- ]year|fiscal\s+year)\b',x['text'],re.I) and not re.search(r'\bquarter',x['text'],re.I)) and
          re.search(r'(?:earnings\s+per\s+diluted\s+share|diluted\s+earnings\s+per\s+share|\bEPS\b)',x['text'],re.I)]
    record={'id':event['id'],'role':event['role'],'SEC_8K_accession':event['8K']['accessionNumber'],
            'SEC_financial_form_accession':event['financial_filing']['accessionNumber'],
            'financial_form':event['financial_filing']['form'],'reported_period_end':event['financial_filing']['reportDate'],
            'fiscal_year':None,'fiscal_quarter':None,'expected_fiscal_quarter_from_metadata':event['expected_quarter_for_URL_only'],
            'period_is_explicit_quarter':False,'duration_weeks':None,'duration_weeks_status':'not_extracted_or_assumed',
            'reported_quarter_diluted_EPS_decimal_string':None,'currency':'USD_assumed_from_US_issuer_reporting',
            'source_dollar_symbol_explicit':False,'original_share_basis':'pre2020_4to1split' if event['id']<'2020-08-31' else 'post2020_4to1split',
            'separate_canonical_share_unit_factor':'0.25' if event['id']<'2020-08-31' else '1',
            'source_body_sha256':source['body_sha256'],'source_file':source['body_file'],
            'source_datePublished':source['datePublished'],'source_dateModified':source['dateModified'],
            'known_time_assumption_NY':event['known_time_assumption_NY'],
            'historical_original_version_proved':False,'selected_EPS_not_later_revised_assumed':True,
            'GAAP_explicitly_labelled':False,'unqualified_primary_EPS_treated_as_GAAP_assumption':True,
            'market_feature_constructed':False,'innovation_constructed':False,'missing_or_conflict_status':None,
            'matching_introductory_block_count':len(unique),'all_matching_introduction_blocks':list(unique.values())}
    if len(unique)!=1:
        record['missing_or_conflict_status']='unknown_introductory_context_count';return record
    block=next(iter(unique.values()));text=block['text'];matches=list(EPS.finditer(text))
    fy=re.search(r'\bfiscal\s+(\d{4})\s+'+ordinal+r'\s+quarter',text,re.I)
    record.update(introductory_quote=text,source_location={'DOM_block_ordinal':block['DOM_block_ordinal'],'tag':block['tag']},
                  period_is_explicit_quarter=bool(fy),fiscal_quarter=event['expected_quarter_for_URL_only'] if fy else None,fiscal_year=int(fy.group(1)) if fy else None,
                  all_diluted_EPS_text_candidates=[{'decimal_string':m.group(1),'phrase':m.group(0),'start':m.start(),'end':m.end()} for m in matches],
                  disallowed_qualifiers_found=DISALLOWED.findall(text))
    record['other_adjusted_or_annual_EPS_disclosures']=qualified_other_disclosures
    comparison_attribution=False;other_current_values=[];week_evidence=[]
    for p in parser.blocks:
        # Only an explicit quarter assignment counts as a same-quarter competitor.
        # Annual/comparative facts remain separate source material, never a fallback.
        if re.search(r'\b(?:quarterly|'+ordinal+r'\s+quarter)\b',p['text'],re.I) and not re.search(r'\b(?:annual|full[- ]year|fiscal\s+year)\b',p['text'],re.I):
            for m in DIRECT_EPS.finditer(p['text']):
                if not attribution_prefix(p['text'],m.start()):other_current_values.append({'decimal_string':m.group(1),'quote':p['text'],'DOM_block_ordinal':p['DOM_block_ordinal']})
            for m in re.finditer(r'\b(\d{1,2})[- ]weeks?\b',p['text'],re.I):
                week_evidence.append({'weeks':int(m.group(1)),'quote':p['text'],'DOM_block_ordinal':p['DOM_block_ordinal'],
                                      'current_quarter_assignment':not attribution_prefix(p['text'],m.start())})
    record['all_explicit_quarter_EPS_evidence']=other_current_values
    record['duration_week_evidence']=week_evidence
    unique_weeks=sorted({v['weeks'] for v in week_evidence if v['current_quarter_assignment']})
    if len(unique_weeks)==1:record.update(duration_weeks=unique_weeks[0],duration_weeks_status='explicit_quarter_assigned_source_text')
    elif unique_weeks:record['duration_weeks_status']='multiple_explicit_week_counts_unassigned'
    for match in matches:
        comparison_attribution|=attribution_prefix(text,match.start())
    record['comparison_attribution_detected']=comparison_attribution
    conflicting_other_values=len(matches)==1 and any(Decimal(v['decimal_string'])!=Decimal(matches[0].group(1)) for v in other_current_values)
    record['conflicting_explicit_quarter_EPS']=conflicting_other_values
    if not fy or int(fy.group(1))!=event['expected_fiscal_year_from_event'] or len(matches)!=1 or DISALLOWED.search(text) or qualified_other_disclosures or comparison_attribution or conflicting_other_values:
        record['missing_or_conflict_status']='unknown_ambiguous_or_qualified_EPS';return record
    value=matches[0].group(1)
    require(Decimal(value)>=0,'Negative or invalid EPS source fact')
    record.update(reported_quarter_diluted_EPS_decimal_string=value,source_dollar_symbol_explicit=True,
                  missing_or_conflict_status='qualified_current_official_quarter_fact_under_version_and_currency_assumptions')
    return record

def main():
    os.sched_setaffinity(0,{24});os.environ['CUDA_VISIBLE_DEVICES']=''
    def expired(signum,frame):raise TimeoutError('30second source-fact extraction deadline')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,30)
    spec=read(BASE/'EXTRACTION_SPEC.json');design=read(BASE/'DESIGN_REVIEW.json');code=read(BASE/'CODE_ACCEPTANCE.json')
    require(design['admission']['source_fact_extraction_admitted'] and code['source_fact_extraction_accepted'],'Review required')
    require(not (BASE/'SOURCE_FACTS.json').exists(),'Do not overwrite first extraction')
    for file,expected in code['reviewed_files'].items():require(sha(file)==expected,'Reviewed source changed')
    roster=read(CACHE/'ROSTER.json')['events'];sources=read(CACHE/'SOURCE_CONTENT_INSPECTION.json')['records']
    require(len(roster)==len(sources)==26,'All fixed sources required')
    facts=[]
    for event,source in zip(roster,sources):
        require(event['id']==source['id'] and sha(source['body_file'])==source['body_sha256'],'Fixed source identity')
        facts.append(extract_fact(Path(source['body_file']).read_text(),event,source))
        write(BASE/'PARTIAL_SOURCE_FACTS.json',{'completed':len(facts),'facts':facts,'market_execution':False})
    write(BASE/'SOURCE_FACTS.json',{'source_count':26,'facts':facts,'qualified_count':sum(x['reported_quarter_diluted_EPS_decimal_string'] is not None for x in facts),
          'qualified_count_means_parsed_not_independent_source_acceptance':True,'final_source_fact_qualification':False,
          'source_fact_extraction_only':True,'market_feature_or_innovation_constructed':False,'source_assumptions':spec['source_assumptions'],
          'source_spec_sha256':sha(BASE/'EXTRACTION_SPEC.json'),'market_execution_admitted':False,'GPU':False,'Q3_market_arrays_read':False,'new_model_fits':0})
    print({'source_facts':len(facts),'qualified':sum(x['reported_quarter_diluted_EPS_decimal_string'] is not None for x in facts)})

if __name__=='__main__':main()
