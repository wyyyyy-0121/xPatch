"""Independent saved-source audit. No HTTP, EPS values or market arrays."""
from pathlib import Path
from html.parser import HTMLParser
from datetime import datetime,date,timedelta,timezone
from zoneinfo import ZoneInfo
import csv,hashlib,json,os,re

BASE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class Doc(HTMLParser):
    def __init__(self):
        super().__init__();self.script=None;self.scriptparts=[];self.ld=[];self.title=False;self.titles=[];self.blocks=[];self.capture=None;self.depth=0;self.pp=[]
    def handle_starttag(self,t,a):
        d=dict(a)
        if t=='script':self.script=d.get('type');self.scriptparts=[]
        if t=='title':self.title=True
        if self.capture:
            if t==self.capture:self.depth+=1
        elif t=='p' or (t=='div' and 'pagebody-copy' in d.get('class','').split()):
            self.capture=t;self.depth=1;self.pp=[]
    def handle_endtag(self,t):
        if t=='script':
            if self.script=='application/ld+json':self.ld.append(json.loads(''.join(self.scriptparts)))
            self.script=None
        if t=='title':self.title=False
        if t==self.capture:
            self.depth-=1
            if self.depth==0:
                self.blocks.append(re.sub(r'\s+',' ',''.join(self.pp)).strip());self.capture=None
    def handle_data(self,s):
        if self.script is not None:self.scriptparts.append(s)
        if self.title:self.titles.append(s)
        if self.capture and self.script is None:self.pp.append(s)

def main():
    os.sched_setaffinity(0,{26});checks=[]
    def ck(ok,msg):
        if not ok:raise RuntimeError(msg)
        checks.append(msg)
    spec=read(BASE/'COLLECTION_SPEC.json');p=read(BASE/'COLLECTION_PROTOCOL.json');r=read(BASE/'run/COLLECTION_RESULT.json');cp=read(BASE/'run/CHECKPOINT.json');roster=read(BASE/'ROSTER.json')['events'];ins=read(BASE/'SOURCE_CONTENT_INSPECTION.json');ca=read(BASE/'CODE_ACCEPTANCE.json');sr=read(BASE/'SOURCE_ROSTER_AND_SCHEMA_REVIEW.json')
    ck(p['deadline_epoch']-p['started_epoch']==180,'one180sdeadline')
    ck(r['protocol_sha256']==cp['protocol_sha256']==ins['collection_protocol_sha256']==sha(BASE/'COLLECTION_PROTOCOL.json'),'protocol hashes')
    for f,h in p['locked_files'].items():ck(sha(f)==h,'frozen '+Path(f).name)
    for f,h in ca['reviewed_files'].items():ck(sha(f)==h,'source-reviewed '+Path(f).name)
    ck(ca['accepted_before_execution'] and ca['source_collection_accepted'],'code acceptance')
    ck(datetime.fromisoformat(ca['created_at_UTC']).timestamp()<p['started_epoch'],'review before run')
    ck(sr['admission']['source_collection_only_admitted'] and not sr['admission']['market_execution_admitted'],'source schema scope')
    ck(len(roster)==26 and len({e['id'] for e in roster})==26,'26unique roster')
    ck(sum(e['role']=='development_event' for e in roster)==22,'22development events')
    ck(sum(e['role']=='historical_baseline_only' for e in roster)==4,'4historical baselines')
    ck(len(r['records'])==r['receipts']==r['events_in_roster']==26,'26receipts')
    ck(sum(x['HTTP_attempts'] for x in r['records'])==r['HTTP_attempts']==24,'24totalattempts')
    ck(sum('reused_body' in x for x in r['records'])==2,'2reused sources')
    ck(r['complete_HTTP200_HTML_or_reused']==26 and all(x['complete_body'] and x['HTTP_200_HTML'] for x in r['records']),'26completeHTML')
    ck(not cp['active'] and len(cp['completed'])==26 and not r['unstarted_ids'] and r['all_jobs_terminal'],'terminal checkpoint')
    for c in cp['completed']:ck(sha(BASE/'run'/c['id']/'RECEIPT.json')==c['receipt_sha256'],'commit '+c['id'])
    ck(0<r['seconds']<180 and not r['stopped_by_signal'],'natural completion within180s, no recovery claim')
    ck(not r['market_execution_admitted'] and not r['GPU'] and not r['Q3_market_arrays_read'] and r['new_model_fits']==0,'no market/GPU/Q3/fits')
    ck(spec['automatic_redirects'] is False and spec['retries']==0,'no redirects retry')
    ck(spec['worker_affinity']==spec['parent_affinity']==p['CPU']==[24],'CPU24')
    ck(spec['response_cap_bytes']==524288 and spec['aggregate_response_cap_bytes']==16777216,'fixed bytes')
    ck(24*(524288+1)<16777216,'totalcap implied by perjob bound')
    ck(spec['selected_EPS_not_later_changed_assumed_not_proved'] and not spec['original_PIT_proved'],'currentcopyassumption not PIT')
    own=[];netbytes=rawbytes=0;points=[];refmap={x['id']:x for x in ins['records']};recmap={x['id']:x for x in r['records']}
    for e in roster:
        id=e['id'];rec=recmap[id]
        ck(read(BASE/'run'/id/'RECEIPT.json')==rec,'copied receipt '+id)
        ck(rec['URL']==e['URL'],'fixed URL '+id)
        ck(rec['current_historical_version_assumed'] and not rec['original_PIT_proved'],'source assumption '+id)
        body=Path(rec['reused_body']['path']) if 'reused_body' in rec else BASE/'run'/id/'BODY.html';data=body.read_bytes();digest=hashlib.sha256(data).hexdigest();rawbytes+=len(data)
        expected=rec['reused_body']['sha256'] if 'reused_body' in rec else rec['body_sha256'];ck(digest==expected,'bodyhash '+id)
        if 'reused_body' not in rec:
            ck(rec['status']==200 and rec['content_type'].lower().startswith('text/html'),'HTTP200HTML '+id)
            ck(rec['bytes']==len(data)<=524288,'actual bytecap '+id);netbytes+=len(data)
            if rec['declared_content_length'] is not None:ck(rec['declared_content_length']==len(data),'declared length '+id)
            else:ck(rec['completeness_basis']=='EOF_without_declared_length','EOF basis '+id)
            ck(rec['CPU_affinity']==[24],'worker affinity '+id)
            ck(p['started_epoch']<=rec['started_epoch']<rec['finished_epoch']<p['deadline_epoch'],'inside180s '+id)
            ck(0<=rec['elapsed_seconds']<=12.2 and abs(rec['elapsed_seconds']-(rec['finished_epoch']-rec['started_epoch']))<1e-6,'request duration '+id)
            ck(not Path('/proc/'+str(rec['pid'])).exists(),'worker absent '+id)
            points += [(rec['started_epoch'],1),(rec['finished_epoch'],-1)]
        ck(data.lstrip().lower().startswith(b'<!doctype html>') and data.rstrip().lower().endswith(b'</html>'),'HTML boundaries '+id)
        doc=Doc();doc.feed(data.decode('utf-8'));news=[x for x in doc.ld if x.get('@type')=='NewsArticle'];ck(len(news)==1,'oneNewsArticle '+id);news=news[0]
        ck(news['mainEntityOfPage']==e['URL'],'structured URL '+id)
        ck(news['datePublished'][:10]==id,'structured publish date '+id)
        ordinal={1:'first',2:'second',3:'third',4:'fourth'}[e['expected_quarter_for_URL_only']]
        ck(ordinal+' quarter' in news['headline'].lower(),'headline quarter '+id)
        title=''.join(doc.titles).strip();ck(title==news['headline']+' - Apple','title headline '+id)
        end=date.fromisoformat(e['financial_filing']['reportDate']);long=end.strftime('%B')+' '+str(end.day)+', '+str(end.year)
        blocks=[v for v in doc.blocks if 'today announced' in v.lower() and 'financial results' in v.lower()]
        sentence=next((v.split('.')[0] for v in blocks if long.lower() in v.lower()),None)
        ck(sentence is not None and ordinal+' quarter' in sentence.lower(),'announcement samequarter period '+id)
        known=datetime.combine(date.fromisoformat(e['financial_filing']['filingDate'])+timedelta(days=1),datetime.min.time(),ZoneInfo('America/New_York'))
        ck(e['known_time_assumption_NY']==known.isoformat() and e['known_time_assumption_UTC']==known.astimezone(timezone.utc).isoformat(),'known time '+id)
        ck(not e['historical_initial_version_proof'] and not e['observed_received_time_proof'],'unknown historical version '+id)
        ref=refmap[id]
        for k,v in [('body_sha256',digest),('title',title),('datePublished',news['datePublished']),('dateModified',news.get('dateModified')),('financial_report_end',end.isoformat()),('financial_form',e['financial_filing']['form']),('known_time_assumption_NY',known.isoformat())]:ck(ref[k]==v,'independent '+k+' '+id)
        ck(ref['all_identity_checks_passed'] and not ref['EPS_numerical_extraction_qualified'] and not ref['historical_initial_version_proof'],'inspection scope '+id)
        own.append({'id':id,'role':e['role'],'title':title,'datePublished':news['datePublished'],'dateModified':news.get('dateModified'),'financial_report_end':end.isoformat(),'financial_form':e['financial_filing']['form'],'body_sha256':digest,'body_bytes':len(data),'report_identity_sentence':sentence,'EPS_values_extracted_by_review':False,'historical_original_version_proven':False})
    ck(netbytes<16777216,'aggregate newbodybytes')
    active=maximum=0
    for t,d in sorted(points,key=lambda x:(x[0],x[1])):active+=d;maximum=max(maximum,active)
    ck(active==0 and maximum<=2,'maxIO interval overlap2')
    ck(sum(x['financial_form']=='10-K' for x in own)==6,'six10Kidentity not EPS qualification')
    ck(all('fourth quarter' in x['title'].lower() for x in own if x['financial_form']=='10-K'),'Q4release identities')
    idx=list(csv.DictReader((BASE/'SOURCE_CACHE_INDEX.csv').open()));ck(len(idx)==26,'CSV26rows');csvchecks=0
    for row in idx:
        ref=refmap[row['id']]
        for k in row:ck(row[k]==str(ref[k]),'CSV '+k+' '+row['id']);csvchecks+=1
    ck(ins['identity_all_pass_count']==ins['count']==26,'26content inspection complete')
    files=['ROSTER.json','SOURCE_SCHEMA.json','COLLECTION_SPEC.json','COLLECTION_PROTOCOL.json','CODE_ACCEPTANCE.json','SOURCE_ROSTER_AND_SCHEMA_REVIEW.json','run/COLLECTION_RESULT.json','run/CHECKPOINT.json','SOURCE_CONTENT_INSPECTION.json','SOURCE_CACHE_INDEX.csv','REVIEW_CHECK_REVISION.json','review_saved_cache.py']
    output={'accepted':True,'status':'accepted_raw_official_release_cache_only','reviewer':'/root/earnings_cache_result_review','at_UTC':datetime.now(timezone.utc).isoformat(),'artifact_identity_resource_content_and_CSV_checks_passed':len(checks),'CSV_cells_checked':csvchecks,'checks':checks,'reviewed_artifacts_sha256':{str(BASE/f):sha(BASE/f) for f in files},'independent_identity_records':own,'source_cache':{'events':26,'development_events':22,'historical_baselines':4,'new_GET_attempts':24,'reused_bodies':2,'new_body_bytes':netbytes,'all_body_bytes_including_reuse':rawbytes,'max_request_interval_overlap':maximum,'actual_window_seconds':r['seconds'],'all24worker_PIDs_missing_now':True,'all26checkpoint_receipt_hashes_valid':True,'raw_content_and_cache_identity_accepted':True},'scientific_scope':{'current_historical_copy_not_rewritten_assumption_required':True,'original_as_filed_bytes_or_actual_received_time_proven':False,'six10Kmatches_are_not_quarter_EPS_qualification':True,'EPS_numeric_dataset_qualified':False,'all_event_EPS_extraction_or_innovation_admitted':False,'market_execution_admitted':False,'model_fit_or_candidate_admitted':False,'independent_market_confirmation':False,'research_goal_achieved':False,'new_evidence_integrated_in_PDF':False},'review_execution':{'CPU_affinity':[26],'HTTP_requests':0,'market_arrays_weights_Q3_read':False,'EPS_values_extracted':False,'model_fits':0,'GPU':False,'standard_library_and_saved_source_text_only':True},'material_blockers':[],'claim_review_pending_parent_stage_report':True}
    (BASE/'RESULT_REVIEW.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print({'checks':len(checks),'CSV_cells':csvchecks,'newbytes':netbytes,'allbytes':rawbytes,'maxoverlap':maximum,'accepted':True})

if __name__=='__main__':main()
