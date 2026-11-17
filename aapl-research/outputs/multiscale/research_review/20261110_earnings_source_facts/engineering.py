"""Artificial HTML and two already-qualified source facts, before full extraction."""
import hashlib
import json
import os
from pathlib import Path
import time
from extract import BASE,extract_fact

def main():
    os.sched_setaffinity(0,{24});started=time.time()
    cache=BASE.parent/'20261110_earnings_release_cache'
    events={x['id']:x for x in json.loads((cache/'ROSTER.json').read_text())['events']}
    sources={x['id']:x for x in json.loads((cache/'SOURCE_CONTENT_INSPECTION.json').read_text())['records']}
    checks=[]
    def check(ok,name):
        if not ok:raise RuntimeError(name)
        checks.append(name)
    for id,expected in [('2020-01-28','4.99'),('2021-01-27','1.68')]:
        value=extract_fact(Path(sources[id]['body_file']).read_text(),events[id],sources[id])
        check(value['reported_quarter_diluted_EPS_decimal_string']==expected,'Previously qualified direct source '+id)
    e=events['2021-01-27'];s=sources['2021-01-27']
    intro='Apple today announced financial results for its fiscal 2021 first quarter ended December 26, 2020. Quarterly earnings per diluted share of $1.68.'
    wrap=lambda html:'<article class="article">'+html+'</article>'
    cases=[('<div class="pagebody-copy">'+intro+'</div>','Olddiv'),('<div class="pagebody-copy"><p>'+intro+'</p></div>','Nestedpdiv'),('<meta name="x"><div><br><p>'+intro+'</p><img src="x"></div>','Voidelements')]
    for html,name in cases:
        value=extract_fact(wrap(html),e,s)
        check(value['reported_quarter_diluted_EPS_decimal_string']=='1.68',name)
    invalid=[('<p>Apple reports EPS $99.99</p>','Headlineonly'),('<p>'+intro.replace('diluted','basic')+'</p>','BasicEPS'),
             ('<p>'+intro.replace('first quarter','full year')+'</p>','Annualcontext'),('<p>'+intro.replace('December 26','December 25')+'</p>','Wrongperiod'),
             ('<p>'+intro.replace('fiscal 2021','fiscal 2022')+'</p>','Wrongfiscalyear'),('<p>'+intro.replace('Quarterly earnings','In the year-ago quarter quarterly earnings')+'</p>','Yearagocomparator'),
             ('<p>'+intro.replace('Quarterly earnings','In the year-ago quarter, diluted earnings').replace('diluted earnings per diluted share','diluted earnings per share')+'</p>','Yearagocommaassignment'),
             ('<p>'+intro+' Excluding charges diluted earnings per share was $2.00.</p>','Adjustedintro'),
             ('<p>'+intro+'</p><p>Excluding charges diluted earnings per share was $2.00 for the first quarter.</p>','Adjustedotherquarter'),
             ('<p>'+intro+'</p><p>'+intro.replace('1.68','1.69')+'</p>','Twoinconsistentintros'),
             ('<p>'+intro+'</p><p>For the first quarter diluted earnings per share was $1.69.</p>','Conflictingcurrentquarter')]
    for html,name in invalid:
        check(extract_fact(wrap(html),e,s)['reported_quarter_diluted_EPS_decimal_string'] is None,name+'unknown')
    check(extract_fact('<p>'+intro+'</p>',e,s)['reported_quarter_diluted_EPS_decimal_string'] is None,'Articlecontainerrequired')
    for text,name in [('For the full year earnings per diluted share of $9.99.','Separateannual'),('For the full year excluding charges diluted earnings per share was $9.99.','Separateadjustedannual')]:
        check(extract_fact(wrap('<p>'+intro+'</p><p>'+text+'</p>'),e,s)['reported_quarter_diluted_EPS_decimal_string']=='1.68',name)
    check(extract_fact(wrap('<p>'+intro.replace('Quarterly earnings','Diluted earnings').replace('Diluted earnings per diluted share','Diluted earnings per share')+'</p>'),e,s)['reported_quarter_diluted_EPS_decimal_string']=='1.68','Unambiguousquarterdirectsentence')
    weeks=extract_fact(wrap('<p>'+intro+'</p><p>The first quarter was 14 weeks.</p>'),e,s)
    check(weeks['duration_weeks']==14,'Explicitweeksretained')
    prior=extract_fact(wrap('<p>'+intro+'</p><p>The year-ago first quarter was 14 weeks.</p>'),e,s)
    check(prior['duration_weeks'] is None,'Priorquarterweeksnotcurrent')
    check(weeks['unqualified_primary_EPS_treated_as_GAAP_assumption'] and not weeks['GAAP_explicitly_labelled'],'ImplicitGAAPdistinctfromproof')
    out={'status':'passed_artificial_and_two_previously_qualified_sources_only','checks':len(checks),'check_descriptions':checks,
         'elapsed_seconds':time.time()-started,'extractor_sha256':hashlib.sha256((BASE/'extract.py').read_bytes()).hexdigest(),
         'engineering_source_sha256':hashlib.sha256((BASE/'engineering.py').read_bytes()).hexdigest(),
         'all26numeric_extraction_executed':False,'HTTP_requests':0,'market_arrays_weights_Q3_read':False,'CPU':[24],'GPU':False}
    (BASE/'ENGINEERING_ACCEPTANCE.json').write_text(json.dumps(out,indent=2)+'\n');print('Engineering passed',len(checks),'checks')

if __name__=='__main__':main()
