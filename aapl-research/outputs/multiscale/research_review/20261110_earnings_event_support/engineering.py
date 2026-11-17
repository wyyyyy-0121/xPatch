"""Artificial arithmetic, calendar boundary, membership and exact-purge fixtures."""
from runtime import BASE, setup, read, write, require, qualify_dependencies
setup(24)
import math
import numpy as np
import pandas as pd
import exchange_calendars as xc
from support import pair_value, structural_rank, rank_report, next_session, ns, phase_report, partition, known_from_filing

spec=read(BASE/'DESIGN_SPEC.json')
qualify_dependencies(spec)
checks=[]
def check(name,condition):
    checks.append({'name':name,'pass':bool(condition)})
    require(condition,name)

def fact(value,factor='1'):
    return {'reported_quarter_diluted_EPS_decimal_string':value,'separate_canonical_share_unit_factor':factor,'currency':'USD_assumed_from_US_issuer_reporting'}

x=pair_value(fact('3'),fact('8','0.25'),50)
check('Exact split alignment retained separately',x['current_aligned_EPS']=='3' and x['prior_aligned_EPS']=='2.00' and x['ratio']=='3/2')
check('Positive natural log',x['sign']==1 and abs(x['z']-math.log(1.5))<1e-15)
x=pair_value(fact('1'),fact('2'),50)
check('Negative natural log',x['sign']==-1 and abs(x['z']+math.log(2))<1e-15)
x=pair_value(fact('2'),fact('8','0.25'),50)
check('True zero stays valid',x['valid'] and x['z']==0 and x['sign']==0)
check('Current missing propagated',pair_value(fact(None),fact('2'),50)['missing_reasons']==['current_unqualified'])
check('Prior missing propagated',pair_value(fact('2'),fact(None),50)['missing_reasons']==['prior_unqualified'])
check('Both missing separately retained',len(pair_value(fact(None),fact(None),50)['missing_reasons'])==2)
check('Nonpositive EPS unqualified',not pair_value(fact('0'),fact('2'),50)['valid'])
def rows(values):
    return [dict(id=str(i),**pair_value(fact(str(v)),fact('1'),50)) for i,v in enumerate(values)]

for name,values,rank in [('empty',[],0),('allzero',[1,1],1),('positive',[2,3,4],2),('negative',[.2,.3,.4],2),('twosigns_onlytwo',[.5,2,.5,2],2),('twosigns_three',[.5,2,3],3),('zero_and_signs',[.5,1,2],3),('one_value',[2,2,2],1)]:
    r=rank_report(rows(values),spec)
    check('Structural/numeric rank '+name,r['numeric_rank']==r['structural_rank']==rank)
v=rows([.5,2,3]);check('Single negative event fragility',structural_rank(v)==3 and structural_rank(v[1:])==2)
v=rows([.4,.5,2,3]);check('Both signs replicated survive every deletion',all(structural_rank(v[:i]+v[i+1:])==3 for i in range(len(v))))
sch=xc.get_calendar('XNYS',start='2021-11-22',end='2021-11-30').schedule
day,op,cl=next_session(sch,pd.Timestamp('2021-11-25T00:00:00',tz='America/New_York'))
check('Holiday skips to next actual session',day=='2021-11-26')
check('Actual half day close',cl-op==210*60_000_000_000)
check('Halfday hourly grid drops residual30min',(cl-op)//(60*60_000_000_000)==3)
day2,op2,cl2=next_session(sch,pd.Timestamp('2021-11-26T14:00:00',tz='America/New_York'))
check('Late Friday knowledge waits through weekend',day2=='2021-11-29')
check('Knowledge exactly at open admits still-unopened session',next_session(sch,pd.Timestamp(op,unit='ns',tz='UTC'))[0]=='2021-11-26')
check('One nanosecond after open waits next session',next_session(sch,pd.Timestamp(op+1,unit='ns',tz='UTC'))[0]=='2021-11-29')
check('Known time uses whole filing date then midnight',known_from_filing({'financial_filing':{'filingDate':'2021-03-13'}})=='2021-03-14T00:00:00-05:00')
clock=np.array([op,op+1,cl,cl+1],dtype=np.int64)
e={'id':'artificial','open_ns':op,'close_ns':cl,**pair_value(fact('2'),fact('1'),50)}
r=phase_report(np.arange(4,dtype=np.int64),{'end_ns':clock},[e],spec)
check('Open exclusive close inclusive',r['event_origin_count']==2 and r['non_event_origin_count']==2)
check('Multiple origins count as one event',r['rank']['event_count']==1 and r['rank']['structural_rank']==1)
e['valid']=False;e['sign']=None;e['z']=None
r=phase_report(np.arange(4,dtype=np.int64),{'end_ns':clock},[e],spec)
check('Unknown has event membership but no rank row',r['independent_event_count']==1 and r['rank']['event_count']==0 and r['missing_event_origin_count']==2)
check('Empty LOO minimum null and pass false',r['minimum_leave_one_out_structural_rank'] is None and not r['retains_rank3_after_any_single_event_deletion'])
fit=np.arange(10,dtype=np.int64);end=np.arange(14,dtype=np.int64)
b,p,t=partition(fit,end,2)
check('Original floor80 split',b==8 and t.tolist()==[8,9])
check('Strict purge excludes equality target boundary',p.tolist()==[0,1,2,3,4,5])
write(BASE/'ENGINEERING_ACCEPTANCE.json',{'accepted':all(x['pass'] for x in checks),'checks':checks,'count':len(checks),'artificial_fixtures_only':True,'actual_source_values_or_NPZ_decoded':False,'actual_pause_resume_test':False,'Q3_read':False,'GPU':False})
print({'artificial_checks':len(checks),'accepted':True})
