"""Independent source/calendar/support replay. Import performs no numerics.

Only runtime resource/dependency/atomic-I/O helpers are shared with main.
No support.py functions, prices, labels, predictions or weights are imported.
"""
import bisect
import datetime as dt
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import io
import math
import os
import signal
import sys
import time
import zipfile
from zoneinfo import ZoneInfo
from runtime import BASE, setup, protocol, qualify_dependencies, guard, read, write, sha, require

CHECKS = 0
MINUTE = 60_000_000_000
PAUSE = False

def request_pause(*args):
    global PAUSE
    PAUSE = True

def pause_boundary(p, events, records):
    if PAUSE:
        write(BASE/'run'/'INDEPENDENT_PARTIAL.json',{'accepted':False,'complete_replay':False,
              'protocol_sha256':sha(BASE/'EXECUTION_PROTOCOL.json'),'checks_passed':CHECKS,
              'events':events,'completed_records':records,'pid':os.getpid(),
              'deadline_epoch':p['deadline_epoch'],'source_or_numeric_result_not_accepted':True})
        raise SystemExit(75)

def verify(ok, label):
    global CHECKS
    CHECKS += 1
    require(ok, 'Independent replay: ' + label)

def index_hash(a):
    import numpy as np
    return hashlib.sha256(np.asarray(a, dtype='<i8').tobytes()).hexdigest()

def decode(path, expected, members, selected):
    import numpy as np
    verify(len(selected)==len(set(selected)) and set(selected)<={'fit','start_ns','end_ns','observed','session_date'}, 'member whitelist')
    verify(sha(path)==expected, 'container hash')
    with zipfile.ZipFile(path) as z:
        verify(sorted(z.namelist())==sorted(members), 'complete central directory')
        return {k:np.load(io.BytesIO(z.read(k+'.npy')),allow_pickle=False) for k in selected}

def calendar_rows(spec):
    import exchange_calendars as xc
    c=spec['calendar']; cal=xc.get_calendar('XNYS',start=c['start'],end=c['end'])
    verify(c['name']=='XNYS', 'calendar name')
    return [{'session_date':str(day.date()),
             'open_ns':int(cal.session_open(day).as_unit('ns').value),
             'close_ns':int(cal.session_close(day).as_unit('ns').value)}
            for day in cal.sessions_in_range(c['start'],c['end'])]

def full_grid(bars, days, entry, profile):
    import numpy as np
    verify(set(bars)=={'start_ns','end_ns','observed','session_date'}, 'four non-price fields only')
    n=entry['rows']
    verify(all(a.shape==(n,) for a in bars.values()), 'bar shapes')
    verify(bars['start_ns'].dtype==bars['end_ns'].dtype==np.dtype('int64'), 'bar time dtypes')
    verify(bars['observed'].dtype==np.dtype('bool') and bars['session_date'].dtype.kind=='U', 'bar identity dtypes')
    step=MINUTE if profile=='minute' else 60*MINUTE
    cursor=0
    for day in days:
        opening,closing=day['open_ns'],day['close_ns']
        verify(closing>opening and (closing-opening)%MINUTE==0, 'session length')
        count=(closing-opening)//step
        starts=np.arange(count,dtype=np.int64)*step+opening
        sl=slice(cursor,cursor+count)
        verify(np.array_equal(bars['start_ns'][sl],starts), 'complete scheduled starts')
        verify(np.array_equal(bars['end_ns'][sl],starts+step), 'complete scheduled ends')
        verify(bool(np.all(bars['session_date'][sl]==day['session_date'])), 'complete scheduled dates')
        cursor+=count
    verify(cursor==n and str(bars['session_date'][0])=='2021-01-04' and str(bars['session_date'][-1])=='2026-06-30', 'complete preQ3 extent')

def instant_ns(value):
    d=dt.datetime.fromisoformat(value).astimezone(dt.timezone.utc)-dt.datetime(1970,1,1,tzinfo=dt.timezone.utc)
    return ((d.days*86400+d.seconds)*1_000_000+d.microseconds)*1000

def independent_pair(a,b,precision):
    missing=[role+'_unqualified' for role,f in [('current',a),('prior',b)]
             if f is None or f['reported_quarter_diluted_EPS_decimal_string'] is None]
    if missing:return {'valid':False,'missing_reasons':missing,'ratio':None,'z':None,'sign':None}
    verify(a['currency']==b['currency']=='USD_assumed_from_US_issuer_reporting','pair currency assumption')
    x=Fraction(a['reported_quarter_diluted_EPS_decimal_string'])*Fraction(a['separate_canonical_share_unit_factor'])
    y=Fraction(b['reported_quarter_diluted_EPS_decimal_string'])*Fraction(b['separate_canonical_share_unit_factor'])
    if x<=0 or y<=0:return {'valid':False,'missing_reasons':['nonpositive_EPS'],'ratio':None,'z':None,'sign':None}
    ratio=x/y
    # A different expression and higher precision than main's rounded quotient ln.
    with localcontext() as c:
        c.prec=precision+20
        z=Decimal(ratio.numerator).ln()-Decimal(ratio.denominator).ln()
    return {'valid':True,'missing_reasons':[],'ratio':str(ratio),'z':float(z),
            'sign':1 if ratio>1 else (-1 if ratio<1 else 0)}

def source_events(spec,days):
    facts=read(spec['facts'])['facts']; roster=read(spec['roster'])['events']; unit=read(spec['unit_source'])
    verify(len(facts)==len(roster)==26,'26 source records')
    byid={r['id']:r for r in roster}; quarters={(f['fiscal_year'],f['fiscal_quarter']):f for f in facts}
    verify(len(byid)==len(quarters)==26,'unique source identities')
    for f in facts:
        r=byid[f['id']]
        day=dt.date.fromisoformat(r['financial_filing']['filingDate'])+dt.timedelta(days=1)
        known=dt.datetime.combine(day,dt.time(),ZoneInfo('America/New_York'))
        verify(known.isoformat()==r['known_time_assumption_NY']==f['known_time_assumption_NY'],'fixed filing known time')
        verify(known.astimezone(dt.timezone.utc).isoformat()==r['known_time_assumption_UTC'],'UTC known time')
        verify(f['SEC_8K_accession']==r['8K']['accessionNumber'] and f['SEC_financial_form_accession']==r['financial_filing']['accessionNumber'],'SEC identities')
        verify(f['fiscal_year']==r['expected_fiscal_year_from_event'] and f['fiscal_quarter']==r['expected_quarter_for_URL_only'] and f['reported_period_end']==r['financial_filing']['reportDate'],'source FY quarter period')
        verify(sha(f['source_file'])==f['source_body_sha256'],'official raw body hash')
        verify(f['separate_canonical_share_unit_factor']==('0.25' if f['id'] in ['2020-01-28','2020-04-30','2020-07-30'] else '1'),'share-unit metadata')
    result=[]
    for a in facts:
        if a['role']!='development_event':continue
        b=quarters.get((a['fiscal_year']-1,a['fiscal_quarter']))
        verify(b is not None,'previous FY same Q present')
        known=max(dt.datetime.fromisoformat(a['known_time_assumption_NY']),dt.datetime.fromisoformat(b['known_time_assumption_NY']),dt.datetime.fromisoformat(unit['known_time_new_york']))
        known_ns=instant_ns(known.isoformat())
        session=next((d for d in days if d['open_ns']>=known_ns),None)
        verify(session is not None,'first not-opened eligible session')
        event={'id':a['id'],'fiscal_year':a['fiscal_year'],'fiscal_quarter':a['fiscal_quarter'],
               'prior_id':b['id'],'current_raw_EPS':a['reported_quarter_diluted_EPS_decimal_string'],
               'prior_raw_EPS':b['reported_quarter_diluted_EPS_decimal_string'],
               'current_unit_factor':a['separate_canonical_share_unit_factor'],'prior_unit_factor':b['separate_canonical_share_unit_factor'],
               'current_known':a['known_time_assumption_NY'],'prior_known':b['known_time_assumption_NY'],
               'unit_known':unit['known_time_new_york'],'effective_known_UTC':known.astimezone(dt.timezone.utc).isoformat(),
               'current_duration_weeks':a['duration_weeks'],'prior_duration_weeks':b['duration_weeks'],**session}
        event.update(independent_pair(a,b,spec['decimal_precision']));result.append(event)
    verify(len(result)==22 and len({e['session_date'] for e in result})==22,'22 distinct event sessions')
    return result

def exact_rank(rows):
    values={Fraction(e['ratio']) for e in rows}
    if not values:return 0
    return 3 if min(values)<1<max(values) and len(values)>=3 else min(2,len(values))

def ranks(rows,spec):
    import numpy as np
    matrix=[[1.,abs(e['z']),e['z']] for e in rows]
    singular=np.linalg.svd(np.asarray(matrix,dtype=np.float64).reshape((-1,3)),compute_uv=False)
    tol=max(spec['rank_absolute_tolerance'],spec['rank_relative_tolerance']*float(singular[0]) if len(singular) else 0.)
    structural=exact_rank(rows); numeric=sum(float(s)>tol for s in singular)
    # Scalar fsum is an additional independent arithmetic record, not a fit.
    gram=[[math.fsum(row[j]*row[k] for row in matrix) for k in range(3)] for j in range(3)]
    return {'event_count':len(rows),'positive':sum(e['sign']>0 for e in rows),'negative':sum(e['sign']<0 for e in rows),
            'zero':sum(e['sign']==0 for e in rows),'structural_rank':structural,'numeric_rank':numeric,
            'singular_values':[float(s) for s in singular],'rank_tolerance':tol,
            'numeric_structural_agree':structural==numeric,'independent_fsum_Gram':gram}

def phase(indices,bars,events,spec):
    times=[int(bars['end_ns'][i]) for i in indices]; members=[]; valid=[]; missing=[]
    for e in events:
        count=bisect.bisect_right(times,e['close_ns'])-bisect.bisect_right(times,e['open_ns'])
        if count:
            members.append({'event_id':e['id'],'origin_count':count,'valid':e['valid'],'sign':e['sign']})
            (valid if e['valid'] else missing).append(e)
    loo=[{'deleted_event_id':e['id'],**ranks([x for x in valid if x['id']!=e['id']],spec)} for e in valid]
    n=sum(m['origin_count'] for m in members)
    return {'original_origin_count':len(indices),'origin_sha256':index_hash(indices),'first_decision_ns':times[0],'last_decision_ns':times[-1],
            'event_membership':members,'independent_event_count':len(members),'valid_event_ids':[e['id'] for e in valid],
            'missing_event_ids':[e['id'] for e in missing],'event_origin_count':n,'non_event_origin_count':len(indices)-n,
            'valid_event_origin_count':sum(m['origin_count'] for m in members if m['valid']),
            'missing_event_origin_count':sum(m['origin_count'] for m in members if not m['valid']),
            'rank':ranks(valid,spec),'leave_one_event_out':loo,
            'minimum_leave_one_out_structural_rank':min((x['structural_rank'] for x in loo),default=None),
            'minimum_leave_one_out_numeric_rank':min((x['numeric_rank'] for x in loo),default=None),
            'retains_rank3_after_any_single_event_deletion':bool(valid) and all(x['structural_rank']==x['numeric_rank']==3 for x in loo)}

def same(actual,expected,label):
    if isinstance(expected,dict):
        for k,v in expected.items():
            if k=='independent_fsum_Gram':continue
            verify(k in actual,label+' key '+k);same(actual[k],v,label+'.'+k)
    elif isinstance(expected,list):
        verify(len(actual)==len(expected),label+' length')
        for i,(a,b) in enumerate(zip(actual,expected)):same(a,b,label+'['+str(i)+']')
    elif isinstance(expected,float):
        verify(isinstance(actual,(int,float)) and math.isfinite(actual) and math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-14),label+' numeric')
    else:verify(actual==expected,label+' value')

def replay_state(entry,bars,events,spec):
    import numpy as np
    f=entry['fit']; fit=decode(f['path'],f['file_sha256'],spec['fit_members'],['fit'])['fit']; h=entry['horizon']
    verify(fit.dtype==np.dtype('int64') and fit.shape==(f['count'],) and np.all(np.diff(fit)>0),'original fit type/order')
    verify(index_hash(fit)==f['fit_sha256'],'original fit payload hash')
    verify(int(fit[0])>=0 and int(fit[-1])+h<len(bars['end_ns']),'fit index limits')
    verify(all(bool(np.all(bars['observed'][fit+j])) for j in range(h+1)),'original complete observed path identity only')
    if entry['profile']=='minute':
        verify(np.all(bars['session_date'][fit]==bars['session_date'][fit+h]) and np.all(bars['end_ns'][fit+h]-bars['end_ns'][fit]==h*MINUTE),'minute original session/continuity')
    boundary=(len(fit)*4)//5; tail=fit[boundary:]
    # Independent scalar membership construction, not main's vector mask.
    cutoff=int(bars['end_ns'][int(tail[0])])
    prefix=np.asarray([int(i) for i in fit[:boundary] if int(bars['end_ns'][int(i)+h])<cutoff],dtype=np.int64)
    partition={'id':entry['id'],'original_fit_count':len(fit),'boundary':boundary,'prefix_count':len(prefix),'tail_count':len(tail),'first_tail_origin':int(tail[0]),'prefix_sha256':index_hash(prefix),'tail_sha256':index_hash(tail)}
    verify(partition==entry['expected_partition'],'canonical strict purged partition')
    phases={k:phase(x,bars,events,spec) for k,x in [('prefix',prefix),('tail',tail)]}
    a={m['event_id'] for m in phases['prefix']['event_membership']};b={m['event_id'] for m in phases['tail']['event_membership']}
    return {'id':entry['id'],'profile':entry['profile'],'partition':partition,**phases,
            'within_state_prefix_tail_event_overlap':sorted(a&b),'tail_exclusive_event_ids':sorted(b-a),
            'tail_exclusive_valid_event_ids':sorted(set(phases['tail']['valid_event_ids'])-a),'decoded_fit_members':['fit.npy']}

def replay_decision(records):
    conditions=[]
    for r in records:
        p,t=r['prefix'],r['tail']
        tests=[('prefix_rank3',p['rank']['structural_rank']==p['rank']['numeric_rank']==3),
               ('leave_one_event_out_rank3',p['retains_rank3_after_any_single_event_deletion']),
               ('tail_has_valid_event',bool(t['valid_event_ids'])),
               ('no_within_state_event_overlap',not r['within_state_prefix_tail_event_overlap'])]
        conditions.extend({'state':r['id'],'condition':k,'pass':v} for k,v in tests)
    cross=[{'prefix_state':a['id'],'tail_state':b['id'],'shared_valid_event_ids':sorted(set(a['prefix']['valid_event_ids'])&set(b['tail']['valid_event_ids']))} for a in records for b in records]
    passed=all(c['pass'] for c in conditions)
    return {'necessary_conditions':conditions,'passed_count':sum(c['pass'] for c in conditions),'condition_count':16,
            'necessary_support_conditions_passed':passed,'may_propose_separately_frozen_small_nested_information_test':passed,
            'cross_state_valid_event_exposure':cross,'model_fit_admitted':False,'candidate_admitted':False,
            'all_earnings_information_useless_conclusion':False,'predictive_result':False,
            'independent_market_confirmation':False,'full_research_goal_achieved':False}

def main():
    setup(26); signal.signal(signal.SIGUSR1,request_pause); signal.signal(signal.SIGTERM,request_pause)
    p=protocol(); spec=read(BASE/'DESIGN_SPEC.json'); out=BASE/'run'
    dependencies=qualify_dependencies(spec)
    for name,h in spec['source_files'].items():verify(sha(name)==h,'source lock '+name)
    days=calendar_rows(spec); events=source_events(spec,days)
    pause_boundary(p,events,[])
    saved_events=read(out/'EVENTS.json')
    verify(saved_events['not_consensus_surprise'] is True and saved_events['PIT_original_version_proved'] is False,'source scope claims')
    same(saved_events['events'],events,'22 events')
    facts={f['id']:f for f in read(spec['facts'])['facts']}
    for actual in saved_events['events']:
        if actual['valid']:
            a,b=facts[actual['id']],facts[actual['prior_id']]
            verify(Fraction(actual['current_aligned_EPS'])==Fraction(a['reported_quarter_diluted_EPS_decimal_string'])*Fraction(a['separate_canonical_share_unit_factor']),'current aligned decimal')
            verify(Fraction(actual['prior_aligned_EPS'])==Fraction(b['reported_quarter_diluted_EPS_decimal_string'])*Fraction(b['separate_canonical_share_unit_factor']),'prior aligned decimal')
    cache={}; records=[]
    for entry in spec['states']:
        guard(p,5);profile=entry['profile']
        if profile not in cache:
            b=spec['bars'][profile]; cache[profile]=decode(b['path'],b['sha256'],spec['bar_members'],spec['decode_bar_members'])
            full_grid(cache[profile],days,b,profile)
        record=replay_state(entry,cache[profile],events,spec)
        same(read(out/(entry['id']+'.json')),record,entry['id']);records.append(record)
        pause_boundary(p,events,records)
    decision=replay_decision(records);same(read(out/'DECISION.json'),decision,'16 conditions and cross-state claims')
    result=read(out/'RESULTS.json')
    same(result['records'],records,'aggregate four-state outputs')
    same(result,{ 'events_total':22,'valid_events':sum(e['valid'] for e in events),
                 'missing_events':[e['id'] for e in events if not e['valid']],
                 'price_return_label_members_decoded':False,'whole_container_hash_includes_compressed_price_member_bytes':True,
                 'per_origin_forecast_matrix_constructed':False,
                 'model_fits':0,'GPU':False,'Q3_read':False},'result scope')
    all_ids=lambda r,phase:{x['event_id'] for x in r[phase]['event_membership']}
    cross_all=[{'prefix_state':a['id'],'tail_state':b['id'],
                'shared_all_event_ids':sorted(all_ids(a,'prefix')&all_ids(b,'tail'))}
               for a in records for b in records]
    exposure=[]
    for profile in ['minute','hourly']:
        first=next(r for r in records if r['id']==profile+'_f1')
        second=next(r for r in records if r['id']==profile+'_f2')
        exposure.append({'profile':profile,
          'fold1_tail_vs_fold2_prefix_shared_all_event_ids':sorted(all_ids(first,'tail')&all_ids(second,'prefix')),
          'fold1_tail_vs_fold2_prefix_shared_valid_event_ids':sorted(set(first['tail']['valid_event_ids'])&set(second['prefix']['valid_event_ids']))})
    pause_boundary(p,events,records);guard(p)
    write(out/'INDEPENDENT_REPLAY.json',{'accepted':True,'checks_passed':CHECKS,'own_Fraction_Decimal_calendar_bisect_partition_rank_LOO':True,
          'shared_numpy_calendar_dependency_and_runtime_helpers':True,'events':events,'records':records,'decision':decision,
          'same_task_cross_fold_exposure':exposure,'cross_state_all_event_exposure':cross_all,'dependencies':dependencies,
          'protocol_sha256':sha(BASE/'EXECUTION_PROTOCOL.json'),'input_output_hashes':{str(out/n):sha(out/n) for n in ['EVENTS.json','RESULTS.json','DECISION.json']},
          'elapsed_from_frozen_start_seconds':time.time()-p['started_epoch'],'deadline_epoch':p['deadline_epoch'],
          'CPU_affinity':sorted(os.sched_getaffinity(0)),'decoded_members':['fit.npy','start_ns.npy','end_ns.npy','observed.npy','session_date.npy'],
          'price_label_prediction_weight_Q3_members_decoded':False,
          'whole_container_hash_includes_compressed_price_member_bytes':True,
          'HTTP_requests':0,'model_fits':0,'GPU':False,'predictive_result':False,'Goal_success':False})

if __name__=='__main__':
    try:main()
    except Exception as e:
        write(BASE/'run'/'INDEPENDENT_FAILURE.json',{'error':repr(e),'accepted':False,'pid':os.getpid()})
        raise
