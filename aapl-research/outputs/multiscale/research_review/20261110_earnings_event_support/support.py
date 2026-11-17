"""Fixed source/calendar/fit-membership diagnostic. No price or target reads."""
import datetime as dt
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import io
import os
import signal
import sys
import time
import zipfile
from zoneinfo import ZoneInfo
from runtime import BASE, require, sha, read, write, setup, guard, protocol, qualify_dependencies

MINUTE=60_000_000_000
PAUSE=False

def request_pause(*args):
    global PAUSE
    PAUSE=True

def digest(a):
    import numpy as np
    return hashlib.sha256(np.asarray(a,dtype='<i8').tobytes()).hexdigest()

def selected(path,expected,members,keys):
    import numpy as np
    require(set(keys)<= {'start_ns','end_ns','session_date','observed','fit'},'Forbidden member')
    require(sha(path)==expected,'Container hash drift')
    with zipfile.ZipFile(path) as z:
        require(sorted(z.namelist())==sorted(members),'Unexpected or duplicate ZIP member')
        return {k:np.load(io.BytesIO(z.read(k+'.npy')),allow_pickle=False) for k in keys}

def schedule(spec):
    import exchange_calendars as xc
    c=spec['calendar']
    return xc.get_calendar(c['name'],start=c['start'],end=c['end']).schedule.loc[c['start']:c['end']]

def ns(value):
    import pandas as pd
    return int(pd.Timestamp(value).as_unit('ns').value)

def next_session(sch,known):
    # No session is active before all three source/unit knowledge boundaries.
    eligible=sch.loc[sch['open'] >= known]
    require(len(eligible)>0,'No eligible planned session')
    day,row=next(iter(eligible.iterrows()))
    return str(day.date()),ns(row['open']),ns(row['close'])

def qualify_grid(a,sch,entry,freq):
    import numpy as np
    starts=[]; ends=[]; dates=[]
    step=MINUTE if freq=='minute' else 60*MINUTE
    for day,row in sch.iterrows():
        op,cl=ns(row['open']),ns(row['close'])
        count=(cl-op)//step
        require(count>0 and (cl-op)%MINUTE==0,'Calendar length')
        s=op+np.arange(count,dtype=np.int64)*step
        starts.append(s);ends.append(s+step)
        dates.append(np.full(count,day.strftime('%Y-%m-%d'),dtype='<U10'))
    for key,parts in [('start_ns',starts),('end_ns',ends),('session_date',dates)]:
        require(a[key].shape==(entry['rows'],) and np.array_equal(a[key],np.concatenate(parts)),'Full planned grid '+key)
    require(a['start_ns'].dtype==a['end_ns'].dtype==np.dtype('int64'),'Clock dtype')
    require(a['observed'].shape==(entry['rows'],) and a['observed'].dtype==np.dtype('bool'),'Observed identity')
    require(a['session_date'].dtype.kind=='U','Session dtype')
    require(str(a['session_date'][0])=='2021-01-04' and str(a['session_date'][-1])=='2026-06-30','PreQ3 span')
    return {'rows':entry['rows'],'full_grid_equal':True,'observed_count':int(a['observed'].sum()),'decoded_members':['start_ns.npy','end_ns.npy','observed.npy','session_date.npy'],'close_decoded':False}

def known_from_filing(roster):
    day=dt.date.fromisoformat(roster['financial_filing']['filingDate'])+dt.timedelta(days=1)
    return dt.datetime.combine(day,dt.time(),ZoneInfo('America/New_York')).isoformat()

def pair_value(current,prior,precision):
    missing=[]
    for role,f in [('current',current),('prior',prior)]:
        if f is None or f['reported_quarter_diluted_EPS_decimal_string'] is None:
            missing.append(role+'_unqualified')
    if missing:return {'valid':False,'missing_reasons':missing,'z':None,'sign':None,'ratio':None}
    require(current['currency']==prior['currency']=='USD_assumed_from_US_issuer_reporting','Currency mismatch')
    with localcontext() as c:
        c.prec=precision
        a=Decimal(current['reported_quarter_diluted_EPS_decimal_string'])*Decimal(current['separate_canonical_share_unit_factor'])
        b=Decimal(prior['reported_quarter_diluted_EPS_decimal_string'])*Decimal(prior['separate_canonical_share_unit_factor'])
        if a<=0 or b<=0:return {'valid':False,'missing_reasons':['nonpositive_EPS'],'z':None,'sign':None,'ratio':None}
        ratio=Fraction(a)/Fraction(b)
        z=(a/b).ln()
        sign=(a>b)-(a<b)
        return {'valid':True,'missing_reasons':[],'current_aligned_EPS':str(a),'prior_aligned_EPS':str(b),'ratio':str(ratio),'z_decimal':str(z),'z':float(z),'sign':sign}

def events(spec,sch):
    import pandas as pd
    facts=read(spec['facts'])['facts'];roster=read(spec['roster'])['events'];unit=read(spec['unit_source'])
    require(len(facts)==len(roster)==26,'Fixed source roster count')
    byid={e['id']:e for e in roster};byq={(f['fiscal_year'],f['fiscal_quarter']):f for f in facts}
    require(len(byid)==len(byq)==26,'Unique source identity')
    for f in facts:
        r=byid[f['id']]
        require(f['known_time_assumption_NY']==r['known_time_assumption_NY']==known_from_filing(r),'Frozen filing-day known time')
        require(f['SEC_financial_form_accession']==r['financial_filing']['accessionNumber'] and f['SEC_8K_accession']==r['8K']['accessionNumber'],'SEC source identity')
        require(f['reported_period_end']==r['financial_filing']['reportDate'],'Fiscal period identity')
        require(f['fiscal_year']==r['expected_fiscal_year_from_event'] and f['fiscal_quarter']==r['expected_quarter_for_URL_only'],'FYQ identity')
        require(sha(f['source_file'])==f['source_body_sha256'],'Official body identity')
        expected='0.25' if f['id'] in ['2020-01-28','2020-04-30','2020-07-30'] else '1'
        require(f['separate_canonical_share_unit_factor']==expected,'Exact unit factor')
    out=[]
    for f in facts:
        if f['role']!='development_event':continue
        prior=byq.get((f['fiscal_year']-1,f['fiscal_quarter']))
        require(prior is not None,'Fixed prior same-quarter record absent')
        known=max(pd.Timestamp(f['known_time_assumption_NY']),pd.Timestamp(prior['known_time_assumption_NY']),pd.Timestamp(unit['known_time_new_york']))
        day,op,cl=next_session(sch,known)
        record={'id':f['id'],'fiscal_year':f['fiscal_year'],'fiscal_quarter':f['fiscal_quarter'],'prior_id':prior['id'],'current_raw_EPS':f['reported_quarter_diluted_EPS_decimal_string'],'prior_raw_EPS':prior['reported_quarter_diluted_EPS_decimal_string'],'current_unit_factor':f['separate_canonical_share_unit_factor'],'prior_unit_factor':prior['separate_canonical_share_unit_factor'],'current_known':f['known_time_assumption_NY'],'prior_known':prior['known_time_assumption_NY'],'unit_known':unit['known_time_new_york'],'effective_known_UTC':known.tz_convert('UTC').isoformat(),'session_date':day,'open_ns':op,'close_ns':cl,'current_duration_weeks':f['duration_weeks'],'prior_duration_weeks':prior['duration_weeks']}
        record.update(pair_value(f,prior,spec['decimal_precision']));out.append(record)
    require(len(out)==22,'All development events retained')
    require(len({e['session_date'] for e in out})==22,'No conflicting active event sessions')
    return out

def structural_rank(rows):
    if not rows:return 0
    ratios={Fraction(r['ratio']) for r in rows}
    pos={r for r in ratios if r>1};neg={r for r in ratios if r<1};zero=Fraction(1) in ratios
    if pos and neg and (zero or len(pos)>1 or len(neg)>1):return 3
    return min(2,len(ratios))

def rank_report(rows,spec):
    import numpy as np
    s=np.linalg.svd(np.array([[1.,abs(r['z']),r['z']] for r in rows],dtype=np.float64).reshape((-1,3)),compute_uv=False)
    tolerance=max(spec['rank_absolute_tolerance'],spec['rank_relative_tolerance']*float(s[0]) if len(s) else 0.)
    nr=int(np.count_nonzero(s>tolerance));sr=structural_rank(rows)
    return {'event_count':len(rows),'positive':sum(r['sign']==1 for r in rows),'negative':sum(r['sign']==-1 for r in rows),'zero':sum(r['sign']==0 for r in rows),'structural_rank':sr,'numeric_rank':nr,'singular_values':[float(x) for x in s],'rank_tolerance':tolerance,'numeric_structural_agree':nr==sr}

def partition(fit,end,h):
    boundary=4*len(fit)//5;tail=fit[boundary:];before=fit[:boundary]
    return boundary,before[end[before+h]<end[tail[0]]],tail

def phase_report(indices,bars,all_events,spec):
    import numpy as np
    decisions=bars['end_ns'][indices];counts=[];valid=[];unknown=[]
    for e in all_events:
        count=int(np.count_nonzero((decisions>e['open_ns'])&(decisions<=e['close_ns'])))
        if count:
            counts.append({'event_id':e['id'],'origin_count':count,'valid':e['valid'],'sign':e['sign']})
            (valid if e['valid'] else unknown).append(e)
    rr=rank_report(valid,spec)
    deleted=[{'deleted_event_id':e['id'],**rank_report(valid[:i]+valid[i+1:],spec)} for i,e in enumerate(valid)]
    return {'original_origin_count':len(indices),'origin_sha256':digest(indices),'first_decision_ns':int(decisions[0]),'last_decision_ns':int(decisions[-1]),'event_membership':counts,'independent_event_count':len(counts),'valid_event_ids':[e['id'] for e in valid],'missing_event_ids':[e['id'] for e in unknown],'non_event_origin_count':len(indices)-sum(x['origin_count'] for x in counts),'event_origin_count':sum(x['origin_count'] for x in counts),'valid_event_origin_count':sum(x['origin_count'] for x in counts if x['valid']),'missing_event_origin_count':sum(x['origin_count'] for x in counts if not x['valid']),'rank':rr,'leave_one_event_out':deleted,'minimum_leave_one_out_structural_rank':min((x['structural_rank'] for x in deleted),default=None),'minimum_leave_one_out_numeric_rank':min((x['numeric_rank'] for x in deleted),default=None),'retains_rank3_after_any_single_event_deletion':bool(valid) and all(x['structural_rank']==x['numeric_rank']==3 for x in deleted)}

def state(entry,bars,all_events,spec):
    import numpy as np
    f=entry['fit']; fit=selected(f['path'],f['file_sha256'],spec['fit_members'],['fit'])['fit'];h=entry['horizon']
    require(fit.dtype==np.dtype('int64') and fit.shape==(f['count'],) and np.all(np.diff(fit)>0),'Original fit order')
    require(digest(fit)==f['fit_sha256'],'Original fit identity')
    require(fit.min()>=0 and fit.max()+h<len(bars['end_ns']),'Fit/target index range')
    # Identity only: do not drop or mask any origin using future observed fields.
    require(bars['observed'][fit].all(),'Original observed origins')
    for j in range(1,h+1):require(bars['observed'][fit+j].all(),'Original complete target-clock identity')
    if entry['profile']=='minute':
        require(np.all(bars['session_date'][fit]==bars['session_date'][fit+h]),'Original minute session identity')
        require(np.all(bars['end_ns'][fit+h]-bars['end_ns'][fit]==h*MINUTE),'Original minute clock continuity')
    boundary,prefix,tail=partition(fit,bars['end_ns'],h)
    actual={'id':entry['id'],'original_fit_count':len(fit),'boundary':boundary,'prefix_count':len(prefix),'tail_count':len(tail),'first_tail_origin':int(tail[0]),'prefix_sha256':digest(prefix),'tail_sha256':digest(tail)}
    require(actual==entry['expected_partition'],'Canonical original partitions differ')
    p=phase_report(prefix,bars,all_events,spec);t=phase_report(tail,bars,all_events,spec)
    ids=lambda phase:{r['event_id'] for r in phase['event_membership']}
    overlap=sorted(ids(p)&ids(t))
    return {'id':entry['id'],'profile':entry['profile'],'partition':actual,'prefix':p,'tail':t,'within_state_prefix_tail_event_overlap':overlap,'tail_exclusive_event_ids':sorted(ids(t)-ids(p)),'tail_exclusive_valid_event_ids':sorted(set(t['valid_event_ids'])-ids(p)),'decoded_fit_members':['fit.npy']}

def decision(records):
    conditions=[]
    for r in records:
        p=r['prefix'];t=r['tail']
        conditions.extend([{'state':r['id'],'condition':'prefix_rank3','pass':p['rank']['structural_rank']==p['rank']['numeric_rank']==3},{'state':r['id'],'condition':'leave_one_event_out_rank3','pass':p['retains_rank3_after_any_single_event_deletion']},{'state':r['id'],'condition':'tail_has_valid_event','pass':len(t['valid_event_ids'])>0},{'state':r['id'],'condition':'no_within_state_event_overlap','pass':not r['within_state_prefix_tail_event_overlap']}])
    cross=[]
    for p in records:
        for t in records:
            a=set(p['prefix']['valid_event_ids']);b=set(t['tail']['valid_event_ids'])
            cross.append({'prefix_state':p['id'],'tail_state':t['id'],'shared_valid_event_ids':sorted(a&b)})
    passed=all(x['pass'] for x in conditions)
    return {'necessary_conditions':conditions,'passed_count':sum(x['pass'] for x in conditions),'condition_count':len(conditions),'necessary_support_conditions_passed':passed,'may_propose_separately_frozen_small_nested_information_test':passed,'model_fit_admitted':False,'candidate_admitted':False,'cross_state_valid_event_exposure':cross,'all_earnings_information_useless_conclusion':False,'predictive_result':False,'independent_market_confirmation':False,'full_research_goal_achieved':False}

def main():
    setup(24);signal.signal(signal.SIGUSR1,request_pause);signal.signal(signal.SIGTERM,request_pause)
    p=protocol();spec=read(BASE/'DESIGN_SPEC.json');out=BASE/'run'
    deps=qualify_dependencies(spec);sch=schedule(spec);guard(p,15)
    cp=out/'CHECKPOINT.json';identity=sha(BASE/'EXECUTION_PROTOCOL.json')
    saved=read(cp) if cp.exists() else {'protocol_sha256':identity,'completed_states':[],'worker_PIDs':[]}
    require(saved['protocol_sha256']==identity,'Checkpoint protocol differs')
    for done in saved['completed_states']:require(sha(out/(done['id']+'.json'))==done['sha256'],'Committed state drift')
    require([x['id'] for x in saved['completed_states']]==[e['id'] for e in spec['states'][:len(saved['completed_states'])]],'Checkpoint state sequence')
    saved['worker_PIDs'].append(os.getpid());write(cp,saved)
    event_rows=events(spec,sch);write(out/'EVENTS.json',{'events':event_rows,'source_assumptions':read(spec['facts'])['source_assumptions'],'not_consensus_surprise':True,'PIT_original_version_proved':False})
    write(out/'DEPENDENCY_RECEIPT.json',deps)
    cache={}
    for entry in spec['states'][len(saved['completed_states']):]:
        guard(p,15);freq=entry['profile']
        if freq not in cache:
            b=spec['bars'][freq];arrays=selected(b['path'],b['sha256'],spec['bar_members'],spec['decode_bar_members'])
            receipt=qualify_grid(arrays,sch,b,freq);cache[freq]=(arrays,receipt)
        arrays,receipt=cache[freq];r=state(entry,arrays,event_rows,spec);r['bar_identity']=receipt
        write(out/(entry['id']+'.json'),r);saved['completed_states'].append({'id':entry['id'],'sha256':sha(out/(entry['id']+'.json'))});write(cp,saved)
        if PAUSE:
            write(out/'PAUSED.json',{'pid':os.getpid(),'checkpoint_sha256':sha(cp),'deadline_epoch':p['deadline_epoch']});return 75
    records=[read(out/(e['id']+'.json')) for e in spec['states']]
    write(out/'RESULTS.json',{'records':records,'events_total':len(event_rows),'valid_events':sum(e['valid'] for e in event_rows),'missing_events':[e['id'] for e in event_rows if not e['valid']],'price_return_label_members_decoded':False,'whole_container_hash_includes_compressed_price_member_bytes':True,'per_origin_forecast_matrix_constructed':False,'model_fits':0,'GPU':False,'Q3_read':False})
    write(out/'DECISION.json',decision(records));guard(p)
    write(out/'DONE.json',{'pid':os.getpid(),'completed_states':4,'elapsed_from_frozen_start_seconds':time.time()-p['started_epoch'],'deadline_epoch':p['deadline_epoch']})
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:
        write(BASE/'run'/'FAILURE.json',{'pid':os.getpid(),'error':repr(e),'incomplete_science_not_accepted':True});raise
