"""Independent clock-only result review. Imports no coverage core/worker/source code."""
from __future__ import annotations
import argparse
import calendar
from datetime import date, datetime, time as day_time, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import time
import zipfile
from zoneinfo import ZoneInfo
import numpy as np

BASE=Path(__file__).resolve().parent
MINUTE=60_000_000_000
CUTOFF=calendar.timegm((2026,7,1,4,0,0))*1_000_000_000
NY=ZoneInfo('America/New_York')
BAR_MEMBERS=('start_ns','end_ns','observed','session_date')
PARTITIONS=('full_fit','purged_prefix','unchanged_tail')
STAMP=re.compile(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:00(?:\.0{1,9})?(?:Z|\+00:00)')

class ReviewError(Exception):
    pass


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path,guard=lambda:None):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            guard(); block=stream.read(65536)
            if not block:
                return h.hexdigest()
            h.update(block)


def atomic(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    tmp.replace(path)


def frozen_table(refs):
    result={}
    for ref in refs:
        key=str(Path(ref['path']).resolve())
        if key in result or not re.fullmatch('[0-9a-f]{64}',str(ref['sha256'])):
            raise ReviewError('duplicate_or_invalid_input_lock')
        result[key]=ref['sha256']
    return result


def locked_json(path,table,guard):
    key=str(Path(path).resolve()); guard()
    if key not in table:
        raise ReviewError('metadata_not_locked')
    payload=Path(path).read_bytes()
    if sha(payload)!=table[key]:
        raise ReviewError('metadata_identity_mismatch')
    return json.loads(payload)


def selected_npz(path,names,audit,guard,table=None):
    allowed_sets=(set(BAR_MEMBERS),{'fit'},{p+suffix for p in PARTITIONS for suffix in ('_origins','_complete_history')})
    if set(names) not in allowed_sets or len(names)!=len(set(names)):
        raise ReviewError('member_selection_outside_clock_fit_mask_allowlist')
    if table is not None:
        key=str(Path(path).resolve())
        if key not in table or file_sha(path,guard)!=table[key]:
            raise ReviewError('archive_not_frozen')
    result={}
    with zipfile.ZipFile(path) as archive:
        listing=archive.namelist()
        for name in names:
            guard(); member=name+'.npy'
            if listing.count(member)!=1:
                raise ReviewError('selected_member_missing_or_duplicated')
            with archive.open(member) as stream:
                result[name]=np.lib.format.read_array(stream,allow_pickle=False)
            audit.append({'path':str(Path(path).resolve()),'member':member})
    return result


def fixed_ranges():
    ranges=[]; first=date(2021,1,4); last=date(2026,6,30)
    while first<=last:
        end=min(last,first+timedelta(days=29)); ranges.append([first.isoformat(),end.isoformat()])
        first=end+timedelta(days=1)
    return ranges


def stamp_ns(text):
    if not isinstance(text,str) or not STAMP.fullmatch(text):
        raise ReviewError('source_minute_stamp_invalid')
    stamp=datetime.fromisoformat(text.replace('Z','+00:00'))
    return calendar.timegm(stamp.utctimetuple())*1_000_000_000


def independent_source(index,protocol,guard):
    ranges=fixed_ranges()
    if (index['kind']!='SPY_qualified_canonical_index_v1' or index['all_67_ranges_complete'] is not True
            or index['ranges']!=ranges or index['Q3_allowed'] is not False
            or index['source_only_not_RTH_or_context_admission'] is not True
            or index['protocol_sha256']!=protocol['source_protocol_sha256']
            or index['inheritance_manifest_sha256']!=protocol['inheritance_manifest_sha256']):
        raise ReviewError('source_index_identity_invalid')
    pages=index['pages']
    if (not 67<=len(pages)<=298 or any(type(p['range']) is not int or not 0<=p['range']<67 for p in pages)
            or [(p['range'],p['page']) for p in pages]!=sorted((p['range'],p['page']) for p in pages)):
        raise ReviewError('source_page_order_invalid')
    all_stamps=[]; page_audit=[]
    for index_range,(first,last) in enumerate(ranges):
        low=datetime.combine(date.fromisoformat(first),day_time(),NY)
        high=datetime.combine(date.fromisoformat(last)+timedelta(days=1),day_time(),NY)
        low_ns=calendar.timegm(low.astimezone(timezone.utc).utctimetuple())*1_000_000_000
        high_ns=calendar.timegm(high.astimezone(timezone.utc).utctimetuple())*1_000_000_000
        group=[p for p in pages if p['range']==index_range]
        if not group:
            raise ReviewError('missing_source_range')
        token=None; last_stamp=None; seen_tokens=set()
        for number,p in enumerate(group):
            guard(); ref=p['canonical_ref']; path=Path(ref['path'])
            if (p['page']!=number or p['request_page_token']!=token or (number and token is None)
                    or not path.is_absolute() or type(ref['bytes']) is not int or not 0<ref['bytes']<=4*1024*1024):
                raise ReviewError('source_reference_or_token_invalid')
            with path.open('rb') as stream:
                payload=stream.read(ref['bytes']+1)
            if len(payload)!=ref['bytes'] or sha(payload)!=ref['sha256']:
                raise ReviewError('canonical_payload_identity_invalid')
            # JSON instantiates SPY price fields; no price is extracted or calculated.
            body=json.loads(payload)
            if (not isinstance(body.get('bars'),dict) or set(body['bars'])!={'SPY'}
                    or 'next_page_token' not in body):
                raise ReviewError('source_symbol_or_terminal_field_invalid')
            rows=body['bars']['SPY']; next_token=body['next_page_token']
            if (not isinstance(rows,list) or not 1<=len(rows)<=10000 or len(rows)!=p['rows']
                    or next_token!=p['next_page_token']
                    or (next_token is not None and (not isinstance(next_token,str) or not next_token or next_token in seen_tokens))):
                raise ReviewError('source_rows_or_token_cycle_invalid')
            current=[]
            for j,bar in enumerate(rows):
                if j%256==0: guard()
                value=stamp_ns(bar['t'])
                if not low_ns<=value<high_ns or value>=CUTOFF or (last_stamp is not None and value<=last_stamp):
                    raise ReviewError('source_time_order_or_boundary_invalid')
                current.append(value); last_stamp=value
            if current[0]!=stamp_ns(p['first_UTC']) or current[-1]!=stamp_ns(p['last_UTC']):
                raise ReviewError('source_index_clock_mismatch')
            all_stamps.extend(current); token=next_token
            if token is not None: seen_tokens.add(token)
            page_audit.append({'range':index_range,'page':number,'canonical_ref':ref})
        if token is not None:
            raise ReviewError('unfinished_range_pagination')
    stamps=np.asarray(all_stamps,dtype=np.int64)
    if not len(stamps) or np.any(stamps>=CUTOFF) or np.any(stamps[1:]<=stamps[:-1]):
        raise ReviewError('global_source_order_invalid')
    return stamps,page_audit


def check_clock(bars,count,duration):
    if (any(v.ndim!=1 or len(v)!=count for v in bars.values())
            or bars['start_ns'].dtype!=np.dtype('int64') or bars['end_ns'].dtype!=np.dtype('int64')
            or bars['observed'].dtype!=np.bool_ or bars['session_date'].dtype.kind!='U'
            or np.any(bars['start_ns'][1:]<=bars['start_ns'][:-1])
            or np.any(bars['end_ns'][1:]<=bars['end_ns'][:-1])
            or np.any(bars['end_ns']-bars['start_ns']!=duration) or np.any(bars['end_ns']>CUTOFF)):
        raise ReviewError('original_clock_invalid')


def independent_hours(minute,hour,source_present,guard=lambda:None):
    starts=[]; ends=[]; sessions=[]; available=[]; first=0
    n=len(minute['start_ns'])
    while first<n:
        guard(); stop=first+1; session=minute['session_date'][first]
        while stop<n and minute['session_date'][stop]==session: stop+=1
        if first and session<=minute['session_date'][first-1]:
            raise ReviewError('session_order_invalid')
        if np.any(minute['start_ns'][first+1:stop]-minute['start_ns'][first:stop-1]!=MINUTE):
            raise ReviewError('original_session_minute_gap')
        blocks=(stop-first)//60
        for block in range(blocks):
            a=first+60*block; b=a+60
            starts.append(int(minute['start_ns'][a])); ends.append(int(minute['end_ns'][b-1])); sessions.append(session)
            available.append(all(bool(x) for x in source_present[a:b]))
        first=stop
    for field,expected in [('start_ns',starts),('end_ns',ends),('session_date',sessions)]:
        if not np.array_equal(hour[field],np.asarray(expected,dtype=hour[field].dtype)):
            raise ReviewError('independent_hour_grid_mismatch')
    return np.asarray(available,dtype=bool)


def history_mask_all(source_present,length):
    """Independent sliding-window reduction; no cumsum/prefix-count implementation."""
    n=len(source_present)
    if n<length:
        return np.zeros(n,dtype=bool)
    windows=np.lib.stride_tricks.sliding_window_view(source_present,length)
    return np.r_[np.zeros(length,dtype=bool),np.all(windows,axis=1)[:-1]]


def independent_partition(fit,ends,horizon):
    if (fit.dtype!=np.dtype('int64') or fit.ndim!=1 or len(fit)<2
            or np.any(fit[1:]<=fit[:-1]) or fit[0]<0 or int(fit[-1])+horizon>=len(ends)):
        raise ReviewError('original_fit_clock_invalid')
    cut=(len(fit)*4)//5; tail=fit[cut:].copy(); decision=int(ends[int(tail[0])])
    prefix=np.asarray([int(i) for i in fit[:cut] if int(ends[int(i)+horizon])<decision],dtype=np.int64)
    return cut,prefix,tail


def partition_summary(origins,mask):
    return {'total':len(origins),'complete':int(np.count_nonzero(mask)),
            'incomplete':int(len(mask)-np.count_nonzero(mask)),
            'origin_array_sha256':sha(origins.astype('<i8').tobytes()),'mask_sha256':sha(mask.tobytes())}


def run(protocol_path):
    os.sched_setaffinity(0,{26})
    raw=protocol_path.read_bytes(); protocol=json.loads(raw); protocol_hash=sha(raw); now=time.time()
    if (protocol.get('kind')!='SPY_original_origin_clock_coverage_v1' or protocol.get('execution_admitted') is not True
            or protocol['deadline_epoch']-protocol['started_epoch']!=180
            or not protocol['started_epoch']<=now<protocol['deadline_epoch']-1
            or protocol.get('independent_reviewer_sha256')!=file_sha(Path(__file__))
            or protocol.get('Q3_allowed') is not False or protocol.get('fitting_allowed') is not False
            or protocol.get('independent_CPU_affinity')!=[26]):
        raise ReviewError('shared_coverage_protocol_invalid')
    def guard():
        if time.time()>=protocol['deadline_epoch']-1:
            raise ReviewError('shared_original_deadline_reached')
    def expired(_sig,_frame): raise ReviewError('shared_original_deadline_reached')
    signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,protocol['deadline_epoch']-1-time.time())
    started=time.time(); audit=[]; mask_audit=[]
    try:
        table=frozen_table(protocol['locks'])
        if (file_sha(BASE/'coverage_worker.py',guard)!=protocol['worker_sha256']
                or file_sha(BASE/'coverage_core.py',guard)!=protocol['core_sha256']):
            raise ReviewError('primary_code_identity_changed')
        if protocol.get('source_final_independent_qualification_required') is not True:
            raise ReviewError('source_final_qualification_required')
        locked_json(protocol['source_acceptance_path'],table,guard)
        locked_json(protocol['root_source_acceptance_path'],table,guard)
        primary_path=BASE/'run/RESULTS.json'; primary_bytes=primary_path.read_bytes(); primary=json.loads(primary_bytes)
        if (primary['status']!='source_clock_coverage_only_complete' or primary['protocol_sha256']!=protocol_hash
                or primary['deadline_epoch']!=protocol['deadline_epoch']
                or not protocol['started_epoch']<=primary['started_epoch']<=primary['ended_epoch']<=protocol['deadline_epoch']
                or primary['features_models_scores_built'] is not False
                or primary['AAPL_close_labels_stop_outer_decoded'] is not False
                or primary['predictive_screen_automatically_admitted'] is not False
                or primary['Q3_read'] is not False or primary['research_goal_achieved'] is not False):
            raise ReviewError('primary_result_identity_or_scope_invalid')
        index=locked_json(protocol['source_index_path'],table,guard)
        stamps,page_audit=independent_source(index,protocol,guard)
        grids={}
        for frequency,duration in [('minute',MINUTE),('hour',60*MINUTE)]:
            ref=protocol['original_bars'][frequency]
            grids[frequency]=selected_npz(ref['path'],BAR_MEMBERS,audit,guard,table)
            check_clock(grids[frequency],ref['rows'],duration)
        # Different mapping algorithm from the primary searchsorted implementation.
        minute_present=np.isin(grids['minute']['start_ns'],stamps,assume_unique=True)
        hourly_present=independent_hours(grids['minute'],grids['hour'],minute_present,guard)
        outside=len(stamps)-int(np.count_nonzero(minute_present))
        for field,value in [('source_rows',len(stamps)),('source_rows_outside_original_RTH',outside),
                            ('source_observed_minutes',int(np.count_nonzero(minute_present))),
                            ('source_complete_hours',int(np.count_nonzero(hourly_present)))]:
            if primary[field]!=value: raise ReviewError('source_or_RTH_count_mismatch')
        if primary['original_hour_clock_exactly_matched'] is not True:
            raise ReviewError('primary_original_hour_not_matched')
        if len(primary['rows'])!=4 or len(protocol['tasks'])!=2:
            raise ReviewError('four_state_count_invalid')
        rows=[]; expected=[('minute_h60_future_log_return','minute',480,60),('hourly_h8_future_log_return','hour',96,8)]
        by_key={(r['task'],r['fold']):r for r in primary['rows']}
        if len(by_key)!=4: raise ReviewError('duplicate_primary_state')
        for task_ref,identity in zip(protocol['tasks'],expected):
            guard(); task=locked_json(task_ref['path'],table,guard); name,frequency,length,horizon=identity
            if (tuple(task[k] for k in ('task','frequency','lookback','horizon'))!=identity or task_ref['folds']!=[1,2]):
                raise ReviewError('original_task_identity_invalid')
            all_history=history_mask_all(minute_present if frequency=='minute' else hourly_present,length)
            for fold in (1,2):
                guard(); meta=task['folds'][str(fold)]
                fit=selected_npz(meta['split_file'],('fit',),audit,guard,table)['fit']
                if (len(fit)!=meta['counts']['fit'] or sha(fit.astype('<i8').tobytes())!=meta['split_sha256']['fit']
                        or np.any(fit<length-1)):
                    raise ReviewError('original_fit_identity_invalid')
                ends=grids[frequency]['end_ns']; cut,prefix,tail=independent_partition(fit,ends,horizon)
                row=by_key[(name,fold)]
                if (row['lookback']!=length or row['horizon']!=horizon or row['original_cut']!=cut
                        or row['purged_count']!=cut-len(prefix) or row['first_tail_decision_ns']!=int(ends[int(tail[0])])
                        or row['last_prefix_label_end_ns']!=(int(ends[int(prefix[-1])+horizon]) if len(prefix) else None)):
                    raise ReviewError('original_partition_clock_mismatch')
                mask_path=Path(row['mask_file'])
                if mask_path.resolve()!=BASE/'run'/(name+'_fold_'+str(fold)+'_clock_masks.npz') or file_sha(mask_path,guard)!=row['mask_file_sha256']:
                    raise ReviewError('primary_mask_identity_invalid')
                names=tuple(p+suffix for p in PARTITIONS for suffix in ('_origins','_complete_history'))
                outputs=selected_npz(mask_path,names,mask_audit,guard)
                summaries={}
                for part,origins in zip(PARTITIONS,(fit,prefix,tail)):
                    guard(); mask=all_history[origins]
                    if (outputs[part+'_origins'].dtype!=np.dtype('int64') or outputs[part+'_complete_history'].dtype!=np.bool_
                            or not np.array_equal(outputs[part+'_origins'],origins)
                            or not np.array_equal(outputs[part+'_complete_history'],mask)):
                        raise ReviewError('original_origins_or_history_mask_mismatch')
                    summaries[part]=partition_summary(origins,mask)
                    if summaries[part]!=row['partitions'][part]:
                        raise ReviewError('partition_counts_or_hashes_mismatch')
                rows.append({'task':name,'fold':fold,'partitions':summaries,'all_origins_and_masks_match':True})
        if audit!=primary['decoded_AAPL_members']:
            raise ReviewError('primary_original_member_audit_mismatch')
        guard(); ended=time.time()
        result={'status':'independent_clock_coverage_review_passed','protocol_sha256':protocol_hash,
            'primary_result_sha256':sha(primary_bytes),'reviewer_sha256':file_sha(Path(__file__),guard),
            'started_epoch':started,'ended_epoch':ended,'shared_original_deadline_epoch':protocol['deadline_epoch'],
            'seconds':ended-started,'same_180_second_window':True,'CPU_affinity':sorted(os.sched_getaffinity(0)),
            'independent_implementations':['np.isin exact-grid mapping','explicit per-session original-hour construction','sliding_window_view/all delayed history','explicit label-clock prefix purge'],
            'source_rows':len(stamps),'source_rows_outside_original_RTH':outside,
            'source_observed_minutes':int(minute_present.sum()),'source_complete_hours':int(hourly_present.sum()),
            'source_acceptance_proof_hashes':{key:table[str(Path(protocol[key]).resolve())] for key in ('source_acceptance_path','root_source_acceptance_path')},
            'source_page_metadata':page_audit,'rows':rows,'decoded_AAPL_members':audit,'decoded_primary_clock_mask_members':mask_audit,
            'SPY_price_fields_instantiated_by_JSON_but_no_price_calculation':True,
            'AAPL_close_labels_stop_outer_decoded':False,'Q3_read':False,'HTTP_calls':0,'GPU':False,
            'features_fits_scores':False,'predictive_screen_automatically_admitted':False,'research_goal_achieved':False}
        out=BASE/'run/INDEPENDENT_COVERAGE_REVIEW.json'
        if out.exists(): raise ReviewError('do_not_overwrite_completed_independent_review')
        atomic(out,result)
        print(json.dumps({'status':result['status'],'seconds':result['seconds'],'states':len(rows)}))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    run(parser.parse_args().protocol)
