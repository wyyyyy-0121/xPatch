"""Original-clock nested roles only; no price, target, prediction or model reads."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import signal
import time
import types
import zipfile

BASE = Path(__file__).resolve().parent
NAMES = ('F','P','B','Q','S','C_cal','T','gap_P_to_T','gap_B_to_C','gap_Q_to_S')
ROLES = ('Q','S','C_cal','gap_Q_to_S','gap_B_to_C','gap_P_to_T','T')
ROLE_LABELS = ('Q_fit','S_early_stop','C_calibration','gap_Q_to_S','gap_B_to_C','gap_P_to_T','T_assessment')
CUTOFF = 1782878400000000000


def sha(body): return hashlib.sha256(body).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def atomic(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    tmp.replace(path)


def run(path):
    raw = path.read_bytes(); p = json.loads(raw); began = time.time()
    if (not all(type(p[k]) in (int,float) and math.isfinite(p[k]) for k in ('started_epoch','deadline_epoch'))
            or p['deadline_epoch']-p['started_epoch'] != 180
            or not p['started_epoch'] <= began < p['deadline_epoch']
            or p['kind'] != 'two_state_nested_clock_roles_v1'
            or p['worker_sha256'] != sha(Path(__file__).read_bytes())
            or p['market_prices_labels_allowed'] is not False
            or p['model_fitting_allowed'] is not False or p['Q3_allowed'] is not False
            or p['GPU'] is not False or p['paid_resources'] is not False):
        raise ValueError('invalid_clock_scope_window_identity')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(key) != '1':raise ValueError('thread_environment')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':raise ValueError('GPU_environment')
    os.sched_setaffinity(0,{24});resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
    def guard():
        if time.time() >= p['deadline_epoch']:raise TimeoutError('original_nested_clock_deadline')
    def expire(*_):raise TimeoutError('nested_clock_deadline_or_termination')
    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):signal.signal(sig,expire)
    signal.setitimer(signal.ITIMER_REAL,p['deadline_epoch']-time.time())
    import numpy as np
    table={r['path']:r['sha256'] for r in p['locks']}
    if len(table)!=len(p['locks']):raise ValueError('duplicate_locks')
    audit=[]
    def body(ref):
        guard(); filename=str(Path(ref).resolve())
        if filename not in table:raise ValueError('unlocked_path')
        value=Path(filename).read_bytes()
        if sha(value)!=table[filename]:raise ValueError('frozen_input_changed')
        return value
    def selected(ref,names):
        if names not in (('end_ns','session_date'),('fit_origins','prefix_mask','tail_mask')):
            raise ValueError('member_allowlist')
        arrays={}
        with zipfile.ZipFile(io.BytesIO(body(ref))) as archive:
            listing=archive.namelist()
            for key in names:
                guard();member=key+'.npy'
                if listing.count(member)!=1:raise ValueError('ambiguous_member')
                with archive.open(member) as stream:arrays[key]=np.lib.format.read_array(stream,allow_pickle=False)
                audit.append({'path':str(Path(ref).resolve()),'member':member})
        return arrays
    def ah(array):return sha(np.asarray(array,dtype='<i8').tobytes())
    for ref in p['locks']:body(ref['path'])
    prior=json.loads(body(p['prior_results_path']))
    accept=json.loads(body(p['prior_acceptance_path']))
    if (prior['status']!='original_five_fold_fit_clock_support_complete'
            or accept['status']!='accepted_all_original_fit_clock_support_metadata_only'
            or accept['task_fold_states']!=10 or prior['Q3_read'] is not False):
        raise ValueError('prior_clock_support_not_accepted')
    nested_source=body(p['nested_module_path'])
    if sha(nested_source)!=p['nested_module_sha256']:raise ValueError('nested_module_changed')
    nested=types.ModuleType('frozen_nested_clock_partitions')
    nested.__file__=p['nested_module_path'];exec(compile(nested_source,p['nested_module_path'],'exec'),nested.__dict__)
    output=BASE/'nested_run';output.mkdir(exist_ok=False)
    clocks={}
    for frequency in ('minute','hour'):
        bars=selected(p['original_bars'][frequency]['path'],('end_ns','session_date'))
        if (bars['end_ns'].dtype!=np.dtype('int64') or bars['end_ns'].ndim!=1
                or bars['session_date'].dtype.kind!='U' or bars['session_date'].shape!=bars['end_ns'].shape
                or len(bars['end_ns'])!=p['original_bars'][frequency]['rows']
                or (np.diff(bars['end_ns'])<=0).any() or (bars['end_ns']>CUTOFF).any()
                or (bars['session_date']>'2026-06-30').any()):raise ValueError('original_clock_invalid')
        clocks[frequency]=bars
    rows=[];in_memory=[]
    for old in prior['rows']:
        guard();frequency=old['frequency'];clock=clocks[frequency]
        if table.get(old['support_file'])!=old['support_file_sha256']:raise ValueError('old_support_identity')
        saved=selected(old['support_file'],('fit_origins','prefix_mask','tail_mask'))
        fit=saved['fit_origins']
        if ah(fit)!=old['partitions']['full_fit']['origin_sha256']:raise ValueError('original_fit_identity')
        result=nested.build_nested_partitions(clock,fit,saved['prefix_mask'],saved['tail_mask'],
                                             horizon=old['H'],lookback=old['L'])
        parts=result['partitions'];part_meta={};payload={}
        for name in NAMES:
            item=parts[name];origins=item['origins']
            for field in ('origins','fit_positions','parent_positions'):payload[name+'_'+field]=item[field]
            part_meta[name]={'count':len(origins),'origin_sha256':ah(origins),
                'fit_positions_sha256':ah(item['fit_positions']),'parent':item['parent'],
                'parent_positions_sha256':ah(item['parent_positions']),
                'session_count':len(np.unique(clock['session_date'][origins])),
                'first_decision_ns':int(clock['end_ns'][origins[0]]) if len(origins) else None,
                'last_decision_ns':int(clock['end_ns'][origins[-1]]) if len(origins) else None,
                'last_label_end_ns':int(clock['end_ns'][origins[-1]+old['H']]) if len(origins) else None}
        if (part_meta['P']['origin_sha256']!=old['partitions']['purged_prefix']['origin_sha256']
                or part_meta['T']['origin_sha256']!=old['partitions']['unchanged_tail']['origin_sha256']):
            raise ValueError('accepted_P_T_changed')
        uid_input={'task':old['task'],'fold':old['fold'],'L':old['L'],'H':old['H'],
            'clock_end_sha256':ah(clock['end_ns']),
            'session_date_sha256':sha(clock['session_date'].tobytes()),
            'original_fit_sha256':ah(fit),'nested_module_sha256':p['nested_module_sha256'],
            'role_origin_hashes':{name:part_meta[name]['origin_sha256'] for name in NAMES}}
        uid=sha(canonical(uid_input))
        payload['terminal_role']=result['ledger']['terminal_role']
        payload['history_suffix_length']=result['history_support']['suffix_length']
        payload['history_boundary_count']=result['history_support']['boundary_count']
        destination=output/(old['task']+'_fold_'+str(old['fold'])+'_nested.npz')
        tmp=destination.with_suffix('.tmp')
        with tmp.open('wb') as stream:
            np.savez_compressed(stream,**payload);stream.flush();os.fsync(stream.fileno())
        tmp.replace(destination)
        row={'task':old['task'],'frequency':frequency,'fold':old['fold'],'L':old['L'],'H':old['H'],
            'stateUID':uid,'state_UID_input':uid_input,'partitions':part_meta,
            'terminal_role_counts':dict(sorted(Counter(payload['terminal_role'].tolist()).items())),
            'terminal_role_sha256':sha(payload['terminal_role'].tobytes()),
            'history_suffix_length_sha256':ah(payload['history_suffix_length']),
            'history_boundary_count_sha256':ah(payload['history_boundary_count']),
            'nested_file':str(destination),'nested_file_sha256':sha(destination.read_bytes())}
        rows.append(row);in_memory.append((row,{name:parts[name]['origins'] for name in ROLES}))
        atomic(output/'CHECKPOINT.json',{'protocol_sha256':sha(raw),'deadline_epoch':p['deadline_epoch'],
            'completed_rows':rows,'decoded_members':audit})
    if len(rows)!=10 or len({row['stateUID'] for row in rows})!=10:raise ValueError('all10states_required')
    overlaps=[]
    for i,(left,lp) in enumerate(in_memory):
        for right,rp in in_memory[i+1:]:
            same=left['frequency']==right['frequency']
            le=clocks[left['frequency']]['end_ns'];re=clocks[right['frequency']]['end_ns']
            for lr in ROLES:
                for rr in ROLES:
                    guard();values=np.intersect1d(le[lp[lr]],re[rp[rr]],assume_unique=True)
                    origins=np.intersect1d(lp[lr],rp[rr],assume_unique=True) if same else None
                    if same and not np.array_equal(le[origins],values):raise ValueError('same_grid_overlap_mismatch')
                    overlaps.append({'left_stateUID':left['stateUID'],'right_stateUID':right['stateUID'],
                        'left_role':lr,'right_role':rr,'same_frequency':same,
                        'decision_clock_intersection_count':len(values),
                        'ordered_decision_clock_sha256':ah(values),
                        'ordered_origin_sha256':ah(origins) if same else None,
                        'first_decision_ns':int(values[0]) if len(values) else None,
                        'last_decision_ns':int(values[-1]) if len(values) else None})
    guard()
    result={'kind':'matched_two_state_original_clock_nested_roles_v1','status':'all10_nested_roles_and2205_clock_overlap_pairs_complete',
        'protocol_sha256':sha(raw),'started_epoch':began,'ended_epoch':time.time(),
        'deadline_epoch':p['deadline_epoch'],'seconds':time.time()-began,'CPU_affinity':sorted(os.sched_getaffinity(0)),
        'rows':rows,'crossstate_role_clock_overlaps':overlaps,'decoded_members':audit,
        'role_order':list(ROLES),'overlap_scope':'Exact decision-clock intersections for all45statepairs and49terminalrolepairs; origin hash only samefrequency. Not exhaustive raw-history/target-path value reuse. ',
        'whole_compressed_npz_bytes_read_for_identity':True,'price_or_label_members_decoded':False,
        'old_stop_outer_members_decoded':False,'model_fits':0,'Q3_read':False,'GPU':False,
        'full_g_tensor_verified':False,'market_training_admitted':False,'research_goal_achieved':False,
        'RLIMIT_AS_bytes':resource.getrlimit(resource.RLIMIT_AS)[0],
        'RSS_ru_maxrss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    atomic(output/'RESULTS.json',result);signal.setitimer(signal.ITIMER_REAL,0)
    return {'status':result['status'],'states':len(rows),'role_pairs':len(overlaps),'seconds':result['seconds']}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',type=Path,required=True);args=parser.parse_args()
    try:print(json.dumps(run(args.protocol)))
    except BaseException as error:
        atomic(BASE/('NESTED_FAILED_'+str(os.getpid())+'.json'),{'error_type':type(error).__name__,
            'message':str(error),'ended_epoch':time.time(),'market_training_admitted':False,'research_goal_achieved':False})
        raise
