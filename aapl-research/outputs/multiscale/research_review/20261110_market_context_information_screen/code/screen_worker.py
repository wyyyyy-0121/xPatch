"""Admitted single-SPY information screen. No market I/O at import."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

BASE=Path(__file__).resolve().parents[1]
BATCH=2048
THREAD_VARS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
SCIENCE_SHA='19586ce75cf9dd8b00d15dc8385e20aff70a86cc306698adb0f97a55f7fbc6bb'
SOURCE_PROTOCOL_SHA='7289278cabbfba50f072355970fc39e153cdd2d45d42efa277ac85dac46033a5'
INHERITANCE_SHA='3addf97adf1a53a0d7d4c25a03f6a8321158eac429d05021a0de21d82f5961b7'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(65536),b''):h.update(block)
    return h.hexdigest()

def frozen_json(path,expected):
    body=Path(path).read_bytes()
    if hashlib.sha256(body).hexdigest()!=expected:raise ValueError('frozen_json_identity_changed')
    return json.loads(body)

def admission(path):
    body=path.read_bytes();p=json.loads(body);identity=hashlib.sha256(body).hexdigest()
    fixture=p.get('kind')=='synthetic_SPY_screen_fixture_v1'
    now=time.time();start=p['started_epoch'];end=p['deadline_epoch']
    if (type(start) not in (int,float) or type(end) not in (int,float)
            or not math.isfinite(start) or not math.isfinite(end) or end-start!=900
            or not start<=now<end-2 or p.get('GPU') is not False or p.get('Q3_allowed') is not False
            or any(os.environ.get(k)!='1' for k in THREAD_VARS)
            or os.environ.get('CUDA_VISIBLE_DEVICES') not in ('','-1')
            or p.get('worker_cpu')!=(26 if fixture else 24)
            or os.sched_getaffinity(0)!={p['worker_cpu']}):
        raise ValueError('immutable_window_or_resource_admission_invalid')
    if fixture:
        if p.get('market_arrays_admitted') is not False:raise ValueError('fixture_cannot_admit_market')
    elif p.get('market_arrays_admitted') is not True:raise ValueError('market_arrays_not_admitted')
    for name in ('screen_worker.py','screen_math.py','source_io.py','screen_lifecycle.py'):
        target=BASE/'code'/name
        if p['code_hashes'].get(str(target))!=sha(target):raise ValueError('screen_code_not_frozen')
    if p['science_rules_sha256']!=SCIENCE_SHA:raise ValueError('science_rules_not_fixed')
    rules=frozen_json(p['science_rules_path'],SCIENCE_SHA)
    if not fixture:
        frozen_json(p['data_references_path'],p['data_references_sha256'])
        if (p['source_protocol_sha256']!=SOURCE_PROTOCOL_SHA or p['inheritance_manifest_sha256']!=INHERITANCE_SHA):
            raise ValueError('accepted_source_ancestry_changed')
    accepted=frozen_json(p['synthetic_acceptance_path'],p['synthetic_acceptance_sha256'])
    if (accepted['status']!='passed' or accepted['market_data_used'] is not False
            or accepted['maximum_shape']!=[2048,979,60]
            or not all(math.isfinite(accepted[k]) and accepted[k]>0 for k in ('max_shape_update_seconds','max_shape_solve_seconds'))):
        raise ValueError('maximum_shape_acceptance_invalid')
    run=Path(p.get('run_directory',str(BASE/'run'))).resolve()
    if not fixture and run!=BASE/'run':raise ValueError('market_run_directory_must_be_fixed')
    return p,identity,rules,accepted,run,fixture

def qualification(source,run,identity,guard):
    import numpy as np
    from screen_lifecycle import atomic_json,digest
    rows=[];extras={}
    for state in source.states:
        guard();P=state.P;PV=state.PV;complete=state.complete_PV
        minimumA=max(100,state.L+19);minimumV=max(100,2*state.L+20)
        if len(P)<minimumA or len(PV)<minimumV or int(complete.sum())<100:
            raise ValueError('prefix_input_domain_qualification_failed:'+state.key)
        for a in range(0,len(P),BATCH):
            x=state.A(P[a:a+BATCH])
            if x.shape!=(len(P[a:a+BATCH]),state.L+18) or not np.isfinite(x).all():raise ValueError('A_qualification_failed')
        for a in range(0,len(PV),BATCH):
            x=state.B(PV[a:a+BATCH])
            if x.shape!=(len(PV[a:a+BATCH]),state.L+20) or not np.isfinite(x).all():raise ValueError('B_qualification_failed')
        values=PV[complete];s=np.zeros(state.L-1);ss=np.zeros(state.L-1);hasher=hashlib.sha256()
        for a in range(0,len(values),BATCH):
            guard();z=state.z(values[a:a+BATCH])
            if z.shape!=(len(values[a:a+BATCH]),state.L-1) or not np.isfinite(z).all():raise ValueError('SPY_complete_feature_qualification_failed')
            s+=z.sum(axis=0);ss+=np.square(z).sum(axis=0);hasher.update(z.astype('<f8').tobytes())
        mean=s/len(values);variance=ss/len(values)-mean*mean
        margin=np.maximum(np.maximum(np.abs(ss/len(values)),mean*mean),1.)
        if (variance<-1e-10*margin).any():raise ValueError('SPY_qualification_negative_variance')
        sd=np.sqrt(np.maximum(variance,0))
        if not np.isfinite(mean).all() or not np.isfinite(sd).all() or not (sd>1e-12).any():
            raise ValueError('SPY_representation_qualification_failed')
        extras['spy_mean_'+state.key]=mean
        domains={}
        for expert,origins,columns in (('A_full',P,state.L+18),('B_valid',PV,state.L+20),('Plus_valid',PV,2*state.L+19)):
            domains[expert]={'n_train':len(origins),'ordered_origins_sha256':digest(origins.astype('<i8').tobytes()),'columns':columns}
        rows.append({'state':state.key,'task':state.task,'fold':state.fold,'P_count':len(P),'PV_count':len(PV),
                     'complete_SPY_in_PV_count':int(complete.sum()),'mVP_sha256':digest(state.mVP.tobytes()),
                     'mSP_sha256':digest(state.mSP.tobytes()),'complete_SPY_features_sha256':hasher.hexdigest(),
                     'SPY_raw_std_max':float(sd.max()),'ordered_input_domains':domains,'passed':True})
    if len(rows)!=4:raise ValueError('all_four_prefix_states_required')
    report={'kind':'fixed_SPY_no_label_qualification_v1','protocol_sha256':identity,'rows':rows,
            'all_four_passed':True,'labels_constructed':False,'tail_predictions_constructed':False}
    atomic_json(run/'QUALIFICATION.json',report)
    return report,extras

def train_features(state,expert,origins,mean):
    if expert=='A_full':return state.A(origins)
    if expert=='B_valid':return state.B(origins)
    positions=__import__('numpy').searchsorted(state.PV,origins)
    return state.plus(origins,state.complete_PV[positions],mean)

def run_screen(p,identity,rules,accepted,run,fixture):
    import numpy as np
    from screen_lifecycle import (Guard,PreserveStop,atomic_json,atomic_npz,save_checkpoint,
                                 load_checkpoint,file_sha,digest,validate_external_pause)
    from screen_math import empty_stats,update,solve,predict,route,score,aggregate,EXPERTS,METHODS
    guard=Guard(p);run.mkdir(parents=True,exist_ok=True);checkpoint=run/'CHECKPOINT.npz'
    meta=None;stats=None;models={};extras={};source=None
    try:
        guard()
        if checkpoint.exists():
            meta,stats,models,extras=load_checkpoint(checkpoint)
            if meta['protocol_sha256']!=identity or meta['started_epoch']!=p['started_epoch'] or meta['deadline_epoch']!=p['deadline_epoch']:
                raise ValueError('checkpoint_original_protocol_changed')
            # Validate actual external new-PID continuation and cost before any further stats.
            if meta.get('required_external_pause'):
                marker,receipt,timing=validate_external_pause(run,p,identity,checkpoint,meta)
                if not meta.get('external_pause_accepted'):
                    actual=timing['actual_first_A_update_seconds']
                    projected=3*(8*79*max(accepted['max_shape_update_seconds'],actual)+12*accepted['max_shape_solve_seconds'])+120
                    passed=bool(time.time()+projected<=p['deadline_epoch'])
                    atomic_json(run/'COST_GATE.json',{'protocol_sha256':identity,'projected_remaining_seconds':projected,
                        'actual_first_A_update_seconds':actual,'accepted_max_shape_update_seconds':accepted['max_shape_update_seconds'],
                        'accepted_max_shape_solve_seconds':accepted['max_shape_solve_seconds'],'evaluated_epoch':time.time(),
                        'deadline_epoch':p['deadline_epoch'],'passed':passed,'external_pause_verified_before_gate':True})
                    if not passed:raise PreserveStop('fixed_end_to_end_cost_gate_refused')
                    atomic_json(run/'RESUME_VALIDATION.json',{'protocol_sha256':identity,'old_pid':receipt['pid'],
                        'new_pid':os.getpid(),'new_PID_verified':True,'checkpoint_bytes_preserved':True,
                        'original_deadline_reset':False,'cost_gate_passed':True})
                    meta['external_pause_accepted']=True
                    meta['external_pause_receipt_sha256']=file_sha(run/'EXTERNAL_PAUSE_RECEIPT.json')
                    save_checkpoint(checkpoint,meta,stats,models,extras)
            if meta['phase']=='complete':
                print(json.dumps({'status':'already_complete','protocol_sha256':identity}));return 0
        if fixture:
            from synthetic_support import ArtificialSource
            source=ArtificialSource(rules,guard)
        else:
            from source_io import ScreenSource
            source=ScreenSource(p,rules,guard)
        if meta is None:
            report,extras=qualification(source,run,identity,guard)
            meta={'protocol_sha256':identity,'started_epoch':p['started_epoch'],'deadline_epoch':p['deadline_epoch'],
                  'phase':'fit','fit_index':0,'cursor':0,'tail_state':0,'tail_cursor':0,'rows':[],
                  'required_external_pause':False,'external_pause_accepted':False,
                  'qualification_sha256':file_sha(run/'QUALIFICATION.json'),'models':[],'transforms':[]}
            save_checkpoint(checkpoint,meta,None,models,extras)
        else:
            report=frozen_json(run/'QUALIFICATION.json',meta['qualification_sha256'])
            if report.get('all_four_passed') is not True or report.get('labels_constructed') is not False:
                raise ValueError('saved_all_state_no_label_qualification_invalid')
        source.allow_prefix_labels()
        jobs=[(state,expert) for state in source.states for expert in EXPERTS]
        while meta['fit_index']<12:
            guard();index=meta['fit_index'];state,expert=jobs[index]
            origins=state.P if expert=='A_full' else state.PV
            columns=state.L+18 if expert=='A_full' else state.L+20 if expert=='B_valid' else 2*state.L+19
            if stats is None:stats=empty_stats(columns,state.H)
            if int(stats['n'])!=meta['cursor']:raise ValueError('committed_stats_cursor_mismatch')
            while meta['cursor']<len(origins):
                guard();start=meta['cursor'];selected=origins[start:start+BATCH]
                began=time.perf_counter()
                x=train_features(state,expert,selected,extras['spy_mean_'+state.key])
                y=state.labels(selected,'prefix');update(stats,x,y)
                meta['cursor']+=len(selected)
                first=index==0 and start==0
                if first:
                    if len(selected)!=2048:raise ValueError('first_prefix_batch_must_have_2048_origins')
                    meta['required_external_pause']=True
                save_checkpoint(checkpoint,meta,stats,models,extras)
                elapsed=time.perf_counter()-began
                if first:
                    atomic_json(run/'FIRST_BATCH_TIMING.json',{'protocol_sha256':identity,'pid':os.getpid(),
                        'actual_first_A_update_seconds':elapsed,'includes_X_Y_stats_checkpoint_and_fsync':True,
                        'source_load_and_qualification_excluded':True,'cursor':2048})
                    atomic_json(run/'READY_FOR_EXTERNAL_PAUSE.json',{'protocol_sha256':identity,'pid':os.getpid(),
                        'cursor':2048,'checkpoint_path':str(checkpoint),'checkpoint_sha256':file_sha(checkpoint),
                        'timing_sha256':file_sha(run/'FIRST_BATCH_TIMING.json')})
                    while True:
                        guard();time.sleep(.02)
                guard()
            model=solve(stats);models[index]=model
            path=run/'models'/(state.key+'__'+expert+'.npz');atomic_npz(path,model)
            domain=report['rows'][index//3]['ordered_input_domains'][expert]
            meta['models'].append({'state':state.key,'task':state.task,'fold':state.fold,'expert':expert,
                'path':str(path),'sha256':file_sha(path),'n_train':int(stats['n']),
                'domain_sha256':domain['ordered_origins_sha256'],'columns':columns})
            meta['fit_index']+=1;meta['cursor']=0;stats=None
            save_checkpoint(checkpoint,meta,stats,models,extras);guard()
        if meta['phase']=='fit':
            if len(models)!=12 or len(meta['models'])!=12:raise ValueError('all_twelve_models_barrier_failed')
            for si,state in enumerate(source.states):
                path=run/(state.key+'_PREFIX_TRANSFORMS.npz')
                atomic_npz(path,{'spy_raw_mean':extras['spy_mean_'+state.key],'prefix_mean':models[si*3]['y_mean']})
                meta['transforms'].append({'state':state.key,'path':str(path),'sha256':file_sha(path)})
            manifest={'kind':'fixed_SPY_all_twelve_prefix_models_v1','protocol_sha256':identity,
                'qualification_path':str(run/'QUALIFICATION.json'),'qualification_sha256':meta['qualification_sha256'],
                'models':meta['models'],'transforms':meta['transforms'],'total_fits':12,
                'all_models_frozen_before_any_tail_prediction_or_label':True}
            atomic_json(run/'FROZEN_MODELS.json',manifest)
            meta['frozen_manifest_sha256']=file_sha(run/'FROZEN_MODELS.json');meta['phase']='tail'
            save_checkpoint(checkpoint,meta,None,models,extras)
        manifest=frozen_json(run/'FROZEN_MODELS.json',meta['frozen_manifest_sha256'])
        if manifest['total_fits']!=12 or manifest['all_models_frozen_before_any_tail_prediction_or_label'] is not True:
            raise ValueError('tail_barrier_identity_invalid')
        for row in manifest['models']+manifest['transforms']:
            if file_sha(row['path'])!=row['sha256']:raise ValueError('frozen_prefix_artifact_changed')
        guard();source.allow_tail_labels()
        while meta['tail_state']<4:
            si=meta['tail_state'];state=source.states[si];n=len(state.T);h=state.H
            if meta['tail_cursor']==0:
                for name in ('targets',)+METHODS:extras['tail_'+name]=np.zeros((n,h),dtype=np.float64)
            while meta['tail_cursor']<n:
                guard();a=meta['tail_cursor'];b=min(a+BATCH,n);origins=state.T[a:b]
                v=state.mVT[a:b];s=state.mST[a:b]
                A=predict(models[si*3],state.A(origins))
                B=predict(models[si*3+1],state.B(origins[v])) if v.any() else np.empty((0,h))
                keep=v&s
                Plus=predict(models[si*3+2],state.plus(origins[keep],np.ones(int(keep.sum()),dtype=bool),
                    extras['spy_mean_'+state.key])) if keep.any() else np.empty((0,h))
                base,plus=route(A,B,Plus,v,s)
                predictions={'A_full':A,'base':base,'plus':plus,'zero':np.zeros_like(A),
                             'prefix_mean':np.broadcast_to(models[si*3]['y_mean'],A.shape).copy()}
                extras['tail_targets'][a:b]=state.labels(origins,'tail')
                for name,value in predictions.items():extras['tail_'+name][a:b]=value
                meta['tail_cursor']=b;save_checkpoint(checkpoint,meta,None,models,extras);guard()
            arrays={'origins':state.T,'mV':state.mVT,'mS':state.mST}
            arrays.update({name:extras['tail_'+name] for name in ('targets',)+METHODS})
            path=run/(state.key+'_TAIL.npz');atomic_npz(path,arrays)
            metrics,gates,groups=score(arrays['targets'],{name:arrays[name] for name in METHODS},state.mVT,state.mST)
            meta['rows'].append({'state':state.key,'task':state.task,'fold':state.fold,'n_tail':n,
                'tail_path':str(path),'tail_sha256':file_sha(path),'ordered_tail_origins_sha256':digest(state.T.astype('<i8').tobytes()),
                'metrics':metrics,'gates':gates,'state_pass':bool(all(gates.values())),
                'diagnostic_groups':groups,'fallback_bitwise_verified':True})
            for name in ('targets',)+METHODS:del extras['tail_'+name]
            meta['tail_state']+=1;meta['tail_cursor']=0
            save_checkpoint(checkpoint,meta,None,models,extras);guard()
        primaries,passed=aggregate(meta['rows'])
        results={'kind':'fixed_SPY_information_screen_result_v1','protocol_sha256':identity,
            'status':'complete','rows':meta['rows'],'primaries':primaries,'overall_information_gate':passed,
            'numerically_qualified':True,'total_fits':12,'frozen_manifest_sha256':meta['frozen_manifest_sha256'],
            'Q3_read':False,'old_stop_or_outer_read':False,'GPU_used':False,'paid_resources':False,
            'market_data_used':not fixture,'independent_confirmation_claimed':False,'formal_method_success_claimed':False,
            'goal_achieved':False,'ended_epoch':time.time(),'deadline_epoch':p['deadline_epoch'],
            'maximum_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'selected_npz_reads':getattr(getattr(source,'locked',None),'audit',[])}
        atomic_json(run/'RESULTS.json',results);meta['phase']='complete';save_checkpoint(checkpoint,meta,None,models,extras)
        print(json.dumps({'status':'complete','overall_information_gate':passed,'protocol_sha256':identity}));return 0
    except PreserveStop as error:
        atomic_json(run/('PRESERVED_STOP_'+str(os.getpid())+'.json'),{'protocol_sha256':identity,
            'reason':str(error),'pid':os.getpid(),'checkpoint_path':str(checkpoint),
            'checkpoint_sha256':file_sha(checkpoint) if checkpoint.exists() else None,
            'original_deadline_epoch':p['deadline_epoch'],'original_deadline_reset':False,'ended_epoch':time.time()})
        print(json.dumps({'status':'preserved_stop','reason':str(error)}));return 75
    except Exception as error:
        atomic_json(run/('FAILED_'+str(os.getpid())+'.json'),{'protocol_sha256':identity,'status':'failed',
            'error_type':type(error).__name__,'reason':str(error),'checkpoint_retained':checkpoint.exists(),
            'models_or_effect_success_claimed':False,'ended_epoch':time.time()})
        raise

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',required=True,type=Path)
    parser.add_argument('--run',action='store_true',required=True);args=parser.parse_args()
    p,identity,rules,accepted,run,fixture=admission(args.protocol)
    resource.setrlimit(resource.RLIMIT_AS,(3221225472,3221225472))
    raise SystemExit(run_screen(p,identity,rules,accepted,run,fixture))

if __name__=='__main__':main()
