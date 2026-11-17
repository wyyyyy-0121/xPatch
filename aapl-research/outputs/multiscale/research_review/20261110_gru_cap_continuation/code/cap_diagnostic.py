"""Frozen one-cell cap20-to60 diagnostic using the original training loop.

Verification mode never scores models or updates parameters. train_score mode
requires actual verification and synthetic resume evidence frozen into protocol.
All outer scoring occurs only after the continuation best is selected.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict,replace
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import struct
import sys
import tempfile
import time
import zipfile
import numpy as np
import torch
import torch._dynamo
from cap_guard import install_guard

ROOT=Path('/home/ayaka/wslcode/aapl-research')
SOURCE=ROOT/'outputs/multiscale/controlled_followup/20261108_v2/code/reference_source'
JOB=ROOT/'outputs/multiscale/private_three_scale_development/20261107T134128870773Z/hourly_h8_future_log_return_f2'
BAR=ROOT/'outputs/multiscale/admission/alpaca_20261107T052228573242Z/bars_hour.npz'
SPLIT=ROOT/'outputs/multiscale/controlled_followup/20261108_v2/cache/hourly_h8_future_log_return/fold_2.npz'
ADMISSION=BAR.parent/'admission.json'
OLD='910553b6471b9fcef06f325b5b30381a1c4c2e764d6d09001f4179e6685815e4'
SEVEN=('multiscale/training.py','multiscale/models.py','multiscale/features.py','multiscale/bars.py','multiscale/mechanism.py','aapl/models.py','aapl/vendor/xpatch_network.py')
STOP=False
ACTIVE=None


def signalled(*_):
    global STOP
    STOP=True


def check(p,reserve=0):
    if STOP or time.time()+reserve>=p['deadline_epoch']:raise TimeoutError('Signal or immutable cap deadline')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):
            if ACTIVE is not None:check(ACTIVE)
            h.update(block)
    return h.hexdigest()


def same(a,b):
    if isinstance(a,torch.Tensor) or isinstance(b,torch.Tensor):
        return isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and a.device==b.device and torch.equal(a.detach().reshape(-1).contiguous().view(torch.uint8),b.detach().reshape(-1).contiguous().view(torch.uint8))
    if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):
        return isinstance(a,np.ndarray) and isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
    if isinstance(a,dict) or isinstance(b,dict):return isinstance(a,dict) and isinstance(b,dict) and a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)) or isinstance(b,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a,float) or isinstance(b,float):return type(a) is type(b) and struct.pack('!d',a)==struct.pack('!d',b)
    return type(a) is type(b) and a==b


def require(value,message):
    if not value:raise ValueError(message)


def atomic_json(path,value):
    path=Path(path);fd,name=tempfile.mkstemp(dir=path.parent,prefix='.state-',suffix='.json')
    try:
        with os.fdopen(fd,'w') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def load_registered_metadata(p,path):
    require(str(path) in p['metadata_sha256'],'Unregistered metadata')
    require(sha(path)==p['metadata_sha256'][str(path)],'Changed metadata')
    return json.loads(path.read_text())


def gate(p,protocol_path):
    require(p['status']=='frozen_before_execution' and p['mode'] in ('verify','train_score'),'Wrong phase')
    check(p)
    require(p['market_arrays_admitted'] is True,'Data not admitted')
    access=install_guard(p,protocol_path)
    for key in ('code_sha256','metadata_sha256','input_sha256'):
        for path,h in p[key].items():require(sha(path)==h,'Frozen identity changed: '+path)
    for path in (BAR,SPLIT,JOB/'gru/latest.pt',JOB/'gru/best.pt'):
        require(str(path) in p['input_sha256'],'Required exact input absent')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    sys.path.insert(0,str(SOURCE))
    return access


def read_split(member):
    with zipfile.ZipFile(SPLIT) as z:
        require(z.namelist().count(member+'.npy')==1,'Ambiguous split member')
        with z.open(member+'.npy') as f:return np.load(io.BytesIO(f.read()),allow_pickle=False)


def load_data_and_verify(p):
    from multiscale.bars import Bars
    from multiscale.features import CloseWindows,fit_scaler
    from multiscale.training import TrainConfig,identity,digest_indices
    receipt=load_registered_metadata(p,JOB/'fold_receipt.json')
    summary=load_registered_metadata(p,JOB/'gru/training_summary.json')
    admission=load_registered_metadata(p,ADMISSION)
    require(sha(ADMISSION)=='f6fc738205532cf5016d74b2baa4691692cb03674c3db9b6a8ccf8ce8126ae86','Admission differs')
    require(admission['files']['bars_hour.npz']==p['input_sha256'][str(BAR)],'Wrong admitted bars')
    with np.load(BAR,allow_pickle=False) as z:
        require(set(z.files)=={'start_ns','end_ns','close','observed','session_date'},'Wrong bar keys')
        a={k:z[k].copy() for k in z.files}
    require(np.all(a['session_date']<='2026-06-30'),'Beyond preQ3 boundary')
    bars=Bars('hour',**a)
    # Exact original development.bars_identity serialization, independently read
    # from source before protocol freezing; no production loader/foundation data.
    h=hashlib.sha256(bars.frequency.encode('utf8'))
    for name in ('start_ns','end_ns','close','observed','session_date'):
        value=np.ascontiguousarray(getattr(bars,name))
        h.update(name.encode('ascii'));h.update(value.dtype.str.encode('ascii'))
        h.update(str(value.shape).encode('ascii'));h.update(value.tobytes())
    require(h.hexdigest()==receipt['identity_fields']['bars_sha256'],'Decoded bar identity differs')
    fit,stop=read_split('fit'),read_split('stop')
    for name,arr,count in (('fit',fit,2294),('stop',stop,406)):
        require(arr.dtype==np.int64 and arr.shape==(count,) and np.all(np.diff(arr)>0),'Invalid split order/count')
        require(digest_indices(arr)==receipt['identity_fields'][name+'_indices_sha256'],'Canonical split differs')
        require(arr[0]>=95 and arr[-1]+8<len(bars.close),'Bad windows')
        bad=np.r_[0,np.cumsum(~bars.observed,dtype=np.int64)]
        require(np.all(bad[arr+9]==bad[arr-95]),'Incomplete observed history/labels')
    require(bars.end_ns[fit[-1]+8]<bars.end_ns[stop[0]],'Unpurged fit/stop')
    windows=CloseWindows(bars,96,admission['split_actions'])
    config=TrainConfig('gru',96,8,'future_log_return',20,3,128,.001,2026)
    dataid=receipt['data_identity']+':'+receipt['config']['protocol_identity']+':'+receipt['fold_identity']
    require(dataid.startswith('private_exploration_not_publication_admitted:'),'Historical data identity changed')
    for rel in SEVEN:require(sha(SOURCE/rel)==receipt['source_sha256'][rel],'Original identity source differs')
    require(identity(config,dataid,fit,stop,'cpu')==OLD,'Actual training identity differs')
    latest=torch.load(JOB/'gru/latest.pt',weights_only=True,map_location='cpu')
    best=torch.load(JOB/'gru/best.pt',weights_only=True,map_location='cpu')
    scaler=asdict(fit_scaler(windows,fit,8,'future_log_return'))
    require(same(scaler,latest['scaler']) and same(scaler,best['scaler']),'Actual scaler differs bitwise')
    require(latest['identity']==OLD and best['identity']==OLD and best['epoch']==20,'Original payload identity differs')
    require(same(best['config'],asdict(config)),'Original best config differs')
    require(latest['state']['epoch']==20 and latest['state']['steps']==360 and same(latest['state']['history'],summary['history']),'Original state differs')
    return windows,fit,stop,config,dataid,latest,best,receipt,{'decoded_bar_fingerprint':h.hexdigest(),'bar_rows':len(bars.close),'last_date':str(bars.session_date[-1]),'fit_count':len(fit),'stop_count':len(stop),'scaler_actual_recomputation_bitwise':True,'actual_training_identity':OLD,'fit_stop_observed_and_purge_verified':True,'split_members_decoded':['fit.npy','stop.npy'],'outer_decoded':False,'Q3_read':False}


def metrics(pred,truth):
    e=np.square(pred-truth)
    return {'endpoint_mse':float(e[:,-1].mean()),'path_mse':float(e.mean()),'per_step_mse':e.mean(axis=0).tolist()}


def run(p,path):
    global ACTIVE
    ACTIVE=p
    access=gate(p,path);out=Path(p['output_directory']);out.mkdir(parents=True,exist_ok=True)
    if (out/'RESULTS.json').exists():
        previous=json.loads((out/'RESULTS.json').read_text())
        require(previous['protocol_sha256']==sha(path),'Completed output belongs to another protocol')
        return
    status={'status':'running_'+p['mode'],'protocol_sha256':sha(path),'deadline_epoch':p['deadline_epoch'],'training_updates':0,'Q3_read':False,'access_guard':access}
    atomic_json(out/'STATUS.json',status)
    try:
        windows,fit,stop,c20,dataid,original_latest,original_best,receipt,evidence=load_data_and_verify(p)
        status['data_verification']=evidence
        if p['mode']=='verify':
            require(p['training_updates_admitted'] is False and p['outer_scoring_admitted'] is False,'Verification cannot train/score')
            check(p,5)
            status.update(status='passed_actual_data_scaler_identity_verification',completed_epoch=time.time(),market_forward_calls=0,outer_scored=False)
            atomic_json(out/'RESULTS.json',status);atomic_json(out/'STATUS.json',status);return
        require(p['training_updates_admitted'] is True and p['outer_scoring_admitted'] is True,'Training/scoring not admitted')
        verify=load_registered_metadata(p,Path(p['verification_results']))
        synthetic=load_registered_metadata(p,Path(p['synthetic_results']))
        require(verify['status']=='passed_actual_data_scaler_identity_verification','Actual verification missing')
        require(synthetic['status']==p['synthetic_required_status'],'Synthetic acceptance missing')
        from multiscale.training import run_fit,identity,atomic_torch_save,predict_best
        from multiscale.bars import targets
        c60=replace(c20,max_epochs=60);newid=identity(c60,dataid,fit,stop,'cpu')
        require(newid!=OLD,'Cap change must have distinct identity')
        control=out/'control20';extended=out/'extended60'
        if not (out/'BRIDGE.json').exists():
            require(not control.exists() and not extended.exists(),'Unrecorded prior bridge directories')
            control.mkdir();extended.mkdir()
            for directory in (control,extended):
                for name,value in (('latest',original_latest),('best',original_best)):
                    bridged=copy.deepcopy(value)
                    if directory==extended:
                        bridged['identity']=newid
                        if name=='best':bridged['config']=asdict(c60)
                    stripped=copy.deepcopy(bridged);stripped['identity']=OLD
                    if name=='best':stripped['config']=asdict(c20)
                    require(same(stripped,value),'Bridge modified a scientific field')
                    atomic_torch_save(bridged,directory/(name+'.pt'))
                    require(same(bridged,torch.load(directory/(name+'.pt'),weights_only=True,map_location='cpu')),'Bridge disk roundtrip differs')
            atomic_json(out/'BRIDGE.json',{'old_identity':OLD,'new_identity':newid,'only_configuration_change':{'max_epochs':[20,60]},'scientific_payloads_bitwise_unchanged':True,'latest_changed_fields':['identity'],'best_changed_fields':['identity','config.max_epochs'],'original_checkpoints':{str(JOB/'gru'/n):p['input_sha256'][str(JOB/'gru'/n)] for n in ('latest.pt','best.pt')}})
        else:
            bridge=json.loads((out/'BRIDGE.json').read_text());require(bridge['new_identity']==newid and bridge['old_identity']==OLD,'Wrong resumed bridge')
        check(p,30)
        before=copy.deepcopy(original_latest)
        control_summary=run_fit(windows,fit,stop,c20,control,dataid,device='cpu',resume=True)
        require(same(torch.load(control/'latest.pt',weights_only=True,map_location='cpu'),before),'No-update control changed state')
        require(same(control_summary['history'],original_latest['state']['history']),'Control history differs')
        calls=0;max_call=0.;times=[]
        while True:
            check(p,30)
            start=time.monotonic()
            summary=run_fit(windows,fit,stop,c60,extended,dataid,device='cpu',resume=True,stop_after_steps=1)
            elapsed=time.monotonic()-start;max_call=max(max_call,elapsed);times.append(elapsed);calls+=1
            status.update(training_updates=summary['steps']-360,calls_this_invocation=calls,max_batch_call_seconds=max_call,latest_training_status=summary['status'])
            if summary['status']=='completed':break
            if calls%18==0:atomic_json(out/'STATUS.json',status)
        selected=torch.load(extended/'best.pt',weights_only=True,map_location='cpu')
        require(selected['identity']==newid and summary['epochs']<=60,'Continuation selection identity differs')
        require(same(summary['history'][:20],original_latest['state']['history']),'Original history prefix changed')
        atomic_json(out/'SELECTED_BEFORE_OUTER.json',{'summary_sha256':sha(extended/'training_summary.json'),'best_sha256':sha(extended/'best.pt'),'identity':newid,'best_epoch':summary['best_epoch'],'selection_frozen_epoch':time.time(),'outer_scored_yet':False})
        check(p,15)
        outer=read_split('validation')
        from multiscale.training import digest_indices
        require(outer.dtype==np.int64 and outer.shape==(1356,) and np.all(np.diff(outer)>0),'Bad outer cohort')
        require(digest_indices(outer)==receipt['identity_fields']['validation_indices_sha256'],'Outer canonical identity differs')
        require(windows.bars.end_ns[stop[-1]+8]<windows.bars.end_ns[outer[0]],'Stop labels overlap outer')
        truth=targets(windows.bars,outer,8,'future_log_return')
        oldpred=predict_best(control/'best.pt',windows,outer,OLD,device='cpu',batch_size=512)
        newpred=predict_best(extended/'best.pt',windows,outer,newid,device='cpu',batch_size=512)
        score={'cap20':metrics(oldpred,truth),'cap60':metrics(newpred,truth),'zero':metrics(np.zeros_like(truth),truth)}
        require(abs(score['cap20']['endpoint_mse']-receipt['models']['gru']['metrics']['endpoint_mse'])<=1e-15,'Original outer metric not reproduced')
        score['endpoint_gain_cap60_vs_cap20_percent']=100*(1-score['cap60']['endpoint_mse']/score['cap20']['endpoint_mse'])
        score['endpoint_gain_cap60_vs_zero_percent']=100*(1-score['cap60']['endpoint_mse']/score['zero']['endpoint_mse'])
        atomic_json(out/'OUTER_SCORES.json',{'development_only':True,'independent_market_confirmation':False,'scores':score,'outer_count':len(outer),'selection_frozen_before_scoring':True})
        np.savez_compressed(out/'DEVELOPMENT_PREDICTIONS.npz',origins=outer,cap20=oldpred,cap60=newpred,truth=truth)
        for name in ('latest.pt','best.pt'):require(sha(JOB/'gru'/name)==p['input_sha256'][str(JOB/'gru'/name)],'Original checkpoint modified')
        check(p,5)
        status.update(status='completed_single_cap_diagnostic',completed_epoch=time.time(),old_identity=OLD,new_identity=newid,cap20_summary=control_summary,cap60_summary=summary,scores=score,outer_scored=True,independent_market_confirmation=False,predictive_method_success=False,full_goal_achieved=False,max_call_seconds=max_call,call_count=calls)
        atomic_json(out/'RESULTS.json',status);atomic_json(out/'STATUS.json',status)
    except BaseException as exc:
        status.update(status='paused_deadline_or_signal' if isinstance(exc,TimeoutError) else 'failed',error_type=type(exc).__name__,error=str(exc),same_deadline_on_resume=True)
        atomic_json(out/'STATUS.json',status)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);args=parser.parse_args()
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):signal.signal(sig,signalled)
    path=Path(args.protocol).resolve();run(json.loads(path.read_text()),path)
