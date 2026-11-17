"""Independent post-termination replay of this stage's final market artifacts.

No training helper/native-core import. Original AST admission is Recurrent only.
Python audit hooks are an explicit file/source boundary, not an OS sandbox.
"""
import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import time
import zipfile


def require(value,message):
    if not bool(value): raise RuntimeError(message)


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary,path)


def check_deadline(rules):
    require(time.time()<rules['deadline_epoch'],'Original market absolute deadline exhausted; no replay extension')


def preflight(args):
    rules_path=args.rules.resolve()
    require(sha(rules_path)==args.rules_sha256,'Replay rule identity changed')
    rules=read_json(rules_path)
    require(sha(__file__)==rules['script_sha256'],'Replay source changed')
    protocol_path=Path(rules['market_protocol']['path']).resolve()
    require(sha(protocol_path)==rules['market_protocol']['sha256'],'Original market protocol changed')
    protocol=read_json(protocol_path)
    require(protocol['phase']=='market' and protocol['market_admission'] is True,'Not an admitted original market protocol')
    require(rules['deadline_epoch']==protocol['deadline_epoch'],'Replay cannot reset original deadline')
    require(protocol['fixed_epochs']==8 and protocol['planned_neural_fits']==12 and protocol['planned_optimizer_updates']==29520,'Original study size changed')
    check_deadline(rules)
    require(sorted(os.sched_getaffinity(0))==[24],'Replay requires CPU24 alone')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1','GPU must be hidden')
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
        require(os.environ.get(name)=='1','Numerical thread limit not frozen: '+name)
    stage=Path(rules['stage']).resolve()
    metadata_path=Path(protocol['input_metadata_path']).resolve()
    require(sha(metadata_path)==protocol['input_metadata_sha256']==rules['input_metadata']['sha256'],'Six-input metadata changed')
    require(str(metadata_path)==rules['input_metadata']['path'],'Metadata path changed')
    metadata=read_json(metadata_path)
    expected={r['path']:r['sha256'] for r in metadata['bars'].values()}
    expected.update({r['path']:r['file_sha256'] for group in metadata['fits'].values() for r in group.values()})
    require(expected==protocol['market_files']==rules['market_files'] and len(expected)==6,'Replay requires precisely the admitted six caches')
    require(set(protocol['fixed_states'])=={s['id'] for s in rules['states']} and len(rules['states'])==4,'Fixed state cohort changed')
    market_root=Path(protocol['output_directory']).resolve()
    require(market_root==Path(rules['market_output_root']).resolve() and market_root.is_relative_to(stage),'Wrong new-stage output root')
    own=market_root/'market'
    controller_path=market_root/'EXECUTION_RECEIPT.json'
    controller=read_json(controller_path)
    require(controller['protocol_sha256']==rules['market_protocol']['sha256'], 'Terminal controller belongs to another protocol')
    require(controller['status']=='completed_worker_sequence_pending_independent_acceptance' and not controller.get('error'),'Original training controller has not terminated successfully')
    events=[e for e in controller['events'] if e['mode']=='market']
    require(len(events)==1 and events[0]['exit_code']==0 and events[0].get('finished_epoch'),'Original market worker not in successful terminal state')
    worker_pid=events[0]['pid']
    try:
        os.kill(worker_pid,0)
    except ProcessLookupError:
        pass
    else:
        raise RuntimeError('Original training worker PID still exists; do not overlap numerical jobs')
    final_results_path=own/'RESULTS.json'
    final_results=read_json(final_results_path)
    require(final_results['status']=='completed_all_fixed_states' and final_results['market_read'] is True and final_results['Q3_read'] is False,'Global market terminal results missing')
    require({r['state'] for r in final_results['results']}=={s['id'] for s in rules['states']} and len(final_results['results'])==4,'Global final cohort incomplete')
    freeze_path=own/'GLOBAL_FREEZE.json'; freeze=read_json(freeze_path)
    require(freeze['all_models_controls_frozen'] is True and freeze['learner_count']==12 and freeze['control_state_count']==4,'Original global pre-score freeze incomplete')
    require(set(freeze['states'])=={s['id'] for s in rules['states']} and len(freeze['final_fits'])==4,'Wrong global freeze states')
    require(freeze['protocol_sha256']==rules['market_protocol']['sha256'],'Wrong global freeze protocol')
    require(freeze['single_global_postfreeze_tail_pass'] is True and freeze['prediction_path_reloaded_from_committed_payloads'] is True,'Original global freeze does not qualify final reload/tail sequence')
    allowed={rules_path,Path(__file__).resolve(),protocol_path,metadata_path,controller_path,final_results_path,freeze_path}
    commits={}
    for spec in rules['states']:
        directory=(own/spec['id']).resolve()
        require(directory.parent==own,'State directory escaped new output')
        manifest_path=directory/'CHECKPOINT.json'; summary_path=directory/'SUMMARY.json'
        manifest=read_json(manifest_path); summary=read_json(summary_path)
        require(re.fullmatch(r'payload_[0-9]{8}_[0-9a-f]{32}\.pt',manifest['payload']) is not None,'Invalid committed own-payload name')
        payload=(directory/manifest['payload']).resolve()
        require(payload.parent==directory,'Own payload escaped exact state directory')
        prediction=(directory/'PREDICTIONS.npz').resolve()
        require(prediction.parent==directory,'Own prediction path escaped state directory')
        identity=manifest['identity']
        require(identity['protocol_sha256']==rules['market_protocol']['sha256'] and identity['state']==spec['id'] and identity['epochs']==8 and identity['batch']==128 and identity['seed']==2026,'Final checkpoint protocol identity mismatch')
        require(identity['code_files']==protocol['code_files'] and identity['source_files']==protocol['source_files'],'Final checkpoint source identity mismatch')
        n=spec['prefix_count']; updates=((n+127)//128-int(n%128==1))*8
        require(manifest['epoch']==8 and manifest['steps']==updates,'Final checkpoint update budget incomplete')
        require(summary['status']=='complete' and summary['epochs']==8 and summary['steps']==updates and summary['actual_optimizer_updates']==3*updates,'State final summary incomplete')
        require(summary['manifest']==manifest and summary['market_read'] is True and summary['synthetic'] is False and summary['Q3_read'] is False,'Summary does not commit current final own checkpoint')
        for key in ['original_fit_count','boundary','prefix_count','tail_count','first_tail_origin','prefix_sha256','tail_sha256']:
            require(identity['partition'][key]==spec[key],'Checkpoint partition differs from fixed design: '+key)
        frozen=next(r for r in freeze['final_fits'] if r['state']==spec['id'])
        require(frozen['phase']=='prefix' and frozen['dedicated_tail_rows_before_freeze']==0 and frozen['common_batches']==updates and frozen['actual_optimizer_updates']==3*updates,'Pre-score phase or update receipt mismatch')
        require(frozen['checkpoint']['identity']==identity,'Global freeze did not cover this final model identity')
        commits[spec['id']]={'directory':directory,'manifest':manifest,'payload':payload,'prediction':prediction,'summary':summary}
        allowed.update([manifest_path,summary_path,payload,prediction])
    model_path=Path(rules['original_Recurrent']['path']).resolve()
    require(protocol['source_files'][str(model_path)]==rules['original_Recurrent']['sha256'],'Original Recurrent source not admitted')
    allowed.add(model_path)
    output=Path(rules['output_path']).resolve()
    require(output.is_relative_to(stage/'review') and not output.exists(),'Replay report must be fresh and inside this stage review')
    temporary=output.with_suffix(output.suffix+'.tmp')
    runtime_roots={Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve(),Path(rules['repository_venv']).resolve()}
    runtime_roots.update(Path(p).resolve() for p in rules['readonly_installed_package_roots'])
    system_roots=[Path(p) for p in ['/usr','/lib','/lib64','/opt','/proc','/sys','/dev','/etc','/tmp']]
    market_paths={Path(p).resolve() for p in expected}
    def audit(event,values):
        if event in ('socket.connect','socket.bind','socket.getaddrinfo'):
            raise PermissionError('Independent replay admits no network')
        if event!='open' or not values or isinstance(values[0],int): return
        raw=values[0]
        if not isinstance(raw,(str,bytes,os.PathLike)): return
        path=Path(os.fsdecode(raw)).resolve()
        mode=values[1] if len(values)>1 else 'r'
        flags=values[2] if len(values)>2 and isinstance(values[2],int) else 0
        write=(isinstance(mode,str) and any(x in mode for x in ['w','a','+','x'])) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if write:
            if path in {output,temporary} or path.is_relative_to(Path('/tmp')) or str(path)=='/dev/null': return
            raise PermissionError('Replay write outside own report/temp: '+str(path))
        if path in allowed or path in market_paths: return
        if any(path.is_relative_to(root) for root in runtime_roots): return
        if any(path.is_relative_to(root) for root in system_roots): return
        raise PermissionError('Replay read outside exact source/six caches/own final artifacts: '+str(path))
    sys.dont_write_bytecode=True; sys.addaudithook(audit)
    check_deadline(rules)
    require(sha(model_path)==rules['original_Recurrent']['sha256'],'Original Recurrent source bytes changed')
    input_receipts=[]
    for path,expected_sha in expected.items():
        check_deadline(rules); require(sha(path)==expected_sha,'Original six-cache bytes changed: '+path)
        input_receipts.append({'path':path,'sha256':expected_sha})
    for item in commits.values():
        require(sha(item['payload'])==item['manifest']['sha256'],'Own committed final payload byte identity mismatch')
        item['prediction_sha256']=sha(item['prediction'])
    return rules,protocol,metadata,commits,output,input_receipts,worker_pid


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--rules',required=True,type=Path); parser.add_argument('--rules-sha256',required=True); args=parser.parse_args()
    rules,protocol,metadata,commits,output,input_receipts,worker_pid=preflight(args)
    import numpy as np
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    runtime={'torch':str(torch.__version__),'mkldnn_enabled':torch.backends.mkldnn.enabled,'mkldnn_available':torch.backends.mkldnn.is_available(),'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'threads':torch.get_num_threads(),'interop':torch.get_num_interop_threads(),'CPU':sorted(os.sched_getaffinity(0)),'GPU_hidden':os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'}
    check_deadline(rules)
    source=Path(rules['original_Recurrent']['path'])
    nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Recurrent']
    require(len(nodes)==1,'Exactly one original Recurrent class required')
    namespace={'nn':torch.nn,'torch':torch,'__name__':'_independent_market_recurrent'}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(source)+':Recurrent-only','exec'),namespace)
    Recurrent=namespace['Recurrent']
    checks=[]
    def compare(name,actual,expected,atol=1e-10,rtol=1e-8):
        actual=np.asarray(actual,dtype=np.float64); expected=np.asarray(expected,dtype=np.float64)
        require(actual.shape==expected.shape and actual.size>0,name+' shape/empty mismatch')
        require(np.isfinite(actual).all() and np.isfinite(expected).all(),name+' nonfinite')
        error=np.abs(actual-expected); bound=atol+rtol*np.maximum(np.abs(actual),np.abs(expected))
        okay=bool(np.all(error<=bound))
        checks.append({'name':name,'passed':okay,'max_abs':float(error.max()),'max_normalized':float((error/bound).max()),'atol':atol,'rtol':rtol})
        require(okay,name+' numerical replay mismatch; no tolerance rescue')
    def member(path,name):
        check_deadline(rules)
        with zipfile.ZipFile(path,'r') as archive:
            require(archive.namelist().count(name)==1,'Missing/duplicate literal npy member '+name)
            with archive.open(name) as stream: return np.load(io.BytesIO(stream.read()),allow_pickle=False)
    def indices_sha(indices): return hashlib.sha256(np.asarray(indices,dtype='<i8').tobytes()).hexdigest()
    bars={}
    for profile in ['minute','hourly']:
        record=metadata['bars'][profile]
        fields={k:member(record['path'],k+'.npy') for k in ['start_ns','end_ns','close','observed','session_date']}
        n=len(fields['close'])
        require(n>0 and all(a.ndim==1 and len(a)==n for a in fields.values()),'Raw bar dimensions invalid')
        require(fields['start_ns'].dtype==np.int64 and fields['end_ns'].dtype==np.int64 and fields['close'].dtype==np.float64 and fields['observed'].dtype==np.bool_ and fields['session_date'].dtype.kind=='U','Raw bar dtypes invalid')
        require(np.all(np.diff(fields['start_ns'])>0) and np.all(np.diff(fields['end_ns'])>0) and np.all(fields['start_ns']<fields['end_ns']) and np.all(fields['start_ns'][1:]>=fields['end_ns'][:-1]),'Raw bar intervals invalid')
        require(np.all(fields['session_date']<='2026-06-30') and np.all(fields['session_date'][1:]>=fields['session_date'][:-1]),'Q3 or unordered raw date')
        require(np.all(fields['end_ns']<=1782864000000000000),'Raw timestamps cross July1UTC')
        observed=fields['observed']; require(np.isfinite(fields['close'][observed]).all() and np.all(fields['close'][observed]>0),'Invalid observed nominal closes')
        logclose=np.full(n,np.nan,dtype=np.float64); logclose[observed]=np.log(fields['close'][observed])
        split=np.zeros(n,dtype=np.float64)
        previous=None
        for action in metadata['split_actions']:
            date,ratio=str(action['date']),float(action['ratio'])
            require(previous is None or previous<date,'Unordered split actions'); require(np.isfinite(ratio) and ratio>0,'Invalid split ratio')
            split[fields['session_date']>=date]+=np.log(ratio); previous=date
        fields['logclose']=logclose; fields['split']=split; bars[profile]=fields
    ledgers=[]; partitions={}; states=[]
    for spec in rules['states']:
        check_deadline(rules)
        item=commits[spec['id']]; record=metadata['fits'][spec['profile']][str(spec['fold'])]
        fit=member(record['path'],'fit.npy')
        require(fit.dtype==np.int64 and fit.ndim==1 and len(fit)==record['count']==spec['original_fit_count'] and np.all(np.diff(fit)>0),'Original fit member dimensions/order/count invalid')
        require(indices_sha(fit)==record['fit_sha256'],'Original fit canonical digest changed')
        raw=bars[spec['profile']]; L,H=spec['lookback'],spec['horizon']
        require(fit[0]>=L-1 and fit[-1]+H<len(raw['close']),'Original fit outside input bars')
        boundary=4*len(fit)//5; tail=fit[boundary:]; before=fit[:boundary]
        prefix=before[raw['end_ns'][before+H]<raw['end_ns'][tail[0]]]
        partition={'original_fit_count':len(fit),'boundary':boundary,'prefix_count':len(prefix),'tail_count':len(tail),'first_tail_origin':int(tail[0]),'prefix_sha256':indices_sha(prefix),'tail_sha256':indices_sha(tail)}
        for key,value in partition.items(): require(value==spec[key]==item['manifest']['identity']['partition'][key],'Independent strict-purge partition mismatch: '+key)
        partitions[spec['id']]={'profile':spec['profile'],'prefix':prefix,'tail':tail}
        state= torch.load(item['payload'],map_location='cpu',weights_only=True)
        require(state['identity']==item['manifest']['identity'] and state['state']['epoch']==8 and state['state']['steps']==item['manifest']['steps'],'Own final payload identity/epoch mismatch')
        require(state['state']['permutation'] is None and state['state']['cursor']==0 and state['state']['seen']==0 and len(state['state']['history'])==8,'Own fit is not a completed eight-epoch state')
        require(state['identity']['backend_identity']==runtime,'Replay backend differs from original; no toggle rescue')
        require(state['evaluation'] is not None and state['evaluation']['state']==spec['id'] and state['evaluation']['count']==len(tail) and state['evaluation']['horizon']==H,'Own checkpoint lacks complete final evaluation')
        require(set(state['learners'])=={'D','Z','C'} and state['controls'] is not None,'Own final learner/control set incomplete')
        require(state['modes']=={'D':False,'Z':False,'C':False,'reference':False},'Saved final models were not in eval mode')
        with np.load(item['prediction'],allow_pickle=False) as archive:
            expected={'origins','labels','D','Z','C','zero','prefix_update_mean','same_information_Ridge'}
            require(set(archive.files)==expected,'Saved own prediction members changed')
            saved={key:archive[key] for key in archive.files}
        require(saved['origins'].dtype==np.int64 and np.array_equal(saved['origins'],tail),'Saved prediction origins differ from original tail')
        require(all(saved[key].shape==(len(tail),H) and np.isfinite(saved[key]).all() for key in expected-{'origins'}),'Saved prediction shape/finiteness invalid')
        def arrays(origins):
            locations=origins[:,None]+np.arange(1-L,1)
            require(raw['observed'][locations].all(),'Original history has missing observations')
            history=raw['logclose'][locations]+raw['split'][locations]-raw['split'][origins,None]
            x=history-raw['logclose'][origins,None]
            anchor=raw['close'][origins,None]
            require(np.isfinite(x).all() and np.isfinite(anchor).all(),'Rebuilt original features nonfinite')
            return x,anchor
        def labels(origins):
            future=origins[:,None]+np.arange(1,H+1)
            require(raw['observed'][origins].all() and raw['observed'][future].all(),'Original path has missing observations')
            if spec['profile']=='minute': require(np.all(raw['session_date'][origins]==raw['session_date'][origins+H]),'Minute target crosses session')
            return np.log(raw['close'][future]/raw['close'][origins,None])
        xsum=xsq=asum=asq=0.; ysum=np.zeros(H); ysq=np.zeros(H)
        for first in range(0,len(prefix),4096):
            check_deadline(rules); x,a=arrays(prefix[first:first+4096]); y=labels(prefix[first:first+4096])
            xsum+=float(x.sum(dtype=np.float64)); xsq+=float(np.square(x).sum(dtype=np.float64))
            asum+=float(a.sum(dtype=np.float64)); asq+=float(np.square(a).sum(dtype=np.float64))
            ysum+=y.sum(axis=0,dtype=np.float64); ysq+=np.square(y).sum(axis=0,dtype=np.float64)
        n=len(prefix); xm=xsum/(n*L); am=asum/n; ym=ysum/n
        xs=np.sqrt(max(xsq/(n*L)-xm**2,0.)); astdev=np.sqrt(max(asq/n-am**2,0.)); ys=np.sqrt(np.maximum(ysq/n-ym**2,0.))
        xs=float(xs if xs>1e-12 else 1.); astdev=float(astdev if astdev>1e-12 else 1.); ys=np.where(ys>1e-12,ys,1.)
        scaler={'x_mean':float(xm),'x_std':xs,'anchor_mean':float(am),'anchor_std':astdev,'y_mean':ym,'y_std':ys}
        require(state['scaler']['fitted_origin_count']==n,'Saved scaler fit count differs from purged prefix')
        for key,value in scaler.items(): compare(spec['id']+':scaler:'+key,value,state['scaler'][key])
        def design(origins):
            x,a=arrays(origins)
            return np.column_stack((((x-xm)/xs).astype(np.float32),((a-am)/astdev).astype(np.float32))).astype(np.float64)
        sx=np.zeros(L+1); sxx=np.zeros(L+1); ry=np.zeros(H)
        for first in range(0,n,4096):
            check_deadline(rules); origins=prefix[first:first+4096]; X=design(origins); Y=labels(origins)
            sx+=X.sum(axis=0,dtype=np.float64); sxx+=np.square(X).sum(axis=0,dtype=np.float64); ry+=Y.sum(axis=0,dtype=np.float64)
        rmean=sx/n; second=sxx/n; variance=second-rmean**2; ymean=ry/n
        allowance=1e-10*np.maximum(1.,np.maximum(np.abs(second),rmean**2))
        require(np.all(variance>=-allowance),'Independent Ridge variance beyond frozen tolerance')
        rsd=np.sqrt(np.maximum(variance,0.)); rsd=np.where(rsd>1e-12,rsd,1.)
        gram=np.zeros((L+1,L+1)); cross=np.zeros((L+1,H))
        for first in range(0,n,4096):
            check_deadline(rules); origins=prefix[first:first+4096]; Z=(design(origins)-rmean)/rsd
            gram+=Z.T@Z; cross+=Z.T@(labels(origins)-ymean)
        A=gram/n+.1*np.eye(L+1); R=cross/n; coefficient=np.linalg.solve(A,R)
        residual=float(np.linalg.norm(A@coefficient-R,ord=np.inf)); denominator=float(np.linalg.norm(A,ord=np.inf)*np.linalg.norm(coefficient,ord=np.inf)+np.linalg.norm(R,ord=np.inf))
        backward=residual/denominator if denominator else (0. if residual==0 else float('inf'))
        require(np.isfinite(coefficient).all() and np.isfinite(backward) and backward<=1e-10,'Independent fixed Ridge stationarity gate failed')
        controls=state['controls']
        require(controls['normalized_lambda']==.1 and controls['observations']==n and controls['backward_error']<=1e-10,'Saved fixed Ridge metadata invalid')
        for key,value in [('coefficients',coefficient),('x_mean',rmean),('x_std',rsd),('y_mean',ymean)]:
            compare(spec['id']+':Ridge:'+key,value,controls[key].numpy())
        baseline=torch.as_tensor(-np.asarray(state['scaler']['y_mean'])/np.asarray(state['scaler']['y_std']),dtype=torch.float32)
        require(torch.equal(state['baseline'],baseline),'Saved baseline not original FP64 ratio then FP32')
        torch.manual_seed(2026); initial=Recurrent('lstm',H,width=32).float()
        initial_dict=initial.state_dict()
        require(set(initial_dict)==set(state['reference_initial']) and all(torch.equal(value,state['reference_initial'][key]) for key,value in initial_dict.items()),'Saved C reference_initial is not original same-seed theta0')
        reference=Recurrent('lstm',H,width=32).float(); reference.load_state_dict(state['reference']); reference.eval()
        require(all(torch.equal(value,state['reference_initial'][key]) for key,value in reference.state_dict().items()),'Saved C reference changed during training')
        for p in reference.parameters(): p.requires_grad_(False)
        models={}
        for arm in ['D','Z','C']:
            model=Recurrent('lstm',H,width=32).float(); model.load_state_dict(state['learners'][arm]); model.eval()
            require(all(torch.isfinite(p).all() for p in model.parameters()),'Nonfinite saved final learner')
            for p in model.parameters(): p.requires_grad_(False)
            models[arm]=model
        batches=0; neural_physical_max={arm:0. for arm in models}
        for first in range(0,len(tail),512):
            check_deadline(rules); origins=tail[first:first+512]; actual_y=labels(origins); saved_slice=slice(first,first+len(origins))
            require(np.array_equal(actual_y,saved['labels'][saved_slice]),'Independent raw nominal tail labels differ from saved labels')
            rawx,rawanchor=arrays(origins); x=((rawx-xm)/xs).astype(np.float32); a=((rawanchor-am)/astdev).astype(np.float32)
            tx,ta=torch.from_numpy(x[...,None]),torch.from_numpy(a)
            with torch.no_grad():
                reference_q=reference(tx,ta)
                for arm,model in models.items():
                    q=model(tx,ta)
                    if arm=='C': q=(q-reference_q)+baseline
                    saved_q=(saved[arm][saved_slice]-np.asarray(state['scaler']['y_mean']))/np.asarray(state['scaler']['y_std'])
                    compare(spec['id']+':batch%d:q:%s'%(batches,arm),q.numpy(),saved_q,atol=1e-7,rtol=1e-5)
                    physical=q.numpy()*ys+ym
                    neural_physical_max[arm]=max(neural_physical_max[arm],float(np.max(np.abs(physical-saved[arm][saved_slice]))))
            require(np.count_nonzero(saved['zero'][saved_slice])==0,'Saved zero control is not zero')
            compare(spec['id']+':batch%d:prefixmean'%batches,np.broadcast_to(ym,actual_y.shape),saved['prefix_update_mean'][saved_slice])
            ridge=((np.column_stack((x,a)).astype(np.float64)-rmean)/rsd)@coefficient+ymean
            compare(spec['id']+':batch%d:Ridgeprediction'%batches,ridge,saved['same_information_Ridge'][saved_slice])
            batches+=1
        states.append({'state':spec['id'],'partition':partition,'native_prefix_scaler_recomputed':True,'independent_Ridge_refit':True,'Ridge_backward_error':backward,'tail_batches':batches,'neural_standardized_q_replayed':['D','Z','C'],'physical_prediction_max_abs_difference_descriptive':neural_physical_max,'raw_tail_labels_exact':True,'own_payload_sha256':item['manifest']['sha256'],'own_prediction_sha256':item['prediction_sha256'],'eight_epochs':True,'training_added':False})
        del models,reference,initial,state,saved
    for a,a_rows in partitions.items():
        for b,b_rows in partitions.items():
            if a!=b and a_rows['profile']==b_rows['profile']:
                ledgers.append({'prefix_state':a,'tail_state':b,'origin_intersection_count':int(np.intersect1d(a_rows['prefix'],b_rows['tail']).size)})
    check_deadline(rules)
    # Detect a late concurrent writer rather than bless changing own final artifacts.
    for item in commits.values():
        require(sha(item['payload'])==item['manifest']['sha256'] and sha(item['prediction'])==item['prediction_sha256'],'Own final artifacts changed during replay')
        require(read_json(item['directory']/'CHECKPOINT.json')==item['manifest'],'Final checkpoint pointer changed during replay')
    report={'status':'passed_independent_final_market_preprocessing_Ridge_and_neural_output_replay','rules_sha256':args.rules_sha256,'script_sha256':rules['script_sha256'],'market_protocol_sha256':rules['market_protocol']['sha256'],'deadline_epoch':rules['deadline_epoch'],'deadline_reset':False,'original_training_worker_pid_exited_before_replay':worker_pid,'runtime':runtime,'original_six_inputs':input_receipts,'whole_preQ3_raw_bar_fields_decoded':True,'raw_fields_decoded':['start_ns','end_ns','close','observed','session_date'],'fit_archive_members_decoded':['fit.npy'],'old_weights_or_predictions_read':False,'only_this_stage_final_weights_predictions_read':True,'Recurrent_only_original_AST':True,'worker_or_native_core_imported':False,'new_training_updates':0,'states':states,'cross_state_origin_membership':ledgers,'not_globally_blind':True,'checks_passed':len(checks),'checks':checks,'comparison_tolerances':rules['comparison_tolerances'],'scientific_decision_changed':False,'separate_saved_prediction_score_verification_required':True,'Q3_read':False,'forecast_success':False,'full_goal_achieved':False}
    atomic_json(output,report)
    print(json.dumps({'status':report['status'],'checks_passed':len(checks),'states':len(states)}))

if __name__=='__main__': main()
