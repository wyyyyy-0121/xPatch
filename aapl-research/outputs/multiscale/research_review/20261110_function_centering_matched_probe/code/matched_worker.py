"""Original-native D/Z/C lockstep; preparation synthetic only unless separately admitted."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import uuid

import numpy as np
import torch

from native_core import (require, digest, original_pipeline, synthetic_state,
                         fixed_ridge, ridge_predict, control_fit, load_market)

PAUSE = False

def signal_pause(signum, frame):
    global PAUSE
    PAUSE = True


def json_write(path, value):
    path = Path(path)
    temp = path.with_name(path.name+'.tmp')
    with temp.open('w') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp,path)


def save_checkpoint(output, payload):
    """Immutable generation, then atomic manifest pointer: old committed pair survives."""
    name = 'payload_%08d_%s.pt' % (payload['state']['steps'],uuid.uuid4().hex)
    path = output/name
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        torch.save(payload,stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary,path)
    manifest = {'schema':1,'payload':name,'sha256':digest(path),'steps':payload['state']['steps'],
                'epoch':payload['state']['epoch'],'identity':payload['identity']}
    json_write(output/'CHECKPOINT.json',manifest)
    return manifest


def load_checkpoint(output):
    pointer = output/'CHECKPOINT.json'
    require(pointer.is_file(),'Missing committed checkpoint: never silently reinitialize')
    manifest = json.loads(pointer.read_text())
    path = output/manifest['payload']
    require(path.parent == output and path.is_file() and digest(path) == manifest['sha256'], 'Corrupt checkpoint manifest/payload')
    value = torch.load(path,map_location='cpu',weights_only=True)
    require(value['identity'] == manifest['identity'] and value['state']['steps'] == manifest['steps'], 'Manifest semantic mismatch')
    return value


def make_arms(native, H, scaler):
    torch.manual_seed(2026)
    default = native['Recurrent']('lstm',H,width=32).float()
    arms = {'D':default,'Z':copy.deepcopy(default),'C':copy.deepcopy(default)}
    reference = copy.deepcopy(default).eval()
    for p in reference.parameters(): p.requires_grad_(False)
    initial = {k:v.detach().clone() for k,v in reference.state_dict().items()}
    baseline = torch.as_tensor(-np.asarray(scaler.y_mean)/np.asarray(scaler.y_std),dtype=torch.float32)
    with torch.no_grad():
        arms['Z'].head.weight.zero_(); arms['Z'].head.bias.copy_(baseline)
    optimizers = {a:torch.optim.Adam(m.parameters(),lr=.001,weight_decay=0,betas=(.9,.999),eps=1e-8,amsgrad=False) for a,m in arms.items()}
    return arms,reference,initial,baseline,optimizers


def forward(arm, arms, reference, baseline, x, anchor):
    value = arms[arm](x,anchor)
    if arm == 'C':
        with torch.no_grad(): fixed = reference(x,anchor)
        value = (value-fixed)+baseline
    return value


def reference_gate(arms, reference, initial, optimizers):
    for name,value in reference.state_dict().items():
        require(torch.equal(value,initial[name]),'Reference state mutated: '+name)
    forbidden = {id(p) for p in reference.parameters()}
    pointers = {p.untyped_storage().data_ptr() for p in reference.parameters()}
    for p in reference.parameters(): require(p.grad is None and not p.requires_grad,'Reference has gradient')
    for arm,model in arms.items():
        require(all(p.untyped_storage().data_ptr() not in pointers for p in model.parameters()),'Reference storage alias')
        require(not any(id(p) in forbidden for group in optimizers[arm].param_groups for p in group['params']),'Reference in Adam')


def qualify_initial(native, adapter, scaler):
    arms,reference,initial,baseline,opts = make_arms(native,adapter.horizon,scaler)
    x,a,y = adapter.batch(scaler,adapter.prefix[:8])
    q = {key:forward(key,arms,reference,baseline,x,a) for key in arms}
    mu,sigma = torch.tensor(scaler.y_mean,dtype=torch.float64),torch.tensor(scaler.y_std,dtype=torch.float64)
    bound = 32*torch.finfo(torch.float32).eps*max(1.,float(mu.abs().max()),float(sigma.abs().max()))
    checks = []
    for key in ['Z','C']:
        error = float((q[key].double()*sigma+mu).abs().max())
        require(error <= bound,'Initial physicalzero failure '+key)
        checks.append({'check':key+'_physical_zero','max_abs':error,'bound':bound})
    require(torch.allclose(q['Z'],q['C'],atol=1e-7,rtol=1e-5),'Z/C initial function mismatch')
    for name,tensor in arms['D'].state_dict().items(): require(torch.equal(tensor,arms['C'].state_dict()[name]),'D/C theta0 mismatch')
    for label,v in [('ones',torch.ones_like(q['D'])),('sin',torch.sin(torch.arange(q['D'].numel()).reshape_as(q['D'])+1.)),('cos',torch.cos(torch.arange(q['D'].numel()).reshape_as(q['D'])+1.))]:
        grads = {}
        for key in ['D','C']:
            prediction = forward(key,arms,reference,baseline,x,a)
            grads[key] = torch.autograd.grad((prediction*v).sum(),tuple(arms[key].parameters()))
        maximum = 0.
        for gd,gc in zip(grads['D'],grads['C']):
            require(torch.allclose(gd,gc,atol=1e-7,rtol=1e-5),'D/C fixed-cotangent VJP mismatch')
            maximum = max(maximum,float((gd-gc).abs().max()))
        checks.append({'check':'VJP_'+label,'max_abs':maximum})
    grads = {}
    for key in arms:
        opts[key].zero_grad(set_to_none=True)
        loss = (forward(key,arms,reference,baseline,x,a)-y).square().mean()
        loss.backward()
        grads[key] = {n:p.grad.detach().clone() for n,p in arms[key].named_parameters()}
        require(all(torch.isfinite(v).all() for v in grads[key].values()),'Initial nonfinite gradient')
    zbone = max(float(v.abs().max()) for n,v in grads['Z'].items() if n.startswith('rnn.'))
    cbone = max(float(v.abs().max()) for n,v in grads['C'].items() if n.startswith('rnn.'))
    require(zbone == 0. and cbone > 10*(1e-7+1e-5*cbone),'Zzero/Cnonzero first-backbone gate')
    for name in ['head.weight','head.bias']:
        require(torch.allclose(grads['Z'][name],grads['C'][name],atol=1e-7,rtol=1e-5),'Z/C initial head gradient mismatch')
    reference_gate(arms,reference,initial,opts)
    return {'status':'passed_native_initial_identities','checks':checks,'Z_firstbone_inf':zbone,'C_firstbone_inf':cbone,'reference_immutable':True,'forecast_success':False}


def fixture_payload(adapter):
    return {k:torch.from_numpy(getattr(adapter.bars,k).copy()) for k in ['start_ns','end_ns','close','observed']} | {'session_date':[str(value) for value in adapter.bars.session_date],'fit':torch.from_numpy(adapter.fit.copy()),'prefix':torch.from_numpy(adapter.prefix.copy()),'tail':torch.from_numpy(adapter.tail.copy())}


def train_state(native, adapter, protocol, output, epochs, pause_kind=None, resume=False, synthetic=False):
    output.mkdir(parents=True,exist_ok=True)
    if not resume:
        require(not (output/'STARTED.json').exists(), 'Started state without committed resume must not reinitialize')
        json_write(output/'STARTED.json',{'state':adapter.id,'protocol_sha256':digest(protocol['_path'])})
    scaler = adapter.scaler()
    identity = {'protocol_sha256':digest(protocol['_path']),'state':adapter.id,'partition':adapter.partition,
                'epochs':epochs,'batch':128,'seed':2026,'backend_identity':{'torch':str(torch.__version__),'mkldnn_enabled':torch.backends.mkldnn.enabled,'mkldnn_available':torch.backends.mkldnn.is_available(),'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'threads':torch.get_num_threads(),'interop':torch.get_num_interop_threads(),'CPU':sorted(os.sched_getaffinity(0)),'GPU_hidden':os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'},'source_files':protocol['source_files'],'code_files':protocol['code_files']}
    arms,reference,initial,baseline,opts = make_arms(native,adapter.horizon,scaler)
    random.seed(2026)
    rng = np.random.default_rng(2026)
    state = {'epoch':0,'cursor':0,'permutation':None,'steps':0,'seen':0,'loss_sums':{a:0. for a in arms},'history':[]}
    controls = None
    evaluation = None
    if resume:
        saved = load_checkpoint(output)
        require(saved['identity'] == identity and saved['scaler'] == asdict(scaler),'Resume protocol/data/scaler changed')
        for key in arms:
            arms[key].load_state_dict(saved['learners'][key]); opts[key].load_state_dict(saved['optimizers'][key]); arms[key].train(saved['modes'][key])
        require(torch.equal(saved['baseline'],baseline), 'Saved baseline differs from original FP64-ratio initialization')
        require(set(saved['reference_initial']) == set(initial) and all(torch.equal(initial[k],saved['reference_initial'][k]) for k in initial), 'Saved reference_initial is not same seed theta0')
        reference.load_state_dict(saved['reference']); initial = saved['reference_initial']
        reference.train(saved['modes']['reference'])
        state = saved['state']; controls = saved['controls']; evaluation = saved['evaluation']
        rng.bit_generator.state = saved['numpy_rng']; random.setstate(saved['python_rng']); torch.set_rng_state(saved['torch_rng'])
    else:
        require(not (output/'CHECKPOINT.json').exists() and not list(output.glob('payload_*.pt')),'Existing state: explicit resume required')
    def payload():
        return {'schema':1,'identity':identity,'state':copy.deepcopy(state),'scaler':asdict(scaler),
                'learners':{a:m.state_dict() for a,m in arms.items()},'reference':reference.state_dict(),
                'reference_initial':initial,'baseline':baseline,'optimizers':{a:o.state_dict() for a,o in opts.items()},
                'modes':{a:m.training for a,m in arms.items()}|{'reference':reference.training},
                'numpy_rng':rng.bit_generator.state,'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),
                'controls':controls,'evaluation':evaluation,'fixture':fixture_payload(adapter) if synthetic else None}
    if not resume: save_checkpoint(output,payload())
    local_deadline = protocol['deadline_epoch']
    def pause_check():
        if PAUSE or time.time() >= local_deadline:
            save_checkpoint(output,payload())
            json_write(output/'PAUSED.json',{'reason':'signal' if PAUSE else 'expired_deadline','steps':state['steps'],'deadline_epoch':protocol['deadline_epoch'],'deadline_reset':False})
            return True
        return False
    while state['epoch'] < epochs:
        if pause_check(): return None,75
        if state['permutation'] is None:
            state['permutation'] = torch.from_numpy(rng.permutation(len(adapter.prefix)))
            state['cursor'] = 0
        while state['cursor'] < len(adapter.prefix):
            if pause_check(): return None,75
            first = state['cursor']; end = min(first+128,len(adapter.prefix))
            if len(adapter.prefix)-end == 1: end = len(adapter.prefix)
            indices = adapter.prefix[state['permutation'][first:end].numpy()]
            x,anchor,y = adapter.batch(scaler,indices)
            for key in ['D','Z','C']:
                arms[key].train(); opts[key].zero_grad(set_to_none=True)
                pred = forward(key,arms,reference,baseline,x,anchor)
                loss = (pred-y).square().mean()
                require(torch.isfinite(loss),'Nonfinite training loss')
                loss.backward()
                require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in arms[key].parameters()),'Missing/nonfinite training gradients')
                opts[key].step()
                require(all(torch.isfinite(p).all() for p in arms[key].parameters()),'Nonfinite parameter after update')
                state['loss_sums'][key] += float(loss.detach())*len(indices)
            reference_gate(arms,reference,initial,opts)
            state['cursor'] = end; state['steps'] += 1; state['seen'] += len(indices)
            if state['steps'] % 100 == 0: save_checkpoint(output,payload())
            if synthetic and pause_kind and state['steps'] == 4:
                if pause_kind == 'deadline': local_deadline = time.time()-1.
                else:
                    json_write(output/'READY_FOR_SIGNAL.json',{'pid':os.getpid(),'steps':4})
                    until = time.monotonic()+20.
                    while not PAUSE and time.monotonic() < until: time.sleep(.01)
                    require(PAUSE,'External SIGUSR1 did not arrive')
                if pause_check(): return None,75
        state['history'].append({'epoch':state['epoch']+1,'steps':state['steps'],
                                 'training_scaled_fullpath_loss':{a:state['loss_sums'][a]/state['seen'] for a in arms}})
        state['epoch'] += 1; state['cursor'] = 0; state['permutation'] = None
        state['seen'] = 0; state['loss_sums'] = {a:0. for a in arms}
        save_checkpoint(output,payload())
    # A complete training checkpoint is already committed before this noniterative solve.
    if pause_check(): return None,75
    if controls is None:
        control = control_fit(adapter,scaler)
        controls = {k:torch.from_numpy(v.copy()) if isinstance(v,np.ndarray) else v for k,v in control.items()}
        save_checkpoint(output,payload())
    if pause_check(): return None,75
    return {'adapter':adapter,'scaler':scaler,'arms':arms,'reference':reference,'reference_initial':initial,
            'baseline':baseline,'opts':opts,'state':state,'controls':controls,'payload':payload,'save':lambda:save_checkpoint(output,payload()),
            'output':output},0


def evaluate_runs(runs, protocol, output, synthetic=False):
    # Re-read every committed final fit, not a caller's boolean assertion.
    committed = []
    if synthetic:
        require(len(runs) == 1 and runs[0]['adapter'].id == 'synthetic_native', 'Wrong synthetic freeze state set')
    else:
        require({r['adapter'].id for r in runs} == {'minute_f1','minute_f2','hourly_f1','hourly_f2'} and len(runs) == 4, 'Global freeze requires exact four states and twelve learners')
    for run in runs:
        require(run['adapter'].target_calls['tail_rows'] == 0 and run['adapter'].phase == 'prefix', 'Dedicated tail labels constructed before global freeze')
        saved = load_checkpoint(run['output'])
        n = len(run['adapter'].prefix)
        per_epoch = (n+127)//128-int(n%128 == 1)
        require(saved['state']['epoch'] == saved['identity']['epochs'] and saved['state']['steps'] == per_epoch*saved['identity']['epochs'], 'Incomplete final model at global freeze')
        require(saved['controls'] is not None and set(saved['learners']) == {'D','Z','C'},'Missing global controls/models')
        require(saved['identity'] == run['payload']()['identity'] and saved['scaler'] == asdict(run['scaler']), 'Final reload identity/scaler mismatch')
        ram = run['payload']()
        for arm in ['D','Z','C']:
            require(all(torch.equal(value,saved['learners'][arm][name]) for name,value in run['arms'][arm].state_dict().items()), 'RAM/final learner mismatch')
        require(all(torch.equal(value,saved['reference'][name]) for name,value in run['reference'].state_dict().items()), 'RAM/final reference mismatch')
        for name,value in run['controls'].items():
            require(torch.equal(value,saved['controls'][name]) if isinstance(value,torch.Tensor) else value == saved['controls'][name], 'RAM/final control mismatch')
        run['scaler'] = type(run['scaler'])(**saved['scaler'])
        for arm in ['D','Z','C']:
            run['arms'][arm].load_state_dict(saved['learners'][arm])
            run['opts'][arm].load_state_dict(saved['optimizers'][arm])
            run['arms'][arm].train(saved['modes'][arm])
        run['reference'].load_state_dict(saved['reference'])
        run['reference'].train(saved['modes']['reference'])
        run['reference_initial'] = saved['reference_initial']
        run['baseline'] = saved['baseline']
        run['controls'] = saved['controls']
        reference_gate(run['arms'],run['reference'],run['reference_initial'],run['opts'])
        committed.append({'state':run['adapter'].id,'dedicated_tail_rows_before_freeze':run['adapter'].target_calls['tail_rows'],'phase':'prefix','common_batches':saved['state']['steps'],'actual_optimizer_updates':3*saved['state']['steps'],'checkpoint':json.loads((run['output']/'CHECKPOINT.json').read_text())})
    freeze = {'all_models_controls_frozen':True,'states':[r['adapter'].id for r in runs],
              'final_fits':committed,'learner_count':3*len(runs),'control_state_count':len(runs),'prediction_path_reloaded_from_committed_payloads':True,'protocol_sha256':digest(protocol['_path']),
              'single_global_postfreeze_tail_pass':True,'not_globally_blind':True}
    json_write(output/'GLOBAL_FREEZE.json',freeze)
    results = []
    for run in runs:
        adapter,scaler = run['adapter'],run['scaler']
        adapter.permit_tail(freeze)
        for model in run['arms'].values(): model.eval()
        sums = {key:{'endpoint_sse':0.,'path_sse':0.,'sum_error':0.,'sum_pred':0.,'sum_pred_square':0.} for key in ['D','Z','C','zero','prefix_update_mean','same_information_Ridge']}
        control = {k:v.numpy() if isinstance(v,torch.Tensor) else v for k,v in run['controls'].items()}
        ridge_predictions = []
        prediction_chunks = {key:[] for key in sums}
        label_chunks = []
        for first in range(0,len(adapter.tail),512):
            require(time.time() < protocol['deadline_epoch'],'Deadline during scoring: no partial results admitted')
            indices = adapter.tail[first:first+512]
            rawx,rawanchor = adapter.arrays(indices)
            x,anchor = scaler.transform_x(rawx,rawanchor)
            y = adapter.labels(indices)
            label_chunks.append(y)
            predictions = {'zero':np.zeros_like(y),'prefix_update_mean':np.broadcast_to(np.asarray(scaler.y_mean),y.shape),
                           'same_information_Ridge':ridge_predict(control,np.column_stack((x,anchor)))}
            ridge_predictions.append(predictions['same_information_Ridge'])
            with torch.no_grad():
                for key in run['arms']:
                    q = forward(key,run['arms'],run['reference'],run['baseline'],torch.from_numpy(x[...,None]),torch.from_numpy(anchor))
                    predictions[key] = scaler.invert_y(q.numpy())
            for key,pred in predictions.items():
                prediction_chunks[key].append(pred.copy())
                require(np.isfinite(pred).all(),'Nonfinite final tail prediction')
                err = pred-y
                sums[key]['endpoint_sse'] += float(np.square(err[:,-1]).sum(dtype=np.float64))
                sums[key]['path_sse'] += float(np.square(err).sum(dtype=np.float64))
                sums[key]['sum_error'] += float(err.sum(dtype=np.float64)); sums[key]['sum_pred'] += float(pred.sum(dtype=np.float64)); sums[key]['sum_pred_square'] += float(np.square(pred).sum(dtype=np.float64))
        record = {'state':adapter.id,'count':len(adapter.tail),'horizon':adapter.horizon,'predictors':{}}
        for key,stat in sums.items():
            count = len(adapter.tail)*adapter.horizon
            record['predictors'][key] = stat|{'endpoint_MSE':stat['endpoint_sse']/len(adapter.tail),'path_MSE':stat['path_sse']/count,'bias':stat['sum_error']/count,'prediction_variance':max(0.,stat['sum_pred_square']/count-(stat['sum_pred']/count)**2)}
        results.append(record)
        np.savez(run['output']/'PREDICTIONS.npz',origins=adapter.tail,labels=np.concatenate(label_chunks),**{key:np.concatenate(chunks) for key,chunks in prediction_chunks.items()})
        if synthetic:
            b = adapter.bars
            np.savez(output/'NATIVE_FIXTURE.npz',close=b.close,start_ns=b.start_ns,end_ns=b.end_ns,observed=b.observed,session_date=b.session_date,fit=adapter.fit,prefix=adapter.prefix,tail=adapter.tail,
                     ridge_coefficients=control['coefficients'],ridge_x_mean=control['x_mean'],ridge_x_std=control['x_std'],ridge_y_mean=control['y_mean'],ridge_tail_predictions=np.concatenate(ridge_predictions))
            json_write(output/'SCALER.json',{'scaler':asdict(scaler)})
    return freeze,results


def success_gate(results):
    state_rules=[]
    for row in results:
        p=row['predictors']; checks=[]
        for control in ['D','Z','zero','prefix_update_mean','same_information_Ridge']:
            for metric in ['endpoint_MSE','path_MSE']:
                v=p[control][metric]
                checks.append({'control':control,'metric':metric,'passed':bool(np.isfinite(v) and v>0 and p['C'][metric] <= .999*v)})
        state_rules.append({'state':row['state'],'checks':checks,'passed':all(x['passed'] for x in checks)})
    tasks=[]
    for prefix in ['minute','hourly']:
        rows=[r for r in results if r['state'].startswith(prefix+'_')]
        if rows:
            require(len(rows)==2,'Missing task folds for decision')
            gain=sum(1-r['predictors']['C']['endpoint_MSE']/r['predictors']['zero']['endpoint_MSE'] for r in rows)/2
            tasks.append({'task':prefix,'equal_fold_zero_endpoint_improvement':gain,'passed':gain>=.01})
    return {'four_states_required':True,'per_state':state_rules,'per_task':tasks,'continue_to_formal_development_review':len(results)==4 and len(tasks)==2 and all(x['passed'] for x in state_rules+tasks),'forecast_success':False,'independent_confirmation':False,'full_goal_achieved':False}


def ridge_qualification(spec):
    rng=np.random.default_rng(spec['seed']); X=rng.standard_normal((37,7)); Y=rng.standard_normal((37,3))
    records=[]
    for case in spec['cases']:
        x=X.copy()
        if case=='constant_last_column': x[:,-1]=3.25
        if case=='singular_duplicate_columns': x[:,1]=x[:,0]
        fit=fixed_ridge(x,Y)
        mean=x.mean(axis=0); sd=x.std(axis=0); sd=np.where(sd>1e-12,sd,1.); z=(x-mean)/sd
        ym=Y.mean(axis=0); beta=np.linalg.solve(z.T@z/len(z)+.1*np.eye(z.shape[1]),z.T@(Y-ym)/len(z))
        expected=z@beta+ym; actual=ridge_predict(fit,x)
        require(np.allclose(actual,expected,atol=1e-10,rtol=1e-8),'Independent direct Ridge prediction mismatch')
        require(np.allclose(fit['coefficients'],beta,atol=1e-10,rtol=1e-8),'Independent direct Ridge coefficient mismatch')
        records.append({'case':case,'max_pred_difference':float(np.max(np.abs(actual-expected))),'backward_error':fit['backward_error']})
    return records


def maximum_cost(native,adapter,scaler):
    arms,ref,initial,baseline,opts=make_arms(native,adapter.horizon,scaler)
    first=time.perf_counter(); x,a,y=adapter.batch(scaler,adapter.prefix[:128]); preparation=time.perf_counter()-first
    records=[]
    for key in ['D','Z','C']:
        steps=[]; inference=[]
        for trial in range(3):
            first=time.perf_counter(); opts[key].zero_grad(set_to_none=True)
            pred=forward(key,arms,ref,baseline,x,a)
            require(torch.isfinite(pred).all(), 'Maximumshape nonfinite training prediction')
            loss=(pred-y).square().mean()
            require(torch.isfinite(loss), 'Maximumshape nonfinite training loss')
            loss.backward()
            require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in arms[key].parameters()), 'Maximumshape missing/nonfinite gradients')
            opts[key].step()
            require(all(torch.isfinite(p).all() for p in arms[key].parameters()), 'Maximumshape nonfinite updated parameters')
            elapsed=time.perf_counter()-first
            require(elapsed<=30.,'Maximumshape cost gate')
            steps.append(elapsed)
            first=time.perf_counter()
            with torch.no_grad(): inference_output=forward(key,arms,ref,baseline,x,a)
            require(torch.isfinite(inference_output).all(), 'Maximumshape nonfinite inference output')
            inference.append(time.perf_counter()-first)
        records.append({'arm':key,'trainstep_seconds':steps,'inference_seconds':inference})
    reference_gate(arms,ref,initial,opts)
    first=time.perf_counter(); control=control_fit(adapter,scaler); ridgetime=time.perf_counter()-first
    return {'shape':{'B':128,'L':480,'H':60,'dtype':'float32'},'batch_prepare_seconds':preparation,'arms':records,'native_prefix1280_Ridge_seconds':ridgetime,'Ridge_backward_error':control['backward_error'],'forecast_success':False}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--protocol',required=True,type=Path); parser.add_argument('--mode',choices=['qualify','cost','trajectory','market','compare'],required=True); parser.add_argument('--output',required=True,type=Path); parser.add_argument('--resume',action='store_true'); parser.add_argument('--pause-kind',choices=['signal','deadline']); args=parser.parse_args()
    require(os.environ.get('AAPL_MATCHED_GUARDED')=='1','Must use guarded entry')
    protocol=json.loads(args.protocol.read_text()); protocol['_path']=str(args.protocol.resolve())
    require(time.time()<protocol['deadline_epoch'],'Absolute deadline exhausted')
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    runtime={'torch':torch.__version__,'mkldnn_enabled':torch.backends.mkldnn.enabled,'CPU':sorted(os.sched_getaffinity(0)),'GPU_hidden':os.environ.get('CUDA_VISIBLE_DEVICES')=='-1','threads':torch.get_num_threads(),'interop':torch.get_num_interop_threads()}
    signal.signal(signal.SIGUSR1,signal_pause); signal.signal(signal.SIGTERM,signal_pause)
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    if args.mode=='compare':
        from compare_payloads import compare_all
        json_write(output/'PAYLOAD_COMPARISON.json',compare_all(Path(protocol['output_directory'])))
        return 0
    native=original_pipeline(protocol)
    fixture=json.loads((Path(__file__).parent/'FIXTURE_SPEC.json').read_text())
    if args.mode=='market':
        require(protocol['phase']=='market' and protocol.get('market_admission') is True,'Preparation does not admit market')
        design=json.loads(Path(protocol['design_path']).read_text()); states=load_market(protocol,native,design,output); epochs=8
    else:
        require(protocol['phase']=='synthetic','Synthetic mode with wrong phase')
        states=[synthetic_state(native,fixture['native'])]; epochs=2
    if args.mode=='qualify':
        state=states[0]; scaler=state.scaler()
        denied=False
        try: state.labels(state.tail[:1])
        except RuntimeError: denied=True
        require(denied,'Tail target gate did not reject before freeze')
        result={'status':'passed_new_native_pipeline_qualification','native_initial':qualify_initial(native,state,scaler),'Ridge':ridge_qualification(fixture['ridge_fixtures']),'tail_target_denied_before_freeze':denied,'source_ast_receipt':native['source_ast_receipt'],'runtime':runtime,'market_read':False,'Q3_read':False}
        json_write(output/'RESULTS.json',result); return 0
    if args.mode=='cost':
        state=states[0]; result=maximum_cost(native,state,state.scaler())|{'runtime':runtime,'market_read':False,'Q3_read':False}
        json_write(output/'RESULTS.json',result); return 0
    runs=[]
    for state in states:
        destination=output if args.mode=='trajectory' else output/state.id
        if args.mode == 'trajectory' and args.resume:
            require((destination/'CHECKPOINT.json').is_file(), 'Trajectory resume requires committed checkpoint; never initialize fresh')
        use_resume=args.resume and (destination/'CHECKPOINT.json').exists()
        if args.resume and destination.exists() and list(destination.glob('payload_*.pt')) and not (destination/'CHECKPOINT.json').exists():
            raise RuntimeError('Uncommitted existing payload without manifest: no silent reinit')
        destination.mkdir(parents=True,exist_ok=True)
        qualification=qualify_initial(native,state,state.scaler())
        json_write(destination/'INITIAL_QUALIFICATION.json',qualification)
        run,code=train_state(native,state,protocol,destination,epochs,args.pause_kind,use_resume,args.mode=='trajectory')
        if code: return code
        runs.append(run)
    freeze,results=evaluate_runs(runs,protocol,output,args.mode=='trajectory')
    # Final semantic payload deliberately excludes local paths, pids, wallclock and pause kind.
    for run,row in zip(runs,results):
        final=run['payload'](); final['evaluation']=row; final['modes']={a:m.training for a,m in run['arms'].items()}|{'reference':run['reference'].training}
        manifest=save_checkpoint(run['output'],final)
        json_write(run['output']/'SUMMARY.json',{'status':'complete','steps':run['state']['steps'],'step_unit':'common_lockstep_batch','actual_optimizer_updates':3*run['state']['steps'],'epochs':run['state']['epoch'],'manifest':manifest,'synthetic':args.mode=='trajectory','market_read':args.mode=='market','Q3_read':False,'runtime':runtime})
    json_write(output/'RESULTS.json',{'status':'completed_all_fixed_states','results':results,'decision':success_gate(results) if args.mode=='market' else {'synthetic_only':True,'forecast_success':False},'runtime':runtime,'market_read':args.mode=='market','whole_preQ3_containers_decoded':args.mode=='market','Q3_read':False,'full_goal_achieved':False})
    return 0

if __name__=='__main__':
    sys.exit(main())
