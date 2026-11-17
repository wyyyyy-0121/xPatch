"""Four fixed temporal development probes, with resumable prefix-only training.

No old experiment engine is imported. Original stop/outer members are never
loaded. The later segment is evaluated only after the fixed prefix training
and all fixed readouts have numerically qualified.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import time

BASE = Path(__file__).resolve().parents[1]
STOP = False


def stop(*_):
    global STOP
    STOP = True


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)


def atomic_torch(torch, path, value):
    with tempfile.NamedTemporaryFile('wb', dir=path.parent, delete=False) as stream:
        torch.save(value, stream)
        stream.flush(); os.fsync(stream.fileno()); temporary = stream.name
    os.replace(temporary, path)


def check_files(files):
    for path, expected in files.items():
        if sha(path) != expected:
            raise ValueError('Frozen bytes changed: '+path)


def array_sha(np, array):
    return hashlib.sha256(np.asarray(array, dtype='<i8').tobytes()).hexdigest()


def next_batch_end(cursor, count, nominal=128):
    end = min(cursor+nominal, count)
    return count if count-end == 1 else end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['market', 'synthetic'], required=True)
    parser.add_argument('--state-id')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--synthetic-batches', type=int, default=8)
    parser.add_argument('--synthetic-epochs', type=int, default=2)
    parser.add_argument('--pause-after-steps', type=int)
    parser.add_argument('--segment-seconds', type=float)
    args = parser.parse_args()
    assert set(os.sched_getaffinity(0)) == {24}
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
        assert os.environ.get(name) == '1'
    for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGUSR1]:
        signal.signal(sig, stop)
    scope = load(BASE/'STAGE_SCOPE.json')
    deadline = scope['deadline_epoch']
    assert time.time() < deadline
    parent = Path(scope['parent_directory'])
    for name, expected in scope['parent_files_sha256'].items():
        assert sha(parent/name) == expected
    freeze = load(BASE/'CODE_FREEZE.json')
    check_files(freeze['files'])
    stage_inputs=load(BASE/'STAGE_INPUT_IDENTITIES.json')
    assert stage_inputs['stage_scope_sha256']==sha(BASE/'STAGE_SCOPE.json')
    check_files(stage_inputs['verified_files'])
    inputs = load(parent/'INPUT_IDENTITIES.json')
    if args.mode == 'market':
        admission = load(BASE/'MARKET_ADMISSION.json')
        assert admission['status'] == 'admitted_four_fixed_temporal_development_states'
        assert admission['scope_sha256'] == sha(BASE/'STAGE_SCOPE.json')
        assert admission['code_freeze_sha256'] == sha(BASE/'CODE_FREEZE.json')
        assert admission['input_identities_sha256'] == sha(parent/'INPUT_IDENTITIES.json')
        assert admission['stage_input_identities_sha256'] == sha(BASE/'STAGE_INPUT_IDENTITIES.json')
        assert args.pause_after_steps is None and args.segment_seconds is None
        check_files(inputs['verified_files'])
        check_files(stage_inputs['inherited_data_container_identities_requiring_recheck_before_open'])
        source = next(s for s in inputs['states'] if s['id'] == args.state_id)
        cfg = source['cfg']
        fixed_epochs = load(parent/'NEXT_TEMPORAL_READOUT_PROBE_PLAN.json')['fixed_epoch_budgets'][source['id']]
    else:
        assert not args.state_id
        assert 2 <= args.synthetic_batches <= 100 and 1 <= args.synthetic_epochs <= 20
        cfg = {'model':'lstm', 'initialization':'baseline', 'anchor':'observed',
               'output':'return', 'target':'future_log_return', 'loss':'path',
               'seed':2026, 'learning_rate':0.001, 'batch_size':128,
               'lookback':480, 'horizon':60, 'frequency':'minute', 'fold':1}
        source = None
        fixed_epochs = args.synthetic_epochs

    import numpy as np
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0, inputs['reference_root'])
    from multiscale.bars import Bars, targets
    from multiscale.features import CloseWindows, fit_scaler
    from multiscale.models import make_model
    for name in ['multiscale.bars','multiscale.features','multiscale.models']:
        Path(sys.modules[name].__file__).resolve().relative_to(Path(inputs['reference_root']).resolve())
    h, lookback = cfg['horizon'], cfg['lookback']
    assert cfg['model'] == 'lstm' and cfg['initialization'] == 'baseline'
    assert cfg['target'] == 'future_log_return' and cfg['output'] == 'return'
    assert cfg['anchor'] == 'observed' and cfg['loss'] == 'path'
    if args.mode == 'market':
        with np.load(source['split_file'], allow_pickle=False) as archive:
            fit = archive['fit']
        assert len(fit) == source['fit_count'] and array_sha(np,fit) == source['fit_array_sha256']
        with np.load(source['source_bars'], allow_pickle=False) as archive:
            assert set(archive.files) == {'start_ns','end_ns','close','observed','session_date'}
            arrays = {name:archive[name] for name in archive.files}
        assert np.all(arrays['session_date'] <= '2026-06-30')
        cutoff = np.datetime64('2026-07-01T04:00:00','ns').astype('int64')
        assert np.all(arrays['end_ns'][arrays['observed']] < cutoff)
        bars = Bars(cfg['frequency'], **arrays)
        actions = load(inputs['source_admission_path'])['split_actions']
    else:
        nfit = 128*args.synthetic_batches
        count = lookback+nfit+h
        generator = np.random.default_rng(8112001)
        close = 100*np.exp(np.cumsum(generator.normal(0, 0.0005, count)))
        start = np.datetime64('2026-01-01T00:00:00','ns').astype('int64')
        end_ns = start+np.arange(1,count+1,dtype=np.int64)*60_000_000_000
        bars = Bars('minute', end_ns-60_000_000_000, end_ns, close,
                    np.ones(count,dtype=bool), np.full(count,'2026-01-01'))
        fit = np.arange(lookback-1, lookback-1+nfit, dtype=np.int64)
        actions = []
    assert fit.ndim == 1 and len(fit)>h+128 and np.all(np.diff(fit)>0)
    assert fit[0]>=lookback-1 and fit[-1]+h<len(bars.close)
    cut = (4*len(fit))//5
    tail = fit[cut:]
    update0 = fit[:cut]
    update = update0[bars.end_ns[update0+h] < bars.end_ns[tail[0]]]
    assert len(update)>=128 and len(tail)>=32
    assert bars.end_ns[update[-1]+h] < bars.end_ns[tail[0]]
    assert not np.intersect1d(update,tail).size
    windows = CloseWindows(bars, lookback, actions)
    # All fitted statistics and all training labels are restricted to update.
    scaler = asdict(fit_scaler(windows,update,h,cfg['target']))
    scaler = json.loads(json.dumps(scaler))
    assert scaler['fitted_origin_count'] == len(update)
    ym, ys = np.asarray(scaler['y_mean']), np.asarray(scaler['y_std'])
    assert np.isfinite(ym).all() and np.all(ys>0)
    directory = args.run_dir.resolve(); directory.relative_to(BASE)
    directory.mkdir(parents=True, exist_ok=True)
    binding = {'mode':args.mode, 'state_id':args.state_id or 'synthetic', 'cfg':cfg,
        'fixed_epochs':fixed_epochs, 'scope_sha256':sha(BASE/'STAGE_SCOPE.json'),
        'stage_input_identities_sha256':sha(BASE/'STAGE_INPUT_IDENTITIES.json'),
        'code_freeze_sha256':sha(BASE/'CODE_FREEZE.json'), 'stage_deadline_epoch':deadline,
        'original_fit_sha256':array_sha(np,fit), 'update_sha256':array_sha(np,update),
        'tail_sha256':array_sha(np,tail), 'update_count':len(update), 'tail_count':len(tail),
        'purged_count':len(update0)-len(update), 'scaler':scaler,
        'all_fitted_statistics_restricted_to_update':True, 'old_model_or_scaler_reused':False}
    identity = digest(binding)

    def batch(indices, need_raw=False):
        x, anchor = windows.arrays(indices)
        x=((x-scaler['x_mean'])/scaler['x_std']).astype(np.float32)
        anchor=((anchor-scaler['anchor_mean'])/scaler['anchor_std']).astype(np.float32)
        raw = targets(bars,indices,h,cfg['target'])
        standardized=((raw-ym)/ys).astype(np.float32)
        return (torch.from_numpy(x[:,:,None]),torch.from_numpy(anchor)), torch.from_numpy(standardized), raw

    torch.manual_seed(cfg['seed'])
    net = make_model('lstm',lookback,h)
    # Exact return anchor_baseline initialization copied from v3 networks.create.
    with torch.no_grad():
        net.head.weight.zero_()
        net.head.bias.copy_(torch.as_tensor(-ym/ys,dtype=torch.float32))
    initial = copy.deepcopy(net.state_dict())
    opt = torch.optim.Adam(net.parameters(),lr=cfg['learning_rate'])
    rng = np.random.default_rng(cfg['seed'])
    state={'epoch':0,'cursor':0,'permutation':None,'steps':0,'phase':'training',
           'loss_sum':0.0,'seen':0,'epoch_losses':[], 'training_seconds':0.0}
    checkpoint = directory/'CHECKPOINT.json'
    generation = 0
    if checkpoint.exists():
        manifest=load(checkpoint)
        payload=directory/manifest['payload_file']
        if (payload.parent!=directory or manifest['identity']!=identity
                or sha(payload)!=manifest['payload_sha256']):
            raise ValueError('Training checkpoint payload checksum or identity changed')
        generation=manifest['generation']
        saved = torch.load(payload,map_location='cpu',weights_only=True)
        if saved['identity']!=identity or saved['binding']!=binding:
            raise ValueError('Training checkpoint identity changed')
        net.load_state_dict(saved['model'],strict=True); opt.load_state_dict(saved['optimizer'])
        initial=saved['initial_model']; state=saved['state']
        rng.bit_generator.state=saved['numpy_rng'];torch.set_rng_state(saved['torch_rng'])
    invocation_start=time.time(); first_steps=state['steps']

    def status(status_name, **extra):
        atomic_json(directory/'STATUS.json',{'status':status_name,'identity':identity,
            'epoch':state['epoch'],'cursor':state['cursor'],'steps':state['steps'],
            'phase':state['phase'],'update_count':len(update),'tail_count':len(tail),
            'fixed_epochs':fixed_epochs,'stage_deadline_epoch':deadline,
            'GPU':False,'stop_outer_scored':False,'Q3_values_read':False,**extra})

    def save(status_name):
        nonlocal generation
        generation+=1
        payload=directory/f'CHECKPOINT.{generation:06d}.pt'
        while payload.exists():
            generation+=1
            payload=directory/f'CHECKPOINT.{generation:06d}.pt'
        atomic_torch(torch,payload,{'identity':identity,'binding':binding,
            'model':net.state_dict(),'optimizer':opt.state_dict(),'initial_model':initial,
            'state':state,'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state()})
        atomic_json(checkpoint,{'identity':identity,'generation':generation,
            'payload_file':payload.name,'payload_sha256':sha(payload),
            'stage_deadline_epoch':deadline,'epoch':state['epoch'],'steps':state['steps']})
        status(status_name)

    def boundary():
        if STOP or time.time()>=deadline or (args.segment_seconds is not None and
                time.time()-invocation_start>=args.segment_seconds) or (
                args.pause_after_steps is not None and state['steps']-first_steps>=args.pause_after_steps):
            save('paused_atomic_training_state_preserved')
            raise InterruptedError('Signal, original deadline or synthetic boundary reached')

    if not checkpoint.exists():save('running')
    atomic_json(directory/'PARTITION.json',{'identity':identity,'original_fit_count':len(fit),
        'update_before_purge':len(update0),'update_count':len(update),'tail_count':len(tail),
        'update_sha256':array_sha(np,update),'tail_sha256':array_sha(np,tail),
        'last_update_label_end_ns':int(bars.end_ns[update[-1]+h]),
        'first_tail_decision_ns':int(bars.end_ns[tail[0]]),
        'old_stop_outer_split_members_loaded':False,'full_preQ3_bar_cache_loaded':args.mode=='market',
        'historical_development_informed_design':True,'independent_confirmation':False})
    try:
        while state['epoch']<fixed_epochs:
            if state['permutation'] is None:
                state['permutation']=torch.from_numpy(rng.permutation(len(update)))
                state['cursor']=0
            while state['cursor']<len(update):
                boundary(); end=next_batch_end(state['cursor'],len(update))
                positions=state['permutation'][state['cursor']:end].numpy()
                args0,y,_=batch(update[positions])
                started=time.perf_counter()
                net.train();opt.zero_grad(set_to_none=True)
                loss=(net(*args0)-y).square().mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite original path objective')
                loss.backward()
                if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()):
                    raise ValueError('Missing or nonfinite original Adam gradient')
                opt.step();state['cursor']=end;state['steps']+=1
                state['loss_sum']+=float(loss.detach())*len(positions);state['seen']+=len(positions)
                state['training_seconds']+=time.perf_counter()-started
                if state['steps']%25==0:save('running')
            state['epoch_losses'].append(state['loss_sum']/state['seen'])
            state['epoch']+=1;state['cursor']=0;state['permutation']=None
            state['loss_sum']=0.0;state['seen']=0;save('running')
        state['phase']='fixed_readouts';save('running')
        # Reconstructible from the saved fixed trained and initialized encoders.
        # A partial dense assembly is transient; completed training never reruns.
        with torch.random.fork_rng(devices=[]):
            random_net=make_model('lstm',lookback,h)
        random_net.load_state_dict(initial,strict=True)
        for model in [net,random_net]:
            model.eval()
            for parameter in model.parameters():parameter.requires_grad_(False)
        b0=np.vstack([net.head.weight.detach().numpy().T,net.head.bias.detach().numpy()]).astype(np.float64)

        def assemble(indices, label_partition):
            x0=np.empty((len(indices),34),dtype=np.float64)
            xd=np.empty_like(x0);yy=np.empty((len(indices),h),dtype=np.float64)
            native=np.empty_like(yy);raws=np.empty_like(yy)
            for start in range(0,len(indices),128):
                boundary();end=min(start+128,len(indices))
                (xt,at),yt,raw=batch(indices[start:end])
                with torch.inference_mode():
                    hidden=net.rnn(xt)[0][:,-1]
                    rhidden=random_net.rnn(xt)[0][:,-1]
                    feature=torch.cat([hidden,at],dim=1)
                    native[start:end]=net(xt,at).numpy().astype(np.float64)
                    if start==0 and not np.array_equal(net.head(feature).numpy(),native[start:end].astype(np.float32)):
                        raise ValueError('Native forward differs from extracted-feature route')
                x0[start:end,:33]=feature.numpy().astype(np.float64);x0[start:end,33]=1.0
                xd[start:end,:32]=rhidden.numpy().astype(np.float64)
                xd[start:end,32]=at.numpy()[:,0];xd[start:end,33]=1.0
                yy[start:end]=yt.numpy().astype(np.float64);raws[start:end]=raw
                status('running',analysis_partition=label_partition,analysis_cursor=end,
                       transient_matrix_recomputed_on_resume=True)
            rebuilt=x0@b0
            scale=max(1.0,float(np.abs(native).max()),float((np.abs(x0)@np.abs(b0)).max()))
            error=float(np.abs(native-rebuilt).max())/scale
            if error>64*np.finfo(np.float32).eps:raise ValueError('Affine output reconstruction failed')
            return x0,xd,yy,native,raws,error

        ux,ud,uy,ua,ur,urecon=assemble(update,'update')
        from affine_lstsq import fit_affine, loss_summary, compare_losses, transform_prediction
        readouts={'B':fit_affine(ux,uy),'C':fit_affine(ux[:,32:],uy),'D':fit_affine(ud,uy)}
        summaries={name:fit.summary() for name,fit in readouts.items()}
        update_metrics={'A':float(np.mean((ua-uy)**2)),
                        'zero':float(np.mean((-ym/ys-uy)**2)),
                        'mean':float(np.mean(uy**2))}
        update_metrics.update({name:value['residual_MSE'] for name,value in summaries.items()})
        atomic_json(directory/'UPDATE_METRICS.json',{'identity':identity,
            'path_standardized_MSE':update_metrics,
            'A_minus_B_MSE':update_metrics['A']-update_metrics['B'],
            'all_fit_statistics_restricted_to_update':True,'tail_labels_scored':False})
        atomic_json(directory/'READOUT_QUALIFICATION.json',{'identity':identity,
            'readouts':summaries,'tail_labels_scored':False})
        # Qualification interface is deliberately strict: no smaller-rank rescue.
        if not all(value['solver_qualified'] for value in summaries.values()):
            state['phase']='completed_nonqualified';save('completed_nonqualified')
            atomic_json(directory/'RESULT.json',{'status':'completed_numerical_nonqualification',
                'identity':identity,'binding':binding,'readouts':summaries,'tail_scored':False,
                'Q3_values_read':False,'stop_outer_scored':False,'new_candidate_admitted':False})
            atomic_json(directory/'RESULT_COMMIT.json',{'result_sha256':sha(directory/'RESULT.json'),
                'identity':identity,'stage_deadline_epoch':deadline})
            return 0
        boundary();state['phase']='fixed_tail_scoring';save('running')
        tx,td,ty,ta,tr,trecon=assemble(tail,'original_fit_tail')
        predictions={'A':ta,'zero':np.broadcast_to(-ym/ys,ty.shape),
                     'mean':np.zeros_like(ty)}
        margins={name:np.zeros_like(ty) for name in predictions}
        query_details={}
        for name,fit in readouts.items():
            query={'B':tx,'C':tx[:,32:],'D':td}[name]
            prediction,qualification=fit.predict(query)
            predictions[name]=prediction
            margins[name]=qualification['absolute_error_margin']
            query_details[name]={key:value for key,value in qualification.items()
                                 if key!='absolute_error_margin'}
        metrics={};comparisons={};losses={};raw_predictions={};raw_margins={}
        for name,prediction in predictions.items():
            if not np.isfinite(prediction).all():raise ValueError('Nonfinite tail prediction')
            raw_prediction,raw_margin=transform_prediction(prediction,margins[name],ys,ym)
            raw_predictions[name]=raw_prediction;raw_margins[name]=raw_margin
            losses[name]={'path':loss_summary(prediction,ty,margins[name]),
                          'endpoint':loss_summary(raw_prediction[:,-1:],tr[:,-1:],raw_margin[:,-1:])}
            metrics[name]={'path_standardized_MSE':losses[name]['path']['MSE'],
                          'endpoint_raw_MSE':losses[name]['endpoint']['MSE']}
        for reference in ['A','C','D','zero','mean']:
            comparisons[reference]={
                'path':compare_losses(losses['B']['path'],losses[reference]['path']),
                'endpoint':compare_losses(losses['B']['endpoint'],losses[reference]['endpoint'])}
        # Save per-origin losses for dependence-aware summaries; no predictor.
        paired={'origins':tail}
        for name,prediction in predictions.items():
            paired[name+'_path']=np.mean((prediction-ty)**2,axis=1)
            paired[name+'_endpoint']=(raw_predictions[name][:,-1]-tr[:,-1])**2
        with tempfile.NamedTemporaryFile('wb',dir=directory,delete=False) as stream:
            np.savez_compressed(stream,**paired);stream.flush();os.fsync(stream.fileno());temporary=stream.name
        os.replace(temporary,directory/'PAIRED_TAIL_LOSSES.npz')
        result={'status':'completed_fixed_temporal_diagnostic','identity':identity,'binding':binding,
            'fixed_epochs_completed':state['epoch'],'training_steps':state['steps'],
            'training_seconds':state['training_seconds'],'epoch_training_losses':state['epoch_losses'],
            'readouts':summaries,'tail_query_details':query_details,'tail_metrics':metrics,
            'tail_loss_numerical_sensitivity':losses,
            'update_path_standardized_MSE':update_metrics,
            'B_vs_controls':comparisons,'update_affine_reconstruction_error':urecon,
            'tail_affine_reconstruction_error':trecon,
            'paired_loss_sha256':sha(directory/'PAIRED_TAIL_LOSSES.npz'),
            'tail_scored':True,'new_neural_training':True,'new_candidate_admitted':False,
            'historically_exposed_development':True,'independent_confirmation':False,
            'original_stop_outer_scored':False,'Q3_values_read':False,'GPU':False,
            'finished_at_UTC':datetime.now(timezone.utc).isoformat(),
            'research_goal_achieved':False}
        boundary();atomic_json(directory/'RESULT.json',result)
        atomic_json(directory/'RESULT_COMMIT.json',{'result_sha256':sha(directory/'RESULT.json'),
            'identity':identity,'stage_deadline_epoch':deadline})
        state['phase']='completed';save('completed')
        print(json.dumps({'status':result['status'],'state_id':binding['state_id'],
                          'training_steps':state['steps'],'result':str(directory/'RESULT.json')}))
        return 0
    except InterruptedError as error:
        print(json.dumps({'status':'paused','reason':str(error),'epoch':state['epoch'],
                          'steps':state['steps'],'checkpoint':str(checkpoint)}))
        return 2
    except Exception:
        save('error_training_state_preserved')
        raise


if __name__=='__main__':
    raise SystemExit(main())
