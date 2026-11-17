"""Budgeted tied/untied native-decoder experiment; original H60 partitions."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import sys
import time
import resource
import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
V3 = BASE.parents[1] / 'mechanism_repair/20261109_v3'
sys.path.insert(0, str(V3 / 'code'))
from study import read, sha, atomic, npz, now, make_job, source_hashes, PRIOR
from engine import Data
from multiscale.training import atomic_torch_save
sys.path.insert(0, str(BASE / 'code'))
from shared_network import Bundle, HORIZONS
from gpu_graph import GraphBackward

DEADLINE = 1794216709.0
STOP = False
DEVICE = torch.device("cuda:0")


def interrupted(*_):
    global STOP
    STOP = True


def setup():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') != '-1' and len(os.sched_getaffinity(0)) <= 2
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    assert torch.cuda.is_available(), 'Local authorized CUDA device is required'
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)


def cpu_snapshot(value):
    """Detached immutable CPU snapshots keep all checkpoints CPU-loadable."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_snapshot(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(v) for v in value)
    return value


def device_batch(data, part, positions):
    args, truth = data.batch(part, positions)
    return tuple(value.to(DEVICE) for value in args), truth.to(DEVICE)


def finite_gradients(net):
    gradients = [p.grad for p in net.parameters()]
    if any(g is None for g in gradients):
        return False
    # Infinity norms have no sum-of-squares overflow. NaN/Inf in any gradient
    # makes its norm nonfinite, preserving the original all-finite contract
    # while synchronizing CUDA only once rather than once per parameter.
    maxima = torch._foreach_norm(gradients, float('inf'))
    return bool(torch.stack(maxima).isfinite().all())


def make_optimizer(net):
    return torch.optim.Adam(net.parameters(), lr=.001, fused=True)


def training_backward(net, optimizer, args, truth, runner=None):
    """Same full-path backward; optimizer update and validation stay outside."""
    net.train()
    if len(truth) == 128:
        if runner is None:
            runner = GraphBackward(net, optimizer, loss, args, truth)
        value = runner(args, truth)
    else:
        optimizer.zero_grad(set_to_none=True)
        value = loss(net(*args), truth)
        value.backward()
    assert torch.isfinite(value)
    assert finite_gradients(net)
    return value, runner


def training_sources():
    return {name: sha(BASE / 'code' / name) for name in ('experiment.py', 'shared_network.py', 'controller.py', 'gpu_graph.py')}


def data_for(fold):
    return Data(make_job('minute', 60, 'future_log_return', 'xpatch', 'anchor_baseline', fold))


def loss(predictions, truth):
    return sum((predictions[h] - truth[:, :h]).square().mean() for h in HORIZONS)


def benchmark(arm_only=None, worker_index=None):
    assert time.time() < DEADLINE
    started = time.perf_counter(); data = data_for(5)
    data_ready = time.perf_counter()-started
    rows = []
    for arm in ((arm_only,) if arm_only else ('tied', 'untied')):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        prepare = time.perf_counter()
        net = Bundle(480, data.scaler, 2026, arm == 'tied').to(DEVICE)
        opt = make_optimizer(net)
        torch.cuda.synchronize(); prepare = time.perf_counter()-prepare
        timings = []; cpu_start = time.process_time(); runner = None
        permutation = np.random.default_rng(2026).permutation(len(data.splits['fit']))
        capture_seconds = None
        for step in range(64):
            if STOP or time.time() >= DEADLINE:
                raise InterruptedError('benchmark deadline or signal')
            torch.cuda.synchronize(); start = time.perf_counter()
            pos = permutation[step*128:(step+1)*128]
            args, y = device_batch(data, 'fit', pos)
            value, runner = training_backward(net, opt, args, y, runner); opt.step()
            torch.cuda.synchronize()
            if step == 0:
                capture_seconds = time.perf_counter()-start
            if step >= 8:
                timings.append(time.perf_counter()-start)
        check_times = []
        check_path = BASE/'audit'/('gpu_probe_checkpoint_'+arm+('_'+str(worker_index) if worker_index is not None else '')+'.pt')
        check_path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(3):
            torch.cuda.synchronize(); before = time.perf_counter()
            atomic_torch_save(cpu_snapshot({'model':net.state_dict(), 'optimizer':opt.state_dict(),
                'numpy_rng':np.random.default_rng(2026).bit_generator.state,
                'torch_rng':torch.get_rng_state(), 'cuda_rng':torch.cuda.get_rng_state_all()}), check_path)
            check_times.append(time.perf_counter()-before)
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        rows.append({'arm': arm, 'batch_size': 128, 'lookback': 480,
            'four_heads': list(HORIZONS), 'fit_only_forward_backward_steps':64,
            'measured_steps_after_warmup':len(timings), 'step_seconds_mean':float(np.mean(timings)),
            'step_seconds_max':max(timings),'step_seconds_samples':timings,
            'parameters':sum(p.numel() for p in net.parameters()),
            'first_step_including_graph_capture_seconds':capture_seconds,
            'model_optimizer_prepare_seconds':prepare,'data_prepare_seconds':data_ready,
            'checkpoint_cpu_snapshot_atomic_save_seconds_mean':float(np.mean(check_times)),
            'checkpoint_cpu_snapshot_atomic_save_seconds_max':max(check_times),
            'checkpoint_bytes':check_path.stat().st_size,'cpu_seconds_for_probe':time.process_time()-cpu_start,
            'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'cuda_peak_allocated_bytes':peak_allocated,'cuda_peak_reserved_bytes':peak_reserved,
            'cuda_synchronized_each_step':True,'weights_are_cost_probe_only':True})
        check_path.unlink()
        del net,opt,args,y,value,runner
    counts = {}
    meta = read(PRIOR/'cache/minute_h60_future_log_return/TASK.json')
    for fold in range(1,6):
        rec = meta['folds'][str(fold)]
        with np.load(rec['split_file'],allow_pickle=False) as parts:
            counts[str(fold)] = {part:len(parts[part]) for part in ('fit','stop','validation')}
    value = {'status':'measured_before_new_stop_or_outer_scores','rows':rows,'fold_counts':counts,
        'affinity':sorted(os.sched_getaffinity(0)),'stop_or_outer_targets_constructed':False,
        'Q3_values_read':False,'GPU':True,'device':torch.cuda.get_device_name(),
        'torch_version':torch.__version__,'cuda_version':torch.version.cuda,
        'deadline_epoch':DEADLINE,'at_utc':now(),'network_sha256':sha(BASE/'code/shared_network.py'),
        'experiment_sha256':sha(BASE/'code/experiment.py'),'deterministic_algorithms':True,
        'tf32':False,'mixed_precision':False,'probe_weights_retained_as_candidate':False,
        'optimizer_execution':'Adam fused=True, unchanged lr/betas/eps/weight_decay',
        'training_execution':'CUDA graph forward/backward for batch128; eager tail; finite checks and Adam step outside graph',
        'fit_positions':'fixed seed2026 permutation, no stop/outer inputs',
        'graph_source_sha256':sha(BASE/'code/gpu_graph.py')}
    path = BASE/'audit/GPU_FIT_ONLY_COST.json' if worker_index is None else BASE/'audit/gpu_concurrent_cost'/f'worker_{worker_index}.json'
    atomic(path,value)
    print(json.dumps(value),flush=True)


def freeze():
    """Validate the explicit authorized GPU revision; never modify its budget."""
    preceding = source_hashes()
    assert preceding == read(V3/'protocol/CODE_FREEZE.json')['hashes']
    auth_path = BASE/'protocol/AUTHORIZATION.json'; auth = read(auth_path)
    assert auth['authorized'] and auth['local_gpu_authorized'] and not auth['paid_resources']
    assert auth['deadline_epoch'] == DEADLINE and auth['fixed_epochs'] == 20
    saved = read(BASE/'protocol/RULES.json')
    assert saved['sources'] == training_sources()
    assert saved['authorization_sha256'] == sha(auth_path)
    assert saved['plan_sha256'] == sha(BASE/'PLAN_ZH.md')
    assert saved['preceding_sources'] == preceding
    assert saved['cost_sha256'] == sha(BASE/saved['cost_file'])
    assert saved['data_identity'] == sha(PRIOR/'protocol/DATA.json')
    assert saved['deadline_epoch'] == DEADLINE and saved['fixed_epochs'] == 20
    assert saved['GPU'] is True and saved['Q3_values_read'] is False
    assert saved['independent_confirmation'] is False and saved['all_cells_required'] is True
    assert saved['horizons'] == list(HORIZONS) and saved['seeds'] == [2026,2027,2028]
    assert saved['folds'] == [1,2,3,4,5] and len(saved['jobs']) == 30
    expected = {(arm,fold,seed) for arm in ('tied','untied') for fold in range(1,6) for seed in (2026,2027,2028)}
    assert {(j['arm'],j['fold'],j['seed']) for j in saved['jobs']} == expected
    assert len({j['id'] for j in saved['jobs']}) == 30
    meta = read(PRIOR/'cache/minute_h60_future_log_return/TASK.json')
    for fold in range(1,6):
        rec = meta['folds'][str(fold)]; registered = saved['partitions'][str(fold)]
        assert registered['split_npz_sha256'] == sha(rec['split_file'])
        assert registered['array_hashes'] == rec['split_sha256']
        assert registered['scaler_sha256'] == sha(rec['scaler_file']) == rec['scaler_sha256']
        assert registered['registered_outer_targets_sha256'] == rec['targets_sha256']
        with np.load(rec['split_file'],allow_pickle=False) as parts:
            assert {key:hashlib.sha256(np.asarray(parts[key],dtype='<i8').tobytes()).hexdigest() for key in parts.files} == rec['split_sha256']
    return saved


@torch.inference_mode()
def predict(net, data, part, collect=False):
    net.eval(); idx = data.splits[part]; sums = {h: 0. for h in HORIZONS}
    values = {h: [] for h in HORIZONS}
    for start in range(0, len(idx), 128):
        if STOP or time.time() >= DEADLINE:
            raise InterruptedError('deadline or signal during evaluation')
        positions = np.arange(start, min(start+128, len(idx)))
        if part == 'stop':
            args, y = device_batch(data, part, positions)
        else:
            args = tuple(a.to(DEVICE) for a in input_batch(data, part, positions))
        p = net(*args)
        for h in HORIZONS:
            if part == 'stop':
                sums[h] += float((p[h][:, -1] - y[:, h-1]).double().square().sum())
            if collect:
                values[h].append(p[h].cpu().numpy()*data.ys[:h]+data.ym[:h])
    return sum(sums.values())/(len(idx)*len(HORIZONS)), \
        {h: np.concatenate(p) for h, p in values.items()} if collect else {}


def input_batch(data, part, positions):
    origins = data.splits[part][positions]
    x, anchor = data.windows.arrays(origins)
    x = ((x-data.scaler['x_mean'])/data.scaler['x_std']).astype('float32')
    anchor = ((anchor-data.scaler['anchor_mean'])/data.scaler['anchor_std']).astype('float32')
    return torch.from_numpy(x[:, :, None]), torch.from_numpy(anchor)


def run(job, steps_limit=None):
    rules = freeze(); assert rules['admitted']
    folder = BASE / 'runs' / job['id']; folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'DONE.json').exists():
        return read(folder / 'DONE.json')
    data = data_for(job['fold']); torch.manual_seed(job['seed'])
    net = Bundle(480, data.scaler, job['seed'], job['arm'] == 'tied').to(DEVICE)
    opt = make_optimizer(net); rng = np.random.default_rng(job['seed'])
    identity = hashlib.sha256(json.dumps({'job': job, 'rule': sha(BASE / 'protocol/RULES.json')}, sort_keys=True).encode()).hexdigest()
    state = {'epoch': 0, 'cursor': 0, 'permutation': None, 'steps': 0,
        'best_score': float('inf'), 'best_epoch': None, 'history': [], 'loss_sum': 0., 'seen': 0,
        'wall_seconds': 0., 'cpu_seconds': 0.}
    latest = folder / 'latest.pt'; before = time.time(); cpu = time.process_time(); resumed_steps = 0
    if latest.exists():
        saved = torch.load(latest, map_location='cpu', weights_only=True)
        assert saved['identity'] == identity and saved['scaler'] == data.scaler
        net.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer']); state = saved['state']
        rng.bit_generator.state = saved['numpy_rng']; torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        resumed_steps = state['steps']

    def record():
        return {'identity': identity, 'job': job, 'model': cpu_snapshot(net.state_dict()), 'scaler': data.scaler,
            'epoch': state['epoch'], 'parameters': sum(p.numel() for p in net.parameters()),
            'rules_sha256': sha(BASE / 'protocol/RULES.json')}

    def checkpoint():
        saved = record(); saved.update(optimizer=cpu_snapshot(opt.state_dict()), state=state,
            numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(), device=str(DEVICE))
        atomic_torch_save(saved, latest)
        atomic(folder / 'STATE.json', {'job': job, 'epoch': state['epoch'], 'cursor': state['cursor'],
            'steps': state['steps'], 'resumed_from_steps': resumed_steps,
            'deadline_epoch': DEADLINE, 'affinity': sorted(os.sched_getaffinity(0)), 'at_utc': now()})

    if not latest.exists():
        atomic_torch_save(record(), folder / 'initial.pt'); checkpoint()
    first_steps = state['steps']; runner = None
    try:
        while state['epoch'] < rules['fixed_epochs']:
            if state['permutation'] is None:
                state['permutation'] = torch.from_numpy(rng.permutation(len(data.splits['fit'])))
                state['cursor'] = 0
            while state['cursor'] < len(data.splits['fit']):
                if STOP or time.time() >= DEADLINE or steps_limit is not None and state['steps']-first_steps >= steps_limit:
                    raise InterruptedError('save at batch boundary')
                end = min(state['cursor']+128, len(data.splits['fit']))
                if len(data.splits['fit'])-end == 1:
                    end = len(data.splits['fit'])
                pos = state['permutation'][state['cursor']:end].numpy(); args, y = device_batch(data, 'fit', pos)
                value, runner = training_backward(net, opt, args, y, runner)
                opt.step(); state['cursor'] = end; state['steps'] += 1
                state['loss_sum'] += float(value.detach())*len(pos); state['seen'] += len(pos)
                # Release eager-tail autograd references before a later fresh
                # graph capture (notably after a mid-epoch process resume).
                del value
                if state['steps'] % 100 == 0:
                    checkpoint()
            score, _ = predict(net, data, 'stop'); epoch = state['epoch']+1
            state['history'].append({'epoch': epoch, 'aggregate_standardized_inner_endpoint_mse': score,
                'sum_four_path_mse': state['loss_sum']/state['seen'], 'steps': state['steps']})
            if score < state['best_score']:
                state['best_score'] = score; state['best_epoch'] = epoch
                best_record = record(); best_record['epoch'] = epoch
                atomic_torch_save(best_record, folder / 'best.pt')
            state['epoch'] = epoch; state['cursor'] = 0; state['permutation'] = None
            state['loss_sum'] = 0.; state['seen'] = 0; checkpoint()
        # Save the last fixed update before changing the in-memory network.
        atomic_torch_save(record(), folder / 'last.pt')
        outputs = {}; alphas = {}; example = None
        for mode, path in [('last', folder / 'last.pt'), ('selected', folder / 'best.pt')]:
            saved = torch.load(path, map_location='cpu', weights_only=True); net.load_state_dict(saved['model'])
            _, stop_pred = predict(net, data, 'stop', collect=True)
            from multiscale.bars import targets
            stop_y = targets(data.bars, data.splits['stop'], 60, 'future_log_return')
            _, outer = predict(net, data, 'validation', collect=True)
            alphas[mode] = {}
            for h in HORIZONS:
                p = stop_pred[h][:, -1]; y = stop_y[:, h-1]
                alpha = float(np.clip(p@y/(p@p), 0, 1)) if p@p else 0.
                alphas[mode][str(h)] = alpha
                outputs[f'{mode}_h{h}'] = outer[h]
                outputs[f'{mode}_shrink_h{h}'] = outer[h]*alpha
        vi = data.splits['validation']; npz(folder / 'predictions.npz', origins=vi, **outputs)
        args = input_batch(data, 'validation', np.arange(min(128, len(vi))))
        npz(folder / 'EXAMPLE.npz', x=args[0].numpy(), anchor=args[1].numpy(),
            **{key: value[:len(args[0])] for key, value in outputs.items()})
        state['wall_seconds'] += time.time()-before; state['cpu_seconds'] += time.process_time()-cpu
        receipt = {'status': 'completed', 'job': job, 'identity': identity,
            'fixed_epochs': rules['fixed_epochs'], 'steps': state['steps'], 'best_epoch': state['best_epoch'],
            'history': state['history'], 'alphas': alphas, 'scaler': data.scaler,
            'parameters': sum(p.numel() for p in net.parameters()), 'resumed_from_steps': resumed_steps,
            'prediction_sha256': sha(folder / 'predictions.npz'),
            'checkpoint_hashes': {p.name: sha(p) for p in (folder / 'initial.pt', folder / 'last.pt', folder / 'best.pt')},
            'rules_sha256': sha(BASE / 'protocol/RULES.json'), 'wall_seconds': state['wall_seconds'],
            'cpu_seconds': state['cpu_seconds'], 'deadline_epoch': DEADLINE,
            'affinity': sorted(os.sched_getaffinity(0)), 'GPU': True, 'device':torch.cuda.get_device_name(),
            'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(), 'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'Q3_values_read': False,
            'independent_confirmation': False, 'at_utc': now()}
        atomic(folder / 'DONE.json', receipt)
        print(json.dumps({'job': job['id'], 'status': 'completed'}), flush=True)
        return receipt
    except InterruptedError as exc:
        # Restore latest train-state model/optimizer consistency if interruption
        # occurs during post-training scoring after best weights were loaded.
        if state['epoch'] < rules['fixed_epochs']:
            state['wall_seconds'] += time.time()-before; state['cpu_seconds'] += time.process_time()-cpu
            checkpoint()
        atomic(folder / 'PAUSE.json', {'status': 'paused', 'reason': str(exc), 'steps': state['steps'],
            'epoch': state['epoch'], 'deadline_epoch': DEADLINE, 'at_utc': now()})
        return {'status': 'paused'}


def main():
    setup()
    parser = argparse.ArgumentParser(); parser.add_argument('--stage', choices=['benchmark','freeze','worker'])
    parser.add_argument('--job'); parser.add_argument('--steps-limit', type=int)
    parser.add_argument('--bench-arm', choices=['tied','untied']); parser.add_argument('--bench-index', type=int)
    args = parser.parse_args()
    if args.stage == 'benchmark':
        benchmark(args.bench_arm, args.bench_index)
    elif args.stage == 'freeze':
        freeze()
    else:
        rules = freeze(); job = next(j for j in rules['jobs'] if j['id'] == args.job)
        run(job, args.steps_limit)


if __name__ == '__main__':
    main()
