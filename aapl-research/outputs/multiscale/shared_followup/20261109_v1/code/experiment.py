"""Budgeted tied/untied native-decoder experiment; original H60 partitions."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import sys
import time
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

DEADLINE = 1794205909.6228158
STOP = False


def interrupted(*_):
    global STOP
    STOP = True


def setup():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1' and len(os.sched_getaffinity(0)) <= 2
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)


def training_sources():
    return {name: sha(BASE / 'code' / name) for name in ('experiment.py', 'shared_network.py', 'controller.py')}


def data_for(fold):
    return Data(make_job('minute', 60, 'future_log_return', 'xpatch', 'anchor_baseline', fold))


def loss(predictions, truth):
    return sum((predictions[h] - truth[:, :h]).square().mean() for h in HORIZONS)


def benchmark(arm_only=None, worker_index=None):
    if worker_index is None:
        assert set(os.sched_getaffinity(0)).issubset({24, 26})
    assert time.time() < DEADLINE
    data = data_for(5); rows = []
    for arm in ((arm_only,) if arm_only else ('tied', 'untied')):
        net = Bundle(480, data.scaler, 2026, arm == 'tied')
        opt = torch.optim.Adam(net.parameters(), lr=.001)
        timings = []
        for step in range(40):
            start = time.perf_counter()
            pos = np.arange(step*128, (step+1)*128)
            args, y = data.batch('fit', pos)
            net.train(); opt.zero_grad(set_to_none=True)
            value = loss(net(*args), y); value.backward(); opt.step()
            if step >= 8:
                timings.append(time.perf_counter()-start)
            assert time.time() < DEADLINE
        rows.append({'arm': arm, 'batch_size': 128, 'lookback': 480,
            'four_heads': list(HORIZONS), 'fit_only_forward_backward_steps': 40,
            'measured_steps_after_warmup': 32, 'step_seconds_mean': float(np.mean(timings)),
            'step_seconds_max': max(timings), 'parameters': sum(p.numel() for p in net.parameters())})
    if worker_index is not None:
        atomic(BASE / 'audit/concurrent_fit_cost' / f'worker_{worker_index}.json',
            {'rows': rows, 'affinity': sorted(os.sched_getaffinity(0)),
             'fit_only': True, 'at_utc': now()})
        print(json.dumps(rows), flush=True)
        return
    counts = {}
    for fold in range(1, 6):
        data = data_for(fold)
        counts[str(fold)] = {part: len(data.splits[part]) for part in ('fit', 'stop', 'validation')}
    atomic(BASE / 'audit/FIT_ONLY_COST.json', {'status': 'measured_before_stop_or_outer_scores',
        'rows': rows, 'fold_counts': counts, 'affinity': sorted(os.sched_getaffinity(0)),
        'stop_or_outer_targets_constructed': False, 'Q3_values_read': False,
        'deadline_epoch': DEADLINE, 'at_utc': now(), 'network_sha256': sha(BASE / 'code/shared_network.py')})
    print(json.dumps(rows), flush=True)


def freeze():
    assert source_hashes() == read(V3 / 'protocol/CODE_FREEZE.json')['hashes']
    assert read(V3 / 'protocol/BUDGET.json')['deadline_epoch'] == DEADLINE
    cost = read(BASE / 'audit/FIT_ONLY_COST.json')
    concurrent = read(BASE / 'audit/CONCURRENT_FIT_COST.json')
    assert cost['network_sha256'] == sha(BASE / 'code/shared_network.py')
    path = BASE / 'protocol/RULES.json'
    if path.exists():
        saved = read(path); assert saved['sources'] == training_sources()
        assert saved['cost_sha256'] == sha(BASE / 'audit/FIT_ONLY_COST.json')
        assert saved['concurrent_cost_sha256'] == sha(BASE / 'audit/CONCURRENT_FIT_COST.json')
        assert saved['preceding_sources'] == source_hashes()
        assert saved['plan_sha256'] == sha(BASE / 'PLAN_ZH.md')
        return saved
    jobs = [{'id': f'{arm}_f{fold}_s{seed}', 'arm': arm, 'fold': fold, 'seed': seed}
            for fold in range(5, 0, -1) for seed in (2026, 2027, 2028) for arm in ('untied', 'tied')]
    projections = []; schedules = {}
    for epochs in (20, 6):
        durations = []
        for job in jobs:
            count = cost['fold_counts'][str(job['fold'])]
            per_step = max(r['rows'][0]['step_seconds_mean'] for r in concurrent['workers']
                           if r['rows'][0]['arm'] == job['arm'])
            steps = epochs * ((count['fit']+127)//128)
            # Actual eight-slot fit-only throughput, maximum worker per arm,
            # plus25% margin. Full backward time also bounds inference batches.
            other = (epochs+2)*((count['stop']+127)//128) + 2*((count['validation']+127)//128)
            checkpoint_seconds = (steps//100+epochs)*.1
            durations.append((steps+other)*per_step*1.25+checkpoint_seconds+5.)
        loads = [0.]*8
        order = sorted(range(len(jobs)), key=lambda k: (-durations[k], jobs[k]['id']))
        for index in order:
            duration = durations[index]
            k = int(np.argmin(loads)); loads[k] += duration
        schedules[epochs] = [jobs[i] for i in order]
        projections.append({'fixed_epochs': epochs, 'eight_slot_projected_seconds': max(loads),
                            'longest_job_seconds': max(durations),
                            'inflation_on_measured_concurrent_maximum': 1.25,
                            'dispatch': 'LPT in the exact frozen job order',
                            'checkpoint_io_seconds_per_save': .1, 'fixed_setup_seconds_per_job': 5.})
    remaining = DEADLINE-time.time(); reserve = 600.
    chosen = next((p for p in projections if p['eight_slot_projected_seconds']+reserve < remaining), None)
    parts = {}
    meta = read(PRIOR / 'cache/minute_h60_future_log_return/TASK.json')
    for fold in range(1, 6):
        rec = meta['folds'][str(fold)]
        with np.load(rec['split_file'], allow_pickle=False) as a:
            assert {key: hashlib.sha256(np.asarray(a[key], dtype='<i8').tobytes()).hexdigest()
                    for key in a.files} == rec['split_sha256']
        assert sha(rec['scaler_file']) == rec['scaler_sha256']
        parts[str(fold)] = {'split_npz_sha256': sha(rec['split_file']),
            'array_hashes': rec['split_sha256'], 'scaler_sha256': rec['scaler_sha256'],
            'registered_outer_targets_sha256': rec['targets_sha256']}
    value = {'status': 'frozen_before_new_stop_or_outer_scores', 'admitted': chosen is not None,
        'fixed_epochs': chosen['fixed_epochs'] if chosen else None,
        'jobs': schedules[chosen['fixed_epochs']] if chosen else schedules[6],
        'cost_sha256': sha(BASE / 'audit/FIT_ONLY_COST.json'), 'cost_projections': projections,
        'concurrent_cost_sha256': sha(BASE / 'audit/CONCURRENT_FIT_COST.json'),
        'remaining_wall_seconds_at_freeze': remaining, 'reserve_seconds': reserve,
        'sources': training_sources(), 'preceding_sources': source_hashes(),
        'plan_sha256': sha(BASE / 'PLAN_ZH.md'), 'partitions': parts,
        'horizons': list(HORIZONS), 'seeds': [2026, 2027, 2028], 'folds': [1, 2, 3, 4, 5],
        'data_identity': sha(PRIOR / 'protocol/DATA.json'), 'batch_size': 128, 'learning_rate': .001,
        'loss': 'sum of four distinct standardized full-path head MSEs',
        'primary_mechanism': 'tied last fixed epoch vs untied last fixed epoch with identical update counts',
        'secondary': 'one aggregate inner-stop-best whole-bundle epoch per arm; raw and headwise shrunk',
        'candidate': 'tied_selected_shrink', 'counterfactual_patience': 3,
        'early_termination': False, 'all_cells_required': True,
        'original_question_complete_only_if_all_30_bundles_and_three_seed_five_fold_analysis_finish': True,
        'GPU': False, 'Q3_values_read': False, 'independent_confirmation': False,
        'deadline_epoch': DEADLINE, 'at_utc': now(), 'at_epoch': time.time()}
    atomic(path, value); print(json.dumps({k: value[k] for k in ('admitted','fixed_epochs','cost_projections','remaining_wall_seconds_at_freeze')}), flush=True)
    return value


@torch.inference_mode()
def predict(net, data, part, collect=False):
    net.eval(); idx = data.splits[part]; sums = {h: 0. for h in HORIZONS}
    values = {h: [] for h in HORIZONS}
    for start in range(0, len(idx), 128):
        if STOP or time.time() >= DEADLINE:
            raise InterruptedError('deadline or signal during evaluation')
        positions = np.arange(start, min(start+128, len(idx)))
        if part == 'stop':
            args, y = data.batch(part, positions)
        else:
            args = input_batch(data, part, positions)
        p = net(*args)
        for h in HORIZONS:
            if part == 'stop':
                sums[h] += float((p[h][:, -1] - y[:, h-1]).double().square().sum())
            if collect:
                values[h].append(p[h].numpy()*data.ys[:h]+data.ym[:h])
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
    net = Bundle(480, data.scaler, job['seed'], job['arm'] == 'tied')
    opt = torch.optim.Adam(net.parameters(), lr=.001); rng = np.random.default_rng(job['seed'])
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
        resumed_steps = state['steps']

    def record():
        return {'identity': identity, 'job': job, 'model': net.state_dict(), 'scaler': data.scaler,
            'epoch': state['epoch'], 'parameters': sum(p.numel() for p in net.parameters()),
            'rules_sha256': sha(BASE / 'protocol/RULES.json')}

    def checkpoint():
        saved = record(); saved.update(optimizer=opt.state_dict(), state=state,
            numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state())
        atomic_torch_save(saved, latest)
        atomic(folder / 'STATE.json', {'job': job, 'epoch': state['epoch'], 'cursor': state['cursor'],
            'steps': state['steps'], 'resumed_from_steps': resumed_steps,
            'deadline_epoch': DEADLINE, 'affinity': sorted(os.sched_getaffinity(0)), 'at_utc': now()})

    if not latest.exists():
        atomic_torch_save(record(), folder / 'initial.pt'); checkpoint()
    first_steps = state['steps']
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
                pos = state['permutation'][state['cursor']:end].numpy(); args, y = data.batch('fit', pos)
                net.train(); opt.zero_grad(set_to_none=True); value = loss(net(*args), y)
                assert torch.isfinite(value); value.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
                opt.step(); state['cursor'] = end; state['steps'] += 1
                state['loss_sum'] += float(value.detach())*len(pos); state['seen'] += len(pos)
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
            'affinity': sorted(os.sched_getaffinity(0)), 'Q3_values_read': False,
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
