"""Input-only, weights-only four-head replay for an extracted context bundle.

The bundle supplies code/shared_network.py, code/reference_source/, runs/,
and MANIFEST.json. No training, label loading, or original workspace is needed.
The optional --deny-path audit guard makes original-workspace reads fail.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import numpy as np
import torch


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    parser.add_argument('--deny-path', action='append', default=[])
    parser.add_argument('--guard-self-test', action='store_true')
    args = parser.parse_args(); root = args.root.resolve()
    assert args.device == 'cuda' or os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    device = torch.device(args.device)
    if args.device == 'cuda':
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        assert os.environ['CUBLAS_WORKSPACE_CONFIG'] in (':4096:8', ':16:8')
        assert torch.cuda.is_available()
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    assert len(os.sched_getaffinity(0)) <= 2, 'Use at most two logical CPUs'
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    denied = tuple(Path(p).resolve() for p in args.deny_path)
    guard_counts = {'network_attempts': 0, 'forbidden_path_attempts': 0}

    def audit(event, values):
        if event in ('socket.connect', 'socket.connect_ex', 'socket.getaddrinfo', 'socket.bind'):
            guard_counts['network_attempts'] += 1
            raise PermissionError('Network disabled during replay')
        if event in ('open', 'os.listdir', 'os.scandir') and values:
            value = values[0]
            if isinstance(value, (str, bytes, os.PathLike)):
                path = Path(os.fsdecode(value)).resolve()
                if any(path == base or path.is_relative_to(base) for base in denied):
                    guard_counts['forbidden_path_attempts'] += 1
                    raise PermissionError('Original workspace disabled during replay')

    sys.addaudithook(audit)
    guard_tests = {}
    if args.guard_self_test:
        assert denied, 'Guard self-test needs at least one denied path'
        try:
            (denied[0]/'DO_NOT_READ_GUARD_PROBE').read_bytes()
        except PermissionError:
            guard_tests['original_path_open_blocked'] = True
        else:
            raise AssertionError('Original-workspace read guard did not fire')
        import socket
        probe = socket.socket()
        try:
            probe.connect(('127.0.0.1', 9))
        except PermissionError:
            guard_tests['socket_connect_blocked_before_network'] = True
        else:
            raise AssertionError('Network guard did not fire')
        finally:
            probe.close()
    manifest = json.loads((root/'MANIFEST.json').read_text())
    assert {'BUNDLE_SCOPE.json', 'code/shared_network.py', 'code/reference_source/multiscale/models.py',
            'replay.py'}.issubset(manifest['files']), 'Required replay sources must be hash bound'
    for name, expected in manifest['files'].items():
        path = (root/name).resolve()
        assert path.is_relative_to(root), name
        assert digest(path) == expected, f'File hash mismatch: {name}'
    scope = json.loads((root/'BUNDLE_SCOPE.json').read_text())
    market_count = scope['formal_market_bundles']; synthetic_count = scope['synthetic_example_bundles']
    assert (market_count, synthetic_count) in ((30, 0), (0, 2)), 'Explicit complete-market or two-arm synthetic scope required'
    assert manifest['market_bundles'] == market_count
    assert scope['independent_confirmation'] is False and scope['Q3_values_read'] is False
    expected_runs = market_count+synthetic_count
    folders = sorted((root/'runs').iterdir())
    assert len(folders) == expected_runs and all(p.is_dir() for p in folders), 'Replay run count differs from declared scope'
    required_names = {'DONE.json', 'EXAMPLE.npz', 'initial.pt', 'last.pt', 'best.pt'}
    if market_count:
        assert scope['research_experiment_complete'] is True
        assert 'protocol/RULES.json' in manifest['files']
        rules = json.loads((root/'protocol/RULES.json').read_text())
        assert len(rules['jobs']) == 30 and rules['fixed_epochs'] == 20
        assert {j['id'] for j in rules['jobs']} == {p.name for p in folders}
        assert {(j['arm'], j['fold'], j['seed']) for j in rules['jobs']} == {
            (a, f, s) for a in ('tied', 'untied') for f in range(1, 6) for s in (2026, 2027, 2028)}
        jobs = {j['id']: j for j in rules['jobs']}
        rule_hash = digest(root/'protocol/RULES.json')
        required_names.add('predictions.npz')
    else:
        assert scope['research_experiment_complete'] is False
    for folder in folders:
        assert {f'runs/{folder.name}/{name}' for name in required_names}.issubset(manifest['files'])
    reference = root/'code/reference_source'
    sys.path.insert(0, str(reference)); sys.path.insert(0, str(root/'code'))
    from shared_network import Bundle, HORIZONS
    import shared_network
    import multiscale.models
    assert Path(shared_network.__file__).resolve().is_relative_to(root)
    assert Path(multiscale.models.__file__).resolve().is_relative_to(root)
    receipts = []; comparisons = 0
    for folder in folders:
        done = json.loads((folder/'DONE.json').read_text())
        assert done['job']['id'] == folder.name and done['job']['arm'] in ('tied', 'untied')
        if market_count:
            assert done['status'] == 'completed' and done['job'] == jobs[folder.name]
            assert done['rules_sha256'] == rule_hash and done['fixed_epochs'] == 20
            assert done['identity'] == hashlib.sha256(json.dumps({'job': done['job'], 'rule': rule_hash}, sort_keys=True).encode()).hexdigest()
            assert done['Q3_values_read'] is False and done['independent_confirmation'] is False
            assert digest(folder/'predictions.npz') == done['prediction_sha256']
        else:
            assert done['synthetic_only'] is True
        with np.load(folder/'EXAMPLE.npz', allow_pickle=False) as example:
            expected_keys = {'x', 'anchor'} | {f'{mode}{shrink}_h{h}' for mode in ('last', 'selected')
                for shrink in ('', '_shrink') for h in HORIZONS}
            assert set(example.files) == expected_keys, 'Input-only example must have exactly the declared replay arrays'
            x = torch.from_numpy(example['x']).to(device); anchor = torch.from_numpy(example['anchor']).to(device)
            assert x.dtype == torch.float32 and anchor.dtype == torch.float32
            assert x.ndim == 3 and x.shape[-1] == 1 and 0 < len(x) <= 128
            assert anchor.shape == (len(x), 1)
            assert torch.isfinite(x).all() and torch.isfinite(anchor).all()
            models = []
            for mode, name in (('last', 'last.pt'), ('selected', 'best.pt')):
                checkpoint = torch.load(folder/name, map_location='cpu', weights_only=True)
                job = checkpoint['job']; scaler = checkpoint['scaler']
                assert job == done['job']
                if market_count:
                    assert checkpoint['identity'] == done['identity'] and checkpoint['rules_sha256'] == rule_hash
                    assert scaler == done['scaler']
                    assert checkpoint['epoch'] == (done['fixed_epochs'] if mode == 'last' else done['best_epoch'])
                    assert digest(folder/name) == done['checkpoint_hashes'][name]
                model = Bundle(x.shape[1], scaler, job['seed'], job['arm'] == 'tied').to(device).eval()
                model.load_state_dict(checkpoint['model'], strict=True)
                with torch.inference_mode():
                    values = model(x, anchor)
                ys = np.asarray(scaler['y_std']); ym = np.asarray(scaler['y_mean'])
                assert ys.shape == ym.shape == (60,) and np.isfinite(ys).all() and np.isfinite(ym).all() and (ys > 0).all()
                max_error = 0.
                for horizon in HORIZONS:
                    raw = values[horizon].cpu().numpy()*ys[:horizon]+ym[:horizon]
                    alpha = float(done['alphas'][mode][str(horizon)])
                    assert np.isfinite(alpha) and 0. <= alpha <= 1.
                    shrunk = raw*alpha
                    for key, prediction in ((f'{mode}_h{horizon}', raw),
                                            (f'{mode}_shrink_h{horizon}', shrunk)):
                        expected = example[key]
                        assert prediction.shape == expected.shape == (len(x), horizon)
                        assert np.isfinite(prediction).all() and np.isfinite(expected).all()
                        max_error = max(max_error, float(np.max(np.abs(prediction-expected))))
                        if args.device == 'cuda':
                            assert np.array_equal(prediction, expected), f'GPU prediction mismatch: {folder.name}/{key}'
                        else:
                            assert np.allclose(prediction, expected, rtol=2e-5, atol=5e-7), f'CPU prediction tolerance exceeded: {folder.name}/{key}'
                        comparisons += 1
                assert sum(p.numel() for p in model.parameters()) == checkpoint['parameters']
                models.append({'mode': mode, 'checkpoint': name, 'epoch': checkpoint['epoch'],
                    'parameters': checkpoint['parameters'], 'bitwise_raw_and_shrunk': args.device == 'cuda',
                    'CPU_replay_tolerance': {'rtol':2e-5,'atol':5e-7} if args.device == 'cpu' else None,
                    'maximum_absolute_error': max_error})
            initial = torch.load(folder/'initial.pt', map_location='cpu', weights_only=True)
            assert initial['job'] == done['job']
            if market_count:
                assert initial['identity'] == done['identity'] and initial['rules_sha256'] == rule_hash
                assert initial['scaler'] == done['scaler'] and initial['epoch'] == 0
                assert digest(folder/'initial.pt') == done['checkpoint_hashes']['initial.pt']
            baseline = Bundle(x.shape[1], initial['scaler'], initial['job']['seed'],
                              initial['job']['arm'] == 'tied').to(device).eval()
            baseline.load_state_dict(initial['model'], strict=True)
            with torch.inference_mode():
                predictions = baseline(x, anchor)
            zero_error = max(float(np.abs(predictions[h].cpu().numpy()*np.asarray(initial['scaler']['y_std'])[:h]
                +np.asarray(initial['scaler']['y_mean'])[:h]).max()) for h in HORIZONS)
            assert zero_error < 1e-9
            receipts.append({'job': done['job'], 'models': models,
                'initial_baseline_float32_inversion_max_error': zero_error,
                'input_rows': len(x), 'horizons': list(HORIZONS)})
    assert len(receipts) == expected_runs and comparisons == expected_runs*16
    if not market_count:
        assert {r['job']['arm'] for r in receipts} == {'tied', 'untied'}
    result = {'status': 'passed', 'model_bundles': len(receipts)*2,
        'declared_market_bundles': market_count, 'declared_synthetic_bundles': synthetic_count,
        'initial_checkpoint_bundles': len(receipts), 'raw_or_shrunk_head_arrays_checked': comparisons,
        'manifest_files_verified': len(manifest['files']), 'runs': receipts,
        'module_paths_inside_extracted_bundle': True, 'weights_only': True,
        'data_used': 'provided standardized input-only EXAMPLE.npz; no target arrays',
        'market_training_or_scoring': False, 'independent_predictive_confirmation': False,
        'affinity': sorted(os.sched_getaffinity(0)), 'GPU': args.device == 'cuda',
        'forbidden_roots': [str(path) for path in denied], 'audit_guard': guard_counts}
    result['audit_guard_positive_controls'] = guard_tests
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
