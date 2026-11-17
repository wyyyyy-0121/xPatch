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
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--deny-path', action='append', default=[])
    args = parser.parse_args(); root = args.root.resolve()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1', 'CPU replay requires CUDA_VISIBLE_DEVICES=-1'
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
    manifest = json.loads((root/'MANIFEST.json').read_text())
    for name, expected in manifest['files'].items():
        path = (root/name).resolve()
        assert path.is_relative_to(root), name
        assert digest(path) == expected, f'File hash mismatch: {name}'
    reference = root/'code/reference_source'
    sys.path.insert(0, str(reference)); sys.path.insert(0, str(root/'code'))
    from shared_network import Bundle, HORIZONS
    import shared_network
    import multiscale.models
    assert Path(shared_network.__file__).resolve().is_relative_to(root)
    assert Path(multiscale.models.__file__).resolve().is_relative_to(root)
    receipts = []; comparisons = 0
    for folder in sorted((root/'runs').iterdir()):
        if not folder.is_dir() or not (folder/'DONE.json').exists():
            continue
        done = json.loads((folder/'DONE.json').read_text())
        with np.load(folder/'EXAMPLE.npz', allow_pickle=False) as example:
            x = torch.from_numpy(example['x']); anchor = torch.from_numpy(example['anchor'])
            assert x.dtype == torch.float32 and anchor.dtype == torch.float32
            assert x.ndim == 3 and x.shape[-1] == 1 and len(x) <= 128
            assert anchor.shape == (len(x), 1)
            models = []
            for mode, name in (('last', 'last.pt'), ('selected', 'best.pt')):
                checkpoint = torch.load(folder/name, map_location='cpu', weights_only=True)
                job = checkpoint['job']; scaler = checkpoint['scaler']
                assert job['id'] == done['job']['id'] and job['arm'] in ('tied', 'untied')
                model = Bundle(x.shape[1], scaler, job['seed'], job['arm'] == 'tied').eval()
                model.load_state_dict(checkpoint['model'], strict=True)
                with torch.inference_mode():
                    values = model(x, anchor)
                ys = np.asarray(scaler['y_std']); ym = np.asarray(scaler['y_mean'])
                max_error = 0.
                for horizon in HORIZONS:
                    raw = values[horizon].numpy()*ys[:horizon]+ym[:horizon]
                    shrunk = raw*done['alphas'][mode][str(horizon)]
                    for key, prediction in ((f'{mode}_h{horizon}', raw),
                                            (f'{mode}_shrink_h{horizon}', shrunk)):
                        expected = example[key]
                        assert prediction.shape == expected.shape and np.isfinite(prediction).all()
                        max_error = max(max_error, float(np.max(np.abs(prediction-expected))))
                        assert np.array_equal(prediction, expected), f'Prediction mismatch: {folder.name}/{key}'
                        comparisons += 1
                assert sum(p.numel() for p in model.parameters()) == checkpoint['parameters']
                models.append({'mode': mode, 'checkpoint': name, 'epoch': checkpoint['epoch'],
                    'parameters': checkpoint['parameters'], 'bitwise_raw_and_shrunk': True,
                    'maximum_absolute_error': max_error})
            initial = torch.load(folder/'initial.pt', map_location='cpu', weights_only=True)
            baseline = Bundle(x.shape[1], initial['scaler'], initial['job']['seed'],
                              initial['job']['arm'] == 'tied').eval()
            baseline.load_state_dict(initial['model'], strict=True)
            with torch.inference_mode():
                predictions = baseline(x, anchor)
            zero_error = max(float(np.abs(predictions[h].numpy()*np.asarray(initial['scaler']['y_std'])[:h]
                +np.asarray(initial['scaler']['y_mean'])[:h]).max()) for h in HORIZONS)
            assert zero_error < 1e-9
            receipts.append({'job': done['job'], 'models': models,
                'initial_baseline_float32_inversion_max_error': zero_error,
                'input_rows': len(x), 'horizons': list(HORIZONS)})
    assert receipts, 'No complete replay examples supplied'
    result = {'status': 'passed', 'model_bundles': len(receipts)*2,
        'initial_checkpoint_bundles': len(receipts), 'raw_or_shrunk_head_arrays_checked': comparisons,
        'manifest_files_verified': len(manifest['files']), 'runs': receipts,
        'module_paths_inside_extracted_bundle': True, 'weights_only': True,
        'data_used': 'provided standardized input-only EXAMPLE.npz; no target arrays',
        'market_training_or_scoring': False, 'independent_predictive_confirmation': False,
        'affinity': sorted(os.sched_getaffinity(0)), 'GPU': False,
        'forbidden_roots': [str(path) for path in denied], 'audit_guard': guard_counts}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
