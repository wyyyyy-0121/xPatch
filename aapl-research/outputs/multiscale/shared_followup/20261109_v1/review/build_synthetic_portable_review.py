"""Prepare and independently replay a synthetic-only four-head context ZIP."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
REPO = BASE.parents[3]
REFERENCE = REPO/'outputs/multiscale/controlled_followup/20261108_v2/code/reference_source'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def main():
    start = time.time()
    assert set(os.sched_getaffinity(0)).issubset({24, 26})
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0, str(REFERENCE)); sys.path.insert(0, str(BASE/'code'))
    from shared_network import Bundle, Decoder, HORIZONS, native_reference
    spec = importlib.util.spec_from_file_location('shared_experiment_for_input_review', BASE/'code/experiment.py')
    experiment = importlib.util.module_from_spec(spec); spec.loader.exec_module(experiment)

    # Poison every possible label-reading method, then execute the actual
    # production validation predictor, not a duplicate helper implementation.
    class Windows:
        def __init__(self):
            self.calls = []

        def arrays(self, origins):
            self.calls.append(origins.copy())
            x = np.sin(origins[:, None]*.01 + np.arange(480)[None, :]*.03)
            return x, (10+origins[:, None]*.001)

    class PoisonLabels:
        def __init__(self, scaler):
            self.scaler = scaler; self.ym = np.asarray(scaler['y_mean']); self.ys = np.asarray(scaler['y_std'])
            self.splits = {'validation': np.arange(500, 793)}
            self.windows = Windows()

        def batch(self, *args):
            raise AssertionError('Validation Data.batch would construct labels')

        @property
        def bars(self):
            raise AssertionError('Validation predictor must not read labels or market bars')

    scaler = {'x_mean': .1, 'x_std': 1.1, 'anchor_mean': 10., 'anchor_std': 2.,
        'y_mean': np.linspace(-.002, .002, 60).tolist(),
        'y_std': np.linspace(.003, .015, 60).tolist()}
    poison = PoisonLabels(scaler)
    network = Bundle(480, scaler, 2026, True).eval()
    # A nonzero complete function ensures this is more than a zero-head test.
    for h in HORIZONS:
        network.heads[str(h)] = Decoder(native_reference(480, h, 2026))
    _, predicted = experiment.predict(network, poison, 'validation', collect=True)
    assert [len(c) for c in poison.windows.calls] == [128, 128, 37]
    assert np.array_equal(np.concatenate(poison.windows.calls), poison.splits['validation'])
    assert all(predicted[h].shape == (293, h) for h in HORIZONS)

    stage = HERE/'synthetic_portable_bundle_v2'
    assert not stage.exists(), 'Preserve prior acceptance; choose a new review directory'
    (stage/'code').mkdir(parents=True)
    shutil.copy2(BASE/'code/shared_network.py', stage/'code/shared_network.py')
    shutil.copy2(HERE/'portable_replay.py', stage/'replay.py')
    relatives = ('multiscale/__init__.py', 'multiscale/models.py', 'aapl/__init__.py',
        'aapl/models.py', 'aapl/common.py', 'aapl/vendor/__init__.py',
        'aapl/vendor/revin.py', 'aapl/vendor/xpatch_network.py',
        'aapl/vendor/patchtst_backbone.py', 'aapl/vendor/patchtst_layers.py',
        'aapl/vendor/SOURCE.json', 'aapl/vendor/XPATCH_LICENSE.txt', 'aapl/vendor/PATCHTST_LICENSE.txt')
    for name in relatives:
        destination = stage/'code/reference_source'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REFERENCE/name, destination)
    generator = torch.Generator().manual_seed(471)
    x = torch.randn(32, 480, 1, generator=generator)
    anchor = torch.randn(32, 1, generator=generator)
    target = torch.randn(32, 60, generator=generator)*.1
    for tied in (True, False):
        arm = 'tied' if tied else 'untied'
        job = {'id': f'{arm}_synthetic_portable_s2026', 'arm': arm, 'seed': 2026, 'fold': 0}
        folder = stage/'runs'/job['id']; folder.mkdir(parents=True)
        model = Bundle(480, scaler, 2026, tied)

        def record(epoch):
            return {'model': model.state_dict(), 'job': job, 'scaler': scaler,
                    'epoch': epoch, 'parameters': sum(p.numel() for p in model.parameters()),
                    'synthetic_only': True}

        torch.save(record(0), folder/'initial.pt')
        # Restore nonzero native decoders; exercise BN and actual Adam updates.
        for h in HORIZONS:
            model.heads[str(h)] = Decoder(native_reference(480, h, 2026))
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        for epoch in (1, 2):
            model.train(); optimizer.zero_grad(set_to_none=True)
            result = model(x, anchor)
            sum((result[h]-target[:, :h]).square().mean() for h in HORIZONS).backward()
            optimizer.step()
            torch.save(record(epoch), folder/('best.pt' if epoch == 1 else 'last.pt'))
        expected = {}; alphas = {}
        for mode, name in (('last', 'last.pt'), ('selected', 'best.pt')):
            saved = torch.load(folder/name, map_location='cpu', weights_only=True)
            model.load_state_dict(saved['model']); model.eval()
            with torch.inference_mode():
                result = model(x, anchor)
            alphas[mode] = {}
            for index, h in enumerate(HORIZONS):
                alpha = (.1 if mode == 'selected' else .2)*(index+1)
                alphas[mode][str(h)] = alpha
                raw = result[h].numpy()*np.asarray(scaler['y_std'])[:h]+np.asarray(scaler['y_mean'])[:h]
                expected[f'{mode}_h{h}'] = raw
                expected[f'{mode}_shrink_h{h}'] = raw*alpha
        np.savez_compressed(folder/'EXAMPLE.npz', x=x.numpy(), anchor=anchor.numpy(), **expected)
        write(folder/'DONE.json', {'job': job, 'alphas': alphas, 'synthetic_only': True,
            'market_training_or_scoring': False, 'meaning': 'engineering fixture, not research evidence'})
    write(stage/'BUNDLE_SCOPE.json', {'status': 'synthetic_engineering_fixture_only',
        'market_data_or_training': False, 'research_experiment_complete': False,
        'independent_confirmation': False, 'horizons': list(HORIZONS),
        'sources': {'shared_network': sha(BASE/'code/shared_network.py'),
                    'native_models': sha(REFERENCE/'multiscale/models.py')}})
    files = {str(p.relative_to(stage)): sha(p) for p in sorted(stage.rglob('*')) if p.is_file()}
    write(stage/'MANIFEST.json', {'files': files, 'synthetic_only': True})
    archive = HERE/'SYNTHETIC_FOUR_HEAD_REPLAY_V2.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(stage.rglob('*')):
            if path.is_file():
                zipped.write(path, str(path.relative_to(stage)))
    # Extract outside the workspace; original source/data paths and network
    # operations are blocked by the replay's real Python audit hook.
    with tempfile.TemporaryDirectory(prefix='aapl-shared-external-replay-') as temporary:
        outside = Path(temporary)
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(outside)
        receipt = outside/'ACCEPTANCE.json'
        env = dict(os.environ, PYTHONPATH='', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1')
        result = subprocess.run([sys.executable, str(outside/'replay.py'), '--root', str(outside),
            '--deny-path', str(REPO), '--guard-self-test', '--output', str(receipt)], cwd=outside, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
        (HERE/'portable_replay_stdout.log').write_text(result.stdout)
        (HERE/'portable_replay_stderr.log').write_text(result.stderr)
        assert result.returncode == 0, result.stderr
        replay = json.loads(receipt.read_text())
    report = {'status': 'passed', 'scope': 'synthetic engineering only',
        'actual_validation_predictor_rejects_all_label_paths': True,
        'validation_input_only_chunks': [128, 128, 37],
        'external_extraction_workspace_and_network_audit_guard': True,
        'archive': {'path': str(archive), 'bytes': archive.stat().st_size, 'sha256': sha(archive)},
        'replay': replay, 'sources': {'experiment.py': sha(BASE/'code/experiment.py'),
            'shared_network.py': sha(BASE/'code/shared_network.py'),
            'portable_replay.py': sha(HERE/'portable_replay.py')},
        'market_training_or_scoring': False, 'Q3_values_read': False,
        'research_experiment_complete': False, 'elapsed_seconds': time.time()-start,
        'affinity': sorted(os.sched_getaffinity(0)), 'GPU': False}
    write(HERE/'PORTABLE_FOUR_HEAD_ACCEPTANCE.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
