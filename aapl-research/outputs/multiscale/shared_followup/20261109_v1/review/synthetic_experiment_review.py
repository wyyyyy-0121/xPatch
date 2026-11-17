"""Independent, fully synthetic audit; never instantiate the market Data loader."""
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import numpy as np
import torch

REVIEW = Path(__file__).resolve().parent
BASE = REVIEW.parent
sys.path.insert(0, str(BASE / 'code'))
exp = importlib.import_module('experiment')
from shared_network import Bundle, HORIZONS, native_reference
bars_module = importlib.import_module('multiscale.bars')

assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
assert len(os.sched_getaffinity(0)) <= 2
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)


class SyntheticData:
    def __init__(self):
        rng = np.random.default_rng(119)
        self.scaler = {'x_mean': 0., 'x_std': 1., 'anchor_mean': 0., 'anchor_std': 1.,
            'y_mean': np.linspace(-.0001, .0001, 60).tolist(),
            'y_std': np.linspace(.003, .02, 60).tolist()}
        self.ym = np.array(self.scaler['y_mean'])
        self.ys = np.array(self.scaler['y_std'])
        self.splits = {'fit': np.arange(257), 'stop': np.arange(257, 274),
            'validation': np.arange(274, 293)}
        self.x = rng.normal(0, .25, (293, 480, 1)).astype('float32')
        self.anchor = rng.normal(0, .25, (293, 1)).astype('float32')
        self.y = rng.normal(0, 1, (293, 60)).astype('float32')
        self.raw = self.y*self.ys+self.ym
        self.bars = SimpleNamespace(synthetic=True, synthetic_targets=self.raw)
        self.windows = SimpleNamespace(arrays=lambda origins: (self.x[origins, :, 0], self.anchor[origins]))
        self.reads = []

    def batch(self, part, pos):
        self.reads.append(part)
        ix = self.splits[part][pos]
        return (torch.from_numpy(self.x[ix]), torch.from_numpy(self.anchor[ix])), torch.from_numpy(self.y[ix])


def compare(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(compare(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(compare(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    return a == b


def original_native_state(bundle, h):
    encoder = bundle.encoders[str(h)]
    decoder = bundle.heads[str(h)]
    state = {}
    for name, value in encoder.state_dict().items():
        state[name if name.startswith('ema.') else 'backbone.'+name] = value
    for name, value in decoder.state_dict().items():
        state[name if name.startswith('anchor_head.') else 'backbone.'+name] = value
    return state


def native_optimizer_equivalence():
    data = SyntheticData()
    bundle = Bundle(480, data.scaler, 2026, False)
    opt = torch.optim.Adam(bundle.parameters(), lr=.001)
    natives = {h: native_reference(480, h, 2026, data.scaler) for h in HORIZONS}
    opts = {h: torch.optim.Adam(natives[h].parameters(), lr=.001) for h in HORIZONS}
    assert all(compare(original_native_state(bundle, h), natives[h].state_dict()) for h in HORIZONS)
    grad_receipts = []
    for step in range(3):
        args, y = data.batch('fit', np.arange(step*16, (step+1)*16))
        bundle.train(); opt.zero_grad(set_to_none=True)
        value = exp.loss(bundle(*args), y); value.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in bundle.parameters())
        grad_receipts.append({'step': step+1,
            'parameters_with_gradient': sum(p.grad is not None for p in bundle.parameters()),
            'nonzero_gradient_parameters': sum(torch.count_nonzero(p.grad).item() > 0 for p in bundle.parameters())})
        opt.step()
        for h in HORIZONS:
            native = natives[h]; native.train(); opts[h].zero_grad(set_to_none=True)
            loss = (native(*args)-y[:, :h]).square().mean(); loss.backward(); opts[h].step()
            assert compare(original_native_state(bundle, h), native.state_dict()), (step, h)
    return {'updates': 3, 'all_native_weights_and_BN_buffers_bitwise_equal': True,
        'gradient_receipts': grad_receipts}


def run_resume_audit():
    root = REVIEW / 'synthetic_only'
    retry = 1
    while root.exists():
        root = REVIEW / ('synthetic_only_retry_'+str(retry))
        retry += 1
    root.mkdir()
    all_data = []
    def factory(_):
        data = SyntheticData(); all_data.append(data); return data
    exp.data_for = factory
    exp.freeze = lambda: {'admitted': True, 'fixed_epochs': 6}
    original_targets = bars_module.targets
    def synthetic_targets(bars, origins, horizon, target):
        assert getattr(bars, 'synthetic', False), 'Market data reached synthetic audit'
        assert horizon == 60 and target == 'future_log_return'
        assert np.isin(origins, np.arange(257, 274)).all(), 'Non-stop labels reached post-training targets()'
        return bars.synthetic_targets[origins, :horizon]
    bars_module.targets = synthetic_targets
    rows = []
    try:
        for arm in ('tied', 'untied'):
            job = {'id': arm+'_synthetic', 'arm': arm, 'fold': 1, 'seed': 2026}
            dirs = {}
            for mode in ('uninterrupted', 'interrupted'):
                work = root / arm / mode
                (work / 'protocol').mkdir(parents=True)
                (work / 'protocol/RULES.json').write_text('{"synthetic_only": true, "fixed_epochs": 6}\n')
                exp.BASE = work
                exp.STOP = False
                if mode == 'interrupted':
                    paused = exp.run(job, steps_limit=1)
                    assert paused['status'] == 'paused'
                    saved = torch.load(work/'runs'/job['id']/'latest.pt', weights_only=True)
                    assert saved['state']['cursor'] == 128 and saved['state']['steps'] == 1
                done = exp.run(job)
                assert done['status'] == 'completed' and done['steps'] == 12
                dirs[mode] = work/'runs'/job['id']
            a = torch.load(dirs['uninterrupted']/'latest.pt', weights_only=True)
            b = torch.load(dirs['interrupted']/'latest.pt', weights_only=True)
            for key in ('identity','model','optimizer','numpy_rng','torch_rng','epoch','scaler'):
                assert compare(a[key], b[key]), (arm, key)
            for key in ('epoch','cursor','permutation','steps','best_score','best_epoch','history','loss_sum','seen'):
                assert compare(a['state'][key], b['state'][key]), (arm, 'state', key)
            for mode in ('last','best'):
                ca = torch.load(dirs['uninterrupted']/(mode+'.pt'), weights_only=True)
                cb = torch.load(dirs['interrupted']/(mode+'.pt'), weights_only=True)
                assert compare(ca, cb), (arm, mode)
            with np.load(dirs['uninterrupted']/'predictions.npz') as pa, np.load(dirs['interrupted']/'predictions.npz') as pb:
                assert set(pa.files) == set(pb.files) and len(pa.files) == 17
                assert all(np.array_equal(pa[k], pb[k]) for k in pa.files)
                assert all(pa[f'{mode}{shrink}_h{h}'].shape == (19,h)
                    for mode in ('last','selected') for shrink in ('','_shrink') for h in HORIZONS)
            bn = {k: int(v) for k,v in a['model'].items() if k.endswith('num_batches_tracked')}
            assert all(v == 12 for v in bn.values()), bn
            assert len(bn) == (3 if arm == 'tied' else 12)
            rows.append({'arm': arm, 'fixed_epochs': 6, 'steps': 12,
                'resume_model_optimizer_RNG_cursor_history_bitwise_equal': True,
                'last_and_best_checkpoints_bitwise_equal': True,
                'all_16_prediction_arrays_bitwise_equal': True,
                'batchnorm_counters': bn})
    finally:
        bars_module.targets = original_targets
        exp.BASE = BASE
    assert all('validation' not in data.reads for data in all_data), 'Outer targets constructed through data.batch'
    return {'evidence_root': str(root), 'rows': rows, 'synthetic_data_instances': len(all_data),
        'observed_batch_part_calls': {part: sum(d.reads.count(part) for d in all_data)
            for part in ('fit','stop','validation')}}


hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (BASE/'code').glob('*.py')}
results = {'scope': 'synthetic_only_no_market_data_instantiation_or_targets',
    'source_hashes_before': hashes, 'optimizer': native_optimizer_equivalence(),
    'resume': run_resume_audit(), 'Q3_values_read': False, 'GPU': False}
results['source_hashes_after'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (BASE/'code').glob('*.py')}
results['same_sources_during_audit'] = results['source_hashes_before'] == results['source_hashes_after']
(REVIEW/'SYNTHETIC_EXPERIMENT_REVIEW.json').write_text(json.dumps(results, indent=2)+'\n')
print(json.dumps(results, indent=2))
