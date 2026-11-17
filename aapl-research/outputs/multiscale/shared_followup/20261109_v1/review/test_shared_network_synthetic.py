"""Synthetic-only native restoration, actual sharing, gradient and load checks."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
REPO = BASE.parents[3]
REFERENCE = REPO/'outputs/multiscale/controlled_followup/20261108_v2/code/reference_source'
sys.path.insert(0, str(REFERENCE))
sys.path.insert(0, str(BASE/'code'))
from shared_network import Bundle, Decoder, HORIZONS, native_reference


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def all_equal(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a)


def main():
    start = time.time()
    assert set(os.sched_getaffinity(0)).issubset({24, 26})
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True); torch.manual_seed(731)
    scaler = {'y_mean': np.linspace(-.002, .002, 60).tolist(), 'y_std': np.linspace(.003, .015, 60).tolist()}
    rng_before = torch.get_rng_state().clone()
    tied = Bundle(480, scaler, 2026, tied=True)
    untied = Bundle(480, scaler, 2026, tied=False)
    assert torch.equal(rng_before, torch.get_rng_state())
    assert all_equal(tied.heads.state_dict(), untied.heads.state_dict())
    for h in HORIZONS:
        assert all_equal(tied.encoder.state_dict(), untied.encoders[str(h)].state_dict())
    tied_counts = tied.parameter_counts(); untied_counts = untied.parameter_counts()
    assert untied_counts['total']-tied_counts['total'] == 3*tied_counts['one_nonlinear_encoder']
    encoders = [untied.encoders[str(h)] for h in HORIZONS]
    pointers = [{p.data_ptr() for p in encoder.parameters()} for encoder in encoders]
    assert all(not pointers[i].intersection(pointers[j]) for i in range(4) for j in range(i))
    head_pointers = [{p.data_ptr() for p in tied.heads[str(h)].parameters()} for h in HORIZONS]
    assert all(not head_pointers[i].intersection(head_pointers[j]) for i in range(4) for j in range(i))
    x = torch.randn(8, 480, 1); anchor = torch.randn(8, 1)
    tied.eval(); untied.eval(); baseline_max = 0.
    with torch.inference_mode():
        first = tied(x, anchor); second = untied(x, anchor)
        for h in HORIZONS:
            reference = native_reference(480, h, 2026, scaler).eval()
            assert torch.equal(first[h], second[h]) and torch.equal(first[h], reference(x, anchor))
            assert first[h].shape == (len(x), h)
            raw = first[h].double().numpy()*np.asarray(scaler['y_std'])[:h]+np.asarray(scaler['y_mean'])[:h]
            baseline_max = max(baseline_max, float(np.max(np.abs(raw))))
    assert baseline_max < 1e-9
    # A zero head hides encoder errors; restore every original random native
    # decoder to test full nonzero functions through the extracted boundary.
    references = {h: native_reference(480, h, 2026).eval() for h in HORIZONS}
    random_heads = nn.ModuleDict({str(h): Decoder(references[h]) for h in HORIZONS})
    tied.heads.load_state_dict(random_heads.state_dict())
    untied.heads.load_state_dict(random_heads.state_dict())
    with torch.inference_mode():
        a, b = tied(x, anchor), untied(x, anchor)
        for h in HORIZONS:
            assert torch.equal(a[h], b[h]) and torch.equal(a[h], references[h](x, anchor)), h
        assert not torch.equal(a[5], a[60][:, :5])
    # Each shared BN counter advances once, not four times, per bundle batch.
    for model in (tied, untied):
        model.train()
        before = {name: value.clone() for name, value in model.named_buffers() if name.endswith('num_batches_tracked')}
        model(x, anchor)
        after = dict(model.named_buffers())
        assert len(before) == (3 if model.tied else 12)
        assert all(int(after[name]-value) == 1 for name, value in before.items())
        model.eval(); model.zero_grad(set_to_none=True)
        model(x, anchor)[15].square().mean().backward()
        for h in HORIZONS:
            for p in model.heads[str(h)].parameters():
                assert (p.grad is not None and torch.isfinite(p.grad).all()) if h == 15 else p.grad is None
        if model.tied:
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.encoder.parameters())
        else:
            for h in HORIZONS:
                for p in model.encoders[str(h)].parameters():
                    assert (p.grad is not None and torch.isfinite(p.grad).all()) if h == 15 else p.grad is None
    # Four task losses really back-propagate into the same encoder parameters.
    per_horizon = []
    for h in HORIZONS:
        tied.zero_grad(set_to_none=True)
        tied(x, anchor)[h].square().mean().backward()
        per_horizon.append({name: p.grad.clone() for name, p in tied.encoder.named_parameters()})
    tied.zero_grad(set_to_none=True)
    sum(value.square().mean() for value in tied(x, anchor).values()).backward()
    for name, p in tied.encoder.named_parameters():
        expected = sum(part[name] for part in per_horizon)
        assert torch.allclose(p.grad, expected, rtol=2e-5, atol=2e-7), name
    for model in (tied, untied):
        model.eval()
        with torch.inference_mode():
            expected = {h: value.clone() for h, value in model(x, anchor).items()}
        with tempfile.TemporaryDirectory(prefix='shared-bundle-synthetic-') as temporary:
            path = Path(temporary)/'model.pt'
            torch.save({'model': model.state_dict(), 'lookback': 480, 'seed': 2026,
                        'scaler': scaler, 'tied': model.tied}, path)
            saved = torch.load(path, map_location='cpu', weights_only=True)
            restored = Bundle(saved['lookback'], saved['scaler'], saved['seed'], saved['tied']).eval()
            restored.load_state_dict(saved['model'])
            with torch.inference_mode():
                actual = restored(x, anchor)
            assert all(torch.equal(expected[h], actual[h]) for h in HORIZONS)
    result = {'status': 'passed', 'checks': {
        'native_random_nonzero_functions_bitwise_restored_all4heads': True,
        'native_baseline_functions_bitwise_restored_all4heads': True,
        'tied_untied_matched_initial_encoder_heads_and_functions': True,
        'construction_does_not_consume_caller_rng': True,
        'true_H_specific_heads_not_H60_prefix': True,
        'one_shared_BN_update_per_batch': True, 'untied_parameter_and_BN_storage_independent': True,
        'head_gradient_isolation_and_shared_encoder_accumulation': True,
        'weights_only_state_dict_roundtrip_bitwise_predictions': True},
        'baseline_raw_max_absolute_float32_inversion_error': baseline_max,
        'parameters': {'tied': tied_counts, 'untied': untied_counts},
        'scaler': 'synthetic H60 means/std only; no market scaler or labels',
        'source_sha256': sha(BASE/'code/shared_network.py'),
        'native_module_file': str(Path(sys.modules[Bundle.__init__.__globals__['NativeXPatch'].__module__].__file__).resolve()),
        'native_module_sha256': sha(Path(sys.modules[Bundle.__init__.__globals__['NativeXPatch'].__module__].__file__)),
        'test_sha256': sha(Path(__file__)), 'elapsed_seconds': time.time()-start,
        'affinity': sorted(os.sched_getaffinity(0)), 'GPU': False,
        'market_training_or_scoring': False, 'Q3_values_read': False}
    (HERE/'SHARED_NETWORK_SYNTHETIC_ACCEPTANCE.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
