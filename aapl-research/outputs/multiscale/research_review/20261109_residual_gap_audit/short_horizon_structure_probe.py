"""Check original xPatch trend-stream dimensionality on synthetic inputs only."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    assert set(os.sched_getaffinity(0)) == {24}
    protocol = json.loads((BASE/'SHORT_HORIZON_STRUCTURE_PROTOCOL.json').read_text())
    assert sha(Path(__file__)) == protocol['script_sha256']
    for path, expected in protocol['source_sha256'].items():
        assert sha(ROOT/path) == expected
    sys.path.insert(0, str(ROOT/protocol['reference_root']))
    import torch
    from aapl.vendor.xpatch_network import Network

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    rows = []
    for horizon in protocol['native_horizons']:
        for dtype_name in protocol['dtypes']:
            assert time.time() < protocol['deadline_epoch']
            dtype = getattr(torch, dtype_name)
            torch.manual_seed(protocol['seed'])
            net = Network(protocol['lookback'], horizon, 8, 4, 'end').to(dtype=dtype).eval()
            generator = torch.Generator().manual_seed(protocol['data_seed'])
            seasonal = torch.randn(protocol['batch_size'], protocol['lookback'], 1,
                                   generator=generator, dtype=dtype)
            trend1 = torch.randn(protocol['batch_size'], protocol['lookback'], 1,
                                 generator=generator, dtype=dtype).requires_grad_()
            trend2 = torch.randn(protocol['batch_size'], protocol['lookback'], 1,
                                 generator=generator, dtype=dtype)
            captured = []
            hook = net.ln2.register_forward_hook(lambda module, args, output: captured.append(output))
            output1 = net(seasonal, trend1)
            output2 = net(seasonal, trend2)
            hook.remove()
            # Fixed linear combination avoids an accidental sum cancellation.
            weights = torch.linspace(0.7, 1.3, horizon, dtype=dtype)[None, :, None]
            gradient = torch.autograd.grad((output1*weights).sum(), trend1)[0]
            pre_fc7_change = float((captured[0]-captured[1]).detach().abs().max())
            output_change = float((output1-output2).detach().abs().max())
            gradient_max = float(gradient.abs().max())
            dimension = horizon//2
            if dimension == 1:
                tol = protocol['tolerances'][dtype_name]
                passed = (pre_fc7_change <= tol['output_atol'] and
                          output_change <= tol['output_atol'] and
                          gradient_max <= tol['gradient_atol'])
            else:
                passed = output_change > protocol['positive_control_min_output_change']
            rows.append({'native_horizon':horizon,'dtype':dtype_name,
                         'second_LayerNorm_dimension':dimension,
                         'second_LayerNorm_max_difference_between_trends':pre_fc7_change,
                         'combined_output_max_difference_fixed_seasonal_changed_trend':output_change,
                         'weighted_output_gradient_wrt_trend_max_abs':gradient_max,
                         'constant_second_norm_identity_expected':dimension==1,
                         'passed_frozen_criterion':passed})
    result = {'status':'completed','protocol_sha256':sha(BASE/'SHORT_HORIZON_STRUCTURE_PROTOCOL.json'),
              'units':rows,'all_frozen_checks_passed':all(r['passed_frozen_criterion'] for r in rows),
              'market_data_or_weights_loaded':False,'market_fitting_or_scoring':False,
              'GPU':False,'Q3_values_read':False,'new_candidate_admitted':False,
              'scope':'Existing native trend substream only; the seasonal substream and price-anchor head remain input-dependent.',
              'full_research_goal_achieved':False}
    (BASE/'SHORT_HORIZON_STRUCTURE_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
