"""Synthetic native-model identification check; never load market data or weights."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import os
import sys
import tempfile
import time

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
REFERENCE = ROOT / 'outputs/multiscale/mechanism_repair/20261109_v3/code/reference_source'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, value):
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    assert set(os.sched_getaffinity(0)) == {24}
    protocol = json.loads((BASE / 'SCALE_EQUIVALENCE_PROTOCOL.json').read_text())
    assert sha(Path(__file__)) == protocol['script_sha256']
    for name, expected in protocol['source_sha256'].items():
        assert sha(ROOT / name) == expected
    sys.path.insert(0, str(REFERENCE))
    import torch
    from multiscale.models import make_model

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.time()
    progress = {'status': 'running', 'units': [], 'scope': 'synthetic_models_only',
                'protocol_sha256': sha(BASE / 'SCALE_EQUIVALENCE_PROTOCOL.json'),
                'market_data_or_saved_weights_loaded': False, 'Q3_values_read': False,
                'GPU': False, 'forecasting_candidate_admitted': False}

    def persist():
        atomic(BASE / 'SCALE_EQUIVALENCE_PROGRESS.json', progress)

    def deadline():
        if time.time() >= protocol['deadline_epoch']:
            progress['status'] = 'stopped_at_frozen_deadline'
            persist()
            raise SystemExit(2)

    def head_names(model):
        return {'head.weight', 'head.bias'} if model == 'lstm' else {
            'backbone.fc8.weight', 'backbone.fc8.bias', 'anchor_head.weight', 'anchor_head.bias'}

    def mapped_parameters(model, residual, ratio):
        # Physical price: mP+sP*zP = P+mR+sR*zR, P=mA+sA*a.
        values = {n: p.detach().clone() for n, p in residual.named_parameters()}
        for n in head_names(model):
            values[n].mul_(ratio)
        scale_p = protocol['residual_scale'] / ratio
        if model == 'lstm':
            values['head.weight'][:, -1].add_(protocol['anchor_std'] / scale_p)
            values['head.bias'].add_((protocol['anchor_mean'] + protocol['residual_mean']
                                     - protocol['price_mean']) / scale_p)
        else:
            values['anchor_head.weight'][:, 0].add_(protocol['anchor_std'] / scale_p)
            values['anchor_head.bias'].add_((protocol['anchor_mean'] + protocol['residual_mean']
                                            - protocol['price_mean']) / scale_p)
        return values

    def normalized_error(actual, expected):
        if not actual.is_floating_point():
            return 0.0 if torch.equal(actual, expected) else float('inf')
        return float((actual - expected).abs().max() / (1 + expected.abs().max()))

    def make_pair(model, dtype, initialization, ratio):
        torch.manual_seed(protocol['seed'])
        residual = make_model(model, protocol['lookback'], 1).to(dtype=dtype)
        if initialization == 'baseline':
            with torch.no_grad():
                if model == 'lstm':
                    residual.head.weight.zero_()
                    residual.head.bias.fill_(-protocol['residual_mean'] / protocol['residual_scale'])
                else:
                    residual.backbone.fc8.weight.zero_()
                    residual.backbone.fc8.bias.zero_()
                    residual.anchor_head.weight.zero_()
                    residual.anchor_head.bias.fill_(-protocol['residual_mean'] / protocol['residual_scale'])
        direct = copy.deepcopy(residual)
        with torch.no_grad():
            for name, param in direct.named_parameters():
                param.copy_(mapped_parameters(model, residual, ratio)[name])
        return residual, direct

    def run_unit(model, dtype_name, initialization, ratio, steps):
        deadline()
        dtype = getattr(torch, dtype_name)
        residual, direct = make_pair(model, dtype, initialization, ratio)
        optimizers = [torch.optim.Adam(net.parameters(), lr=protocol['learning_rate'],
                                      eps=protocol['adam_epsilon'], weight_decay=0,
                                      foreach=False, fused=False) for net in [residual, direct]]
        generator = torch.Generator().manual_seed(protocol['data_seed'])
        result = {'model': model, 'horizon': 1, 'lookback': protocol['lookback'],
                  'dtype': dtype_name, 'initialization': initialization,
                  'residual_to_price_scale_ratio': ratio, 'steps': [],
                  'initial_physical_forecast_max_abs_difference': None,
                  'primary_common_scale_null': ratio == 1.0}
        for step in range(steps):
            deadline()
            x = torch.randn(protocol['batch_size'], protocol['lookback'], 1,
                            generator=generator, dtype=dtype)
            a = torch.randn(protocol['batch_size'], 1, generator=generator, dtype=dtype)
            anchor = protocol['anchor_mean'] + protocol['anchor_std'] * a
            noise = torch.randn(protocol['batch_size'], 1, generator=generator, dtype=dtype) * 0.2
            truth = anchor + 0.3*x[:, -8:].mean(dim=1) + noise
            s_r = protocol['residual_scale']
            s_p = s_r / ratio
            target_r = (truth - anchor - protocol['residual_mean']) / s_r
            target_p = (truth - protocol['price_mean']) / s_p
            for net, opt in zip([residual, direct], optimizers):
                net.train()
                opt.zero_grad(set_to_none=True)
            z_r, z_p = residual(x, a), direct(x, a)
            p_r = anchor + protocol['residual_mean'] + s_r*z_r
            p_p = protocol['price_mean'] + s_p*z_p
            error_r, error_p = z_r-target_r, z_p-target_p
            loss_r, loss_p = error_r.square().mean(), error_p.square().mean()
            forecast_difference = float((p_r-p_p).detach().abs().max())
            if step == 0:
                result['initial_physical_forecast_max_abs_difference'] = forecast_difference
            loss_r.backward()
            loss_p.backward()
            rp, dp = dict(residual.named_parameters()), dict(direct.named_parameters())
            gradient_identity = max(normalized_error(dp[n].grad, rp[n].grad *
                (ratio if n in head_names(model) else ratio**2)) for n in rp)
            row = {'step': step+1, 'physical_forecast_max_abs_difference_before_update': forecast_difference,
                   'standardized_error_scaling_max_abs_difference': float(
                       (error_p-ratio*error_r).detach().abs().max()),
                   'gradient_scale_identity_normalized_max_abs_difference': gradient_identity,
                   'loss_scale_identity_abs_difference': abs(float(loss_p.detach())-ratio**2*float(loss_r.detach()))}
            for opt in optimizers:
                opt.step()
            mapped = mapped_parameters(model, residual, ratio)
            row['mapped_parameter_normalized_max_abs_difference_after_update'] = max(
                normalized_error(dp[n].detach(), mapped[n]) for n in rp)
            buffers_r, buffers_p = dict(residual.named_buffers()), dict(direct.named_buffers())
            row['buffer_normalized_max_abs_difference'] = max(
                [normalized_error(buffers_p[n], buffers_r[n]) for n in buffers_r] or [0.0])
            state_differences = []
            for n in rp:
                sr, sp = optimizers[0].state[rp[n]], optimizers[1].state[dp[n]]
                for key in ('step', 'exp_avg', 'exp_avg_sq'):
                    state_differences.append(normalized_error(sp[key], sr[key]))
            row['adam_state_normalized_max_abs_difference'] = max(state_differences)
            if ratio == 1.0:
                tol = protocol['tolerances'][dtype_name]
                row['common_scale_null_passed'] = (
                    forecast_difference <= tol['physical_forecast_atol'] and
                    row['standardized_error_scaling_max_abs_difference'] <= tol['standardized_error_atol'] and
                    gradient_identity <= tol['normalized_atol'] and
                    row['mapped_parameter_normalized_max_abs_difference_after_update'] <= tol['normalized_atol'] and
                    row['buffer_normalized_max_abs_difference'] <= tol['normalized_atol'] and
                    row['adam_state_normalized_max_abs_difference'] <= tol['normalized_atol'])
            result['steps'].append(row)
            progress['current_unit'] = {k: v for k, v in result.items() if k != 'steps'}
            progress['current_unit_steps_saved'] = result['steps']
            persist()
        if ratio == 1.0:
            result['passed'] = all(r['common_scale_null_passed'] for r in result['steps'])
        else:
            tol = protocol['tolerances'][dtype_name]
            result['passed'] = (result['initial_physical_forecast_max_abs_difference'] <= tol['physical_forecast_atol']
                and result['steps'][0]['gradient_scale_identity_normalized_max_abs_difference'] <= tol['normalized_atol']
                and result['steps'][1]['physical_forecast_max_abs_difference_before_update'] > protocol['scale_effect_min_gap'])
            result['interpretation'] = 'Matched initial physical forecasts; isolated target scale changes gradient scales and subsequent Adam trajectory on synthetic inputs.'
        return result

    persist()
    for model in protocol['models']:
        for dtype in protocol['dtypes']:
            for initialization in protocol['initializations']:
                result = run_unit(model, dtype, initialization, 1.0, protocol['common_scale_steps'])
                progress['units'].append(result)
                persist()
    for model in protocol['models']:
        result = run_unit(model, 'float64', 'random_mapped', protocol['scale_ratio_control'],
                          protocol['different_scale_steps'])
        progress['units'].append(result)
        persist()
    progress.pop('current_unit', None)
    progress.pop('current_unit_steps_saved', None)
    progress.update(status='completed', elapsed_seconds=time.time()-started,
                    all_common_scale_nulls_passed=all(u['passed'] for u in progress['units'] if u['primary_common_scale_null']),
                    both_scale_controls_passed=all(u['passed'] for u in progress['units'] if not u['primary_common_scale_null']),
                    native_model_updates_not_market_training=True,
                    scientific_scope='Synthetic parameterization identity and isolated numerical scale effect; not attribution of observed market MSE or predictive success.',
                    research_goal_achieved=False, completed_at_UTC=datetime.now(timezone.utc).isoformat())
    atomic(BASE / 'SCALE_EQUIVALENCE_RESULTS.json', progress)
    persist()
    print(json.dumps({k:v for k,v in progress.items() if k != 'units'}, indent=2))


if __name__ == '__main__':
    main()
