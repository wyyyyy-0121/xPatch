"""CPU-only single-step diagnostic adapter. Importing does not run a model.

No data loader, market runner, file writes, or automatic execution is present.
Actual model acceptance and market use require the separate authorization gate.
The original GPU training trajectory is not replayed bitwise on CPU.
"""
from copy import deepcopy
import hashlib
import json
import numpy as np
import torch
from local_update_probe import fixed_order_projection

HORIZONS = (5, 15, 30, 60)


def state_fingerprint(value):
    """Hash ephemeral states for ordinary-backward equality without saving weights."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            array = item.detach().contiguous().cpu().numpy()
            digest.update(b'tensor:')
            digest.update(str(array.dtype).encode())
            digest.update(json.dumps(list(array.shape)).encode())
            digest.update(array.tobytes())
        elif isinstance(item, dict):
            digest.update(b'dict:')
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                visit(key); visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode()+b':')
            for member in item:
                visit(member)
        else:
            digest.update(type(item).__name__.encode()+b':')
            digest.update(json.dumps(item, allow_nan=False).encode())
        digest.update(b';')
    visit(value)
    return digest.hexdigest()


def copied_state(net):
    return {name: value.detach().cpu().clone()
            for name, value in net.state_dict().items()}


def flatten(values):
    return torch.cat([value.detach().reshape(-1).double().cpu() for value in values])


def make_cpu_adam(net, saved):
    """Preserve moments/step/recipe; use explicit CPU, non-fused Adam arithmetic."""
    named = dict(net.named_parameters())
    groups = saved.get('param_groups', [])
    if len(groups) != 1 or groups[0]['params'] != list(range(len(named))):
        raise ValueError('Unexpected original optimizer parameter registration order')
    states = saved.get('state', {})
    if states and set(states) != set(range(len(named))):
        raise ValueError('Incomplete or unregistered Adam parameter histories')
    for number, parameter in enumerate(named.values()):
        if states:
            state = states[number]
            for name in ('exp_avg', 'exp_avg_sq'):
                if state[name].shape != parameter.shape or not torch.isfinite(state[name]).all():
                    raise ValueError('Moment-to-parameter layout or finiteness mismatch')
            if (state['exp_avg_sq'] < 0).any():
                raise ValueError('Negative second moment')
            step = state['step']
            if not isinstance(step, torch.Tensor) or step.numel() != 1 or (
                    not torch.isfinite(step).all() or float(step) < 0 or
                    float(step) != int(float(step))):
                raise ValueError('Adam step must be a finite nonnegative integer scalar')
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001,
                                 fused=False, foreach=False)
    optimizer.load_state_dict(deepcopy(saved))
    if len(optimizer.param_groups) != 1:
        raise ValueError('Expected the original single optimizer parameter group')
    group = optimizer.param_groups[0]
    expected = {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-8,
                'weight_decay': 0, 'amsgrad': False, 'maximize': False}
    for key, value in expected.items():
        if group.get(key, False if key == 'maximize' else None) != value:
            raise ValueError('Original Adam recipe differs: ' + key)
    group['fused'] = False
    group['foreach'] = False
    group['capturable'] = False
    return optimizer


def diagnostic_metrics(net, args, truth, guard):
    guard()
    net.eval()
    with torch.no_grad():
        prediction = net(*args)
        result = {}
        for horizon in HORIZONS:
            error = (prediction[horizon] - truth[:, :horizon]).double()
            result[str(horizon)] = {
                'standardized_path_mse': float(error.square().mean()),
                'standardized_endpoint_mse': float(error[:, -1].square().mean())}
    if not all(np.isfinite(value) for row in result.values() for value in row.values()):
        raise ValueError('Nonfinite diagnostic metric')
    guard()
    return result


def local_unit(net, saved_optimizer, update_args, update_truth,
               diagnostic_args, diagnostic_truth, guard):
    """Compare A/B/C on cloned parameters, keeping buffers and heads matched.

    The caller owns loading and provenance. This function clones its model,
    never updates input parameters, and returns JSON-compatible scalar evidence.
    Completed units are independently resumable at the runtime layer.
    """
    guard()
    if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in net.parameters()):
        raise ValueError('Only explicit CPU float32 diagnostic models are admitted')
    original = copied_state(net)
    original_buffers = {name: value.detach().clone() for name, value in net.named_buffers()}
    work = deepcopy(net)
    parameters = dict(work.named_parameters())
    encoder_names = [name for name in parameters if name.startswith('encoder.')]
    if not encoder_names or not getattr(work, 'tied', False):
        raise ValueError('Require the original tied native bundle')
    encoder_parameters = [parameters[name] for name in encoder_names]
    work.train()
    guard()
    prediction = work(*update_args)  # Exactly one shared train-mode BN update.
    losses = [(prediction[h] - update_truth[:, :h]).square().mean() for h in HORIZONS]
    total = sum(losses)
    if not torch.isfinite(total):
        raise ValueError('Nonfinite update-batch loss')
    all_gradients = torch.autograd.grad(total, tuple(parameters.values()), retain_graph=True)
    if any(not torch.isfinite(value).all() for value in all_gradients):
        raise ValueError('Nonfinite total gradient')
    gradients = dict(zip(parameters, all_gradients))
    head_rows = []
    for number, loss in enumerate(losses):
        guard()
        row = torch.autograd.grad(loss, encoder_parameters,
                                  retain_graph=number < len(losses)-1)
        head_rows.append(flatten(row).numpy())
    head_rows = np.stack(head_rows)
    original_sum = flatten([gradients[name] for name in encoder_names]).numpy()
    if not np.isfinite(head_rows).all():
        raise ValueError('Nonfinite per-head gradient')
    reconstructed = head_rows.sum(axis=0)
    residual = reconstructed-original_sum
    # FP32 total-backward and per-head-backward use different accumulation
    # paths. Judge backward error relative to the component scale, including
    # cancellation, then require a stricter actual-Adam-update null below.
    component_scale = float(np.linalg.norm(head_rows, axis=1).sum())
    # This is an engineering tolerance revised after the original coordinate
    # assertion failed; it is not a rigorous floating-point error bound.
    backward_error_bound = 16*np.finfo(np.float32).eps*component_scale
    if float(np.linalg.norm(residual)) > backward_error_bound:
        raise ValueError('FP32 per-head reconstruction backward error exceeds the frozen bound')
    cursor = 0
    reconstruction_layers = {}
    for name, parameter in zip(encoder_names, encoder_parameters):
        n = parameter.numel(); rows = head_rows[:, cursor:cursor+n]
        r = residual[cursor:cursor+n]
        scale = float(np.linalg.norm(rows, axis=1).sum())
        tolerance = 16*np.finfo(np.float32).eps*scale
        if float(np.linalg.norm(r)) > tolerance:
            raise ValueError('Per-parameter FP32 reconstruction tolerance failed: '+name)
        reconstruction_layers[name] = {'residual_L2': float(np.linalg.norm(r)),
                                      'component_L2_scale': scale, 'tolerance': tolerance}
        cursor += n
    worst = int(np.argmax(np.abs(residual)/(1e-7+2e-4*np.abs(original_sum))))
    worst_coordinate = {'flat_index': worst, 'head_values': head_rows[:, worst].tolist(),
                        'total_gradient': float(original_sum[worst]),
                        'summed_gradient': float(reconstructed[worst]),
                        'original_coordinate_check_would_pass': bool(np.allclose(
                            reconstructed, original_sum, rtol=2e-4, atol=1e-7))}
    # Independent higher-precision identity check: same pre-forward model,
    # inputs and targets, no update and no diagnostic-batch/outer scoring.
    reference64 = deepcopy(net).double(); reference64.train()
    p64 = reference64(*(value.double() for value in update_args))
    ls64 = [(p64[h]-update_truth[:, :h].double()).square().mean() for h in HORIZONS]
    enc64 = [value for name, value in reference64.named_parameters()
             if name.startswith('encoder.')]
    total64 = flatten(torch.autograd.grad(sum(ls64), enc64, retain_graph=True)).numpy()
    rows64 = np.stack([flatten(torch.autograd.grad(value, enc64,
                       retain_graph=i < len(ls64)-1)).numpy()
                       for i, value in enumerate(ls64)])
    residual64 = rows64.sum(axis=0)-total64
    bound64 = 4096*np.finfo(np.float64).eps*float(np.linalg.norm(rows64, axis=1).sum())
    if float(np.linalg.norm(residual64)) > bound64:
        raise ValueError('Higher-precision gradient decomposition identity failed')
    del reference64, p64, ls64, enc64
    projected_sum = fixed_order_projection(head_rows).sum(axis=0)
    post_forward_buffers = {name: value.detach().clone()
                            for name, value in work.named_buffers()}
    before_parameters = {name: value.detach().clone() for name, value in parameters.items()}

    def updated(projected, replacement=None):
        guard()
        candidate = deepcopy(work)
        optimizer = make_cpu_adam(candidate, saved_optimizer)
        current = dict(candidate.named_parameters())
        cursor = 0
        for name, parameter in current.items():
            if (projected or replacement is not None) and name in encoder_names:
                count = parameter.numel()
                vector = projected_sum if replacement is None else replacement
                value = torch.from_numpy(vector[cursor:cursor+count].copy())
                parameter.grad = value.reshape_as(parameter).to(dtype=parameter.dtype)
                cursor += count
            else:
                parameter.grad = gradients[name].detach().clone()
        if (projected or replacement is not None) and cursor != len(projected_sum):
            raise ValueError('Encoder gradient layout mismatch')
        optimizer.step()
        guard()
        return candidate, optimizer

    a, opt_a = updated(False)
    b, _ = updated(True)
    names_a, names_b = dict(a.named_parameters()), dict(b.named_parameters())
    delta_a = flatten([names_a[name] - before_parameters[name] for name in encoder_names])
    delta_b = flatten([names_b[name] - before_parameters[name] for name in encoder_names])
    norm_a, norm_b = float(delta_a.norm()), float(delta_b.norm())
    if not np.isfinite([norm_a, norm_b]).all():
        raise ValueError('Nonfinite actual Adam update')
    null, _ = updated(False, replacement=reconstructed)
    null_parameters = dict(null.named_parameters())
    delta_null = flatten([null_parameters[name]-before_parameters[name] for name in encoder_names])
    null_residual = float((delta_null-delta_a).norm())
    null_norm = float(delta_null.norm())
    null_cosine = float(delta_a @ delta_null/(norm_a*null_norm)) if norm_a and null_norm else None
    if norm_a == 0:
        if null_norm != 0:
            raise ValueError('Reconstructed no-projection gradient changed a zero Adam update')
    elif null_norm == 0 or null_residual/norm_a > 1e-4 or null_cosine < 0.99999999:
        raise ValueError('Actual Adam no-projection numerical null failed')
    per_layer_null = {}
    for name in encoder_names:
        da = flatten([names_a[name]-before_parameters[name]])
        dn = flatten([null_parameters[name]-before_parameters[name]])
        base_norm = float(da.norm()); difference = float((dn-da).norm())
        if base_norm == 0 and difference != 0 or base_norm and difference/base_norm > 1e-4:
            raise ValueError('Per-parameter actual Adam no-projection null failed: '+name)
        per_layer_null[name] = {'A_norm': base_norm, 'difference_L2': difference}
    # Ordinary total.backward uses a separate fresh train-mode forward. Its
    # actual A parameters/Adam must be bitwise identical at every market unit.
    ordinary = deepcopy(net); ordinary.train()
    ordinary_opt = make_cpu_adam(ordinary, saved_optimizer)
    ordinary_opt.zero_grad(); ordinary_pred = ordinary(*update_args)
    sum((ordinary_pred[h]-update_truth[:, :h]).square().mean() for h in HORIZONS).backward()
    ordinary_opt.step()
    if state_fingerprint(ordinary.state_dict()) != state_fingerprint(a.state_dict()) or (
            state_fingerprint(ordinary_opt.state_dict()) != state_fingerprint(opt_a.state_dict())):
        raise ValueError('A does not exactly match independent ordinary total backward')
    del ordinary, ordinary_opt, ordinary_pred, null
    for name in parameters:
        if name not in encoder_names and not torch.equal(names_a[name], names_b[name]):
            raise ValueError('A and B differ in head/decoder parameters')
    for name, value in a.named_buffers():
        if not torch.equal(value, dict(b.named_buffers())[name]):
            raise ValueError('A/B BatchNorm buffers differ')

    c = None
    norm_c = cosine_ac = None
    if norm_a == 0 and norm_b > 0:
        eligibility = 'ineligible_original_update_has_no_direction'
    else:
        c = deepcopy(a)  # All non-encoder parameters exactly equal A.
        ratio = norm_b/norm_a if norm_a else 0.0
        cursor = 0
        with torch.no_grad():
            for name, parameter in c.named_parameters():
                if name in encoder_names:
                    count = parameter.numel()
                    proposed = delta_a[cursor:cursor+count].reshape_as(parameter)*ratio
                    parameter.copy_((before_parameters[name].double()+proposed).float())
                    cursor += count
        params_c = dict(c.named_parameters())
        actual_c = flatten([params_c[name] - before_parameters[name] for name in encoder_names])
        norm_c = float(actual_c.norm())
        if norm_a and norm_c:
            cosine_ac = float(delta_a @ actual_c/(norm_a*norm_c))
        if not norm_a:
            eligibility = 'degenerate_both_updates_zero'
        elif norm_b == 0:
            eligibility = 'degenerate_projected_update_zero'
        elif norm_c == 0:
            eligibility = 'ineligible_float32_control_update_zero'
        else:
            eligibility = 'eligible'
        # In float32 a matched proposed displacement need not survive rounding.
        # Refuse a direction attribution if actual norm/direction are unmatched.
        if not np.isclose(norm_c, norm_b, rtol=1e-3, atol=0) or (
                cosine_ac is not None and cosine_ac < 0.99999):
            eligibility = 'ineligible_float32_norm_or_direction_mismatch'
        for name in parameters:
            if name not in encoder_names and not torch.equal(names_a[name], params_c[name]):
                raise ValueError('C changes a head/decoder update')

    projection_update_difference = float((delta_b-delta_a).norm())
    if projection_update_difference == 0 or null_residual > 0.01*projection_update_difference:
        if eligibility == 'eligible':
            eligibility = 'ineligible_projection_effect_not_separated_from_numerical_null'

    # Baseline uses the SAME post-update-batch BN state, before any weight step.
    baseline = deepcopy(work)
    scores = {'before_with_common_postforward_BN': diagnostic_metrics(
        baseline, diagnostic_args, diagnostic_truth, guard),
        'A': diagnostic_metrics(a, diagnostic_args, diagnostic_truth, guard),
        'B': diagnostic_metrics(b, diagnostic_args, diagnostic_truth, guard)}
    if c is not None:
        scores['C'] = diagnostic_metrics(c, diagnostic_args, diagnostic_truth, guard)
    for candidate in (baseline, a, b, c):
        if candidate is not None:
            for name, value in candidate.named_buffers():
                if not torch.equal(value, post_forward_buffers[name]):
                    raise ValueError('Diagnostic eval modified common buffers')
    if any(not torch.equal(value, original[name]) for name, value in net.state_dict().items()):
        raise ValueError('Input model was mutated')
    if any(not torch.equal(value, original_buffers[name]) for name, value in net.named_buffers()):
        raise ValueError('Input model buffer was mutated')
    guard()
    return {'eligibility': eligibility, 'horizons': list(HORIZONS),
            'gradient_gram': (head_rows @ head_rows.T).tolist(),
            'gradient_norms': np.linalg.norm(head_rows, axis=1).tolist(),
            'gradient_sum_reconstruction_max_abs': float(np.max(
                np.abs(head_rows.sum(axis=0)-original_sum))),
            'gradient_reconstruction_checks': {
                'FP32_residual_L2': float(np.linalg.norm(residual)),
                'FP32_component_scale': component_scale,
                'FP32_bound': backward_error_bound,
                'FP32_per_parameter': reconstruction_layers,
                'worst_original_coordinate_check': worst_coordinate,
                'FP64_residual_L2': float(np.linalg.norm(residual64)), 'FP64_bound': bound64,
                'A_ordinary_backward_model_Adam_exact': True,
                'no_projection_null_update_difference_L2': null_residual,
                'no_projection_null_update_relative_L2': null_residual/norm_a if norm_a else 0.0,
                'no_projection_null_A_cosine': null_cosine,
                'projection_update_difference_L2': projection_update_difference,
                'no_projection_null_per_parameter': per_layer_null},
            'encoder_parameter_names': encoder_names,
            'actual_update_norms': {'A': norm_a, 'B': norm_b, 'C': norm_c},
            'actual_A_C_cosine': cosine_ac, 'metrics': scores,
            'A_B_C_nonencoder_updates_equal': True,
            'A_model_state_sha256': state_fingerprint(a.state_dict()),
            'A_optimizer_state_sha256': state_fingerprint(opt_a.state_dict()),
            'common_postforward_BN_verified': True, 'input_model_unchanged': True,
            'optimizer_execution': 'CPU float32 non-fused Adam with original moments; not GPU bitwise replay',
            'new_candidate_weights_saved': False}
