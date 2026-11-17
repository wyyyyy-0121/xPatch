"""Pure numerical core for a future local, fit-only gradient intervention.

No market data loader, model, training runner, or checkpoint writer is present.
Rows must be encoder gradients in predeclared horizon order. This diagnostic
uses float64 arithmetic; it is not a replay of fused float32 Torch Adam.
"""
import numpy as np


def adam_proposal(gradient, first_moment, second_moment, completed_steps,
                  lr=0.001, betas=(0.9, 0.999), eps=1e-8):
    g, m, v = [np.asarray(x, dtype=np.float64) for x in
               (gradient, first_moment, second_moment)]
    if g.ndim != 1 or g.shape != m.shape or g.shape != v.shape:
        raise ValueError('Require matching flat encoder vectors')
    if not all(np.isfinite(x).all() for x in (g, m, v)) or (v < 0).any():
        raise ValueError('Require finite gradients/moments and nonnegative variance')
    if not isinstance(completed_steps, int) or completed_steps < 0:
        raise ValueError('completed_steps must be a nonnegative integer')
    b1, b2 = betas
    if not (0 <= b1 < 1 and 0 <= b2 < 1 and lr > 0 and eps > 0):
        raise ValueError('Invalid Adam hyperparameters')
    t = completed_steps + 1
    m_next = b1*m + (1-b1)*g
    v_next = b2*v + (1-b2)*g*g
    delta = -lr*(m_next/(1-b1**t))/(np.sqrt(v_next/(1-b2**t))+eps)
    if not np.isfinite(delta).all():
        raise ValueError('Nonfinite update')
    return delta


def fixed_order_projection(gradients):
    """Project each row against original other rows, in ascending row order.

    This is one explicitly specified, order-dependent intervention. It does
    not guarantee that every pair of final projected rows has positive dot.
    """
    g = np.asarray(gradients, dtype=np.float64)
    if g.ndim != 2 or not len(g) or not g.shape[1] or not np.isfinite(g).all():
        raise ValueError('Require a finite nonempty head by parameter matrix')
    changed = g.copy()
    for i in range(len(g)):
        for j in range(len(g)):
            if i == j:
                continue
            norm_squared = float(g[j] @ g[j])
            dot = float(changed[i] @ g[j])
            if dot < 0 and norm_squared > 0:
                changed[i] -= (dot/norm_squared)*g[j]
    return changed


def counterfactual_updates(gradients, first_moment, second_moment, completed_steps):
    """Return original A, projected B and update-norm-matched direction control C.

    Moments are read without mutation. A and B use the same original history.
    C is an imposed parameter displacement, not a persistent Adam optimizer.
    If A has no direction and B moves, C is undefined and the unit is ineligible.
    """
    g = np.asarray(gradients, dtype=np.float64)
    projected = fixed_order_projection(g)
    a = adam_proposal(g.sum(axis=0), first_moment, second_moment, completed_steps)
    b = adam_proposal(projected.sum(axis=0), first_moment, second_moment,
                      completed_steps)
    norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if norm_a == 0 and norm_b > 0:
        c, status = None, 'ineligible_original_update_has_no_direction'
    elif norm_a == 0:
        c, status = a.copy(), 'degenerate_both_updates_zero'
    else:
        c, status = a*(norm_b/norm_a), 'eligible'
    return {'A': a, 'B': b, 'C': c, 'status': status,
            'original_gram': g @ g.T,
            'projected_gradients': projected,
            'norm_A': norm_a, 'norm_B': norm_b,
            'norm_C': None if c is None else float(np.linalg.norm(c))}


def synthetic_acceptance():
    # Positive aligned task gradients: intervention and direction control are null.
    m = np.array([0.2, -0.1, 0.03]); v = np.array([0.04, 0.06, 0.02])
    aligned = np.array([[1., 2., 3.], [2., 4., 6.], [3., 6., 9.]])
    m_before, v_before = m.copy(), v.copy()
    same = counterfactual_updates(aligned, m, v, 7)
    assert np.array_equal(same['A'], same['B'])
    assert np.allclose(same['A'], same['C'], rtol=1e-14, atol=1e-16)
    assert np.array_equal(m, m_before) and np.array_equal(v, v_before)

    # Conflicting heads with anisotropic adaptive history: norm must be matched
    # after Adam, preserving A's direction rather than just its gradient norm.
    conflict = np.array([[2., 0., 1.], [-1., 1., -0.5], [0., 0.3, 1.]])
    different = counterfactual_updates(conflict, m, v, 7)
    assert different['status'] == 'eligible'
    assert not np.allclose(different['A'], different['B'], rtol=1e-8, atol=1e-14)
    assert np.isclose(different['norm_B'], different['norm_C'], rtol=1e-14)
    assert np.allclose(different['C'], different['A']*
                       (different['norm_B']/different['norm_A']), rtol=1e-14)

    # Zero rows must not divide by zero. Zero first-step gradients are uninformative.
    zero = counterfactual_updates(np.zeros((4, 3)), np.zeros(3), np.zeros(3), 0)
    assert zero['status'] == 'degenerate_both_updates_zero'
    assert np.array_equal(zero['A'], np.zeros(3))
    assert np.isfinite(fixed_order_projection(np.vstack([conflict, [0., 0., 0.]]))).all()

    # A positive global scaling is not a fourfold Adam learning rate. Exact
    # algebra uses scaled moments AND epsilon/c when comparing histories.
    g = np.array([1e-9, -0.5, 2.]); c = 0.25
    scaled = adam_proposal(c*g, c*m, c*c*v, 7)
    equivalent = adam_proposal(g, m, v, 7, eps=1e-8/c)
    assert np.allclose(scaled, equivalent, rtol=1e-14, atol=1e-16)
    for invalid in [np.array([[float('nan'), 0., 0.]]), np.empty((0, 3))]:
        try:
            fixed_order_projection(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid gradient fixture was admitted')
    return {'status': 'passed', 'synthetic_only': True,
            'market_data_or_checkpoint_loaded': False,
            'training_or_model_inference': False,
            'checks': ['aligned_null', 'same_moments_not_mutated',
                       'conflict_direction_vs_actual_update_norm',
                       'zero_norm_and_zero_head', 'Adam_scale_epsilon_identity',
                       'reject_nonfinite_or_empty'],
            'float64_numerical_core_not_fused_float32_replay': True}


if __name__ == '__main__':
    import json
    print(json.dumps(synthetic_acceptance()))
