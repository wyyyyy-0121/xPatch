"""Independent synthetic fixtures; never imports a model or market loader."""
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time

DEADLINE = 1794216709.0
if time.time() >= DEADLINE:
    raise SystemExit('Review budget expired before execution')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'
os.sched_setaffinity(0, {24, 26})
import numpy as np

root = Path(__file__).resolve().parent
source = root / 'local_update_probe.py'
spec = importlib.util.spec_from_file_location('reviewed_probe', source)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
acceptance = json.loads((root / 'LOCAL_PROBE_SYNTHETIC_ACCEPTANCE.json').read_text())
assert source_hash == acceptance['source_sha256']

checks = []
def passed(name, **evidence):
    checks.append({'check': name, 'status': 'passed', **evidence})

# Hand reduction: [1,0] - (-1/2)[-1,1] = [1/2,1/2].
# The second row uses original [1,0], giving [-1,1] - (-1)[1,0] = [0,1].
g = np.array([[1., 0.], [-1., 1.]])
expected = np.array([[0.5, 0.5], [0., 1.]])
actual = probe.fixed_order_projection(g)
assert np.array_equal(actual, expected)
passed('hand_derived_two_head_original_reference_projection',
       gradients=g.tolist(), expected=expected.tolist(), actual=actual.tolist())

# This exact fixture is sensitive to ascending order and original references.
# Row 0: [2,0] -> [1,1] -> [0,0].
# Row 1: [-1,1] -> [0,1] -> [-1/2,1/2].
# Row 2: [-1,-1] -> [0,-1] -> [-1/2,-1/2].
g = np.array([[2., 0.], [-1., 1.], [-1., -1.]])
expected = np.array([[0., 0.], [-0.5, 0.5], [-0.5, -0.5]])
g_before = g.copy()
m, v = np.zeros(2), np.zeros(2)
m_before, v_before = m.copy(), v.copy()
out = probe.counterfactual_updates(g, m, v, 0)
assert np.array_equal(out['projected_gradients'], expected)
assert np.array_equal(out['A'], [0., 0.])
assert np.linalg.norm(out['B']) > 0
assert out['C'] is None
assert out['status'] == 'ineligible_original_update_has_no_direction'
assert g.tobytes() == g_before.tobytes()
assert m.tobytes() == m_before.tobytes() and v.tobytes() == v_before.tobytes()
passed('hand_derived_A_zero_B_nonzero_ineligible', gradients=g.tolist(),
       expected_projected=expected.tolist(), A=out['A'].tolist(),
       B=out['B'].tolist(), C=out['C'], eligibility=out['status'])

# Independent scalar arithmetic with nondefault betas and existing history.
# t=3; m'=[.6,-1.05], v'=[.28,1.0675]; corrections .875 and .578125.
gradient = [1., -2.]
moment = [0.2, -0.1]
variance = [0.04, 0.09]
expected_m = [0.6, -1.05]
expected_v = [0.28, 1.0675]
oracle = np.array([
    -0.2 * (mm / 0.875) / (math.sqrt(vv / 0.578125) + 0.1)
    for mm, vv in zip(expected_m, expected_v)
])
actual = probe.adam_proposal(gradient, moment, variance, 2,
                             lr=0.2, betas=(0.5, 0.75), eps=0.1)
assert np.allclose(actual, oracle, rtol=1e-14, atol=1e-16)
passed('independent_nondefault_Adam_bias_correction_with_history',
       completed_steps=2, proposal_step=3, manual_m_next=expected_m,
       manual_v_next=expected_v, bias_corrections=[0.875, 0.578125],
       expected=oracle.tolist(), actual=actual.tolist(),
       max_abs_diff=float(np.max(np.abs(oracle-actual))))

g = np.array([[2., 0., 1.], [-1., 1., -0.5], [0., 0.3, 1.]])
m = np.array([0.2, -0.1, 0.03])
v = np.array([0.04, 0.06, 0.02])
snapshots = [x.tobytes() for x in (g, m, v)]
out = probe.counterfactual_updates(g, m, v, 7)
gradient_ratio = float(np.linalg.norm(out['projected_gradients'].sum(0)) /
                       np.linalg.norm(g.sum(0)))
delta_ratio = float(out['norm_B'] / out['norm_A'])
assert not math.isclose(gradient_ratio, delta_ratio, rel_tol=1e-3)
assert np.allclose(out['C'], out['A'] * delta_ratio, rtol=1e-14, atol=1e-16)
assert math.isclose(out['norm_B'], out['norm_C'], rel_tol=1e-14)
assert np.dot(out['A'], out['C']) > 0
assert all(x.tobytes() == old for x, old in zip((g, m, v), snapshots))
passed('C_matches_actual_parameter_update_norm_and_A_direction',
       actual_update_norm_ratio=delta_ratio, gradient_norm_ratio=gradient_ratio,
       norm_A=out['norm_A'], norm_B=out['norm_B'], norm_C=out['norm_C'],
       inputs_and_moments_bytewise_unchanged=True)

# A gradient sum of zero does not mean A has no direction when momentum exists.
g = np.array([[2., 0.], [-1., 1.], [-1., -1.]])
out = probe.counterfactual_updates(g, [0.2, -0.1], [0.04, 0.06], 7)
assert out['norm_A'] > 0 and out['status'] == 'eligible'
assert out['C'] is not None
passed('zero_original_gradient_with_existing_momentum_stays_eligible',
       original_gradient_sum=g.sum(0).tolist(), A=out['A'].tolist(),
       eligibility=out['status'])

zero = probe.counterfactual_updates(np.zeros((4, 2)), np.zeros(2), np.zeros(2), 0)
assert zero['status'] == 'degenerate_both_updates_zero'
assert np.array_equal(zero['C'], [0., 0.])
passed('both_updates_zero_marked_degenerate')

assert time.time() < DEADLINE
report = {
    'status': 'passed_with_explicit_scope_limits',
    'material_defects': [],
    'source_sha256': source_hash,
    'acceptance_hash_matches_current_source': True,
    'independent_fixtures_not_self_test': True,
    'self_test_called': False,
    'checks': checks,
    'scope_limits': [
        'Pure float64 NumPy arithmetic only; no fused float32 Torch Adam equivalence established.',
        'No market data, checkpoint, model, training, inference, or scoring loaded or run.',
        'Encoder-only targeting and unchanged head updates require later runner integration checks.',
        'C is a one-step imposed displacement; no persistent optimizer state transition is implemented.',
        'No claim that final projected rows are pairwise nonnegative; intervention is order dependent.',
        'Fixtures use ordinary finite magnitudes; extreme-value overflow/underflow is not validated.'
    ],
    'nonblocking_notes': [
        'Original self-test did not cover A-zero/B-nonzero; independent hand-derived fixture covers it here.'
    ],
    'cpu_affinity': sorted(os.sched_getaffinity(0)),
    'gpu_disabled': os.environ['CUDA_VISIBLE_DEVICES'] == '-1',
    'at_beijing': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    'deadline_epoch': DEADLINE,
    'completed_before_deadline': True,
}
dest = root / 'LOCAL_PROBE_INDEPENDENT_REVIEW.json'
dest.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'path': str(dest), 'status': report['status'],
                  'checks_passed': len(checks), 'source_sha256': source_hash,
                  'material_defects': []}))
