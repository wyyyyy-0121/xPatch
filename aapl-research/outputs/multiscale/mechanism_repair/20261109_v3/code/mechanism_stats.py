"""Paired day-block summaries; folds remain distinct chronological strata."""
from study import np


def day_groups(bars, origins, losses):
    dates = bars.session_date[origins]
    grid = np.unique(bars.session_date[(bars.session_date >= dates[0]) & (bars.session_date <= dates[-1])])
    positions = np.searchsorted(grid, dates); names = list(losses)
    return {'names': names, 'counts': np.bincount(positions, minlength=len(grid)),
            'sums': np.column_stack([np.bincount(positions, weights=losses[n], minlength=len(grid)) for n in names])}


def block_samples(groups, block, repetitions=5000):
    names = groups[0]['names']; rng = np.random.default_rng(2026)
    if any(len(g['counts']) <= block for g in groups):
        return None
    samples = np.zeros((repetitions, len(names))); counts = np.zeros(repetitions)
    total = np.zeros(len(names)); n_origins = 0
    for group in groups:
        assert group['names'] == names
        n = len(group['counts']); total += group['sums'].sum(0); n_origins += group['counts'].sum()
        for start in range(0, repetitions, 100):
            r = min(100, repetitions-start)
            begins = rng.integers(0, n-block+1, size=(r, (n+block-1)//block))
            indices = (begins[:, :, None]+np.arange(block)).reshape(r, -1)[:, :n]
            samples[start:start+r] += group['sums'][indices].sum(1)
            counts[start:start+r] += group['counts'][indices].sum(1)
    assert (counts > 0).all()
    return total/n_origins, samples/counts[:, None]


def paired_intervals(groups, pairs, block, task):
    sampled = block_samples(groups, block)
    if sampled is None:
        return [{'task': task, 'primary': a, 'reference': b, 'block': block,
                 'status': 'insufficient_groups', 'supports_improvement': False} for a, b in pairs]
    point, samples = sampled; names = groups[0]['names']; rows = []
    for primary, reference in pairs:
        i, j = names.index(primary), names.index(reference)
        delta = samples[:, j]-samples[:, i]
        varying = bool(np.any(delta != delta[0]))
        ci = np.quantile(delta, [.025, .975]).tolist() if varying else None
        relative = 100*(1-samples[:, i]/samples[:, j])
        rows.append({'task': task, 'primary': primary, 'reference': reference, 'block': block,
                     'repetitions': 5000, 'status': 'descriptive' if varying else 'degenerate',
                     'mse_reference_minus_primary': float(point[j]-point[i]), 'difference_ci95': ci,
                     'relative_improvement_pct': float(100*(1-point[i]/point[j])),
                     'relative_ci95_pct': np.quantile(relative, [.025, .975]).tolist() if varying else None,
                     'supports_improvement': bool(ci and ci[0] > 0),
                     'scope': 'exposed development, nominal intervals, not independent or multiplicity-adjusted'})
    return rows


def interaction(groups, weights, block, task, mode):
    converted = []
    for group in groups:
        value = sum(w*group['sums'][:, group['names'].index(name)] for name, w in weights.items())
        converted.append({'names': ['interaction'], 'counts': group['counts'], 'sums': value[:, None]})
    sampled = block_samples(converted, block)
    if sampled is None:
        return {'task': task, 'block': block, 'mode': mode, 'status': 'insufficient_groups'}
    point, values = sampled; varying = bool(np.any(values[:, 0] != values[0, 0]))
    ci = np.quantile(values[:, 0], [.025, .975]).tolist() if varying else None
    return {'task': task, 'block': block, 'mode': mode, 'weights': weights,
            'mean_loss_interaction': float(point[0]), 'difference_ci95': ci,
            'joint_gain_beyond_additivity_supported': bool(ci and ci[1] < 0),
            'status': 'descriptive' if varying else 'degenerate',
            'definition': 'loss(both factors)-loss(factor one)-loss(factor two)+loss(neither)',
            'scope': 'per-origin average seed loss, exposed historical development'}
