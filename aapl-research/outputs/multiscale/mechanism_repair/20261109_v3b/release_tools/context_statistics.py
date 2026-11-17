"""Portable paired trading-day block arithmetic; NumPy only, no market-data I/O."""
import numpy as np


def close(actual, expected, label):
    assert np.allclose(actual, expected, rtol=1e-11, atol=1e-12, equal_nan=False), (label, actual, expected)


def grouped_days(session_dates, day_grid, losses):
    dates = np.asarray(session_dates)
    grid = np.asarray(day_grid)
    assert dates.ndim == grid.ndim == 1 and len(dates) and len(grid)
    assert np.all(grid[1:] > grid[:-1])
    assert np.all(dates[1:] >= dates[:-1])
    positions = np.searchsorted(grid, dates)
    assert np.all(positions < len(grid)) and np.array_equal(grid[positions], dates)
    names = list(losses)
    assert all(np.asarray(value).shape == dates.shape and np.isfinite(value).all() for value in losses.values())
    return {'names': names, 'counts': np.bincount(positions, minlength=len(grid)),
        'sums': np.column_stack([np.bincount(positions, weights=losses[name], minlength=len(grid)) for name in names])}


def block_samples(groups, block, repetitions=5000):
    names = groups[0]['names']; rng = np.random.default_rng(2026)
    if any(len(group['counts']) <= block for group in groups):
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
        return [{'task': task, 'primary': primary, 'reference': reference, 'block': block,
            'status': 'insufficient_groups', 'supports_improvement': False} for primary, reference in pairs]
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


def compare_intervals(computed, expected):
    assert len(computed) == len(expected)
    for actual, desired in zip(computed, expected):
        for field in ('task', 'primary', 'reference', 'block', 'status', 'supports_improvement'):
            assert actual[field] == desired[field], (field, actual, desired)
        for field in ('mse_reference_minus_primary', 'relative_improvement_pct', 'difference_ci95', 'relative_ci95_pct'):
            if field in desired:
                if desired[field] is None:
                    assert actual[field] is None
                else:
                    close(actual[field], desired[field], field)
