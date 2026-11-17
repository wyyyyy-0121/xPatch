"""Fixed CLV screen arithmetic. No data I/O or execution at import."""
import numpy as np

PENALTY = 0.1
STD_FLOOR = 1e-12
SPAN_EIGEN_REL = 1e-10
SPAN_RESIDUAL_MIN = 1e-8


def empty_stats(columns, horizon):
    return dict(n=0, sx=np.zeros(columns), sy=np.zeros(horizon),
                xx=np.zeros((columns, columns)), xy=np.zeros((columns, horizon)))


def update(stats, x, y):
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or not len(x):
        raise ValueError('Need aligned nonempty matrices')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite screen matrix')
    stats['n'] += len(x)
    stats['sx'] += x.sum(axis=0)
    stats['sy'] += y.sum(axis=0)
    stats['xx'] += x.T @ x
    stats['xy'] += x.T @ y


def moments(stats):
    n = int(stats['n'])
    if n < 2:
        raise ValueError('Insufficient prefix samples')
    mx, my = stats['sx']/n, stats['sy']/n
    cov = stats['xx']/n-np.outer(mx, mx)
    magnitude = np.maximum(np.maximum(np.abs(np.diag(stats['xx'])/n), mx*mx), 1.)
    if (np.diag(cov) < -1e-10*magnitude).any():
        raise ValueError('Negative prefix variance beyond fixed numerical margin')
    # This is the old fixed per-column standardization, including constants.
    sd = np.sqrt(np.maximum(np.diag(cov), 0))
    sd = np.where(sd > STD_FLOOR, sd, 1.)
    gram = cov/sd[:, None]/sd[None, :]
    cross = (stats['xy']/n-np.outer(mx, my))/sd[:, None]
    if not all(np.isfinite(v).all() for v in (mx, my, sd, gram, cross)):
        raise ValueError('Nonfinite sufficient-stat moments')
    return mx, my, sd, (gram+gram.T)/2, cross


def solve(stats, columns):
    mx, my, sd, gram, cross = moments(stats)
    matrix = gram[:columns, :columns]+PENALTY*np.eye(columns)
    right = cross[:columns]
    coef = np.linalg.solve(matrix, right)
    if not np.isfinite(coef).all():
        raise ValueError('Nonfinite fixed Ridge coefficients')
    denominator = np.linalg.norm(matrix, np.inf)*np.linalg.norm(coef, np.inf)+np.linalg.norm(right, np.inf)
    residual = np.linalg.norm(matrix@coef-right, np.inf)/max(float(denominator), np.finfo(float).tiny)
    if not np.isfinite(residual) or residual > 1e-10:
        raise ValueError('Fixed Ridge normwise stationarity gate failed')
    return dict(coef=coef, x_mean=mx[:columns], x_std=sd[:columns], y_mean=my)


def predict(model, x):
    p = len(model['x_mean'])
    return model['y_mean']+(x[:, :p]-model['x_mean'])/model['x_std']@model['coef']


def clv_prefix_qualification(stats, x_batches):
    """CLV residual against retained B eigenspace; no label moment or tail.

    The residual is evaluated directly on the prefix, avoiding subtraction of
    nearly equal projected variances. x_batches supplies only prefix features.
    """
    n = int(stats['n'])
    mx = stats['sx']/n
    cov = stats['xx']/n-np.outer(mx, mx)
    v = float(cov[-1, -1])
    if not np.isfinite(cov).all() or v <= STD_FLOOR**2:
        return dict(passed=False, reason='constant_or_nonfinite_CLV', clv_variance=v)
    sd = np.sqrt(np.maximum(np.diag(cov), 0))
    sd = np.where(sd > STD_FLOOR, sd, 1.)
    gram = cov/sd[:, None]/sd[None, :]
    b = (gram[:-1, :-1]+gram[:-1, :-1].T)/2
    eigen, vectors = np.linalg.eigh(b)
    maximum = float(eigen[-1])
    minimum = float(eigen[0])
    if maximum <= 0 or minimum < -1e-8*maximum:
        raise ValueError('Prefix Gram violates the fixed numerical PSD gate')
    keep = eigen > SPAN_EIGEN_REL*maximum
    q = gram[:-1, -1]
    projection = vectors[:, keep].T@q
    coef = vectors[:, keep]@(projection/eigen[keep])
    rss = denominator = 0.
    seen = 0
    for x in x_batches:
        if x.ndim != 2 or x.shape[1] != len(mx) or not np.isfinite(x).all():
            raise ValueError('Invalid prefix-only residual features')
        z = (x-mx)/sd
        residual = z[:, -1]-z[:, :-1]@coef
        rss += float(residual@residual)
        denominator += float(z[:, -1]@z[:, -1])
        seen += len(x)
    if seen != n or denominator <= 0:
        raise ValueError('Residual evaluation must cover the same complete prefix')
    residual_ratio = rss/denominator
    return dict(passed=bool(np.isfinite(residual_ratio) and maximum > 0 and
                            residual_ratio > SPAN_RESIDUAL_MIN),
                clv_variance=v, residual_ratio=residual_ratio,
                retained_rank=int(keep.sum()), baseline_columns=len(q),
                minimum_B_eigenvalue=minimum, maximum_B_eigenvalue=maximum,
                eigen_relative_cutoff=SPAN_EIGEN_REL, negative_eigen_relative_limit=1e-8,
                minimum_residual_ratio=SPAN_RESIDUAL_MIN,
                interpretation='finite_prefix_residual_vs_retained_B_eigenspace_only')


def partition_fit(fit, end_ns, horizon, available):
    boundary = int(np.floor(.8*len(fit)))
    first = int(fit[boundary])
    prefix, tail = fit[:boundary], fit[boundary:]
    prefix = prefix[end_ns[prefix+horizon] < end_ns[first]]
    return boundary, first, prefix[available(prefix)], tail[available(tail)]


def state_checks(metrics):
    c = metrics['C']
    result = {f'C_endpoint_strictly_below_{a}':bool(c['endpoint_mse'] < metrics[a]['endpoint_mse'])
              for a in ('A', 'B', 'zero', 'prefix_mean')}
    result.update({f'C_path_not_above_{a}':bool(c['path_mse'] <= metrics[a]['path_mse'])
                   for a in ('A', 'B')})
    return result


def synthetic_acceptance():
    """Direct centered solve and serialized sufficient-stat continuation."""
    import tempfile
    from pathlib import Path
    rng = np.random.default_rng(20261110)
    x = rng.normal(size=(560, 501))
    x[:, 0] = 0
    x[:, 1] = x[:, 2]  # Predeclared dictionaries may contain duplicate columns.
    y = rng.normal(size=(560, 60))
    stats = empty_stats(x.shape[1], y.shape[1])
    update(stats, x[:177], y[:177])
    with tempfile.TemporaryDirectory(prefix='aapl-clv-stats-') as d:
        file = Path(d)/'checkpoint.npz'
        np.savez(file, **stats, cursor=177, identity='synthetic-only')
        with np.load(file, allow_pickle=False) as f:
            restored = {k:f[k].copy() for k in ('n','sx','sy','xx','xy')}
            assert int(f['cursor']) == 177 and str(f['identity']) == 'synthetic-only'
        restored['n'] = int(restored['n'])
    update(restored, x[177:], y[177:])
    errors = []
    for columns in (498, 500, 501):
        model = solve(restored, columns)
        mx = x[:, :columns].mean(axis=0)
        sd = x[:, :columns].std(axis=0)
        sd = np.where(sd > STD_FLOOR, sd, 1.)
        z = (x[:, :columns]-mx)/sd
        direct = np.linalg.solve(z.T@z/len(z)+PENALTY*np.eye(columns),
                                 z.T@(y-y.mean(axis=0))/len(z))
        np.testing.assert_allclose(model['coef'], direct, rtol=1e-9, atol=1e-11)
        errors.append(float(np.max(np.abs(model['coef']-direct))))
    assert clv_prefix_qualification(restored, [x])['passed']
    redundant = x.copy(); redundant[:, -1] = redundant[:, 2]
    duplicate = empty_stats(x.shape[1], y.shape[1]); update(duplicate, redundant, y)
    assert not clv_prefix_qualification(duplicate, [redundant])['passed']
    constant = x.copy(); constant[:, -1] = .5
    duplicate = empty_stats(x.shape[1], y.shape[1]); update(duplicate, constant, y)
    assert not clv_prefix_qualification(duplicate, [constant])['passed']
    almost_constant = x.copy(); almost_constant[:, -1] = 1e-14*x[:, -1]
    duplicate = empty_stats(x.shape[1], y.shape[1]); update(duplicate, almost_constant, y)
    assert not clv_prefix_qualification(duplicate, [almost_constant])['passed']
    stamps = np.arange(200, dtype=np.int64)*60_000_000_000
    origins = np.arange(20, 120, dtype=np.int64)
    cut, first, prefix, tail = partition_fit(origins, stamps, 8, lambda o:o%3 != 0)
    assert cut == 80 and first == 100
    assert (stamps[prefix+8] < stamps[first]).all() and (tail >= first).all()
    assert (prefix%3 != 0).all() and (tail%3 != 0).all()
    return dict(status='passed', direct_centered_max_coef_error=max(errors),
                maximum_shape=[560, 501, 60],
                saved_stat_resume=True, duplicate_and_constant_tests=True,
                boundary_before_mask_and_strict_label_purge=True, market_data_used=False)
