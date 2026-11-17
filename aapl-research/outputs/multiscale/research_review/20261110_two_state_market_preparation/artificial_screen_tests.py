"""Artificial numerical controls and frozen screen semantics, no market reads."""
from __future__ import annotations
from datetime import date, timedelta
import json
import os
import resource
import time
import numpy as np
from fixed_controls import load_core
from screen_evaluation import (MODEL_NAMES, SEEDS, origin_losses, day_groups,
                               aggregate_primary, paired_intervals)


def check(value, name):
    if not bool(value):
        raise AssertionError(name)


def main():
    check(sorted(os.sched_getaffinity(0)) == [24], 'CPU24')
    check(all(os.environ.get(k) == '1' for k in
              ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')),
          'Numerical threads1')
    check(os.environ.get('CUDA_VISIBLE_DEVICES') == '-1', 'GPU hidden')
    checks = []; states = []; days = []; manual_zero = []; manual_candidate = []
    d = date(2020, 1, 2)
    while len(days) < 20:
        if d.weekday() < 5:
            days.append(d.isoformat())
        d += timedelta(days=1)
    sessions = np.asarray(days, dtype='<U10')
    for fold, origins in ((1, np.arange(0, 20, 2, dtype=np.int64)),
                          (2, np.arange(0, 20, dtype=np.int64))):
        truth = (.01*(fold+np.arange(len(origins), dtype=np.float64)))[:, None]*np.arange(1., 9.)[None, :]
        raw = {n: 2*truth for n in MODEL_NAMES}; alphas = {n: 1. for n in MODEL_NAMES}
        scales = (.4, .5, .6) if fold == 1 else (.5, .6, .7)
        for seed, scale in zip(SEEDS, scales):
            raw[f'R_seed{seed}'] = scale*truth
        losses = origin_losses(truth, raw, alphas)
        multiplier = sum((1-scale)**2 for scale in scales)/3
        manual_zero.append(sum((.08*(fold+k))**2 for k in range(len(origins))))
        manual_candidate.append(multiplier*manual_zero[-1])
        check(np.allclose(losses['R_seedmean_shrink']['endpoint'], multiplier*truth[:, -1]**2,
                          rtol=1e-13, atol=1e-15), 'Seedmean is mean losses')
        check(np.all(losses['R_seedmean_shrink']['endpoint'] >
                     ((raw['R_seed2026']+raw['R_seed2027']+raw['R_seed2028'])/3-truth)[:, -1]**2),
              'Prediction ensemble excluded')
        groups = day_groups(sessions, origins, losses)
        check(len(groups['counts']) == int(origins[-1])+1, 'Uncompressed original session grid')
        if fold == 1:
            check(np.count_nonzero(groups['counts'] == 0) == 9, 'Nine empty-origin dates retained')
        states.append({'task': 'hourly_h8_future_log_return', 'fold': fold,
                       'state_uid': 'artificial_'+str(fold), 'session_date': sessions,
                       'origins': origins, 'losses': losses})
    result = aggregate_primary(states)
    check(result['passed'] and len(result['endpoint_intervals']) == 26, 'Known strict positive conjunction')
    pooled_gain = 100*(1-sum(manual_candidate)/sum(manual_zero))
    equal_fold_gain = sum(100*(1-c/z) for c, z in zip(manual_candidate, manual_zero))/2
    check(abs(result['gain_overzero_pct']-pooled_gain) < 1e-11, 'Pooled gain manual value')
    check(abs(result['gain_overzero_pct']-equal_fold_gain) > 1., 'Equal-fold ratio cannot pass pooled oracle')
    check(result['T_origin_occurrences'] == 30, 'Unequal fold counts pooled')
    checks.append('manual_known_gain_unequal_folds_all13_controls_two_blocks_and_seedloss_mean')
    # A single seed exactly equal to zero-return must fail the fixed seed gate.
    failure = [{**s, 'losses': {n: {m: v.copy() for m, v in row.items()}
                             for n, row in s['losses'].items()}} for s in states]
    for s in failure:
        s['losses']['R_seed2027_shrink'] = {m: v.copy() for m, v in s['losses']['zero'].items()}
    check(not aggregate_primary(failure)['gates']['R_seed2027_shrink_overzero'], 'Strict seed equality fails')
    checks.append('one_seed_tie_cannot_be_rescued_by_seedmean')
    # Identical constant endpoint differences yield a degenerate, ineligible CI.
    groups = []
    for s in states:
        constant = {n: {'endpoint': np.ones(len(s['origins'])), 'path': np.ones(len(s['origins']))}
                    for n in s['losses']}
        groups.append(day_groups(sessions, s['origins'], constant))
    rows = paired_intervals(groups, 2)
    check(all(r['status'] == 'degenerate' and not r['supports_improvement'] for r in rows),
          'Degenerate intervals fail')
    groups[0]['sums'][:] = 0.; groups[1]['sums'][:] = 0.
    rows = paired_intervals(groups, 2)
    check(all(r['relative_ci95_pct'] is None and r['relative_improvement_pct'] is None
              and not r['supports_improvement'] for r in rows), 'Zero denominators NA, no epsilon')
    checks.append('degenerate_absolute_CI_and_zero_relative_denominators_fail_closed')
    # A full-width singular augmented dictionary tests the actual 1938-column solver.
    core = load_core(); rng = np.random.Generator(np.random.PCG64(314159))
    x = rng.normal(size=(512, 1938)).astype(np.float64)
    x[:, 900] = 1.; x[:, 901] = x[:, 899]  # Constant and duplicated columns.
    y = (.002*x[:, :1]+.001*rng.normal(size=(512, 60))).astype(np.float64)
    stats = core.empty_stats(1938, 60); batch_seconds = []
    for start in range(0, len(x), 128):
        began = time.perf_counter(); core.update(stats, x[start:start+128], y[start:start+128])
        batch_seconds.append(time.perf_counter()-began)
    began = time.perf_counter(); model = core.solve(stats); solve_seconds = time.perf_counter()-began
    # Independently form centered X rather than the raw second-moment covariance.
    mx = x.mean(axis=0); my = y.mean(axis=0); xc = x-mx
    sd = np.sqrt(np.mean(xc*xc, axis=0)); sd = np.where(sd > 1e-12, sd, 1.)
    z = xc/sd; matrix = z.T@z/len(x)+.1*np.eye(x.shape[1]); right = z.T@(y-my)/len(x)
    independent = np.linalg.solve(matrix, right)
    check(np.allclose(model['coef'], independent, rtol=1e-8, atol=1e-10), 'Direct-centered coefficient reconstruction')
    check(float(model['stationarity']) <= 1e-10, 'Fixed stationarity threshold')
    maximum_difference = float(np.max(np.abs(model['coef']-independent)))
    check(np.allclose(core.predict(model, x[:17]), my+((x[:17]-mx)/sd)@independent,
                      rtol=1e-8, atol=1e-10), 'Independent raw path reconstruction')
    checks.append('max1938col_fixedRidge_vs_independent_centered_singular_dictionary_60path')
    print(json.dumps({'status': 'artificial_screen_numerics_complete', 'checks': checks,
                      'maximum_Ridge_update128_seconds': max(batch_seconds),
                      'Ridge_solve1938x60_seconds': solve_seconds,
                      'maxabs_Ridge_coefficient_difference': maximum_difference,
                      'Ridge_stationarity': float(model['stationarity']),
                      'ru_maxrss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      'market_values_read': False, 'GPU': False, 'research_goal_achieved': False}, allow_nan=False))


if __name__ == '__main__':
    main()
