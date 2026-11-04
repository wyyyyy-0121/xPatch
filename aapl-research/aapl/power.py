"""Pre-lock power planning from fixed training-period pseudo-residuals only."""
import numpy as np

from . import baselines
from .common import ROOT, digest, utc_now, write_json, runtime
from .data import Market, sessions


def main():
    runtime()
    market = Market.load()
    days = sorted(set(market.dates))
    train_dates = [d for d in days if d <= '2021-12-30']
    validation_dates = [d for d in days if '2022-01-03' <= d <= '2022-12-30']
    train, val = market.select(train_dates), market.select(validation_dates)
    scale = market.scale(train_dates)
    config = {'model': 'ridge', 'lookback': 96, 'alpha': 1.0, 'seed': 2026}
    model = baselines.fit_ridge(market, train, config, scale)
    prediction = baselines.predict(model, market, val, config, scale)
    y = market.y(val)
    delta = (y - prediction) ** 2 - y ** 2
    dates = market.dates[val]
    unique, inv = np.unique(dates, return_inverse=True)
    sums, count = np.bincount(inv, weights=delta), np.bincount(inv)
    center = delta.mean()
    # A 240-return context gives no origins on early-close days. Use calendar geometry only.
    holdout_days = sum(len(grid) >= 253 for grid in sessions('2026-07-01', '2026-09-30').values())
    rng = np.random.default_rng(20261104)
    rows = []
    for block in (1, 5, 10):
        draws = (rng.integers(len(unique), size=(5000, int(np.ceil(holdout_days / block))))[..., None] + np.arange(block)) % len(unique)
        draws = draws.reshape(5000, -1)[:, :holdout_days]
        means = sums[draws].sum(axis=1) / count[draws].sum(axis=1) - center
        # Worst first Holm test for two candidates: alpha=.025. Find shift giving 80% power.
        mde = max(0.0, float(np.quantile(means, 0.975) - np.quantile(means, 0.2))) / float(np.mean(y ** 2))
        rows.append({'block_days': block, 'minimum_detectable_relative_mse_gain_proxy': mde,
                     'sample_days': holdout_days, 'one_sided_alpha': 0.025, 'target_power': 0.8})
    main = next(r for r in rows if r['block_days'] == 5)
    record = {'completed_at_utc': utc_now(), 'plan_sha256': digest(ROOT / 'IMPLEMENTATION_PLAN.md'),
              'configuration_fixed_before_computing_residuals': config, 'source_fit_end': train_dates[-1],
              'residual_period': [validation_dates[0], validation_dates[-1]], 'residual_days': len(unique),
              'holdout_effective_days_calendar_only': holdout_days, 'sensitivity': rows,
              'primary_mde_proxy': main['minimum_detectable_relative_mse_gain_proxy'],
              'exploratory_only_if_proxy_exceeds_3pct': main['minimum_detectable_relative_mse_gain_proxy'] > 0.03,
              'holdout_outcomes_read': False,
              'limitation': 'Prospective planning proxy from one fixed Ridge residual series; actual power depends on the selected model loss-difference variance and cannot be guaranteed by this estimate'}
    write_json(ROOT / 'outputs/power_planning.json', record)
    print(record, flush=True)


if __name__ == '__main__':
    main()
