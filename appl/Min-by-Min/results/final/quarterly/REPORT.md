# Quarterly AAPL experiment

Status: `complete`

Bars: 499 at 1mo; coverage: 41.50 years.

## Final test

Selected by validation, then evaluated on the final chronological 20%:

| Target | Model | Window | Horizon | Test RMSE | Test MASE |
|---|---|---:|---:|---:|---:|
| future_close | last_close | 24 | 3 | 9.80967 | 30.0571 |
| future_log_return | gru | 24 | 3 | 9.65976 | 29.4996 |

## Validation comparison

| Target | Model | Window | Horizon | Median MASE | Median RMSE | Baseline RMSE gain | Folds improved |
|---|---|---:|---:|---:|---:|---:|---:|
| future_log_return | gru | 24 | 3 | 3.2502 | 0.160553 | -2.02% | 20% |
| future_log_return | gru | 36 | 3 | 3.2747 | 0.162156 | -2.24% | 20% |
| future_close | last_close | 24 | 3 | 3.2758 | 0.157367 | 0.00% | 0% |
| future_log_return | last_close | 24 | 3 | 3.2758 | 0.157367 | 0.00% | 0% |
| future_log_return | xpatch | 24 | 3 | 3.2920 | 0.159641 | -1.45% | 20% |
| future_log_return | lstm | 24 | 3 | 3.3042 | 0.157587 | -0.14% | 20% |
| future_close | last_close | 36 | 3 | 3.3062 | 0.158605 | 0.00% | 0% |
| future_log_return | last_close | 36 | 3 | 3.3062 | 0.158605 | 0.00% | 0% |
| future_log_return | lstm | 36 | 3 | 3.3173 | 0.15918 | -0.36% | 20% |
| future_log_return | xpatch | 36 | 3 | 3.3208 | 0.159301 | -0.44% | 40% |
| future_log_return | drift | 24 | 3 | 3.3298 | 0.178393 | -7.89% | 0% |
| future_log_return | mlp | 24 | 3 | 3.3421 | 0.162552 | -3.30% | 0% |
| future_close | drift | 24 | 3 | 3.3515 | 0.167001 | -4.98% | 0% |
| future_log_return | drift | 36 | 3 | 3.3928 | 0.178859 | -5.05% | 0% |
| future_log_return | mlp | 36 | 3 | 3.3936 | 0.163856 | -2.60% | 0% |
| future_close | drift | 36 | 3 | 3.4551 | 0.166951 | -3.63% | 20% |
| future_log_return | gru | 6 | 3 | 3.7906 | 0.15691 | -0.63% | 40% |
| future_close | last_close | 12 | 3 | 3.8113 | 0.155246 | 0.00% | 0% |
| future_log_return | last_close | 12 | 3 | 3.8113 | 0.155246 | 0.00% | 0% |
| future_close | last_close | 6 | 3 | 3.8173 | 0.154081 | 0.00% | 0% |

Acceptance threshold met: `false`.

## Weak validation periods

- gru / future_log_return (window 24, horizon 3): fold 1 down period, 1988-02-01T00:00:00 to 1993-07-01T00:00:00.
- gru / future_log_return (window 24, horizon 3): fold 2 down period, 1993-05-01T00:00:00 to 1998-10-01T00:00:00.
- gru / future_log_return (window 24, horizon 3): fold 3 up period, 1998-08-01T00:00:00 to 2003-12-01T00:00:00.
- gru / future_log_return (window 24, horizon 3): fold 4 up period, 2003-10-01T00:00:00 to 2009-02-01T00:00:00.
- gru / future_log_return (window 36, horizon 3): fold 1 down period, 1988-12-01T00:00:00 to 1994-03-01T00:00:00.
- gru / future_log_return (window 36, horizon 3): fold 2 up period, 1994-01-01T00:00:00 to 1999-04-01T00:00:00.
- gru / future_log_return (window 36, horizon 3): fold 3 up period, 1999-02-01T00:00:00 to 2004-05-01T00:00:00.
- gru / future_log_return (window 36, horizon 3): fold 4 up period, 2004-03-01T00:00:00 to 2009-06-01T00:00:00.
- xpatch / future_log_return (window 24, horizon 3): fold 1 down period, 1988-02-01T00:00:00 to 1993-07-01T00:00:00.
- xpatch / future_log_return (window 24, horizon 3): fold 2 down period, 1993-05-01T00:00:00 to 1998-10-01T00:00:00.
- xpatch / future_log_return (window 24, horizon 3): fold 3 up period, 1998-08-01T00:00:00 to 2003-12-01T00:00:00.
- xpatch / future_log_return (window 24, horizon 3): fold 4 up period, 2003-10-01T00:00:00 to 2009-02-01T00:00:00.
- 250 additional underperforming folds are listed in `report.json`.

No tested configuration met all acceptance criteria; the validation results do not support a stable predictive advantage.

## Data limitations

- Input source is recorded in a manifest but has not been independently verified.

Per-fold metrics, per-step errors, data manifest, and the complete experiment configuration are in `report.json`.
Final test scores are reported separately and do not affect selection.
