# Quarterly AAPL experiment

Status: `complete`

Bars: 499 at 1mo; coverage: 41.50 years.

## Final test

Selected by validation, then evaluated on the final chronological 20%:

| Target | Model | Window | Horizon | Test RMSE | Test MASE |
|---|---|---:|---:|---:|---:|
| future_close | last_close | 36 | 3 | 9.71632 | 29.6089 |
| future_log_return | gru | 24 | 3 | 9.56251 | 29.8263 |

## Validation comparison

| Target | Model | Window | Horizon | Median MASE | Median RMSE | Baseline RMSE gain | Folds improved |
|---|---|---:|---:|---:|---:|---:|---:|
| future_log_return | gru | 24 | 3 | 3.2840 | 0.162995 | -1.74% | 20% |
| future_log_return | gru | 36 | 3 | 3.2953 | 0.163395 | -1.96% | 20% |
| future_close | last_close | 36 | 3 | 3.3146 | 0.159701 | 0.00% | 0% |
| future_log_return | last_close | 36 | 3 | 3.3146 | 0.159701 | 0.00% | 0% |
| future_close | last_close | 24 | 3 | 3.3155 | 0.159701 | 0.00% | 0% |
| future_log_return | last_close | 24 | 3 | 3.3155 | 0.159701 | 0.00% | 0% |
| future_log_return | xpatch | 36 | 3 | 3.3302 | 0.159636 | 0.04% | 60% |
| future_log_return | lstm | 36 | 3 | 3.3305 | 0.160375 | -0.42% | 20% |
| future_log_return | lstm | 24 | 3 | 3.3337 | 0.160073 | -0.26% | 20% |
| future_log_return | xpatch | 24 | 3 | 3.3349 | 0.161508 | -0.98% | 20% |
| future_log_return | drift | 24 | 3 | 3.3735 | 0.181049 | -9.04% | 0% |
| future_log_return | mlp | 24 | 3 | 3.3745 | 0.165149 | -3.41% | 0% |
| future_close | drift | 24 | 3 | 3.3999 | 0.169522 | -4.68% | 0% |
| future_log_return | drift | 36 | 3 | 3.4062 | 0.180191 | -5.60% | 0% |
| future_log_return | mlp | 36 | 3 | 3.4077 | 0.164465 | -2.37% | 0% |
| future_log_return | mlp | 12 | 3 | 3.4110 | 0.156711 | -5.09% | 20% |
| future_log_return | gru | 12 | 3 | 3.4124 | 0.163994 | -2.56% | 20% |
| future_close | last_close | 12 | 3 | 3.4371 | 0.159701 | 0.00% | 0% |
| future_log_return | last_close | 12 | 3 | 3.4371 | 0.159701 | 0.00% | 0% |
| future_log_return | gru | 6 | 3 | 3.4377 | 0.163823 | -1.83% | 20% |

Acceptance threshold met: `false`.

## Weak validation periods

- gru / future_log_return (window 24, horizon 3): fold 1 down period, 1988-11-01T00:00:00 to 1994-02-01T00:00:00.
- gru / future_log_return (window 24, horizon 3): fold 2 up period, 1993-12-01T00:00:00 to 1999-03-01T00:00:00.
- gru / future_log_return (window 24, horizon 3): fold 3 up period, 1999-01-01T00:00:00 to 2004-03-01T00:00:00.
- gru / future_log_return (window 24, horizon 3): fold 4 up period, 2004-01-01T00:00:00 to 2009-04-01T00:00:00.
- gru / future_log_return (window 36, horizon 3): fold 1 down period, 1988-11-01T00:00:00 to 1994-02-01T00:00:00.
- gru / future_log_return (window 36, horizon 3): fold 2 up period, 1993-12-01T00:00:00 to 1999-03-01T00:00:00.
- gru / future_log_return (window 36, horizon 3): fold 3 up period, 1999-01-01T00:00:00 to 2004-03-01T00:00:00.
- gru / future_log_return (window 36, horizon 3): fold 4 up period, 2004-01-01T00:00:00 to 2009-04-01T00:00:00.
- xpatch / future_log_return (window 36, horizon 3): fold 1 down period, 1988-11-01T00:00:00 to 1994-02-01T00:00:00.
- xpatch / future_log_return (window 36, horizon 3): fold 2 up period, 1993-12-01T00:00:00 to 1999-03-01T00:00:00.
- lstm / future_log_return (window 36, horizon 3): fold 1 down period, 1988-11-01T00:00:00 to 1994-02-01T00:00:00.
- lstm / future_log_return (window 36, horizon 3): fold 2 up period, 1993-12-01T00:00:00 to 1999-03-01T00:00:00.
- 252 additional underperforming folds are listed in `report.json`.

No tested configuration met all acceptance criteria; the validation results do not support a stable predictive advantage.

## Data limitations

- Input source is recorded in a manifest but has not been independently verified.

Per-fold metrics, per-step errors, data manifest, and the complete experiment configuration are in `report.json`.
Final test scores are reported separately and do not affect selection.
