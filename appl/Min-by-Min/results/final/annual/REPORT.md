# Annual AAPL experiment

Status: `complete`

Bars: 499 at 1mo; coverage: 41.50 years.

## Final test

Selected by validation, then evaluated on the final chronological 20%:

| Target | Model | Window | Horizon | Test RMSE | Test MASE |
|---|---|---:|---:|---:|---:|
| future_close | xpatch | 60 | 12 | 72.2136 | 291.1713 |
| future_log_return | last_close | 60 | 12 | 18.618 | 59.3269 |

## Validation comparison

| Target | Model | Window | Horizon | Median MASE | Median RMSE | Baseline RMSE gain | Folds improved |
|---|---|---:|---:|---:|---:|---:|---:|
| future_close | xpatch | 60 | 12 | 4.9035 | 0.273641 | -20.05% | 20% |
| future_close | gru | 60 | 12 | 4.9054 | 0.273568 | -20.11% | 20% |
| future_close | lstm | 60 | 12 | 5.0170 | 0.278621 | -19.61% | 0% |
| future_close | mlp | 60 | 12 | 5.2206 | 0.280646 | -19.46% | 0% |
| future_close | last_close | 60 | 12 | 5.5092 | 0.277967 | 0.00% | 0% |
| future_log_return | last_close | 60 | 12 | 5.5092 | 0.277967 | 0.00% | 0% |
| future_log_return | lstm | 60 | 12 | 5.5917 | 0.284854 | -0.21% | 40% |
| future_log_return | xpatch | 60 | 12 | 5.6100 | 0.287603 | -0.93% | 20% |
| future_log_return | mlp | 60 | 12 | 5.6761 | 0.286778 | -3.17% | 20% |
| future_log_return | gru | 60 | 12 | 5.6848 | 0.293235 | -1.64% | 20% |
| future_close | drift | 60 | 12 | 5.8495 | 0.30103 | -4.86% | 20% |
| future_log_return | drift | 60 | 12 | 6.1594 | 0.321905 | -8.62% | 20% |
| future_log_return | xpatch | 24 | 12 | 6.3074 | 0.282303 | -0.62% | 40% |
| future_log_return | lstm | 24 | 12 | 6.3387 | 0.281496 | -1.05% | 40% |
| future_log_return | gru | 24 | 12 | 6.3507 | 0.28559 | -0.23% | 40% |
| future_close | last_close | 24 | 12 | 6.3540 | 0.284941 | 0.00% | 0% |
| future_log_return | last_close | 24 | 12 | 6.3540 | 0.284941 | 0.00% | 0% |
| future_log_return | xpatch | 12 | 12 | 6.4066 | 0.284604 | -0.71% | 40% |
| future_close | last_close | 12 | 12 | 6.4354 | 0.282606 | 0.00% | 0% |
| future_log_return | last_close | 12 | 12 | 6.4354 | 0.282606 | 0.00% | 0% |

Acceptance threshold met: `false`.

## Weak validation periods

- xpatch / future_close (window 60, horizon 12): fold 1 down period, 1990-06-01T00:00:00 to 1995-12-01T00:00:00.
- xpatch / future_close (window 60, horizon 12): fold 2 up period, 1995-01-01T00:00:00 to 2000-07-01T00:00:00.
- xpatch / future_close (window 60, horizon 12): fold 4 up period, 2004-03-01T00:00:00 to 2009-09-01T00:00:00.
- xpatch / future_close (window 60, horizon 12): fold 5 up period, 2008-10-01T00:00:00 to 2014-04-01T00:00:00.
- gru / future_close (window 60, horizon 12): fold 1 down period, 1990-06-01T00:00:00 to 1995-12-01T00:00:00.
- gru / future_close (window 60, horizon 12): fold 2 up period, 1995-01-01T00:00:00 to 2000-07-01T00:00:00.
- gru / future_close (window 60, horizon 12): fold 4 up period, 2004-03-01T00:00:00 to 2009-09-01T00:00:00.
- gru / future_close (window 60, horizon 12): fold 5 up period, 2008-10-01T00:00:00 to 2014-04-01T00:00:00.
- lstm / future_close (window 60, horizon 12): fold 1 down period, 1990-06-01T00:00:00 to 1995-12-01T00:00:00.
- lstm / future_close (window 60, horizon 12): fold 2 up period, 1995-01-01T00:00:00 to 2000-07-01T00:00:00.
- lstm / future_close (window 60, horizon 12): fold 3 up period, 1999-08-01T00:00:00 to 2005-02-01T00:00:00.
- lstm / future_close (window 60, horizon 12): fold 4 up period, 2004-03-01T00:00:00 to 2009-09-01T00:00:00.
- 228 additional underperforming folds are listed in `report.json`.

No tested configuration met all acceptance criteria; the validation results do not support a stable predictive advantage.

## Data limitations

- Input source is recorded in a manifest but has not been independently verified.

Per-fold metrics, per-step errors, data manifest, and the complete experiment configuration are in `report.json`.
Final test scores are reported separately and do not affect selection.
