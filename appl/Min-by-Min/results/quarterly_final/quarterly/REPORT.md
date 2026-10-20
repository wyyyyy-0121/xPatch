# Quarterly AAPL experiment

Status: `complete`

Bars: 499 at 1mo; coverage: 41.50 years.

## Final test

Selected by validation, then evaluated on the final chronological 20%:

| Target | Model | Window | Horizon | Test RMSE | Test MASE |
|---|---|---:|---:|---:|---:|
| future_close | last_close | 24 | 3 | 9.80967 | 30.0571 |
| future_log_return | gru | 24 | 3 | 9.65976 | 29.4996 |

## Validation

Fold-level metrics, baseline comparisons, window rankings, and the acceptance decision are in `report.json`.
Final test scores are reported separately and do not affect selection.
