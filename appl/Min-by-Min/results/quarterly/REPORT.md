# Quarterly AAPL experiment

Status: `complete`

Bars: 499 at 1mo; coverage: 41.50 years.

## Final test

Selected by validation, then evaluated on the final chronological 20%:

| Target | Model | Window | Horizon | Test RMSE | Test MASE |
|---|---|---:|---:|---:|---:|
| future_close | last_close | 24 | 3 | 9.80967 | 30.0571 |
| future_log_return | gru | 36 | 3 | 6.71479e+79 | 34462082404637940419815985609543045954952520501760765522082041113991650304065536.0000 |

## Validation

Fold-level metrics, baseline comparisons, window rankings, and the acceptance decision are in `report.json`.
Final test scores are reported separately and do not affect selection.
