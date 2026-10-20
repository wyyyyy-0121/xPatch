# Annual AAPL experiment

Status: `complete`

Bars: 499 at 1mo; coverage: 41.50 years.

## Final test

Selected by validation, then evaluated on the final chronological 20%:

| Target | Model | Window | Horizon | Test RMSE | Test MASE |
|---|---|---:|---:|---:|---:|
| future_close | xpatch | 60 | 12 | 72.2136 | 291.1713 |
| future_log_return | last_close | 60 | 12 | 18.618 | 59.3269 |

## Validation

Fold-level metrics, baseline comparisons, window rankings, and the acceptance decision are in `report.json`.
Final test scores are reported separately and do not affect selection.
