# Monthly AAPL experiment

Status: `complete`

Bars: 10468 at 1d; coverage: 41.51 years.

## Final test

Selected by validation, then evaluated on the final chronological 20%:

| Target | Model | Window | Horizon | Test RMSE | Test MASE |
|---|---|---:|---:|---:|---:|
| future_close | last_close | 252 | 21 | 4.73987 | 68.9178 |
| future_log_return | last_close | 252 | 21 | 4.73987 | 68.9178 |

## Validation comparison

| Target | Model | Window | Horizon | Median MASE | Median RMSE | Baseline RMSE gain | Folds improved |
|---|---|---:|---:|---:|---:|---:|---:|
| future_close | last_close | 252 | 21 | 8.6221 | 0.0802478 | 0.00% | 0% |
| future_log_return | last_close | 252 | 21 | 8.6221 | 0.0802478 | 0.00% | 0% |
| future_log_return | gru | 252 | 21 | 8.7005 | 0.0805403 | -1.29% | 0% |
| future_log_return | lstm | 252 | 21 | 8.7048 | 0.0816414 | -1.64% | 0% |
| future_close | drift | 252 | 21 | 8.7263 | 0.0832611 | -3.09% | 0% |
| future_log_return | xpatch | 252 | 21 | 8.7320 | 0.0802284 | -1.94% | 40% |
| future_close | last_close | 126 | 21 | 8.7808 | 0.0802478 | 0.00% | 0% |
| future_log_return | last_close | 126 | 21 | 8.7808 | 0.0802478 | 0.00% | 0% |
| future_close | last_close | 21 | 21 | 8.7848 | 0.0802478 | 0.00% | 0% |
| future_log_return | last_close | 21 | 21 | 8.7848 | 0.0802478 | 0.00% | 0% |
| future_log_return | drift | 252 | 21 | 8.8037 | 0.0846231 | -5.17% | 0% |
| future_close | last_close | 63 | 21 | 8.8135 | 0.0802478 | 0.00% | 0% |
| future_log_return | last_close | 63 | 21 | 8.8135 | 0.0802478 | 0.00% | 0% |
| future_log_return | lstm | 126 | 21 | 8.8386 | 0.0821878 | -2.06% | 0% |
| future_log_return | gru | 21 | 21 | 8.8488 | 0.0807431 | -1.28% | 0% |
| future_log_return | lstm | 21 | 21 | 8.8522 | 0.0814067 | -1.44% | 0% |
| future_log_return | gru | 126 | 21 | 8.8584 | 0.0806652 | -1.45% | 0% |
| future_log_return | lstm | 63 | 21 | 8.8672 | 0.0813509 | -1.37% | 0% |
| future_log_return | gru | 63 | 21 | 8.8762 | 0.0807894 | -1.23% | 0% |
| future_log_return | mlp | 21 | 21 | 8.8890 | 0.0817896 | -1.92% | 0% |

Acceptance threshold met: `false`.

## Weak validation periods

- gru / future_log_return (window 252, horizon 21): fold 1 up period, 1987-04-20T00:00:00 to 1992-09-23T00:00:00.
- gru / future_log_return (window 252, horizon 21): fold 2 down period, 1992-08-25T00:00:00 to 1998-02-02T00:00:00.
- gru / future_log_return (window 252, horizon 21): fold 3 up period, 1998-01-02T00:00:00 to 2003-06-23T00:00:00.
- gru / future_log_return (window 252, horizon 21): fold 4 up period, 2003-05-23T00:00:00 to 2008-11-05T00:00:00.
- gru / future_log_return (window 252, horizon 21): fold 5 up period, 2008-10-08T00:00:00 to 2014-02-25T00:00:00.
- lstm / future_log_return (window 252, horizon 21): fold 1 up period, 1987-04-20T00:00:00 to 1992-09-23T00:00:00.
- lstm / future_log_return (window 252, horizon 21): fold 2 down period, 1992-08-25T00:00:00 to 1998-02-02T00:00:00.
- lstm / future_log_return (window 252, horizon 21): fold 3 up period, 1998-01-02T00:00:00 to 2003-06-23T00:00:00.
- lstm / future_log_return (window 252, horizon 21): fold 4 up period, 2003-05-23T00:00:00 to 2008-11-05T00:00:00.
- lstm / future_log_return (window 252, horizon 21): fold 5 up period, 2008-10-08T00:00:00 to 2014-02-25T00:00:00.
- drift / future_close (window 252, horizon 21): fold 1 up period, 1987-04-20T00:00:00 to 1992-09-23T00:00:00.
- drift / future_close (window 252, horizon 21): fold 2 down period, 1992-08-25T00:00:00 to 1998-02-02T00:00:00.
- 264 additional underperforming folds are listed in `report.json`.

No tested configuration met all acceptance criteria; the validation results do not support a stable predictive advantage.

## Data limitations

- Input source is recorded in a manifest but has not been independently verified.

Per-fold metrics, per-step errors, data manifest, and the complete experiment configuration are in `report.json`.
Final test scores are reported separately and do not affect selection.
