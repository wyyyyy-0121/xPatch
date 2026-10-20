# Monthly AAPL experiment

Status: `complete`

Bars: 10468 at 1d; coverage: 41.51 years.

## Final test

Selected by validation, then evaluated on the final chronological 20%:

| Target | Model | Window | Horizon | Test RMSE | Test MASE |
|---|---|---:|---:|---:|---:|
| future_close | last_close | 252 | 21 | 4.77674 | 68.8691 |
| future_log_return | last_close | 252 | 21 | 4.77674 | 68.8691 |

## Validation comparison

| Target | Model | Window | Horizon | Median MASE | Median RMSE | Baseline RMSE gain | Folds improved |
|---|---|---:|---:|---:|---:|---:|---:|
| future_close | last_close | 252 | 21 | 8.6093 | 0.0801959 | 0.00% | 0% |
| future_log_return | last_close | 252 | 21 | 8.6093 | 0.0801959 | 0.00% | 0% |
| future_log_return | gru | 252 | 21 | 8.6896 | 0.0805002 | -1.12% | 0% |
| future_log_return | lstm | 252 | 21 | 8.6911 | 0.0822091 | -1.54% | 0% |
| future_log_return | xpatch | 252 | 21 | 8.7008 | 0.0801961 | -1.74% | 20% |
| future_close | drift | 252 | 21 | 8.7198 | 0.083211 | -3.12% | 0% |
| future_log_return | drift | 252 | 21 | 8.7975 | 0.0845711 | -5.19% | 0% |
| future_log_return | lstm | 21 | 21 | 8.8611 | 0.0808852 | -1.77% | 20% |
| future_log_return | gru | 21 | 21 | 8.8666 | 0.0800364 | -0.70% | 20% |
| future_close | last_close | 21 | 21 | 8.8849 | 0.0794816 | 0.00% | 0% |
| future_log_return | last_close | 21 | 21 | 8.8849 | 0.0794816 | 0.00% | 0% |
| future_log_return | lstm | 63 | 21 | 8.8972 | 0.0807548 | -1.70% | 40% |
| future_log_return | gru | 63 | 21 | 8.9035 | 0.0797438 | -0.42% | 40% |
| future_log_return | mlp | 21 | 21 | 8.9291 | 0.0808189 | -1.68% | 20% |
| future_close | last_close | 63 | 21 | 8.9300 | 0.0794069 | 0.00% | 0% |
| future_log_return | last_close | 63 | 21 | 8.9300 | 0.0794069 | 0.00% | 0% |
| future_log_return | xpatch | 63 | 21 | 8.9464 | 0.0803124 | -1.14% | 20% |
| future_log_return | xpatch | 21 | 21 | 8.9542 | 0.0804273 | -0.85% | 20% |
| future_log_return | mlp | 63 | 21 | 9.0971 | 0.0847938 | -4.35% | 0% |
| future_log_return | lstm | 126 | 21 | 9.1133 | 0.0802317 | -0.82% | 20% |

Acceptance threshold met: `false`.

## Weak validation periods

- gru / future_log_return (window 252, horizon 21): fold 1 up period, 1987-04-23T00:00:00 to 1992-09-30T00:00:00.
- gru / future_log_return (window 252, horizon 21): fold 2 down period, 1992-09-01T00:00:00 to 1998-02-11T00:00:00.
- gru / future_log_return (window 252, horizon 21): fold 3 up period, 1998-01-13T00:00:00 to 2003-07-07T00:00:00.
- gru / future_log_return (window 252, horizon 21): fold 4 up period, 2003-06-06T00:00:00 to 2008-11-20T00:00:00.
- gru / future_log_return (window 252, horizon 21): fold 5 up period, 2008-10-23T00:00:00 to 2014-04-11T00:00:00.
- lstm / future_log_return (window 252, horizon 21): fold 1 up period, 1987-04-23T00:00:00 to 1992-09-30T00:00:00.
- lstm / future_log_return (window 252, horizon 21): fold 2 down period, 1992-09-01T00:00:00 to 1998-02-11T00:00:00.
- lstm / future_log_return (window 252, horizon 21): fold 3 up period, 1998-01-13T00:00:00 to 2003-07-07T00:00:00.
- lstm / future_log_return (window 252, horizon 21): fold 4 up period, 2003-06-06T00:00:00 to 2008-11-20T00:00:00.
- lstm / future_log_return (window 252, horizon 21): fold 5 up period, 2008-10-23T00:00:00 to 2014-04-11T00:00:00.
- xpatch / future_log_return (window 252, horizon 21): fold 1 up period, 1987-04-23T00:00:00 to 1992-09-30T00:00:00.
- xpatch / future_log_return (window 252, horizon 21): fold 2 down period, 1992-09-01T00:00:00 to 1998-02-11T00:00:00.
- 254 additional underperforming folds are listed in `report.json`.

No tested configuration met all acceptance criteria; the validation results do not support a stable predictive advantage.

## Data limitations

- Input source is recorded in a manifest but has not been independently verified.

Per-fold metrics, per-step errors, data manifest, and the complete experiment configuration are in `report.json`.
Final test scores are reported separately and do not affect selection.
