# Annual AAPL experiment

Status: `complete`

Bars: 499 at 1mo; coverage: 41.50 years.

## Final test

Selected by validation, then evaluated on the final chronological 20%:

| Target | Model | Window | Horizon | Test RMSE | Test MASE |
|---|---|---:|---:|---:|---:|
| future_close | xpatch | 60 | 12 | 71.5273 | 285.9534 |
| future_log_return | last_close | 60 | 12 | 18.4409 | 58.4350 |

## Validation comparison

| Target | Model | Window | Horizon | Median MASE | Median RMSE | Baseline RMSE gain | Folds improved |
|---|---|---:|---:|---:|---:|---:|---:|
| future_close | xpatch | 60 | 12 | 5.7890 | 0.34654 | -31.31% | 0% |
| future_close | gru | 60 | 12 | 5.7920 | 0.346753 | -31.48% | 0% |
| future_close | lstm | 60 | 12 | 5.8953 | 0.351283 | -32.16% | 0% |
| future_close | mlp | 60 | 12 | 6.0342 | 0.34921 | -30.48% | 0% |
| future_close | last_close | 60 | 12 | 6.4181 | 0.337533 | 0.00% | 0% |
| future_log_return | last_close | 60 | 12 | 6.4181 | 0.337533 | 0.00% | 0% |
| future_log_return | lstm | 60 | 12 | 6.4858 | 0.342067 | 0.55% | 60% |
| future_log_return | xpatch | 60 | 12 | 6.5382 | 0.345428 | -0.65% | 40% |
| future_log_return | gru | 60 | 12 | 6.5766 | 0.347316 | -2.05% | 40% |
| future_log_return | mlp | 60 | 12 | 6.6658 | 0.348243 | -4.26% | 20% |
| future_close | drift | 60 | 12 | 6.7393 | 0.354507 | -0.83% | 40% |
| future_close | gru | 36 | 12 | 6.8232 | 0.359362 | -60.89% | 0% |
| future_close | mlp | 24 | 12 | 6.8367 | 0.355509 | -71.01% | 0% |
| future_close | xpatch | 36 | 12 | 6.9511 | 0.363994 | -56.80% | 0% |
| future_close | xpatch | 24 | 12 | 6.9566 | 0.364178 | -73.84% | 0% |
| future_log_return | mlp | 36 | 12 | 6.9744 | 0.329602 | -4.35% | 40% |
| future_close | gru | 24 | 12 | 7.0175 | 0.364543 | -73.17% | 0% |
| future_close | mlp | 36 | 12 | 7.0183 | 0.360503 | -59.93% | 0% |
| future_close | lstm | 36 | 12 | 7.0276 | 0.366111 | -66.98% | 0% |
| future_log_return | drift | 60 | 12 | 7.0278 | 0.37101 | -9.92% | 0% |

Acceptance threshold met: `false`.

## Weak validation periods

- xpatch / future_close (window 60, horizon 12): fold 1 down period, 1990-07-01T00:00:00 to 1996-03-01T00:00:00.
- xpatch / future_close (window 60, horizon 12): fold 2 down period, 1995-04-01T00:00:00 to 2000-12-01T00:00:00.
- xpatch / future_close (window 60, horizon 12): fold 3 up period, 2000-01-01T00:00:00 to 2005-08-01T00:00:00.
- xpatch / future_close (window 60, horizon 12): fold 4 up period, 2004-09-01T00:00:00 to 2010-05-01T00:00:00.
- xpatch / future_close (window 60, horizon 12): fold 5 up period, 2009-06-01T00:00:00 to 2014-02-01T00:00:00.
- gru / future_close (window 60, horizon 12): fold 1 down period, 1990-07-01T00:00:00 to 1996-03-01T00:00:00.
- gru / future_close (window 60, horizon 12): fold 2 down period, 1995-04-01T00:00:00 to 2000-12-01T00:00:00.
- gru / future_close (window 60, horizon 12): fold 3 up period, 2000-01-01T00:00:00 to 2005-08-01T00:00:00.
- gru / future_close (window 60, horizon 12): fold 4 up period, 2004-09-01T00:00:00 to 2010-05-01T00:00:00.
- gru / future_close (window 60, horizon 12): fold 5 up period, 2009-06-01T00:00:00 to 2014-02-01T00:00:00.
- lstm / future_close (window 60, horizon 12): fold 1 down period, 1990-07-01T00:00:00 to 1996-03-01T00:00:00.
- lstm / future_close (window 60, horizon 12): fold 2 down period, 1995-04-01T00:00:00 to 2000-12-01T00:00:00.
- 232 additional underperforming folds are listed in `report.json`.

No tested configuration met all acceptance criteria; the validation results do not support a stable predictive advantage.

## Data limitations

- Input source is recorded in a manifest but has not been independently verified.

Per-fold metrics, per-step errors, data manifest, and the complete experiment configuration are in `report.json`.
Final test scores are reported separately and do not affect selection.
