# Current long-term validation

This directory contains the six model artifacts, validation reports, validation
plots/CSVs, and prediction JSON files regenerated with the current code on
2026-10-20.

## Decision

Annual, quarterly, and monthly validation each evaluated both `future_close`
and `future_log_return`. No candidate passed all acceptance gates (MASE < 1,
at least 2% median RMSE improvement over the baseline, and improvement in at
least 3 of 5 folds). The formal selection for both targets at all three
scales is therefore `last_close`. These results do not claim predictive
advantage.

The six regenerated artifacts are:

- `annual/future_close_h12.joblib`
- `annual/future_log_return_h12.joblib`
- `quarterly/future_close_h3.joblib`
- `quarterly/future_log_return_h3.joblib`
- `monthly/future_close_h21.joblib`
- `monthly/future_log_return_h21.joblib`

The matching `*_prediction.json` files in this directory were produced through
the current `predict.py` entry point. Artifact code hashes were checked during
prediction; no hash was bypassed.

## Data and test status

The available long-term source is an unverified legacy mirror. The monthly
data ends at `2022-05-01` and the daily data ends at `2022-06-17`; the source
could not be updated in this run. These outputs are regression evidence for
the pipeline, not current-market forecasts. A final untouched test was not
run because the source provenance and recency do not satisfy the conditions
for a reliable final test.

The minute experiment remains in
`../iex_exploratory/minute/REPORT.md` with status
`exploratory_quality_limited`. Its observed regular-session coverage is 97%,
below the required 98%, and its cleaning procedure is not suitable evidence
for a strict causal minute forecast. It is excluded from formal model ranking.

See each profile's `REPORT.md`, `report.json`, and `frozen_config.json` for
per-fold metrics, data hashes, and the exact training configuration.
