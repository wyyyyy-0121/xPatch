# minute AAPL

Status: `insufficient_data`

Final test: `not_requested`. Historical test periods are recorded in the shared exposure ledger.

## Data limitations

- calendar_minute_coverage 0.9700 < required 0.9800; full XNYS sessions including missing days
- minimum_session_coverage 0.1513 < required 0.9800; full XNYS sessions including missing days

## Validation selection

Candidates must pass MASE < 1, median baseline RMSE gain >= 2%, and improvement in at least 3/5 folds.
Passing candidates are ordered by normalized log-return RMSE, fold variability, then simplicity. Otherwise select last close.

| Target | Model | Window | Horizon | Accepted |
|---|---|---:|---:|---|

## Window and model comparison

| Target | Model | Window | Horizon | MASE | Return error / baseline | RMSE gain | Fold std |
|---|---|---:|---:|---:|---:|---:|---:|

Full per-fold, per-step and training errors, skips, hashes and diagnostic flags: `report.json`.
Price charts/CSVs show at most 100 final-lead examples per fold; metrics use all validation samples.
Diagnostic actions and hybrid/stopping rules: `DIAGNOSTICS.md` in the project root.
Hourly bars exclude the incomplete terminal 30 minutes; horizons count complete trading bars across sessions.
No predictive advantage is claimed unless every acceptance gate passes.
