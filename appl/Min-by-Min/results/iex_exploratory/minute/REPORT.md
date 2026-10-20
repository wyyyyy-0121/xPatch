# minute AAPL

Status: `exploratory_quality_limited`

Final test: `not_requested`. Historical test periods are recorded in the shared exposure ledger.

## Data limitations

- Coverage is below the formal 98% requirement; this run is exploratory only.
- IEX HIST is a single-exchange sample, not consolidated U.S. market activity.
- HF Data Library Clean filtering uses a centered 50-bar window and can depend on future neighboring bars.
- The exact local file version is not checksum-matched to the provider release; consult the quality audit.

## Validation selection

Candidates must pass MASE < 1, median baseline RMSE gain >= 2%, and improvement in at least 3/5 folds.
Passing candidates are ordered by normalized log-return RMSE, fold variability, then simplicity. Otherwise select last close.

| Target | Model | Window | Horizon | Accepted |
|---|---|---:|---:|---|
| future_close | last_close | 60 | 5 | False |
| future_log_return | last_close | 60 | 5 | False |
| future_close | last_close | 60 | 30 | False |
| future_log_return | last_close | 60 | 30 | False |

## Window and model comparison

| Target | Model | Window | Horizon | MASE | Return error / baseline | RMSE gain | Fold std |
|---|---|---:|---:|---:|---:|---:|---:|
| future_close | last_close | 60 | 5 | 1.7172 | 1.0000 | 0.00% | 0.0000 |
| future_log_return | last_close | 60 | 5 | 1.7172 | 1.0000 | 0.00% | 0.0000 |
| future_close | last_close | 60 | 30 | 4.4766 | 1.0000 | 0.00% | 0.0000 |
| future_log_return | last_close | 60 | 30 | 4.4766 | 1.0000 | 0.00% | 0.0000 |
| future_close | last_close | 120 | 5 | 1.7172 | 1.0000 | 0.00% | 0.0000 |
| future_log_return | last_close | 120 | 5 | 1.7172 | 1.0000 | 0.00% | 0.0000 |
| future_close | last_close | 120 | 30 | 4.4766 | 1.0000 | 0.00% | 0.0000 |
| future_log_return | last_close | 120 | 30 | 4.4766 | 1.0000 | 0.00% | 0.0000 |
| future_log_return | xpatch | 60 | 5 | 1.7182 | 1.0002 | -0.02% | 0.0008 |
| future_log_return | xpatch | 120 | 5 | 1.7194 | 1.0004 | -0.04% | 0.0010 |
| future_log_return | xpatch | 60 | 30 | 4.5019 | 1.0017 | -0.18% | 0.0035 |
| future_log_return | ridge | 60 | 5 | 1.7182 | 1.0035 | -0.35% | 0.0053 |
| future_log_return | xpatch | 120 | 30 | 4.5016 | 1.0038 | -0.41% | 0.0088 |
| future_log_return | ridge | 120 | 5 | 1.7270 | 1.0046 | -0.45% | 0.0086 |
| future_log_return | ridge | 60 | 30 | 4.5682 | 1.0105 | -1.40% | 0.0117 |
| future_log_return | ridge | 120 | 30 | 4.5989 | 1.0207 | -2.04% | 0.0217 |
| future_log_return | drift | 120 | 5 | 1.7725 | 1.0233 | -2.36% | 0.0049 |
| future_close | drift | 120 | 5 | 1.7725 | 1.0233 | -2.37% | 0.0050 |
| future_log_return | drift | 60 | 5 | 1.7856 | 1.0383 | -3.88% | 0.0025 |
| future_close | drift | 60 | 5 | 1.7859 | 1.0390 | -3.95% | 0.0023 |
| future_close | drift | 120 | 30 | 5.0560 | 1.1109 | -11.26% | 0.0337 |
| future_log_return | drift | 120 | 30 | 5.0558 | 1.1111 | -11.27% | 0.0322 |
| future_close | drift | 60 | 30 | 5.2440 | 1.1992 | -20.24% | 0.0312 |
| future_log_return | drift | 60 | 30 | 5.2443 | 1.2012 | -20.16% | 0.0298 |
| future_close | seasonal | 60 | 30 | 25.2121 | 5.1103 | -404.22% | 0.3345 |
| future_log_return | seasonal | 60 | 30 | 25.2121 | 5.1103 | -404.22% | 0.3345 |
| future_close | seasonal | 120 | 30 | 25.2121 | 5.1103 | -404.22% | 0.3345 |
| future_log_return | seasonal | 120 | 30 | 25.2121 | 5.1103 | -404.22% | 0.3345 |
| future_close | ridge | 120 | 30 | 32.0065 | 6.1764 | -572.53% | 2.8126 |
| future_close | ridge | 60 | 30 | 32.1157 | 6.1944 | -569.74% | 2.8424 |
| future_close | seasonal | 60 | 5 | 26.9053 | 12.8416 | -1187.71% | 0.7492 |
| future_log_return | seasonal | 60 | 5 | 26.9053 | 12.8416 | -1187.71% | 0.7492 |
| future_close | seasonal | 120 | 5 | 26.9053 | 12.8416 | -1187.71% | 0.7492 |
| future_log_return | seasonal | 120 | 5 | 26.9053 | 12.8416 | -1187.71% | 0.7492 |
| future_close | xpatch | 120 | 30 | 62.5695 | 15.1297 | -1487.02% | 11.6001 |
| future_close | ridge | 60 | 5 | 34.5822 | 16.1187 | -1559.74% | 7.0944 |
| future_close | ridge | 120 | 5 | 34.4968 | 16.1510 | -1563.89% | 7.0480 |
| future_close | xpatch | 60 | 30 | 65.4243 | 16.2324 | -1598.63% | 10.9984 |
| future_close | xpatch | 60 | 5 | 70.1007 | 37.1083 | -3708.39% | 27.1320 |
| future_close | xpatch | 120 | 5 | 66.2845 | 40.1681 | -4230.61% | 25.6343 |

Full per-fold, per-step and training errors, skips, hashes and diagnostic flags: `report.json`.
Price charts/CSVs show at most 100 final-lead examples per fold; metrics use all validation samples.
Diagnostic actions and hybrid/stopping rules: `DIAGNOSTICS.md` in the project root.
Hourly bars exclude the incomplete terminal 30 minutes; horizons count complete trading bars across sessions.
No predictive advantage is claimed unless every acceptance gate passes.
