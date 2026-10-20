# quarterly AAPL

Status: `validation_complete`

Final test: `not_requested`. Historical test periods are recorded in the shared exposure ledger.

## Data limitations

- Source remains unverified: results are exploratory/regression evidence.

## Validation selection

Candidates must pass MASE < 1, median baseline RMSE gain >= 2%, and improvement in at least 3/5 folds.
Passing candidates are ordered by normalized log-return RMSE, fold variability, then simplicity. Otherwise select last close.

| Target | Model | Window | Horizon | Accepted |
|---|---|---:|---:|---|
| future_close | last_close | 6 | 3 | False |
| future_log_return | last_close | 6 | 3 | False |

## Window and model comparison

| Target | Model | Window | Horizon | MASE | Return error / baseline | RMSE gain | Fold std |
|---|---|---:|---:|---:|---:|---:|---:|
| future_log_return | xpatch | 24 | 3 | 3.5061 | 0.9967 | -0.27% | 0.0502 |
| future_log_return | xpatch | 12 | 3 | 3.4599 | 0.9970 | 0.09% | 0.0456 |
| future_log_return | xpatch | 36 | 3 | 3.5429 | 0.9977 | -1.92% | 0.0564 |
| future_close | last_close | 6 | 3 | 3.4948 | 1.0000 | 0.00% | 0.0000 |
| future_log_return | last_close | 6 | 3 | 3.4948 | 1.0000 | 0.00% | 0.0000 |
| future_close | last_close | 12 | 3 | 3.4948 | 1.0000 | 0.00% | 0.0000 |
| future_log_return | last_close | 12 | 3 | 3.4948 | 1.0000 | 0.00% | 0.0000 |
| future_close | last_close | 24 | 3 | 3.4948 | 1.0000 | 0.00% | 0.0000 |
| future_log_return | last_close | 24 | 3 | 3.4948 | 1.0000 | 0.00% | 0.0000 |
| future_close | last_close | 36 | 3 | 3.4948 | 1.0000 | 0.00% | 0.0000 |
| future_log_return | last_close | 36 | 3 | 3.4948 | 1.0000 | 0.00% | 0.0000 |
| future_log_return | lstm | 12 | 3 | 3.4959 | 1.0008 | -1.24% | 0.0960 |
| future_log_return | lstm | 6 | 3 | 3.4972 | 1.0012 | -1.27% | 0.0897 |
| future_log_return | gru | 12 | 3 | 3.4146 | 1.0014 | -1.22% | 0.1071 |
| future_log_return | xpatch | 6 | 3 | 3.4841 | 1.0014 | 0.07% | 0.0217 |
| future_log_return | gru | 6 | 3 | 3.4119 | 1.0028 | -1.35% | 0.1076 |
| future_log_return | gru | 24 | 3 | 3.4176 | 1.0047 | -1.76% | 0.1141 |
| future_log_return | lstm | 24 | 3 | 3.4935 | 1.0059 | -1.99% | 0.0955 |
| future_log_return | gru | 36 | 3 | 3.4288 | 1.0090 | -2.46% | 0.1079 |
| future_log_return | lstm | 36 | 3 | 3.4879 | 1.0097 | -2.54% | 0.0948 |
| future_log_return | mlp | 6 | 3 | 3.5319 | 1.0129 | -1.23% | 0.5303 |
| future_close | drift | 36 | 3 | 3.6612 | 1.0261 | -3.47% | 0.0644 |
| future_log_return | mlp | 12 | 3 | 3.5106 | 1.0270 | -4.94% | 0.1227 |
| future_log_return | mlp | 24 | 3 | 3.3970 | 1.0362 | -4.85% | 0.2096 |
| future_log_return | drift | 36 | 3 | 3.5956 | 1.0369 | -5.10% | 0.0534 |
| future_log_return | drift | 24 | 3 | 3.5703 | 1.0474 | -7.75% | 0.0416 |
| future_close | drift | 24 | 3 | 3.5914 | 1.0535 | -4.65% | 0.0761 |
| future_log_return | ridge | 6 | 3 | 3.5631 | 1.0754 | -7.48% | 0.3205 |
| future_log_return | drift | 12 | 3 | 3.8061 | 1.0991 | -11.88% | 0.0796 |
| future_close | drift | 12 | 3 | 3.8710 | 1.1331 | -9.16% | 0.1497 |
| future_log_return | mlp | 36 | 3 | 3.6687 | 1.1376 | -10.21% | 0.8089 |
| future_log_return | drift | 6 | 3 | 4.0714 | 1.1817 | -21.17% | 0.0774 |
| future_log_return | ridge | 12 | 3 | 4.1143 | 1.1860 | -28.72% | 0.4012 |
| future_log_return | ridge | 24 | 3 | 5.1576 | 1.2697 | -41.59% | 0.4631 |
| future_close | drift | 6 | 3 | 4.1391 | 1.3734 | -16.96% | 4.6824 |
| future_close | ridge | 6 | 3 | 4.5853 | 1.4890 | -45.17% | 2.7980 |
| future_close | mlp | 6 | 3 | 5.6502 | 1.6444 | -66.99% | 6.2362 |
| future_log_return | ridge | 36 | 3 | 8.0647 | 1.9668 | -133.46% | 0.4184 |
| future_close | mlp | 12 | 3 | 5.9396 | 1.9695 | -105.27% | 6.7456 |
| future_close | gru | 12 | 3 | 6.0827 | 1.9768 | -95.32% | 5.5330 |
| future_close | gru | 24 | 3 | 5.7434 | 2.0545 | -105.36% | 5.6327 |
| future_close | gru | 6 | 3 | 6.3469 | 2.0587 | -93.95% | 5.6808 |
| future_close | ridge | 12 | 3 | 4.9019 | 2.0827 | -142.86% | 2.6579 |
| future_close | gru | 36 | 3 | 5.5881 | 2.1298 | -115.06% | 5.8381 |
| future_close | lstm | 36 | 3 | 6.9084 | 2.2946 | -108.91% | 6.8560 |
| future_close | xpatch | 6 | 3 | 6.9333 | 2.3041 | -110.11% | 7.9706 |
| future_close | lstm | 24 | 3 | 7.2424 | 2.4096 | -116.65% | 6.7580 |
| future_close | mlp | 36 | 3 | 6.5312 | 2.4237 | -156.44% | 34.2936 |
| future_close | seasonal | 6 | 3 | 10.2110 | 2.4522 | -135.23% | 0.6678 |
| future_log_return | seasonal | 6 | 3 | 10.2110 | 2.4522 | -135.23% | 0.6678 |
| future_close | seasonal | 12 | 3 | 10.2110 | 2.4522 | -135.23% | 0.6678 |
| future_log_return | seasonal | 12 | 3 | 10.2110 | 2.4522 | -135.23% | 0.6678 |
| future_close | seasonal | 24 | 3 | 10.2110 | 2.4522 | -135.23% | 0.6678 |
| future_log_return | seasonal | 24 | 3 | 10.2110 | 2.4522 | -135.23% | 0.6678 |
| future_close | seasonal | 36 | 3 | 10.2110 | 2.4522 | -135.23% | 0.6678 |
| future_log_return | seasonal | 36 | 3 | 10.2110 | 2.4522 | -135.23% | 0.6678 |
| future_close | lstm | 12 | 3 | 7.6035 | 2.5273 | -124.66% | 6.6535 |
| future_close | lstm | 6 | 3 | 7.8580 | 2.6183 | -130.55% | 6.5927 |
| future_close | mlp | 24 | 3 | 5.5619 | 2.6381 | -192.11% | 6.0250 |
| future_close | ridge | 24 | 3 | 5.3877 | 2.7376 | -237.78% | 2.5075 |
| future_close | xpatch | 24 | 3 | 8.2468 | 2.7976 | -151.78% | 6.3330 |
| future_close | xpatch | 36 | 3 | 8.8819 | 3.1461 | -158.19% | 6.9697 |
| future_close | xpatch | 12 | 3 | 9.3675 | 3.3314 | -169.23% | 6.3723 |
| future_close | ridge | 36 | 3 | 7.6066 | 3.5855 | -230.46% | 2.0631 |

Full per-fold, per-step and training errors, skips, hashes and diagnostic flags: `report.json`.
Price charts/CSVs show at most 100 final-lead examples per fold; metrics use all validation samples.
Diagnostic actions and hybrid/stopping rules: `DIAGNOSTICS.md` in the project root.
Hourly bars exclude the incomplete terminal 30 minutes; horizons count complete trading bars across sessions.
No predictive advantage is claimed unless every acceptance gate passes.
