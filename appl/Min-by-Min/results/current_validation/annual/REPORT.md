# annual AAPL

Status: `validation_complete`

Final test: `not_requested`. Historical test periods are recorded in the shared exposure ledger.

## Data limitations

- Source remains unverified: results are exploratory/regression evidence.

## Validation selection

Candidates must pass MASE < 1, median baseline RMSE gain >= 2%, and improvement in at least 3/5 folds.
Passing candidates are ordered by normalized log-return RMSE, fold variability, then simplicity. Otherwise select last close.

| Target | Model | Window | Horizon | Accepted |
|---|---|---:|---:|---|
| future_close | last_close | 12 | 12 | False |
| future_log_return | last_close | 12 | 12 | False |

## Window and model comparison

| Target | Model | Window | Horizon | MASE | Return error / baseline | RMSE gain | Fold std |
|---|---|---:|---:|---:|---:|---:|---:|
| future_log_return | lstm | 12 | 12 | 5.8903 | 0.9641 | -2.94% | 0.1080 |
| future_log_return | lstm | 24 | 12 | 5.8916 | 0.9643 | -3.16% | 0.1093 |
| future_log_return | lstm | 60 | 12 | 5.8889 | 0.9645 | -3.38% | 0.1151 |
| future_log_return | lstm | 36 | 12 | 5.8933 | 0.9647 | -3.26% | 0.1116 |
| future_log_return | mlp | 12 | 12 | 5.3522 | 0.9814 | 3.05% | 0.9300 |
| future_log_return | gru | 12 | 12 | 5.6077 | 0.9851 | -14.10% | 0.1598 |
| future_log_return | mlp | 24 | 12 | 5.7425 | 0.9879 | 1.84% | 0.2388 |
| future_log_return | gru | 24 | 12 | 5.6250 | 0.9947 | -12.06% | 0.1586 |
| future_close | last_close | 12 | 12 | 6.1329 | 1.0000 | 0.00% | 0.0000 |
| future_log_return | last_close | 12 | 12 | 6.1329 | 1.0000 | 0.00% | 0.0000 |
| future_close | last_close | 24 | 12 | 6.1329 | 1.0000 | 0.00% | 0.0000 |
| future_log_return | last_close | 24 | 12 | 6.1329 | 1.0000 | 0.00% | 0.0000 |
| future_close | last_close | 36 | 12 | 6.1329 | 1.0000 | 0.00% | 0.0000 |
| future_log_return | last_close | 36 | 12 | 6.1329 | 1.0000 | 0.00% | 0.0000 |
| future_close | last_close | 60 | 12 | 6.1329 | 1.0000 | 0.00% | 0.0000 |
| future_log_return | last_close | 60 | 12 | 6.1329 | 1.0000 | 0.00% | 0.0000 |
| future_log_return | gru | 36 | 12 | 5.6404 | 1.0025 | -10.35% | 0.1559 |
| future_log_return | gru | 60 | 12 | 5.6800 | 1.0074 | -9.05% | 0.1464 |
| future_log_return | xpatch | 24 | 12 | 6.2493 | 1.0131 | -6.43% | 0.0671 |
| future_log_return | xpatch | 60 | 12 | 6.2554 | 1.0132 | -3.36% | 0.0662 |
| future_log_return | xpatch | 12 | 12 | 6.5045 | 1.0173 | -6.66% | 0.0628 |
| future_log_return | drift | 36 | 12 | 8.8946 | 1.0183 | -38.48% | 0.1451 |
| future_log_return | drift | 24 | 12 | 9.1568 | 1.0307 | -36.93% | 0.1416 |
| future_close | drift | 60 | 12 | 6.5363 | 1.0386 | -5.00% | 0.1210 |
| future_log_return | drift | 60 | 12 | 6.8979 | 1.0489 | -9.10% | 0.1185 |
| future_log_return | xpatch | 36 | 12 | 6.7122 | 1.0651 | -10.42% | 0.1155 |
| future_log_return | mlp | 36 | 12 | 5.7803 | 1.0929 | -11.95% | 0.3718 |
| future_close | drift | 36 | 12 | 7.8121 | 1.1482 | -6.94% | 1.8989 |
| future_log_return | mlp | 60 | 12 | 6.0847 | 1.1974 | -29.75% | 0.1880 |
| future_log_return | ridge | 12 | 12 | 6.1693 | 1.2202 | -6.09% | 0.6255 |
| future_close | ridge | 12 | 12 | 6.3600 | 1.2607 | -4.96% | 0.9933 |
| future_log_return | ridge | 60 | 12 | 11.2007 | 1.3223 | -41.09% | 1.1339 |
| future_log_return | drift | 12 | 12 | 11.1941 | 1.3390 | -61.09% | 0.2506 |
| future_close | drift | 24 | 12 | 7.6310 | 1.3641 | -11.19% | 2.0336 |
| future_close | seasonal | 12 | 12 | 9.6433 | 1.3676 | -28.66% | 0.1122 |
| future_log_return | seasonal | 12 | 12 | 9.6433 | 1.3676 | -28.66% | 0.1122 |
| future_close | seasonal | 24 | 12 | 9.6433 | 1.3676 | -28.66% | 0.1122 |
| future_log_return | seasonal | 24 | 12 | 9.6433 | 1.3676 | -28.66% | 0.1122 |
| future_close | seasonal | 36 | 12 | 9.6433 | 1.3676 | -28.66% | 0.1122 |
| future_log_return | seasonal | 36 | 12 | 9.6433 | 1.3676 | -28.66% | 0.1122 |
| future_close | seasonal | 60 | 12 | 9.6433 | 1.3676 | -28.66% | 0.1122 |
| future_log_return | seasonal | 60 | 12 | 9.6433 | 1.3676 | -28.66% | 0.1122 |
| future_close | gru | 12 | 12 | 5.5641 | 1.3794 | -16.76% | 2.5416 |
| future_close | lstm | 12 | 12 | 5.6823 | 1.4051 | -18.82% | 2.9715 |
| future_close | lstm | 24 | 12 | 5.6959 | 1.4076 | -18.95% | 3.0255 |
| future_close | lstm | 36 | 12 | 5.7068 | 1.4096 | -19.05% | 3.0887 |
| future_close | lstm | 60 | 12 | 5.7241 | 1.4128 | -19.19% | 3.1022 |
| future_close | xpatch | 36 | 12 | 5.7465 | 1.4163 | -19.20% | 2.9323 |
| future_close | mlp | 12 | 12 | 5.8012 | 1.4251 | -19.62% | 10.7939 |
| future_close | mlp | 36 | 12 | 5.8505 | 1.4395 | -21.04% | 27.0061 |
| future_close | xpatch | 12 | 12 | 5.8799 | 1.4408 | -20.03% | 3.2305 |
| future_close | gru | 24 | 12 | 5.8956 | 1.4443 | -20.79% | 2.5898 |
| future_close | gru | 60 | 12 | 5.8987 | 1.4446 | -20.68% | 3.3459 |
| future_close | gru | 36 | 12 | 5.8981 | 1.4447 | -20.79% | 2.6824 |
| future_close | xpatch | 60 | 12 | 5.9089 | 1.4471 | -20.70% | 3.0848 |
| future_close | mlp | 24 | 12 | 5.9414 | 1.4510 | -20.16% | 12.8542 |
| future_close | xpatch | 24 | 12 | 6.0071 | 1.4641 | -21.88% | 3.0995 |
| future_close | mlp | 60 | 12 | 6.1194 | 1.4853 | -22.99% | 17.4353 |
| future_log_return | ridge | 36 | 12 | 9.9679 | 1.7601 | -37.60% | 0.7570 |
| future_log_return | ridge | 24 | 12 | 8.3556 | 1.9535 | -41.50% | 0.7337 |
| future_close | ridge | 60 | 12 | 13.0179 | 2.9197 | -61.34% | 17.6813 |
| future_close | ridge | 24 | 12 | 9.3281 | 3.2355 | -30.08% | 5.4927 |
| future_close | ridge | 36 | 12 | 12.6973 | 3.5793 | -58.49% | 11.6564 |
| future_close | drift | 12 | 12 | 9.4498 | 6.3790 | -29.30% | 5.4117 |

Full per-fold, per-step and training errors, skips, hashes and diagnostic flags: `report.json`.
Price charts/CSVs show at most 100 final-lead examples per fold; metrics use all validation samples.
Diagnostic actions and hybrid/stopping rules: `DIAGNOSTICS.md` in the project root.
Hourly bars exclude the incomplete terminal 30 minutes; horizons count complete trading bars across sessions.
No predictive advantage is claimed unless every acceptance gate passes.
