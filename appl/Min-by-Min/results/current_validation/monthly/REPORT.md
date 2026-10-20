# monthly AAPL

Status: `validation_complete`

Final test: `not_requested`. Historical test periods are recorded in the shared exposure ledger.

## Data limitations

- Source remains unverified: results are exploratory/regression evidence.

## Validation selection

Candidates must pass MASE < 1, median baseline RMSE gain >= 2%, and improvement in at least 3/5 folds.
Passing candidates are ordered by normalized log-return RMSE, fold variability, then simplicity. Otherwise select last close.

| Target | Model | Window | Horizon | Accepted |
|---|---|---:|---:|---|
| future_close | last_close | 21 | 21 | False |
| future_log_return | last_close | 21 | 21 | False |

## Window and model comparison

| Target | Model | Window | Horizon | MASE | Return error / baseline | RMSE gain | Fold std |
|---|---|---:|---:|---:|---:|---:|---:|
| future_log_return | lstm | 252 | 21 | 8.7090 | 0.9985 | -1.16% | 0.0699 |
| future_close | last_close | 21 | 21 | 8.5646 | 1.0000 | 0.00% | 0.0000 |
| future_log_return | last_close | 21 | 21 | 8.5646 | 1.0000 | 0.00% | 0.0000 |
| future_close | last_close | 63 | 21 | 8.5646 | 1.0000 | 0.00% | 0.0000 |
| future_log_return | last_close | 63 | 21 | 8.5646 | 1.0000 | 0.00% | 0.0000 |
| future_close | last_close | 126 | 21 | 8.5646 | 1.0000 | 0.00% | 0.0000 |
| future_log_return | last_close | 126 | 21 | 8.5646 | 1.0000 | 0.00% | 0.0000 |
| future_close | last_close | 252 | 21 | 8.5646 | 1.0000 | 0.00% | 0.0000 |
| future_log_return | last_close | 252 | 21 | 8.5646 | 1.0000 | 0.00% | 0.0000 |
| future_log_return | lstm | 21 | 21 | 8.6985 | 1.0089 | -0.82% | 0.0614 |
| future_log_return | lstm | 63 | 21 | 8.7023 | 1.0092 | -1.13% | 0.0860 |
| future_log_return | lstm | 126 | 21 | 8.7054 | 1.0095 | -1.15% | 0.0958 |
| future_log_return | xpatch | 21 | 21 | 8.7299 | 1.0114 | -0.67% | 0.0183 |
| future_log_return | xpatch | 126 | 21 | 8.8717 | 1.0144 | -0.98% | 0.0437 |
| future_close | drift | 252 | 21 | 8.6605 | 1.0265 | -3.05% | 0.0308 |
| future_log_return | drift | 252 | 21 | 8.7361 | 1.0298 | -5.12% | 0.0227 |
| future_log_return | xpatch | 252 | 21 | 8.8041 | 1.0320 | -2.17% | 0.0585 |
| future_log_return | mlp | 63 | 21 | 12.2175 | 1.0394 | -4.23% | 0.7292 |
| future_log_return | gru | 63 | 21 | 9.1184 | 1.0401 | -2.06% | 0.1037 |
| future_log_return | xpatch | 63 | 21 | 8.9580 | 1.0416 | -3.73% | 0.0601 |
| future_log_return | gru | 126 | 21 | 9.1307 | 1.0418 | -2.21% | 0.1049 |
| future_log_return | gru | 252 | 21 | 9.1409 | 1.0432 | -2.32% | 0.1052 |
| future_log_return | drift | 126 | 21 | 9.2568 | 1.0510 | -5.19% | 0.0231 |
| future_log_return | gru | 21 | 21 | 9.1007 | 1.0515 | -2.81% | 0.1019 |
| future_close | drift | 126 | 21 | 9.1061 | 1.0815 | -5.63% | 0.0452 |
| future_log_return | mlp | 21 | 21 | 11.2495 | 1.0857 | -8.07% | 0.7531 |
| future_log_return | mlp | 126 | 21 | 17.6078 | 1.1007 | -11.29% | 0.8882 |
| future_log_return | drift | 63 | 21 | 9.7778 | 1.1112 | -10.57% | 0.0397 |
| future_log_return | ridge | 252 | 21 | 25.4851 | 1.1212 | -9.73% | 0.7454 |
| future_log_return | ridge | 21 | 21 | 10.1856 | 1.1486 | -13.40% | 0.0777 |
| future_log_return | ridge | 126 | 21 | 16.4229 | 1.1505 | -12.78% | 0.3087 |
| future_close | drift | 63 | 21 | 9.7684 | 1.1777 | -12.07% | 1.1644 |
| future_log_return | mlp | 252 | 21 | 19.4565 | 1.1930 | -30.99% | 0.7154 |
| future_log_return | ridge | 63 | 21 | 15.7038 | 1.2111 | -18.16% | 0.2843 |
| future_log_return | drift | 21 | 21 | 10.8297 | 1.2811 | -27.01% | 0.0475 |
| future_close | drift | 21 | 21 | 10.8744 | 1.4226 | -27.96% | 5.7443 |
| future_close | ridge | 252 | 21 | 22.9745 | 2.2920 | -130.63% | 5.3767 |
| future_close | ridge | 126 | 21 | 32.3254 | 3.2067 | -202.66% | 5.4351 |
| future_close | ridge | 63 | 21 | 36.5218 | 3.6450 | -233.79% | 5.5713 |
| future_close | ridge | 21 | 21 | 36.9130 | 3.6502 | -235.06% | 5.6493 |
| future_close | mlp | 21 | 21 | 37.7288 | 3.8484 | -248.41% | 8.3944 |
| future_close | mlp | 63 | 21 | 39.0400 | 3.9543 | -253.02% | 9.0114 |
| future_close | seasonal | 21 | 21 | 50.9837 | 4.8608 | -321.46% | 1.3818 |
| future_log_return | seasonal | 21 | 21 | 50.9837 | 4.8608 | -321.46% | 1.3818 |
| future_close | seasonal | 63 | 21 | 50.9837 | 4.8608 | -321.46% | 1.3818 |
| future_log_return | seasonal | 63 | 21 | 50.9837 | 4.8608 | -321.46% | 1.3818 |
| future_close | seasonal | 126 | 21 | 50.9837 | 4.8608 | -321.46% | 1.3818 |
| future_log_return | seasonal | 126 | 21 | 50.9837 | 4.8608 | -321.46% | 1.3818 |
| future_close | seasonal | 252 | 21 | 50.9837 | 4.8608 | -321.46% | 1.3818 |
| future_log_return | seasonal | 252 | 21 | 50.9837 | 4.8608 | -321.46% | 1.3818 |
| future_close | lstm | 21 | 21 | 53.6328 | 5.8440 | -381.70% | 11.7011 |
| future_close | lstm | 63 | 21 | 53.7988 | 5.8629 | -382.75% | 11.7737 |
| future_close | lstm | 126 | 21 | 53.9661 | 5.8839 | -383.92% | 11.6937 |
| future_close | gru | 252 | 21 | 53.9712 | 5.8859 | -383.09% | 11.3846 |
| future_close | gru | 21 | 21 | 54.1554 | 5.9071 | -384.38% | 11.3593 |
| future_close | gru | 63 | 21 | 54.2281 | 5.9189 | -385.00% | 11.3754 |
| future_close | gru | 126 | 21 | 54.2640 | 5.9254 | -385.33% | 11.3783 |
| future_close | lstm | 252 | 21 | 54.3109 | 5.9298 | -386.51% | 11.7208 |
| future_close | mlp | 126 | 21 | 58.9354 | 6.6341 | -413.11% | 7.6285 |
| future_close | xpatch | 21 | 21 | 59.7616 | 6.8496 | -426.97% | 12.0584 |
| future_close | xpatch | 63 | 21 | 63.3239 | 7.2867 | -449.41% | 12.1151 |
| future_close | xpatch | 126 | 21 | 67.8549 | 7.9847 | -484.04% | 12.0565 |
| future_close | mlp | 252 | 21 | 88.2603 | 8.6787 | -573.18% | 8.6645 |
| future_close | xpatch | 252 | 21 | 80.1727 | 10.2944 | -581.28% | 11.9652 |

Full per-fold, per-step and training errors, skips, hashes and diagnostic flags: `report.json`.
Price charts/CSVs show at most 100 final-lead examples per fold; metrics use all validation samples.
Diagnostic actions and hybrid/stopping rules: `DIAGNOSTICS.md` in the project root.
Hourly bars exclude the incomplete terminal 30 minutes; horizons count complete trading bars across sessions.
No predictive advantage is claimed unless every acceptance gate passes.
