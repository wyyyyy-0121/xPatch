# Supplementary Material: AAPL RV30 Forecast Comparison

This supplement accompanies *Forecasting 30-Minute AAPL Realized Variance: Temporal Stability and Calibration Across Model Families*. It documents completed historical research. The planned matched-input, multi-target, simulation, and timely prospective extensions have no results in this package.

## S1. Numerical sources and study identity

The historical protocol signature is `65a4daf1ebacb8a7dedca5e866010bd94cda25acc7124509ec81d1c14dc2fbf8`. `evidence.json` identifies the development and confirmation evaluation files, search coverage, data manifest, and the two local verification rounds for each phase. The three original CSV files contain all 12 development model/panel combinations, the four admitted confirmation model/panel combinations, and quarterly diagnostics.

The manuscript uses the saved session-weighted results. It does not re-estimate model performance, add hypothesis tests, change the qualification rule, or read interim shadow-observation scores. The annual, seed, clock, and calibration exports in `diagnostics/` retain the source evaluation identities in their own `evidence.json`. Relative annual changes and square roots of saved covariance diagonals are arithmetic summaries of those records.

## S2. Selection and prediction details

`configs/rv30.json`, `configs/rv30_search.json`, and the original freeze record identify the feature lists and enumerated configuration order. HAR, Ridge, LightGBM, xPatch, and Chronos-2 use respectively 12, 32, 80, 32, and 8 configurations per panel and outer fold. The previous-RV rule is fixed. Counts differ by design and do not imply equal compute budgets.

For stochastic models, inner screening uses seed 2026; the three retained configurations are also evaluated with 2027 and 2028. Formal predictions retain the three seeds and their arithmetic mean on the calibrated variance scale. For each seed, empirical interval endpoints are raw variance predictions multiplied by exponentiated calibration residual quantiles. Ensemble intervals instead multiply the ensemble mean by exponentiated residual quantiles around the calibration ensemble mean. These operations preserve distinct point and interval calibration definitions.

The saved group `experiment.json` records selected configurations and outer job paths. Each completed job retains fitted transformation and calibration state, predictions, parameters, and resource receipts. The search coverage file covers 100 tuned groups and 12,000 inner jobs; fixed references and formal outer predictions bring the ledger to 120 groups and 12,200 jobs. Failed or pruned work remains traceable.

The 2025H2–2026H1 stage admits the two qualified Ridge candidates and their two corresponding HAR baselines. Their fitting history ends on 28 May 2025 and calibration on 27 June 2025. Later historical labels do not guide their hyperparameter choice or calibration. The period had prior project exposure and is not described as a blind holdout.

## S3. Diagnostic interpretation

`annual.csv` preserves 2023, 2024, and 2025H1 development segments and 2025H2/2026H1 confirmation segments. `seeds.csv` contains paired mean QLIKE differences versus the matching HAR for each available seed and segment. Deterministic outputs are not rerun under new seed labels to create replication.

The clock groups use origin times in Eastern Time: open through 10:30 inclusive, midday after 10:30 through 14:00 inclusive, and late after 14:00. `clock.csv` gives each group's actual session and origin counts. Early closes and eligibility rules mean that the groups need not contain the same sessions. Group losses are descriptive and are not a new family of significance tests.

`calibration.csv` contains daily-mean Mincer–Zarnowitz intercepts and slopes, HAC standard errors, intercept/slope covariance, and session counts. These regressions use five HAC lags and daily means of overlapping intraday RV targets. The main text reports coefficients and standard errors without a new joint test. Confirmation and pooled development have different refitting structures, limiting direct attribution of coefficient changes.

The nominal 80% and 90% intervals derive from calibration log-residual quantiles. They have no distribution-free or conditional-coverage guarantee. Near-nominal pooled coverage does not establish coverage during each annual or intraday segment. Width is in variance units, with a factor of 10⁵ used only for presentation.

## S4. Local numerical verification

Development and confirmation each passed two local verification rounds. Supervised models received full saved-model formal-prediction replay. The fixed pretrained comparison received the preregistered first/last batch replay, alongside independent aggregate-statistic recomputation; this does not constitute a replay of every pretrained inference.

The evidence preserves 42 earlier xPatch replay discrepancies and a separate precision correction. Restoring the recorded training numerical settings produced exact agreement for the 60 saved xPatch models and 1,620,216 formal predictions within the declared replay scope. The correction changed neither the selected hyperparameters nor the qualification thresholds. The original discrepancies remain part of the provenance record. Local repeatability is distinct from external research replication.

## S5. Prior task and computational implementation

The project previously evaluated 12-minute cumulative returns. Its negative results remain in the preserved earlier research records. RV30 has a different label and loss; the present variance results do not reverse that return-forecasting conclusion.

The implementation supports chronological audits, model selection, forecasting, reporting, and acceptance checks. Separate performance records describe fused training execution and inference/data-preparation optimization. The observed approximately fivefold inference-path speedup concerns four fitted models at paired historical origins; it excludes network acquisition and does not imply a comparable speedup for the full training study. Computational optimization is not a scientific mechanism for the model ranking.

All experiments share a ceiling of 22 logical CPU threads, including four allocated to the observation service. CPU affinity within WSL does not reserve physical processors exclusively for Windows. The stored performance qualifications for the completed historical run do not automatically admit new simulation or multi-target workloads.

## S6. Observation and planned extensions

The free delayed-SIP pipeline retains each original market origin, cutoff, and target but generates outputs approximately 15 minutes later, with actual receipt and issue timestamps. It is delayed shadow replay and cannot establish timely issuance before the target. Its frozen 60-complete-session evaluation and any subsequent timely prospective evaluation are separate from the historical paper results. Interim accumulated observation scores are not used here.

The execution plan proposes matched available information, AAPL/SPY/QQQ as separate targets after data/label audits, and controlled stable, shifted, leverage, and tail-distribution simulations. They remain planned studies. Their protocol, resource checks, parameters, and comparison families require separate locks before evaluation. The main paper contains no placeholder simulation results and makes no claim that these limitations have already been resolved.

## S7. Reproduction and availability

From the repository root, the following commands export saved historical results and diagnostics without training or querying the data supplier:

```bash
python scripts/export_rv30_paper_tables.py
python scripts/export_rv30_paper_diagnostics.py
```

The separate document/plotting environment can regenerate figures and compile the article:

```bash
/home/ayaka/.venvs/doc-tools/bin/python scripts/plot_rv30_paper.py
python scripts/build_rv30_paper.py --compile
```

`latex/README.md` describes standalone compilation. `delivery/manifest.json` indexes the local package; the paper export records contain source and output fingerprints. Access to a complete externally downloadable package is not established. Source market data require appropriate Alpaca access and applicable permissions; no redistribution license is claimed. Credentials, raw authorized data, and large checkpoints are not embedded in the paper.

## S8. Pre-submission editorial checks

`SOURCES.md`, `source_evidence.json`, and `JOURNAL_READING.md` distinguish original text read, publication metadata checked, and access-limited references. Classic references still have documented full-text gaps, including the original HAR paper and some statistical sources. The original xPatch conference PDF has now been read for its abstract, decomposition, dual-stream architecture, and channel-independence description. The attempted additional HAR link returned a non-PDF response and was not counted as a successful original-source retrieval.

The manuscript is a historical empirical draft, with the corresponding author email deliberately blank. No venue-specific template or public release has been asserted. Journal formatting, remaining original-source checks, and the actual access route to a replication package require completion before submission.
