# Intraday Realized-Variance Forecasting for AAPL: A Reproducible Comparison of Regularized Regression, Trees, and Time-Series Models

Working manuscript, 4 November 2026. Author: Wang Yang. This draft reports completed historical experiments. The delayed observation service has not yet produced an evaluation sample. Numerical sources are indexed in `evidence.json`; reference verification and reading scope are recorded in `SOURCES.md` and `source_evidence.json`.

## Abstract

We compare regularized regression, gradient-boosted trees, and time-series models for forecasting AAPL's realized variance over the next 30 minutes. Using SIP minute OHLCV data from January 2021 through June 2026, we evaluate an AAPL-only information set and a market information set that also includes SPY and QQQ history. The design combines ten quarterly expanding-window evaluations, three inner temporal validation folds, separate calibration windows, and whole-session purge gaps. The primary score is session-weighted QLIKE on the variance scale. Candidate qualification additionally requires dependence-aware inference, Holm adjustment across ten comparisons, and quarterly, annual-segment, and seed stability. The development sample contains 602 eligible sessions and 207,235 forecast origins per model. Ridge reduces QLIKE relative to the corresponding HAR baseline by 7.74% and 7.41% in the two information sets and meets all specified development gates. A subsequent historical confirmation sample of 251 sessions yields improvements of 6.29% and 7.97%. The evaluated xPatch and fixed-weight Chronos-2 configurations do not meet the joint gates. AAPL-only LightGBM improves aggregate QLIKE but fails the annual and seed stability requirements. These findings support retaining carefully tuned regularized baselines for this particular task. The evidence remains limited to one target asset, previously exposed historical periods, and incompletely established pretraining contamination boundaries. It does not establish timely prospective performance or trading profitability.

**Keywords:** realized variance; intraday forecasting; HAR; Ridge; xPatch; temporal validation

## 1. Introduction

Minute-level stock forecasting can refer to several different statistical problems. Return forecasting concerns future signed price changes; realized-variance forecasting concerns the magnitude of subsequent price fluctuations. The labels, loss functions, and possible decision uses differ. Our project initially investigated a 12-minute cumulative-return task. None of its candidates passed the development gates against a zero-return benchmark. We preserve that negative result separately and define the present RV30 study as a different task. A change of target is not evidence of an improvement on the earlier task.

Research on realized volatility provides a basis for aggregating high-frequency returns and forecasting their variation [1]. HAR models describe persistence through multiple time scales [2], while intraday periodicity motivates explicit treatment of the trading clock [3]. More flexible sequence models can represent additional nonlinear relationships, but their performance on broad forecasting benchmarks does not establish an advantage for a single stock and a short intraday horizon.

We ask whether Ridge, LightGBM, xPatch, and Chronos-2 provide stable improvements over a tuned HAR benchmark when information cutoffs, temporal selection, calibration, and statistical qualification are controlled. We also report how the model comparisons change when historical SPY and QQQ information is included. This second comparison has an architectural qualification: the market panel supplies conditional covariates to some models and auxiliary training tasks to the channel-independent xPatch implementation.

The contribution is an auditable empirical comparison rather than a new forecasting architecture. The study retains unsuccessful candidates, selected configurations, failed computations, and replay boundaries. Before the formal RV30 outer evaluation, the protocol, configurations, dependencies, and data identities were signed locally. This does not constitute registration on a public platform with an independent timestamp. The historical periods had also already been exposed within the project. We therefore describe the results as retrospective out-of-sample comparisons under temporal splitting, not as an untouched blind test.

## 2. Related work and methodological context

Andersen et al. [1] study modeling and forecasting realized volatility. We use a sum of squared minute returns as an observable target without assuming that one-minute sampling eliminates microstructure noise or that this target equals latent integrated variance. Corsi's HAR model [2] motivates a multiscale benchmark. Our implementation adapts this idea to a 30-minute intraday task using rolling measures, previous-session aggregates, and seasonality; it is not a replication of every specification in the original paper.

Patton [4] studies forecast comparisons based on imperfect volatility proxies. We use QLIKE on the original variance scale and retain its assumptions as a limitation: a suitable loss function does not resolve historical revisions, missing bars, or the timing of data availability. The optional positive and negative semivariance features relate to work on signed variation and volatility persistence [5].

Ridge introduces quadratic regularization into linear regression [9]. LightGBM supplies a gradient-boosted tree implementation [10]. We evaluate their saved local specifications rather than attributing all options or published speed claims of these methods to our implementation. xPatch uses exponential seasonal-trend decomposition in a dual-stream forecasting architecture [7]. We retain the original network architecture and search only parameters supported by the preserved implementation. Chronos-2 provides the fixed pretrained comparison [8]. Its general forecasting benchmarks do not replace local evaluation, and the training-data boundary of the exact checkpoint has not been independently established for this study.

Forecast errors at neighboring minute origins share most of their target returns. We therefore compare session-level loss differences instead of treating each minute as an independent observation. Diebold and Mariano [6], Newey and West [11], and Holm [12] provide the statistical context for predictive-accuracy comparison, serial-correlation adjustment, and multiplicity control. The particular aggregation, lags, resampling scheme, and joint qualification gates are choices fixed by our protocol.

## 3. Data and forecasting task

### 3.1. Data coverage and quality

The data are Alpaca SIP historical minute OHLCV bars queried with raw adjustment, covering 4 January 2021 through 30 June 2026. We construct the regular-session grid using the New York Stock Exchange calendar and `America/New_York`, including daylight-saving changes and early closes. The supplier labels bars by their start time. Across 1,378 sessions, the expected grid contains 535,620 minute bars for each asset.

| Asset | Expected bars | Observed bars | Missing bars | Role |
|---|---:|---:|---:|---|
| AAPL | 535,620 | 535,620 | 0 | Forecast target and historical input |
| SPY | 535,620 | 535,611 | 9 | Historical market input |
| QQQ | 535,620 | 535,620 | 0 | Historical market input |

We reconstruct the CSV data from the paginated raw responses and check all supplied fields and selected unusual sessions. The nine missing SPY bars remain missing; they are not interpolated. Some overlapping AAPL prices were checked against another source. SPY and QQQ prices and all three volume series have not received equivalent independent-provider verification. Final historical bars do not identify their original publication times or subsequent revisions. These checks establish the retained dataset's internal identity and consistency, not point-in-time certification.

### 3.2. Target and information cutoff

Let \(c(t)\) be the close of a one-minute bar ending at time \(t\). For two actually adjacent bars within the same session, define

\[
r(t)=\log c(t)-\log c(t-1\text{ min}).
\]

The first bar of a session, overnight gaps, missing bars, and the first return following a missing bar do not provide valid adjacent intraday returns. At market origin \(u\), the target is

\[
RV_{30}(u)=\sum_{k=1}^{30} r(u+k\text{ min})^2.
\]

All 30 future returns must exist, be adjacent, and belong to the same session. Scheduled origins begin at 09:45 Eastern Time, recur every minute, and stop when the target endpoint would exceed the actual close. A full regular session has 346 scheduled origins; early-close sessions have fewer.

Inputs end at \(u-1\text{ min}\). The last permitted supplier bar therefore starts at \(u-2\text{ min}\). For example, a 09:45 origin can use a bar starting at 09:43 and ending at 09:44. This buffer allows time for publication and calculation in the task definition, but it does not prove that the supplier historically delivered that bar on time.

Both information sets use common eligible evaluation origins. Eligibility requires the AAPL future label and the required historical inputs of all three assets. Future SPY or QQQ data are not used to select evaluation origins. Their future labels, when required for auxiliary training, have a separate training eligibility rule. Historical contexts must also meet the completeness requirements of the longest sequence input and the 22-session aggregates. A historical evaluation session may retain only some origins; its mean loss uses those origins. Deployment requires complete qualifying sessions and is counted separately.

## 4. Models and evaluation design

### 4.1. Models and information sets

The comparison includes the previous 30-minute RV, HAR/HAR-X, Ridge, LightGBM, xPatch, and Chronos-2. The fixed previous-RV reference receives a training-estimated seasonal ratio adjustment. Tabular features include prespecified combinations of 5-, 15-, 30-, 60-, and 120-minute RV, positive and negative semivariances, cumulative returns, previous-session and 5-/22-session aggregates, historical volume and high-low ranges, and trading-clock terms. The exact compact and rich feature lists are stored in the signed configurations.

Ridge standardizes predictors using fitting-window means and standard deviations, retains an intercept, and uses an SVD solver. LightGBM searches its registered tree, sampling, learning-rate, and regularization settings. None of these transformations is fitted on calibration or evaluation observations.

Sequence models receive rolling 30-minute RV in either log form or a log seasonal-ratio representation. Because the most recent known value is at \(u-1\text{ min}\), they predict 31 steps and use the last step: the trailing RV at \(u+30\text{ min}\), which equals the required future RV30 target. Selecting step 30 would shift the target. All sequence training labels remain within a session and the relevant temporal partition.

The preserved xPatch architecture is channel-independent. In its market panel, shared parameters learn auxiliary asset tasks without adding a cross-channel fusion layer to the AAPL output. This differs from adding conditional market covariates to a tabular model. Chronos-2 uses a fixed checkpoint revision and file fingerprints, with cross-learning between different forecast origins disabled. We do not interpret the market panels as identical architectural ablations across all model families.

### 4.2. Calibration and retransformation

For log-location forecasts, we estimate a multiplicative smearing factor from a separate calibration tail as the mean of \(\exp(\log y-\mu)\). Predictions in seasonal-ratio form are also multiplied by the fitting-window seasonal expectation for the known future calendar positions. Seasonal estimates and feature scaling use fitting data only; calibration residuals are never used for fitting the forecasting model.

Empirical 80% and 90% intervals use log-residual quantiles from the same calibration tail. Coverage and width are descriptive diagnostics, without a distribution-free or conditional-coverage guarantee. Calibration gives eligible origins equal weight; evaluation gives sessions equal weight. LightGBM and xPatch use seeds 2026, 2027, and 2028 for formal predictions. Their primary prediction is the arithmetic mean of individually calibrated variance forecasts. Ensemble intervals are calibrated on log residuals of that mean without changing the point prediction. Deterministic models are not rerun to manufacture seed replication.

### 4.3. Nested temporal validation and search

Development covers ten quarterly outer folds from 2023Q1 through 2025Q2. Each fold uses only preceding dates for model selection and fitting. Immediately before the outer quarter, we exclude one whole session, allocate the preceding 20 sessions to calibration, and exclude another whole session before calibration. Earlier observations form the final fitting set.

Three expanding inner folds use non-overlapping 20-session validation blocks within that earlier history. Each inner fold applies the same two purge boundaries and separate 20-session calibration tail. Training origins are sampled every five minutes and inner validation every ten minutes; calibration and formal evaluation use every minute. This fixed clock-based sampling reduces duplicated overlapping work without selecting observations by their targets.

| Model | Unique configurations per information set and outer fold | Selection details |
|---|---:|---|
| Previous RV | 1 fixed rule | No hyperparameter search |
| HAR/HAR-X | 12 | Regularization, semivariances, target representation |
| Ridge | 32 | Eight penalties, two feature sets, two representations |
| LightGBM | 80 | Fixed random search and inner early stopping |
| xPatch | 32 | Supported context, patch, decomposition, and training settings |
| Chronos-2 | 8 | Contexts 64/128/256/512, two representations; fixed weights |

LightGBM and xPatch first screen with seed 2026, then evaluate the top three configurations with seeds 2027 and 2028 on all three inner folds. xPatch uses registered successive halving, with cumulative epoch limits of 5, 15, and 30 and 32, 16, and 8 retained configurations. Optimizer and random states are carried forward. Final training duration is determined from inner results, not by outer early stopping. The model-specific search counts are registered budgets, not equal computing allocations.

The search-coverage record contains 100 tuned model groups and 12,000 inner jobs. Including fixed references and formal outer prediction jobs, the development run contains 120 groups and 12,200 jobs. Failed and pruned jobs remain in the ledger. Only candidates meeting the development gates enter historical confirmation for July 2025 through June 2026. Confirmation fixes the last development fold's selected configuration and refits before the confirmation period under the same calibration and purge discipline; it performs no new search.

### 4.4. Losses, dependence, and qualification gates

For target \(y\) and prediction \(p\), define \(\tilde y=\max(y,10^{-12})\), \(\tilde p=\max(p,10^{-12})\), and

\[
L(y,p)=\frac{\tilde y}{\tilde p}
-\log\left(\frac{\tilde y}{\tilde p}\right)-1.
\]

We first average this loss within each eligible session, then give sessions equal weight. Early-close sessions consequently receive the same aggregate weight as full sessions. Minute sample counts describe forecast coverage rather than independent sample size.

For each candidate, daily differences are candidate loss minus the same-panel HAR loss. We conduct a one-sided DM comparison using Bartlett-weighted HAC with five session lags. We also use single-session resampling and circular five-session block bootstrap, each with 5,000 replicates. Uncentered differences produce two-sided percentile intervals; zero-mean-centered differences produce null tail probabilities. A block consists of observed eligible sessions, without inserting zero losses at missing dates.

The conservative p-value for each comparison is the maximum of the three p-values. Holm adjustment covers the fixed ten-member family: five candidates against HAR in each information set. A development candidate must have lower QLIKE, Holm-adjusted p below 0.05, both 95% interval upper bounds below zero, improvement in at least seven of ten quarters, improvement in each of 2023, 2024, and 2025H1, and consistent pooled and annual-segment directions for every specified seed. Confirmation uses the corresponding prespecified significance and stability requirements, including improvement in at least three of four quarters and in both half-year segments. Secondary diagnostics include log-RV MSE, interval coverage and width, and a daily-mean Mincer–Zarnowitz regression. They do not replace the primary qualification rule.

## 5. Results

### 5.1. Development performance

The common development sample contains 602 eligible sessions and 207,235 origins per model. In both panels the target remains AAPL; SPY and QQQ are historical inputs or auxiliary training assets, not additional forecast targets.

| Information set | Model | QLIKE | Log-RV MSE | QLIKE improvement vs. corresponding HAR | Holm p | All gates |
|---|---|---:|---:|---:|---:|---|
| AAPL | Previous RV | 0.192281 | 0.345949 | −16.00% | 1 | Fail |
| AAPL | HAR | 0.165755 | 0.286719 | Reference | — | Reference |
| AAPL | Ridge | 0.152919 | 0.257131 | 7.74% | 0.0019996 | Pass |
| AAPL | LightGBM | 0.156501 | 0.251059 | 5.58% | 0.0136551 | Fail |
| AAPL | xPatch | 0.160833 | 0.270772 | 2.97% | 0.330452 | Fail |
| AAPL | Chronos-2 | 0.177211 | 0.308747 | −6.91% | 1 | Fail |
| Market | Previous RV | 0.192281 | 0.345949 | −16.91% | 1 | Fail |
| Market | HAR-X | 0.164462 | 0.289572 | Reference | — | Reference |
| Market | Ridge | 0.152271 | 0.257211 | 7.41% | 0.0019996 | Pass |
| Market | LightGBM | 0.158525 | 0.259838 | 3.61% | 0.330452 | Fail |
| Market | xPatch | 0.167298 | 0.276986 | −1.72% | 1 | Fail |
| Market | Chronos-2 | 0.178908 | 0.310045 | −8.78% | 1 | Fail |

Negative improvement denotes higher loss than HAR. Both Ridge candidates meet all development gates. AAPL-only Ridge improves in all ten quarters; market Ridge improves in nine. Their five-session block intervals for the daily QLIKE difference are [−0.017024, −0.009415] and [−0.016843, −0.008327], respectively.

![Development loss differences and five-session block intervals](figures/development_loss_differences.png)

**Figure 1.** Candidate minus corresponding HAR daily QLIKE differences. Bars are pointwise 95% percentile bootstrap intervals with five-session blocks; the intervals themselves are not multiplicity-adjusted. Qualification separately uses Holm adjustment and the stability gates. Teal indicates all gates passed. The comparison is retrospective and historical exposure is disclosed.

### 5.2. Aggregate significance and stability

AAPL-only LightGBM improves QLIKE by 5.58%, has Holm p = 0.013655, and has both bootstrap intervals below zero. It nevertheless fails the annual and seed consistency requirements. Its log-RV MSE is lower than Ridge's, illustrating that the losses assess different forecast properties. We retain the registered primary loss and joint decision rule.

AAPL-only xPatch improves average QLIKE by 2.97%, but its five-session interval [−0.010903, 0.000159] includes zero and its Holm p is 0.330452. It improves in six quarters and does not establish a stable gain under the protocol. Chronos-2 has higher QLIKE than HAR in both panels. This result concerns the locked checkpoint, representations, and eight searched configurations; it does not establish a general ranking of pretrained models.

Market Ridge has slightly lower numerical QLIKE than AAPL-only Ridge. The registered tests compare each candidate with its corresponding HAR, not the two Ridge forecasts with each other. We therefore do not infer a statistically established incremental contribution from SPY and QQQ. The other market-panel results also do not show a uniform improvement.

![Quarterly loss differences relative to HAR](figures/quarterly_loss_differences.png)

**Figure 2.** Session-weighted mean loss differences in all ten development quarters. Negative values favor the candidate. The two panels share the same vertical scale. Points are descriptive quarterly means; no additional quarter-specific significance tests or intervals were computed.

### 5.3. Historical confirmation

Confirmation contains 251 eligible sessions and 86,306 origins per model. It admits only the development-qualified Ridge candidates and their corresponding HAR baselines.

| Information set | HAR QLIKE | Ridge QLIKE | Improvement | Holm p | Five-session 95% difference interval |
|---|---:|---:|---:|---:|---|
| AAPL | 0.175851 | 0.164782 | 6.29% | 0.0019996 | [−0.013920, −0.008246] |
| Market | 0.174630 | 0.160710 | 7.97% | 0.0019996 | [−0.017701, −0.009876] |

Both candidates meet the historical confirmation gates, with improvement in every confirmation quarter. The remaining models have no equivalent confirmation results under this gated design, and we do not impute them. Because this historical period was previously exposed within the project, confirmation supplies subsequent-period consistency evidence rather than an independent blind validation.

### 5.4. Empirical interval diagnostics

Coverage is not a substitute for point-forecast accuracy. The following development diagnostics report both empirical coverage and average width. Width is on the original RV scale, multiplied by \(10^5\) for display. All quantities use equal session weights.

| Information set | Model | 80% coverage | 80% width × 10⁵ | 90% coverage | 90% width × 10⁵ |
|---|---|---:|---:|---:|---:|
| AAPL | Previous RV | 78.79% | 1.5071 | 89.31% | 2.1601 |
| AAPL | HAR | 79.12% | 1.1610 | 88.83% | 1.6126 |
| AAPL | Ridge | 79.33% | 1.4147 | 88.88% | 1.9232 |
| AAPL | LightGBM | 79.40% | 0.9966 | 88.90% | 1.3708 |
| AAPL | xPatch | 80.16% | 1.1575 | 89.59% | 1.5940 |
| AAPL | Chronos-2 | 79.63% | 1.2958 | 89.57% | 1.8322 |
| Market | Previous RV | 78.79% | 1.5071 | 89.31% | 2.1601 |
| Market | HAR-X | 78.89% | 1.1492 | 88.70% | 1.5827 |
| Market | Ridge | 78.66% | 1.2913 | 88.59% | 1.7851 |
| Market | LightGBM | 78.05% | 0.9894 | 88.01% | 1.3691 |
| Market | xPatch | 80.37% | 1.0791 | 89.45% | 1.4880 |
| Market | Chronos-2 | 79.43% | 1.3144 | 89.46% | 1.8565 |

AAPL-only Ridge has wider development intervals than HAR, whereas xPatch coverage is closer to the nominal levels. These observations address interval behavior and do not contradict the point-loss results. In confirmation, AAPL-only Ridge has 81.63%/90.90% coverage and widths 1.0291/1.4839 on the displayed scale; market Ridge has 81.52%/90.56% coverage and widths 1.0471/1.4768. Both have narrower aggregate confirmation intervals than their corresponding HAR. We do not claim conditional calibration or a statistically established interval-quality ranking. Exact values for every admitted model are supplied in the exported CSV files.

## 6. Reproducibility and observation service

The implementation records partitions, parameters, fitted transformations, calibration state, predictions, resource use, and file fingerprints in stable code and artifact paths. Development and confirmation each passed two local verification rounds. Supervised models received full formal-prediction replay; pretrained models received the fixed first/last batch replay specified by the verification plan, with independently recomputed aggregate metrics. This is a local computational check, not replication by an external research team or a full replay of every pretrained prediction. The record preserves 42 earlier xPatch replay failures and the numerical-precision correction that restored the training settings, without a new search or relaxed gates.

The deployment pipeline contains two qualified Ridge models and their two HAR references. Profiling identified repeated conversion of historical Timestamp dictionaries into tabular data as a major cost. We changed the record-construction path and cached only fixed calendar metadata, while recomputing prices, features, and forecasts from the supplied data.

Continuous tests cover 1,204 origins across four historical sessions, including a regular session, an early close, and dates before and after a daylight-saving transition. Mean four-model computation time is 0.1395–0.1438 seconds across the tested sessions, with a maximum of 0.2731 seconds. At 32 paired origins, a separate instance of the original implementation produced exactly equal four-model outputs. Paired speed ratios are 4.99–5.77. These measurements concern this inference and data-preparation path; they do not measure model training, network acquisition, or total project runtime. Experimental processes share a ceiling of 22 logical CPU threads, and the observer is allocated four.

The free data account supports delayed historical SIP queries. The observation service keeps the original market origin, input cutoff, and target, but schedules generation around 15 minutes later. It records actual receipt, generation, and write times without backdating. Part of the target window has therefore elapsed by publication. We classify this as delayed shadow replay, separate from timely prospective forecasting. At this manuscript's status snapshot, the service is waiting for a regular session and has no complete observation-day score. Evaluation is deferred until 60 complete qualifying shadow sessions under the frozen rules. Model parameters are not updated from interim scores. The original timely prospective acceptance requirement remains unfulfilled.

## 7. Discussion and limitations

Ridge's development and confirmation improvements show that a carefully tuned regularized baseline is competitive in this particular task. The engineered inputs already encode multiscale variation, the trading clock, and historical session information. This may make a linear predictor effective, but the study does not isolate the causal contributions of feature engineering, regularization, or sample size. The observed ranking should not be presented as a mechanism result.

Failure of a complex model to meet the joint gates is also not evidence that deep learning is generally unsuitable for financial forecasting. The target asset, horizon, feature representations, checkpoint, and model-specific search budgets are fixed. Sequence models and tabular models receive different representations, and the market panel has different structural effects across architectures. These differences bound interpretation of the comparison.

The dataset supplies one forecast target. Its historical bars may contain revisions, cross-provider checks are incomplete, and minute RV is an imperfect volatility proxy. The historical periods were previously exposed, and pretrained contamination boundaries remain uncertain. Local signatures and repeatable calculations improve auditability without removing those limitations. A new multi-asset study or genuinely unexposed evaluation would require a separately protected design rather than another round of tuning on these results.

The deployment models are also deliberately frozen: their fitting data end on 28 May 2025 and calibration on 27 June 2025. Later observations will assess stability across new market conditions without refitting. Delayed shadow replay can test data handling and frozen-model behavior under later observations, but its issue times prevent it from establishing a tradable, timely forecast. Neither historical loss reductions nor inference speed can be converted directly into stock returns. No directional strategy, execution rule, spread, slippage, or fee model has been evaluated here.

## 8. Conclusion

For the specified AAPL 30-minute realized-variance task, Ridge meets the development and subsequent historical confirmation gates in both information sets. AAPL-only LightGBM has an aggregate improvement without the required stability; xPatch and the locked Chronos-2 configurations do not meet the joint gates. The results support retaining regularized baselines and jointly reporting average loss, uncertainty, and temporal stability in this setting.

The present evidence is a retrospective comparison on previously exposed historical data. Delayed observation has been deployed but has no completed evaluation sample. Timely prospective performance, broader asset generalization, and economic decision value remain open empirical questions.

## Data and code availability

The local study index is `delivery/manifest.json`. It links the protocol, configuration and weight identities, search ledger, model artifacts, evaluation records, and verification scope. Paper tables can be regenerated with `scripts/export_rv30_paper_tables.py`; figures can be regenerated with `scripts/plot_rv30_paper.py` in a separate plotting environment. Neither script retrains models or recomputes hypothesis tests. Public access to the complete code/artifact package has not been established by this draft, and availability of the market data for redistribution has not been established. The submission version must describe the actual accessible package and data-access procedure.

## References

[1] Andersen, T. G., Bollerslev, T., Diebold, F. X., and Labys, P. (2003). Modeling and Forecasting Realized Volatility. *Econometrica*, 71(2), 579–625. https://doi.org/10.1111/1468-0262.00418.

[2] Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility. *Journal of Financial Econometrics*, 7(2), 174–196. https://doi.org/10.1093/jjfinec/nbp001.

[3] Andersen, T. G., and Bollerslev, T. (1997). Intraday periodicity and volatility persistence in financial markets. *Journal of Empirical Finance*, 4(2–3), 115–158. https://doi.org/10.1016/S0927-5398(97)00004-2.

[4] Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*, 160(1), 246–256. https://doi.org/10.1016/j.jeconom.2010.03.034.

[5] Patton, A. J., and Sheppard, K. (2015). Good Volatility, Bad Volatility: Signed Jumps and the Persistence of Volatility. *Review of Economics and Statistics*, 97(3), 683–697. https://doi.org/10.1162/REST_a_00503.

[6] Diebold, F. X., and Mariano, R. S. (1995). Comparing Predictive Accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263. https://doi.org/10.1080/07350015.1995.10524599.

[7] Stitsyuk, A., and Choi, J. (2025). xPatch: Dual-Stream Time Series Forecasting with Exponential Seasonal-Trend Decomposition. *Proceedings of the AAAI Conference on Artificial Intelligence*, 39(19), 20601–20609. https://doi.org/10.1609/aaai.v39i19.34270.

[8] Ansari, A. F., Shchur, O., Küken, J., et al. (2025). Chronos-2: From Univariate to Universal Forecasting. arXiv:2510.15821. https://arxiv.org/abs/2510.15821. The exact model revision used here is recorded separately in the weight lock.

[9] Hoerl, A. E., and Kennard, R. W. (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems. *Technometrics*, 12(1), 55–67. https://doi.org/10.1080/00401706.1970.10488634.

[10] Ke, G., Meng, Q., Finley, T., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems*, 30. Author-institution copy: https://www.microsoft.com/en-us/research/wp-content/uploads/2017/11/lightgbm.pdf.

[11] Newey, W. K., and West, K. D. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica*, 55(3), 703–708. https://doi.org/10.2307/1913610.

[12] Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70. https://www.jstor.org/stable/4615733.

## Editorial notes for the submission version

The reference list is a working bibliography. For [11], the retrieved Crossref record confirms the opening page but not the final page; the original pagination remains to be checked. The publisher page for [12] returned an access-challenge page, so its conventional citation has not yet been verified against the original. Some other classic references have metadata verification without full-text reading. `SOURCES.md` distinguishes these cases from the original passages actually read. The supporting literature review and availability of a public reproducibility package require further work before submission. These editorial notes should be resolved or moved out of the submitted article; unresolved scientific limitations must remain in the article.
