# Supplement to Multi-Horizon Forecasting of Apple Stock Prices and Returns Across Temporal Scales

Wang Yang. Cumulative post-observation supplement: preceding v3a/v3b stages and the completed shared-horizon stage; exposed historical development, 2026Q3 unscored.

Cumulative-stage note for this integrated edition: S1–S18 reproduce the preceding reviewed supplement verbatim. Statements such as “current” and “preceding” within those sections retain their original stage reference. S19–S21 append the later local-probe nonqualification, completed fixed-volume screen and subsequent primary-text review. These later stages neither replace preceding results nor provide independent market confirmation. The source changelog records the new text and preserved identities.

## S1. Scope, chronology and accounting

The original title, five temporal scales, twenty tasks and minute-H60/hourly-H8 return criteria remain fixed. This supplement follows inspection of v3, whose complete text, tables, factorial experiments and acceptance records remain unchanged in the preceding archive. The updated main Table 1 replaces only six long-scale single-seed entries with the same recipes at three seeds. Fourteen entries retain the earlier replication depth. Cross-task losses are never pooled across price/return units.

The long-scale matrix has six tasks, two initialization recipes, five folds and three seeds: 180 cells, including thirty exact reused v3 configurations and 150 new fits. All receive the original raw and inner-stop-shrunk output rules. The unconditional mean study has twenty tasks and one hundred purged folds. The history probe uses ninety baseline-initialized checkpoints and three fixed-weight input conditions per checkpoint. These are different stage counts, not independent statistical samples.

For the preceding completed CPU stages described in S1–S14, the original deadline was epoch 1794205909.6228158 (2026-11-09 14:31:49.623 Beijing), with at most eight two-core training processes and shared analysis CPUs 24 and 26. Their no-GPU statement applies to those earlier stages only. The current separate shared-horizon stage in S15–S18 uses the subsequently authorized local RTX 5070 Ti Laptop GPU and extended absolute deadline of 2026-11-09 17:31:49 Beijing. No paid resources or protected-period reads are authorized in either stage. Checkpoints preserve optimizer/RNG state where training can resume; the actual two-update long-scale exit/resume has a passed recovery receipt.

## S2. Six long-scale paired replications

Each row reports the mean of per-origin losses across seeds 2026/2027/2028, followed by pooling origins. It is not the loss of an averaged forecast. Positive folds compare this mean seed loss with zero/persistence. Default and repaired recipes differ only in baseline-function initialization within each target-specific implementation.

| Task | Default | Default shrink | Repair | Repair shrink | Positive folds |
| --- | --- | --- | --- | --- | --- |
| Annual H12 price | -79.642% | -31.557% | +33.667% | +33.731% | 4/5 |
| Annual H12 return | -39.405% | +23.776% | +38.019% | +38.105% | 5/5 |
| Quarterly H3 price | -967.809% | -18.330% | +1.105% | +1.226% | 2/5 |
| Quarterly H3 return | -613.495% | -10.794% | +0.653% | +0.698% | 1/5 |
| Monthly H21 price | -203.029% | -4.116% | +0.658% | +0.063% | 1/5 |
| Monthly H21 return | -13.683% | -10.988% | +0.799% | +0.741% | 4/5 |

## S3. Seed and strong-control comparisons

| Task | Seed 2026 | Seed 2027 | Seed 2028 | Fit mean | Mean shrink |
| --- | --- | --- | --- | --- | --- |
| Annual H12 price | +36.689% | +29.670% | +34.834% | +9.597% | +23.249% |
| Annual H12 return | +42.031% | +34.384% | +37.901% | +11.893% | +32.804% |
| Quarterly H3 price | +0.989% | +1.550% | +1.138% | +1.434% | +0.761% |
| Quarterly H3 return | +0.528% | +1.136% | +0.431% | +1.784% | +0.145% |
| Monthly H21 price | +0.129% | +0.068% | -0.009% | -0.286% | -0.335% |
| Monthly H21 return | +0.650% | +0.762% | +0.812% | -0.840% | -0.940% |

We retain Ridge with and without the nominal-price anchor, their original inner selections, both default outputs and both mean controls in the saved paired losses and the complete POOLED.csv. The following paired bounds compare the repaired-shrunk forecast with the shrunk mean control. Negative bounds permit the mean control to have lower error. The annual intervals do not establish superiority to this simple control. They are nominal descriptive intervals after exposure and multiple development decisions.

| Task | Block | Relative gain CI | Supports gain |
| --- | --- | --- | --- |
| Annual H12 price | 12 | [-57.21, +22.76]% | False |
| Annual H12 price | 24 | [-33.65, +19.70]% | False |
| Annual H12 return | 12 | [-133.48, +34.53]% | False |
| Annual H12 return | 24 | [-72.94, +31.65]% | False |
| Quarterly H3 price | 3 | [-0.03, +1.28]% | False |
| Quarterly H3 price | 6 | [+0.07, +1.03]% | True |
| Quarterly H3 return | 3 | [+0.15, +1.18]% | True |
| Quarterly H3 return | 6 | [+0.18, +1.04]% | True |
| Monthly H21 price | 21 | [-7.48, +1.88]% | False |
| Monthly H21 price | 42 | [-9.17, +2.49]% | False |
| Monthly H21 return | 21 | [-5.53, +3.65]% | False |
| Monthly H21 return | 42 | [-6.93, +4.37]% | False |

## S4. Unconditional controls across all twenty tasks

Mean return paths average inner-fit cumulative-log-return targets. Mean price paths average inner-fit dollar changes and add each actual current-price anchor. Neither control reads the historical window beyond that anchor. The raw control uses the full fit mean; the shrunk control estimates the same scalar clipped to [0,1] from the purged inner-stop endpoints and applies it across the path. Outer targets never fit a mean or scalar.

| Task | Origins | Mean raw | Mean shrink | Repair shrink |
| --- | --- | --- | --- | --- |
| Annual H12 price | 47 | +9.597% | +23.249% | +33.731% |
| Annual H12 return | 47 | +11.893% | +32.804% | +38.105% |
| Hourly H1 price | 6782 | +0.019% | -0.001% | -0.209% |
| Hourly H1 return | 6782 | +0.016% | -0.003% | -0.796% |
| Hourly H4 price | 6779 | +0.074% | -0.005% | -0.061% |
| Hourly H4 return | 6779 | +0.062% | -0.013% | -0.517% |
| Hourly H8 price | 6781 | +0.148% | -0.003% | -1.636% |
| Hourly H8 return | 6781 | +0.133% | -0.016% | -0.795% |
| Minute H15 price | 428505 | +0.001% | -0.001% | -0.035% |
| Minute H15 return | 428505 | -0.000% | -0.002% | +0.003% |
| Minute H30 price | 411300 | +0.001% | -0.001% | -0.138% |
| Minute H30 return | 411300 | +0.000% | -0.002% | -0.339% |
| Minute H5 price | 439975 | +0.000% | +0.000% | -0.011% |
| Minute H5 return | 439975 | +0.000% | +0.000% | -0.010% |
| Minute H60 price | 376890 | -0.007% | -0.000% | -0.066% |
| Minute H60 return | 376890 | -0.011% | -0.002% | -0.038% |
| Monthly H21 price | 1970 | -0.286% | -0.335% | +0.063% |
| Monthly H21 return | 1970 | -0.840% | -0.940% | +0.741% |
| Quarterly H3 price | 52 | +1.434% | +0.761% | +1.226% |
| Quarterly H3 return | 52 | +1.784% | +0.145% | +0.698% |

The shrunk mean controls fail to beat zero on minute H60 and hourly H8. The raw hourly-H8 mean has a small positive gain, below the original 1% threshold; neither form meets the original primary criteria. Adding these controls after v3 does not rewrite those criteria or create independent validation.

## S5. Fixed-weight history sensitivity

The actual-past condition replays every original calibrated path exactly. Fit-mean history averages only the raw log-relative inner-fit input windows; flat history sets that raw input to zero before the unchanged input scaler. Both retain the real current anchor, checkpoint, target scaler and original stopping alpha. No input substitution causes refitting or recalibration. Twenty-seven of ninety configurations have alpha zero; their calibrated outputs conceal any underlying network sensitivity. These potentially out-of-distribution substitutions cannot establish what an independently refitted no-history architecture would achieve.

| Task | Actual past | Fit-mean past | Flat past |
| --- | --- | --- | --- |
| Annual H12 price | +33.731% | +2.814% | +49.756% |
| Annual H12 return | +38.105% | +14.826% | +49.176% |
| Quarterly H3 price | +1.226% | +1.086% | +1.196% |
| Quarterly H3 return | +0.698% | +0.596% | +0.631% |
| Monthly H21 price | +0.063% | +0.025% | +0.285% |
| Monthly H21 return | +0.741% | +0.227% | +0.595% |

The two substitutions have different annual effects. The actual-versus-flat paired error contrasts favor the flat substitution at both frozen block lengths, but we do not promote a function selected after inspecting outer scores. These observations narrow the supported explanation to the implemented/calibrated functions.

## S6. Nonoverlap, phase, year and corporate-action sensitivity

| Task | Paths | Repair shrink | Mean shrink |
| --- | --- | --- | --- |
| Annual H12 price | 4 | +17.585% | +48.349% |
| Annual H12 return | 4 | +7.470% | +51.317% |
| Quarterly H3 price | 18 | +1.471% | +1.403% |
| Quarterly H3 return | 18 | +1.000% | +0.707% |
| Monthly H21 price | 94 | +0.149% | -0.220% |
| Monthly H21 return | 94 | +0.842% | -0.726% |

Annual greedy nonoverlap leaves four paths. The absolute-bar phase subsets use the original origin index modulo the horizon, without optimizing a phase. Their common historical market exposure prevents independent-sample interpretation. Year exclusions remove origins by their origin calendar year; they do not remove all labels touching that year. We report ranges as descriptive dependence checks only.

| Task | Subset rule | Subsets | Origins range | Repair gain range |
| --- | --- | --- | --- | --- |
| Annual H12 price | absolute bar phase | 12 | 3–4 | +1.454% to +56.071% |
| Annual H12 price | leave origin year out | 5 | 35–42 | +26.124% to +40.229% |
| Annual H12 return | absolute bar phase | 12 | 3–4 | +7.470% to +66.467% |
| Annual H12 return | leave origin year out | 5 | 35–42 | +34.404% to +43.796% |
| Quarterly H3 price | absolute bar phase | 3 | 17–18 | +0.678% to +1.582% |
| Quarterly H3 price | leave origin year out | 6 | 40–50 | -0.595% to +2.132% |
| Quarterly H3 return | absolute bar phase | 3 | 17–18 | +0.110% to +1.192% |
| Quarterly H3 return | leave origin year out | 6 | 40–50 | -1.115% to +1.968% |
| Monthly H21 price | absolute bar phase | 21 | 93–94 | -0.214% to +0.326% |
| Monthly H21 price | leave origin year out | 9 | 1717–1869 | -0.836% to +0.334% |
| Monthly H21 return | absolute bar phase | 21 | 93–94 | +0.458% to +1.068% |
| Monthly H21 return | leave origin year out | 9 | 1717–1869 | +0.280% to +1.188% |

Annual and quarterly evaluation labels cross no reviewed future split events. Their split-exclusion robustness contrast is therefore not estimable; identical all/no-split scores do not count as a successful event-exclusion test. Monthly labels cross one split event on 2020-08-31 through twenty-one overlapping paths. Future nominal-price discontinuities and dividends remain as defined in the original targets.

## S7. BatchNorm follow-up and numerical provenance

The fixed-weight normalization study retained seventeen checkpoints: fifteen minute-H60 cells at five folds and three seeds after inner-only admission, plus the two hourly-H8 screening folds at seed 2026 that failed admission. We replace only the running mean and variance of three BatchNorm layers, sequentially using upstream updated buffers and exact pooled inner-fit activation moments. Network parameters and other buffers remain unchanged. Float64 Welford accumulators and the current fit-input cursor support recovery. The actual interrupted/resumed accumulation also matches an independent pass from cursor zero.

| Task / scope | Repair raw | Adapted raw | Repair shrink | Adapted shrink |
| --- | --- | --- | --- | --- |
| Minute H60 return / 5 folds, 3 seeds | -17.765% | -16.653% | -0.038% | -0.044% |
| Hourly H8 return / 2 folds, 1 seed only | -2.776% | -3.276% | -2.513% | -2.847% |

Minute raw error decreases by 0.943571% relative to the same baseline-initialized raw checkpoint, with descriptive one/five-day paired intervals supporting this reduction. It still loses to zero and to calibrated strong controls. Adapted-shrunk pooled loss is slightly higher than the original shrunk repair; the corresponding relative-change intervals include zero. The hourly screen is not a full-task replication.

| Task | Over 1% | 3 seeds | 4 folds | All controls | Pass |
| --- | --- | --- | --- | --- | --- |
| Minute BN | False | False | False | False | False |
| Minute BN shrink | False | False | False | False | False |

| Seed | BN raw | BN shrink |
| --- | --- | --- |
| 2026 | -18.752% | -0.045% |
| 2027 | -13.487% | -0.048% |
| 2028 | -17.722% | -0.039% |

The all-control audit restores the original default and baseline-initialized raw/shrunk forecasts, zero, all four same-information Ridge variants and raw/shrunk mean paths. It recomputes per-origin mean seed losses, seed directions, fold directions and all 44 nominal paired intervals at one/five-day block lengths. No performance gate passes. The archive retains the control identities and losses.

Interpreting this result requires its boundary: training used batch statistics to optimize the fixed weights. Sequential replacement of running moments can change inference error without removing every training/inference discrepancy or demonstrating useful return signal. We do not add a retraining, architecture or economic-mechanism claim.

The first implementation required cross-batch allclose rtol 1e-6 and atol 1e-9. Before any adapter outer scoring, a screen failed this tolerance and the corresponding original checkpoint also failed it. A documented versioned revision changes only the return absolute tolerance to 1e-8, retaining rtol 1e-6; original and adapted checkpoints must both pass. We preserve the original source, freeze and failure receipt. Canonical batch-128 metrics must replay exactly, so this portability tolerance does not authorize changing a gate or saved score.

## S8. Restoration, replay and limits of acceptance

All thirty seed-2026 default restorations recover the original weights and selected/stopping epochs exactly. Twenty annual/quarterly paths match the original predictions bitwise. Ten monthly paths differ by small float32 batch-shape rounding because old inference used batch 512 and the frozen follow-up engine uses 128; rerunning the archived batch 512 reproduces the old predictions exactly. The frozen analysis initially demanded cross-batch bitwise equality and stopped. The retained diagnostic, analysis repair and preceding source hash explain that assertion correction; targets, training, statistics and gates did not change. We do not claim monthly cross-batch bitwise identity.

Independent checks re-evaluate all 180 long-scale predictors, 180 stopping alphas, 360 seed/mode metric rows, 66 pooled rows and 120 paired intervals. Separate mean-control replay checks the one hundred folds. The history audit independently replays ninety checkpoints under three input conditions, re-sums fit inputs in a different batch order, and checks all 24 pooled conditions and 24 paired intervals. These checks concern computation and source/data boundaries. They provide no independent market history.

## S9. Research status and remaining requirements

The original primary repaired return gains remain −0.0380704% at minute H60 and −0.7950608% at hourly H8. No result in this supplement substitutes annual performance for the original designated task criteria. Improved default implementations, successful replay and editable paper delivery do not establish the requested reliable better forecasting method.

The separate matched four-head shared-versus-independent horizon market experiment is now complete and reported in S15–S18 below. Eligible independent confirmation also remains unfinished: it needs an unexposed period, complete target labels, a frozen method/control set and a rule fixed before scoring. A single held-out quarter cannot confirm twelve-month outcomes. Protected 2026Q3 remains unscored. We claim no trading profitability, external submission, DOI or public deposit. The research Goal remains active and unachieved.

## S10. Reproduction

The addendum retains supplemental source freezes, rules, checkpoint identities, results, paired losses and inference examples. Its manifest identifies the immutable preceding v3 archive by hash. The README states which saved-function/statistical entrypoints were actually executed after extraction, which sources require registered-data path remapping for retraining, and which checks remain pending. The unchanged v3 supplement remains in manuscript/preceding_version for the original factorial/control tables and references.


## S11. Fixed conditional-context hypothesis and chronology

This development follow-up was designed after v3/v3a. It asks whether origin-known clock and weekday information adds conditional-mean value to one compressed-history representation. Timing or overnight exposure can affect volatility without making conditional expected returns predictable. We do not claim a novel Ridge architecture, a repair of xPatch internals, or an isolated test of adding context to the original full-history input.

All forty fixed cells were admitted before new stop/outer scoring, after a fit-only timing probe. The sole candidate is D_joint_shrink. Raw D is diagnostic. Failed first-two-fold raw-D comparisons against both zero and C are retained; all four task-fold directions failed. The full five-fold matrix cannot retrospectively pass that qualification. No raw/shrunk alternate or A/B/C model is promoted after evaluation.

The original absolute compute deadline and two-core analysis affinity apply. Protocol files freeze the plan, sources, original partition-array hashes, separate NPZ file hashes, scaler and target identities. An initial new assertion incorrectly compared the array-hash dictionary to a file-hash string and rejected the first freeze attempt before any stop/outer scoring. All ten partitions matched the original release byte-for-byte; the assertion was repaired and independently tested with a synthetic tamper-rejection fixture before formal freeze. No market data were changed.

## S12. Exact inputs, objective and deterministic acceptance

The local origin-bar completion clock is converted to America/New_York. With u=(clock minutes-570)/390, features contain sin(2 pi u), cos(2 pi u), sin(4 pi u), cos(4 pi u), and Monday-through-Thursday dummies with Friday as reference. The denominator is always 390, including half days. No observed future session length, future date, remaining bars or label endpoint enters the predictor. The cost-only probe used a different dummy reference; only the formal Friday-reference encoding determines these models.

Past features use cumulative log returns at lags 1/15/30/60/120/240/479 for minute H60, or 1/2/4/8/16/32/95 for hourly H8, plus RMS adjacent log returns over H and 4H. Original split-aware completed CloseWindows supply histories; all feature indices are at or before the origin. Fit labels finish before the stop origins and stop labels finish before outer origins.

A uses an unpenalized fit target-mean intercept. B, C and D add eight context features, nine history features, or all seventeen. Fit-only population means and standard deviations transform each column. A constant feature uses scale one. The normalized solve is (X'X/n+0.1I) inverse times X'(Y-mean Y)/n, with no search. The original complete cumulative-log-return path is the target. Each cell applies its own clipped [0,1] scalar estimated only from inner-stop endpoint predictions to its entire path.

The solver is deterministic, so training seeds are N/A. The translated acceptance rule was fixed before scoring: at least 1% improvement against zero; at least four positive folds; and positive paired difference intervals against every prelisted control at both task-specific block lengths, including each existing stochastic seed. Both designated tasks are required for broad success, and both inner direction and performance criteria are required for confirmation eligibility. The earlier neural three-seed gate remains unchanged.

## S13. Complete context matrix and paired results

Endpoint-MSE improvement against zero, pooling each origin once. Positive means lower error. B, C and D are explicitly matched to each other rather than called the original full-history architectures.

| Method | Minute H60 | Hourly H8 |
| --- | --- | --- |
| A mean raw | -0.011% | +0.133% |
| A mean shrink | -0.002% | -0.016% |
| B context raw | -0.169% | -0.064% |
| B context shrink | +0.000% | +0.031% |
| C history raw | -0.289% | -1.093% |
| C history shrink | -0.117% | -0.779% |
| D joint raw | -0.489% | -1.282% |
| D joint shrink | -0.222% | -0.684% |

The fixed candidate's positive-fold counts are 1/5 and 2/5. Both inner qualification and complete performance gates fail. All 31 controls, their two block intervals per task, all seed-specific controls, and all negative/degenerate intervals are preserved in STATISTICS.json and the companion CSV files. No multiplicity correction is claimed. The interval resamples moving trading-day blocks within each original fold, then pools by origin count; minute blocks are one/five days and hourly blocks are two/five days, with 5,000 draws and seed 2026.

Selected fixed-candidate contrasts are shown below; the package preserves the full control set. The percentage is relative to the listed reference, not necessarily to zero.

| Task | Reference | Block | Gain | 95% interval |
| --- | --- | --- | --- | --- |
| Minute H60 | zero | 1 | -0.222% | [-0.462, +0.001]% |
| Minute H60 | zero | 5 | -0.222% | [-0.489, -0.013]% |
| Minute H60 | C history shrink | 1 | -0.105% | [-0.264, +0.057]% |
| Minute H60 | C history shrink | 5 | -0.105% | [-0.256, +0.061]% |
| Minute H60 | repair three seed shrink | 1 | -0.183% | [-0.549, +0.152]% |
| Minute H60 | repair three seed shrink | 5 | -0.183% | [-0.564, +0.121]% |
| Minute H60 | ridge history selected | 1 | -0.184% | [-0.667, +0.277]% |
| Minute H60 | ridge history selected | 5 | -0.184% | [-0.563, +0.138]% |
| Hourly H8 | zero | 2 | -0.684% | [-1.625, +0.220]% |
| Hourly H8 | zero | 5 | -0.684% | [-1.583, +0.148]% |
| Hourly H8 | C history shrink | 2 | +0.095% | [-0.264, +0.444]% |
| Hourly H8 | C history shrink | 5 | +0.095% | [-0.266, +0.405]% |
| Hourly H8 | repair three seed shrink | 2 | +0.111% | [-1.275, +1.521]% |
| Hourly H8 | repair three seed shrink | 5 | +0.111% | [-1.190, +1.487]% |
| Hourly H8 | ridge history selected | 2 | -0.324% | [-1.355, +0.708]% |
| Hourly H8 | ridge history selected | 5 | -0.324% | [-1.271, +0.701]% |

Raw diagnostic comparisons evaluate context beyond history (D/C), history beyond context (D/B), and context beyond the unconditional mean (B/A). Relative improvement uses the second method as reference. A positive descriptive interval describes only this representation on exposed historical data.

| Task | Comparison | Block | Gain | 95% interval |
| --- | --- | --- | --- | --- |
| Minute H60 | D joint / C history | 1 | -0.199% | [-0.366, -0.034]% |
| Minute H60 | D joint / B context | 1 | -0.319% | [-0.692, +0.064]% |
| Minute H60 | B context / A mean | 1 | -0.158% | [-0.286, -0.037]% |
| Minute H60 | D joint / C history | 5 | -0.199% | [-0.369, -0.020]% |
| Minute H60 | D joint / B context | 5 | -0.319% | [-0.664, +0.047]% |
| Minute H60 | B context / A mean | 5 | -0.158% | [-0.283, -0.014]% |
| Hourly H8 | D joint / C history | 2 | -0.187% | [-0.725, +0.329]% |
| Hourly H8 | D joint / B context | 2 | -1.218% | [-2.182, -0.356]% |
| Hourly H8 | B context / A mean | 2 | -0.197% | [-0.709, +0.317]% |
| Hourly H8 | D joint / C history | 5 | -0.187% | [-0.756, +0.322]% |
| Hourly H8 | D joint / B context | 5 | -1.218% | [-2.228, -0.382]% |
| Hourly H8 | B context / A mean | 5 | -0.197% | [-0.738, +0.344]% |

All inner-only scalars are retained, including exact zero values that suppress raw outputs.

| Task | Fold | A alpha | B alpha | C alpha | D alpha |
| --- | --- | --- | --- | --- | --- |
| Minute H60 | 1 | 1.000000 | 0.000000 | 0.417023 | 0.000000 |
| Minute H60 | 2 | 1.000000 | 0.000000 | 0.079156 | 0.110776 |
| Minute H60 | 3 | 0.179722 | 0.000000 | 1.000000 | 0.124470 |
| Minute H60 | 4 | 0.174807 | 0.000000 | 1.000000 | 1.000000 |
| Minute H60 | 5 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Hourly H8 | 1 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| Hourly H8 | 2 | 0.000000 | 0.730334 | 0.000000 | 0.000000 |
| Hourly H8 | 3 | 0.958715 | 0.000000 | 0.000000 | 0.000000 |
| Hourly H8 | 4 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| Hourly H8 | 5 | 0.000000 | 1.000000 | 0.000000 | 0.228741 |

## S14. Context replay and remaining research requirements

The real market fit process exited after 4,096 origins and resumed from the stored cursor. Prefix and complete sufficient statistics, and all four solutions, match an uninterrupted run bit-for-bit at the same batch order. This verifies recoverability, not forecasting skill. An independent reviewer reconstructs all forty models, inner scalars and complete outer paths with a different accumulation order, then checks source controls and block statistics. The separate portable entrypoints reconstruct predictions from input-only examples and statistics from saved per-origin endpoint/path losses with complete day grids. These entrypoints need no original workspace or target series. Full retraining still needs the identified original sources and environment.

The mechanism-and-improvement research Goal remains active. Initialization and normalization diagnostics provide bounded implementation evidence. This context recipe does not achieve the required method contribution; no eligible independent confirmation has been run. The full five-scale, twenty-task results and the original title are preserved, 2026Q3 remains sealed, and the matched four-head shared-versus-independent horizon market experiment is now complete, with its separate development decisions reported below. Predictive success, independent confirmation and manuscript/replay delivery remain separate acceptance items.


## S15. Genuine four-head sharing experiment and numerical implementation

### Matched shared and independent horizon functions

We add a controlled development study of whether sharing a nonlinear history representation changes minute-return forecasting. This study addresses a previously unexecuted shared-learning question; it does not introduce a new architecture. The horizons are 5, 15, 30 and 60 regular-session minutes. Each horizon has its own native xPatch decoder, trend branch and anchor head. The shared arm uses one nonlinear patch encoder, whereas the independent arm uses four copies. These are four native forecast functions; the shorter forecasts are not prefixes extracted from an H60 decoder.

Both arms use the same H60-legal origins, 480-bar histories, observed price anchors, purged partitions, fit-only scalers and target standardization. Within each fold and seed, corresponding functions start from identical encoder and decoder states. Output initialization implements the previously specified zero-return baseline. The independent arm starts from copies of the same encoder, rather than separately randomized encoders. Sharing changes the encoder gradients contributed by the four losses. The arm comparison therefore evaluates the complete parameter-tying intervention, including its gradient aggregation; it cannot distinguish a general representation-sharing effect from every consequence of that intervention. Total parameter counts are 472,549 and 524,797. We do not describe the comparison as equal capacity.

Training minimizes the sum of four mean squared full-path errors in standardized cumulative log returns. Each horizon contributes one mean, irrespective of its path length. Both arms use Adam with learning rate 0.001 and batch size 128, merging a one-sample remainder into the preceding batch and otherwise retaining the final-batch size. They run exactly twenty complete epochs with matched permutations and update counts. The matrix comprises five expanding folds, seeds 2026/2027/2028 and two arms: thirty trained bundles and 120 horizon readouts. This fixed-update comparison differs from the early-stopped historical architecture comparison.

We retain the final epoch and a selected epoch. Selection minimizes the average of the four standardized inner-stop endpoint MSEs among epochs 1–20, choosing one epoch for the entire bundle and the earliest epoch in a tie. The epoch-zero baseline is an external control, not a selectable epoch. It does not choose an epoch separately for each head. Each checkpoint supplies a raw path and a path multiplied by a scalar in [0,1], estimated separately by horizon on inner-stop endpoints. For each horizon and checkpoint, alpha = clip((pᵀy)/(pᵀp), 0, 1) uses inner-stop endpoint forecasts p and observed cumulative returns y, with alpha set to zero when pᵀp is zero. This minimizes inner-stop squared endpoint error for forecasts shrunk toward the zero-return baseline; the same scalar is applied to every step of that head. This yields sixteen arrays per bundle, or 480 complete forecast-path arrays. The mechanism comparison is the unshrunk final-epoch H60 shared forecast against its matched independent forecast. The only designated predictive candidate is the selected, shrunk shared forecast.

### Device and interruption controls

The authorized execution uses one local NVIDIA GeForce RTX 5070 Ti Laptop GPU, with one worker assigned to each arm. Both arms use deterministic FP32 execution. Automatic mixed precision, TF32 and convolution benchmarking are disabled. Fused Adam preserves the stated mathematical hyperparameters but can differ in rounding from a CPU or non-fused implementation. CUDA graphs capture forward and backward operations for full batches; finite checks and optimizer updates remain outside the graph, and irregular tail batches use eager execution. Graph warmup restores the pre-warmup model, BatchNorm and random states. We do not claim bitwise equality between complete CPU and GPU training trajectories or attribute cross-backend historical differences solely to sharing.

Before market dispatch, synthetic checks verified matched initial functions, native independent-arm equivalence, BatchNorm updates, graph/eager transitions, warmup rollback and actual process interruption followed by fresh-process recovery. Resumable latest checkpoints atomically preserve model and BatchNorm state, Adam, NumPy permutations, CPU/CUDA random states and the batch cursor; initial, selected and final checkpoints retain inference weights and identity metadata. Saving occurs every hundred updates and at epoch boundaries. The fixed authorization deadline is November 9, 2026, 17:31:49 Beijing time; interruption or timeout cannot reset it. These engineering checks establish execution and recovery behavior, not forecasting skill.

### Complete-matrix assessment and decision rules

New outer scoring requires all thirty completed bundles, identity-matched sources and checkpoints, matching paired update counts and the complete prediction matrix. The evaluator then freezes its code and input identities before scoring. It reports endpoint MSE, endpoint MAE and full-path MSE for every horizon, arm, seed and retained form. Averaging per-origin seed losses is distinguished from scoring the average forecast.

The primary H60 mechanism condition requires improvements for all three seeds, at least four of five paired folds, and positive lower bounds for independent-minus-shared final-epoch raw loss differences under both one-day and five-day moving-block intervals. The predictive candidate is assessed separately. All three seeds and at least four of five folds must improve on zero returns; the pooled endpoint-MSE improvement over zero, using mean seed losses, must be at least 1%. For every registered control entry, the lower bound of the reference-minus-candidate loss difference must be positive under both block lengths. The mechanism and predictive decisions are reported separately; one does not imply the other. The H60 control set has forty entries: zero returns, raw and inner-shrunk fit means, four independent-arm forms and thirty-three archived loss columns. The archived zero-loss column duplicates the current zero control exactly, and other entries may coincide when selection or shrinkage degenerates to zero. Forty is the registered entry count, not the number of distinct methods, independent discoveries or observations; no frozen control is removed. No new outer score can change the candidate, epochs, seeds, folds or thresholds.

Five thousand block-bootstrap draws use seed 2026, resampling consecutive trading-day blocks independently within each fold and weighting origins through resampled loss sums and counts. Each resampled block preserves its internal day order; concatenated blocks are truncated to the original fold day count and do not preserve the global chronological sequence. The nominal 95% intervals are descriptive for exposed development periods and are not independent or adjusted for the accumulated research search. All four horizons and negative findings must be reported. A minute-horizon result cannot establish the separate hourly predictive objective or cross-scale superiority. Protected 2026Q3 remains unscored; independent predictive confirmation is a separate, uncompleted requirement.



Detached replay additionally verifies GPU-generated synthetic checkpoints on CPU at fixed rtol 2e-5 and atol 5e-7; its observed maximum absolute difference was below 7.4e-10. GPU replay requires exact equality on the specified environment. Hardware and kernel changes can prevent bitwise cross-device equality. These numerical checks remain engineering evidence.

## S16. Separate mechanism and sole-candidate decisions

The mechanism compares raw final tied versus raw final untied. Positive means lower tied error. Each row reports three seed directions, fold directions and two nominal paired block intervals. Only minute H60 is primary.

| Horizon | Raw tied / untied gain | Positive seeds | Positive folds | Block1 interval | Block5 interval | Mechanism |
| --- | --- | --- | --- | --- | --- | --- |
| H5 | -30.667% | 0 | 0/5 | [-37.837, -24.629]% | [-41.588, -21.846]% | fail |
| H15 | -5.562% | 0 | 0/5 | [-7.159, -3.963]% | [-6.918, -4.452]% | fail |
| H30 | +1.318% | 2 | 4/5 | [-0.040, +2.732]% | [-0.097, +2.564]% | fail |
| H60 | +1.942% | 2 | 4/5 | [-0.100, +4.796]% | [-0.214, +4.867]% | fail |

The sole candidate is tied-selected-shrink. Its four required checks are kept separate below. H60 must satisfy both interval lengths against all forty frozen controls; supporting horizons use all seven applicable controls. A threshold or control failure cannot be replaced by a favorable point estimate. All preceding failed methods remain retained.

| Horizon | Gain over zero | 3 seeds | 4 folds | At least 1% | All controls / both blocks | Candidate |
| --- | --- | --- | --- | --- | --- | --- |
| H5 | -0.060% | False | False | False | False | fail |
| H15 | -0.072% | False | False | False | False | fail |
| H30 | -0.120% | False | False | False | False | fail |
| H60 | -0.048% | False | False | False | False | fail |

The matched raw final-checkpoint sharing contrast fails its predefined mechanism gate. The sole candidate fails the predefined minute-H60 development performance gate. Neither a repeatable raw sharing benefit nor a qualifying forecasting candidate is established by this fixed study. No result in this minute-only study completes the original hourly-H8 objective or constitutes independent market confirmation. Existing exposed history remains development data, including all new supporting-horizon results. Nominal bootstrap intervals are not multiplicity-adjusted discovery claims.

## S17. Complete four-horizon endpoint and paired-interval tables

All saved summary rows, individual seeds, prediction-ensemble diagnostics and legacy H60 controls are included below. The candidate gate uses mean seed losses, not averaged forecasts. All 20 fold receipts additionally retain endpoint MSE, path MSE and endpoint MAE for their full registered method sets; those values remain in the hashed analysis rather than being redefined here. A positive interval describes reference-minus-primary error on exposed history. Undefined or degenerate intervals do not pass the gate.

### S17.1 Minute H5 (supporting)

| Method | Origins | Endpoint MSE | Gain over zero | Positive folds |
| --- | --- | --- | --- | --- |
| zero | 376890 | 2.54104334e-06 | +0.000% | 0/5 |
| fit mean raw | 376890 | 2.54105893e-06 | -0.001% | 2/5 |
| fit mean shrink | 376890 | 2.54107483e-06 | -0.001% | 1/5 |
| tied last seed 2026 | 376890 | 4.10306966e-06 | -61.472% | 0/5 |
| tied last seed 2027 | 376890 | 3.84008577e-06 | -51.122% | 0/5 |
| tied last seed 2028 | 376890 | 4.20594023e-06 | -65.520% | 0/5 |
| tied last | 376890 | 4.04969855e-06 | -59.371% | 0/5 |
| tied last prediction ensemble | 376890 | 3.40671101e-06 | -34.067% | 0/5 |
| tied last shrink seed 2026 | 376890 | 2.54675245e-06 | -0.225% | 2/5 |
| tied last shrink seed 2027 | 376890 | 2.55356831e-06 | -0.493% | 2/5 |
| tied last shrink seed 2028 | 376890 | 2.54592596e-06 | -0.192% | 3/5 |
| tied last shrink | 376890 | 2.54874891e-06 | -0.303% | 2/5 |
| tied last shrink prediction ensemble | 376890 | 2.5445326e-06 | -0.137% | 3/5 |
| tied selected seed 2026 | 376890 | 2.6026456e-06 | -2.424% | 0/5 |
| tied selected seed 2027 | 376890 | 2.65699827e-06 | -4.563% | 0/5 |
| tied selected seed 2028 | 376890 | 2.62679949e-06 | -3.375% | 0/5 |
| tied selected | 376890 | 2.62881446e-06 | -3.454% | 0/5 |
| tied selected prediction ensemble | 376890 | 2.58942599e-06 | -1.904% | 1/5 |
| tied selected shrink seed 2026 | 376890 | 2.54645151e-06 | -0.213% | 2/5 |
| tied selected shrink seed 2027 | 376890 | 2.54144456e-06 | -0.016% | 2/5 |
| tied selected shrink seed 2028 | 376890 | 2.53978917e-06 | +0.049% | 2/5 |
| tied selected shrink | 376890 | 2.54256175e-06 | -0.060% | 2/5 |
| tied selected shrink prediction ensemble | 376890 | 2.54011094e-06 | +0.037% | 3/5 |
| untied last seed 2026 | 376890 | 3.0253207e-06 | -19.058% | 0/5 |
| untied last seed 2027 | 376890 | 3.15721689e-06 | -24.249% | 0/5 |
| untied last seed 2028 | 376890 | 3.11524502e-06 | -22.597% | 0/5 |
| untied last | 376890 | 3.09926087e-06 | -21.968% | 0/5 |
| untied last prediction ensemble | 376890 | 2.84954319e-06 | -12.141% | 0/5 |
| untied last shrink seed 2026 | 376890 | 2.54055535e-06 | +0.019% | 1/5 |
| untied last shrink seed 2027 | 376890 | 2.54071426e-06 | +0.013% | 2/5 |
| untied last shrink seed 2028 | 376890 | 2.54112234e-06 | -0.003% | 0/5 |
| untied last shrink | 376890 | 2.54079732e-06 | +0.010% | 2/5 |
| untied last shrink prediction ensemble | 376890 | 2.54056366e-06 | +0.019% | 3/5 |
| untied selected seed 2026 | 376890 | 2.54296952e-06 | -0.076% | 0/5 |
| untied selected seed 2027 | 376890 | 2.55998686e-06 | -0.746% | 1/5 |
| untied selected seed 2028 | 376890 | 2.54853481e-06 | -0.295% | 0/5 |
| untied selected | 376890 | 2.55049706e-06 | -0.372% | 0/5 |
| untied selected prediction ensemble | 376890 | 2.54527782e-06 | -0.167% | 0/5 |
| untied selected shrink seed 2026 | 376890 | 2.54136551e-06 | -0.013% | 0/5 |
| untied selected shrink seed 2027 | 376890 | 2.54105317e-06 | -0.000% | 1/5 |
| untied selected shrink seed 2028 | 376890 | 2.54108618e-06 | -0.002% | 1/5 |
| untied selected shrink | 376890 | 2.54116829e-06 | -0.005% | 0/5 |
| untied selected shrink prediction ensemble | 376890 | 2.54104678e-06 | -0.000% | 1/5 |

| Primary | Reference | Block days | Gain | 95% interval | Supports gain |
| --- | --- | --- | --- | --- | --- |
| tied last | untied last | 1 | -30.667% | [-37.837, -24.629]% | False |
| tied selected shrink | zero | 1 | -0.060% | [-0.152, +0.032]% | False |
| tied selected shrink | fit mean raw | 1 | -0.059% | [-0.151, +0.031]% | False |
| tied selected shrink | fit mean shrink | 1 | -0.059% | [-0.150, +0.032]% | False |
| tied selected shrink | untied last | 1 | +17.962% | [+15.899, +20.045]% | True |
| tied selected shrink | untied last shrink | 1 | -0.069% | [-0.161, +0.021]% | False |
| tied selected shrink | untied selected | 1 | +0.311% | [+0.169, +0.463]% | True |
| tied selected shrink | untied selected shrink | 1 | -0.055% | [-0.147, +0.037]% | False |
| tied last | untied last | 5 | -30.667% | [-41.588, -21.846]% | False |
| tied selected shrink | zero | 5 | -0.060% | [-0.168, +0.048]% | False |
| tied selected shrink | fit mean raw | 5 | -0.059% | [-0.168, +0.048]% | False |
| tied selected shrink | fit mean shrink | 5 | -0.059% | [-0.167, +0.049]% | False |
| tied selected shrink | untied last | 5 | +17.962% | [+14.576, +20.514]% | True |
| tied selected shrink | untied last shrink | 5 | -0.069% | [-0.179, +0.039]% | False |
| tied selected shrink | untied selected | 5 | +0.311% | [+0.134, +0.484]% | True |
| tied selected shrink | untied selected shrink | 5 | -0.055% | [-0.163, +0.051]% | False |
### S17.2 Minute H15 (supporting)

| Method | Origins | Endpoint MSE | Gain over zero | Positive folds |
| --- | --- | --- | --- | --- |
| zero | 376890 | 6.99385096e-06 | +0.000% | 0/5 |
| fit mean raw | 376890 | 6.99406688e-06 | -0.003% | 2/5 |
| fit mean shrink | 376890 | 6.99431859e-06 | -0.007% | 0/5 |
| tied last seed 2026 | 376890 | 1.03840258e-05 | -48.474% | 0/5 |
| tied last seed 2027 | 376890 | 1.02897047e-05 | -47.125% | 0/5 |
| tied last seed 2028 | 376890 | 1.06219286e-05 | -51.875% | 0/5 |
| tied last | 376890 | 1.04318864e-05 | -49.158% | 0/5 |
| tied last prediction ensemble | 376890 | 8.88721234e-06 | -27.072% | 0/5 |
| tied last shrink seed 2026 | 376890 | 6.99446227e-06 | -0.009% | 2/5 |
| tied last shrink seed 2027 | 376890 | 6.99701293e-06 | -0.045% | 1/5 |
| tied last shrink seed 2028 | 376890 | 6.99572639e-06 | -0.027% | 1/5 |
| tied last shrink | 376890 | 6.99573386e-06 | -0.027% | 1/5 |
| tied last shrink prediction ensemble | 376890 | 6.99159931e-06 | +0.032% | 3/5 |
| tied selected seed 2026 | 376890 | 7.91848283e-06 | -13.221% | 0/5 |
| tied selected seed 2027 | 376890 | 8.63027333e-06 | -23.398% | 0/5 |
| tied selected seed 2028 | 376890 | 8.11591658e-06 | -16.044% | 0/5 |
| tied selected | 376890 | 8.22155758e-06 | -17.554% | 0/5 |
| tied selected prediction ensemble | 376890 | 7.74355327e-06 | -10.719% | 0/5 |
| tied selected shrink seed 2026 | 376890 | 6.98907106e-06 | +0.068% | 3/5 |
| tied selected shrink seed 2027 | 376890 | 7.00151866e-06 | -0.110% | 1/5 |
| tied selected shrink seed 2028 | 376890 | 7.00612149e-06 | -0.175% | 2/5 |
| tied selected shrink | 376890 | 6.99890374e-06 | -0.072% | 3/5 |
| tied selected shrink prediction ensemble | 376890 | 6.99107934e-06 | +0.040% | 3/5 |
| untied last seed 2026 | 376890 | 8.983361e-06 | -28.447% | 0/5 |
| untied last seed 2027 | 376890 | 1.02043586e-05 | -45.905% | 0/5 |
| untied last seed 2028 | 376890 | 1.04590308e-05 | -49.546% | 0/5 |
| untied last | 376890 | 9.88225015e-06 | -41.299% | 0/5 |
| untied last prediction ensemble | 376890 | 8.4765763e-06 | -21.200% | 0/5 |
| untied last shrink seed 2026 | 376890 | 6.99272726e-06 | +0.016% | 2/5 |
| untied last shrink seed 2027 | 376890 | 7.01242731e-06 | -0.266% | 2/5 |
| untied last shrink seed 2028 | 376890 | 7.00140267e-06 | -0.108% | 2/5 |
| untied last shrink | 376890 | 7.00218575e-06 | -0.119% | 2/5 |
| untied last shrink prediction ensemble | 376890 | 6.99686606e-06 | -0.043% | 2/5 |
| untied selected seed 2026 | 376890 | 7.54237432e-06 | -7.843% | 0/5 |
| untied selected seed 2027 | 376890 | 8.07056847e-06 | -15.395% | 0/5 |
| untied selected seed 2028 | 376890 | 7.96870475e-06 | -13.939% | 0/5 |
| untied selected | 376890 | 7.86054918e-06 | -12.392% | 0/5 |
| untied selected prediction ensemble | 376890 | 7.50170439e-06 | -7.261% | 0/5 |
| untied selected shrink seed 2026 | 376890 | 6.99366346e-06 | +0.003% | 2/5 |
| untied selected shrink seed 2027 | 376890 | 6.99450196e-06 | -0.009% | 1/5 |
| untied selected shrink seed 2028 | 376890 | 7.01367303e-06 | -0.283% | 2/5 |
| untied selected shrink | 376890 | 7.00061282e-06 | -0.097% | 2/5 |
| untied selected shrink prediction ensemble | 376890 | 6.99474595e-06 | -0.013% | 3/5 |

| Primary | Reference | Block days | Gain | 95% interval | Supports gain |
| --- | --- | --- | --- | --- | --- |
| tied last | untied last | 1 | -5.562% | [-7.159, -3.963]% | False |
| tied selected shrink | zero | 1 | -0.072% | [-0.438, +0.245]% | False |
| tied selected shrink | fit mean raw | 1 | -0.069% | [-0.434, +0.252]% | False |
| tied selected shrink | fit mean shrink | 1 | -0.066% | [-0.433, +0.255]% | False |
| tied selected shrink | untied last | 1 | +29.177% | [+27.878, +30.526]% | True |
| tied selected shrink | untied last shrink | 1 | +0.047% | [-0.274, +0.366]% | False |
| tied selected shrink | untied selected | 1 | +10.962% | [+8.278, +13.823]% | True |
| tied selected shrink | untied selected shrink | 1 | +0.024% | [-0.225, +0.282]% | False |
| tied last | untied last | 5 | -5.562% | [-6.918, -4.452]% | False |
| tied selected shrink | zero | 5 | -0.072% | [-0.365, +0.117]% | False |
| tied selected shrink | fit mean raw | 5 | -0.069% | [-0.365, +0.124]% | False |
| tied selected shrink | fit mean shrink | 5 | -0.066% | [-0.362, +0.127]% | False |
| tied selected shrink | untied last | 5 | +29.177% | [+27.640, +30.333]% | True |
| tied selected shrink | untied last shrink | 5 | +0.047% | [-0.217, +0.251]% | False |
| tied selected shrink | untied selected | 5 | +10.962% | [+6.893, +13.606]% | True |
| tied selected shrink | untied selected shrink | 5 | +0.024% | [-0.165, +0.181]% | False |
### S17.3 Minute H30 (supporting)

| Method | Origins | Endpoint MSE | Gain over zero | Positive folds |
| --- | --- | --- | --- | --- |
| zero | 376890 | 1.30653798e-05 | +0.000% | 0/5 |
| fit mean raw | 376890 | 1.30657538e-05 | -0.003% | 2/5 |
| fit mean shrink | 376890 | 1.306664e-05 | -0.010% | 0/5 |
| tied last seed 2026 | 376890 | 1.87883449e-05 | -43.803% | 0/5 |
| tied last seed 2027 | 376890 | 1.87536283e-05 | -43.537% | 0/5 |
| tied last seed 2028 | 376890 | 1.91583166e-05 | -46.634% | 0/5 |
| tied last | 376890 | 1.89000966e-05 | -44.658% | 0/5 |
| tied last prediction ensemble | 376890 | 1.61000383e-05 | -23.227% | 0/5 |
| tied last shrink seed 2026 | 376890 | 1.30525422e-05 | +0.098% | 3/5 |
| tied last shrink seed 2027 | 376890 | 1.30723597e-05 | -0.053% | 2/5 |
| tied last shrink seed 2028 | 376890 | 1.305809e-05 | +0.056% | 2/5 |
| tied last shrink | 376890 | 1.30609973e-05 | +0.034% | 4/5 |
| tied last shrink prediction ensemble | 376890 | 1.30541805e-05 | +0.086% | 4/5 |
| tied selected seed 2026 | 376890 | 1.54065695e-05 | -17.919% | 0/5 |
| tied selected seed 2027 | 376890 | 1.72046821e-05 | -31.681% | 0/5 |
| tied selected seed 2028 | 376890 | 1.5160387e-05 | -16.035% | 0/5 |
| tied selected | 376890 | 1.59238795e-05 | -21.878% | 0/5 |
| tied selected prediction ensemble | 376890 | 1.47454402e-05 | -12.859% | 0/5 |
| tied selected shrink seed 2026 | 376890 | 1.3085504e-05 | -0.154% | 1/5 |
| tied selected shrink seed 2027 | 376890 | 1.30632854e-05 | +0.016% | 1/5 |
| tied selected shrink seed 2028 | 376890 | 1.30942761e-05 | -0.221% | 1/5 |
| tied selected shrink | 376890 | 1.30810218e-05 | -0.120% | 2/5 |
| tied selected shrink prediction ensemble | 376890 | 1.30579089e-05 | +0.057% | 3/5 |
| untied last seed 2026 | 376890 | 1.91485403e-05 | -46.559% | 0/5 |
| untied last seed 2027 | 376890 | 1.92366949e-05 | -47.234% | 0/5 |
| untied last seed 2028 | 376890 | 1.90722062e-05 | -45.975% | 0/5 |
| untied last | 376890 | 1.91524805e-05 | -46.590% | 0/5 |
| untied last prediction ensemble | 376890 | 1.62491887e-05 | -24.368% | 0/5 |
| untied last shrink seed 2026 | 376890 | 1.30642575e-05 | +0.009% | 3/5 |
| untied last shrink seed 2027 | 376890 | 1.30762371e-05 | -0.083% | 2/5 |
| untied last shrink seed 2028 | 376890 | 1.30648407e-05 | +0.004% | 2/5 |
| untied last shrink | 376890 | 1.30684451e-05 | -0.023% | 2/5 |
| untied last shrink prediction ensemble | 376890 | 1.30613791e-05 | +0.031% | 3/5 |
| untied selected seed 2026 | 376890 | 1.48823921e-05 | -13.907% | 0/5 |
| untied selected seed 2027 | 376890 | 1.74956602e-05 | -33.909% | 0/5 |
| untied selected seed 2028 | 376890 | 1.52074303e-05 | -16.395% | 0/5 |
| untied selected | 376890 | 1.58618275e-05 | -21.403% | 0/5 |
| untied selected prediction ensemble | 376890 | 1.45748505e-05 | -11.553% | 0/5 |
| untied selected shrink seed 2026 | 376890 | 1.30616373e-05 | +0.029% | 2/5 |
| untied selected shrink seed 2027 | 376890 | 1.30660027e-05 | -0.005% | 1/5 |
| untied selected shrink seed 2028 | 376890 | 1.30488058e-05 | +0.127% | 4/5 |
| untied selected shrink | 376890 | 1.30588153e-05 | +0.050% | 3/5 |
| untied selected shrink prediction ensemble | 376890 | 1.30490538e-05 | +0.125% | 4/5 |

| Primary | Reference | Block days | Gain | 95% interval | Supports gain |
| --- | --- | --- | --- | --- | --- |
| tied last | untied last | 1 | +1.318% | [-0.040, +2.732]% | False |
| tied selected shrink | zero | 1 | -0.120% | [-0.384, +0.151]% | False |
| tied selected shrink | fit mean raw | 1 | -0.117% | [-0.390, +0.167]% | False |
| tied selected shrink | fit mean shrink | 1 | -0.110% | [-0.382, +0.172]% | False |
| tied selected shrink | untied last | 1 | +31.701% | [+29.830, +33.617]% | True |
| tied selected shrink | untied last shrink | 1 | -0.096% | [-0.344, +0.159]% | False |
| tied selected shrink | untied selected | 1 | +17.531% | [+13.566, +21.735]% | True |
| tied selected shrink | untied selected shrink | 1 | -0.170% | [-0.458, +0.064]% | False |
| tied last | untied last | 5 | +1.318% | [-0.097, +2.564]% | False |
| tied selected shrink | zero | 5 | -0.120% | [-0.355, +0.077]% | False |
| tied selected shrink | fit mean raw | 5 | -0.117% | [-0.366, +0.081]% | False |
| tied selected shrink | fit mean shrink | 5 | -0.110% | [-0.357, +0.089]% | False |
| tied selected shrink | untied last | 5 | +31.701% | [+30.022, +33.075]% | True |
| tied selected shrink | untied last shrink | 5 | -0.096% | [-0.331, +0.091]% | False |
| tied selected shrink | untied selected | 5 | +17.531% | [+11.123, +22.295]% | True |
| tied selected shrink | untied selected shrink | 5 | -0.170% | [-0.451, +0.060]% | False |
### S17.4 Minute H60 (primary)

| Method | Origins | Endpoint MSE | Gain over zero | Positive folds |
| --- | --- | --- | --- | --- |
| zero | 376890 | 2.40572794e-05 | +0.000% | 0/5 |
| fit mean raw | 376890 | 2.40599024e-05 | -0.011% | 2/5 |
| fit mean shrink | 376890 | 2.40577117e-05 | -0.002% | 2/5 |
| tied last seed 2026 | 376890 | 3.34307586e-05 | -38.963% | 0/5 |
| tied last seed 2027 | 376890 | 3.40957912e-05 | -41.728% | 0/5 |
| tied last seed 2028 | 376890 | 3.45758331e-05 | -43.723% | 0/5 |
| tied last | 376890 | 3.40341277e-05 | -41.471% | 0/5 |
| tied last prediction ensemble | 376890 | 2.91950723e-05 | -21.357% | 0/5 |
| tied last shrink seed 2026 | 376890 | 2.40592919e-05 | -0.008% | 3/5 |
| tied last shrink seed 2027 | 376890 | 2.40780686e-05 | -0.086% | 1/5 |
| tied last shrink seed 2028 | 376890 | 2.40771502e-05 | -0.083% | 1/5 |
| tied last shrink | 376890 | 2.40715036e-05 | -0.059% | 2/5 |
| tied last shrink prediction ensemble | 376890 | 2.40632089e-05 | -0.025% | 3/5 |
| tied selected seed 2026 | 376890 | 3.04534247e-05 | -26.587% | 0/5 |
| tied selected seed 2027 | 376890 | 2.99747362e-05 | -24.597% | 0/5 |
| tied selected seed 2028 | 376890 | 3.03019432e-05 | -25.957% | 0/5 |
| tied selected | 376890 | 3.02433681e-05 | -25.714% | 0/5 |
| tied selected prediction ensemble | 376890 | 2.75360713e-05 | -14.460% | 0/5 |
| tied selected shrink seed 2026 | 376890 | 2.40709342e-05 | -0.057% | 2/5 |
| tied selected shrink seed 2027 | 376890 | 2.40858264e-05 | -0.119% | 0/5 |
| tied selected shrink seed 2028 | 376890 | 2.40498879e-05 | +0.031% | 2/5 |
| tied selected shrink | 376890 | 2.40688829e-05 | -0.048% | 2/5 |
| tied selected shrink prediction ensemble | 376890 | 2.4060452e-05 | -0.013% | 3/5 |
| untied last seed 2026 | 376890 | 3.39537131e-05 | -41.137% | 0/5 |
| untied last seed 2027 | 376890 | 3.6049384e-05 | -49.848% | 0/5 |
| untied last seed 2028 | 376890 | 3.4121023e-05 | -41.832% | 0/5 |
| untied last | 376890 | 3.470804e-05 | -44.273% | 0/5 |
| untied last prediction ensemble | 376890 | 2.97084337e-05 | -23.490% | 0/5 |
| untied last shrink seed 2026 | 376890 | 2.4057333e-05 | -0.000% | 2/5 |
| untied last shrink seed 2027 | 376890 | 2.4096016e-05 | -0.161% | 0/5 |
| untied last shrink seed 2028 | 376890 | 2.40667392e-05 | -0.039% | 2/5 |
| untied last shrink | 376890 | 2.40733628e-05 | -0.067% | 1/5 |
| untied last shrink prediction ensemble | 376890 | 2.40616007e-05 | -0.018% | 1/5 |
| untied selected seed 2026 | 376890 | 2.83561874e-05 | -17.869% | 0/5 |
| untied selected seed 2027 | 376890 | 2.89577909e-05 | -20.370% | 0/5 |
| untied selected seed 2028 | 376890 | 2.8976265e-05 | -20.447% | 0/5 |
| untied selected | 376890 | 2.87634145e-05 | -19.562% | 0/5 |
| untied selected prediction ensemble | 376890 | 2.70634142e-05 | -12.496% | 0/5 |
| untied selected shrink seed 2026 | 376890 | 2.40628953e-05 | -0.023% | 2/5 |
| untied selected shrink seed 2027 | 376890 | 2.4066859e-05 | -0.040% | 2/5 |
| untied selected shrink seed 2028 | 376890 | 2.40566591e-05 | +0.003% | 1/5 |
| untied selected shrink | 376890 | 2.40621378e-05 | -0.020% | 2/5 |
| untied selected shrink prediction ensemble | 376890 | 2.40488942e-05 | +0.035% | 4/5 |
| legacy v3b zero | 376890 | 2.40572794e-05 | +0.000% | 0/5 |
| legacy v3b archived default raw | 376890 | 3.10097433e-05 | -28.900% | 0/5 |
| legacy v3b archived default shrink | 376890 | 2.41178582e-05 | -0.252% | 0/5 |
| legacy v3b fit mean raw | 376890 | 2.40599024e-05 | -0.011% | 2/5 |
| legacy v3b fit mean shrink | 376890 | 2.40577117e-05 | -0.002% | 2/5 |
| legacy v3b ridge history | 376890 | 2.41530505e-05 | -0.398% | 3/5 |
| legacy v3b ridge history anchor | 376890 | 2.42437398e-05 | -0.775% | 1/5 |
| legacy v3b ridge history selected | 376890 | 2.40663042e-05 | -0.038% | 3/5 |
| legacy v3b ridge history anchor selected | 376890 | 2.40888723e-05 | -0.131% | 1/5 |
| legacy v3b default seed 2026 raw | 376890 | 3.10097433e-05 | -28.900% | 0/5 |
| legacy v3b default seed 2026 shrink | 376890 | 2.41178582e-05 | -0.252% | 0/5 |
| legacy v3b default seed 2027 raw | 376890 | 3.00485287e-05 | -24.904% | 0/5 |
| legacy v3b default seed 2027 shrink | 376890 | 2.40848142e-05 | -0.114% | 0/5 |
| legacy v3b default seed 2028 raw | 376890 | 3.29575012e-05 | -36.996% | 0/5 |
| legacy v3b default seed 2028 shrink | 376890 | 2.41359856e-05 | -0.327% | 1/5 |
| legacy v3b default three seed raw | 376890 | 3.13385911e-05 | -30.267% | 0/5 |
| legacy v3b default three seed shrink | 376890 | 2.4112886e-05 | -0.231% | 1/5 |
| legacy v3b repair seed 2026 raw | 376890 | 2.87695412e-05 | -19.588% | 0/5 |
| legacy v3b repair seed 2026 shrink | 376890 | 2.40680525e-05 | -0.045% | 1/5 |
| legacy v3b repair seed 2027 raw | 376890 | 2.76626328e-05 | -14.987% | 0/5 |
| legacy v3b repair seed 2027 shrink | 376890 | 2.40658413e-05 | -0.036% | 1/5 |
| legacy v3b repair seed 2028 raw | 376890 | 2.85607398e-05 | -18.720% | 0/5 |
| legacy v3b repair seed 2028 shrink | 376890 | 2.40654205e-05 | -0.034% | 2/5 |
| legacy v3b repair three seed raw | 376890 | 2.83309713e-05 | -17.765% | 0/5 |
| legacy v3b repair three seed shrink | 376890 | 2.40664381e-05 | -0.038% | 1/5 |
| legacy v3b A mean raw | 376890 | 2.40599024e-05 | -0.011% | 2/5 |
| legacy v3b A mean shrink | 376890 | 2.40577117e-05 | -0.002% | 2/5 |
| legacy v3b B context raw | 376890 | 2.40980075e-05 | -0.169% | 0/5 |
| legacy v3b B context shrink | 376890 | 2.40572794e-05 | +0.000% | 0/5 |
| legacy v3b C history raw | 376890 | 2.41268124e-05 | -0.289% | 2/5 |
| legacy v3b C history shrink | 376890 | 2.40853978e-05 | -0.117% | 1/5 |
| legacy v3b D joint raw | 376890 | 2.41748121e-05 | -0.489% | 1/5 |
| legacy v3b D joint shrink | 376890 | 2.41105915e-05 | -0.222% | 1/5 |

| Primary | Reference | Block days | Gain | 95% interval | Supports gain |
| --- | --- | --- | --- | --- | --- |
| tied last | untied last | 1 | +1.942% | [-0.100, +4.796]% | False |
| tied selected shrink | zero | 1 | -0.048% | [-0.161, +0.045]% | False |
| tied selected shrink | fit mean raw | 1 | -0.037% | [-0.168, +0.090]% | False |
| tied selected shrink | fit mean shrink | 1 | -0.046% | [-0.165, +0.060]% | False |
| tied selected shrink | untied last | 1 | +30.653% | [+28.101, +33.393]% | True |
| tied selected shrink | untied last shrink | 1 | +0.019% | [-0.083, +0.112]% | False |
| tied selected shrink | untied selected | 1 | +16.321% | [+12.156, +20.846]% | True |
| tied selected shrink | untied selected shrink | 1 | -0.028% | [-0.145, +0.076]% | False |
| tied selected shrink | legacy v3b zero | 1 | -0.048% | [-0.161, +0.045]% | False |
| tied selected shrink | legacy v3b archived default raw | 1 | +22.383% | [+18.518, +26.503]% | True |
| tied selected shrink | legacy v3b archived default shrink | 1 | +0.203% | [-0.054, +0.460]% | False |
| tied selected shrink | legacy v3b fit mean raw | 1 | -0.037% | [-0.168, +0.090]% | False |
| tied selected shrink | legacy v3b fit mean shrink | 1 | -0.046% | [-0.165, +0.060]% | False |
| tied selected shrink | legacy v3b ridge history | 1 | +0.348% | [-0.068, +0.783]% | False |
| tied selected shrink | legacy v3b ridge history anchor | 1 | +0.721% | [+0.167, +1.276]% | True |
| tied selected shrink | legacy v3b ridge history selected | 1 | -0.011% | [-0.329, +0.341]% | False |
| tied selected shrink | legacy v3b ridge history anchor selected | 1 | +0.083% | [-0.285, +0.473]% | False |
| tied selected shrink | legacy v3b default seed 2026 raw | 1 | +22.383% | [+18.518, +26.503]% | True |
| tied selected shrink | legacy v3b default seed 2026 shrink | 1 | +0.203% | [-0.054, +0.460]% | False |
| tied selected shrink | legacy v3b default seed 2027 raw | 1 | +19.900% | [+16.793, +23.114]% | True |
| tied selected shrink | legacy v3b default seed 2027 shrink | 1 | +0.066% | [-0.107, +0.222]% | False |
| tied selected shrink | legacy v3b default seed 2028 raw | 1 | +26.970% | [+22.391, +31.814]% | True |
| tied selected shrink | legacy v3b default seed 2028 shrink | 1 | +0.278% | [-0.074, +0.712]% | False |
| tied selected shrink | legacy v3b default three seed raw | 1 | +23.197% | [+19.596, +27.034]% | True |
| tied selected shrink | legacy v3b default three seed shrink | 1 | +0.182% | [+0.003, +0.364]% | True |
| tied selected shrink | legacy v3b repair seed 2026 raw | 1 | +16.339% | [+12.547, +20.634]% | True |
| tied selected shrink | legacy v3b repair seed 2026 shrink | 1 | -0.003% | [-0.150, +0.123]% | False |
| tied selected shrink | legacy v3b repair seed 2027 raw | 1 | +12.991% | [+9.471, +16.903]% | True |
| tied selected shrink | legacy v3b repair seed 2027 shrink | 1 | -0.013% | [-0.128, +0.098]% | False |
| tied selected shrink | legacy v3b repair seed 2028 raw | 1 | +15.727% | [+11.796, +19.898]% | True |
| tied selected shrink | legacy v3b repair seed 2028 shrink | 1 | -0.014% | [-0.363, +0.308]% | False |
| tied selected shrink | legacy v3b repair three seed raw | 1 | +15.044% | [+11.598, +18.991]% | True |
| tied selected shrink | legacy v3b repair three seed shrink | 1 | -0.010% | [-0.149, +0.124]% | False |
| tied selected shrink | legacy v3b A mean raw | 1 | -0.037% | [-0.168, +0.090]% | False |
| tied selected shrink | legacy v3b A mean shrink | 1 | -0.046% | [-0.165, +0.060]% | False |
| tied selected shrink | legacy v3b B context raw | 1 | +0.121% | [-0.050, +0.289]% | False |
| tied selected shrink | legacy v3b B context shrink | 1 | -0.048% | [-0.161, +0.045]% | False |
| tied selected shrink | legacy v3b C history raw | 1 | +0.240% | [-0.142, +0.603]% | False |
| tied selected shrink | legacy v3b C history shrink | 1 | +0.069% | [-0.190, +0.323]% | False |
| tied selected shrink | legacy v3b D joint raw | 1 | +0.438% | [-0.001, +0.870]% | False |
| tied selected shrink | legacy v3b D joint shrink | 1 | +0.173% | [-0.074, +0.430]% | False |
| tied last | untied last | 5 | +1.942% | [-0.214, +4.867]% | False |
| tied selected shrink | zero | 5 | -0.048% | [-0.176, +0.049]% | False |
| tied selected shrink | fit mean raw | 5 | -0.037% | [-0.187, +0.074]% | False |
| tied selected shrink | fit mean shrink | 5 | -0.046% | [-0.182, +0.055]% | False |
| tied selected shrink | untied last | 5 | +30.653% | [+28.074, +33.527]% | True |
| tied selected shrink | untied last shrink | 5 | +0.019% | [-0.109, +0.110]% | False |
| tied selected shrink | untied selected | 5 | +16.321% | [+10.397, +21.017]% | True |
| tied selected shrink | untied selected shrink | 5 | -0.028% | [-0.125, +0.056]% | False |
| tied selected shrink | legacy v3b zero | 5 | -0.048% | [-0.176, +0.049]% | False |
| tied selected shrink | legacy v3b archived default raw | 5 | +22.383% | [+17.267, +26.045]% | True |
| tied selected shrink | legacy v3b archived default shrink | 5 | +0.203% | [-0.103, +0.464]% | False |
| tied selected shrink | legacy v3b fit mean raw | 5 | -0.037% | [-0.187, +0.074]% | False |
| tied selected shrink | legacy v3b fit mean shrink | 5 | -0.046% | [-0.182, +0.055]% | False |
| tied selected shrink | legacy v3b ridge history | 5 | +0.348% | [-0.029, +0.655]% | False |
| tied selected shrink | legacy v3b ridge history anchor | 5 | +0.721% | [+0.201, +1.216]% | True |
| tied selected shrink | legacy v3b ridge history selected | 5 | -0.011% | [-0.273, +0.196]% | False |
| tied selected shrink | legacy v3b ridge history anchor selected | 5 | +0.083% | [-0.207, +0.354]% | False |
| tied selected shrink | legacy v3b default seed 2026 raw | 5 | +22.383% | [+17.267, +26.045]% | True |
| tied selected shrink | legacy v3b default seed 2026 shrink | 5 | +0.203% | [-0.103, +0.464]% | False |
| tied selected shrink | legacy v3b default seed 2027 raw | 5 | +19.900% | [+15.933, +22.705]% | True |
| tied selected shrink | legacy v3b default seed 2027 shrink | 5 | +0.066% | [-0.083, +0.218]% | False |
| tied selected shrink | legacy v3b default seed 2028 raw | 5 | +26.970% | [+21.616, +32.104]% | True |
| tied selected shrink | legacy v3b default seed 2028 shrink | 5 | +0.278% | [-0.059, +0.702]% | False |
| tied selected shrink | legacy v3b default three seed raw | 5 | +23.197% | [+18.554, +26.897]% | True |
| tied selected shrink | legacy v3b default three seed shrink | 5 | +0.182% | [+0.000, +0.353]% | True |
| tied selected shrink | legacy v3b repair seed 2026 raw | 5 | +16.339% | [+11.903, +19.747]% | True |
| tied selected shrink | legacy v3b repair seed 2026 shrink | 5 | -0.003% | [-0.163, +0.102]% | False |
| tied selected shrink | legacy v3b repair seed 2027 raw | 5 | +12.991% | [+8.313, +17.223]% | True |
| tied selected shrink | legacy v3b repair seed 2027 shrink | 5 | -0.013% | [-0.114, +0.089]% | False |
| tied selected shrink | legacy v3b repair seed 2028 raw | 5 | +15.727% | [+10.684, +19.364]% | True |
| tied selected shrink | legacy v3b repair seed 2028 shrink | 5 | -0.014% | [-0.190, +0.204]% | False |
| tied selected shrink | legacy v3b repair three seed raw | 5 | +15.044% | [+10.522, +18.588]% | True |
| tied selected shrink | legacy v3b repair three seed shrink | 5 | -0.010% | [-0.119, +0.092]% | False |
| tied selected shrink | legacy v3b A mean raw | 5 | -0.037% | [-0.187, +0.074]% | False |
| tied selected shrink | legacy v3b A mean shrink | 5 | -0.046% | [-0.182, +0.055]% | False |
| tied selected shrink | legacy v3b B context raw | 5 | +0.121% | [-0.081, +0.281]% | False |
| tied selected shrink | legacy v3b B context shrink | 5 | -0.048% | [-0.176, +0.049]% | False |
| tied selected shrink | legacy v3b C history raw | 5 | +0.240% | [-0.164, +0.561]% | False |
| tied selected shrink | legacy v3b C history shrink | 5 | +0.069% | [-0.205, +0.327]% | False |
| tied selected shrink | legacy v3b D joint raw | 5 | +0.438% | [-0.010, +0.818]% | False |
| tied selected shrink | legacy v3b D joint shrink | 5 | +0.173% | [-0.085, +0.464]% | False |

## S18. Completion, provenance and remaining research goal

The manuscript builder requires all thirty completed training receipts, the complete queue receipt, the 82-file analysis closure and a passed independent statistics review with matching source/input identities before reading these results. The independent reviewer reconstructs endpoint/path/absolute errors, fit-only mean control calibration and trading-day block statistics; it does not rerun a new market period. Model scalar provenance and the exact recorded shrink relation are checked; full inner-stop re-inference is not claimed by that statistics review.

The original title, author and school details, five temporal scales, twenty task table, annual sample limits, split-event sensitivity and completed negative results remain preserved. The author supplies the email before submission. Protected 2026Q3 remains sealed. A candidate that passes this development gate would still require a separately frozen eligible confirmation design. The overall mechanism-and-improvement research goal is not marked achieved by this manuscript or by minute-only development success. No external submission, public deposit, DOI or trading-profitability claim is made.

## S19. Local gradient-direction probe: numerical history and stopped route

### S19.1 Fixed question and controls

The completed sharing intervention changes parameter capacity, head-gradient aggregation and Adam histories together. The follow-up asks whether projecting a negative head-gradient component changes the next fit-batch loss after matching actual encoder-update magnitude. This local question does not identify the cause of the complete training trajectory or test an improved market forecaster.

The plan fixes the earliest two folds, seeds 2026/2027/2028 and three positions in each original purged H60 fit sequence. For N fit origins, a unit starts at floor((N−256)q), with q=0.25, 0.50 or 0.75. Its first 128 origins provide the update batch and the next 128 provide diagnostic losses. Adjacent future labels can overlap. Eighteen units represent six fold/seed states, not eighteen independent market samples.

Each unit clones the final twenty-epoch tied model with matching BatchNorm and Adam states. Recorded identities establish that the latest model matches the registered last checkpoint; the probe does not combine a selected model with another epoch's optimizer. CPU nonfused arithmetic does not claim bitwise reproduction of a GPU fused-Adam step. One train-mode forward supplies common updated BatchNorm buffers. Four head-specific encoder gradients use the existing standardized path losses. A applies the original total-backward Adam update. B projects negative dot products in fixed H5/H15/H30/H60 order, then applies Adam from the same state. This order-dependent projection does not guarantee nonnegative final pairwise dot products. C retains A's actual encoder-displacement direction while matching B's norm. Zero A displacement with nonzero B makes C undefined and the unit inapplicable. Nonencoder updates remain identical. All branches use common post-forward BatchNorm buffers in eval-mode diagnostics.

Continuation requires all eighteen units complete and eligible. Within each fold/seed, the median over three positions of loss_C−loss_B must be positive for H60 endpoint error and nonnegative for path error. All six states must satisfy both conditions before independent review can consider bounded short training. This conservative route criterion supplies no significance test or market forecast candidate, and permits no replacement positions.

### S19.2 Disclosed numerical revisions

The initial actual-model synthetic acceptance passed. The original elementwise float32 gradient-reconstruction check then failed after three saved market-local units; one of 17,416 checked coordinates failed. An instrumentation-only R1 retained the failure. R2 revised the engineering numerical contract after observing it. This revision was not the original registration and is not a numerical theorem. It required:

- Global and per-parameter float32 reconstruction residual L2 no greater than 16 epsilon32 times the sum of the corresponding head-gradient L2 norms.
- A same-preforward-state float64 identity check with residual no greater than 4096 epsilon64 times its component scale.
- Exact model and Adam agreement between A and an independently executed ordinary total backward in each unit.
- An unprojected reconstructed-gradient Adam null with overall and per-parameter relative L2 at most 10^−4 and global cosine at least 0.99999999, without loss-based selection.
- Direction attribution only when the unprojected numerical-null displacement is at most 1% of projected B−A displacement, with the recorded nonzero and norm/direction checks.

R2's actual-model synthetic test passed. Its first market unit passed numerical checks but failed JSON serialization of a NumPy float32 tolerance, so R2 committed no market-unit JSON. R3 converts tolerance scalars to builtin floats and adds full synthetic JSON-serialization acceptance. R3 changes neither numerical expressions nor scientific thresholds. The original failure and R1/R2/R3 records remain separate; their partial units are not pooled.

### S19.3 Saved-unit inventory and interpretation

R3 saves four units, then stops when the fifth, `f1_s2027_q50`, fails the per-parameter float32 reconstruction check in `encoder.fc1.weight`. All four saved units satisfy R2 reconstruction and ordinary-A controls. Only one also qualifies for direction attribution. The following are receipt scalars, not a fresh gradient or market recomputation.

| Saved unit | Negative head pairs | Float64 residual L2 | Null relative L2 | Attribution |
| --- | ---: | ---: | ---: | --- |
| f1_s2026_q25 | 0 | 6.4124479904e−14 | 1.2475000793e−6 | Ineligible |
| f1_s2026_q50 | 1 | 1.2397939500e−14 | 2.8277356288e−7 | Eligible |
| f1_s2026_q75 | 0 | 4.3293409722e−14 | 8.8539669054e−7 | Ineligible |
| f1_s2027_q25 | 0 | 5.3474230535e−14 | 9.1478347248e−7 | Ineligible |

The three ineligible units have no negative gradient pair and do not separate a projection effect from the numerical null. Four of eighteen units exist and one is eligible; the complete-and-eligible prerequisite fails. The six-state criterion remains unevaluated. No further tolerance revision, replacement unit, short training or candidate weights follow. The independent reviewer inspects frozen sources, authorization, unit receipts, synthetic/admission records and the failure log; it does not recalculate market gradients or predictions. Partial numerical nonqualification neither supports nor scientifically refutes the mechanism.

The worker constructs only fit targets and diagnostics. Its frozen loader reads the admitted, previously exposed complete pre-Q3 bar file and split metadata; disk access is not restricted to fit rows alone. No Q3 source is selected, and no original stopping or outer score enters the probe. The reviewed execution path does not constitute an OS-level file-access trace. [Final probe results](review/source_snapshot/evidence/probe/FINAL_LOCAL_PROBE_RESULTS.json) and [independent receipt review](review/source_snapshot/evidence/probe/FINAL_LOCAL_PROBE_INDEPENDENT_REVIEW.json) retain exact identities and limits.

## S20. Fixed price–volume information screen

### S20.1 Source identity, clock and one-field expansion

A separate information hypothesis asks whether completed trading activity modulates the conditional continuation or reversal of a past price move. We fix one new field, unsigned volume, from the archived Alpaca SIP/raw minute CSV for January 4, 2021 through June 30, 2026. Timestamps establish alignment and closes establish identity; the other expanded-source market fields do not enter predictors.

The source receipt accounts for all 1,061,536 rows: 535,620 regular-session minutes and 525,916 exclusions. Aligned closes equal the preceding float64 cache exactly. Hourly volume sums the exact session-open-anchored groups of sixty consecutive regular minutes used by the original hourly closes, producing 8,238 complete hourly bars and excluding 41,340 residual minutes. Existing close-only caches remain unchanged. Source byte identity matches before and after projection. [The source receipt](review/source_snapshot/evidence/volume/VOLUME_SOURCE_RECEIPT.json) records identity and grouping checks. Historical final bars do not prove as-received availability: we assume use after bar completion, while vendor revisions and arrival latency remain unresolved. H8 can cross sessions and does not denote eight clock hours. Minute future labels remain within the regular session.

For completed bar j in session d at slot s, define

\[
a_j=\log(1+V_j)-\operatorname{median}\{\log(1+V_{d',s}):d'\in\mathcal{D}_{20}(d)\},
\]

where the set contains exactly the twenty immediately preceding completed global trading sessions and excludes all of session d. All twenty same-slot observations must be valid. Half-day absent slots are not filled from older sessions or artificial zeros; a recorded real zero remains zero. At origin t,

\[
u_t=H^{-1}\sum_{j=t-H+1}^{t}a_j,\qquad r_t=\log P_t-\log P_{t-H}.
\]

Activity requires H bars and the past return requires H+1 closes. The activity baseline may update using already completed prior diagnostic sessions, while coefficients and standardization stay fixed. A common source-validity mask gives identical origins to A, B and C.

### S20.2 Arms, partitions and acceptance

A uses the preceding seventeen compressed-history/clock/weekday columns: nine history features, four clock-harmonic terms and four weekday indicators. B adds u_t; C adds r_t u_t to B. We form the interaction in its stated physical units before update-pool standardization. All arms fit the same complete cumulative-log-return target path using deterministic normalized Ridge, lambda 0.1, and an unpenalized intercept. C_raw is the sole continuation form, without extra shrinkage calibration. C versus B tests the added interaction representation, while C versus A reports overall change. Added regressors also alter joint Ridge shrinkage; neither contrast isolates an economic cause.

For each fold, we first define the chronological 80/20 boundary on the complete original fit-index sequence. We then purge update origins whose labels finish at or after the first diagnostic decision and apply the common source-only validity mask. New-field validity cannot move that original boundary. Scaling, target means and coefficients use the retained update segment; the tail supplies fit-internal diagnostic scores. The earliest two folds, two tasks and three arms give twelve deterministic fits without artificial seed replication. Update/diagnostic counts are 45,151/12,853 and 93,128/25,753 for minute folds 1/2, and 871/228 and 1,671/459 for hourly folds 1/2.

Synthetic acceptance covers past-only slot normalization, nontrivial response to historical volume, expiry after twenty sessions, future-session/length invariance, CSV matching and rejection, hourly grouping, boundary purging, and H-volume/H+1-close alignment. Streaming/resumed sufficient statistics match a direct centered Ridge solve to a maximum coefficient difference of 2.7755575615628914e−16. A fixed-shape timing probe admits the twelve-fit matrix before formal scoring. These checks establish frozen implementation properties, not predictive validity.

In all four task/fold states, C must have strictly lower endpoint MSE than A and B, no greater path MSE than either, and endpoint MSE no greater than zero. Within each task, the two physical-unit endpoint interaction coefficients must share a nonzero sign. All four states and both sign checks are required. This exposed fit-internal route rule is distinct from the original five-fold predictive gate. [Frozen rules](review/source_snapshot/evidence/volume/RULES.json) have SHA-256 3276502a5c13f659e6e4fad8b42d40a063f131a1a6951a6eb8cdc7535dd56ad9.

### S20.3 Complete results and independent reconstruction

All twelve fits complete. Positive percentages mean lower C endpoint MSE, using 100(1−MSE_C/MSE_reference). The coefficient below refers to the physical r_t u_t interaction.

| Task/fold | C vs zero | C vs A | C vs B | Endpoint coefficient | All conditions |
| --- | ---: | ---: | ---: | ---: | --- |
| Minute H60 / 1 | -2.843273% | -0.276985% | -0.124916% | -0.1467515387 | Fail |
| Minute H60 / 2 | -1.203783% | +0.538688% | +0.591163% | -0.0981053987 | Fail |
| Hourly H8 / 1 | -8.863169% | -0.470969% | -0.018373% | -0.1398159624 | Fail |
| Hourly H8 / 2 | +0.306731% | +1.598943% | +1.122640% | -0.0941639658 | Pass |

Endpoint MSE uses the last horizon; path MSE averages over origins and all horizon steps. The table rounds frozen scalar outputs; the linked JSON retains full precision.

| Task/fold and metric | A | B | C | Zero |
| --- | ---: | ---: | ---: | ---: |
| Minute 1, endpoint | 1.7041595326e−5 | 1.7067477938e−5 | 1.7088798031e−5 | 1.6616349891e−5 |
| Minute 1, path | 8.7587533055e−6 | 8.7772530697e−6 | 8.7928068391e−6 | 8.5972378119e−6 |
| Minute 2, endpoint | 4.3889523306e−5 | 4.3912691246e−5 | 4.3653095863e−5 | 4.3133857771e−5 |
| Minute 2, path | 2.3493028610e−5 | 2.3498235866e−5 | 2.3421971614e−5 | 2.3243370192e−5 |
| Hourly 1, endpoint | 2.4779968315e−4 | 2.4892100938e−4 | 2.4896674251e−4 | 2.2869694520e−4 |
| Hourly 1, path | 1.4023206787e−4 | 1.4064358697e−4 | 1.4091234226e−4 | 1.3296186584e−4 |
| Hourly 2, endpoint | 6.9020015063e−4 | 6.8687537778e−4 | 6.7916424212e−4 | 6.8125386066e−4 |
| Hourly 2, path | 3.8929386050e−4 | 3.8779733783e−4 | 3.8414607723e−4 | 3.8355986013e−4 |

Both task-specific sign checks pass, but only hourly fold 2 meets every local condition. Minute fold 2 beats A and B but remains worse than zero. Hourly fold 2's path MSE also remains worse than zero; the fixed rule required its path error to beat A and B only. The all-four-state criterion fails, so no five-fold candidate follows. No alternate fold, activity window, field, penalty or shrinkage coefficient is selected from these scores. This screen does not establish the original stable 1% objective.

An independent direct-solve review reconstructs all twelve models, predictions, metrics and the Boolean gate from the recorded source, fixed partitions and acceptance artifacts. It reproduces the failed decision without adding a scientific tolerance. Separate claim review binds the reviewed addendum to its final source hash. Reconstruction and document acceptance add no new market period. [Complete results](review/source_snapshot/evidence/volume/RESULTS.json), [numerical review](review/source_snapshot/evidence/volume/INDEPENDENT_NUMERICAL_REVIEW.json) and [final acceptance](review/source_snapshot/evidence/volume/FINAL_ACCEPTANCE.json) preserve these distinctions. No original stopping or outer targets or scores enter the screen; protected 2026Q3 remains unread.

## S21. Subsequent primary-literature verification and reproduction map

### S21.1 Chronology and reading coverage

After completing and reviewing the failed screen, we retrieve two public author-hosted typeset journal PDFs and verify targeted sections with full extracted-text keyword searches. This does not constitute line-by-line verification of all algebra, proofs, tables or references, or independent paper replication. The later findings cannot retrospectively strengthen screen registration, change its failed rule or admit a model. This literature stage adds no market fitting, fields or scores. [The Campbell review](review/source_snapshot/evidence/campbell/PRIMARY_TEXT_REVIEW.json) and [the Llorente review](review/source_snapshot/evidence/llorente/PRIMARY_TEXT_REVIEW.json) record retrieval, versions, pages and actual reading.

Campbell, Grossman and Wang [17] use daily market returns and aggregate NYSE turnover. Their principal CRSP value-weighted NYSE/AMEX sample covers July 1962 through September 1987, with period and universe robustness checks. Detrended log turnover subtracts a backward one-year moving average from market shares traded divided by shares outstanding. They regress the next daily return; second-order analyses use the second subsequent day, not a cumulative intraday path. Even their thirty-two individual-stock regressions use the single aggregate market-volume series (author PDF printed page 922). Our own-stock volume, twenty-session same-slot median and H60/H8 targets differ in information, frequency and normalization.

Their risk-averse-intermediary model allows price direction with unsigned volume to inform inference about latent noninformational demand under its assumptions. The empirical regression does not require observed signed order flow. A negative interaction alone nevertheless cannot identify motives, informed trading or market-maker preferences. The authors discuss theoretical/empirical variable differences and calibration limits (printed pages 929 and 933–935). Their within-sample coefficients are not a prespecified zero-benchmark forecast-MSE test.

The obtained Campbell author PDF has thirty-five parseable pages bearing printed numbers 905–939. The MIT author publication catalog lists 905–940; printed page 940 and that boundary remain unverified. Main reference [17] omits the disputed full range. The journal identity is DOI 10.2307/2118454; the author-hosted journal-body copy has SHA-256 cc27550019281c472483607976f823f7db790985acf8109de1f5e2ce63af6238. The separate NBER w4193 file is partial and unverified and supplies no version comparison.

Llorente et al. [18] study 2,226 NYSE/AMEX stocks with daily CRSP data and TAQ quotes during 1993–1998 (printed pages 1018–1019). Empirical volume is log(turnover+0.00000255) minus its previous two-hundred-day mean, with turnover equal to shares traded divided by shares outstanding (page 1020). Equation 12 regresses the next daily return on the prior return and volume-by-return interaction (page 1022). The interaction form is established prior art. Its baseline, units and target differ from our log1p-share-volume, twenty-session/H-bar construction.

The authors relate coefficients to cross-sectional information-asymmetry proxies, including average opening spreads and capitalization, and examine analyst coverage and market residualization. Some proxies use whole-sample averages and cannot be treated as known at the start of a prospective forecast. Market residualization requires a broader panel. End-of-day midquotes, 10:00-to-next-10:00 returns, trade-count controls and one-/two-/five-day intervals address historical measurement and frequency alternatives; the 10:00 experiment remains daily. Their acknowledgment of multiple autocorrelation sources (page 1041) limits motive attribution. The reviewed design supplies no prespecified held-out AAPL minute/hour MSE gain against zero and matched strong forecasting controls.

The Llorente author copy identifies the 2002 RFS journal version, volume 15(4), pages 1005–1047, DOI 10.1093/rfs/15.4.1005. Its forty-three-page PDF SHA-256 is 84c37ce7344380993bc011611d1561f5dba3ada2477cb43b9ad49d956257f29c. We distinguish it from the 2000 SSRN working-paper DOI; that working paper supplies no detailed findings here. Conditional association, identification under assumptions and forecast improvement remain different claims.

### S21.2 Supported inference and remaining requirements

These papers motivate a conditional volume–return question and identify missing controls. Neither establishes that our AAPL representation should pass its rule. Unsigned volume can be informative without directly recording trade direction; the failed screen establishes neither general volume uselessness nor refutation of either theory. A negative coefficient is not an identified trading regime, and its sign is not the full conditional-return slope. Later literature gives no basis to rescue the failed screen by rotating fields, transformations, windows or models.

The evidence synthesis admits no new candidate. A future mechanism study would need an observable-at-decision information set and a falsifiable design separating the proposed channel from bid–ask bounce, stale prices, liquidity and common-market explanations. A forecasting claim would further need a frozen candidate, strong development controls and an eligible untouched confirmation period. Such a proposal is not an executed or admitted experiment in this edition. The title and twenty-task comparison remain fixed. The broader mechanism-and-improvement objective remains active and unachieved, with independent market confirmation outstanding.

### S21.3 Documentation and immutable computation dependencies

This edition integrates text and evidence receipts as a thin documentation set. [The reproduction guide](REPRODUCTION_GUIDE_EN.md) and its dependency manifest distinguish included documentation from external immutable archives, raw sources, partitions, caches and weights. [The source changelog](SOURCE_CHANGELOG.json) records source identities, new insertions and preservation checks. This integration neither loads market arrays nor executes archived market controllers or builders.

| Evidence layer | External dependency | Integration scope |
| --- | --- | --- |
| Original comparison and controlled/context studies | Preceding archives identified by shared-release and inherited manifests | Preserved results; no fresh historical retraining |
| Completed sharing study | [Shared computation archive](../../shared_followup/20261109_gpu_v2/release/AAPL_SHARED_GPU_COMPLETE_V2.zip) and replay/readme/statistics files | Archive identity and prior replay receipts; no new market period |
| Local direction probe | R3 source freeze, six original model/optimizer checkpoints, four unit receipts and final independent review linked in S19 | Partial numerical nonqualification; no new candidate weights |
| Fixed volume screen | Rules, original source/partitions, volume caches, twelve A/B/C files and reviews linked in S20 | Completed failed screen with prior twelve-fit reconstruction |
| Primary-text review | Two author-copy PDFs and retrieval/page/reading records linked in S21 | Bounded verified reading; no new forecasting candidate |

The sharing computation ZIP is distinct from its smaller manuscript-review ZIP. Earlier v3/v3a dependencies remain separately identified; the two sharing ZIPs do not contain every historical experiment or all licensed raw data. Full retraining requires exact admitted versions, session/corporate-action and split rules, environment and path mapping. The guide identifies archive hashes rechecked in this review and older identities retained as inherited provenance. Small-document checks do not reconstruct volume fits, rerun gradients or repeat prior saved-loss/predictor acceptance. Computational replay, method success and independent confirmation remain separate. No submission, public deposit, DOI or profitability claim follows from this integration.

## S22. Short-horizon trend restriction and target-scale checks

### S22.1 Native structure and scope

After the preceding integrated edition, a source audit traced the native xPatch trend stream through its second pooling and LayerNorm. Pooling leaves floor(H/2) coordinates. For native H2/H3, normalization acts on one coordinate, whose centered value and variance are zero; the normalized output is therefore its learned bias beta and has zero derivative with respect to that trend input. Holding seasonal inputs fixed, we vary only synthetic trend inputs. FP64 and FP32 H2/H3 cases have zero normalized/output differences and zero derivatives. H4/H8 positive controls produce nonzero responses. All eight frozen checks pass; source and checks receive independent review. Dtype-specific random draws are not bitwise paired, so their gradient magnitudes are not compared across dtypes.

The original H1 adapter uses native H2. Of the original twenty tasks, this dimensional condition affects only xPatch hourly-H1 price/return and quarterly-H3 price/return: four tasks. The seasonal stream and price-anchor route remain active. The result neither makes the full model constant nor establishes the restriction's contribution to market error. It does not explain the designated minute-H60/hourly-H8 failures. No market repair training or performance attribution follows.

### S22.2 Common-scale mapping and scale-only controls

Both original architectures already have linear anchor channels. At a common target standard deviation, an external current-price addition can be absorbed into an affine shift of anchor coefficient and bias. Under matched parameters, Adam/BatchNorm states, no weight decay and real arithmetic, corresponding physical predictions, losses, gradients and updates follow. Thus the external addition alone does not enlarge these readout function classes. The original price/residual comparison changed target standard deviation as well, so its market difference does not isolate this algebraic change.

Ten fixed native synthetic cells complete 102 paired update steps. All four FP64 common-scale nulls pass; the maximum physical prediction difference is 2.8422e-14. Three of four FP32 nulls pass. The xPatch randomly initialized mapped head fails the parameter-map tolerance over steps 2–12, with maximum normalized difference 0.00663255; the maximum physical prediction difference remains 1.5259e-5. The complete common-scale gate is false. We retain this failure without adjusting tolerance, deleting the cell, rerunning a preferred result or attributing it to an unsaved operator.

Both scale-only controls pass their fixed first-step gradient-scaling relations. Starting from matched physical functions but changing the target standard deviation separates the next-batch forecasts after the first update, by approximately 0.05090 and 0.05808 synthetic units. This establishes that scale can change the numerical optimization path in these checks. It neither attributes the historical market difference entirely to scale nor establishes which scale is better. These structural/scale checks load no market arrays or saved model weights and do not access Q3. Their frozen protocols, results, source and independent reviews are included under `evidence/structure_scale/`.

### S22.3 Training sufficiency and confirmation

The earlier H1 cap check remains a finite-budget result: seventeen default states received extra updates, and fifteen still reached sixty epochs. Of 150 added long-scale fits, 65 ended at twenty epochs and 61 selected epoch twenty. They do not establish convergence of every architecture. Shared-study training and stopping logs differ in time, mode and objective, so subtracting them would not give a common-objective generalization gap. No qualified forecasting candidate or new market confirmation follows from the present audit. Historical exposure and source identities require a current task-appropriate audit before any future eligible confirmation; a past sealed-data receipt alone is insufficient.

## S23. Conditional affine-readout fit gap in fixed LSTM representations

For the minute H60 and hourly H8 log-return tasks, folds 1 and 2, seed 2026, we froze the previously selected anchor-baseline LSTM. The design consisted of its 32 final hidden coordinates, the original standardized observed anchor, and an intercept. Original complete-path fit targets were standardized with the saved training scaler, cast to FP32 as in training, then promoted to FP64. We decomposed the empirical error of the original head into its conditional affine projection error and the squared prediction gap, using streaming augmented QR followed by SVD. No projected coefficient vector or predictor was retained.

| Task / fold | Fit origins | Selected epoch | Original head MSE | Conditional projected MSE | Fit-error reduction |
|---|---:|---:|---:|---:|---:|
| Minute H60 / 1 | 64261 | 4 | 0.997011183 | 0.979857961 | 1.7205% |
| Minute H60 / 2 | 128761 | 1 | 0.999354795 | 0.994373840 | 0.4984% |
| Hourly H8 / 1 | 1139 | 1 | 0.999377033 | 0.916966243 | 8.2462% |
| Hourly H8 / 2 | 2294 | 9 | 0.996154459 | 0.957857131 | 3.8445% |

All four designs had numerical rank 34 under the prespecified cutoff eps64 * max(n,34) * sigma_max, with no singular value in the ambiguity band [0.1 tau,10 tau]. Reconstruction, stationarity, and the empirical error identity passed their frozen numerical criteria. An independent implementation called the original forward with a head-input hook and projected the original dense X directly by SVD; its largest normalized scalar difference was 4.998e-12 against a frozen 1e-9 threshold. These values concern each state's standardized complete-path fit objective, not raw endpoint-return MSE.

The result establishes a conditional empirical head-minimization gap for these four fixed learned representations. It does not establish optimizer-specific failure, an all-network optimization gap, or an improvement on future observations. Early stopping may have regularized the original heads. The design condition numbers ranged from approximately 4.6e4 to 7.7e6; full numerical rank does not ensure a stable unpenalized extrapolation. The hourly fit sets contained only 1139 and 2294 overlapping origins, so their larger fit reductions warrant particular caution. The remaining projection error is not an estimate of irreducible market noise.

To reproduce historical features exactly, each replay loaded the complete admitted pre-Q3 bar cache. Only the original fit member of the split archive was used to generate labels and compute losses. No new stop/outer predictions or scores were produced, and 2026Q3 remained sealed. Same-data numerical recomputation is separate from independent market confirmation.

The first attempted replay stopped at an unsupported NumPy Unicode maximum reduction before model-weight deserialization or fit-loss computation. A separately archived revision changed only the equivalent ISO-date boundary check and a reporting field name; it preserved the data, numerical contracts, and absolute deadline. Actual process pause, elapsed timeout, signal interruption, recovery, and corruption-rejection checks were rerun before the four replays.

The next temporal intervention was frozen and completed subsequently, as reported in S24. Its encoders and scalers were trained anew on earlier update pools, so its later labels had not trained those representations. The preceding fixed checkpoints were not retrospectively relabeled as unexposed representations.

## S24. Purged temporal test of exact readout refitting

For the minute H60 and hourly H8 return tasks, folds 1 and 2, seed 2026, the original fit origins were divided chronologically at floor(0.8n). Update origins whose complete labels ended at or after the first later-segment decision were purged. Every input, anchor and target scaler was refitted on the remaining update set. A new original LSTM was trained with the original zero-head initialization, Adam recipe, shuffle and batch rules, for fixed budgets of 4, 1, 1 and 9 epochs. These budgets and the 80/20 design were informed by exposed historical development; no independent-confirmation claim is made.

Arm A retained the trained head. Arm B replaced only that head with an unpenalized exact affine readout fitted to the same update data and frozen trained representation. Fixed controls C and D used anchor-plus-intercept and the never-trained same-initialization encoder, respectively, with exact readouts. Zero return and the update-only mean were also retained. Only B could advance; no control was eligible for selection as an alternative candidate.

| Task / fold | Update / later origins | Epochs | B update-MSE reduction versus A | B later path-MSE change | B later endpoint-MSE change |
|---|---:|---:|---:|---:|---:|
| Minute H60 / 1 | 51391 / 12853 | 4 | 1.6975% | +0.6208% | +0.7551% |
| Minute H60 / 2 | 102948 / 25753 | 1 | 0.7320% | +0.3999% | +0.6524% |
| Hourly H8 / 1 | 903 / 228 | 1 | 8.0692% | Not scored: numerical exclusion | Not scored: numerical exclusion |
| Hourly H8 / 2 | 1827 / 459 | 9 | 5.1586% | +24.3626% | +32.2872% |

Positive later-loss changes mean deterioration, with A as the denominator. The hourly fold1 update reduction is a descriptive value from a numerically unqualified solve; it is not a qualified empirical optimum. Path MSE uses the update-derived target-standardization scale; endpoint MSE uses original-unit cumulative log returns. Counts sum row instances and overlap across folds. Four neural fits completed 2556 batch updates. Of 39293 planned later row instances, 39065 were scored.

The hourly fold1 B and D stationarity diagnostics were approximately 1.0189e-10 and 1.4408e-10, exceeding the frozen 1e-10 criterion. Its later labels were not evaluated. No cutoff, tolerance or penalty was altered. In all three evaluable states, exact readout refitting reduced update MSE but increased both later path and endpoint MSE point estimates. The prespecified continuation filter required all four states numerically eligible and a fixed task to beat A/C/D/zero at both losses in both folds beyond the frozen first-order sensitivity estimate. It failed; this exact-readout recipe was stopped without a regularization, epoch, window or seed search.

This is evidence about the temporal consequence of the final-readout intervention under this recipe. It does not identify an Adam-specific defect, distinguish implicit regularization from representation instability or finite-sample effects, or imply that market returns lack predictive information. The original formal forecasting-success requirements, including five folds, three seeds, at least 1% improvement over zero and the strong controls, remain unmet and unchanged.

Numerical ranking used a frozen scale- and conditioning-aware first-order perturbation estimate, explicitly not a certified numerical interval or a statistical confidence interval. The B-versus-A deterioration exceeded this estimate in minute fold1 and hourly fold2; minute fold2 and contrasts involving D had wide estimates and are reported as unresolved where applicable. Agreement between implementations does not convert these estimates into proofs. Existing session-block bootstrap rules were retained: 5000 draws, seed 2026, blocks 1/5 sessions for minute and 2/5 for hourly. The exposed-development intervals are descriptive, unadjusted for multiplicity, and exclude numerical uncertainty. They do not establish statistically significant deterioration in every evaluable state.

Independent reconstruction reproduced update-only scaler values exactly, checked origin hashes and label-completion purges, verified the same-seeded initial clone, and extracted features with original forwards and head hooks. It solved on the original dense X by direct SVD without calling the primary worker or its LS kernel. All four numerical eligibility identities matched, including the excluded hourly fold1; the largest normalized metric difference was approximately 1.21e-13 against the frozen 1e-8 threshold. This is same-data numerical verification, not independent market confirmation.

The full admitted pre-Q3 bar cache was read to reproduce features, but only the original fit split member supplied update or diagnostic labels. No original stop/outer scores were added and 2026Q3 remained sealed. Actual training pause, timeout and signal recovery preserved model, Adam, initialized encoder, random states and batch position. Native synthetic cases exercised training recovery and numerical exclusion; a separate exact-source full-rank composition fixture exercised the later-loss and serialization branch. Two archived serialization-only revisions converted NumPy booleans to native booleans; numerical rules were unchanged.

The completed evidence supports a bounded negative finding: conditional empirical head-minimization gaps need not translate into lower later-development loss. No forecasting candidate, stable improvement over strong baselines, or independent market confirmation is established by this stage.

### S24.1 Complete later losses for all fixed controls

Each task/fold retains its own target-standardization scale. These are later fit-internal losses, not original outer losses; cross-state path values are not pooled. The excluded hourly fold 1 supplies no later loss for any arm.

| State | Arm | Path standardized MSE | Endpoint raw MSE |
|---|---|---:|---:|
| Minute H60 / 1 | A | 0.714623931802 | 1.64803358122e-05 |
| Minute H60 / 1 | B | 0.719060584873 | 1.66047727865e-05 |
| Minute H60 / 1 | C | 0.714909000561 | 1.65024345204e-05 |
| Minute H60 / 1 | D | 0.717112638706 | 1.65221472443e-05 |
| Minute H60 / 1 | zero | 0.717240992345 | 1.66163498905e-05 |
| Minute H60 / 1 | mean | 0.718770070726 | 1.6677404246e-05 |
| Minute H60 / 2 | A | 1.85001762249 | 4.31101033519e-05 |
| Minute H60 / 2 | B | 1.85741636748 | 4.33913580988e-05 |
| Minute H60 / 2 | C | 1.84937884999 | 4.3095436972e-05 |
| Minute H60 / 2 | D | 1.85666098208 | 4.33274535644e-05 |
| Minute H60 / 2 | zero | 1.85032907483 | 4.31338577711e-05 |
| Minute H60 / 2 | mean | 1.85158189261 | 4.31866398887e-05 |
| Hourly H8 / 1 | All | Not scored | Not scored |
| Hourly H8 / 2 | A | 2.00154064022 | 0.00067742369229 |
| Hourly H8 / 2 | B | 2.48916788302 | 0.000896145032726 |
| Hourly H8 / 2 | C | 1.99434550864 | 0.000675178971022 |
| Hourly H8 / 2 | D | 2.31127304798 | 0.000746396109453 |
| Hourly H8 / 2 | zero | 2.00484489052 | 0.000681253860655 |
| Hourly H8 / 2 | mean | 2.00674867764 | 0.000682356945358 |

### S24.2 Complete descriptive block intervals

The following sixty intervals retain the original helper rules, with 5,000 draws and seed 2026. Each row compares B against the stated reference. Positive relative improvement means lower B loss; these signs differ from the deterioration convention in Table 4. Intervals concern exposed development, are unadjusted for multiplicity and exclude numerical uncertainty. No interval changes the failed continuation decision or admits another candidate.

#### Minute H60 / 1

| Loss | Reference | Block sessions | Gain (%) | 95% interval (%) | Supports gain |
|---|---|---:|---:|---|---|
| path | A | 1 | -0.620837 | [-1.400597, +0.094523] | No |
| path | C | 1 | -0.580715 | [-1.334138, +0.069795] | No |
| path | D | 1 | -0.271637 | [-1.148725, +0.413415] | No |
| path | zero | 1 | -0.253693 | [-1.230373, +0.494121] | No |
| path | mean | 1 | -0.040418 | [-1.327269, +1.000893] | No |
| path | A | 5 | -0.620837 | [-1.299976, -0.062478] | No |
| path | C | 5 | -0.580715 | [-1.271623, -0.123667] | No |
| path | D | 5 | -0.271637 | [-1.197262, +0.324223] | No |
| path | zero | 5 | -0.253693 | [-1.168501, +0.314772] | No |
| path | mean | 5 | -0.040418 | [-1.379006, +0.898842] | No |
| endpoint | A | 1 | -0.755063 | [-1.598181, +0.351334] | No |
| endpoint | C | 1 | -0.620140 | [-1.359357, +0.330703] | No |
| endpoint | D | 1 | -0.500090 | [-1.244445, +0.034840] | No |
| endpoint | zero | 1 | +0.069673 | [-1.350407, +1.171905] | No |
| endpoint | mean | 1 | +0.435508 | [-1.712127, +2.064742] | No |
| endpoint | A | 5 | -0.755063 | [-1.572161, +0.408769] | No |
| endpoint | C | 5 | -0.620140 | [-1.455214, +0.241094] | No |
| endpoint | D | 5 | -0.500090 | [-1.309153, -0.091261] | No |
| endpoint | zero | 5 | +0.069673 | [-1.286426, +0.810947] | No |
| endpoint | mean | 5 | +0.435508 | [-1.670021, +1.641697] | No |

#### Minute H60 / 2

| Loss | Reference | Block sessions | Gain (%) | 95% interval (%) | Supports gain |
|---|---|---:|---:|---|---|
| path | A | 1 | -0.399928 | [-0.995662, +0.115796] | No |
| path | C | 1 | -0.434606 | [-1.002018, +0.075775] | No |
| path | D | 1 | -0.040685 | [-0.297186, +0.206918] | No |
| path | zero | 1 | -0.383029 | [-0.946707, +0.158101] | No |
| path | mean | 1 | -0.315108 | [-0.877428, +0.198564] | No |
| path | A | 5 | -0.399928 | [-0.901697, +0.053912] | No |
| path | C | 5 | -0.434606 | [-0.912730, +0.007716] | No |
| path | D | 5 | -0.040685 | [-0.324176, +0.244414] | No |
| path | zero | 5 | -0.383029 | [-0.896436, +0.091358] | No |
| path | mean | 5 | -0.315108 | [-0.809806, +0.144943] | No |
| endpoint | A | 1 | -0.652410 | [-1.373926, +0.033533] | No |
| endpoint | C | 1 | -0.686665 | [-1.412697, +0.005280] | No |
| endpoint | D | 1 | -0.147492 | [-0.464288, +0.145265] | No |
| endpoint | zero | 1 | -0.596980 | [-1.398359, +0.200829] | No |
| endpoint | mean | 1 | -0.474031 | [-1.203227, +0.240222] | No |
| endpoint | A | 5 | -0.652410 | [-1.239364, -0.108994] | No |
| endpoint | C | 5 | -0.686665 | [-1.272923, -0.145363] | No |
| endpoint | D | 5 | -0.147492 | [-0.448906, +0.173491] | No |
| endpoint | zero | 5 | -0.596980 | [-1.355608, +0.088792] | No |
| endpoint | mean | 5 | -0.474031 | [-1.129754, +0.133710] | No |

#### Hourly H8 / 2

| Loss | Reference | Block sessions | Gain (%) | 95% interval (%) | Supports gain |
|---|---|---:|---:|---|---|
| path | A | 2 | -24.362595 | [-53.502515, -4.850528] | No |
| path | C | 2 | -24.811266 | [-56.134500, -4.460898] | No |
| path | D | 2 | -7.696833 | [-26.641884, +5.776041] | No |
| path | zero | 2 | -24.157629 | [-55.694857, -3.344830] | No |
| path | mean | 2 | -24.039842 | [-55.565708, -3.588211] | No |
| path | A | 5 | -24.362595 | [-53.423030, -3.406933] | No |
| path | C | 5 | -24.811266 | [-56.056179, -2.836133] | No |
| path | D | 5 | -7.696833 | [-26.294617, +6.184692] | No |
| path | zero | 5 | -24.157629 | [-55.989458, -1.759899] | No |
| path | mean | 5 | -24.039842 | [-55.495576, -2.074561] | No |
| endpoint | A | 2 | -32.287229 | [-84.004145, -3.580589] | No |
| endpoint | C | 2 | -32.727036 | [-86.451043, -2.728697] | No |
| endpoint | D | 2 | -20.062929 | [-56.059306, +1.702432] | No |
| endpoint | zero | 2 | -31.543480 | [-84.194802, -0.985616] | No |
| endpoint | mean | 2 | -31.330829 | [-84.303580, -1.445688] | No |
| endpoint | A | 5 | -32.287229 | [-82.488646, -2.266896] | No |
| endpoint | C | 5 | -32.727036 | [-85.277004, -1.556639] | No |
| endpoint | D | 5 | -20.062929 | [-54.967969, +2.884950] | No |
| endpoint | zero | 5 | -31.543480 | [-84.318294, +0.379767] | No |
| endpoint | mean | 5 | -31.330829 | [-83.415980, -0.033657] | No |

### S24.3 Provenance, recovery and remaining objective

The primary and independent implementations agree on four eligibility identities and all three scored states. They use the same admitted data identities, so this is computational reconstruction on the same historical data. Full numerical reproduction still requires the external source bars, partitions, original model code and saved checkpoint generations identified in the evidence receipts. The thin documentation bundle contains text, source and JSON snapshots; it does not contain raw bars, arrays or checkpoints and cannot independently retrain the history by itself.

The first attempted fixed-state replay failed at a NumPy Unicode maximum reduction before weight deserialization or fit-loss computation. Its date-check revision preserved the absolute deadline. The temporal route then required two serialization-only revisions for native Python boolean conversion; the archived original failures remain and numerical thresholds are unchanged. Actual pause, timeout and signal checks preserve training/model/optimizer/initial-clone/random/batch state. Native synthetic cases exercise training recovery and numerical exclusion. A separate exact-source full-rank composition fixture exercises the qualified later-loss and serialization branch; it is not a qualified native-LSTM later-recovery test. No stop/outer score or Q3 value is added by these stages.

The fixed-state gap and fresh temporal intervention answer different questions. The former measures remaining empirical head fitting within an already selected representation; the latter tests one fixed consequence of removing that gap on later historical observations with freshly restricted fitting. Their combined negative transfer result supplies no stable method gain, no Adam-specific mechanism, no general rejection of learned representations and no independent market confirmation. Neither the anchor-only nor random-encoder control is eligible to replace B after inspecting these outcomes. The original title, five scales, twenty tasks and formal strong-control success criteria remain unchanged. The broader positive-method research goal remains active and unachieved. The present results support a paper about the tested forecasts and bounded implementation findings; document completion alone does not satisfy that broader goal.

## S25. Calendar maturity and limits of clock identifiability

S1–S24 retain the immediately preceding edition's bytes. S25–S27 append the later calendar, session-representation and cap studies without revising their earlier decisions. These studies use exposed development history; protected 2026Q3 remains sealed.

### S25.1 Audit inputs and complete ten-state counts

The audit reads four calendar arrays and only fit.npy from ten original split archives. It reads no prices, target values or weights and fits no model. Across the five folds, 320,761 unique minute origins account for 963,635 row visits; 5,750 unique hourly origins account for 17,227 visits. Expanding folds reuse origins, so those visits are not independent market replications.

An earlier same-slot path is calendar-usable only when its calendar-defined origin and full target interval fit the permitted history, its target completes by the current decision, and its session rule holds. Calendar usability does not establish observed-bar completeness, real-time vendor arrival or revision identity. The table reports previous-one-session and previous-two-session usability within the original L480/L96 histories. The last column counts clock/weekday groups containing more than one full historical-gap signature.

| Task/fold | Original fit origins | Previous 1 session usable | Previous 2 sessions usable | Groups with multiple gap signatures |
|---|---:|---:|---:|---:|
| Minute H60 / 1 | 64,261 | 64,261 | 0 | 749/1,650 |
| Minute H60 / 2 | 128,761 | 128,581 | 0 | 899/1,650 |
| Minute H60 / 3 | 192,931 | 192,571 | 0 | 899/1,650 |
| Minute H60 / 4 | 256,921 | 256,201 | 0 | 1,229/1,650 |
| Minute H60 / 5 | 320,761 | 319,861 | 0 | 1,650/1,650 |
| Hourly H8 / 1 | 1,139 | 0 | 1,139 | 30/30 |
| Hourly H8 / 2 | 2,294 | 0 | 2,291 | 30/30 |
| Hourly H8 / 3 | 3,449 | 0 | 3,443 | 30/30 |
| Hourly H8 / 4 | 4,595 | 0 | 4,583 | 30/30 |
| Hourly H8 / 5 | 5,750 | 0 | 5,735 | 30/30 |

### S25.2 What the calendar distinctions establish

No audited origin has a mature previous-session same-slot hourly-H8 path. Where that slot exists its path is unfinished; other origins lack the corresponding slot. On a normal six-complete-hour session grid, its origin lies near o−6 and its H8 endpoint near o+2. Treating its realized return as known at o would violate maturity. The audit makes no finding that an existing model used that unavailable feature. In fold 5, the nearest mature visible same-slot path comes from two sessions earlier at 5,735 origins and three sessions earlier at fifteen.

Minute H60 usually has a mature previous-session path within the original 480-bar window. On an ordinary 390-minute session grid, that return is logclose[o−330]−logclose[o−390]. A full linear history control can already express it. In fold 5, 319,861 origins meet the calendar rule; 600 lack a corresponding session slot, and 300 have a mature candidate that violates the same-session target rule. The second-previous-session path does not fit this original window. Adding a separate column for an already expressible return does not add raw information.

The origin clock and weekday do not identify the complete history's elapsed-time pattern. At fold 5, all 1,650 minute groups and all thirty hourly groups contain multiple gap signatures. Only 807 minute groups contain multiple session-transition signatures, whereas all thirty hourly groups do; gap and transition signatures are different objects. These counts do not establish a conditional-mean signal after conditioning on prices or attribute the neural errors to omitted timing. An hourly close-to-close increment across sessions combines the previous day's residual trading time, overnight movement and the next day's opening hour. It is not a pure overnight return.

### S25.3 Independent coverage and remaining availability limits

A separate dense-window implementation reconstructs all ten states and matches the vectorized audit's complete clock-group gap/transition histograms, previous-one/two-session counts, nearest-mature-session distributions, unique unions and row visits exactly. It shares access control and atomic writing. It does not independently reconstruct every bounded timestamp example, boundary_counts or failure_strata entry. The retained first atomic-I/O failure and a pre-read CLI deadline mismatch precede the accepted execution; neither supplies a forecasting result. The accepted run preserves the original absolute deadline.

The audit establishes timestamp relations in the retained calendar. It leaves provider receipt latency, bar revisions and observed-data availability for any new proposed feature unresolved. It adds no stop/outer score, forecasting candidate or independent market period. The subsequent experiment therefore fixes same-information controls rather than assuming that the calendar distinction provides new signal.

## S26. Fixed same-information session representation

### S26.1 Inputs, intervention and falsification rule

The fixed screen retains minute H60 and hourly H8, folds 1 and 2. Each arm receives the complete original split-adjusted log-relative price window x, the L−1 historical session-transition indicators g, L−1 log elapsed-time/native-interval values, and the original eight completed-origin New York clock/weekday columns. It omits the absolute nominal-price anchor. The base arm uses these common columns. The real arm adds `s_real=sum(g*diff(x))`; the mirror arm adds `s_mirror=sum(reverse(g)*diff(x))`. These deterministic summaries use the same underlying history and calendar information. The mirror can carry predictive information; it is not a validated signal-free placebo. Equal feature counts also do not equalize feature covariance or the penalty's effect.

We set each chronological boundary at floor(0.8n) of its original fit cohort before any feature-based filtering. Prefix labels must finish strictly before the first diagnostic-tail origin. The tail retains the original last 20%; no availability-based origin filtering occurs. All columns, including the summary, use prefix-only population standardization. Every fit minimizes complete-path raw nominal cumulative-log-return squared error with normalized Ridge penalty 0.1 and an unpenalized target-mean intercept. There is no target-standard-deviation scaling, extra shrinkage or penalty search.

Before constructing target arrays, all four feature designs must pass the frozen constant-summary, signed-equivalence and residual-rank checks. All four pass. The twelve fitted solutions pass the finite-value and normwise stationarity checks; no state is hidden through numerical exclusion. The real arm is the sole eligible continuation form. In every state, it must beat both fitted controls at endpoint and full-path MSE and beat zero and the prefix mean at endpoint MSE. Each of these six comparisons requires improvement greater than 1e-6 times the control MSE. The screen requires all four states; the separate formal five-fold, three-seed success criteria remain unchanged.

### S26.2 Complete local decisions

Positive gains mean lower endpoint MSE. Prefix/tail counts follow the original completion purge.

| State | Prefix rows | Tail rows | Real vs base (%) | Real vs mirror (%) | Real vs zero (%) | Six-condition screen |
|---|---:|---:|---:|---:|---:|---|
| Minute H60 / 1 | 51,391 | 12,853 | −0.773441 | −0.765404 | −4.379722 | Fail |
| Minute H60 / 2 | 102,948 | 25,753 | −0.015667 | −0.017941 | −0.419700 | Fail |
| Hourly H8 / 1 | 903 | 228 | +3.046931 | +2.722011 | +1.320643 | Pass |
| Hourly H8 / 2 | 1,827 | 459 | +0.584265 | +0.367230 | −2.397295 | Fail |

The minute states fail all six conditions. Hourly fold 2 beats base and mirror at both losses but fails the two endpoint comparisons with zero and the prefix mean. Its full-path MSE also exceeds zero, although that comparison was not a continuation requirement. Only hourly fold 1 passes. We stop the hypothesis with one of four required states; we do not restrict the task to hourly, select the winning fold or alter the summary, mask reversal, penalty or margin after viewing the losses.

Full-path changes agree in direction with the endpoint comparison against the base: real increases minute-fold losses by about 0.5972% and 0.0077% and reduces hourly-fold losses by about 1.9893% and 0.3731%. The second hourly state's path loss still exceeds zero by about 2.7576%. The small first-hourly gain does not establish stable cross-period improvement. Fold cohorts and overlapping targets remain dependent.

### S26.3 Post-observation error description and numerical reconstruction

After the main MSE results were known, a descriptive calculation applied MSE(p)−MSE(0)=E[p²]−2E[py]. For the base prediction b and delta=p−b, MSE(p)−MSE(b)=E[delta²]−2E[delta(y−b)]. These identities describe the full refitted function, including coupled changes to base coefficients and regularization; they do not isolate an economic effect of one coefficient. In both minute states, delta has negative empirical alignment with the base residual and increases loss. Both hourly states reduce base loss, but the second still has negative empirical prediction–target product and loses to zero. We use this calculation only to describe the fixed outcomes, with no new selection or significance claim.

The independent implementation calls no primary core, adapter or worker numerical function and reads no primary fitted model. It constructs the four partitions and feature tests, directly centers the full prefix designs and solves all twelve fits. Across five methods (base, real, mirror, zero, prefix mean), it reconstructs every step MSE plus endpoint and path MSE: 720 quantities. Maximum relative discrepancy is 1.2852004724340962e-15; the maximum error-identity residual relative to original MSE is 1.2359389688691124e-15. It shares only access control and atomic JSON writing. Agreement verifies this computation on the same exposed data.

The executions decode the full five-column admitted pre-Q3 minute/hourly bar caches; they do not read only the close values at fit origins. Split decoding admits only fit.npy. After all four feature gates pass, original fit labels supply prefix fitting and tail diagnosis. Neither execution decodes stop/outer indices or adds stop/outer scores. Two retained pre-score access-hook failures concern a runtime codec and a cross-environment NumPy lazy-load path. The reviewed R1/R2 changes extend runtime I/O admission while retaining the five other scientific source files and thresholds; the original deadline is unchanged. The access hook and source review are not an operating-system sandbox or ZIP-member monitor. Protected 2026Q3 remains sealed.

## S27. Cap sufficiency of the sole capped primary GRU

### S27.1 Local question and restored state

A separate audit of the forty original neural fits for the two primary return tasks finds thirty-nine patience stops. The sole capped state is hourly H8 return, GRU, fold 2, seed 2026, with best epoch 20 at the twenty-epoch limit. The 81 cap hits among all 400 original fits cannot establish general undertraining on these primary tasks; the earlier hourly-H1 price continuation addresses a different experiment.

The local intervention extends this one state's maximum epoch count from 20 to 60 while retaining the original fit/stop origins, architecture, target, scaler, path loss, batch order, Adam state and random states. Patience stays three; selection uses raw endpoint MSE on the original stopping partition. The complete restored state includes the epoch, batch cursor/permutation, accumulated losses, history, stale count and best model. Because max_epochs participates in the training identity, isolated latest copies change only identity; isolated best copies change identity and config.max_epochs. Bitwise checks preserve the scientific payload, and the original files remain unchanged. A cap20 no-update control reproduces the original state.

Before training, actual-data verification decodes all five fields of the 8,238-bar pre-Q3 hourly cache and matches its original semantic identity. It verifies 2,294 fit and 406 stop origins, observed-window completeness and their completion purge; recomputing the original fit-only scaler matches every stored field bitwise. Validation decoding follows selection freeze. No train.npy reconstruction is claimed. The later publication permission does not rewrite the original data-exposure identity.

Synthetic continuous four-epoch, per-batch disk recovery and isolated cap2→4 runs agree bitwise. These one-thread fixtures provide engineering evidence. The primary market continuation and its continuous replay use the original launcher's documented two Torch intra-op threads. The original shell thread variables, precise CPU affinity and inter-op setting remain unknown. The corrective executions use logical CPUs 24/26 and inter-op count one; the retained evidence supports matching the documented intra-op count, without claiming full reconstruction of the original execution resources.

### S27.2 Stopping and the unchanged selected prediction

The continuation performs 54 additional optimizer updates over three epochs and stops at epoch 23 under patience three. No new stopping loss beats the epoch-20 minimum.

| Epoch | Inner-stop endpoint MSE | Selected best epoch |
|---|---:|---:|
| 20, original | 0.0006369894448303129 | 20 |
| 21 | 0.0006371416688671142 | 20 |
| 22 | 0.0006384256933739521 | 20 |
| 23 | 0.0006379284747988018 | 20 |

After freezing the selected best, the worker scores the original exposed outer block of 1,356 overlapping paths. Old and continued predictions are byte-identical.

| Predictor | Outer endpoint MSE | Outer complete-path MSE |
|---|---:|---:|
| Original cap20 selected GRU | 0.0003297163258494623 | 0.00018398185026632948 |
| Cap60 continuation selected GRU | 0.0003297163258494623 | 0.00018398185026632948 |
| Zero cumulative return | 0.0003285151290547691 | 0.00018312938766350156 |

The gain over cap20 is zero; the endpoint gain over zero is −0.3656443%. This cell supplies no method improvement. Under the retained stopping rule, the original cap did not alter the final selection. The result does not establish full optimization of the thirty-nine earlier stops, optimal patience, or the behavior of longer training under a different stopping rule. No new seed, candidate or protected market period enters the diagnostic.

### S27.3 Continuous replay, metadata exceptions and scope

The primary worker calls the original run_fit fifty-five times: fifty-four calls interrupt after one update and the last completes the final epoch's stopping step. An uninterrupted two-thread call repeats the same 54 updates and matches the final latest/best payloads and complete summary bitwise. The replay's independent Python scalar squares and math.fsum match thirty MSE quantities within the predeclared absolute tolerance 1e-15; the largest absolute difference is 4.87890977618477e-19. It also matches validation-index bytes, reconstructs archived labels from the admitted bars and confirms the old/new prediction bytes. Shared data-admission helpers limit this independence to execution trajectory and metric reduction. It does not establish independent data access or an unseen-market result.

The initial one-thread continuation and replay remain auxiliary records. The original launcher's two-thread setting was identified during final resource review; the corrected two-thread execution and replay retained the same absolute stage deadline. Both thread settings give the same full history and reported metrics. The auxiliary independent protocol also retains three inherited metadata fields that disagree with its dedicated purpose and actual source. The archive discloses them without rewriting the frozen protocol:

| Inherited auxiliary field | Actual auxiliary replay and consequence |
|---|---|
| checkpoint_after_every_update=true | One continuous original call used periodic/epoch checkpoints. It did not test per-update timeout recovery; the main stepwise worker saved per update. |
| actual_train_or_validation_array_reconstruction_claimed=false | The replay reconstructed validation.npy and compared origin bytes. It did not decode train.npy or establish complete train-array reconstruction. |
| outer_scoring_admitted=false | The dedicated replay decoded and reduced archived development predictions. It produced no new model forward, period or candidate; the generic flag did not describe that archived-metric operation. |

The corrected two-thread replay protocol states its original periodic/epoch checkpoints, validation reconstruction and archived-metric reduction. Its successful uninterrupted completion must not be described as an interrupted-market recovery test. The primary two-thread continuation and replay perform 108 market updates in total; with the auxiliary one-thread pair, this stage performs 216. These repeated updates do not add independent market observations.

The workers retain original pre-Q3 history and the exposed outer identity. Python audit hooks and frozen-source review constrain access but are not operating-system sandboxes. The production loop checks the fixed deadline between updates, atomically saves state and reserves time for completion; the accepted market runs did not require deadline recovery. Planning estimates are not guaranteed runtime bounds. GPU remains disabled, no paid resource is used and Q3 remains sealed. Matching the saved state and reported metrics verifies computation. A stable improvement over strong controls and eligible independent confirmation remain outstanding.

## S28. Matched function-centering design and complete results

S1–S27 retain the immediately preceding edition's text. This later probe asks whether the original deterministic one-layer LSTM32 benefits from a physical-zero initial prediction while retaining its default initial parameter Jacobian. It is a new matched diagnostic in already exposed historical development data, not a reopening of a previously failed fixed recipe or a claim to a new architecture.

### S28.1 Three recipes and their interpretation

For raw cumulative-return path \(y\), prefix moments give \(z=(y-\mu)/\sigma\) and physical output \(\hat y=\mu+\sigma\odot q\). D uses \(q_D=f_\theta(x,a)\) with default state \(\theta_0\). Z uses the same initial recurrent backbone with zero head weights and bias \(b=-\mu/\sigma\), evaluated as an FP64 ratio then cast to FP32. C uses

\[
q_C=\bigl(f_\theta(x,a)-f_{\theta_0}^{\mathrm{ref}}(x,a)\bigr)+b.
\]

The C learner starts at exactly D's default state; its independent initial reference is frozen and excluded from Adam. In exact arithmetic C and Z begin at physical return zero. At fixed input, parameter state and mode, the initial trainable-parameter Jacobian of C equals D's, since the reference contributes no trainable derivative. Their different residuals do not imply equal loss gradients. C/Z differ in initial readout parameters, a fixed nonlinear offset and their subsequent optimization trajectories. The translated class is \(\mathcal F-f_{\theta_0}^{\mathrm{ref}}+b\); equality with \(\mathcal F\) is not assumed. The comparison estimates a total recipe effect rather than an isolated gradient-delay cause. C has the same trainable parameter count as D, twice the stored network parameters, and one additional forward pass during training and inference.

Earlier separate checks found zero first-step and nonzero second-step backbone gradients in 120 baseline configurations: ninety xPatch and thirty LSTM. Those thirty LSTM checks all concern hourly-H1 price prediction, not replications of the present H60/H8 return experiment. A first-step restriction need not persist; these checks do not establish the mechanism of the later market losses.

### S28.2 Fixed cohorts, inputs and optimization

The minute lookback/horizon are L480/H60 and hourly L96/H8. For an original fit set of size n, the chronological boundary is floor(4n/5). The tail retains every original origin from that boundary onward. An earlier origin enters the update prefix only if its complete label ends strictly before the first tail origin's completion. We check both origin-vector digests. The tail itself is unchanged.

**Table S28.1.** Original and purged cohorts. Boundary is the zero-based split position, not a retained-origin count.

| State | Original fit | Boundary | Prefix | Tail |
| --- | ---: | ---: | ---: | ---: |
| Minute / 1 | 64,261 | 51,408 | 51,391 | 12,853 |
| Minute / 2 | 128,761 | 103,008 | 102,948 | 25,753 |
| Hourly / 1 | 1,139 | 911 | 903 | 228 |
| Hourly / 2 | 2,294 | 1,835 | 1,827 | 459 |

Each arm receives the same full origin-relative log-close history, with only splits effective at the origin, and the same nominal closing-price anchor. Prefix-only population moments standardize histories, anchor and horizon-specific targets. The ordered scaler reductions use blocks of 4,096 origins and preserve the original FP32 transformed-input/target boundary. No scaler from the complete original fit cohort is reused.

Seed 2026 initializes each matched backbone; a separate NumPy generator with seed 2026 fixes shared epoch permutations. All twelve learners complete eight epochs. The four states use 3,216/6,440/64/120 common batch positions and 9,648/19,320/192/360 optimizer updates, totaling 9,840 shared positions and 29,520 updates. Batch size is 128; a lone final sample merges into the preceding batch. Uniform standardized complete-path MSE and Adam are fixed: learning rate 0.001, betas 0.9/0.999, epsilon 1e−8 and zero decay. There is no scheduler, clipping, early stopping or best-epoch choice. The final epoch is used. This equal epoch count does not imply equal training cost for C, or sufficient optimization of every architecture.

Controls are D, Z, zero returns, the prefix horizon-wise mean and same-information Ridge. Ridge takes the same FP32 transformed full history and anchor, then centers/scales columns in FP64 and fits raw multi-output returns with an unpenalized prefix-mean intercept and normalized penalty 0.1. The penalty is fixed before outcomes; it differs from the original study's Ridge selection procedure. Numerical qualification includes finite coefficients, a variance-roundoff check and normwise normal-equation backward error no greater than 1e−10.

All twelve learners and four fitted-control objects are committed before the dedicated tail prediction pass. The worker reloads the committed payloads. No dedicated tail target is requested before the global freeze; state-specific update permissions and nested folds still prevent a globally blind interpretation. We score raw-return endpoint and full-path MSE for all six predictors on the same origins.

### S28.3 All predictor losses

**Table S28.2.** Complete raw-return losses and gains against zero. Positive gains mean lower MSE. Ridge denotes the fixed same-information control; Prefix mean denotes the horizon-wise update-only mean. All values come from the accepted saved market result, independently reconstructed from saved predictions.

| State | Predictor | Endpoint MSE | Path MSE | Endpoint gain | Path gain |
| --- | --- | ---: | ---: | ---: | ---: |
| Minute / 1 | D | 1.6564261782237898e-05 | 8.5867460315480143e-06 | +0.313475033% | +0.122036643% |
| Minute / 1 | Z | 1.6519951244541868e-05 | 8.5746635349756654e-06 | +0.580143332% | +0.262575928% |
| Minute / 1 | C | 1.6555104771028255e-05 | 8.5972055506895158e-06 | +0.368583473% | +0.000375251% |
| Minute / 1 | Zero | 1.6616349890518676e-05 | 8.5972378119493076e-06 | +0.000000000% | +0.000000000% |
| Minute / 1 | Prefix mean | 1.6677404245969916e-05 | 8.6195766032051916e-06 | -0.367435423% | -0.259836842% |
| Minute / 1 | Ridge | 1.6988342014369869e-05 | 8.7901374446376674e-06 | -2.238711428% | -2.243739639% |
| Minute / 2 | D | 4.3426029608273028e-05 | 2.3343893694061358e-05 | -0.677360784% | -0.432482471% |
| Minute / 2 | Z | 4.3042336116577278e-05 | 2.3183399942572629e-05 | +0.212180545% | +0.258010131% |
| Minute / 2 | C | 4.310133208774144e-05 | 2.3187782300008132e-05 | +0.075406386% | +0.239155905% |
| Minute / 2 | Zero | 4.3133857771117002e-05 | 2.3243370192377272e-05 | +0.000000000% | +0.000000000% |
| Minute / 2 | Prefix mean | 4.3186639888650784e-05 | 2.3263287180171635e-05 | -0.122368182% | -0.085688898% |
| Minute / 2 | Ridge | 4.3248955690452851e-05 | 2.3352538977221648e-05 | -0.266838918% | -0.469677090% |
| Hourly / 1 | D | 0.00022463709180077705 | 0.00013191546277992628 | +1.775211032% | +0.786994871% |
| Hourly / 1 | Z | 0.00022722698093725785 | 0.00013218407376749174 | +0.642756405% | +0.584973798% |
| Hourly / 1 | C | 0.00022717993051497189 | 0.00013215124779969746 | +0.663329667% | +0.609662056% |
| Hourly / 1 | Zero | 0.00022869694520114722 | 0.00013296186584427444 | +0.000000000% | +0.000000000% |
| Hourly / 1 | Prefix mean | 0.00023031776271093088 | 0.00013356606099136506 | -0.708718478% | -0.454412356% |
| Hourly / 1 | Ridge | 0.00022326420269646683 | 0.00013070598032604243 | +2.375520364% | +1.696640991% |
| Hourly / 2 | D | 0.00067643444461860849 | 0.00038192102723058965 | +0.707433207% | +0.427269135% |
| Hourly / 2 | Z | 0.00067708550170304587 | 0.00038250571748066895 | +0.611865737% | +0.274831325% |
| Hourly / 2 | C | 0.00067781670339650015 | 0.00038217125131640367 | +0.504533986% | +0.362031837% |
| Hourly / 2 | Zero | 0.00068125386065533123 | 0.00038355986012552457 | +0.000000000% | +0.000000000% |
| Hourly / 2 | Prefix mean | 0.00068235694535790053 | 0.00038401339230681924 | -0.161919773% | -0.118242869% |
| Hourly / 2 | Ridge | 0.00069155213956673643 | 0.00038919491871899713 | -1.511665402% | -1.469147108% |

### S28.4 Complete fixed continuation decisions

C must reduce both MSE measures by at least 0.1% relative to each of five controls in every state. Independently for each task, the arithmetic mean of the two fold-relative endpoint improvements against zero must reach 1%. These conditions are conjunctive; no state subset, alternative seed, epoch budget, penalty or backend can rescue this fixed probe. Passing would only admit review of a formal five-fold, three-seed historical-development recipe, without opening protected data.

**Table S28.3.** All forty control-metric comparisons. Gains use each row's control as denominator. Pass/Fail applies the predeclared 0.1% margin to the unrounded MSE values.

| State | Control | Endpoint gain | Endpoint gate | Path gain | Path gate |
| --- | --- | ---: | --- | ---: | --- |
| Minute / 1 | D | +0.055281734% | Fail | -0.121810044% | Fail |
| Minute / 1 | Z | -0.212794372% | Fail | -0.262890965% | Fail |
| Minute / 1 | Zero | +0.368583473% | Pass | +0.000375251% | Fail |
| Minute / 1 | Prefix mean | +0.733324402% | Pass | +0.259537719% | Pass |
| Minute / 1 | Ridge | +2.550203210% | Pass | +2.194867773% | Pass |
| Minute / 2 | D | +0.747702526% | Pass | +0.668746166% | Pass |
| Minute / 2 | Z | -0.137064984% | Fail | -0.018902997% | Fail |
| Minute / 2 | Zero | +0.075406386% | Fail | +0.239155905% | Pass |
| Minute / 2 | Prefix mean | +0.197532851% | Pass | +0.324566686% | Pass |
| Minute / 2 | Ridge | +0.341334491% | Pass | +0.705519333% | Pass |
| Hourly / 1 | D | -1.131976333% | Fail | -0.178739486% | Fail |
| Hourly / 1 | Z | +0.020706354% | Fail | +0.024833527% | Fail |
| Hourly / 1 | Zero | +0.663329667% | Pass | +0.609662056% | Pass |
| Hourly / 1 | Prefix mean | +1.362392617% | Pass | +1.059260999% | Pass |
| Hourly / 1 | Ridge | -1.753853852% | Fail | -1.105739363% | Fail |
| Hourly / 2 | D | -0.204344824% | Fail | -0.065517232% | Fail |
| Hourly / 2 | Z | -0.107992520% | Fail | +0.087440827% | Fail |
| Hourly / 2 | Zero | +0.504533986% | Pass | +0.362031837% | Pass |
| Hourly / 2 | Prefix mean | +0.665376383% | Pass | +0.479707486% | Pass |
| Hourly / 2 | Ridge | +1.986175067% | Pass | +1.804665751% | Pass |

The state pass counts are 5/10, 7/10, 4/10 and 6/10 in chronological task/fold order. Twenty-two comparisons pass and eighteen fail, so all four state-level conjunctions fail. The equal-fold zero-endpoint gains are 0.2219949295% for minute H60 and 0.5839318268% for hourly H8; both fail 1%. All eight C-versus-Z margin tests fail, although hourly fold 1 has small raw reductions in both losses and hourly fold 2 has a small path reduction. This fixed C recipe is closed and no formal forecasting candidate is admitted.

## S29. Reconstruction, exposure and limits of the centering evidence

### S29.1 Execution and independent numerical checks

The numerical worker uses logical CPU24, one Torch intra-op and one inter-op thread, Torch 2.14.1+cu130 with the default enabled MKLDNN, deterministic algorithms and GPU access disabled. Its two-hour absolute deadline is not reset. Market execution completes normally in 1,312.428 seconds. Checkpoint generations preserve learners, reference states, scalers, baseline, Adam, batch permutations/cursor, epoch loss accumulators, random states and exact source/protocol/partition/backend identities. Preparation includes an actual external-signal exit/new-process resume and an injected expired-local-deadline exit/resume, with 6,480 payload-leaf comparisons against continuous synthetic runs. These are recovery checks, not market performance. The noniterative Ridge solver has no internal checkpoint; after interruption it would recompute from committed final neural state under an eligible continuation.

An independent script, run after the original worker exited, rebuilds the raw partitions, features, scalers and fixed Ridge fits, then replays D/Z/C outputs from this stage's final committed models. It extracts only the original Recurrent class AST and does not import the worker or native-core implementation. All 435 checks pass. It verifies exact origins and raw labels, standardized neural-output agreement with atol 1e−7 and rtol 1e−5, and scaler/Ridge/control agreement with atol 1e−10 and rtol 1e−8. The fixed Ridge backward-error bound remains 1e−10. This establishes output reconstruction within the declared tolerances, not bitwise identity of all neural arrays or independent retraining.

A separate saved-prediction program, without Torch or model helpers, reconstructs 216 score quantities with absolute/relative tolerances 1e−18/1e−12 and reproduces the complete failed continuation decision. Lifecycle/scientific-gate review has 98 checks, and the pre-integration addendum has 49 claim checks. These counts describe distinct verification layers, not market sample sizes, additional random seeds, or independent economic evidence.

### S29.2 Data exposure and access boundaries

The second-fold update prefix contains all 12,853 first-fold minute tail origins and all 228 first-fold hourly tail origins. All cohorts also belong to previously exposed development history. The global prediction freeze prevents within-run tail feedback; it does not make these nested historical observations globally unseen.

Both admitted pre-Q3 bar archives decode the prescribed five fields in full: start/end timestamps, close, observed status and session date. Each split archive decodes only its literal fit.npy member. The original stop/outer members and protected 2026Q3 values are not read. Python audit hooks and exact path permissions constrain this worker; they are not an operating-system sandbox. The independent reconstruction repeats the disclosed input decoding and exposure accounting. A small documentation bundle does not itself execute these numerical procedures.

### S29.3 What the negative result does and does not establish

At this single seed and finite budget, the specific centered recipe does not satisfy its prespecified superiority conditions. Four small positive endpoint differences against zero do not establish stable signal, general C-over-D/Z superiority, or a reason to fit a new fusion. The result also does not establish market unpredictability or impossibility of all centering designs. Initialization, translated function class, implicit regularization and optimization remain combined. No new candidate, protected period or expanded training is inferred from numerical reproducibility. The original title, five scales and twenty tasks remain; stable improvement and eligible unseen-period confirmation remain unachieved.

### S29.4 Separate descriptive-accounting nonqualification

A subsequent, separately frozen zero-training/zero-inference calculation attempts a per-state, per-horizon error decomposition using the already accepted predictions. Its 132 internal checks pass, but its independent scalar verifier exits at the first percentage-sum comparison. The observed discrepancy is approximately 3.10e−14 percentage points, above its frozen approximately 3.76e−16 bound. A source/saved-scalar review finds seven of forty-eight percentage-sum inequalities outside that unchanged condition. The same absolute tolerance is not invariant to converting raw MSE to percentage units. Source, threshold and formula are not changed, and neither program is rerun to obtain a pass. The calculation is recorded as partial numerical nonqualification and supplies no dynamic-contribution table or new mechanism claim. This separate engineering failure does not revoke the accepted market preprocessing/output replay or score reconstruction above. The documentation retains failure receipts and source identities in evidence/accounting_failure, rather than presenting its unchecked decomposition as scientific evidence.

## S30. Artificial controls for finite learning and competing explanations

These artificial diagnostics supplement *Multi-Horizon Forecasting of Apple Stock Prices and Returns Across Temporal Scales*. They assess whether a specified finite recurrent pipeline recovers a known conditional mean and examine narrowly defined competing explanations for its failures. They do not extend the AAPL information set or change the five-scale, twenty-task study. Later stages were chosen after earlier artificial results had been inspected; they are adaptive development diagnostics, not independent confirmation.

For endpoint or raw equal-horizon path predictions \(p\), define conditional excess risk on a fixed artificial input design as \(R(p)=\operatorname{mean}((p-m)^2)\), where \(m\) is the known conditional mean. For positive signal, the recovery fraction is \(1-R(p)/R(0)\). We evaluate error relative to the known conditional mean instead of a realization of future label noise. This recovery measure does not quantify realized market-MSE improvement or trading return. Negative recovery means greater error against m than the zero predictor. Repeated training seeds share an artificial data design and do not constitute independent market histories.

**Table S30.1.** Artificial diagnostic sequence. Counts refer to fixed checks or comparisons within the stated stage, rather than independent market experiments.

| Case | Intervention | Fixed result | Limit |
| --- | --- | --- | --- |
| Known-mean control | Native LSTM; positive and null signals | OLS 4/4; native positive-signal half-recovery 6/12 | L480/H60 fails; L96/H8 passes; L and H co-vary |
| Coordinate pair | Exactly reversible lattice C versus D | Relative and half-recovery each 8/12 | Both overall conjunctions fail; new DGP |
| Saved states | Selected versus terminal; zero updates | No half-threshold crossings in 72 pairs | Two saved states; no counterfactual stopping policy |
| Fixed-H60 context | Delete specified old prefix | Relative improvement 0/6; half-recovery 2/6 | Sufficient signal retained; full information differs |
| Initial gradient | Noisy-label versus mean-label proposals | 72/72 local descent; 0/36 reversals | First batch, zero updates; same autograd backend |

### S30.1. Known conditional mean under the original finite recurrent recipe

The first calibration retained the native width-32 LSTM, default initialization, Adam learning rate 0.001, batch size 128, maximum 20 epochs, and patience 3. Fitting minimized standardized full-path squared error; stopping selected the lowest raw endpoint MSE. Independent artificial fit, stop, and test designs contained 2,048, 512, and 4,096 origins. The configurations were L480/H60 and L96/H8, each with signal strength alpha=0.25 or alpha=0 and seeds 2026, 2027, and 2028: twelve learners in total.

Origin-relative cumulative Gaussian histories were normalized with a fit-only scalar to form the actual FP32 input X. The signal was defined from that visible input, \(z=\sqrt{L}\,x_{\mathrm{std}}[\operatorname{float64}(X_{\mathrm{last}})-\operatorname{float64}(X_{\mathrm{penultimate}})]\), and targets were \(Y_h=\alpha h z/\sqrt{H}+\sum_{j\le h}\epsilon_j\), with independent standard Gaussian innovations. Thus \(m_h(X)=\alpha h z/\sqrt{H}\) was known from the actual input and fitted scalar, rather than an inaccessible prequantization increment. The constant anchor was 100 and standardized to zero. All models and controls were frozen before a single artificial test pass.

A fit-only intercept-plus-z OLS control passed all four shape-by-metric 50% recovery checks: endpoint/path recovery was 99.080449%/98.487532% at L480/H60 and 99.324022%/98.968325% at L96/H8. OLS used an oracle-informed representation derived from the input; equal information does not imply an equal function class or inductive bias. Native endpoint recovery, in seed order, was 25.141867%, −8.453784%, and −8.924383% at L480/H60, versus 60.692558%, 58.045863%, and 58.905958% at L96/H8. All six long-configuration endpoint/path checks failed the 50% requirement; all six short-configuration checks passed. The full requirement was therefore not met. The twelve fits used 119 epochs and 1,904 updates. Null cases were described only by prediction energy relative to known positive noise risk, without a post hoc recovery gate.

This supplies a finite artificial counterexample to inferring weak available conditional mean solely from a model's failure to beat zero. It does not establish that the artificial signal exists in AAPL. Exact global representability of the Gaussian oracle by this recurrent pipeline is unproven. Length and output horizon vary jointly, while independent artificial origins omit market overlap, sessions, and drift; a pure representation, optimization, capacity, or economic explanation is not identified.

### S30.2. Finite-precision qualification and a distinct lattice coordinate pair

An ordinary standardized FP32 first-difference transform failed its pretraining exact-visible-signal qualification at `exact_signal:480:fit`. A whole-history relative reconstruction error of approximately 2.56×10^−8 did not establish exact signal equivalence. That recipe was closed with zero training, without relaxing its gate.

A separate artificial design then quantized Gaussian cumulative histories once to a fixed q=2^−12 lattice. Cumulative coordinates C and full first-difference coordinates D each used a fit-only power-of-two maximum-absolute scale without centering. Decoding the actual FP32 inputs exactly restored their complete common integer history, visible signal, and conditional mean. This was a new DGP with two freshly trained arms, not a rescue of the rejected transform or a comparison against the earlier Gaussian baseline.

The twelve fits used the two original L/H configurations, alpha=0.25, the same three seeds and sample counts, and the fixed recurrent recipe. Initial states, target scalers, and planned epoch permutations matched within pairs. The common prefixes of realized epoch permutations also matched; differing stopping times and update counts remained part of the intervention. All four unique OLS checks passed. Differences reduced conditional excess risk by at least 10% in 8/12 endpoint/path comparisons and achieved 50% recovery in 8/12. Both complete conjunctions failed. All six L96/H8 comparisons passed; at L480/H60 only seed 2026 passed both metrics.

In seed order, C versus D endpoint recovery was (−5.775284%, 70.544532%), (−7.936319%, −9.765795%), and (−9.121577%, −9.205899%) at L480/H60; the corresponding pairs at L96/H8 were (11.552967%, 91.693509%), (0.625394%, 92.093307%), and (21.107577%, 88.639120%). The twelve fits used 142 epochs and 2,272 updates. These results show sensitivity of the combined coordinate, normalization, finite-training, and stopping pipeline to information-equivalent inputs in this design. They neither isolate one component nor meet the predefined cross-configuration success criterion. The fixed recipe was closed without seed, scale, budget, or threshold changes and without admitting a market method.

### S30.3. Selected and terminal states do not reverse the recovery classification

A zero-update audit compared the selected and actual terminal states of the twelve lattice learners. The 144 metric records covered both states, fit/stop/test splits, and endpoint/path risk. None of the 72 selected/terminal pairs crossed the original half-recovery threshold. None of the 48 state-by-metric fit/test pairs disagreed in their half-recovery classification. These are dependent descriptive comparisons of the same saved models, not fresh experiments or a revision of the original success gates.

For the failing long-D seeds 2027 and 2028, terminal artificial-test endpoint recovery improved to 3.631595% and 2.270838%, respectively, from the selected states' negative recovery, but stayed below 50%; the corresponding terminal fit states also remained below half recovery. Thus replacing selection with these particular saved terminal states would not repair the classification. Continuous risk can nevertheless improve without crossing that threshold. The audit did not retain every epoch, and a terminal state is not the counterfactual state reached under a different patience policy. Neither unchanged binary classifications nor fit/test agreement proves that stopping is irrelevant, longer training is futile, or overfitting is absent.

### S30.4. Deleting an old prefix with the horizon and data held fixed

The next intervention kept the original L480/H60 lattice signal, noise, labels, target scalers, full-D input scale, H60 readout, initial parameter values, and planned epoch permutations. Three fresh learners instead received [0, the last 95 original differences], a context of 96. Actual FP32 decoding exactly retained the suffix, visible signal, and conditional mean. The removed prefix supplied no extra conditional-mean information given that suffix; it was not asserted to be independent noise. This retained sufficient signal while removing some full-history information. The original full-context models and results stayed frozen and their weights were not loaded in this intervention.

The three fits used 29 epochs and 464 updates. Both unique OLS checks qualified. None of six endpoint/path comparisons achieved the required 10% relative-risk reduction; only the two seed-2026 comparisons recovered half the signal. Both complete requirements failed. The trained/selected epochs were 20/19, 5/2, and 4/1, matching the full-context counterparts. The maximum absolute relative risk change across six estimates was 1.2418749040232768×10^−7. This is a descriptive fixed-sample difference, not a statistical equivalence result.

Deleting this specified prefix did not remedy the fixed D-LSTM deficit. It narrows the claim that shorter context alone explains the earlier L480/H60 versus L96/H8 contrast; it does not identify horizon as the cause or show that all context lengths are irrelevant. The precise deletion recipe was closed without a market candidate.

### S30.5. The initial direction audit does not support a local noise reversal

At each original full-D L480/H60 initialization, the original first shuffled fit batch and noisy-fit target scaler were reused. Native FP32 gradients were computed for noisy labels and for FP32-standardized known-mean labels. Each gave an idealized FP64 first-Adam proposal, \(\delta=-0.001g/(|g|+10^{-8})\), with no optimizer step, parameter perturbation, or stop/test model evaluation. Parameters and buffers remained unchanged.

All 72 prespecified directional records indicated local descent beyond a fixed numerical-resolution margin. We used \(256\epsilon_{32}\|g_{\mathrm{ref}}\|_2\|\delta\|_2+10^{-12}\) as a numerical resolution policy, not a statistical confidence bound. They covered the three seeds, two proposals, first-batch and fixed-128-block full-fit references, scaled and raw known-mean objectives, and all/head/backbone parameter groups. None of 36 paired comparisons showed a non-descending noisy-label proposal and a descending mean-label proposal. The records are dependent descriptions, not independent significance tests. A negative directional derivative does not establish improvement at the actual finite step size, later in training, or under another stopping policy. Scaled and raw objectives use different horizon weights.

A seven-component signed VJP accounting separated initial output, true mean, fit centering, Brownian noise, FP64 normalization arithmetic, FP32 target conversion, and output-cotangent arithmetic; parameter closure was recorded separately. The frozen numerical qualification criteria passed for all seeds. Proposals used direct gradients rather than reconstructed sums. This accounting does not establish a dominant causal component: fit scalers depend on the same noisy labels, Brownian horizons are correlated, and head/backbone projections are not interventions.

Separate implementations regenerated artificial designs and checked the saved predictions or audit quantities against the stage-specific frozen criteria. For the gradient audit both used the same PyTorch autograd backend; this was not an independent Jacobian engine. Original engineering failures remain in the record. The first calibration's preparation/read-guard failures and subsequent verifier environment/report failures were repaired without retraining or changing numerical science gates, with the original failed supervisor receipt retained. The gradient replay initially failed when serializing a NumPy float32 value into JSON. A supplemental output-only adapter converted that actual type to Python float without changing numerical calculations, formulas, thresholds, or the absolute deadline; only replay was rerun, and the original failure was preserved.

The local audit therefore does not support the specified explanation that noisy labels reverse the initial idealized Adam direction on the known-mean objective. It does not show that label noise is harmless later, explain the complete failed trajectory, or justify a new loss or oracle-supervised market model. Across these artificial stages, we find a scoped finite-pipeline recovery deficit. The specified repairs fail their complete criteria, and the initial local reversal explanation lacks support. They do not establish a natural-market cause or forecasting gain. No market candidate was admitted, 2026Q3 remained sealed, and stable gains over strong controls plus eligible independent market confirmation remain unachieved.

## S31. Fixed completed-bar close-location information screen

### S31.1 Question, source and timing

The earlier close/volume representations cannot determine the high and low of a completed bar. We therefore ask whether one added close-location value supplies incremental prediction within a fixed Ridge procedure. For the latest completed original-frequency bar, CLV=((C−L)−(H−C))/(H−L); a valid H=L bar maps to zero. This is a location within the final range. It does not reveal the order of intrabar events, transaction direction, a quote midpoint or economic causality. We add no other OHLC field, lag, interaction or window.

The hash-bound original Alpaca SIP/raw source ends on June 30, 2026. All 1,061,536 CSV rows pass finite positive OHLC and L≤O,C≤H checks; invalid source rows would fail the whole stage. We align 535,620 regular minutes exactly to the original timestamps and close values. Original timestamps mark bar starts; completed OHLC is assumed available no earlier than bar completion. Actual historical receipt latency and revisions remain unresolved. This source qualification therefore supports the existing final-bar historical study, not a new point-in-time deployment claim.

For each original full sixty-minute group from the session open, hourly OHLC takes the first open, maximum high, minimum low and last close. The resulting 8,238 complete hours match original start/end/session/close arrays. The original 1,378 sessions and exclusion of 41,340 residual minutes remain unchanged. Hourly CLV is calculated after aggregation, never by averaging minute CLV. There is no clipping, epsilon, winsorization or filling. Neither qualified frequency contains a flat-range bar. Source qualification loads no targets or models.

### S31.2 Fixed design and numerical prerequisites

The four states are minute H60/L480 and hourly H8/L96, folds 1/2. A uses the full original FP64 log-relative close window, nominal anchor and prior seventeen close/time features. B adds the unchanged past-volume anomaly u and return_lag_H×u. C adds only the latest completed-bar CLV. The old volume normalization uses the median log1p volume at the same slot across the previous twenty global completed sessions, requires all twenty slots, excludes today’s session, and averages anomalies over the past H bars. The duplicated close-derived A columns stay in all arms; they can affect Ridge’s penalty geometry. We do not claim bitwise equality with the original neural FP32 input representation.

Each original unfiltered fit list sets its eighty-percent boundary before filtering. Prefix labels must complete strictly before the original tail’s first decision. The unchanged past-volume availability mask then applies to all arms and controls. We require at least one hundred prefix and twenty tail origins per state. No original stopping, outer or protected-quarter score is constructed. The two folds reuse exposed historical development periods; a global within-run freeze does not make their observations independent.

All column means/scales, target means and coefficients use only the common prefix. We fit raw cumulative close-log-return paths using normalized Ridge penalty 0.1 and an unpenalized intercept. Ordinary column standard deviations at most 1e−12 map to one, while a CLV standard deviation at most 1e−12 fails its feature prerequisite. The covariance diagonal, normwise stationarity and direct-centered reconstruction rules are fixed before numerical execution.

Before solving any predictor or scoring a tail, all four states must pass the CLV residual prerequisite. For standardized B, the Gram eigenspace retains eigenvalues above 1e−10 of the largest; a nonpositive largest value or a smallest value below −1e−8 of that maximum fails. Direct prefix residual sum of squares, divided by standardized CLV sum of squares, must exceed 1e−8. This describes residual variation against that retained numerical subspace, not exact full-span or population conditional-mean nonredundancy. All four states pass.

**Table S31.1.** Common development cohorts and CLV residual qualification.

| State | Prefix | Tail | Residual ratio |
| --- | ---: | ---: | ---: |
| Minute H60 / 1 | 45,151 | 12,853 | 68.746728% |
| Minute H60 / 2 | 93,128 | 25,753 | 68.605724% |
| Hourly H8 / 1 | 871 | 228 | 58.900010% |
| Hourly H8 / 2 | 1,671 | 459 | 60.808310% |

All twelve finite predictors are committed before any dedicated tail pass. We score zero-return and prefix-only mean controls at the identical origins. We fit no output scalar, choose no sign and search no penalty or model.

### S31.3 All state losses and the failed continuation decision

Every state requires C’s raw endpoint MSE to be strictly below A, B, zero and the prefix mean, with full-path MSE no greater than A and B. Both primary tasks separately require at least a one-percent reduction over zero: \(G=1-\frac{\mathrm{MSE}_{C,1}+\mathrm{MSE}_{C,2}}{\mathrm{MSE}_{0,1}+\mathrm{MSE}_{0,2}}\ge 0.01\). This uses a ratio of equal-fold MSEs, not the mean of foldwise percentage gains or pooled-origin SSE. All state and task conditions form one conjunction. Scientific comparisons use unrounded values, without a numerical-tolerance rescue.

**Table S31.2.** Complete tail losses. Positive endpoint gains indicate lower MSE relative to zero; prefix_mean uses only the fitting prefix. Full-path MSE averages all H raw cumulative-return coordinates.

| State | Arm | Endpoint MSE | Path MSE | Endpoint gain |
| --- | --- | ---: | ---: | ---: |
| Minute H60 / 1 | A | 1.73499341548e-05 | 8.88676235554e-06 | -4.414834% |
| Minute H60 / 1 | B | 1.73551661786e-05 | 8.90413144942e-06 | -4.446321% |
| Minute H60 / 1 | C | 1.73596996453e-05 | 8.90537668913e-06 | -4.473604% |
| Minute H60 / 1 | zero | 1.66163498905e-05 | 8.59723781195e-06 | +0.000000% |
| Minute H60 / 1 | prefix_mean | 1.66696651494e-05 | 8.61595607094e-06 | -0.320860% |
| Minute H60 / 2 | A | 4.38716800572e-05 | 2.35244093752e-05 | -1.710541% |
| Minute H60 / 2 | B | 4.36355882441e-05 | 2.34554601815e-05 | -1.163194% |
| Minute H60 / 2 | C | 4.36345277367e-05 | 2.34549615314e-05 | -1.160735% |
| Minute H60 / 2 | zero | 4.31338577711e-05 | 2.32433701924e-05 | +0.000000% |
| Minute H60 / 2 | prefix_mean | 4.31782819227e-05 | 2.32609096745e-05 | -0.102991% |
| Hourly H8 / 1 | A | 0.000229025054832 | 0.000131279589936 | -0.143469% |
| Hourly H8 / 1 | B | 0.000231428175025 | 0.000132346905988 | -1.194257% |
| Hourly H8 / 1 | C | 0.000231381288731 | 0.000132340146313 | -1.173756% |
| Hourly H8 / 1 | zero | 0.000228696945201 | 0.000132961865844 | +0.000000% |
| Hourly H8 / 1 | prefix_mean | 0.000232241373834 | 0.000134381453335 | -1.549836% |
| Hourly H8 / 2 | A | 0.000700049297449 | 0.000395377968614 | -2.758948% |
| Hourly H8 / 2 | B | 0.000691606581635 | 0.000391408220282 | -1.519657% |
| Hourly H8 / 2 | C | 0.000692227079836 | 0.00039159939025 | -1.610739% |
| Hourly H8 / 2 | zero | 0.000681253860655 | 0.000383559860126 | +0.000000% |
| Hourly H8 / 2 | prefix_mean | 0.00068204579204 | 0.000383891548054 | -0.116246% |

No state meets the complete rule (0/4); C loses to zero at all four endpoints. Equal-fold endpoint gains are −2.0820341368% at minute H60 and −1.5009122056% at hourly H8. C-versus-B endpoint gains are −0.026122%, +0.002430%, +0.020260% and −0.089718%, in table order. The mixed directions do not support stable incremental prediction. Both one-percent task conditions fail. We close this exact linear recipe without selecting a favorable fold, alternate horizon, additional CLV lag or another penalty.

### S31.4 Reconstruction, recovery and interpretation

An external controller sends SIGUSR1 after the first 2,048 prefix observations have been accumulated and saved. The worker records a paused state and preserves checkpoint bytes. It resumes from that cursor with the same source/protocol identity and absolute deadline, then completes all twelve predictors. The first worker’s nonzero exit is the predeclared signal pause. Sufficient statistics are committed per batch; tail scores are committed per completed state. An interrupted tail pass can be recomputed without losing fitted-model progress. The first maximum-shape batch uses about 216.52 MiB peak RSS and passes the frozen cost rule.

A separate implementation rebuilds original origins, label-completion purge, common volume mask, OHLC grouping, CLV and directly centered Ridge matrices. It calls none of the main solve, prediction, partition or decision functions. It reconstructs the twelve solutions and all scores under fixed rtol1e−9/atol1e−11, verifies 16,716 identity/numerical/decision comparisons, and matches every scientific decision; the maximum absolute numerical difference is about \(2.696510\times 10^{-12}\) across different compared quantities. Both implementations reuse the identified original close-window/context code and the same exposed market history. This is numerical reconstruction, not a second historical sample or independent market confirmation.

The negative result concerns one fixed representation, penalty and exposed development design. It does not prove that all CLV methods fail or explain the original neural errors. Adding columns changes regularization geometry. C is itself the Ridge predictor with the enlarged information set, so it cannot establish an architecture/fusion advantage over that identical same-information control. We admit no formal market candidate, leave 2026Q3 sealed and retain the unfinished stable-improvement and independent-confirmation requirements. All original five scales and twenty tasks remain.

## S32. An additional reference using the announced 2020 split

### S32.1. Information and scope

The original references set the next nominal price to the origin close and cumulative log return to zero. They do not account for future splits that had already been announced. After earlier examination of the historical split event, we fixed a single split-aware reference before this calculation. Apple announced the four-for-one split on July 30, 2020, and stated that split-adjusted trading would begin August 31 [19]. Its contemporaneous as-filed Form 10-Q and SEC submission metadata support a conservative information boundary of August 1, 2020, 00:00 America/New_York. The record date of August 24 is not the trading effective date. These sources support a historical public-availability assumption; they do not establish the actual receipt time on an individual machine or the point-in-time status of vendor bars.

For each forecast origin and each future path step, the adjustment applies if the recorded decision time is at or after that boundary and origin session < August 31 ≤ target session. The price reference is origin close / 4 and the return reference is −log(4) at those steps. Other path entries retain the original reference, including origins for which the split was already effective. We keep the original nominal targets, historical predictions, time partitions and origins. The reference contains additional public announcement information unavailable to the original close-only models, and does not cover dividends or all possible corporate actions.

### S32.2. Results on the original long-scale origins

We reconstruct all five folds of the six annual, quarterly and monthly price/return tasks and score the eight original methods and the separately frozen tiny long/aligned model. The latter has three original seeds; we average losses at each origin before pooling. The fourteen minute/hourly task rows are inherited from saved summaries and date qualifications. Those tasks begin in 2021 and have no future 2020 split, and their arrays were not rescored here. A further post-observation scalar bridge divides the existing pooled mean-seed MSE for the main-paper initialized-and-shrunk xPatch candidate by the new reference MSE. The bridge first verifies the old reference MSE and origin count for all six tasks. It does not perform new inference.

**Table S32.1.** Pooled endpoint MSE improvement is 100 × (1 − predictor MSE / reference MSE). “Reference gain” compares the announced-split reference with the original persistence/zero reference. “xPatch old” and “xPatch new” compare the main-paper Table 1 candidate with the old and new references. “Tiny new” refers to a different earlier long/aligned model. Negative gains indicate higher MSE than the named reference. The xPatch-to-new comparison is a post-observation calculation from saved scalar results.

| Task | Origins | Eligible | Reference gain | xPatch old | xPatch new | Tiny new |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Annual H12 price | 47 | 0 | +0.000% | +33.731% | +33.731% | +6.928% |
| Annual H12 return | 47 | 0 | +0.000% | +38.105% | +38.105% | +6.823% |
| Quarterly H3 price | 52 | 0 | +0.000% | +1.226% | +1.226% | +0.944% |
| Quarterly H3 return | 52 | 0 | +0.000% | +0.698% | +0.698% | +0.114% |
| Monthly H21 price | 1970 | 21 | +76.809% | +0.063% | -330.938% | -331.172% |
| Monthly H21 return | 1970 | 21 | +73.597% | +0.741% | -275.931% | -282.327% |

All 21 monthly crossing origins meet the fixed information boundary, with 231 affected forecast-path coordinates per target. They are overlapping outcomes from one event, not independent split replications. The stronger reference reduces monthly price and return endpoint MSE by 76.809319% and 73.596588%, respectively. Neither the original comparison nor the new reference tests what would happen if the learned predictors also used the announcement. No assessed annual or quarterly path crosses the event; their prior point comparisons remain unchanged, including the small, overlapping annual sample. Earlier intervals against the original reference are not intervals against the new reference.

### S32.3. Numerical reconstruction and limits

The calculation used the existing pre-Q3 bars, labels and saved forecasts, with no training or new model inference. A separate implementation reconstructed 30 folds and 330 prediction files for the eight original methods and tiny, checking 37,544 scalar values, including their original pooled losses. The separate xPatch bridge checked saved pooled scalars; that replay did not open xPatch candidate prediction arrays. The first verifier completed the numerical fold reconstruction but failed on an inherited-task metadata wording field. A separate schema adapter corrected that field while preserving the original failed receipt, numerical code, tolerances and absolute deadline; its replay passed. The main process also saved a checkpoint after a deliberate one-fold pause and resumed it in a fresh process under the same deadline. This was a cooperative pause, not a hard-kill or external-signal recovery test. These checks establish numerical reproducibility on already exposed data, not independent market confirmation.

The fixed original minute-H60 and hourly-H8 predictive criteria, shared-versus-independent comparison and 2026Q3 holdout remain separate. The split reference is a descriptive public-information control, not a new learned method, a proof of market predictability or a trading strategy. The source qualification and all result receipts are listed in the separate edition's evidence map.


## S33. Cross-origin innovation reuse under a fixed external design

### S33.1 Question, fixed data and estimators

The preceding artificial controls cumulated innovations within each target path, independently across rows. They did not isolate the reuse of future innovations by neighboring origins. We test that row-dependence question analytically, preserving each row's complete conditional label distribution. This does not repeat the target-column transformation review: the intervention changes the joint distribution across fit rows.

For each L480/H60 and L96/H8 shape, the fixed NumPy PCG64 seed is 20261110+L. We generate 6,656 Gaussian histories, using rows 0–2,047 as fit and 2,560–6,655 as test; the intervening 512 rows preserve input-stream alignment and supply no labels or selection observations. Historical increments are reverse-cumulated and divided by sqrt(L), with the last historical coordinate zero. A single fit-only scalar mean and standard deviation normalize the history before the prescribed FP32 cast. The signal z is sqrt(L) times the fitted historical scale times the difference between the last two visible FP32 coordinates, evaluated in FP64. Fit-only z moments define the two-column designs \(X=[\mathbf{1},u_{\mathrm{fit}}]\) and \(Z=[\mathbf{1},u_{\mathrm{test}}]\). The known target mean is \(\mu_{i,h}=0.25\,h\,z_i/\sqrt{H}\).

The entire external design is fixed and independent of label innovations. In arm I, each row has independent Gaussian future increments. In arm O, origin i reuses increments i+1 through i+h of a common timeline. Both arms have the same row mean and same within-row covariance \(\min(h,k)\). Their fixed-horizon row covariances are \(\Sigma_{I,h}=hI\) and \(\Sigma_{O,h}[i,j]=\max(0,h-|i-j|)\). The difference need not be positive semidefinite; overlapping labels do not mechanically inflate every estimator's variance.

No noise realization or label sample is drawn, and no trainer, model inference or market loader is called. Six linear rules are frozen: zero, fit constant, half constant, OLS, Ridge and half Ridge. For arbitrary fit label y, Ridge's coefficient operator is \(A=(X^\top X/n+0.1\,\operatorname{diag}(0,1))^{-1}X^\top/n\). The intercept is unpenalized. OLS uses penalty zero; the half rules multiply the entire prediction, including intercept, by 0.5. The sole primary rule is Ridge, and controls cannot rescue its decision.

### S33.2 Analytic risk, decomposition and fixed rules

For a prediction operator \(F=ZA\), conditional on the fixed designs, known-mean risk is \(R=\|F\mu_{\mathrm{fit}}-\mu_{\mathrm{test}}\|_2^2/m+\operatorname{tr}(F\Sigma_hF^\top)/m\). The first term is the explicit noise-free-label response error, reported as squared bias; the second is estimation variance. We split the test prediction error into its constant level and centered dynamic component. With \(t=\operatorname{mean}(u_{\mathrm{test}})\), \(v=\operatorname{mean}((u_{\mathrm{test}}-t)^2)\) and coefficient covariance \(C=A\Sigma_hA^\top\), level variance is \([1,t]C[1,t]^\top\) and dynamic variance is \(vC_{\mathrm{slope},\mathrm{slope}}\). Total risk equals the sum of these projected risks. No parameter-variance diagonal alone is interpreted as the level risk.

All 68 horizons, six rules and two arms yield 816 records; 24 full-path summaries and all 408 O−I contrasts are retained. Path risk averages raw coordinate risks, without reselecting a horizon. The primary endpoint rules require all four I-arm OLS/Ridge risks to be no greater than half the matching zero risk. Each of the two primary Ridge endpoints must then satisfy \(\Delta_{\mathrm{total}}>\eta\), \(\Delta_{\mathrm{dynamic}}>\eta\), and \(\Delta_{\mathrm{dynamic}}-0.01R_{0,\mathrm{dynamic}}>\eta\). Here \(\eta=10^{-10}(R_O+R_I+R_0+1)\) is a predeclared numerical decision band, not a confidence interval or rigorous error bound. The one-percent condition is an artificial decision-value criterion; it is not the AAPL predictive-improvement threshold.

Finite/scale qualifications, Gram condition at most \(10^8\), normwise linear stationarity at most \(10^{-11}\), covariance symmetry/PSD and risk identities are checked under fixed rtol1e−9/atol1e−11 where applicable. Within-tolerance negative values would be retained and flagged. None occurs. No denominator floor, clipping, alternate solution, noise seed or penalty search is used.

### S33.3 Results and the failed scientific conjunction

**Table S33.1.** Primary Ridge known-mean risks and O−I differences, in artificial raw units. The last column divides the dynamic difference by zero's dynamic signal risk; it is not a market MSE improvement.

| Shape | I risk | O risk | Level delta | Dynamic delta | Dynamic/signal |
| --- | ---: | ---: | ---: | ---: | ---: |
| L480/H60 | 0.08410403963 | 1.808526144 | 1.711346312 | 0.01307579192 | +0.355093% |
| L96/H8 | 0.01115700401 | 0.0382365613 | 0.02730334426 | -0.0002237869696 | -0.045232% |

All four control qualifications pass. Three of six scientific inequalities pass, but neither complete endpoint conjunction passes (0/2). H60 has a positive dynamic cost below the one-percent criterion; H8 has a negative dynamic difference. Both total risks increase, predominantly through the constant-level component. These observations do not support the specified material dynamic-estimation-cost continuation across both shapes.

**Table S33.2.** All fixed rules at both endpoints. Differences are O−I; the half rules remain controls rather than alternative selected candidates.

| H | Rule | I risk | O risk | Dynamic delta |
| --- | --- | ---: | ---: | ---: |
| 60 | zero | 3.682358528 | 3.682358528 | +0 |
| 60 | constant | 3.71313771 | 5.424491966 | +0 |
| 60 | half_constant | 3.690095232 | 4.117933796 | +0 |
| 60 | OLS | 0.05877513585 | 1.785942943 | +0.01582170822 |
| 60 | Ridge | 0.08410403963 | 1.808526144 | +0.01307579192 |
| 60 | half_Ridge | 1.108999385 | 1.540104911 | +0.003268947979 |
| 8 | zero | 0.4950078075 | 0.4950078075 | +0 |
| 8 | constant | 0.4988205546 | 0.5261242502 | +0 |
| 8 | half_constant | 0.4959349291 | 0.502760853 | +0 |
| 8 | OLS | 0.007730519307 | 0.03476303828 | -0.0002707822332 |
| 8 | Ridge | 0.01115700401 | 0.0382365613 | -0.0002237869696 |
| 8 | half_Ridge | 0.1490391882 | 0.1558090775 | -5.59467424e-05 |

### S33.4 Recovery, reconstruction and interpretation

The source-reviewed 180-second shared window, including structural checks, real pause/resume, main arithmetic and independent reconstruction, completes in 4.239846 seconds. A separate controller sends SIGUSR1 to a live worker after the first complete toy unit. The worker saves and exits with the prescribed pause code; checkpoint bytes remain unchanged while paused. A new PID resumes under the same protocol hash and original absolute deadline. All four toy units, 48 records and saved scientific fields equal uninterrupted execution exactly. This demonstrates this artificial workflow's recovery, not the recovery of a market trainer. The resumed terminal checkpoint replaces the full paused checkpoint; its earlier hash and unchanged-byte check remain in the controller receipt, so a later audit cannot rehash that historical file. The final records and signal/pause receipts remain available.

The main calculation forms 2-by-2 coefficient covariances using lag products. A separate implementation regenerates the prescribed inputs, uses math.fsum moments and an explicit 2-by-2 inverse, then constructs each innovation's coefficient response by interval convolution. It independently computes every risk, full-path summary, contrast and endpoint decision. All 14,444 numerical comparisons pass; maximum absolute difference across compared quantities is 1.554312e-15. Its peak RSS is 136.660 MiB, below the 512 MiB hard limit. Main and independent programs use single numerical threads on CPUs24 and26 respectively. The independent program shares NumPy's fixed RNG and protocol/resource/JSON helpers, while its numerical functions do not import the main arithmetic. Independent FP64 reductions are compared by the frozen tolerances, not claimed to be byte-identical. The structural toy verifies cross-origin/cross-horizon covariance, h=1 arm equality and a negative quadratic form for \(\Sigma_O-\Sigma_I\).

We close this exact artificial explanatory continuation without selecting another shape, rule, sample size, signal strength or penalty. A large total variance increase does not establish a material dynamic effect. This external-design result neither identifies the original LSTM's full training mechanism nor establishes market unpredictability. In market data, a later origin's inputs may contain shocks from an earlier origin's label, so the assumed conditional error/design independence requires additional identification. No GLS/WLS, sparse-origin sampling, new market candidate or independent market confirmation follows automatically. No market arrays, original weights or protected-quarter values are read; no training or inference is performed. The original title, five scales, twenty tasks and unfinished stable-improvement and confirmation requirements remain.


# S34. Fixed-slope Ridge with the prefix return level omitted

This additional development screen evaluates one fixed prediction intervention. Let p be the previously accepted close-history-plus-anchor Ridge prediction (normalized penalty 0.1, unpenalized intercept), and let b be its purged-prefix physical cumulative-log-return mean. The intervention uses g=p-b, retaining the same slopes and fit-centered features. No predictor is refitted. The saved prefix mean and Ridge intercept have a source-qualified common reduction chain, further checked in an artificial same-source fixture; market model checkpoints were not decoded in this screen. Omitting this level is not equivalent to setting the intercept in raw-input coordinates to zero, or to zeroing a standardized-output head bias.

The four previously exposed states are minute_f1, minute_f2, hourly_f1, and hourly_f2. All 136 state-by-horizon positions and nine predictors are disclosed. The fixed continuation rule requires g to improve endpoint and equally weighted path MSE by at least 0.1% against each of p, zero, the prefix mean, 0.5p, and 0.5g in all four states (40 conditions), plus at least 1% equal-fold endpoint improvement over zero for both minute H60 and hourly H8 (two conditions). D/Z/C are disclosed but are not additional gate conditions. Passing this screen would permit only a full-development design review, not candidate qualification.

Positive percentages in the table indicate lower MSE for g than the named reference; negative percentages indicate higher MSE. These quantities are loss comparisons, not investment returns.

| Exposed state | Endpoint vs zero | Full path vs zero | Endpoint vs original Ridge | Endpoint vs 0.5g |
| --- | ---: | ---: | ---: | ---: |
| minute_f1 | -2.015642% | -2.070179% | +0.218185% | -1.354812% |
| minute_f2 | -0.144026% | -0.384793% | +0.122486% | -0.151345% |
| hourly_f1 | +0.921231% | +0.729112% | -1.489677% | -2.303685% |
| hourly_f2 | -1.431248% | -1.409458% | +0.079220% | -1.046274% |

Only 7 of 42 conditions passed. The equal-fold endpoint improvements over zero were -1.079834149% for minute H60 and -0.255008489% for hourly H8. Removing b reduced endpoint loss relative to p in three states but increased it in hourly_f1. The endpoint loss of 0.5g was lower than that of g in every state. The exact recipe was therefore closed without selecting folds or horizons, changing the penalty or level window, tuning a coefficient, or expanding training.

For every horizon the fixed paired identity is $L(p)-L(g)=b^2-2b\,\mathrm{mean}(y-g)$. Positive values favor omitting b. Its terms and unit-aware tolerances are retained in the evidence. The intervention does not change the origin-dependent prediction variation, establish that the true drift is zero, distinguish estimation noise from changing drift, or identify an original neural optimization mechanism. The previously failed accounting protocol and its original failure remain closed; this is a different raw-loss comparison, not its requalification.

A single 180-second shared execution window, including an actual external SIGUSR1 pause, a new-process resume with the same protocol/deadline, and independent replay, used 3.801525 seconds. The independent math.fsum implementation passed 4246 risk, identity, and scope implementation checks; the maximum MSE absolute difference was 1.30104261e-18. The implementations share the Python/NumPy decoding backend and standard runtime helpers, but not their risk/decision functions. There were zero market training updates and zero model-inference calls, with one analysis CPU per implementation, GPU disabled, and no paid resources.

The prefix/tail partitions and exposure identities are unchanged. The second-fold prefix contains the full first-fold tail (12,853 minute origins and 228 hourly origins), and all tails have already informed development. Numeric agreement is not independent market confirmation. Original stop/outer arrays, raw bars, split arrays, and model checkpoints were not read; 2026Q3 remained sealed. The original method-success requirements remain unmet. This addendum is integrated as Supplement S34 in this edition; the manuscript title and its five-scale, twenty-task scope are unchanged.


# S35. A fixed-support opportunity bound for a closing-window hypothesis

This zero-fit diagnostic retained the original AAPL title, complete minute H60 and hourly H8 targets, original origins and four exposed purged-fit development tails. It was motivated by the public abstract of Gao, Han, Li and Zhou (2018), *Market intraday momentum*, DOI:10.1016/j.jfineco.2018.05.009. We did not obtain the full paper and do not claim replication or an AAPL effect from that source.

The proposed transfer class fixes the original prefix mean b outside targets that fully cover the current session's final 30 minutes after an early-session signed-return feature could be complete. We constructed a superset S of all potentially modified cells without reading that feature or any price. For each horizon, any predictor C equal to b outside S has empirical MSE bounded below by \(F=\mathrm{mean}[\mathbf{1}_{\mathrm{outside}\,S}(Y-b)^2]\). Allowing zero error inside S gives an optimistic opportunity bound, not a forecast or population risk estimate.

All 136 state-horizon positions and three fixed controls (zero, b and saved same-information Ridge) were retained. Six of 26 necessary conditions were ruled out with conservative floating-point envelopes; the other twenty were merely not ruled out. The necessary subset cannot establish passage of the full 58-condition recipe.

For minute H60, original complete same-session labels imply zero support at h1--59. At h60 only 39/12,853 and 79/25,753 original origins can change. The equal-fold endpoint improvement over zero cannot exceed 0.106405182070% (conservative numerical upper bound 0.106405293968%), below the prespecified 1% threshold. Even perfect supported-cell predictions can reduce the full-path b risk by only 0.009908065364% and 0.012182788790%, below the 0.1% state requirement. The exact fixed-support class was therefore closed without fitting.

All hourly H8 endpoint cells belong to the relaxed support set, giving a zero floor and a vacuous 100% improvement bound. This does not demonstrate an hourly signal, attribute overnight versus intraday components, or reject the original ETF finding. No partial-overlap, selected-origin, changed-intercept or threshold rescue was performed.

Independent calendar/support reconstruction and columnwise scalar fsum replay passed 82,018 implementation checks, including origin identity, structural, numerical and decision assertions. The maximum loss difference was 1.30104260698e-18; the shared 180-second window used 3.322170496 seconds. A real SIGUSR1 pause, exit75, new-PID recovery and the same absolute deadline were verified. Only saved predictions and four non-price bar fields were decoded; container hashing reads compressed bytes, and Y was not recomputed from prices. Historical final-calendar/final-bar availability limitations remain.

This is deterministic evidence about a restricted class on already exposed development labels. It is not a fitted candidate, independent market confirmation or completion of the research objective. There were no model fits, weight updates, model inference, GPU use or Q3 access. This accepted diagnostic is integrated as Supplement S35 in this edition; it does not change the original method-success criteria.


# S36. Conditional seasonal-EPS direction and independent-event support

We prespecified one proxy: the log ratio of current-quarter diluted EPS to the previous fiscal year's same-quarter EPS, after applying separately retained share-unit factors. This was a self-defined conditional development question, not a verified short-horizon PEAD result or an analyst-consensus earnings-surprise measure. We retained all 22 development events; 18 pairs were usable (15 positive, two negative, one exactly zero), with four missing pairs propagated from the accepted source-fact view. The missing event dates are 2022-10-27, 2023-11-02, 2024-10-31 and 2025-10-30; the 2023 pair is missing because its prior-year source is unqualified, and the 2025 pair has both sources unqualified. These are conservative source/parser-policy states, not proof that quarterly EPS was absent from the actual reports.

The fixed nested question was whether signed change adds information beyond announcement-session membership, source missingness and absolute change. All four original purged fit prefixes contained only positive changes: three events in each first-fold prefix and five in each second-fold prefix. Thus abs(z)=z exactly and the event-level matrix [1,abs(z),z] had structural and numerical rank two rather than three. Its minimum leave-one-event-out rank was also two in every prefix. These are exact sign-support restrictions, not estimates of model forecasting error.


Table S36.1. Each announcement is counted once within a phase. Repeated minute/hour appearances do not supply independent events.

| State | Prefix events | Prefix + / - / 0 | Prefix rank | Min. deletion rank | Tail events |
| --- | ---: | --- | ---: | ---: | ---: |
| minute_f1 | 3 | 3 / 0 / 0 | 2 | 2 | 0 |
| minute_f2 | 5 | 5 / 0 / 0 | 2 | 2 | 1 |
| hourly_f1 | 3 | 3 / 0 / 0 | 2 | 2 | 0 |
| hourly_f2 | 5 | 5 / 0 / 0 | 2 | 2 | 2 |

Neither first-fold tail contained a usable active event. The second-fold minute tail contained one positive event; the hourly tail contained one positive and one negative event. No within-state prefix/tail event overlap was found. We retained cross-state event identities and did not count repeated origins or minute/hour appearances as additional independent announcements. Six of sixteen prespecified necessary conditions passed; ten failed. Consequently, this exact proxy/window/control route did not proceed to fitting. This does not establish that all earnings information is useless or that a market hypothesis has failed a predictive test.

The official historical HTML copies and extracted EPS remain conditional on the selected EPS not having been subsequently revised; unqualified primary EPS is treated as GAAP and dollar amounts as USD. Original publication bytes and actual historical receipt were not proved. The later availability boundary remained New York midnight after the original financial filingDate ended. Each event was active only in the first XNYS session not yet open after current, prior and unit-map boundaries, using the original bar-end decision clock and open<decision<=close. The reported 13/14-week difference was retained without normalization. No adjusted EPS, comparison-column replacement or annual-minus-nine-month EPS was substituted.

The main implementation and independent reconstruction completed within one fixed 180-second window in approximately 1.048256 seconds. The independent reconstruction used separate rational arithmetic, higher-precision logarithms, scalar partition reconstruction and bisect-based event counting, with shared NumPy/calendar dependencies and execution helpers; 14,858 comparisons passed. Only fit.npy and four non-price bar members were decoded. Whole-container hashing included compressed price-member bytes; no price, label or prediction member was decoded. No per-origin forecast matrix, model fit, inference, GPU, paid resource or 2026Q3 market access was used. Formal pause/resume was not exercised on these market inputs and is not claimed.

The original title, all five scales/twenty tasks, full origins/targets and method-success criteria remain unchanged. This accepted support diagnostic is integrated as Supplement S36 in this edition. The full research goal remains active and unachieved.


```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# S37. A fixed delayed market-context screen

We tested whether a single delayed SPY history improved the original AAPL minute H60/L480 and hourly H8/L96 tasks in folds 1 and 2. This was a development information screen with fixed point-error continuation criteria, not a formal method comparison or independent market confirmation. The title and five-scale, twenty-task scope are unchanged.

The original inner 80/20 boundary and clock-based label purge were retained. A_full used the original AAPL history, nominal price anchor and 17 context fields on all purged prefix origins P. B_valid and Plus_valid were fitted on the same volume-valid subset P_V. The latter added only L−1 adjacent log differences within the L delayed SPY closes at original indices i−L through i−1. Missing SPY features were mean-encoded from complete P_V observations without a preliminary standard-deviation division; each expert then used its single final training-domain scaler. Raw future log-return paths, FP64 normalized Ridge penalty λ=0.1 and an unpenalized target mean were fixed. The old shrinkage coefficient selected on a later stop partition was not imported into this inner screen.

All four prefix input domains qualified before targets were enabled; all twelve prefix models were frozen before any tail prediction or label construction. The base route used B where volume was valid and otherwise copied A. Plus copied base bitwise unless both volume and SPY history were valid. All original tail origins were scored. Both inputs were available for every actual tail origin, so missing-input fallback had synthetic acceptance coverage but was not exercised on the observed tails.

| Task | Fold | Tail origins | Zero endpoint MSE | Base endpoint MSE | Plus endpoint MSE | Gain vs zero | State gates |
|---|---:|---:|---:|---:|---:|---:|---:|
| Minute H60 | 1 | 12,853 | 1.661634989e-05 | 1.735516618e-05 | 1.742883669e-05 | -4.889683% | 0/6 |
| Minute H60 | 2 | 25,753 | 4.313385777e-05 | 4.363558824e-05 | 4.384542289e-05 | -1.649667% | 0/6 |
| Hourly H8 | 1 | 228 | 0.0002286969452 | 0.000231428175 | 0.0002194251379 | +4.054189% | 5/6 |
| Hourly H8 | 2 | 459 | 0.0006812538607 | 0.0006916065816 | 0.0007357028166 | -7.992462% | 0/6 |

Gain is 100(1−MSE_plus/MSE_zero). For each primary, the prespecified fold aggregation was 1−sum(MSE_plus)/sum(MSE_zero), giving −2.550705644710% for minute H60 and −4.964790220010% for hourly H8. Each state required strictly lower endpoint MSE than A_full, base, zero and prefix mean, and path MSE no higher than A_full and base. No state passed all six conditions; neither primary passed its minimum 1% gain. Hourly fold 1 improved endpoint error against all four controls but increased path error relative to base. Its isolated endpoint improvement was not used to change the continuation rule. The fixed representation and learner recipe was closed.

An independent implementation reconstructed source alignment, features, all twelve centered Ridge solutions, all tail target/prediction matrices, diagnostic groups and decisions. It passed 14,186,998 scalar comparisons at fixed rtol1e−9/atol1e−11; the maximum absolute difference was 2.69651e−12. The implementations shared the NumPy solve backend. A real external SIGUSR1 pause at the first 2,048-origin batch, exit code 75, unchanged checkpoint bytes and continuation under a different PID were accepted within one unchanged 900-second deadline. Peak main-process RSS was 341,745,664 bytes. Full pre-Q3 bar-cache columns were physically loaded, including closes at times belonging to the original stop/outer periods. No original stop/outer partition arrays, saved predictions or scores were accessed; targets were constructed only for the admitted P/T origins. The 2026Q3 market holdout remained unread, and no GPU or paid resource was used.

This result limits the utility of the specified dictionary and learner on exposed development histories. Correlated added columns can change Ridge penalty geometry, so the comparison does not establish a general absence of conditional information in SPY, a causal effect, or superiority of an architecture. Historical finalized-bar and receipt-time assumptions remain. Overlapping origins and folds are not independent market replications. The screen supplies no confidence-interval or unseen-confirmation claim and does not replace the full five-fold, strong-control, original block-interval and eligible-confirmation requirements. The predictive-method goal remains unachieved.

Evidence: SCIENCE_RULES.json; DATA_REFERENCES.json; run/QUALIFICATION.json; run/FROZEN_MODELS.json; run/RESULTS.json; run/INDEPENDENT_RESULT_REVIEW.json; ROOT_RESULT_INTEGRITY.json; METRICS.csv. This section integrates the accepted fixed-screen result; the reviewed standalone addendum is retained as an evidence snapshot.


# S38. Matched recurrent state routing at trading-session boundaries

## S38.1 Question and fixed comparison

This final development screen tests whether resetting part of a recurrent state at historical session boundaries improves minute-H60 and hourly-H8 cumulative-return prediction. The respective lookbacks remain 480 and 96 bars. We use the first two original folds and three fixed seeds, 2026, 2027 and 2028. Arm C carries both states across the complete input window. Arm R carries its first state and resets its second state immediately before processing each indicated session boundary. The boundary flag at the first window position is false. R after the prespecified output calibration, denoted R_shrink, is the sole candidate; R_raw remains diagnostic.

Both arms contain two independently initialized GRU cells with 32 hidden units. Each receives the same standardized origin-relative close history and boundary flag. An 82-coordinate output-head input concatenates the two final hidden states, the standardized nominal origin close and the original 17 return, volatility and calendar features. The model has 6,912 recurrent parameters and 83H output-head parameters: 11,892 for H60 and 7,576 for H8. There is no dropout. Within each task, fold and seed, the arms copy the same initial parameters and share each actual FP32 training batch. Zero head weights and biases plus a nontrainable scaled-zero buffer give identical initial zero-return functions, subject to the recorded inverse-scaling rounding bound. The reset flag is the sole architectural intervention.

The original fit F, prefix P and assessment tail T remain unchanged. Within P, the original session-tail fraction 0.15 first separates B from C_cal, then separates Q from S within B. Each adjacent split purges any earlier label whose completion time fails to precede the first later decision strictly. Every original F origin remains assigned to a role or recorded purge gap. No origin is selected by its within-session suffix length.

**Table S38.1.** Fixed nested-role counts. These are origin occurrences within each task-fold state, with overlapping labels.

| Task | Fold | Q | S | C_cal | T |
| --- | ---: | ---: | ---: | ---: | ---: |
| Minute H60 | 1 | 36,871 | 6,600 | 7,920 | 12,853 |
| Minute H60 | 2 | 74,491 | 13,020 | 15,437 | 25,753 |
| Hourly H8 | 1 | 635 | 118 | 134 | 228 |
| Hourly H8 | 2 | 1,304 | 232 | 275 | 459 |

## S38.2 Fitting, controls and frozen decision rule

Q alone fits every scaler and weight. The two arms within a task-fold state use the same immutable scaler. PCG64 supplies a common Q permutation each epoch. Adam uses learning rate 0.001 and the frozen optimizer settings; the training loss averages squared errors over standardized path coordinates. Each arm selects its checkpoint by raw FP64 endpoint MSE on S, with strict improvement, patience three and a 20-epoch cap. Ties retain the earlier checkpoint. Epoch zero is ineligible. The selected checkpoint is retained without refitting on the complete prefix.

The fixed controls are zero return, the Q mean path and enhanced Ridge. Ridge receives the neural history, boundary flags, within-session suffix mask, the FP32 product of history and that mask, the origin anchor and the 17 context features. Its width is 4L+18. Lambda=0.1 is added to the Q-normalized standardized-feature Gram matrix, with an unpenalized intercept. Q-only raw moments are accumulated in 128-origin chunks. A separate directly centered implementation reconstructs all four Ridge fits under the original numerical tolerances.

All 24 neural models, four Ridge fits and four Q mean paths are frozen before calibration. For each of these 32 fit records, C_cal determines one scalar: the endpoint inner product between predictions and labels divided by prediction squared norm, clipped to [0,1]. An exactly zero denominator gives zero. This scalar multiplies the complete raw path. All 32 scalar records are frozen before any state scores T.

The primary gain is 100 times one minus the ratio of pooled candidate and zero endpoint-loss sums across both folds. It is not an unweighted average of fold percentage gains. Seed means average per-origin losses, not predictions. Each task must gain at least 1%; its seed-mean loss must improve in each fold, and each R seed must improve after pooling folds. Every candidate seed and its seed-mean loss must have strictly lower pooled endpoint error and no higher path error than each of 13 references: zero; raw and calibrated Q mean; raw and calibrated Ridge; raw and calibrated outputs of each C seed; and raw and calibrated C seed-mean losses.

The seed-mean candidate additionally requires a positive lower bound for the paired endpoint-error difference against every reference at both registered block lengths. The bootstrap uses 5,000 replicates, PCG64 seed 2026 and the 0.025/0.975 quantiles. Folds form separate strata. Each fold retains the full trading-session grid from its first to last T session, including sessions without T origins. Minute blocks have one and five sessions; hourly blocks have two and five. Degenerate or undefined intervals fail. These requirements form 136 Boolean conditions per task. Both tasks must pass. The intervals are nominal descriptions of exposed development data, without adjustment for the wider research sequence.

## S38.3 Complete result and stopping decision

All 12 paired packages finish normally, completing 24 neural fits. The complete 32-model calibration and T evaluation then finish, followed by independent numerical reconstruction. The complete pipeline takes 1,607.498 seconds within the original four-hour window, from November 10, 2026, 21:06:06.986 to 21:32:54.483 Beijing time. No deadline is reset and no GPU is used. Checkpoints and the originally exercised pause/resume records are retained.

**Table S38.2.** R_shrink endpoint gain over zero, pooled across folds using mean seed losses. Intervals are relative gains in percent. Each interval uses the indicated session block length.

| Task | Gain (%) | Short-block 95% interval | Five-session 95% interval | Passed |
| --- | ---: | --- | --- | --- |
| Minute H60 | -1.354923 | [-2.756783, -0.008008] (1) | [-2.513457, -0.153707] | No |
| Hourly H8 | +0.013581 | [-0.028881, +0.053728] (2) | [-0.024825, +0.052641] | No |

The minute candidate's pooled endpoint MSE is 3.4770262058e-5, compared with 3.4305449524e-5 for zero. All three R seeds have negative pooled gains. The hourly candidate's MSE is 5.3098817681e-4, compared with 5.3106029919e-4 for zero. Only seed 2027 has a positive hourly pooled gain; the other two equal zero. Neither task reaches 1%, and neither has strictly positive seed-mean gain in both folds. The full conjunction fails for both tasks, with 70/136 and 88/136 individual conditions satisfied. Counts describe the decision record and are not graded evidence of partial qualification.

**Table S38.3.** Endpoint gains over zero by fold, in percent. C and R use their separately frozen calibration scalars. Both columns average losses over the three seeds.

| Task | Fold | C_shrink gain | R_shrink gain |
| --- | ---: | ---: | ---: |
| Minute H60 | 1 | +0.241596 | +0.170215 |
| Minute H60 | 2 | -1.653238 | -1.648150 |
| Hourly H8 | 1 | +0.089370 | +0.095024 |
| Hourly H8 | 2 | 0.000000 | 0.000000 |

The pooled R_shrink improvement relative to matched C_shrink is -0.007147% for minute H60 and +0.000808% for hourly H8. Both block-length intervals include zero for each contrast. At hourly fold 2, all six neural calibration scalars are zero, making the calibrated arms exactly the zero-return predictor. This masks their raw differences. At hourly fold 1, only seed 2027 retains a nonzero scalar in either arm. The results therefore supply no repeatable predictive benefit from the tested state-routing intervention. They do not establish that session resets are generally ineffective, or identify an overnight economic cause of earlier model errors.

The independent numerical review passes 11,673,968 scalar comparisons, with maximum absolute discrepancy 3.331e-15, and agrees with the failed scientific decision. It reconstructs four directly centered Ridge solutions, calibration arithmetic, losses, pooling, intervals and all gates. It reuses the qualified adapter and NumPy solver backend and does not independently replay neural forward passes. Numerical agreement establishes the scope of computational verification; it supplies no new market sample.

Under the finite closeout rule, this exact recipe is closed. No five-fold replication, replacement seed, changed window, alternative loss or raw-output promotion follows. Stable improvement over strong controls and eligible independent market confirmation remain unachieved.

## S38.4 Exposure and reproduction limits

Within both tasks, every first-fold C_cal and T origin also occurs in the second fold's Q. The implementation isolates weights, scalers and operations by task, fold, state identifier and role, but these timestamps are not globally unseen. The reader physically decodes the approved full pre-Q3 close columns. It does not open original stop/outer partition components or 2026Q3. Finalized historical bars, source revisions and assumed completion times retain their original availability limits. This controlled algorithmic contrast cannot alone explain every earlier model failure or identify market causality.

The reporting directory retains 84 task-fold predictor-loss rows, 42 pooled rows, 52 paired intervals, all 272 Boolean conditions, 32 calibration scalars and 24 training-selection records. The reporting script reads accepted JSON receipts and saved loss members only. It decodes no raw predictions or labels and performs no training, inference or new bootstrap. Exact source identities, original execution commands and the read-only reporting command are recorded in the reproduction guide. Licensed source bars, arrays and checkpoint payloads remain external local dependencies. The original title and five-scale, twenty-task study are preserved; this final two-task screen adds a bounded mechanism test and its negative result.
