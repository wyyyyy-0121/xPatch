"""Generate a result-bound paper and exhaustive supplement from frozen-run evidence."""
from common import *
from figures import LABELS

REFERENCES = r"""
## References

[1] Stitsyuk, A., and Choi, J. “xPatch: Dual-Stream Time Series Forecasting with Exponential Seasonal-Trend Decomposition.” *AAAI*, 39(19), 20601–20609, 2025. https://doi.org/10.1609/aaai.v39i19.34270

[2] Zeng, A., Chen, M., Zhang, L., and Xu, Q. “Are Transformers Effective for Time Series Forecasting?” *AAAI*, 37(9), 11121–11128, 2023. https://doi.org/10.1609/aaai.v37i9.26317

[3] Jacobs, R. A., Jordan, M. I., Nowlan, S. J., and Hinton, G. E. “Adaptive Mixtures of Local Experts.” *Neural Computation*, 3(1), 79–87, 1991. https://doi.org/10.1162/neco.1991.3.1.79

[4] Bates, J. M., and Granger, C. W. J. “The Combination of Forecasts.” *Operational Research Quarterly*, 20(4), 451–468, 1969. https://doi.org/10.1057/jors.1969.103

[5] Campbell, J. Y., and Thompson, S. B. “Predicting Excess Stock Returns Out of Sample: Can Anything Beat the Historical Average?” *Review of Financial Studies*, 21(4), 1509–1531, 2008. https://doi.org/10.1093/rfs/hhm055

[6] Bai, S., Kolter, J. Z., and Koltun, V. “An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.” arXiv:1803.01271, 2018. https://arxiv.org/abs/1803.01271

[7] Hochreiter, S., and Schmidhuber, J. “Long Short-Term Memory.” *Neural Computation*, 9(8), 1735–1780, 1997. https://doi.org/10.1162/neco.1997.9.8.1735

[8] Künsch, H. R. “The Jackknife and the Bootstrap for General Stationary Observations.” *Annals of Statistics*, 17(3), 1217–1241, 1989. https://doi.org/10.1214/aos/1176347265
"""


def main():
    r=read(BASE/'analysis/RESULTS.json'); d=read(BASE/'diagnostics/DIAGNOSTICS.json')
    choice=read(BASE/'protocol/STAGE_C_DECISION.json'); train=read(BASE/'audit/TRAINING_COMPLETE.json')
    replay=read(BASE/'audit/PREDICTOR_REPLAY.json'); recovery=read(BASE/'audit/ENGINE_RECOVERY.json')
    pool={x['kind']:x for x in r['pooled']}; primary=r['primary']; p=pool[primary]
    out=BASE/'manuscript'; out.mkdir(exist_ok=True)
    assert primary=='short_state', 'Rewrite around actual selected branch; do not silently reuse a single-stream manuscript.'
    zero=pool['zero']['h60_mse']; imp=p['h60_improvement_pct']
    seed=list(p['seed_improvements_pct'].values())
    status='met' if r['performance_goal_passed'] else 'did not meet'
    conclusion=(f'The candidate {status} the registered performance requirements. '
        f'Its pooled H60 improvement was {imp:.4f}%, with {p["positive_folds"]} of five positive fold means. ')
    if not r['performance_goal_passed']:
        conclusion += 'These results support a bounded negative result for the tested recipe; they do not establish a forecasting advantage.'
    else:
        conclusion += 'The improvement remains historical development evidence and requires independent confirmation.'
    abstract=(f'We investigate four neural recipes that failed to beat zero cumulative returns for AAPL minute forecasting. '
        f'An exact error decomposition separates prediction energy from alignment with realized returns. '
        f'Twelve retrained structural probes test price anchoring, recurrent readout and xPatch branch inputs on two chronological folds. '
        f'Mean readout does not improve either recurrent model; anchor removal is inconsistent. '
        f'Neither fold supports the registered trend–residual complementarity screen. '
        f'We therefore evaluate a predeclared local-convolution model with state-dependent output scaling, alongside single-stream and constant-scaling controls. '
        f'Five folds and three seeds yield {r["origins"]:,} matched validation origins. '
        f'The primary candidate has 1,310 trainable parameters and {imp:.4f}% pooled H60 MSE improvement over zero, with seed improvements from {min(seed):.4f}% to {max(seed):.4f}%. '
        f'It {status} the registered performance requirements. '
        f'All market periods were previously exposed in development; the protected 2026Q3 series remains unscored. '
        f'The study reports targeted failure evidence, paired uncertainty and reproducible predictors without an independent or trading-profit claim.')
    header='# Structural Diagnostics and Output Scaling for AAPL Minute Return Forecasting\n\n'
    header+='Wang Yang\n\nHigh School Affiliated to Shanghai Jiao Tong University\n\nShanghai, 200439, China\n\nEmail: [to be supplied by author]\n\n## Abstract\n\n'+abstract+'\n\n'
    intro=r"""## 1. Introduction

An unsuccessful forecasting comparison can motivate structural questions when the original predictions, partitions and training histories remain available. In a preceding AAPL comparison, the minute H60 cumulative-return task gave negative pooled improvements for MLP, GRU, LSTM and an xPatch adapter. Ranking these recipes leaves the reason unresolved: a price-level anchor may be unstable, recurrent readout may discard useful information, or a decomposition may add prediction variation without useful correspondence to returns.

This follow-up tests those explanations before choosing a compact candidate. The contribution is a documented chain from an observed error pattern to retrained probes, a conditional design decision and paired controls. It does not introduce trend decomposition, convolution, mixtures or shrinkage as new concepts. xPatch already combines exponential decomposition and two streams [1]; simple decomposition/linear models motivate strong linear controls [2]. Input-dependent expert weighting [3] and forecast combination [4] also require evidence that candidate components contribute complementary information.

We ask whether targeted structural changes improve the original recipes, whether the tested branch inputs exhibit repeatable state complementarity, and whether state-dependent scaling adds value beyond a local model and simple shrinkage. Financial prediction studies emphasize the difficulty of exceeding simple benchmarks [5]; that literature does not establish predictability for AAPL at this frequency. Every result here concerns exposed historical development data, with model-level chronology preserved.

## 2. Data and evaluation design

The admitted source is the existing Alpaca AAPL snapshot, with the analysis grid ending June 30, 2026. Minute data begin in January 2021; all 535,620 expected regular-session grid positions are observed. The XNYS calendar is a grid proxy. Raw provider versions can contain revisions, so this is not a fully point-in-time trading reconstruction. The researcher confirmed permission for Alpaca and Kibot data; only Alpaca contributes to this study, and supplier contracts were not independently inspected.

At origin \(t\), the input contains 480 completed historical closes, expressed on the split basis known at that origin. Targets are cumulative log returns

\[
y_{t,h}=\log(P_{t+h}/P_t),\qquad h=1,\ldots,60.
\]

Every target path remains within one regular session. Historical windows may cross sessions; consecutive historical log-price differences can therefore contain an overnight change. Cash dividends are excluded. H60 endpoint mean squared error is primary. H5/H15/H30 and the full-path error are auxiliary outputs from the same 60-step predictor, not separately trained models or a claimed multi-horizon innovation.

We retain the original five expanding chronological folds and eligible origins. Fitting labels finish before inner stopping decisions; all training labels finish before outer validation decisions. Transformations use the inner fitting partition only, and selected predictors are not refitted on stopping data. Matching the original hashes establishes partition identity, not independence from investigator exposure. The 2026Q3 series remains sealed, with no protected-period fitting or scoring.

Adam uses learning rate 0.001, batch size 128, at most 20 epochs and patience 3. Training minimizes MSE across the standardized return path; stopping selects raw H60 MSE. These objectives are distinct. All 20 original H60 neural fits stopped before the epoch cap, which rules out reaching that cap as the direct explanation for this task's recorded stops; it does not establish convergence.

## 3. Diagnostics and conditional model design

### 3.1 Prediction energy and structural probes

With the zero-return predictor as reference, the exact identity

\[
E[(\hat y-y)^2]=E[y^2]+E[\hat y^2]-2E[\hat y y]
\]

separates prediction energy from alignment. It is descriptive and cannot identify a causal economic mechanism. Figure 1 shows the quantities for the original five-fold H60 predictions. Equal averaging of the four predictions also remains a descriptive combination, without fitting or independent confirmation.

![Figure 1. Prediction energy and alignment, each normalized by zero-return MSE. Excess energy over alignment increases squared error. All quantities use the original exposed validation predictions.](figures/diagnostic_energy.png){width=3.30in}

Six changes are retrained on the earliest two folds at seed 2026: remove the standardized anchor input from MLP; replace GRU/LSTM final-state readout with a mean over hidden states; and, separately, remove the xPatch anchor, zero its trend input or zero its residual input. Anchor removal supplies zero in the original parameter layout. Branch probes preserve both computational networks and their biases; zeroing an input is not literal removal of all branch parameters. Effective capacity and optimization can change.

A probe is a supported screening direction only if it lowers H60 error in both folds. Complementarity additionally requires the trend-retained and residual-retained probes to reverse their error ranking across low/high past-60-bar volatility, with the same reversal direction in both folds. Each threshold is the corresponding inner-fit median. This deliberately narrow screen is not a general test of every possible fusion mechanism.

### 3.2 The predeclared single-stream revision

Before new market fitting, the protocol specified a single-stream revision if the complementarity screen failed. This branch evaluates a local causal convolution with no raw dollar-price anchor. Causal exponential means of spans 8 and 32 initialize at the first close in each supplied 480-bar window. The local input is the 480-position deviation from the span-32 mean, divided by past-60-return standard deviation with floor \(10^{-6}\). Two ReLU convolution layers use eight channels, kernel width three and dilations one/two; last and mean activations feed a 60-output linear head. Causal convolutions are established sequence-modeling components [6].

Four past-state features contain short/long volatility ratios and absolute trend/deviation magnitudes. Inner-fit means and scales standardize those features. An eight-unit ReLU layer and sigmoid output produce \(\lambda_t\). The primary raw-return path is

\[
\hat y_{t,h}=\sigma_h\lambda_t S_h(x_t).
\]

The target-scale offset in the implementation makes this expression zero-centered despite training standardization. The local head is initialized to zero, and inner stopping may select epoch zero. The original LSTM [7] retains its original initialization and stopping recipe; this difference is disclosed. All local variants share the zero option, so module comparisons do not change that option.

The sigmoid is an output factor, not a calibrated confidence probability. Because the head is unconstrained, constant scaling can be absorbed into its weights; it is a parameterization control, not a hard bound on total prediction magnitude. State scaling also changes the function class and optimization. Its incremental value must be established empirically.

The formal matrix is LSTM, trend linear head, local convolution, local with learned constant scaling, and local with state scaling, each at five folds and seeds 2026/2027/2028. Stored small predictors each contain 1,612 physical parameters; trainable counts are 300, 1,252, 1,254 and 1,310 for trend/local/constant/state variants. Inactive modules are frozen and retained for common storage/initialization. The local receptive field is seven positions before temporal pooling.

Controls include zero return, Ridge on all 488 derived candidate inputs, and the LSTM scaled by a single coefficient in [0,1]. Ridge penalties 0.1/1/10 and the scalar coefficient are selected only on inner stopping data. The scalar is the clipped least-squares coefficient \(\sum py/\sum p^2\), with zero used for a zero denominator. This prevents outer outcomes from calibrating the control. Direct trend features and the full residual history are available to Ridge.

### 3.3 Paired analysis and registered requirements

Primary summaries average three seeds' squared losses at each origin, then pool the common origins across folds. Scoring the mean prediction is a separate ensemble result. Seeds measure fitting randomness and do not supply new market samples. We resample the full validation-session grid within each fold using 5,000 non-circular moving-block draws, seed 2026, lengths one and five sessions, and nominal 95% percentile intervals [8]. Every comparison uses the same block draws. These are descriptive, unadjusted intervals on previously exposed data; stationarity is not established.

The registered performance requirements are at least 1% pooled H60 improvement over zero, positive pooled improvement for all three seeds, at least four positive fold means, superiority to LSTM/Ridge/scalar-shrunk LSTM, support under both block lengths, and ablations supporting the claimed state module. The 1% requirement is a practical target, not a significance threshold. Recipes, seeds and horizons are not changed in response to the scores.
"""
    probe_lines=['| Retrained change | Fold 1 | Fold 2 |','| --- | ---: | ---: |']
    plabel={'mlp_noanchor':'MLP: anchor zero','gru_mean':'GRU: mean readout','lstm_mean':'LSTM: mean readout',
        'xpatch_noanchor':'xPatch: anchor zero','xpatch_notrend':'xPatch: trend input zero','xpatch_noresidual':'xPatch: residual input zero'}
    for e in choice['evidence']:
        if 'probe' in e:
            a,b=e['improvement_vs_original_pct']; probe_lines.append(f'| {plabel[e["probe"]]} | {a:+.2f}% | {b:+.2f}% |')
    formal_lines=['| Method | Gain vs zero | Seed range | Folds > zero |','| --- | ---: | ---: | ---: |']
    for k,x in pool.items():
        vals=list(x['seed_improvements_pct'].values()); sr=f'{min(vals):+.2f} / {max(vals):+.2f}' if vals else '—'
        formal_lines.append(f'| {LABELS[k]} | {x["h60_improvement_pct"]:+.3f}% | {sr} | {x["positive_folds"]}/5 |')
    results='\n## 4. Results\n\n'
    original=', '.join(f'{m.upper() if m!="xpatch" else "xPatch"}: {d["pooled"][m]["improvement_pct"]:.4f}%' for m in d['models'])
    results+=f'The original pooled improvements over zero were {original}. The equal prediction mean achieved {d["equal_mean_improvement_pct"]:.4f}%. Prediction energy exceeded its alignment offset for every original recipe. This motivates amplitude diagnostics, without proving that shrinking any one model will generalize.\n\n'
    results+='**Table 1.** H60 MSE improvements of retrained probes relative to their matched original recipe, not relative to zero. Positive values favor the change. These are two-fold screening results.\n\n'+'\n'.join(probe_lines)+'\n\n'
    results+='Mean readout worsened both GRU and LSTM in both screening folds. Anchor removal did not improve either applicable architecture consistently. Both xPatch input-zeroing probes improved on the full adapter, but the trend-retained probe had lower loss in both volatility states in both folds. The required state-rank reversal was absent. This is evidence against the stipulated complementarity screen, with capacity, optimization and the retained anchor still part of the comparison. No two-stream candidate entered the formal matrix.\n\n'
    results+=f'The selected branch completed 75 formal cells, including five identity-verified reused LSTM cells, plus 12 probes: {train["new_fits"]} new market fits in total. The five folds contain {r["origins"]:,} validation origins; zero H60 MSE is {zero:.8g}. Table 2 reports mean seed loss, with ensemble scores kept separate in the supplement.\n\n'
    results+='**Table 2.** Formal H60 improvement over zero. “Seed range” is the minimum/maximum pooled improvement across the three seeds, in percent; Ridge and zero are deterministic. Positive folds use mean seed loss.\n\n'+'\n'.join(formal_lines)+'\n\n'
    results+=f'The primary state-scaled model achieved {imp:.4f}% improvement, with {p["positive_folds"]}/5 positive fold means and seed values '+', '.join(f'{v:.4f}%' for v in seed)+'. '
    results+=f'Its prediction ensemble achieved {p["ensemble_h60_improvement_pct"]:.4f}%; that number does not replace the registered mean-loss outcome.\n\n'
    for ref in ('zero','short_plain','short_const','linear_same_features','lstm_constant_shrink'):
        intervals=sorted([x for x in r['paired_intervals'] if x['reference']==ref],key=lambda x:x['block_sessions'])
        vals='; '.join(f'{x["block_sessions"]}-session: [{x["relative_ci95_pct"][0]:+.3f}, {x["relative_ci95_pct"][1]:+.3f}]%' for x in intervals)
        results+=f'For primary versus {LABELS[ref]}, paired relative-improvement intervals were {vals}. '
    results+='\n\n'+conclusion+' The supplement records each requirement individually.\n\n'
    results+='![Figure 2. Fold/seed H60 improvements. Colored points are runs; black diamonds are pooled mean-loss results. The zero line is the reference and the dotted 1% line is the registered practical target. Fold colors identify time blocks, not independent repetitions.](figures/formal_stability.png){width=3.30in}\n\n'
    discussion=r"""## 5. Discussion and limitations

The structural probes narrow the original explanations. Replacing a final recurrent state with its temporal mean did not rescue the tested recipes. Anchor removal had fold-dependent effects, so the evidence does not support a universal price-anchor explanation. Lower errors after zeroing xPatch inputs suggest that the full adapter's variation is not automatically useful, but the probes do not isolate all capacity or optimization effects. Failure of the specific volatility-state screen does not rule out other mixtures, other branch definitions or other markets.

The compact experiment tests an output-parameterization hypothesis under fixed data and limited computation. Beating an unsuccessful neural comparator alone is insufficient: zero, full-information Ridge and an inner-calibrated scalar are necessary checks. A state factor cannot be interpreted as reliability from its sigmoid range or visually appealing values. If constant scaling or the zero option explains the result, a dynamic mechanism is unsupported. The registered ablations define the empirical comparison; they do not prove a unique causal mechanism inside an overparameterized optimizer.

The study has one stock, a fixed frequency, close-only inputs and a historical provider snapshot. Overlapping minute labels are dependent, and neither three seeds nor many origins create independent market episodes. One/five-session resampling provides sensitivity descriptions rather than an independence guarantee. Historical periods informed the prior comparison and follow-up design; within-fit chronology does not turn those periods into a new confirmatory test. A protected period must remain separate until a later, explicitly approved confirmation design has been fixed.

No execution model, transaction costs or portfolio test is evaluated. Forecast MSE reduction would not by itself imply profitable trading. The earlier five-scale comparison remains preliminary context in the supplement; the new candidate has not been validated across those scales. The research contribution is bounded structural evidence and a reproducible small-model experiment, with its performance outcome reported directly.
"""
    discussion+='\n## 6. Conclusion\n\n'+conclusion+' The two-fold screen did not support the planned trend–residual complementarity. Targeted probes and the complete local-model controls make the tested explanations and their limits explicit. A future forecasting claim requires a distinct frozen confirmation study.\n\n'
    discussion+=f'## Reproducibility and availability\n\nThe isolated study archive contains the frozen configurations, run identities, metrics, paired losses and supplement. All {replay["predictors"]} small predictors were replayed over their complete validation arrays and 60 outputs from historical inputs, without reading target files in the replay. A real process-exit/restart acceptance on synthetic dropout/Adam data reproduced model, optimizer, RNG, history and prediction states exactly; it is an engineering check, not a new market experiment. A standalone CPU package includes the archived fold/seed predictors and preprocessing. Raw licensed data and all full prediction arrays remain in the original workspace with hashes; the portable ZIP is not a complete raw-data training archive. The email field remains for the author to fill. Nothing has been submitted externally.\n\n'
    mainpaper=header+intro+results+discussion+REFERENCES
    (out/'MAIN_PAPER_EN.md').write_text(mainpaper)
    # Exhaustive machine-derived supplement, retaining unfavourable cells.
    supp='# Supplement: Structural Diagnostics and Compact AAPL Return Models\n\nWang Yang\n\n'
    supp+='This supplement reports exposed historical development evidence. The protected 2026Q3 series is unscored. No selection below changes the registered recipe.\n\n'
    supp+='## S1. Protocol and source identity\n\n'
    supp+=f'Primary model: `{primary}`. Data admission SHA256: `{read(BASE/"protocol/DATA.json")["admission_sha256"]}`. Frozen decision SHA256: `{r["decision_sha256"]}`. The original reviewed plan and the pre-fit implementation addendum are preserved in the protocol folder.\n\n'
    supp+='| Fold | Fit dates | Stop dates | Validation dates | Fit / stop / validation origins |\n| --- | --- | --- | --- | ---: |\n'
    for x in r['session_grid']:
        supp+=f'| {x["fold"]} | {x["fit_first"]} to {x["fit_last"]} | {x["stop_first"]} to {x["stop_last"]} | {x["first_validation_session"]} to {x["last_validation_session"]} | {x["fit_origins"]:,} / {x["stop_origins"]:,} / {x["validation_origins"]:,} |\n'
    supp+='\n## S2. Original model diagnostics\n\n'
    structures=read(BASE/'diagnostics/STRUCTURE_COUNTS.json')
    supp+='| Model | Implemented structure | Parameters |\n| --- | --- | ---: |\n'
    for m,x in structures['structures'].items():supp+=f'| {m} | {x} | {structures["parameters"][m]:,} |\n'
    supp+='\n'
    supp+='| Model | Gain over zero (%) | Prediction energy / zero | Alignment / zero | Correlation |\n| --- | ---: | ---: | ---: | ---: |\n'
    for m,x in d['pooled'].items(): supp+=f'| {m} | {x["improvement_pct"]:+.6f} | {x["energy_ratio"]:.8f} | {x["alignment_ratio"]:.8f} | {x["correlation"]:.6f} |\n'
    supp+='\n| Fold | Model | State | Origins | Gain (%) | Bias | Realized mean |\n| --- | --- | --- | ---: | ---: | ---: | ---: |\n'
    for x in d['fold_state_metrics']:
        supp+=f'| {x["fold"]} | {x["model"]} | {x["state"]} | {x["origins"]:,} | {x["improvement_pct"]:+.5f} | {x["prediction_bias"]:.6g} | {x["actual_mean"]:.6g} |\n'
    supp+='\n## S3. All retrained probes and the conditional decision\n\n'+'\n'.join(probe_lines)+'\n\n'
    supp+='Both input-zeroing probes retain the original branch networks and price-anchor head. They are retraining interventions on inputs, not pure removals of network capacity.\n\n'
    supp+='| Fold | Trend-retained low / high MSE | Residual-retained low / high MSE | Rank reversal |\n| --- | ---: | ---: | ---: |\n'
    for x in choice['evidence']:
        if 'branch_losses' in x:
            a=x['branch_losses']['xpatch_noresidual']; b=x['branch_losses']['xpatch_notrend']
            supp+=f'| {x["fold"]} | {a["low"]:.8g} / {a["high"]:.8g} | {b["low"]:.8g} / {b["high"]:.8g} | {x["reversal_orientation"]} |\n'
    supp+='\nThe reversal requirement failed in both folds. The pre-fit single-stream branch, not a score-selected new architecture, was executed. No double-stream formal fit was run.\n\n'
    supp+='## S4. Pooled errors and all fold/seed results\n\n'+'\n'.join(formal_lines)+'\n\n'
    supp+='| Method | H5 MSE | H15 MSE | H30 MSE | H60 MSE | Path MSE | H60 MAE |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n'
    for k,x in pool.items(): supp+='| '+LABELS[k]+' | '+' | '.join(f'{x[key]:.8g}' for key in ('h5_mse','h15_mse','h30_mse','h60_mse','path_mse','h60_mae'))+' |\n'
    supp+='\n| Fold | Method | Seed | H60 MSE | Gain over zero (%) | H60 MAE | Path MSE |\n| --- | --- | ---: | ---: | ---: | ---: | ---: |\n'
    for x in r['fold_seed_metrics']:
        supp+=f'| {x["fold"]} | {LABELS[x["kind"]]} | {x["seed"] or "—"} | {x["h60_mse"]:.8g} | {x["h60_improvement_pct"]:+.6f} | {x["h60_mae"]:.8g} | {x["path_mse"]:.8g} |\n'
    supp+='\n## S5. Seed-loss mean versus prediction ensemble\n\n| Method | Mean-loss gain (%) | Prediction-mean gain (%) |\n| --- | ---: | ---: |\n'
    for k,x in pool.items(): supp+=f'| {LABELS[k]} | {x["h60_improvement_pct"]:+.6f} | {x["ensemble_h60_improvement_pct"]:+.6f} |\n'
    supp+='\n![Auxiliary stepwise MSE ratios for all 60 cumulative-return outputs. The primary endpoint remains H60.](figures/auxiliary_path.png){width=6.6in}\n\n'
    supp+='## S6. Paired uncertainty and success requirements\n\nPositive reference-minus-primary differences favor the primary. Losses are averaged over seeds per origin before session-block sampling. Sessions with no eligible origins remain on the fold session grid. Intervals are nominal 95%, descriptive and not multiplicity adjusted.\n\n'
    supp+='| Reference | Block (sessions) | MSE difference | Difference interval | Relative interval (%) |\n| --- | ---: | ---: | ---: | ---: |\n'
    for x in r['paired_intervals']:
        a,b=x['difference_ci95']; c,e=x['relative_ci95_pct']
        supp+=f'| {LABELS[x["reference"]]} | {x["block_sessions"]} | {x["mse_reference_minus_primary"]:.8g} | [{a:.8g}, {b:.8g}] | [{c:+.5f}, {e:+.5f}] |\n'
    supp+='\n| Registered requirement | Met |\n| --- | --- |\n'
    for k,x in r['performance_checks'].items(): supp+=f'| {k.replace("_"," ")} | {"Yes" if x else "No"} |\n'
    supp+='\n## S7. Volatility-state descriptions\n\n| Fold | Method | State | Origins | MSE | Zero MSE |\n| --- | --- | --- | ---: | ---: | ---: |\n'
    for x in r['state_metrics']: supp+=f'| {x["fold"]} | {LABELS[x["kind"]]} | {x["state"]} | {x["origins"]:,} | {x["mse"]:.8g} | {x["zero_mse"]:.8g} |\n'
    supp+='\n## S8. Controls, parameters and training receipts\n\nRidge sees trend features (4), the full residual history (480), and state features (4). Scaling and intercept fit only inner-fit data. Its penalty is selected by inner-stop H60 MSE. LSTM shrink coefficients minimize inner-stop H60 MSE in [0,1].\n\n'
    scales=read(BASE/'analysis/SCALE_DESCRIPTIONS.json')
    supp+='Trainable counts use the requires-gradient convention. The single-stream gate retains an unused first output (nine parameters); the constant variant retains an unused first constant. No effective-capacity equivalence is claimed from the printed counts.\n\n'
    supp+='| Fold | Local variant | Seed | Best epoch | Scale mean | Scale 5 / 50 / 95 percentiles | Energy / zero | Alignment / zero |\n| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n'
    for x in scales['rows']:
        if x['kind'] not in ('short_plain','short_const','short_state'):continue
        quant=' / '.join(f'{v:.4f}' for v in x['lambda_quantiles'])
        supp+=f'| {x["fold"]} | {LABELS[x["kind"]]} | {x["seed"]} | {x["best_epoch"]} | {x["lambda_mean"]:.5f} | {quant} | {x["prediction_energy_ratio"]:.7f} | {x["alignment_ratio"]:.7f} |\n'
    supp+='\nScale distributions describe learned functions, including epoch-zero choices; they are not validated probabilities or standalone evidence of useful adaptation. Constant factors can be absorbed by an unconstrained head.\n\n'
    supp+='| Fold | Ridge alpha | LSTM scalar (seed 2026 / 2027 / 2028) |\n| --- | ---: | ---: |\n'
    for fold in range(1,6):
        lin=read(BASE/'controls'/f'linear_f{fold}'/'DONE.json')
        vals=[read(BASE/'controls'/f'shrink_lstm_f{fold}_s{s}'/'DONE.json')['alpha'] for s in (2026,2027,2028)]
        supp+=f'| {fold} | {lin["selected_alpha"]} | '+ ' / '.join(f'{v:.8f}' for v in vals)+' |\n'
    supp+='\n| Fold | Method | Seed | Best epoch | Epochs | New training wall (s) | Training CPU (s) | Reused |\n| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |\n'
    for x in r['training_records']:
        supp+=f'| {x["fold"]} | {LABELS[x["kind"]]} | {x["seed"]} | {x["best_epoch"]} | {x["epochs"]} | {x["training_wall_seconds"]:.3f} | {x["training_cpu_seconds"]:.3f} | {x["reused_original"]} |\n'
    supp+='\nTraining wall/CPU fields measure the engine training invocation before final prediction, including validation and data preparation. They exclude final prediction overhead and most analysis. Reused cells show zero new-training time, not zero original cost. The last-invocation fields and individual receipts retain additional timing. Their sum is not comprehensive total-study CPU consumption.\n\n'
    supp+='## S9. Recovery and inference acceptance\n\n'
    supp+=f'{replay["predictors"]} small predictors and {replay["total_origin_predictor_pairs"]:,} origin–predictor pairs were checked at all 60 steps. Exact equality for every saved array: {replay["all_exact"]}. The positive-close API was checked at the first and final prediction batches; the complete replay used exact split-adjusted log-relative histories. The verifier reads historical close inputs and saved predictions, not target arrays. Short, long, zero, nonfinite and negative-input checks are included in the interface acceptance.\n\n'
    supp+='The whole-engine restart check used synthetic data and three separate process invocations: an uninterrupted run, a run exiting after two batches, and its continuation. Model and Adam state, both RNGs, history, best weights and full predictions matched exactly. This validates recovery plumbing; it is not a new market result. Checkpoints use atomic replacement and preserve minibatch position, permutation and optimizer state. The separate original source/data hashes remain bound to the run receipts.\n\n'
    supp+='## S10. Claim and hypothesis ledger\n\n| Hypothesis | Evidence and permitted interpretation |\n| --- | --- |\n'
    supp+='| Price anchor is the common cause | Not supported by consistent improvements in both probe folds. |\n| Mean recurrent readout rescues the recipe | Not supported: GRU and LSTM changes worsen both folds. |\n| Original xPatch inputs are complementary across the predefined states | Required rank reversal absent in both folds; no two-stream formal claim. |\n'
    supp+=f'| Compact state scaling meets registered performance requirements | {"Met on development data only" if r["performance_goal_passed"] else "Not met"}; individual criteria and all paired intervals above. |\n'
    supp+='| Independent predictive or trading success | Not tested. Protected 2026Q3 remains unscored; no execution/cost evaluation. |\n\n'
    supp+='## S11. Relation to preceding five-scale study\n\nThe preceding 20-task, 100-fold core comparison and its 400 neural fits are preserved at their original workspace locations. They motivated this follow-up and remain development evidence. Their results do not validate the new small model on annual, quarterly, monthly, hourly or other minute tasks. Full original reports are included as background in the review package; these earlier fits are not counted as new fits here.\n\n'
    supp+=REFERENCES
    (out/'SUPPLEMENT_EN.md').write_text(supp)
    atomic(out/'SOURCE_AUDIT.json', {'main_sha256':sha(out/'MAIN_PAPER_EN.md'),
        'supplement_sha256':sha(out/'SUPPLEMENT_EN.md'), 'results_sha256':sha(BASE/'analysis/RESULTS.json'),
        'primary':primary, 'performance_goal_passed':r['performance_goal_passed'],
        'Q3_values_read':False, 'independent_confirmation':False})
    print(json.dumps({'manuscript_written':True,'performance_goal_passed':r['performance_goal_passed']}))


if __name__=='__main__': main()
