"""Build an integrated original-title manuscript from completed evidence."""
from common import *
import shutil

def pct(v):return f'{v:+.3f}%'
def mdtable(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)

def main():
    r=read(BASE/'analysis/RESULTS.json');complete=read(BASE/'audit/TRAINING_COMPLETE.json')
    assert r['completed_new_fits']==complete['jobs'] and r['new_primary_tasks']==20
    old=ROOT/'outputs/multiscale/double_column_manuscripts/20261108_parallel8_full_core'
    original=(old/'MAIN_PAPER_EN.md').read_text();refs=original.split('## References\n',1)[1]
    mainrows=[x for x in r['pooled'] if x['variant']=='long_aligned'];Nseed=mainrows[0]['seed_count']
    success=r['both_mechanism_tasks_pass']
    m=next(x for x in mainrows if x['task']=='minute_h60_future_log_return')
    h=next(x for x in mainrows if x['task']=='hourly_h8_future_log_return')
    annual_price=next(x for x in mainrows if x['task']=='annual_h12_future_close')
    annual_return=next(x for x in mainrows if x['task']=='annual_h12_future_log_return')
    tiny_jobs=[job for job in read(BASE/'protocol/SELECTED_JOBS.json')['jobs'] if job['family']=='tiny']
    epoch_zero=sum(read(BASE/'runs'/job['id']/'DONE.json')['best_epoch']==0 for job in tiny_jobs)
    verdict=('Both prespecified mechanism tasks met the registered performance requirements.' if success else
        'The candidate did not meet the registered requirement for stable improvement on both mechanism tasks.')
    core=list(csv.DictReader((CORE/'results/METRICS_POOLED.csv').open()))
    close_neural=[z for z in core if z['target']=='future_close' and z['model'] in ('mlp','gru','lstm','xpatch')]
    oldpositive=sum(float(z['relative_improvement'])>0 for z in close_neural)
    newtable=[]
    for profile,(_,_,hs) in PROFILES.items():
        for horizon in hs:
            a=next(x for x in mainrows if x['task']==key(profile,horizon,'future_close'))
            b=next(x for x in mainrows if x['task']==key(profile,horizon,'future_log_return'))
            newtable.append([f'{profile.capitalize()} H{horizon}',pct(a['improvement_pct']),pct(b['improvement_pct'])])
    ftable=[]
    for v,label in [('short_path','Short / path'),('long_path','Long / path'),('short_aligned','Short / aligned'),('long_aligned','Long / aligned')]:
        ftable.append([label]+[pct(next(x for x in r['pooled'] if x['task']==key(p,hh,t) and x['variant']==v)['improvement_pct']) for p,hh,t in MECHANISM])
    fusiontext=[]
    for result in r['fusion']:
        values={x['method']:x['improvement_pct'] for x in result['pooled']}
        gate=pct(values['gate']) if 'gate' in values else 'not estimable'
        fusiontext.append(f"For {result['task'].split('_h')[0]} returns, equal, constant and gated combinations respectively achieved {pct(values['equal'])}, {pct(values['constant'])} and {gate} improvement over zero on the common three-fold replay. "
            +( 'The gate cleared the registered pairwise replay comparisons.' if result['gate_gain_supported'] else 'The gate did not clear all required comparisons with zero, constant weights and the four single experts.'))
    interactions=r['factorial_interactions'];supported=sum(z.get('joint_gain_beyond_additivity_supported',False) for z in interactions)
    factor_sentences=[]
    for profile,hh,t in MECHANISM:
        task=key(profile,hh,t)
        descriptions=[]
        for reference,description in [('short_aligned','long dilation under aligned loss'),
                                      ('long_path','endpoint weighting under long dilation')]:
            paired=[z for z in r['intervals'] if z['task']==task and z['primary']=='long_aligned'
                    and z['reference']==reference]
            supported_pair=len(paired)==2 and all(z['supports_improvement'] for z in paired)
            descriptions.append(('support' if supported_pair else 'do not support')+' lower error from '+description)
        factor_sentences.append(f"For {profile} H{hh}, the paired intervals at both block lengths {descriptions[0]}, and {descriptions[1]}.")
    probe_both=[]
    for profile,hh,t in MECHANISM:
        for model in ('mlp','gru','lstm','xpatch'):
            rr=[z for z in r['original_loss_probes'] if z['task']==key(profile,hh,t) and z['model']==model]
            if len(rr)==2 and all(z['improvement_vs_original_pct']>0 for z in rr):probe_both.append(f'{profile} {model}')
    probe_description=(', '.join(probe_both) if probe_both else 'none of the eight task-model pairs')
    text=rf'''# Multi-Horizon Forecasting of Apple Stock Prices and Returns Across Temporal Scales

Wang Yang

High School Affiliated to Shanghai Jiao Tong University

Shanghai 200439, China

[Email to be completed by the author]

## Abstract

We compare AAPL nominal-close and cumulative-log-return forecasts across annual, quarterly, monthly, hourly and minute scales. We evaluate twenty tasks with five chronological folds and matched persistence references. An initial eight-method comparison provides 400 neural fits. We then diagnose saved predictions and test two factors in a small causal convolutional network: local receptive field and path-versus-endpoint loss weighting. The fixed main candidate uses the same recipe across twenty tasks, while minute H60 and hourly H8 returns receive complete four-variant, three-seed ablations. We calibrate fixed-expert combinations on expanding past records. The main candidate's mean-seed-loss improvements over zero are {pct(m['improvement_pct'])} for minute H60 and {pct(h['improvement_pct'])} for hourly H8. {verdict} We report all tasks, including adverse results, and distinguish limited historical gains from mechanism claims. We inspected the assessment periods during development and leave the protected 2026Q3 series unscored. We establish neither independent confirmation nor trading profitability.

**Keywords:** financial forecasting; multiple temporal scales; causal convolution; loss weighting; forecast combination.

## 1. Introduction and related work

Stock forecasts at five minutes and twelve months differ in available history, outcome overlap and exposure to market events. Price errors also require an explicit persistence reference: retaining the latest close can outperform a trained predictor even when that predictor appears close to the observed price. Return targets remove the level from the label but remain sensitive to corporate-action conventions.

We study one stock under matched origins and fixed finite-budget recipes. The original comparison covers persistence, drift, seasonal repetition, ridge regression, MLP, GRU, LSTM and an adapted xPatch model. We extend that comparison by examining four neural implementations, testing a parameter-matched two-factor model and evaluating actual expert combinations. We ask which changes improve error under the stated rules, and which corresponding ablations support an explanation. A planned shared-versus-separate-horizon market experiment remains unrun; the present path output does not supply that comparison.

xPatch combines trend/residual processing with exponential decomposition [1]; PatchTST and Temporal Fusion Transformers provide established patch and multi-horizon designs [2,3]. Financial forecasting studies use different information sets and outcomes [4–8]. Mattera et al. include Apple in multi-horizon forecasting with clustering and reconciliation [9], providing direct precedent. We use established causal dilated convolutions [13], forecast combinations [14] and input-dependent expert weights [15] to test behavior within the same AAPL tasks and matched references. Financial shrinkage results for aggregate markets [16] motivate a control, without establishing its transfer to minute AAPL returns.

## 2. Data, targets and evaluation

Experiments use Alpaca SIP/raw data through the protected boundary of July 1, 2026 in New York. The daily source has 2,637 rows from January 2016 through June 2026; conservative completion retains 2,636 dates and 125 full months. Intraday history runs from January 4, 2021 through June 30, 2026. Its regular-session minute grid contains all 535,620 expected observations and yields 8,238 complete hourly bars. We exclude incomplete hourly remnants and retain expected bar positions. XNYS supplies the session calendar; minute-derived closes need not equal official closing-auction prices. The researcher confirmed Alpaca and Kibot permission; only Alpaca contributes reported results. The annual task uses the accepted approximately ten-year history.

At a completed origin \(t\), we fit the following targets separately for \(j=1,\ldots,H\):

\[
Y^P_{{t,j}}=P_{{t+j}},\qquad Y^R_{{t,j}}=\log(P_{{t+j}}/P_t).
\]

Annual and quarterly tasks use monthly bars at H12 and H3. Monthly uses daily H21, hourly uses H1/H4/H8, and minute uses H5/H15/H30/H60. Lookbacks are 12, 36, 252, 96 and 480 bars. These ten horizons and two targets define twenty tasks. Minute labels finish within one regular session. H8 counts complete trading-hour bars and always crosses a session; it does not mean eight clock hours. Historical windows may cross sessions.

Historical closes account only for splits effective at the origin. Future splits can create nominal-target discontinuities; we exclude cash dividends. We assign daily completion to the next New York midnight [12]. This conservative convention fixes label boundaries but cannot prove historical vendor availability or remove revisions.

Five expanding folds purge training labels that finish at or after the first assessment decision. Inner fitting and stopping partitions use the same completion-time rule. Annual and quarterly folds retain the original warm-up scheme. We reuse exact split identities and common origins. Neural training uses Adam 0.001, batch 128, at most twenty epochs and patience three; stopping selects raw endpoint MSE without refitting on the stopping segment.

We report endpoint MSE, MAE and full-path error. We define relative improvement as

\[
I=100(1-\mathrm{{MSE}}_{{model}}/\mathrm{{MSE}}_{{baseline}})\%.
\]

The reference is latest-close persistence for prices and zero for returns. We pool origins, and average the three seed losses at each origin before pooling. This differs from scoring averaged predictions. We never average dollar-squared and return MSEs across tasks.

Paired moving-block intervals use 5,000 draws, seed 2026 and nominal 95% bounds [10,11]. Minute blocks contain one/five sessions; hourly blocks contain two/five. Longer-scale primary/sensitivity rules remain 12/24 monthly groups, 3/6 monthly groups and 21/42 daily groups. Intraday draws stay within folds; longer-scale grids concatenate chronological fold groups because annual fold windows can be shorter than a block. Zero-origin groups remain on the grids. Intervals are descriptive, without multiplicity correction, after investigator exposure to these periods.

## 3. Structural diagnostics and controlled methods

For return predictions, or price changes relative to the origin, endpoint error satisfies

\[
E[(y-p)^2]=E[y^2]+E[p^2]-2E[py].
\]

The decomposition quantifies prediction energy and alignment; it does not identify their causes. We report it for all twenty tasks, along with bias, per-step errors, training histories and forecast correlations. In both designated return tasks, all four original recipes have energy costs above their alignment offsets. At minute H60, xPatch's normalized energy is 0.30353 against a 0.01454 alignment offset; LSTM's corresponding values are 0.00551 and 0.00430. This describes the error gap without identifying a responsible branch. Fixed-checkpoint perturbations remove recent/older input quarters, anchors or selected xPatch branch inputs. They describe sensitivity only. Actual retraining probes provide separate evidence. Of 400 core neural fits, 81 reached the twenty-epoch cap, including 36 hourly-price fits; none of the twenty minute-H60 return fits did. The absence of a cap explanation in that one task cannot establish adequate optimization across all scales.

The earlier minute H60 study found inconsistent anchor-removal effects and no two-fold gain from recurrent mean readout. Its local state-scaled candidate remained 0.1982% worse than zero across five folds and three seeds. Its two-layer convolution had local receptive field seven, with historical information also present through pooling and input transformation. That result neither proves longer context ineffective nor validates state scaling. Simultaneously trained output heads can absorb constant scaling, limiting that mechanism's interpretation.

### 3.1 Matched receptive-field and loss factors

All four new variants use eight causal convolutional layers, width eight, kernel three and ReLU. A common head reads the last state and temporal mean. Short-field layers use dilation one; long-field layers use 1,2,4,8,16,32,64,128. Their maximum local receptive fields are 17 and 511, with identical parameter count \(1432+17H\): 2,452 at H60 and 1,568 at H8. A shorter input offers only its available history; padding supplies no observations. Pooling still integrates the window, so this factor does not isolate all forms of long memory.

Inputs are split-aware past log closes, demeaned within each window and scaled using the inner fit. Return outputs start at zero. Price models separately learn dollar residuals added to \(P_t\); we do not exponentiate returns and call the result a squared-loss price mean. All tiny variants share zero-residual head initialization and the option for inner stopping to retain epoch-zero persistence. Original models include an explicit standardized price anchor at their readout and retain their earlier initialization. Those input and initialization differences limit comparisons across model families.

The original loss averages standardized path error. The aligned loss fixes equal weight on that path and its endpoint:

\[
L_A=\tfrac12 H^{{-1}}\sum_{{j=1}}^H e_j^2/\sigma_j^2+\tfrac12 e_H^2/\sigma_H^2.
\]

All \(\sigma_j\) come from the fitting segment. H1 makes the two losses identical. We also change only the loss in the original four architectures on the first two folds of the two mechanism tasks, producing sixteen retraining probes.

Before observing new scores, we fixed long-field/aligned loss as the main candidate. The full four-cell factorial covers minute H60 and hourly H8 returns at five folds and three seeds. The same main recipe enters all twenty tasks. Throughput admission selected {r['selected_fits']} new neural fits within an eight-hour CPU wall budget: eight workers, two logical CPUs each, no GPU. We retain same-input ridge and inner-calibrated scalar shrinkage of the fixed original LSTM. Paired factor comparisons and the loss contrast \(L_{{LA}}-L_{{LP}}-L_{{SA}}+L_{{SP}}\) assess main effects and beyond-additive interaction; the complete candidate alone cannot justify every module.

### 3.2 Fixed-expert temporal combination

We freeze original seed-2026 MLP, GRU, LSTM and xPatch predictions and add a zero-return expert. Controls are zero, four-expert equal averaging and nonnegative constant weights on the five-expert simplex. A linear four-input softmax gate has 25 trainable parameters. Inputs are short/long volatility ratio, absolute normalized short trend, signed normalized long change and known session position. Minute windows are 60/240 bars; hourly windows are 6/48. The supplement defines the formulas, training variance normalization and fixed 0.001 equal-weight penalty.

Folds one and two initialize past calibration; folds three, four and five receive expanding replay evaluations. Every calibration label must finish before the scored fold starts. Gate scaling, fitting and stopping use only those past records, with a purged 80/20 chronological inner split. We never fit weights using scored-fold labels. Three gate seeds measure gate randomness while the experts remain at one seed. Although the gate is small, the H60 four-expert system has 318,929 parameters including the gate.

## 4. Results

### 4.1 Original comparison and new cross-scale candidate

The original comparison completes 100 folds and 400 neural fits. Among forty nominal-price neural comparisons, {oldpositive} beat persistence. Minute H60 original improvements are -0.9240% for MLP, -1.1294% for GRU, -0.1204% for LSTM and -28.8996% for xPatch. Their equal prediction mean gives -1.6352%; different outputs do not guarantee useful complementarity.

Table 1 reports the fixed new recipe across all twenty tasks, with {Nseed} seeds for the main candidate. A positive entry alone does not establish stability, supported factors or independent forecasting value. The supplementary tables retain original eight-method errors and every new fold and seed.

Annual price and return gains are {pct(annual_price['improvement_pct'])} and {pct(annual_return['improvement_pct'])}, with all three seeds positive. Their paired 12/24-month interval bounds favor the candidate, but 47 overlapping origins supply few resampling blocks and no independent confirmation. Quarterly point gains are below 1%; both block lengths give intervals crossing zero. We retain these cross-scale findings without selecting a new primary task after seeing them.

Inner stopping retains epoch-zero persistence in {epoch_zero} of {len(tiny_jobs)} tiny-model fits. Many near-zero differences therefore accompany rejection of trained deviations. This behavior does not establish that the market contains no predictable signal.

**Table 1.** Main-candidate endpoint-MSE improvements over matched persistence/zero. Positive is better; units remain separate. Values rounded to 0.000% show no resolved gain at this precision.

{mdtable(['Task','Price','Return'],newtable)}

### 4.2 Mechanism tasks and loss probes

**Table 2.** Four-cell mean-seed-loss improvements over zero. Short/long changes only dilation; path/aligned changes only loss.

{mdtable(['Variant','Minute H60','Hourly H8'],ftable)}

{verdict} The supplement lists each requirement, all control intervals and seed ranges. {' '.join(factor_sentences)} Of the eight original task-model loss probes, {probe_description} improve on both screening folds; two folds cannot establish a general loss principle. The interaction contrast supports a beyond-additive joint benefit in {supported} of the four prespecified task/block combinations. Unsupported factors receive no mechanism advantage claim.

![Figure 1. Fold-mean improvements (gray dots) and origin-pooled mean-seed-loss improvements (black diamonds) for the fixed four variants. S/L denote short/long field; P/A denote path/aligned loss. Axes differ between tasks.](figures/factorial_folds.png){{width=3.3in}}

### 4.3 Fixed-expert replay

{' '.join(fusiontext)} These figures use only folds three to five and cannot be compared directly with the five-fold percentages in Table 2. We retain common replay origins for all single experts and combinations. The past-fit record and exact source identities accompany the supplement.

## 5. Interpretation and limits

The original task definitions reveal issues that model changes cannot remove. Annual evaluation has only 47 overlapping origins, with four paths in a nonoverlapping sensitivity. Monthly H21 has 21 overlapping paths crossing the August 2020 split; they account for 74.01% of zero-return and 76.91% of price-persistence squared error. They represent one event. A prior split-neutral retraining changes the target and yields unstable seed results, including GRU improvements of 3.96%, -24.67% and 3.99%. Those diagnostics remain separate from unchanged nominal labels.

We isolate two implementation factors within the tiny family; comparisons with original neural models also change input representation, size and initialization. A short-field model still pools the whole window, limiting any long-memory interpretation. The gate produces mixture weights without calibrated confidence. Low prediction correlation and a post-hoc oracle combination provide no deployable advantage without past-only calibration.

We had inspected all historical assessment periods before this follow-up. We exclude a quote seen during earlier provider research from features and scoring. The protected 2026Q3 series remains sealed; a three-month interval cannot supply a full annual target. None of these experiments supports an independent test claim. Historical vendor revision and point-in-time availability remain limitations. We evaluate no strategy, execution cost or profitability.

We assess computational reproducibility through checkpoints that retain Adam, RNG, epoch and batch position under one absolute deadline. We reproduce uninterrupted optimizer state after an actual process exit and resume. Saved-predictor replay and a detached inference package verify the implementation; those checks cannot improve predictive evidence. The shared-horizon comparison remains unfinished.

## 6. Conclusion

We preserve a twenty-task, five-scale AAPL comparison and add targeted structural diagnosis, a parameter-matched receptive-field/loss experiment, and past-calibrated fixed-expert combinations. The main candidate achieves {pct(m['improvement_pct'])} and {pct(h['improvement_pct'])} improvement over zero on the two designated mechanism tasks. {verdict} The complete task tables and ablations delimit any local advantage and retain negative outcomes. Future confirmation requires a frozen design and eligible observations beyond the exposed development data.

## Data and code availability

The accompanying package includes editable main/supplementary manuscripts, all original and new result tables, source identities, protocols, weights and an input-only inference interface. Full raw-data training requires the identified data versions and environment. The author will supply the email before external submission. No archival DOI, public deposit or external submission has been made.

## References

{refs.strip()}

[13] Bai, S., Kolter, J. Z., and Koltun, V. *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*. arXiv:1803.01271, 2018. https://arxiv.org/abs/1803.01271

[14] Bates, J. M., and Granger, C. W. J. “The Combination of Forecasts.” *Operational Research Quarterly*, 20(4), 451–468, 1969. https://doi.org/10.1057/jors.1969.103

[15] Jacobs, R. A., Jordan, M. I., Nowlan, S. J., and Hinton, G. E. “Adaptive Mixtures of Local Experts.” *Neural Computation*, 3(1), 79–87, 1991. https://doi.org/10.1162/neco.1991.3.1.79

[16] Campbell, J. Y., and Thompson, S. B. “Predicting Excess Stock Returns Out of Sample: Can Anything Beat the Historical Average?” *The Review of Financial Studies*, 21(4), 1509–1531, 2008. https://doi.org/10.1093/rfs/hhm055
'''
    # Raw f strings keep TeX backslashes literal; this source uses one TeX slash.
    text=text.replace('\\\\','\\')
    (BASE/'manuscript/MAIN_PAPER_EN.md').write_text(text,encoding='utf-8')
    supplement(r,core)
    shutil.copy2(old/'AUTHOR_INFORMATION.json',BASE/'manuscript/AUTHOR_INFORMATION.json')
    atomic(BASE/'manuscript/SOURCE_AUDIT.json',{'result_sha256':sha(BASE/'analysis/RESULTS.json'),
        'core_metrics_sha256':sha(CORE/'results/METRICS_POOLED.csv'),'original_title_preserved':True,
        'all_twenty_tasks_present':True,'new_training_fits':r['completed_new_fits'],
        'performance_goal_passed':success,'Q3_values_read':False,'email_pending_author':True})
    from research_report import main as explain_completed_results
    explain_completed_results()

def supplement(r,core):
    sections=['# Supplement: Multi-Horizon Forecasting of Apple Stock Prices and Returns Across Temporal Scales\n',
        'Wang Yang\n\nAll results concern exposed historical development. This supplement distinguishes the original seed-2026 matrix, the earlier minute mechanism study, and the new registered follow-up. The protected 2026Q3 series remains unscored.\n',
        '## S1. Original eight-method comparison\n']
    for task in read(BASE/'protocol/DATA.json')['task_files']:
        meta=read(BASE/'cache'/task/'TASK.json')
        rr=[z for z in core if z['profile']==meta['profile'] and int(z['horizon'])==meta['horizon'] and z['target']==meta['target']]
        sections+=['### '+task+'\n',mdtable(['Method','Origins','Endpoint MSE','Improvement'],
            [[z['model'],z['origins'],f"{float(z['endpoint_mse']):.8g}",pct(100*float(z['relative_improvement']))] for z in rr])]
    sections+=['## S2. New primary and all ablations\n',mdtable(['Task / variant','N','Seeds','MSE','Improvement','Positive folds'],
        [[z['task']+' / '+z['variant'],z['origins'],z['seed_count'],f"{z['endpoint_mse']:.8g}",pct(z['improvement_pct']),z['positive_fold_means']] for z in r['pooled']]),
        '### Seed ranges and formal criteria\n']
    sections.append('![Supplementary Figure S1. Main-candidate endpoint-MSE ratios across all twenty tasks. A ratio below one favors the candidate; price and return losses retain separate units.](figures/new_twenty_task_ratios.png){width=3.5in}\n')
    sections.append(mdtable(['Task / variant','Seed 2026','Seed 2027','Seed 2028'],
        [[z['task']+' / '+z['variant']]+[pct(z['seed_improvements_pct'][str(s)]) if str(s) in z['seed_improvements_pct'] else 'not run' for s in (2026,2027,2028)] for z in r['pooled']]))
    sections.append(mdtable(['Task','At least 1%','All seeds positive','At least 4 folds','Both-block controls','All met'],
        [[z['task'],z['checks']['minimum_one_percent'],z['checks']['all_three_seeds_positive'],
          z['checks']['four_positive_folds'],z['checks']['all_control_intervals_both_blocks'],
          z['performance_goal_passed']] for z in r['mechanism_performance']]))
    sections+=['## S3. Paired controls, factors and intervals\n',
        'Positive differences favor the listed primary. The intervals are nominal and descriptive, without multiplicity adjustment. Intraday blocks stay within folds; longer-scale grids concatenate fold groups. All three seeds contribute through mean per-origin loss.\n',
        mdtable(['Task / comparison','Block','Improvement','95% difference interval'],
            [[z['task']+' / '+z['primary']+' vs '+z['reference'],z['block'],pct(z['relative_improvement_pct']) if 'relative_improvement_pct' in z else z['status'],
             str(z.get('difference_ci95'))] for z in r['intervals']]),
        '### Factorial interaction\n',
        'The contrast is loss(long,aligned) minus loss(long,path) minus loss(short,aligned) plus loss(short,path). A strictly negative interval supports a beyond-additive joint gain.\n',
        mdtable(['Task','Block','Loss interaction','95% interval','Supported'],
            [[z['task'],z['block'],f"{z['mean_loss_interaction']:.8g}",str(z.get('difference_ci95')),
              z.get('joint_gain_beyond_additivity_supported',False)] for z in r['factorial_interactions']]),
        '## S4. Original-model loss probes\n',mdtable(['Task / model','Fold','Original MSE','Aligned MSE','Change'],
            [[z['task']+' / '+z['model'],z['fold'],f"{z['original_endpoint_mse']:.8g}",f"{z['aligned_endpoint_mse']:.8g}",pct(z['improvement_vs_original_pct'])] for z in r['original_loss_probes']]),
        '## S5. Fusion replay\n',
        'The expert seed is fixed at 2026. The gate has four inputs and five outputs, or 25 parameters. Experts are frozen; three gate seeds do not measure full-expert training robustness. Each calibration label completes strictly before the scored fold. The gate uses a past 80/20 chronological fit/stop split, with incomplete fit labels purged.\n',
        r'''For completed adjacent split-aware log changes \(r\), define the past-window quantities

\[
v_k=\sqrt{k^{-1}\sum_{j=0}^{k-1}r_{t-j}^{2}},\qquad q_k=\sum_{j=0}^{k-1}r_{t-j}.
\]

For short/long windows \(s,l\), the four gate inputs are

\[
f=\left(\frac{v_s}{\max(v_l,\epsilon)},\frac{|q_s|}{\max(\sqrt{s}v_l,\epsilon)},\frac{q_l}{\max(\sqrt{l}v_l,\epsilon)},\frac{\tau}{T}\right).
\]

Here \(\epsilon=10^{-8}\), \(\tau\) is known elapsed regular-session time and \(T\) is scheduled duration. Short/long windows are 60/240 minutes or 6/48 complete hourly bars. Fit-only scaling applies to the four features. Gate endpoint loss divides by inner-fit label variance, with penalty 0.001 times the sample mean squared Euclidean distance of the five weights from the uniform vector. Adam uses 0.01, at most 300 updates, patience 20, and unpenalized stop loss.
''']
    for fr in r['fusion']:
        sections+=['### '+fr['task']+'\n',mdtable(['Method','MSE','Improvement'],
            [[z['method'],f"{z['endpoint_mse']:.8g}",pct(z['improvement_pct'])] for z in fr['pooled']]),
            mdtable(['Gate versus','Block','Improvement','95% difference interval'],
                [[z['reference'],z['block'],pct(z['relative_improvement_pct']),str(z['difference_ci95'])] for z in fr['intervals']])]
    sections.append('![Supplementary Figure S2. Equal, constant and gated combinations on the same three evaluation folds. These replay percentages are separate from the five-fold main-model results.](figures/fusion_replay.png){width=3.8in}\n')
    sections+=['## S6. Scope, reproducibility and unresolved claims\n',
        'The earlier minute candidate failed its registered performance goal; it remains an exploratory substudy of the original five-scale paper. Original shared-versus-independent-horizon market comparisons remain unfinished. Neither the new path head nor expert fusion supplies that missing control.\n',
        'Original annual and corporate-action diagnostics remain in the earlier full-core report. These use exposed periods: 47 annual origins, four nonoverlapping annual paths and 21 overlapping monthly paths around one split event. Do not treat those paths as independent event replicates.\n',
        'The source CSVs include every fold/seed, per-step error and training history. JSON files retain source identities, uncertainty details and all admission criteria. Run receipts identify the checkpoint and prediction hashes. Full training/Adam progress remains in the working directory. The release manifest states which inputs, checkpoints and results it contains.\n',
        'The main author supplies the remaining email. No conference-specific sample screenshot was accessible in this environment; the layout follows the requested two-column style with a complete author affiliation, without claiming exact compliance with an unidentified venue template.\n']
    facts=read(BASE/'diagnostics/POOLED_DIAGNOSTIC_FACTS.json')
    mechanism_keys={key(*z) for z in MECHANISM}
    diagnostic_rows=[z for z in facts['model_task_rows'] if z['task'] in mechanism_keys]
    sections+=['## S7. Original-model diagnostic evidence\n',
        'The pooled diagnostic values, first-fold fixed-checkpoint input sensitivities and all training histories are retained in diagnostics/. Perturbation responses are not causal attributions or substitutes for retraining. Prediction energy and alignment use the zero-return or origin-relative-price reference.\n',
        mdtable(['Task / model','Energy / baseline','Alignment / baseline','Improvement'],
            [[z['task']+' / '+z['model'],f"{z['prediction_energy_over_baseline']:.6f}",
              f"{z['alignment_offset_over_baseline']:.6f}",pct(z['improvement_pct'])] for z in diagnostic_rows]),
        'Energy minus alignment offset equals the relative MSE excess over the baseline. These decompositions describe each saved recipe, not an architecture-wide law.\n',
        'The original twenty-epoch cap was reached by 81 of 400 neural fits, including 36 hourly-price fits. None of the twenty minute-H60 return fits reached it. Increasing the cap was not tested in this fixed-budget follow-up.\n']
    split_rows=[];bar_rows=[];task_ranges=[]
    for profile,(frequency,lookback,horizons) in PROFILES.items():
        bars=admitted_bars(frequency)
        observed_dates=bars.session_date[bars.observed]
        bar_rows.append([profile,frequency,lookback,str(observed_dates[0]),str(observed_dates[-1]),int(bars.observed.sum())])
        for horizon in horizons:
            for target in ('future_close','future_log_return'):
                task=key(profile,horizon,target);meta=read(BASE/'cache'/task/'TASK.json')
                validation=[]
                for fn,fold in meta['folds'].items():
                    with np.load(fold['split_file']) as arrays:parts={k:arrays[k] for k in arrays.files}
                    row={'task':task,'fold':int(fn),'horizon':horizon,'lookback':lookback}
                    for name,indices in parts.items():
                        row.update({name+'_origins':len(indices),
                            name+'_first_decision_ns':int(bars.end_ns[indices[0]]),
                            name+'_last_decision_ns':int(bars.end_ns[indices[-1]]),
                            name+'_last_label_completion_ns':int(bars.end_ns[indices[-1]+horizon])})
                    assert row['fit_last_label_completion_ns']<row['stop_first_decision_ns']
                    assert row['train_last_label_completion_ns']<row['validation_first_decision_ns']
                    split_rows.append(row);validation.extend(parts['validation'])
                if target=='future_log_return':
                    task_ranges.append([profile+' H'+str(horizon),len(validation),str(bars.session_date[validation[0]]),
                        str(bars.session_date[validation[-1]]),str(bars.session_date[validation[-1]+horizon])])
    with (BASE/'analysis/EXACT_SPLIT_RANGES.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(split_rows[0]));writer.writeheader();writer.writerows(split_rows)
    sections+=['## S8. Admitted history and exact split ranges\n',
        'These observed-bar counts follow the conservative completion cutoff used by training, rather than the counts in raw provider exports. Expected grid positions are retained even when a bar is unavailable. Annual and quarterly inputs are monthly bars; scale labels name the forecast horizon.\n',
        mdtable(['Scale','Bar unit','Lookback','First session','Last session','Bars'],bar_rows),
        'The following ranges cover the five exposed assessment folds. Price and return tasks share origins within each horizon. Exact per-fold fit, stop, training and validation decision/completion timestamps are included in analysis/EXACT_SPLIT_RANGES.csv; partitions/ retains the index arrays and fitted scalers.\n',
        mdtable(['Task','Origins','First origin','Last origin','Last completion'],task_ranges)]
    benchmarks=read(BASE/'audit/SYNTHETIC_THROUGHPUT.json')['models']
    implementations={
        'mlp':'Flattened log-relative history and anchor; 64-unit ReLU hidden layer, dropout 0.1, linear path head.',
        'gru':'One GRU layer of width 32; final hidden state concatenated with anchor before the linear path head.',
        'lstm':'One LSTM layer of width 32; final hidden state concatenated with anchor before the linear path head.',
        'xpatch':'Causal EMA with alpha 0.1; native trend/residual backbone, patch 8, stride 4, end padding; separate linear anchor head.'}
    architecture_rows=[]
    for model in ('mlp','gru','lstm','xpatch'):
        sizes=[next(row['parameters'] for row in benchmarks if row['family']=='original' and
                    row['model']==model and row['lookback']==lookback and row['horizon']==horizon)
               for lookback,horizon in ((480,60),(96,8))]
        architecture_rows.append([model,implementations[model],'Explicit',*sizes])
    sections+=['## S9. Original implementations and comparison limits\n',
        'Original neural models use the split-aware log-relative close window and a separately fit standardized latest-close anchor. Targets are standardized by step on the inner fit. The original seed is 2026. Adam, learning rate, batch size, twenty-epoch cap and patience follow Section 2. Original neural fits do not use the new tiny model\'s epoch-zero persistence option.\n',
        mdtable(['Model','Implementation','Anchor','H60 parameters','H8 parameters'],architecture_rows),
        'The xPatch adapter runs a native H2 output for H1 and exposes its first step because the native half-horizon pooling cannot represent H1. This is an implementation limitation, not a new xPatch design. Parameter totals include the anchor head. The reference source and third-party source/license metadata accompany the package.\n',
        'The new four variants match each other in width, depth, parameter count, input transformation, zero initialization and stopping rules. Dilation and loss are their two experimental factors. Comparisons against original architectures do not isolate a single factor; the sixteen original-model loss probes preserve their original inputs and architecture to test that narrower change.\n']
    (BASE/'manuscript/SUPPLEMENT_EN.md').write_text('\n\n'.join(sections),encoding='utf-8')
    status='achieved_on_exposed_development_only' if r['both_mechanism_tasks_pass'] else 'not_achieved'
    (BASE/'analysis/CONCLUSION_LEDGER_ZH.md').write_text(f'''# 结论与交付状态

原题、五尺度与二十任务保留。新增主模型拟合 {r['completed_new_fits']} 格，两个机制任务五折三种子四变体；主候选跨二十任务完整评价。

性能目标：{status}。实验完成、工程恢复、论文排版与独立验证分别验收。2026Q3 未读未评分，历史开发结果不能成为独立确认。

原多跨度共享市场实验仍未完成。邮箱由作者填写。任何摘要/结论引用的具体优势均以 RESULTS.json、配对区间与对应消融为准。
''',encoding='utf-8')

if __name__=='__main__':main()
