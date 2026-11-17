"""Generate a new review version from complete controlled evidence, retaining title."""
from study import *
import shutil


TITLE = 'Multi-Horizon Forecasting of Apple Stock Prices and Returns Across Temporal Scales'


def pct(value, digits=3):
    return f'{value:+.{digits}f}%'


def interval(value):
    if value is None:
        return 'not estimable'
    return f'[{value[0]:.3g}, {value[1]:.3g}]'


def label(task):
    profile, rest = task.split('_h', 1); horizon, target = rest.split('_', 1)
    return f'{profile.capitalize()} H{horizon} '+('price' if target == 'future_close' else 'return')


def table(headings, rows):
    return '\n'.join(['| '+' | '.join(headings)+' |', '| '+' | '.join(['---']*len(headings))+' |']+
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])


def main():
    result = read(BASE/'analysis/CANDIDATE_RESULTS.json')
    assert not result['primary_tasks_both_pass'], 'Reassess negative-result wording if the fixed gate changes'
    rep = read(BASE/'analysis/REPLICATION_RESULTS.json'); cap = read(BASE/'analysis/CAP_RESULTS.json')
    probes = read(BASE/'analysis/STRUCTURAL_PROBES.json'); screen = read(BASE/'analysis/SCREENING_RESULTS.json')
    rules = read(BASE/'protocol/CANDIDATE_RULES.json'); selected = read(BASE/'protocol/CANDIDATE_JOBS.json')
    assert result['task_count'] == 20 and len(rep['original_restorations']) == 20
    assert all(r['model_and_stopping_exact'] for r in rep['original_restorations'])
    prior = read(PRIOR/'analysis/RESULTS.json')
    oldtext = (PRIOR/'manuscript/MAIN_PAPER_EN.md').read_text()
    references = oldtext.split('## References\n', 1)[1]
    manuscript = BASE/'manuscript'; manuscript.mkdir(exist_ok=True)
    author = read(PRIOR/'manuscript/AUTHOR_INFORMATION.json')
    atomic(manuscript/'AUTHOR_INFORMATION.json', author)
    mtask = 'minute_h60_future_log_return'; htask = 'hourly_h8_future_log_return'
    get = lambda task, method: next(r for r in result['pooled'] if r['task'] == task and r['method'] == method)
    minute, hourly = get(mtask, 'candidate'), get(htask, 'candidate')
    annual_p, annual_r = get('annual_h12_future_close', 'candidate'), get('annual_h12_future_log_return', 'candidate')
    return_contrasts = [r for r in rep['contrasts'] if r['model'] == 'xpatch' and r['mode'] == 'trained' and
                        r['primary'] == 'anchor_baseline' and r['reference'] == 'anchor_random']
    mc = next(r for r in return_contrasts if r['task'] == mtask)
    hc = next(r for r in return_contrasts if r['task'] == htask)
    cap_value = lambda model, method: next(r['endpoint_mse'] for r in cap['pooled'] if r['model'] == model and r['method'] == method)
    tasks_table = []
    for profile, (_, _, hs) in PROFILES.items():
        for h in hs:
            a, b = (get(task_key(profile, h, t), 'candidate') for t in ('future_close', 'future_log_return'))
            count = f"{a['seed_count']}/{b['seed_count']}"
            tasks_table.append([f'{profile.capitalize()} H{h}', pct(a['improvement_pct']), pct(b['improvement_pct']), count])
    table1 = table(['Task', 'Price', 'Return', 'Seeds P/R'], tasks_table)
    cap_table = table(['Model', 'Default 20', 'Default 60', 'Init. 20'],
        [[model.upper() if model == 'lstm' else 'xPatch', f'{cap_value(model,"default20"):.2f}',
          f'{cap_value(model,"default60"):.2f}', f'{cap_value(model,"price_baseline20"):.4f}'] for model in ('lstm', 'xpatch')])
    text = f'''# {TITLE}

Wang Yang

High School Affiliated to Shanghai Jiao Tong University

Shanghai 200439, China

[Email to be completed by the author]

## Abstract

We compare AAPL nominal-price and cumulative-return forecasts across five temporal scales and twenty tasks. An eight-method, five-fold historical comparison supplies 400 neural fits. We investigate whether differences attributed to architectures instead arise from input anchors, output initialization, target representation and stopping. A 52-cell screen precedes complete four-cell, five-fold, three-seed replications in four architecture-task groups; the default recipes reproduce the original weights and stopping decisions exactly. For xPatch, baseline-function initialization reduces minute H60 and hourly H8 return error by {mc['pooled_improvement_over_reference_pct']:.2f}% and {hc['pooled_improvement_over_reference_pct']:.2f}% relative to the default head, yet still fails to outperform zero returns. On hourly H1 prices, extending the default training cap does not reproduce the reduction obtained from initialization. We freeze one initialization-and-inner-shrinkage recipe and evaluate every original task. Its improvements over zero are {pct(minute['improvement_pct'])} for minute H60 and {pct(hourly['improvement_pct'])} for hourly H8; neither satisfies the registered performance criteria. The results identify consequential implementation factors without establishing a predictive method superior to strong references. All assessment periods are exposed development data; the protected 2026Q3 series remains unscored.

**Keywords:** multi-horizon forecasting; temporal scales; persistence; initialization; controlled comparison.

## 1. Introduction and related work

Predicting a stock's next five-minute return and its price twelve months ahead involves different outcome overlap, market events and effective sample sizes. Price forecasts also need an explicit persistence reference. A network that spends its budget learning the current price can appear much worse than a smaller residual model even when neither predicts future changes usefully.

We retain the original five-scale, twenty-task AAPL study and ask which implementation choices explain its model differences. The original methods are persistence, drift, seasonal repetition, ridge regression, MLP, GRU, LSTM and adapted xPatch. An earlier follow-up tested receptive field, endpoint weighting and past-calibrated combinations. Those candidates failed the designated return-task criteria. Comparing that small network with the original architectures also changed inputs, initialization, output representation and stopping. The present controlled phase separates these factors before testing one fixed repair across all tasks.

xPatch provides a trend/residual architecture [1], while PatchTST and Temporal Fusion Transformers exemplify established patch and multi-horizon designs [2,3]. Financial applications and their information sets vary [4–8]. Apple has direct precedent in multi-horizon clustering and reconciliation [9]. We assess established architectures under explicit finite-budget implementations; we claim neither a new architecture nor the first multi-horizon Apple study. Forecast combinations and shrinkage provide controls [14–16]. A separate shared-versus-independent-horizon market experiment remains unrun.

## 2. Data, tasks and evaluation

We use admitted Alpaca SIP/raw data before July 1, 2026 in New York. The daily source contains 2,637 rows from January 2016 through June 2026; conservative completion retains 2,636 dates and 125 full months. Intraday observations run from January 4, 2021 through June 30, 2026: 535,620 expected regular-session minutes and 8,238 complete hourly bars. Incomplete hourly remnants are excluded. XNYS defines the session grid, and minute-derived closes need not equal official closing-auction prices. The author reported permission from Alpaca and Kibot; only Alpaca supplies the reported experiments.

For a completed origin \\(t\\), the two separately fitted targets are \\(Y^P_{{t,j}}=P_{{t+j}}\\) and \\(Y^R_{{t,j}}=\\log(P_{{t+j}}/P_t)\\), for \\(j=1,\\ldots,H\\). Annual H12 and quarterly H3 use monthly bars; monthly H21 uses daily bars; hourly uses H1/H4/H8; minute uses H5/H15/H30/H60. Lookbacks are 12, 36, 252, 96 and 480 bars. Ten horizons and two targets give twenty tasks. Minute labels finish in the same regular session. H8 counts complete trading-hour bars and crosses a session; it is not eight clock hours.

Historical windows adjust only for splits already effective at the origin. Nominal future prices retain future split discontinuities, and dividends are excluded. Daily completion is conservatively assigned to the next New York midnight [12]. Vendor revisions and historical availability are not established by this convention.

Five expanding folds purge fit and stopping labels that reach the next partition's first decision. Scalers use only the inner fit; early stopping and calibration use only its purged tail. Original split identities and origins are preserved. Adam uses learning rate 0.001, batch 128, patience three and a twenty-epoch cap. We score endpoint MSE and retain MAE and full-path error. Relative improvement is

\\[
I=100(1-\\mathrm{{MSE}}_{{model}}/\\mathrm{{MSE}}_{{reference}})\\%.
\\]

References are current-close persistence for prices and zero for returns. Three-seed comparisons average loss at each origin before pooling, rather than scoring an averaged forecast. Single-seed cross-scale results are identified. Dollar and return errors are never averaged together.

Paired moving-block intervals use 5,000 draws, seed 2026 and nominal 95% bounds [10,11]. Minute blocks have one/five trading days, hourly two/five. Annual, quarterly and monthly blocks have 12/24 monthly groups, 3/6 monthly groups and 21/42 daily groups. Intraday folds remain separate strata; long-scale chronological groups concatenate because individual annual folds can be shorter than one block. All intervals are descriptive after historical exposure and screening, without multiplicity adjustment.

## 3. Controlled mechanism study

### 3.1 Audit and paired interventions

The earlier tiny model used window-demeaned inputs without the original nominal-price anchor, zero-function initialization and dollar residual price outputs. Its patience compared trained epochs with the epoch-zero baseline, whereas the original networks stopped relative to trained epochs only. Of 400 original fits, 81 reached twenty epochs; 206 of 390 tiny fits selected epoch zero and stopped at three epochs. These observations motivate hypotheses; they do not identify causes on their own.

Return interventions retain the original LSTM and xPatch, past log-relative histories, parameter layout and training. Their four cells cross observed versus constant-zero standardized price anchor with default versus exact-zero-return head initialization. The price diagnostic fixes hourly H1 and crosses direct price versus dollar residual output with default versus current-price-function initialization. Each output representation uses its own fit-only target scaler. Thus that parameterization intervention combines scale and representation; it does not isolate a skip connection alone.

The initial screen fixes the first two chronological folds and seed 2026: 32 return cells, 16 price cells and four tiny-model stopping probes. Every cell records the best trained epoch, optional final baseline selection, and the counterfactual baseline-coupled stopping prefix. Only prelisted comparisons improving in both screening folds admit an entire four-cell replication. This admits xPatch on minute H60 and hourly H8 returns, and LSTM/xPatch on hourly H1 prices: 240 cells at five folds and three seeds, including 32 reused screening cells. The LSTM return and tiny stopping branches do not pass this screen.

All twelve default screening restorations and all twenty default restorations in the full replication match the archived model weights and stopping epochs exactly. The four tiny counterfactual prefixes also match. Differences can therefore be interpreted as interventions within these concrete implementations, subject to the exposed development design. Algebraic output-path decompositions and the first two update gradients provide additional diagnostics, without proving economic mechanisms.

### 3.2 Training cap and fixed cross-scale recipe

Thirty hourly-H1 default price fits retain their optimizer, random state, batch cursor and history while a copied checkpoint permits at most sixty epochs under the same early-stop rule. Seventeen original fits are cap-bound; the other thirteen correctly receive no extra updates. This isolates the cap from a new random restart.

Before reading the full replication summary, we specify an architecture precedence and a mechanism condition: baseline-function initialization must improve hourly-H1 price error in all seed directions, at least four folds and both block intervals. The condition supports xPatch. We then freeze the original observed-anchor inputs, direct original targets, path loss and architecture; only the head starts at zero returns or current prices. A scalar calibrated on inner-stop endpoints shrinks the trained forecast toward persistence:

\\[
\\hat y=b+\\alpha(f-b),\\quad
\\alpha=\\operatorname{{clip}}_{{[0,1]}}\\!\\left(\\frac{{\\sum q r}}{{\\sum q^2}}\\right).
\\]

Here \\(q=f-b\\) and \\(r=y-b\\), where \\(f\\) is the trained forecast, \\(b\\) persistence and \\(y\\) the observed target. The scalar is fixed before each fold is scored and applies to every path step. This familiar calibration is a control and repair recipe, not a new model architecture. All twenty tasks receive the same rule: five folds, seed 2026 throughout, plus seeds 2027/2028 at the two designated returns and the price diagnostic. The 130 cells reuse 45 exact replication cells and add 85 fits. We compare both unshrunk and calibrated outputs with original defaults, inner-shrunk defaults, and same-information Ridge with/without the nominal-price anchor. Ridge fits only the inner fit and chooses from 0.1/1/10 regularization using the stopping tail.

Predictive success remains stricter than repairing a weak default: each designated return task must improve at least 1% over zero, have all three seeds positive and at least four positive folds, and beat the specified controls under both interval lengths. A successful price diagnostic cannot replace failed return criteria.

## 4. Results

### 4.1 What the interventions explain

Baseline-function initialization reduces xPatch return MSE relative to its default by {mc['pooled_improvement_over_reference_pct']:.2f}% at minute H60 and {hc['pooled_improvement_over_reference_pct']:.2f}% at hourly H8. Both contrasts improve in all five fold means and all three pooled seed directions, with supporting intervals at both lengths. Removing the default hourly anchor also reduces error. These are implementation effects in the fitted recipes; the initialized, unshrunk models remain worse than zero. Their final inner-shrunk return improvements are {pct(minute['improvement_pct'])} and {pct(hourly['improvement_pct'])}. Correcting an output failure does not establish usable return signal.

The two repairs have overlapping benefits. For both return tasks, the trained-loss interaction (both changes minus each separate change plus the default) is positive under both block intervals. The complete four-cell test therefore supports diminishing joint reductions, without evidence for an additional predictive component from combining those interventions.

Hourly-H1 results further distinguish learning a price level from forecasting its next change. Table 2 shows that the extended default retains substantial error, whereas current-price initialization is already near persistence under twenty epochs. Persistence MSE is {cap_value('lstm','persistence'):.4f} dollars squared. The initialized networks remain slightly worse. This supports a consequential initialization problem within the tested budget; it does not establish impossibility of training any default sufficiently.

**Table 1.** Fixed initialization-and-inner-shrinkage recipe: endpoint-MSE improvement over persistence/zero. P/R seed counts identify the single-seed scales.

{table1}

**Table 2.** Hourly-H1 mean-seed price MSE. Default-60 continues the exact default-20 prefix; initialized-20 retains the direct-price target.

{cap_table}

![Figure 1. Matched xPatch return recipes, with zero-return MSE set to one. Each bar pools per-origin mean seed loss. The four factorial cells and the calibrated baseline-initialized recipe are retained; a ratio below one would beat zero.](figures/return_controls.png){{width=3.3in}}

### 4.2 Cross-scale interpretation and preceding experiments

Table 1 retains every original task. The fixed candidate's annual price and return improvements are {pct(annual_p['improvement_pct'])} and {pct(annual_r['improvement_pct'])}, based on the same 47 overlapping origins. The four-path nonoverlapping sensitivity limits long-scale inference; a second training seed cannot create a new year. Detailed tables compare all controls and retain negative results. Neither designated return task passes the predictive gate. The prior tiny convolution and fixed-expert gates also remain negative on their designated return criteria and are retained as preceding experiments in the supplement.

For returns or price changes, the error decomposes as

\\[
E[(y-p)^2]=E[y^2]+E[p^2]-2E[py].
\\]

This separates forecast energy and alignment. A large price-anchor coefficient can inflate energy when nominal levels shift. The exact additive-path decomposition describes a fitted function; the matched refit supplies the separate intervention evidence. Zeroed final heads also yield zero backbone gradient on the first update in {sum(r['first_updates'][0]['backbone_all_zero'] for r in probes['rows'] if r['variant'].endswith('baseline'))}/{sum(r['variant'].endswith('baseline') for r in probes['rows'])} replicated baseline-initialized cells, followed by learned gradients. Initialization changes optimization as well as the starting forecast.

## 5. Limitations and conclusion

All assessment periods had been inspected before this phase. Screening, repeated development and multiple contrasts prevent independent-confirmation claims. Blocks measure a descriptive dependence sensitivity, not a new unexposed test. The annual sample is small and overlapping. Monthly H21 includes 21 overlapping paths spanning the August 2020 split, previously accounting for 74.01% of zero-return and 76.91% of price-persistence squared error; one event is not 21 independent shocks. Input completion, source revisions, session definitions and omitted dividends further limit interpretation.

The study identifies initialization, nominal-price anchors and finite training as consequential parts of the tested implementation. It does not infer a market mechanism from a sensitivity probe, and residual-output comparisons cannot separate their scaler changes. Cross-scale transfer of a price initialization repair is tested rather than assumed. No new recipe meets the designated stable predictive-improvement objective. The protected 2026Q3 series remains sealed, and a single quarter could not confirm annual H12 targets. Shared-versus-independent horizon learning and eligible independent confirmation remain unfinished. We evaluate no strategy, execution cost or trading profitability.

The original title and twenty-task comparison are retained. The additional contribution is controlled evidence about why particular implementations fail, together with a fully reported test of a fixed repair. Checkpoint recovery, predictor replay and detached saved-loss reproduction assess computational correctness; they do not turn these historical findings into independent forecasting success.

## Data and code availability

The local review package contains editable manuscripts, registered protocols, all result tables and paired losses, source identities, model checkpoints and input-only inference. The accompanying earlier package retains the original eight-method comparison and preceding causal-convolution/fusion experiments. Full retraining needs the identified raw-data versions, partitions, path remapping and recorded environment. The author will supply the email before submission. No external submission, public deposit or archival DOI has been created.

## References

{references}
'''
    assert TITLE in text and 'MECHANISM' not in text
    (manuscript/'MAIN_PAPER_EN.md').write_text(text, encoding='utf-8')
    sections = [f'# Supplement to {TITLE}', 'Wang Yang. Exposed historical development; 2026Q3 unscored.',
        '## S1. Provenance and accounting',
        'The original comparison supplies 400 neural fits. The preceding follow-up has 406 new fits. This phase has a 52-cell screen, 240-cell replication including 32 reused screen cells, 30 copied-prefix cap checks, and a 130-cell candidate including 45 reused replication cells. The phase therefore trains 345 fresh configurations and executes 30 cap checks; 17 cap checks add updates and 13 retain already-stopped fits. These overlapping stage counts must not be summed as independent experiments.',
        'The main title and five scales are unchanged. The same purged partitions, targets, source capsule and CPU environment are retained. Code and configuration hashes identify each trained function. GPU training is disabled; at most eight training processes use two logical CPUs each, while analysis shares two CPUs. One four-hour absolute training deadline includes screening, recovery and continuation, without reset.',
        '## S2. Controlled screening',
        'The first two chronological folds and seed 2026 are development screening. Twelve original models and four tiny stopping prefixes match exactly before attributing intervention differences.',
        table(['Task / model / variant', 'Fold', 'Trained gain', 'Selected gain'],
            [[label(r['task'])+' / '+r['model']+' / '+r['variant'], r['fold'], pct(r['improvement_pct']),
              pct(next(s['improvement_pct'] for s in screen['rows'] if all(s[k] == r[k] for k in ('task','model','variant','fold','seed')) and s['mode'] == 'selected'))]
             for r in screen['rows'] if r['mode'] == 'trained']),
        '## S3. Complete factorial replications',
        'All admitted groups retain four cells and five folds at seeds 2026, 2027 and 2028. The table reports per-origin mean seed loss. Coupled means the original baseline-participating stopping prefix; selected allows the final exact baseline after trained-only patience; shrink is the scalar inner-stop calibration.',
        table(['Task / model / method', 'MSE', 'Gain', 'Positive folds'],
            [[label(r['task'])+' / '+r['model']+' / '+r['method'], f"{r['endpoint_mse']:.7g}", pct(r['improvement_pct']), f"{r['positive_folds']}/5"]
             for r in rep['pooled'] if r['method'].endswith(('__trained','__shrink'))]),
        '## S4. Prespecified causal contrasts and interactions',
        'Supported means all three seed directions positive, at least four positive fold means, and lower error under both nominal block intervals. These are exposed development effects; they do not establish independent discovery. Every unrounded interval and all four output modes are retained in REPLICATION_INTERVALS.csv and REPLICATION_CONTRASTS.csv.',
        table(['Task / model / trained contrast', 'Relative reduction', 'Positive folds', 'Supported'],
            [[label(r['task'])+' / '+r['model']+' / '+r['primary']+' vs '+r['reference'],
              pct(r['pooled_improvement_over_reference_pct']), f"{r['positive_folds']}/5", r['replicated_development_mechanism_supported']]
             for r in rep['contrasts'] if r['mode'] == 'trained']),
        table(['Task / model', 'Block', 'Trained interaction', '95% interval'],
            [[label(r['task'])+' / '+r['model'], r['block'], f"{r.get('mean_loss_interaction',0):.6g}",
              interval(r.get('difference_ci95'))] for r in rep['interactions'] if r['mode'] == 'trained']),
        '## S5. Exact-prefix training-cap diagnostic',
        'The maximum changes from 20 to 60 epochs; the early-stopping rule is unchanged. Only originally cap-bound fits receive updates. Original checkpoints remain untouched. Thirteen already-stopped fits produce the same result. The following aggregate values use all thirty original cases, including unchanged ones.',
        table(['Model / method', 'MSE', 'Gain over persistence'],
            [[r['model']+' / '+r['method'], f"{r['endpoint_mse']:.7g}", pct(r['improvement_over_persistence_pct'])] for r in cap['pooled']]),
        '## S6. The fixed candidate and matched controls',
        'The architecture rule was fixed after the initial screen and before the complete replication summary. It tests baseline initialization under direct original targets and observed anchors, followed by standard scalar inner-stop shrinkage. The data are exposed throughout; this sequence is mechanism-guided development, not prospective confirmation. All twenty tasks use seed 2026; the two designated returns and H1 price diagnostic retain three seeds. Same-information Ridge includes both historical-window-only and historical-window-plus-anchor versions, with optional inner baseline selection. Original xPatch and its independently inner-calibrated scalar shrinkage remain controls.',
        table(['Task / method', 'MSE', 'Gain', 'Positive folds'],
            [[label(r['task'])+' / '+r['method'], f"{r['endpoint_mse']:.7g}", pct(r['improvement_pct']), f"{r['positive_folds']}/5"]
             for r in result['pooled'] if r['method'] in ('candidate','trained','original','original_shrink','ridge_history','ridge_history_anchor')]),
        '## S7. Primary return criteria',
        table(['Task', 'At least 1%', 'Three positive seeds', 'Four positive folds', 'All control intervals'],
            [[label(r['task']), r['checks']['minimum_one_percent'], r['checks']['all_three_seeds_positive'],
              r['checks']['four_positive_folds'], r['checks']['all_control_intervals_both_blocks']] for r in result['primary_performance_gates']]),
        'Neither mechanism repair nor successful replay is predictive qualification. The final design fails if any registered condition fails. Cross-scale positives do not change these prespecified endpoints.',
        '## S8. Preceding studies and complete original comparison',
        'The earlier convolutional factorial and past-only expert combination are retained as preceding studies. Their full methods, original eight-method tables, split records, negative findings and source audit remain in the preserved prior manuscript and supplement. The following twenty-task values retain its fixed long-field/aligned candidate and are not overwritten by the present recipe.',
        table(['Task', 'Prior candidate gain', 'Seeds', 'Origins'],
            [[label(r['task']), pct(r['improvement_pct']), r['seed_count'], r['origins']] for r in prior['pooled'] if r['variant'] == 'long_aligned']),
        '## S9. Structural interpretation and sample limits',
        'The nominal-price anchor enters the tested recurrent readout linearly and xPatch through a separate additive linear head. The saved-coefficient decomposition includes the residual skip when present and subtracts persistence before computing incremental prediction energy. It is algebraic attribution. A separate matched retraining contrast supplies intervention evidence. The first-step backbone gradient is zero when the final head is zeroed; later updates may change the backbone. Residual target standardization also changes optimization, preventing a pure skip-connection claim.',
        'The annual evaluation has 47 overlapping origins and four nonoverlapping sensitivity paths. Monthly H21 has 21 paths crossing one 2020 split event. All prior investigator exposure is retained. Increasing seeds does not create market history, and changing vendors does not create an independent period.',
        '## S10. Independent confirmation and outstanding scope',
        'The protected 2026Q3 series is neither opened nor included. A future confirmation requires an eligible, unexposed period, complete labels, purged boundaries, a frozen single design and controls, and an assessment rule fixed before scoring. Annual H12 needs twelve future monthly observations; one quarter cannot supply that test. No existing fold is relabeled independent. The original shared-versus-independent horizon market comparison is separate and remains unrun. The Goal for a reliable better forecasting method remains unmet.',
        '## S11. Reproduction and files',
        'The input-only predictor reconstructs forecasts from a stored checkpoint and completed past closes; it never retrieves target labels. Batched replay covers every new configuration and appropriate coupled/shrink outputs. The portable statistics script recomputes all mechanism, cap and twenty-task pooled losses, intervals and interactions from included per-origin loss arrays. Detached checks block access to the original research workspace and network while allowing the named installed Python environment. Training reproduction still requires the specified environment, admitted raw/derived data and path remapping; those limits are separate from saved-function replay.',
        'Unrounded full seed/fold rows: SCREENING_ROWS.csv, REPLICATION_ROWS.csv and CANDIDATE_ROWS.csv. Paired results: REPLICATION_INTERVALS.csv, CANDIDATE_INTERVALS.csv and CAP_RESULTS.json. Configuration, source, budget, recovery, visual and package receipts are retained alongside the manuscripts.',
        '## References', references]
    (manuscript/'SUPPLEMENT_EN.md').write_text('\n\n'.join(sections)+'\n', encoding='utf-8')
    prior_dest = manuscript/'preceding_version'; prior_dest.mkdir(exist_ok=True)
    for name in ('MAIN_PAPER_EN.md', 'SUPPLEMENT_EN.md', 'MAIN_PAPER_EN.pdf', 'SUPPLEMENT_EN.pdf'):
        shutil.copy2(PRIOR/'manuscript'/name, prior_dest/name)
    atomic(manuscript/'SOURCE_AUDIT.json', {'title': TITLE, 'author_source': str(PRIOR/'manuscript/AUTHOR_INFORMATION.json'),
        'prior_main_sha256': sha(PRIOR/'manuscript/MAIN_PAPER_EN.md'),
        'current_results_sha256': sha(BASE/'analysis/CANDIDATE_RESULTS.json'),
        'mechanism_results_sha256': sha(BASE/'analysis/REPLICATION_RESULTS.json'),
        'independent_confirmation': False, 'research_goal_achieved': False, 'Q3_values_read': False, 'at_utc': now()})
    print(json.dumps({'manuscripts_written': 2, 'title_preserved': TITLE, 'twenty_tasks_retained': True}), flush=True)


if __name__ == '__main__':
    main()
