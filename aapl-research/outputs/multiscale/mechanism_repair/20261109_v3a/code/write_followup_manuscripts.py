"""Build an isolated v3a review draft from immutable v3 and verified supplements."""
from pathlib import Path
import hashlib
import json
import re
import shutil

BASE = Path(__file__).resolve().parents[1]
V3 = BASE.parent / '20261109_v3'
TITLE = 'Multi-Horizon Forecasting of Apple Stock Prices and Returns Across Temporal Scales'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pct(x):
    return f'{x:+.3f}%'


def label(task):
    scale, tail = task.split('_h', 1)
    h, target = tail.split('_', 1)
    return f'{scale.capitalize()} H{h} ' + ('price' if target == 'future_close' else 'return')


def table(headings, rows):
    return '\n'.join(['| ' + ' | '.join(headings) + ' |',
        '| ' + ' | '.join(['---'] * len(headings)) + ' |'] +
        ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def select(rows, task, method):
    return next(r for r in rows if r['task'] == task and r['method'] == method)


def interval(row):
    lo, hi = row['relative_ci95_pct']
    return f'[{lo:+.2f}, {hi:+.2f}]%'


def main():
    long = read(BASE / 'longscale_replication/analysis/RESULTS.json')
    means = read(BASE / 'mean_controls/audit/RESULTS.json')
    history = read(BASE / 'history_sensitivity/analysis/RESULTS.json')
    bnfile = BASE / 'bn_recalibration/RESULTS.json'
    controlfile = BASE / 'bn_recalibration/CONTROL_AUDIT.json'
    bn = read(bnfile) if bnfile.exists() and controlfile.exists() else None
    bncontrol = read(controlfile) if bn is not None else None
    original = read(V3 / 'analysis/CANDIDATE_RESULTS.json')
    assert long['paired_cells'] == 180 and long['new_cells'] == 150
    assert read(BASE / 'longscale_review/REVIEW.json')['status'] == 'passed'
    assert read(BASE / 'longscale_review/HISTORY_REVIEW.json')['status'] == 'passed'
    assert read(BASE / 'mean_controls/audit/REPLAY.json')['status'] == 'passed'
    assert original['primary_tasks_both_pass'] is False
    tasks = list(dict.fromkeys(r['task'] for r in original['pooled']))
    longtasks = list(dict.fromkeys(r['task'] for r in long['pooled']))
    assert len(tasks) == 20 and len(longtasks) == 6
    manuscript = BASE / 'manuscript'
    manuscript.mkdir(exist_ok=True)
    shutil.copytree(V3 / 'manuscript/figures', manuscript / 'figures', dirs_exist_ok=True)
    previous = manuscript / 'preceding_version'
    previous.mkdir(exist_ok=True)
    for name in ('MAIN_PAPER_EN.md', 'MAIN_PAPER_EN.pdf', 'MAIN_PAPER_EN.docx',
                 'SUPPLEMENT_EN.md', 'SUPPLEMENT_EN.pdf', 'SUPPLEMENT_EN.docx'):
        shutil.copy2(V3 / 'manuscript' / name, previous / name)
    shutil.copy2(V3 / 'manuscript/AUTHOR_INFORMATION.json', manuscript / 'AUTHOR_INFORMATION.json')

    def candidate(task):
        return select(long['pooled'], task, 'repair_shrink') if task in longtasks else select(original['pooled'], task, 'candidate')

    annualp = candidate('annual_h12_future_close')
    annualr = candidate('annual_h12_future_log_return')
    meanp = select(means['pooled'], 'annual_h12_future_close', 'shrink')
    meanr = select(means['pooled'], 'annual_h12_future_log_return', 'shrink')
    flatp = select(history['pooled'], 'annual_h12_future_close', 'flat_history')
    flatr = select(history['pooled'], 'annual_h12_future_log_return', 'flat_history')
    originaltext = (V3 / 'manuscript/MAIN_PAPER_EN.md').read_text()
    text = originaltext
    abstract = f'''We compare AAPL nominal-price and cumulative-return forecasts across five temporal scales and twenty tasks. An eight-method, five-fold historical comparison supplies 400 neural fits. We investigate input anchors, output initialization, target representation and stopping through a 52-cell screen, matched five-fold three-seed factorial replications, and exact-prefix training-cap checks. For xPatch, baseline-function initialization reduces minute H60 and hourly H8 return error by 9.60% and 44.11% relative to the default head, yet the fixed initialization-and-shrinkage recipe remains below zero returns by 0.038% and 0.795%. A post-observation supplement extends paired initialization replication to six long-scale tasks and adds fit-only unconditional mean controls. Annual price and return improvements over persistence/zero average {pct(annualp['improvement_pct'])} and {pct(annualr['improvement_pct'])} across three seeds, but their paired intervals against the shrunk mean controls include zero; the original nonoverlap sensitivity has only four paths. Fixed-weight history substitutions further limit a historical-pattern explanation. We identify implementation effects and report failed predictive criteria. We have not established a forecasting method superior to the strong controls. All assessment periods are exposed development data; protected 2026Q3 remains unscored.'''
    text = re.sub(r'(?<=## Abstract\n\n).*?(?=\n\n\*\*Keywords)', abstract, text, count=1, flags=re.S)
    text = text.replace('Single-seed cross-scale results are identified.',
        'Seed counts identify the fourteen task entries that retain the original replication depth; the six long-scale tasks now use three seeds.')
    followup = '''### 3.3 Post-observation long-scale and normalization follow-up

After inspecting the preceding controlled phase, we froze a separate supplemental plan. It is development follow-up, not the original registration or independent discovery. Six annual, quarterly and monthly price/return tasks compare the same default and baseline-initialized xPatch recipes at five folds and seeds 2026/2027/2028. The 180 cells reuse thirty exact preceding cells and add 150 fits. We retain the original inputs, targets, optimizer, stopping and scalar calibration; no annual outcome replaces the designated return criteria.

An additional unconditional control uses the inner-fit mean cumulative-log-return path, or the mean dollar-change path added to the current price. Its raw and inner-stop-shrunk forms use the same purging and endpoint scalar rule. A separate fixed-weight probe substitutes a fit-input mean history or a flat raw log-relative history while retaining the actual current-price anchor, learned parameters, scalers and original scalar. These substitutions can be out of distribution. They measure sensitivity of the calibrated pipeline, including twenty-seven of ninety configurations with scalar zero, and do not establish a market cause or a refitted forecasting improvement.

We also freeze a BatchNorm diagnostic on the two designated return tasks. Only inner-fit inputs contribute sequential estimates of the three BatchNorm layers' running moments; learned parameters remain fixed. Two earliest folds and seed 2026 must both improve raw inner-stop endpoint error before a task enters five-fold, three-seed continuation. The absolute compute deadline also applies. Adapter outer scores cannot determine admission. The supplement records the full outcome and a numerical-acceptance revision made before adapter outer scoring.

'''
    text = text.replace('## 4. Results\n', followup + '## 4. Results\n', 1)
    table1 = []
    for scale, hs in [('annual', [12]), ('quarterly', [3]), ('monthly', [21]),
                      ('hourly', [1, 4, 8]), ('minute', [5, 15, 30, 60])]:
        for h in hs:
            p, r = [candidate(f'{scale}_h{h}_{target}') for target in ('future_close', 'future_log_return')]
            table1.append([f'{scale.capitalize()} H{h}', pct(p['improvement_pct']), pct(r['improvement_pct']),
                           f"{p['seed_count']}/{r['seed_count']}"])
    newtable = ('**Table 1.** Fixed initialization-and-inner-shrinkage recipe: endpoint-MSE improvement over persistence/zero. '
                'P/R give seed counts; six long-scale entries use the post-observation three-seed supplement.\n\n' +
                table(['Task', 'Price', 'Return', 'Seeds P/R'], table1) + '\n\n')
    start, end = text.index('**Table 1.**'), text.index('**Table 2.**')
    text = text[:start] + newtable + text[end:]
    oldparagraph = text[text.index('Table 1 retains every original task.'):text.index('\n\nFor returns or price changes')]
    newparagraph = f'''Table 1 retains every original task. The supplemental three-seed annual price and return gains are {pct(annualp['improvement_pct'])} and {pct(annualr['improvement_pct'])}, each on the same 47 overlapping origins. Shrunk fit-only mean paths gain {pct(meanp['improvement_pct'])} and {pct(meanr['improvement_pct'])}. Both annual paired intervals against those means cross zero at the 12- and 24-month block lengths. On the original four-path nonoverlapping subset, the repair gains 17.585% and 7.470%, compared with 48.349% and 51.317% for the mean control. Additional seeds check training randomness and cannot supply new market years. Neither designated return task passes the original predictive gate. The preceding tiny convolution and fixed-expert gates also remain negative.'''
    text = text.replace(oldparagraph, newparagraph)
    historysection = f'''### 4.3 History sensitivity and robustness limits

With all parameters, anchors, scalers and scalars fixed, annual improvements fall to 2.814%/14.826% under the fit-mean history and rise to {pct(flatp['improvement_over_baseline_pct'])}/{pct(flatr['improvement_over_baseline_pct'])} under the flat history (price/return). The contrasting directions limit a general claim that reading the historical trajectory explains the annual gains. They do not prove that the raw networks ignore history, especially where scalar zero masks their outputs. We retain these as unrefitted diagnostic interventions and do not select the flat-input function as a candidate from its outer performance.

All twelve annual origin phases remain small, with three or four paths each; their shared historical labels do not form twelve independent tests. Leave-one-origin-year-out analyses also reuse exposed data. Annual and quarterly outer labels cross none of the reviewed future split events, so their split-exclusion contrasts are not estimable. Monthly H21 has twenty-one paths crossing the single August 2020 split. Detailed phase, year-exclusion, event and matched-control results accompany the supplement.

'''
    text = text.replace('## 5. Limitations and conclusion\n', historysection + '## 5. Limitations and conclusion\n', 1)
    if bn is not None:
        minute_bn = bn['tasks']['minute_h60_future_log_return']
        gain = minute_bn['improvement_over_zero_pct']
        rawcontrast = next(r for r in minute_bn['paired_intervals'] if r['primary'] == 'adapted' and
                           r['reference'] == 'original' and r['block'] == 1)
        assert all(not r['performance_gate_passed'] for r in
                   bncontrol['tasks']['minute_h60_future_log_return']['performance_modes'].values())
        bnsection = f'''### 4.4 Fixed-weight normalization diagnostic

Minute H60 passed the inner-only two-fold screen and completed five folds and three seeds. Fit-input moment replacement lowers its raw MSE by {rawcontrast['relative_improvement_pct']:.3f}% relative to the same initialized checkpoint; both day-block intervals support that local reduction. Raw forecasts nevertheless remain {gain['adapted']:.3f}% relative to zero. After separate inner-stop calibration, the gain is {pct(gain['adapted_shrink'])}, compared with {pct(gain['original_shrink'])} for the original repair. All three adapted-shrunk seed gains are negative; only one pooled fold is positive, and neither raw nor calibrated outputs pass the original all-control criteria. Hourly H8 fails its inner screen and remains a two-fold, single-seed diagnostic.

These fixed-weight comparisons show that the stored running moments affect inference error under this implementation. They do not establish that recomputing moments corrects every train/evaluation discrepancy: the learned weights were optimized with training-batch statistics. The final calibrated error and the strong controls provide no basis for claiming a better predictive method from this adjustment.

'''
        text = text.replace('## 5. Limitations and conclusion\n', bnsection + '## 5. Limitations and conclusion\n', 1)
    text = text.replace('The local review package contains editable manuscripts, registered protocols, all result tables and paired losses, source identities, model checkpoints and input-only inference.',
        'The local review addendum contains editable updated manuscripts, supplemental protocols, result tables, paired losses, source identities, supplemental checkpoints and input-only replay examples. The immutable preceding v3 archive supplies the controlled factorial study and its detached inference/statistics checks; the addendum identifies that dependency by SHA-256 rather than duplicating it.')
    (manuscript / 'MAIN_PAPER_EN.md').write_text(text)

    sections = [f'# Supplement to {TITLE}',
        'Wang Yang. Post-observation supplement v3a; exposed historical development, 2026Q3 unscored.',
        '## S1. Scope, chronology and accounting',
        'The original title, five temporal scales, twenty tasks and minute-H60/hourly-H8 return criteria remain fixed. This supplement follows inspection of v3, whose complete text, tables, factorial experiments and acceptance records remain unchanged in the preceding archive. The updated main Table 1 replaces only six long-scale single-seed entries with the same recipes at three seeds. Fourteen entries retain the earlier replication depth. Cross-task losses are never pooled across price/return units.',
        'The long-scale matrix has six tasks, two initialization recipes, five folds and three seeds: 180 cells, including thirty exact reused v3 configurations and 150 new fits. All receive the original raw and inner-stop-shrunk output rules. The unconditional mean study has twenty tasks and one hundred purged folds. The history probe uses ninety baseline-initialized checkpoints and three fixed-weight input conditions per checkpoint. These are different stage counts, not independent statistical samples.',
        'The original absolute compute deadline is epoch 1794205909.6228158 (2026-11-09 14:31:49.623 Beijing). At most eight training processes use two logical CPUs each; analysis processes share CPUs 24 and 26. No GPU, payment, additional download or protected-period read occurs. Checkpoints preserve optimizer/RNG state where training can resume; the actual two-update long-scale exit/resume has a passed recovery receipt.',
        '## S2. Six long-scale paired replications',
        'Each row reports the mean of per-origin losses across seeds 2026/2027/2028, followed by pooling origins. It is not the loss of an averaged forecast. Positive folds compare this mean seed loss with zero/persistence. Default and repaired recipes differ only in baseline-function initialization within each target-specific implementation.',
        table(['Task', 'Default', 'Default shrink', 'Repair', 'Repair shrink', 'Positive folds'],
            [[label(t)] + [pct(select(long['pooled'], t, m)['improvement_pct']) for m in
                ('default_trained', 'default_shrink', 'repair_trained', 'repair_shrink')] +
             [f"{candidate(t)['positive_folds']}/5"] for t in longtasks]),
        '## S3. Seed and strong-control comparisons',
        table(['Task', 'Seed 2026', 'Seed 2027', 'Seed 2028', 'Fit mean', 'Mean shrink'],
            [[label(t)] + [pct(candidate(t)['seed_improvements_pct'][str(s)]) for s in (2026, 2027, 2028)] +
             [pct(select(long['pooled'], t, m)['improvement_pct']) for m in ('fit_mean', 'fit_mean_shrink')]
             for t in longtasks]),
        'We retain Ridge with and without the nominal-price anchor, their original inner selections, both default outputs and both mean controls in the saved paired losses and the complete POOLED.csv. The following paired bounds compare the repaired-shrunk forecast with the shrunk mean control. Negative bounds permit the mean control to have lower error. The annual intervals do not establish superiority to this simple control. They are nominal descriptive intervals after exposure and multiple development decisions.',
        table(['Task', 'Block', 'Relative gain CI', 'Supports gain'],
            [[label(r['task']), r['block'], interval(r), str(r['supports_improvement'])]
             for r in long['paired_intervals'] if r['primary'] == 'repair_shrink' and r['reference'] == 'fit_mean_shrink']),
        '## S4. Unconditional controls across all twenty tasks',
        'Mean return paths average inner-fit cumulative-log-return targets. Mean price paths average inner-fit dollar changes and add each actual current-price anchor. Neither control reads the historical window beyond that anchor. The raw control uses the full fit mean; the shrunk control estimates the same scalar clipped to [0,1] from the purged inner-stop endpoints and applies it across the path. Outer targets never fit a mean or scalar.',
        table(['Task', 'Origins', 'Mean raw', 'Mean shrink', 'Repair shrink'],
            [[label(t), candidate(t)['origins'], pct(select(means['pooled'], t, 'raw')['improvement_pct']),
              pct(select(means['pooled'], t, 'shrink')['improvement_pct']), pct(candidate(t)['improvement_pct'])] for t in tasks]),
        'The shrunk mean controls fail to beat zero on minute H60 and hourly H8. The raw hourly-H8 mean has a small positive gain, below the original 1% threshold; neither form meets the original primary criteria. Adding these controls after v3 does not rewrite those criteria or create independent validation.',
        '## S5. Fixed-weight history sensitivity',
        'The actual-past condition replays every original calibrated path exactly. Fit-mean history averages only the raw log-relative inner-fit input windows; flat history sets that raw input to zero before the unchanged input scaler. Both retain the real current anchor, checkpoint, target scaler and original stopping alpha. No input substitution causes refitting or recalibration. Twenty-seven of ninety configurations have alpha zero; their calibrated outputs conceal any underlying network sensitivity. These potentially out-of-distribution substitutions cannot establish what an independently refitted no-history architecture would achieve.',
        table(['Task', 'Actual past', 'Fit-mean past', 'Flat past'],
            [[label(t)] + [pct(select(history['pooled'], t, m)['improvement_over_baseline_pct']) for m in
                ('actual', 'fit_mean_history', 'flat_history')] for t in longtasks]),
        'The two substitutions have different annual effects. The actual-versus-flat paired error contrasts favor the flat substitution at both frozen block lengths, but we do not promote a function selected after inspecting outer scores. These observations narrow the supported explanation to the implemented/calibrated functions.',
        '## S6. Nonoverlap, phase, year and corporate-action sensitivity',
        table(['Task', 'Paths', 'Repair shrink', 'Mean shrink'],
            [[label(r['task']), r['origins'], pct(r['methods']['repair_shrink']['improvement_pct']),
              pct(r['methods']['fit_mean_shrink']['improvement_pct'])] for r in long['sensitivity']
             if r['rule'] == 'original_global_earliest_greedy']),
        'Annual greedy nonoverlap leaves four paths. The absolute-bar phase subsets use the original origin index modulo the horizon, without optimizing a phase. Their common historical market exposure prevents independent-sample interpretation. Year exclusions remove origins by their origin calendar year; they do not remove all labels touching that year. We report ranges as descriptive dependence checks only.',
        table(['Task', 'Subset rule', 'Subsets', 'Origins range', 'Repair gain range'],
            [[label(t), rule.replace('_', ' '), len(rs),
              f"{min(r['origins'] for r in rs)}–{max(r['origins'] for r in rs)}",
              f"{min(r['methods']['repair_shrink']['improvement_pct'] for r in rs):+.3f}% to {max(r['methods']['repair_shrink']['improvement_pct'] for r in rs):+.3f}%"]
             for t in longtasks for rule in ('absolute_bar_phase', 'leave_origin_year_out')
             for rs in [[r for r in long['sensitivity'] if r['task'] == t and r['rule'] == rule and r['origins']]]]),
        'Annual and quarterly evaluation labels cross no reviewed future split events. Their split-exclusion robustness contrast is therefore not estimable; identical all/no-split scores do not count as a successful event-exclusion test. Monthly labels cross one split event on 2020-08-31 through twenty-one overlapping paths. Future nominal-price discontinuities and dividends remain as defined in the original targets.',
        '## S7. BatchNorm follow-up and numerical provenance',
    ]
    if bn is not None:
        sections.extend([
            'The fixed-weight normalization study retained seventeen checkpoints: fifteen minute-H60 cells at five folds and three seeds after inner-only admission, plus the two hourly-H8 screening folds at seed 2026 that failed admission. We replace only the running mean and variance of three BatchNorm layers, sequentially using upstream updated buffers and exact pooled inner-fit activation moments. Network parameters and other buffers remain unchanged. Float64 Welford accumulators and the current fit-input cursor support recovery. The actual interrupted/resumed accumulation also matches an independent pass from cursor zero.',
            table(['Task / scope', 'Repair raw', 'Adapted raw', 'Repair shrink', 'Adapted shrink'],
                [[label(t) + (' / 5 folds, 3 seeds' if r['scope'] == 'five_fold_three_seed' else ' / 2 folds, 1 seed only')] +
                 [pct(r['improvement_over_zero_pct'][m]) for m in ('original', 'adapted', 'original_shrink', 'adapted_shrink')]
                 for t, r in bn['tasks'].items()]),
            'Minute raw error decreases by 0.943571% relative to the same baseline-initialized raw checkpoint, with descriptive one/five-day paired intervals supporting this reduction. It still loses to zero and to calibrated strong controls. Adapted-shrunk pooled loss is slightly higher than the original shrunk repair; the corresponding relative-change intervals include zero. The hourly screen is not a full-task replication.',
            table(['Task', 'Over 1%', '3 seeds', '4 folds', 'All controls', 'Pass'],
                [['Minute ' + mode.replace('adapted', 'BN').replace('_', ' ')] +
                 [str(r['checks'][name]) for name in ('minimum_one_percent_over_zero',
                    'all_three_seed_directions_positive', 'four_positive_fold_means',
                    'all_controls_positive_both_block_intervals')] + [str(r['performance_gate_passed'])]
                 for mode, r in bncontrol['tasks']['minute_h60_future_log_return']['performance_modes'].items()]),
            table(['Seed', 'BN raw', 'BN shrink'],
                [[seed, pct(r['adapted']), pct(r['adapted_shrink'])] for seed, r in
                 bncontrol['tasks']['minute_h60_future_log_return']['per_seed_improvement_over_zero_pct'].items()]),
            'The all-control audit restores the original default and baseline-initialized raw/shrunk forecasts, zero, all four same-information Ridge variants and raw/shrunk mean paths. It recomputes per-origin mean seed losses, seed directions, fold directions and all 44 nominal paired intervals at one/five-day block lengths. No performance gate passes. The archive retains the control identities and losses.',
            'Interpreting this result requires its boundary: training used batch statistics to optimize the fixed weights. Sequential replacement of running moments can change inference error without removing every training/inference discrepancy or demonstrating useful return signal. We do not add a retraining, architecture or economic-mechanism claim.',
        ])
        (manuscript / 'BN_RESULTS_SNAPSHOT.json').write_text(json.dumps(bn, indent=2))
        (manuscript / 'BN_CONTROL_SNAPSHOT.json').write_text(json.dumps(bncontrol, indent=2))
    else:
        sections.append('The fixed diagnostic is still in progress at this draft. Minute H60 passed the two-fold raw inner-stop admission; hourly H8 did not because its first fold worsened. No full-task predictive conclusion is available from this admission screen. The draft must be updated with the final result before release.')
    sections.extend([
        'The first implementation required cross-batch allclose rtol 1e-6 and atol 1e-9. Before any adapter outer scoring, a screen failed this tolerance and the corresponding original checkpoint also failed it. A documented versioned revision changes only the return absolute tolerance to 1e-8, retaining rtol 1e-6; original and adapted checkpoints must both pass. We preserve the original source, freeze and failure receipt. Canonical batch-128 metrics must replay exactly, so this portability tolerance does not authorize changing a gate or saved score.',
        '## S8. Restoration, replay and limits of acceptance',
        'All thirty seed-2026 default restorations recover the original weights and selected/stopping epochs exactly. Twenty annual/quarterly paths match the original predictions bitwise. Ten monthly paths differ by small float32 batch-shape rounding because old inference used batch 512 and the frozen follow-up engine uses 128; rerunning the archived batch 512 reproduces the old predictions exactly. The frozen analysis initially demanded cross-batch bitwise equality and stopped. The retained diagnostic, analysis repair and preceding source hash explain that assertion correction; targets, training, statistics and gates did not change. We do not claim monthly cross-batch bitwise identity.',
        'Independent checks re-evaluate all 180 long-scale predictors, 180 stopping alphas, 360 seed/mode metric rows, 66 pooled rows and 120 paired intervals. Separate mean-control replay checks the one hundred folds. The history audit independently replays ninety checkpoints under three input conditions, re-sums fit inputs in a different batch order, and checks all 24 pooled conditions and 24 paired intervals. These checks concern computation and source/data boundaries. They provide no independent market history.',
        '## S9. Research status and remaining requirements',
        'The original primary repaired return gains remain −0.0380704% at minute H60 and −0.7950608% at hourly H8. No result in this supplement substitutes annual performance for the original designated task criteria. Improved default implementations, successful replay and editable paper delivery do not establish the requested reliable better forecasting method.',
        'We have not run the separate shared-versus-independent horizon market experiment. Eligible independent confirmation also remains unfinished: it needs an unexposed period, complete target labels, a frozen method/control set and a rule fixed before scoring. A single held-out quarter cannot confirm twelve-month outcomes. Protected 2026Q3 remains unscored. We claim no trading profitability, external submission, DOI or public deposit. The research Goal remains active and unachieved.',
        '## S10. Reproduction',
        'The addendum retains supplemental source freezes, rules, checkpoint identities, results, paired losses and inference examples. Its manifest identifies the immutable preceding v3 archive by hash. The README states which saved-function/statistical entrypoints were actually executed after extraction, which sources require registered-data path remapping for retraining, and which checks remain pending. The unchanged v3 supplement remains in manuscript/preceding_version for the original factorial/control tables and references.',
    ])
    (manuscript / 'SUPPLEMENT_EN.md').write_text('\n\n'.join(sections) + '\n')
    inputs = {str(p.relative_to(BASE)) if p.is_relative_to(BASE) else str(p): sha(p) for p in
              [V3 / 'manuscript/MAIN_PAPER_EN.md', V3 / 'analysis/CANDIDATE_RESULTS.json',
               BASE / 'longscale_replication/analysis/RESULTS.json', BASE / 'mean_controls/audit/RESULTS.json',
               BASE / 'history_sensitivity/analysis/RESULTS.json', Path(__file__)]}
    if bn is not None:
        inputs['bn_recalibration/RESULTS.json'] = sha(bnfile)
        inputs['bn_recalibration/CONTROL_AUDIT.json'] = sha(controlfile)
    (manuscript / 'SOURCE_AUDIT.json').write_text(json.dumps({'status': 'BN_complete_render_acceptance_pending' if bn is not None else 'draft_pending_BN_and_render_acceptance',
        'title': TITLE, 'original_v3_unchanged': True, 'inputs': inputs, 'table1': table1,
        'all20_tasks': tasks, 'supplemental_longscale_tasks': longtasks,
        'primary_success': False, 'independent_confirmation': False, 'Q3_values_read': False}, indent=2))
    print('v3a Markdown written; rendered and independent final acceptance pending.')


if __name__ == '__main__':
    main()
