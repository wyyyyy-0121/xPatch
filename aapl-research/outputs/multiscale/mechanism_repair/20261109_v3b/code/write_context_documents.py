"""Integrate the bounded context experiment into an isolated review draft."""
from pathlib import Path
import json
import shutil

BASE = Path(__file__).resolve().parents[1]
V3A = BASE.with_name('20261109_v3a')


def read(p):
    return json.loads(Path(p).read_text())


def pct(x):
    return f'{x:+.3f}%'


def table(headings, rows):
    return '\n'.join(['| ' + ' | '.join(headings) + ' |',
        '| ' + ' | '.join(['---'] * len(headings)) + ' |'] +
        ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def tasklabel(task):
    return 'Minute H60' if task.startswith('minute') else 'Hourly H8'


def ci(row):
    bounds = row['relative_ci95_pct']
    return 'degenerate' if bounds is None else f'[{bounds[0]:+.3f}, {bounds[1]:+.3f}]%'


def main():
    results = read(BASE / 'analysis/STATISTICS.json')
    outer = read(BASE / 'analysis/OUTER_RESULTS.json')
    assert outer['matrix_cells'] == 40 and len(outer['metrics']) == 80
    assert not results['independent_confirmation']
    tasks = list(results['tasks'])
    candidate = results['candidate']
    entries = [results['tasks'][task] for task in tasks]
    gains = [entry['summary'][candidate]['improvement_pct'] for entry in entries]
    positive = [entry['summary'][candidate]['positive_folds'] for entry in entries]
    def diagnostic(task_index, primary, reference):
        return next(row for row in entries[task_index]['diagnostic_contrasts']
                    if row['primary'] == primary and row['reference'] == reference)
    minute_dc = diagnostic(0, 'D_joint_raw', 'C_history_raw')['relative_improvement_pct']
    minute_ba = diagnostic(0, 'B_context_raw', 'A_mean_raw')['relative_improvement_pct']
    hourly_db = diagnostic(1, 'D_joint_raw', 'B_context_raw')['relative_improvement_pct']
    # The current documents describe this exact negative fixed recipe, and must
    # fail visibly rather than silently produce contradictory future prose.
    assert not any(entry['inner_direction_gate_passed'] or entry['performance_gate_passed'] for entry in entries)
    folder = BASE / 'manuscript'; folder.mkdir(exist_ok=True)
    shutil.copytree(V3A / 'manuscript/figures', folder / 'figures', dirs_exist_ok=True)
    shutil.copy2(V3A / 'manuscript/AUTHOR_INFORMATION.json', folder / 'AUTHOR_INFORMATION.json')
    text = (V3A / 'manuscript/MAIN_PAPER_EN.md').read_text()
    text = text.replace('We identify implementation effects and report failed predictive criteria.',
        f'A later fixed forty-cell study of origin-clock and weekday context with compressed history also fails its designated criteria ({pct(gains[0])} and {pct(gains[1])} against zero at minute H60 and hourly H8). We identify implementation effects and report failed predictive criteria.', 1)
    method = '''### 3.4 Origin-known context and compressed history

After the normalization follow-up, a separate development hypothesis asks whether information absent from the original price-history inputs can improve a direct conditional-mean forecast. For minute H60 and hourly H8, four matched deterministic regressions use an intercept only (A), origin-clock/weekday context (B), compressed past history (C), or their additive union (D). Context uses two fixed local-clock harmonics and four weekday indicators with Friday as reference, from the completed origin bar alone. Nine history features comprise seven cumulative-return lags and two trailing RMS increments. C and D omit the nominal-price anchor; this is a new low-dimensional information design, not an isolated addition to xPatch.

Fit-only standardized regressions use a fixed normalized ridge penalty of 0.1 and the same complete cumulative-return paths. Each model has its own inner-stop scalar; D with that scalar is the sole candidate. A fit-only timing probe admitted all two tasks by five folds by four cells before any new stop or outer scoring. The original first-two-fold raw-D direction check remains required for independent-confirmation eligibility. We compare the fixed candidate against 31 prelisted controls at both original day-block lengths, including every seed of the preceding stochastic benchmarks. Deterministic fits have no training-seed replication. Full specifications and all failed directions appear in the supplement.

'''
    text = text.replace('## 4. Results\n', method + '## 4. Results\n', 1)
    resultsection = f'''### 4.5 Bounded conditional-context test

Both tasks fail the first-two-fold inner direction check. The complete forty-cell matrix nevertheless remains reported under the pre-score cost admission. The fixed D-shrink candidate gains {pct(gains[0])} at minute H60 and {pct(gains[1])} at hourly H8 against zero, with {positive[0]}/5 and {positive[1]}/5 positive folds. Neither task passes the fixed performance gate or qualifies for independent confirmation.

At minute H60, raw D versus matched history C gains {pct(minute_dc)}, and context B versus unconditional A gains {pct(minute_ba)}; both nominal intervals are wholly negative at both block lengths. At hourly H8, raw D versus context-only B gains {pct(hourly_db)} with negative intervals, while D-versus-C intervals include zero. These comparisons limit the selected additive conditional-mean representation; they cannot establish that calendar context is irrelevant to other forecasting or volatility models. No alternate cell, lag set or regularization is promoted using outer scores.

'''
    text = text.replace('## 5. Limitations and conclusion\n', resultsection + '## 5. Limitations and conclusion\n', 1)
    text = text.replace('The local review addendum contains editable updated manuscripts,',
        'The local context addendum contains the updated manuscripts and forty-model input-only inference/statistics replay. It identifies the immutable v3 and v3a dependencies by SHA-256 for the preceding experiments. The preceding review addendum contains editable manuscripts,', 1)
    (folder / 'MAIN_PAPER_EN.md').write_text(text)

    rows = []
    for cell in ('A_mean', 'B_context', 'C_history', 'D_joint'):
        for mode in ('raw', 'shrink'):
            name = cell + '_' + mode
            rows.append([name.replace('_', ' ')] + [pct(e['summary'][name]['improvement_pct']) for e in entries])
    contrasts = []
    for task, entry in zip(tasks, entries):
        for row in entry['diagnostic_contrasts']:
            contrasts.append([tasklabel(task), row['primary'].replace('_raw', '').replace('_', ' ') + ' / ' +
                row['reference'].replace('_raw', '').replace('_', ' '), row['block'],
                pct(row['relative_improvement_pct']), ci(row)])
    intervals = []
    for task, entry in zip(tasks, entries):
        for control in ('zero', 'C_history_shrink', 'repair_three_seed_shrink', 'ridge_history_selected'):
            for row in entry['comparisons'][control]:
                intervals.append([tasklabel(task), control.replace('_', ' '), row['block'],
                    pct(row['relative_improvement_pct']), ci(row)])
    alphas = []
    for task in tasks:
        for fold in range(1, 6):
            done = read(BASE / 'fits' / task / f'fold_{fold}' / 'INNER_DONE.json')
            alphas.append([tasklabel(task), fold] + [f"{done['inner_metrics'][cell]['alpha']:.6f}" for cell in ('A_mean','B_context','C_history','D_joint')])
    supplemental = f'''

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

{table(['Method', 'Minute H60', 'Hourly H8'], rows)}

The fixed candidate's positive-fold counts are {positive[0]}/5 and {positive[1]}/5. Both inner qualification and complete performance gates fail. All 31 controls, their two block intervals per task, all seed-specific controls, and all negative/degenerate intervals are preserved in STATISTICS.json and the companion CSV files. No multiplicity correction is claimed. The interval resamples moving trading-day blocks within each original fold, then pools by origin count; minute blocks are one/five days and hourly blocks are two/five days, with 5,000 draws and seed 2026.

Selected fixed-candidate contrasts are shown below; the package preserves the full control set. The percentage is relative to the listed reference, not necessarily to zero.

{table(['Task', 'Reference', 'Block', 'Gain', '95% interval'], intervals)}

Raw diagnostic comparisons evaluate context beyond history (D/C), history beyond context (D/B), and context beyond the unconditional mean (B/A). Relative improvement uses the second method as reference. A positive descriptive interval describes only this representation on exposed historical data.

{table(['Task', 'Comparison', 'Block', 'Gain', '95% interval'], contrasts)}

All inner-only scalars are retained, including exact zero values that suppress raw outputs.

{table(['Task', 'Fold', 'A alpha', 'B alpha', 'C alpha', 'D alpha'], alphas)}

## S14. Context replay and remaining research requirements

The real market fit process exited after 4,096 origins and resumed from the stored cursor. Prefix and complete sufficient statistics, and all four solutions, match an uninterrupted run bit-for-bit at the same batch order. This verifies recoverability, not forecasting skill. An independent reviewer reconstructs all forty models, inner scalars and complete outer paths with a different accumulation order, then checks source controls and block statistics. The separate portable entrypoints reconstruct predictions from input-only examples and statistics from saved per-origin endpoint/path losses with complete day grids. These entrypoints need no original workspace or target series. Full retraining still needs the identified original sources and environment.

The mechanism-and-improvement research Goal remains active. Initialization and normalization diagnostics provide bounded implementation evidence. This context recipe does not achieve the required method contribution; no eligible independent confirmation has been run. The full five-scale, twenty-task results and the original title are preserved, 2026Q3 remains sealed, and the original shared-versus-independent horizon market experiment is still unexecuted. Predictive success, independent confirmation and manuscript/replay delivery remain separate acceptance items.
'''
    supplement = (V3A / 'manuscript/SUPPLEMENT_EN.md').read_text() + supplemental
    (folder / 'SUPPLEMENT_EN.md').write_text(supplement)
    (BASE / 'analysis/DOCUMENT_RESULT_SNAPSHOT.json').write_text(json.dumps({
        'candidate': candidate, 'gains_pct': gains, 'positive_folds': positive,
        'inner_gates': [e['inner_direction_gate_passed'] for e in entries],
        'performance_gates': [e['performance_gate_passed'] for e in entries],
        'original_title_preserved': True, 'old_manuscripts_modified': False}, indent=2) + '\n')
    print('Isolated v3b main and complete supplement sources written.')


if __name__ == '__main__':
    main()
