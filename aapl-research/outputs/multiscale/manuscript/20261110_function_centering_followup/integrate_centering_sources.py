"""Integrate accepted small reports without loading market arrays or weights."""
import json
import hashlib
from pathlib import Path

BASE = Path(__file__).resolve().parent
PREVIOUS = BASE / 'review/previous_edition'
EVIDENCE = BASE / 'evidence/function_centering'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def main():
    results = load(EVIDENCE / 'market_runs/market/RESULTS.json')
    protocol = load(EVIDENCE / 'MARKET_PROTOCOL.json')
    rules = load(EVIDENCE / 'review/INDEPENDENT_MARKET_REPLAY_RULES.json')
    assert results['status'] == 'completed_all_fixed_states'
    assert not results['decision']['continue_to_formal_development_review']
    source = (PREVIOUS / 'MAIN_PAPER_EN.md').read_text()
    original_supplement = (PREVIOUS / 'SUPPLEMENT_EN.md').read_text()
    old_phrase = 'A same-information session screen passes only one of four states, and extending the sole capped primary GRU leaves its selected predictor unchanged.'
    assert source.count(old_phrase) == 1
    source = source.replace(old_phrase, old_phrase + ' A matched twelve-learner LSTM function-centering probe also fails its fixed continuation rule, with equal-fold endpoint gains over zero of 0.222% and 0.584% on the two primary return tasks.')
    methods = r'''### 3.9 Matched function-centering probe

We compare three recipes in the original one-layer LSTM32 on minute-H60 and hourly-H8 returns, folds 1/2 and seed 2026. D uses default initialization. Z retains the same initial recurrent backbone but sets readout weights to zero and its standardized bias to \(b=-\mu/\sigma\), where \(\mu,\sigma\) are horizon-wise prefix target moments. C starts its learner at the same complete default state as D and subtracts a separate frozen copy of that initial network:

\[
q_C=f_\theta(x,a)-f_{\theta_0}^{\mathrm{ref}}(x,a)+b,
\qquad \hat y=\mu+\sigma\odot q_C.
\]

All arms share the original log-relative history, nominal-price anchor, prefix-only scalers and batch order. The original fit's last 20% of origins form the tail; the earlier prefix is purged by label completion. Twelve learners complete eight fixed epochs, with 9,840 common batch positions and 29,520 optimizer updates. Adam and standardized full-path loss are fixed, without stopping or best-epoch selection. All twelve learners and four sets of fitted controls are committed before the dedicated tail pass. We compare C with D, Z, zero, the prefix horizon-wise mean and same-information Ridge at normalized penalty 0.1. Every state's endpoint and full-path MSE must improve by at least 0.1% over each control; each task's equal-fold endpoint improvement over zero must reach 1%. This is a finite-budget development filter, separate from formal five-fold, three-seed success (S28).

'''
    source = source.replace('## 4. Results\n', methods + '## 4. Results\n', 1)
    labels = {'minute_f1': 'Minute / 1', 'minute_f2': 'Minute / 2', 'hourly_f1': 'Hourly / 1', 'hourly_f2': 'Hourly / 2'}
    short = {'D': 'D', 'Z': 'Z', 'C': 'C', 'zero': 'Zero', 'prefix_update_mean': 'Prefix mean', 'same_information_Ridge': 'Ridge'}
    summaries = []
    score_rows = []
    comparison_rows = []
    scalar_rows = []
    for state in results['results']:
        sid = state['state']
        p = state['predictors']
        gain = lambda control, metric: 100 * (1 - p['C'][metric] / p[control][metric])
        gate = next(x for x in results['decision']['per_state'] if x['state'] == sid)
        passes = sum(x['passed'] for x in gate['checks'])
        summaries.append(f"| {labels[sid]} | {gain('zero', 'path_MSE'):+.6f}% | {gain('zero', 'endpoint_MSE'):+.6f}% | {passes}/10 |")
        for name in ['D', 'Z', 'C', 'zero', 'prefix_update_mean', 'same_information_Ridge']:
            endpoint = p[name]['endpoint_MSE']
            path = p[name]['path_MSE']
            ge = 100 * (1 - endpoint / p['zero']['endpoint_MSE'])
            gp = 100 * (1 - path / p['zero']['path_MSE'])
            score_rows.append(f'| {labels[sid]} | {short[name]} | {endpoint:.17g} | {path:.17g} | {ge:+.9f}% | {gp:+.9f}% |')
            scalar_rows.append({'state': sid, 'predictor': name, 'endpoint_MSE': endpoint, 'path_MSE': path, 'endpoint_gain_percent': ge, 'path_gain_percent': gp})
        for control in ['D', 'Z', 'zero', 'prefix_update_mean', 'same_information_Ridge']:
            row = {x['metric']: x['passed'] for x in gate['checks'] if x['control'] == control}
            comparison_rows.append(f"| {labels[sid]} | {short[control]} | {gain(control, 'endpoint_MSE'):+.9f}% | {'Pass' if row['endpoint_MSE'] else 'Fail'} | {gain(control, 'path_MSE'):+.9f}% | {'Pass' if row['path_MSE'] else 'Fail'} |")
    result_text = '''### 4.10 No qualifying gain from matched function centering

The complete four-state probe meets its numerical prerequisites. C's equal-fold endpoint reductions against zero are 0.2219949% for minute H60 and 0.5839318% for hourly H8, below the prespecified 1% criterion. No state passes all ten control comparisons (Table 5); twenty-two of forty comparisons pass and eighteen fail. All eight C-versus-Z comparisons fail the required 0.1% margin, although some raw MSE differences favor C by less than that margin. The fixed recipe stops without admission to five-fold, three-seed development. S28 reports all six predictors and both losses.

**Table 5.** Matched function-centering probe. Gains are relative MSE reductions for C against zero. A state requires all ten control comparisons to pass, as well as its task-level rule. Minute H60 and hourly H8 each retain two historical development folds.

| State | Path gain | Endpoint gain | Passed |
| --- | ---: | ---: | ---: |
''' + '\n'.join(summaries) + '''

C's fixed nonlinear offset, translated function class and optimization trajectory belong to the total recipe effect. The comparison does not isolate the causal effect of first-step backbone gradients. C additionally stores a frozen network and requires another forward pass. The second-fold prefix includes every first-fold tail origin within each task, so the globally frozen prediction pass is not an unseen-market test. Independent reconstruction verifies preprocessing, Ridge and neural outputs; saved-prediction statistics reproduce the same failed decision. Neither computation adds independent market observations (S29).

'''
    source = source.replace('## 5. Limitations and conclusion\n', result_text + '## 5. Limitations and conclusion\n', 1)
    conclusion = 'The broader objective of an explained, reproducible forecasting improvement remains unachieved, with eligible independent market confirmation still outstanding.'
    assert source.count(conclusion) == 1
    source = source.replace(conclusion, 'The later function-centering probe further constrains one fixed initialization recipe; four small positive endpoint differences against zero do not establish superiority over the other controls. A single seed and eight fixed epochs do not establish an architecture\'s fully optimized performance. ' + conclusion)
    marker = '[The reproduction guide](REPRODUCTION_GUIDE_EN.md)'
    assert source.count(marker) == 1
    source = source.replace(marker, 'S28–S29 append the complete matched function-centering probe and its independent reconstruction receipts. Their small source/result snapshots are in evidence/function_centering. The separate failed descriptive-percentage verification is documented as an engineering limitation, without admitting a dynamic-contribution claim. ' + marker)
    source = source.replace('The integrated documentation retains the partial-probe receipts, complete fixed-volume results and bounded primary-text findings in S19–S21. S22–S24 retain the structural/scale checks, fixed-readout fit decomposition and fresh temporal intervention. This edition adds the calendar audit, fixed session-representation screen and matched-resource cap diagnostic in S25–S27, with small source/result snapshots in evidence/calendar_audit, evidence/session_alignment and evidence/gru_cap. Numerical replay of those market diagnostics requires the identified external inputs and checkpoints; the documentation bundle contains neither raw licensed bars nor model arrays.', 'The retained S19–S21 describe the partial-probe receipts, complete fixed-volume results and bounded primary-text findings. S22–S24 retain the structural/scale checks, fixed-readout fit decomposition and fresh temporal intervention; S25–S27 retain the calendar audit, fixed session-representation screen and matched-resource cap diagnostic. The immediately preceding immutable documentation archive supplies those earlier source/result snapshots and dependency maps without recursive duplication in this edition. Numerical replay requires their identified external inputs and checkpoints; neither documentation bundle contains raw licensed bars or model arrays.')
    source = source.replace('The current shared-horizon addendum adds', 'The earlier shared-horizon computation addendum supplies')
    (BASE / 'MAIN_PAPER_EN.md').write_text(source)

    cohorts = []
    for s in rules['states']:
        cohorts.append(f"| {labels[s['id']]} | {s['original_fit_count']:,} | {s['boundary']:,} | {s['prefix_count']:,} | {s['tail_count']:,} |")
    supp = r'''

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
''' + '\n'.join(cohorts) + r'''

Each arm receives the same full origin-relative log-close history, with only splits effective at the origin, and the same nominal closing-price anchor. Prefix-only population moments standardize histories, anchor and horizon-specific targets. The ordered scaler reductions use blocks of 4,096 origins and preserve the original FP32 transformed-input/target boundary. No scaler from the complete original fit cohort is reused.

Seed 2026 initializes each matched backbone; a separate NumPy generator with seed 2026 fixes shared epoch permutations. All twelve learners complete eight epochs. The four states use 3,216/6,440/64/120 common batch positions and 9,648/19,320/192/360 optimizer updates, totaling 9,840 shared positions and 29,520 updates. Batch size is 128; a lone final sample merges into the preceding batch. Uniform standardized complete-path MSE and Adam are fixed: learning rate 0.001, betas 0.9/0.999, epsilon 1e−8 and zero decay. There is no scheduler, clipping, early stopping or best-epoch choice. The final epoch is used. This equal epoch count does not imply equal training cost for C, or sufficient optimization of every architecture.

Controls are D, Z, zero returns, the prefix horizon-wise mean and same-information Ridge. Ridge takes the same FP32 transformed full history and anchor, then centers/scales columns in FP64 and fits raw multi-output returns with an unpenalized prefix-mean intercept and normalized penalty 0.1. The penalty is fixed before outcomes; it differs from the original study's Ridge selection procedure. Numerical qualification includes finite coefficients, a variance-roundoff check and normwise normal-equation backward error no greater than 1e−10.

All twelve learners and four fitted-control objects are committed before the dedicated tail prediction pass. The worker reloads the committed payloads. No dedicated tail target is requested before the global freeze; state-specific update permissions and nested folds still prevent a globally blind interpretation. We score raw-return endpoint and full-path MSE for all six predictors on the same origins.

### S28.3 All predictor losses

**Table S28.2.** Complete raw-return losses and gains against zero. Positive gains mean lower MSE. Ridge denotes the fixed same-information control; Prefix mean denotes the horizon-wise update-only mean. All values come from the accepted saved market result, independently reconstructed from saved predictions.

| State | Predictor | Endpoint MSE | Path MSE | Endpoint gain | Path gain |
| --- | --- | ---: | ---: | ---: | ---: |
''' + '\n'.join(score_rows) + '''

### S28.4 Complete fixed continuation decisions

C must reduce both MSE measures by at least 0.1% relative to each of five controls in every state. Independently for each task, the arithmetic mean of the two fold-relative endpoint improvements against zero must reach 1%. These conditions are conjunctive; no state subset, alternative seed, epoch budget, penalty or backend can rescue this fixed probe. Passing would only admit review of a formal five-fold, three-seed historical-development recipe, without opening protected data.

**Table S28.3.** All forty control-metric comparisons. Gains use each row's control as denominator. Pass/Fail applies the predeclared 0.1% margin to the unrounded MSE values.

| State | Control | Endpoint gain | Endpoint gate | Path gain | Path gate |
| --- | --- | ---: | --- | ---: | --- |
''' + '\n'.join(comparison_rows) + '''

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
'''
    (BASE / 'SUPPLEMENT_EN.md').write_text(original_supplement.rstrip('\n') + supp + '\n')
    (BASE / 'review/NEW_TABLE_SCALARS.json').write_text(json.dumps({'source_result_sha256': sha(EVIDENCE / 'market_runs/market/RESULTS.json'), 'score_rows': scalar_rows, 'all_four_states_and_six_predictors': True, 'not_new_market_scoring': True}, indent=2) + '\n')
    (BASE / 'SOURCE_CHANGELOG.json').write_text(json.dumps({'original_title_preserved': True, 'five_scales_twenty_tasks_preserved': True, 'previous_sources': {name: sha(PREVIOUS / name) for name in ['MAIN_PAPER_EN.md', 'SUPPLEMENT_EN.md']}, 'changes': ['Abstract: accepted centering outcome', 'Methods3.9: exact three recipes and fixed filter', 'Results4.10/Table5: complete local outcome', 'Limits and availability: qualified scope', 'Supplement S28/S29: complete scores, gates and reconstruction; S1-S27 untouched', 'No dynamic-decomposition scientific claim from failed accounting'], 'stable_method_success': False, 'unseen_confirmation_complete': False, 'full_goal_achieved': False}, indent=2) + '\n')
    print(json.dumps({'status': 'integrated_pending_source_claim_and_render_acceptance', 'score_rows': len(scalar_rows), 'comparison_rows': len(comparison_rows)}))


if __name__ == '__main__':
    main()
