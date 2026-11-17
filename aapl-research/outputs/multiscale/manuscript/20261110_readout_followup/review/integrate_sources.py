"""Integrate accepted text/JSON evidence; never load market or model arrays."""
from pathlib import Path
import hashlib
import json
import re

BASE = Path(__file__).resolve().parents[1]
PREVIOUS = BASE.parent / '20261109_integrated_followup'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


main = (PREVIOUS / 'MAIN_PAPER_EN.md').read_text()
abstract = """We compare AAPL nominal-price and cumulative-return forecasts across five temporal scales and twenty tasks, then test specific explanations for differences between implementations. The eight-method, five-fold comparison supplies 400 neural fits. Matched initialization, anchor and training-cap interventions show that a large improvement over a default network can coexist with failure against persistence or zero returns. Baseline-function initialization reduces xPatch minute-H60 and hourly-H8 return error by 9.60% and 44.11% relative to its default head, but the fixed initialized-and-shrunk recipe remains below zero by 0.038% and 0.795%. Three-seed long-scale follow-up reports annual price/return gains of 33.731%/38.105% over persistence/zero; intervals against shrunk unconditional means include zero, and the original nonoverlap sensitivity has only four paths. A genuine four-head shared-versus-independent experiment completes thirty training bundles but fails its separate mechanism and predictive gates. Fixed context and price–volume designs also fail their designated criteria. Source analysis and eight synthetic checks identify a one-coordinate LayerNorm restriction in the xPatch trend branch for four short-horizon tasks, without attributing their market errors to that restriction. For four fixed LSTM states, exact affine-readout projection reduces fit-path MSE by 0.498%–8.246%. A matched temporal intervention then refits all scalers and trains four fresh encoders on purged earlier observations. Exact readout refitting increases later path and endpoint point losses in all three numerically eligible states; one state is excluded under the frozen numerical rule. Its continuation gate fails. The evidence separates structural restrictions, empirical fitting gaps and temporal predictive improvement. No tested recipe meets the designated stable improvement over strong controls. Assessment periods remain exposed development data, and protected 2026Q3 remains unscored."""
start = main.index('## Abstract\n') + len('## Abstract\n\n')
end = main.index('\n\n**Keywords:**', start)
main = main[:start] + abstract + main[end:]

methods = r"""### 3.7 Structural checks and conditional-readout intervention

We inspect native xPatch dimensions and test its trend-input response with fixed synthetic seasonal inputs. Separately, matched price/residual parameter maps test a common target scale in FP64 and FP32; scale-only controls change the target standard deviation while matching the initial physical forecast. These are implementation and numerical checks without market fitting (S22).

For minute H60 and hourly H8 returns, folds 1/2 and seed 2026, we freeze four previously selected LSTM representations. Their 32 hidden coordinates, standardized anchor and intercept give 34 readout columns. Streaming augmented QR and SVD project the original standardized complete-path fit targets; independent original-forward hooks and direct SVD of the original design verify the empirical gap. No projected predictor is retained (S23).

A separate temporal intervention divides each original fit chronologically at floor(0.8n), purges update labels reaching the later segment, and refits every scaler and a fresh LSTM on the earlier update pool. Fixed training budgets are 4/1/1/9 epochs in the task/fold order above. A retains the trained head; B replaces only its readout by unpenalized affine least squares. Controls C and D use an anchor-only readout and the same-seeded never-trained encoder; zero and the update-only mean remain. Only B can advance. All four states must qualify numerically, and one fixed task must beat A/C/D/zero on both path and endpoint loss in both folds beyond the frozen first-order sensitivity estimate. This development filter precedes the unchanged formal five-fold, three-seed success gate. The temporal design and budgets reflect exposed historical development (S24).

"""
main = main.replace('## 4. Results\n', methods + '## 4. Results\n', 1)
results = r"""### 4.8 Structural restriction and failure of fit-gap transfer

For native H2/H3, the second trend-stream LayerNorm receives one coordinate: its output is the learned bias, independent of its trend input. Eight fixed synthetic checks pass. The original H1 adapter calls native H2, so the restriction affects xPatch hourly-H1 and quarterly-H3 price/return tasks. Seasonal and anchor routes remain active; this does not make the whole model constant or explain the primary H60/H8 failures. Common-scale parameter mapping passes all four FP64 cases but only three of four FP32 cases. We retain that numerical failure and make no isolated market-error attribution (S22).

All four fixed LSTM designs qualify at rank 34. Conditional readout fit-error reductions are 1.7205%/0.4984% for minute folds 1/2 and 8.2462%/3.8445% for hourly folds 1/2. These are empirical reductions within a frozen representation; they do not identify an Adam-specific failure or irreducible noise. The 196,455 fit row instances overlap across folds.

Four fresh temporal fits complete 2,556 batch updates. B lowers update loss, but every eligible later-state point estimate increases (Table 4). Hourly fold 1 fails the original stationarity criterion, so its later labels are not scored. The continuation gate fails and this unpenalized readout recipe stops. Conditioning-aware perturbation estimates are not certificates: minute fold 2 and comparisons involving D remain unresolved where estimates are wide. Descriptive block intervals do not establish deterioration in every state. Independent original-design reconstruction matches the eligibility decisions and later metrics, with a largest normalized metric difference of approximately 1.21e-13. S24 reports all controls, intervals and exclusions. This verifies numerical replay, not unseen-market performance.

**Table 4.** Fresh temporal readout intervention. Positive changes mean higher B loss relative to A. Path loss uses the update-derived target scale; endpoint loss uses raw cumulative log returns. The encoders and scalers differ from the preceding fixed-checkpoint diagnostic.

| State | Path change | Endpoint change | Eligibility |
|---|---:|---:|---|
| Minute H60 / 1 | +0.6208% | +0.7551% | Qualified |
| Minute H60 / 2 | +0.3999% | +0.6524% | Qualified |
| Hourly H8 / 1 | Not scored | Not scored | Excluded |
| Hourly H8 / 2 | +24.3626% | +32.2872% | Qualified |

"""
main = main.replace('## 5. Limitations and conclusion\n', results + '## 5. Limitations and conclusion\n', 1)
old = 'The broader objective of an explained, reproducible forecasting improvement remains unachieved, with eligible independent market confirmation still outstanding.'
new = ('The structural and readout follow-ups narrow the explanation further: a short-horizon trend restriction is identifiable, and an empirical affine-head gap can coexist with worse later point losses. '
       'The temporal intervention does not distinguish implicit regularization, unstable representations and finite-sample effects. It supports no penalty, epoch or window rescue, and its controls are not alternative selected candidates. '
       'The broader objective of an explained, reproducible forecasting improvement remains unachieved, with eligible independent market confirmation still outstanding.')
assert old in main
main = main.replace(old, new)
main = main.replace('The integrated documentation adds the partial-probe receipts, complete fixed-volume results and bounded primary-text findings in S19–S21.',
                    'The integrated documentation retains the partial-probe receipts, complete fixed-volume results and bounded primary-text findings in S19–S21. This edition adds the structural/scale checks, fixed-readout fit decomposition and fresh temporal intervention in S22–S24, with small source/result snapshots. Numerical replay of those market diagnostics requires the identified external inputs and checkpoints; the documentation bundle contains neither raw licensed bars nor model arrays.')
(BASE / 'MAIN_PAPER_EN.md').write_text(main)

supp = (PREVIOUS / 'SUPPLEMENT_EN.md').read_text().rstrip('\r\n') + '\n\n'
supp += r"""## S22. Short-horizon trend restriction and target-scale checks

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

"""
fixed_note = (BASE / 'evidence/fixed_readout/MANUSCRIPT_ADDENDUM_EN.md').read_text()
fixed_note = fixed_note[fixed_note.index('For the minute H60'):]
last = fixed_note.index('A prospective development intervention would require')
fixed_note = fixed_note[:last] + ('The next temporal intervention was frozen and completed subsequently, as reported in S24. Its encoders and scalers were trained anew on earlier update pools, so its later labels had not trained those representations. The preceding fixed checkpoints were not retrospectively relabeled as unexposed representations.\n')
supp += fixed_note.rstrip() + '\n\n'
supp += '## S24. Purged temporal test of exact readout refitting\n\n'
temporal_note = (BASE / 'evidence/temporal_readout/MANUSCRIPT_ADDENDUM_EN.md').read_text()
temporal_note = temporal_note[temporal_note.index('For the minute H60'):]
temporal_note = temporal_note.replace(
    'a fixed task to beat A/C/D/zero at both losses in both folds.',
    'a fixed task to beat A/C/D/zero at both losses in both folds beyond the frozen first-order sensitivity estimate.')
supp += temporal_note.rstrip() + '\n\n'
res = json.loads((BASE / 'evidence/temporal_readout/RESULTS.json').read_text())
supp += '### S24.1 Complete later losses for all fixed controls\n\n'
supp += ('Each task/fold retains its own target-standardization scale. These are later fit-internal losses, not original outer losses; cross-state path values are not pooled. The excluded hourly fold 1 supplies no later loss for any arm.\n\n')
supp += '| State | Arm | Path standardized MSE | Endpoint raw MSE |\n|---|---|---:|---:|\n'
labels = {}
for st in res['states']:
    label = ('Minute H60' if st['task']=='minute_h60' else 'Hourly H8') + ' / ' + str(st['fold'])
    labels[st['state_id']] = label
    if not st['tail_scored']:
        supp += f'| {label} | All | Not scored | Not scored |\n'
        continue
    for arm in ['A','B','C','D','zero','mean']:
        metrics=st['tail_metrics'][arm]
        supp += f"| {label} | {arm} | {metrics['path_standardized_MSE']:.12g} | {metrics['endpoint_raw_MSE']:.12g} |\n"
supp += '\n### S24.2 Complete descriptive block intervals\n\n'
supp += ('The following sixty intervals retain the original helper rules, with 5,000 draws and seed 2026. Each row compares B against the stated reference. Positive relative improvement means lower B loss; these signs differ from the deterioration convention in Table 4. Intervals concern exposed development, are unadjusted for multiplicity and exclude numerical uncertainty. No interval changes the failed continuation decision or admits another candidate.\n\n')
for st in res['states']:
    if not st['tail_scored']: continue
    sid=st['state_id']; supp += '#### '+labels[sid]+'\n\n'
    supp += '| Loss | Reference | Block sessions | Gain (%) | 95% interval (%) | Supports gain |\n|---|---|---:|---:|---|---|\n'
    for row in res['intervals']:
        if row['state_id']!=sid:continue
        lo,hi=row['relative_ci95_pct']
        supp += f"| {row['metric']} | {row['reference']} | {row['block']} | {row['relative_improvement_pct']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {'Yes' if row['supports_improvement'] else 'No'} |\n"
    supp += '\n'
supp += r"""### S24.3 Provenance, recovery and remaining objective

The primary and independent implementations agree on four eligibility identities and all three scored states. They use the same admitted data identities, so this is computational reconstruction on the same historical data. Full numerical reproduction still requires the external source bars, partitions, original model code and saved checkpoint generations identified in the evidence receipts. The thin documentation bundle contains text, source and JSON snapshots; it does not contain raw bars, arrays or checkpoints and cannot independently retrain the history by itself.

The first attempted fixed-state replay failed at a NumPy Unicode maximum reduction before weight deserialization or fit-loss computation. Its date-check revision preserved the absolute deadline. The temporal route then required two serialization-only revisions for native Python boolean conversion; the archived original failures remain and numerical thresholds are unchanged. Actual pause, timeout and signal checks preserve training/model/optimizer/initial-clone/random/batch state. Native synthetic cases exercise training recovery and numerical exclusion. A separate exact-source full-rank composition fixture exercises the qualified later-loss and serialization branch; it is not a qualified native-LSTM later-recovery test. No stop/outer score or Q3 value is added by these stages.

The fixed-state gap and fresh temporal intervention answer different questions. The former measures remaining empirical head fitting within an already selected representation; the latter tests one fixed consequence of removing that gap on later historical observations with freshly restricted fitting. Their combined negative transfer result supplies no stable method gain, no Adam-specific mechanism, no general rejection of learned representations and no independent market confirmation. Neither the anchor-only nor random-encoder control is eligible to replace B after inspecting these outcomes. The original title, five scales, twenty tasks and formal strong-control success criteria remain unchanged. The broader positive-method research goal remains active and unachieved. The present results support a paper about the tested forecasts and bounded implementation findings; document completion alone does not satisfy that broader goal.
"""
(BASE / 'SUPPLEMENT_EN.md').write_text(supp)
changes = {
    'previous_edition': str(PREVIOUS),
    'previous_main_sha256':sha(PREVIOUS/'MAIN_PAPER_EN.md'),
    'previous_supplement_sha256':sha(PREVIOUS/'SUPPLEMENT_EN.md'),
    'new_main_sha256':sha(BASE/'MAIN_PAPER_EN.md'),
    'new_supplement_sha256':sha(BASE/'SUPPLEMENT_EN.md'),
    'new_main_sections':['3.7','4.8'],
    'revised_main_sections':['Abstract','5. Limitations and conclusion','Data and code availability'],
    'preserved_supplement_sections':[f'S{i}' for i in range(1,22)],
    'new_supplement_sections':['S22','S23','S24'],
    'old_documents_modified':False,
    'market_arrays_or_weights_loaded':False,
    'new_training_or_scoring':False,
    'research_goal_achieved':False,
}
(BASE/'SOURCE_CHANGELOG.json').write_text(json.dumps(changes,indent=2)+'\n')
print(json.dumps(changes,indent=2))
