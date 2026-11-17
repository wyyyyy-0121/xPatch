"""Carry verified results into a draft with current permission and restore evidence."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re
import shutil


ROOT = Path('/home/ayaka/wslcode/aapl-research')
OUT = ROOT / 'outputs/multiscale'
PARENT = OUT / 'private_working_manuscripts/20261107T173744618637Z'
PERMISSION = OUT / 'permission_confirmations/20261108_ALPACA_KIBOT_USER_CONFIRMED.json'
PROVENANCE = OUT / 'data_provenance/20261108_CURRENT_DEVELOPMENT_SOURCE_MANIFEST.json'
RESTORE = OUT / 'private_model_restoration/20261108_pre0330_same_batches/RESTORATION_AUDIT.json'
MINUTE = OUT / 'private_minute_recomputation/20261108_pre0330_all_completed.json'
FIGURES = OUT / 'private_result_figures/20261108_pre0330_v2'
COVERAGE = OUT / 'private_coverage_verified/20261108_pre0330_snapshot.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('Expected one revision fragment: ' + before[:100])
    return text.replace(before, after, 1)


def section(text, start, end):
    return text.split(start, 1)[1].split(end, 1)[0]


def main():
    permission, source, restore, minute, coverage = map(load, (PERMISSION, PROVENANCE, RESTORE, MINUTE, COVERAGE))
    if (permission['status'] != 'permitted_as_confirmed_by_user' or
            permission['providers'] != ['Alpaca', 'Kibot'] or
            restore['status'] != 'completed_task_saved_neural_restoration_verified' or
            restore['model_restorations_checked'] != 240 or
            restore['origin_model_paths_checked'] != 179288 or
            restore['all_saved_predictions_exactly_reproduced'] is not True or
            minute['completed_folds_recomputed'] != 8 or coverage['completed_folds'] != 68 or
            coverage['five_fold_tasks'] != 12):
        raise ValueError('Permission, restoration or coverage evidence differs')
    if source['permission_confirmation_sha256'] != digest(PERMISSION):
        raise ValueError('Provenance is not bound to the current permission record')
    for item in restore['results']:
        if digest(item['check']) != item['check_sha256']:
            raise ValueError('Restoration child receipt changed')
    before = (PARENT / 'PRIVATE_WORKING_MANUSCRIPT_EN.md').read_text(encoding='utf8')
    text = before
    text = replace_once(text,
        '**PRIVATE WORKING MANUSCRIPT — NOT FOR SUBMISSION OR DISTRIBUTION.** This version combines previously exposed, local exploratory experiments with the working methods draft. Account-specific data and publication rights, a final protocol, and independent confirmation remain unresolved. Bracketed items identify work that has not been completed.',
        '**WORKING MANUSCRIPT: EXPERIMENTS IN PROGRESS.** This revision reports exposed historical development experiments. The researcher has confirmed permission for Alpaca and Kibot data. The minute matrix, final protocol documentation and manuscript review remain incomplete. Bracketed items identify unfinished work.')
    text = replace_once(text,
        'These previously exposed development results do not establish independent confirmation or trading profitability; source publication rights and the remaining experiments are unresolved.',
        'These previously exposed development results do not establish independent confirmation or trading profitability. The remaining minute experiments are incomplete.')
    text = replace_once(text,
        'This working snapshot reports the completed development comparisons without claiming independent confirmation. **[State the final contribution only after the remaining work and literature review are complete.]**',
        'This working snapshot provides matched-origin comparisons, documents the effect of target definitions and split events, and checks saved-model reproducibility. The reported findings concern the completed tasks and stated training recipes; the remaining minute tasks have yet to establish a complete five-scale comparison.')
    text = replace_once(text,
        '**[Complete a targeted literature search and state the final contribution only after the direct comparisons and any shared-horizon experiment have been completed.]**',
        'We therefore evaluate close-derived path regression under a common set of origin and label rules. The study distinguishes its information set and target definitions from these precedents, and treats architectural comparisons as empirical results under a finite compute budget.')
    text = replace_once(text,
        'Longer public candidates have been investigated, but their upstream provenance, adjustment rules, and applicable publication rights are not yet fully established. The source review is documented separately. **[Insert the admitted provider, version, licence, exact periods, coverage, missing observations, source hashes, and retrieval dates. Do not name a candidate as the study dataset before admission.]**',
        'The reported experiments use Alpaca SIP/raw observations. The daily-source retrieval receipt is dated November 7, 2026 at 04:34 UTC; the prior minute-source batch is identified as alpaca_development_20261102, with its prepared-array manifest dated November 5. The versioned source manifest accompanying this draft records the full SHA-256 digests of both raw sources, the daily receipt, the minute manifest, the four derived bar files and the current permission confirmation. Raw-source identity and the regular-session analysis grid are recorded separately.\n\n'
        'On November 8, 2026, the researcher confirmed that Alpaca and Kibot permit use of all their data in the context of research and publication. We record this as researcher-confirmed permission; we have not independently inspected a supplier contract. Kibot supplies none of the results reported here. Its earlier price-unit discrepancies remain in the separate source-quality record, and no alternative series was substituted into the current experiments.')
    text = replace_once(text,
        'The following inventory describes the Alpaca SIP/raw candidate used in the private development snapshot. It is a date-coverage and processing record, not a publication-admission decision, a guarantee of historical point-in-time availability, or a substitute for checking the vendor’s close convention.',
        'The following inventory describes the Alpaca SIP/raw source used in this development snapshot. The coverage and processing checks do not establish historical point-in-time availability. We retain the vendor-close definition and the conservative completion convention throughout the recorded folds.')
    text = replace_once(text,
        '| Minute candidate | 1,061,536 source rows recorded in the candidate inventory. |',
        '| Minute source and regular grid | 1,061,536 raw source rows; 535,620 of 535,620 expected regular-session minutes observed, from 2021-01-04 to 2026-06-30. |')
    text = replace_once(text,
        '| Rights and close definition | Account-specific publication and redistribution rights remain unverified. A minute-derived session close is not necessarily the official closing-auction price. |',
        '| Permission and close definition | Researcher-confirmed permission for Alpaca and Kibot data, recorded on 2026-11-08. A minute-derived session close is not necessarily the official closing-auction price. |')
    text = replace_once(text,
        '| Daily cutoff | 2,636 dates usable before cutoff; 2026-06-30 was excluded under the conservative next-midnight completion rule. |',
        '| Daily and monthly cutoff | 2,636 daily dates and 125 complete months usable before cutoff; the 2026-06-30 daily slot and June 2026 monthly slot are unobserved under the conservative completion rule. |')
    text = replace_once(text,
        'This comparison can establish a difference between the stated recipes under the measured budget; it cannot isolate a universal causal effect of parameter sharing. **[Report the comparison only if all planned paired runs actually finish.]**',
        'A completed comparison could establish a difference between the stated recipes under the measured budget; it could not isolate a universal causal effect of parameter sharing. This comparison has not been run and contributes no empirical result to the present snapshot.')
    text = replace_once(text,
        'The interface has been exercised with fictional prices across the planned task shapes, including restoration in a separate process. These engineering checks do not establish reproducibility on an admitted market dataset, confer rights to publish trained parameters, or provide forecasting results. **[Document the final source, environment, model and independent restoration receipts after the formal work is completed.]**',
        'The interface has been exercised with fictional prices across the planned task shapes, including restoration in a separate process. A separate historical audit now covers every validation origin in all 60 folds of the 12 completed tasks. It restores 240 saved neural predictors in 60 fresh CPU processes, checks the recorded source, model and fold identities, and repeats the original prediction batches. Each input window ends at its own forecast origin; the audit does not open saved target arrays. All 179,288 origin-model prediction paths reproduce the saved predictions exactly, with maximum absolute difference zero. The audit uses Python 3.13.13, NumPy 2.5.4 and PyTorch 2.14.1+cu130 on two logical CPU cores, with GPU visibility disabled, and takes approximately 107 seconds. This verifies the saved predictors for the completed tasks; it does not complete the twenty-task inference package or test recovery from an interrupted training step.')
    marker = '### Realized training budget\n'
    if text.count(marker) != 1:
        raise ValueError('Training-budget section differs')
    figure = ('![Figure 2. Endpoint MSE ratios for the twelve completed tasks. Cell labels show each model’s pooled endpoint MSE divided by the matched persistence or zero-return MSE; values below one favor the model. Both panels use the same logarithmic color scale centered at a ratio of one. Labels are rounded; Appendix A reports the underlying errors. The annual rows use the short-history variant, and the incomplete minute tasks are excluded.](figures/completed_task_error_ratios.png){width=98%}\n\n')
    text = text.replace(marker, figure + marker, 1)
    text = replace_once(text,
        'Company actions, uncertain historical data revisions, source permissions, and prior exposure constrain what can be concluded.',
        'Company actions, uncertain historical data revisions, limited history and prior exposure constrain what can be concluded.')
    text = replace_once(text,
        'The proposed evaluation code is maintained in a version-controlled research repository, and its execution records identify the source files used. Access to the eventual data and fitted artifacts depends on the applicable provider agreements. **[Insert the archived code release, exact data version, access procedure, licence, and any permitted artifact hashes after data admission and experiments.]**',
        'The research repository retains the task definitions, source-file digests, saved model and prediction identities, chronological split records, and independent arithmetic checks. The accompanying development-source manifest identifies the exact Alpaca daily and minute versions and derived bar files. The permission record contains the researcher’s confirmation for Alpaca and Kibot. The restoration audit records all 240 completed-task model checks, while the minute recomputation record identifies the eight completed minute folds. The code archive and final data-access instructions will accompany the completed experimental release; no archive DOI has been assigned to this working snapshot.')
    text = replace_once(text,
        'Overlapping targets, splits, prior exposure, and unverified publication rights constrain interpretation.',
        'Overlapping targets, splits, limited history and prior exposure constrain interpretation.')

    # Numerical evidence is carried unchanged. The sole inserted result figure
    # derives from the same 96 pooled model rows, rather than a new selection.
    if section(text, '### Cumulative log-return target', '### Realized training budget').replace(figure, '') != section(
            before, '### Cumulative log-return target', '### Realized training budget'):
        raise ValueError('Primary results text or tables changed')
    if section(text, '### Realized training budget', '## 6. Limitations and conclusion') != section(
            before, '### Realized training budget', '## 6. Limitations and conclusion'):
        raise ValueError('Training, minute or sensitivity results changed')
    if section(text, '## References\n', '## Appendix A.') != section(before, '## References\n', '## Appendix A.'):
        raise ValueError('Bibliography changed')
    old_math = re.findall(r'\\\[.*?\\\]|\\\(.*?\\\)', before, re.DOTALL)
    new_math = re.findall(r'\\\[.*?\\\]|\\\(.*?\\\)', text, re.DOTALL)
    if old_math != new_math:
        raise ValueError('An equation changed')

    directory = OUT / 'private_working_manuscripts' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory.mkdir()
    md = directory / 'PRIVATE_WORKING_MANUSCRIPT_EN.md'
    md.write_text(text, encoding='utf8')
    csv_hashes = {}
    for path in PARENT.glob('*.csv'):
        shutil.copy2(path, directory / path.name)
        csv_hashes[path.name] = digest(path)
    for path, name in ((PERMISSION, 'DATA_PERMISSION_CONFIRMATION.json'),
                       (PROVENANCE, 'DATA_PROVENANCE_MANIFEST.json'),
                       (MINUTE, 'MINUTE_RECOMPUTATION.json')):
        shutil.copy2(path, directory / name)
    shutil.copytree(FIGURES, directory / 'figures')
    (directory / 'paper_artifacts').mkdir()
    shutil.copy2(OUT / 'paper_artifacts/chronological_split_schematic.png',
                 directory / 'paper_artifacts/chronological_split_schematic.png')
    audit = {'status': 'permission_provenance_and_real_restoration_added_to_working_manuscript',
             'created_at_utc': datetime.now(timezone.utc).isoformat(),
             'parent_directory': str(PARENT), 'parent_manuscript_sha256': digest(PARENT / md.name),
             'manuscript_sha256': digest(md), 'script_sha256': digest(Path(__file__)),
             'permission_sha256': digest(PERMISSION), 'source_manifest_sha256': digest(PROVENANCE),
             'coverage_sha256': digest(COVERAGE), 'restoration_audit': str(RESTORE),
             'restoration_audit_sha256': digest(RESTORE), 'minute_recomputation_sha256': digest(MINUTE),
             'figure_audit_sha256': digest(FIGURES / 'FIGURE_AUDIT.json'), 'csv_sha256': csv_hashes,
             'checks': {'primary_result_tables_unchanged': True, 'training_and_sensitivity_results_unchanged': True,
                        'equations_unchanged': True, 'bibliography_unchanged': True,
                        'complete_tasks': 12, 'completed_task_folds': 60, 'restored_neural_models': 240,
                        'exact_reproduced_prediction_paths': 179288, 'minute_folds_recomputed': 8,
                        'source_permission_current_user_confirmation': True},
             'independent_test': False, 'Q3_values_read': False,
             'limitations': ['The remaining minute matrix and final manuscript review are incomplete.',
                             'Permission is recorded from the researcher confirmation, not independent contract inspection.',
                             'The saved-predictor audit is separate from the final twenty-task deployment package.']}
    (directory / 'MANUSCRIPT_UPDATE_AUDIT.json').write_text(json.dumps(audit, indent=2) + '\n')
    (OUT / 'LATEST_PRIVATE_WORKING_MANUSCRIPT.txt').write_text(str(directory) + '\n')
    print(json.dumps({'status': audit['status'], 'directory': str(directory), 'manuscript_sha256': digest(md)}))


if __name__ == '__main__':
    main()
