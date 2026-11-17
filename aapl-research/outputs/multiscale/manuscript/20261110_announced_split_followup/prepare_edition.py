"""Create an additive split-benchmark manuscript edition from accepted sources.

This script reads saved summaries and document files, not market or model arrays.
It does not train, rescore, or change the accepted preceding edition.
"""
from pathlib import Path
import hashlib
import json
import shutil

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
OLD = BASE.parent / '20261110_bar_location_followup'
STUDY = ROOT / 'outputs/multiscale/research_review/20261110_announced_split_benchmark'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def main():
    assert not (BASE / 'MAIN_PAPER_EN.md').exists(), 'Do not overwrite a prepared edition'
    old_main = (OLD / 'MAIN_PAPER_EN.md').read_text(encoding='utf8')
    old_supp = (OLD / 'SUPPLEMENT_EN.md').read_text(encoding='utf8')
    accepted = read(OLD / 'review/FINAL_DOCUMENT_ACCEPTANCE.json')
    assert accepted['checks_failed'] == 0 and accepted['research_goal_achieved'] is False
    assert sha(OLD / 'MAIN_PAPER_EN.pdf') == accepted['main_pdf_sha256']
    assert sha(OLD / 'SUPPLEMENT_EN.pdf') == accepted['supplement_pdf_sha256']
    result = read(STUDY / 'audit/RESULTS.json')
    replay = read(STUDY / 'audit/INDEPENDENT_REPLAY_SCHEMA_R1.json')
    bridge = read(STUDY / 'V3A_SCALAR_BRIDGE.json')
    assert result['completed_folds'] == 30 and replay['status'] == 'passed_independent_numerical_replay'
    assert bridge['source_audit_result_sha256'] == sha(STUDY / 'audit/RESULTS.json')
    assert bridge['old_reference_and_origin_count_matched_all_six']
    source_rows = {r['task']: r for r in result['task_summaries']}
    v3a = {r['task']: r for r in bridge['rows']}
    assert len(source_rows) == len(v3a) == 6 and set(source_rows) == set(v3a)

    insert_abstract = ('A later post-observation reference incorporating the announced 2020 split '
                       'reduces monthly price and return endpoint MSE by 76.809% and 73.597% relative '
                       'to the original persistence and zero references. ')
    anchor = 'Source analysis and eight synthetic checks identify a one-coordinate LayerNorm restriction'
    assert old_main.count(anchor) == 1
    main_text = old_main.replace(anchor, insert_abstract + anchor, 1)
    section = '''### 4.12 A reference using the announced 2020 split

After earlier inspection of the split event, we fixed a separate reference using Apple's public four-for-one announcement [19] before calculating this comparison. At origins whose decision time is at or after August 1, 2020, 00:00 New York time, and whose session precedes August 31, it predicts current close divided by four, or a cumulative log return of −log(4), for path steps reaching the August 31 effective trading date. Other steps retain the original persistence or zero reference. We keep the nominal targets, saved forecasts, and all original assessment origins (S32).

Among 1,970 monthly H21 origins, 21 overlapping paths cross the event and meet the announcement boundary. The new reference reduces pooled endpoint MSE by 76.809% for price and 73.597% for return compared with the original reference. The initialized-and-shrunk xPatch in Table 1 has relative endpoint changes of −330.938% and −275.931% against the new reference, calculated afterward from saved pooled losses. The earlier tiny long/aligned model is a different predictor and is reported separately in S32. No assessed annual or quarterly path crosses this event. The announcement is additional public information absent from the original learned inputs; these comparisons do not isolate architecture effects or establish independent market performance.

'''
    assert main_text.count('## 5. Limitations and conclusion') == 1
    main_text = main_text.replace('## 5. Limitations and conclusion', section + '## 5. Limitations and conclusion', 1)
    main_text = main_text.rstrip() + ('\n\n[19] Apple Inc. *Form 10-Q for the quarter ended June 27, 2020*, '
        'pp. 18 and 32; SEC accession 0000320193-20-000062, filed July 31, 2020. '
        'https://s2.q4cdn.com/470004039/files/doc_financials/2020/q3/_10-Q-Q3-2020-(As-Filed).pdf\n')

    task_order = ['annual_h12_future_close', 'annual_h12_future_log_return',
                  'quarterly_h3_future_close', 'quarterly_h3_future_log_return',
                  'monthly_h21_future_close', 'monthly_h21_future_log_return']
    names = {'annual_h12_future_close': 'Annual H12 price', 'annual_h12_future_log_return': 'Annual H12 return',
             'quarterly_h3_future_close': 'Quarterly H3 price', 'quarterly_h3_future_log_return': 'Quarterly H3 return',
             'monthly_h21_future_close': 'Monthly H21 price', 'monthly_h21_future_log_return': 'Monthly H21 return'}
    lines = []
    for task in task_order:
        s, b = source_rows[task], v3a[task]
        old = s['old_benchmark']['endpoint_mse']
        new = s['announced_benchmark']['endpoint_mse']
        tiny = s['tiny_long_aligned']
        assert abs(old - b['old_reference_mse']) <= 1e-12 * max(abs(old), 1)
        assert abs(new - b['announced_reference_mse']) <= 1e-12 * max(abs(new), 1)
        lines.append(f"| {names[task]} | {s['origins']} | {s['known_event_origins']} | "
                     f"{100*(1-new/old):+.3f}% | {b['xpatch_gain_pct_vs_old']:+.3f}% | "
                     f"{b['xpatch_gain_pct_vs_announced']:+.3f}% | "
                     f"{tiny['endpoint_mse_gain_pct_vs_announced_benchmark']:+.3f}% |")
    supplement = '''## S32. An additional reference using the announced 2020 split

### S32.1. Information and scope

The original references set the next nominal price to the origin close and cumulative log return to zero. They do not account for future splits that had already been announced. After earlier examination of the historical split event, we fixed a single split-aware reference before this calculation. Apple announced the four-for-one split on July 30, 2020, and stated that split-adjusted trading would begin August 31 [19]. Its contemporaneous as-filed Form 10-Q and SEC submission metadata support a conservative information boundary of August 1, 2020, 00:00 America/New_York. The record date of August 24 is not the trading effective date. These sources support a historical public-availability assumption; they do not establish the actual receipt time on an individual machine or the point-in-time status of vendor bars.

For each forecast origin and each future path step, the adjustment applies if the recorded decision time is at or after that boundary and origin session < August 31 ≤ target session. The price reference is origin close / 4 and the return reference is −log(4) at those steps. Other path entries retain the original reference, including origins for which the split was already effective. We keep the original nominal targets, historical predictions, time partitions and origins. The reference contains additional public announcement information unavailable to the original close-only models, and does not cover dividends or all possible corporate actions.

### S32.2. Results on the original long-scale origins

We reconstruct all five folds of the six annual, quarterly and monthly price/return tasks and score the eight original methods and the separately frozen tiny long/aligned model. The latter has three original seeds; we average losses at each origin before pooling. The fourteen minute/hourly task rows are inherited from saved summaries and date qualifications. Those tasks begin in 2021 and have no future 2020 split, and their arrays were not rescored here. A further post-observation scalar bridge divides the existing pooled mean-seed MSE for the main-paper initialized-and-shrunk xPatch candidate by the new reference MSE. The bridge first verifies the old reference MSE and origin count for all six tasks. It does not perform new inference.

**Table S32.1.** Pooled endpoint MSE improvement is 100 × (1 − predictor MSE / reference MSE). “Reference gain” compares the announced-split reference with the original persistence/zero reference. “xPatch old” and “xPatch new” compare the main-paper Table 1 candidate with the old and new references. “Tiny new” refers to a different earlier long/aligned model. Negative gains indicate higher MSE than the named reference. The xPatch-to-new comparison is a post-observation calculation from saved scalar results.

| Task | Origins | Eligible | Reference gain | xPatch old | xPatch new | Tiny new |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
''' + '\n'.join(lines) + '''

All 21 monthly crossing origins meet the fixed information boundary, with 231 affected forecast-path coordinates per target. They are overlapping outcomes from one event, not independent split replications. The stronger reference reduces monthly price and return endpoint MSE by 76.809319% and 73.596588%, respectively. Neither the original comparison nor the new reference tests what would happen if the learned predictors also used the announcement. No assessed annual or quarterly path crosses the event; their prior point comparisons remain unchanged, including the small, overlapping annual sample. Earlier intervals against the original reference are not intervals against the new reference.

### S32.3. Numerical reconstruction and limits

The calculation used the existing pre-Q3 bars, labels and saved forecasts, with no training or new model inference. A separate implementation reconstructed 30 folds and 330 prediction files for the eight original methods and tiny, checking 37,544 scalar values, including their original pooled losses. The separate xPatch bridge checked saved pooled scalars; that replay did not open xPatch candidate prediction arrays. The first verifier completed the numerical fold reconstruction but failed on an inherited-task metadata wording field. A separate schema adapter corrected that field while preserving the original failed receipt, numerical code, tolerances and absolute deadline; its replay passed. The main process also saved a checkpoint after a deliberate one-fold pause and resumed it in a fresh process under the same deadline. This was a cooperative pause, not a hard-kill or external-signal recovery test. These checks establish numerical reproducibility on already exposed data, not independent market confirmation.

The fixed original minute-H60 and hourly-H8 predictive criteria, shared-versus-independent comparison and 2026Q3 holdout remain separate. The split reference is a descriptive public-information control, not a new learned method, a proof of market predictability or a trading strategy. The source qualification and all result receipts are listed in the separate edition's evidence map.
'''
    for name in ('code', 'figures'):
        shutil.copytree(OLD / name, BASE / name, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(OLD / 'AUTHOR_INFORMATION.json', BASE / 'AUTHOR_INFORMATION.json')
    build = (OLD / 'build_integrated_documents.py').read_text(encoding='utf8')
    build = build.replace('PREVIOUS = BASE.parent / "20261110_artificial_mechanism_followup"',
                          'PREVIOUS = BASE.parent / "20261110_bar_location_followup"')
    (BASE / 'build_integrated_documents.py').write_text(build, encoding='utf8')
    (BASE / 'MAIN_PAPER_EN.md').write_text(main_text, encoding='utf8')
    (BASE / 'SUPPLEMENT_EN.md').write_text(old_supp.rstrip() + '\n\n' + supplement, encoding='utf8')
    assert (BASE / 'SUPPLEMENT_EN.md').read_text().startswith(old_supp.rstrip() + '\n\n')
    evidence = BASE / 'evidence/split_benchmark'
    evidence.mkdir(parents=True)
    selected = ['PLAN_ZH.md', 'SOURCE_QUALIFICATION.json', 'SOURCE_REVIEW.md', 'PROTOCOL.json',
                'SYNTHETIC_ACCEPTANCE.json', 'LIFECYCLE_ACCEPTANCE.json', 'SCHEMA_ADAPTER_PROTOCOL.json',
                'V3A_SCALAR_BRIDGE.json', 'MANUSCRIPT_ADDENDUM_EN.md',
                'audit/RESULTS.json', 'audit/INDEPENDENT_REPLAY.json', 'audit/INDEPENDENT_REPLAY_SCHEMA_R1.json',
                'sources/apple_q3_2020_as_filed.receipt.json', 'sources/SEC_matching_2020_filings.receipt.json']
    entries = []
    for raw in selected:
        src, dest = STUDY / raw, evidence / raw.replace('/', '__')
        shutil.copy2(src, dest)
        entries.append({'original_path': str(src), 'snapshot': str(dest.relative_to(BASE)),
                        'sha256': sha(src), 'bytes': src.stat().st_size})
        assert sha(src) == sha(dest)
    manifest = {'previous_accepted_edition': str(OLD), 'previous_main_sha256': sha(OLD/'MAIN_PAPER_EN.md'),
                'previous_supplement_sha256': sha(OLD/'SUPPLEMENT_EN.md'),
                'previous_evidence_map_sha256': sha(OLD/'EVIDENCE_MAP.json'), 'new_snapshots': entries,
                'inherited_bar_location_evidence': {
                    'external_map': str(OLD/'EVIDENCE_MAP.json'),
                    'sha256': sha(OLD/'EVIDENCE_MAP.json'),
                    'map_contents': read(OLD/'EVIDENCE_MAP.json'),
                    'copy_mode': 'external; snapshot paths resolve against previous edition'},
                'market_arrays_weights_included': False, 'full_numeric_replay_performed_by_document_build': False,
                'independent_market_confirmation': False, 'research_goal_achieved': False}
    (BASE / 'EVIDENCE_MAP.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    guide = '''# Manuscript and numerical reproduction boundaries

This edition preserves the title, author block and twenty-task scope of the previous accepted edition, and appends a post-observation announced-split reference in section 4.12 and S32. The author supplies the email. The original results remain available; the new reference uses public announcement information not supplied to the close-only learned models. The xPatch scalar bridge is post-observation. No independent confirmation or improved learned method is claimed.

EVIDENCE_MAP.json identifies the previous accepted edition and the copied small protocol, source and result snapshots. The original calculation, full source files, market inputs and retained failure receipt remain at outputs/multiscale/research_review/20261110_announced_split_benchmark. The previous edition's EVIDENCE_MAP.json and REPRODUCTION_GUIDE_EN.md describe S1–S31 and earlier dependencies. The original absolute deadline is historical and is not reset by document building. The small snapshots are not a portable training or market-data archive.

Rebuild the documents from the project root with `/home/ayaka/.venvs/doc-tools/bin/python outputs/multiscale/manuscript/20261110_announced_split_followup/build_integrated_documents.py`. This requires Pandoc, python-docx, LibreOffice, Poppler and the recorded font environment. The build neither opens market data nor reruns experiments. New builds need independent claim and visual review. No external submission, publication, DOI or new ZIP is asserted.
'''
    (BASE/'REPRODUCTION_GUIDE_EN.md').write_text(guide, encoding='utf8')
    (BASE/'SOURCE_CHANGELOG.json').write_text(json.dumps({
        'previous_edition': str(OLD), 'old_main': sha(OLD/'MAIN_PAPER_EN.md'),
        'old_supplement': sha(OLD/'SUPPLEMENT_EN.md'), 'supplement_old_prefix_preserved': True,
        'main_changes': ['one abstract sentence', 'new section 4.12', 'new reference 19'],
        'new_main': sha(BASE/'MAIN_PAPER_EN.md'), 'new_supplement': sha(BASE/'SUPPLEMENT_EN.md'),
        'title_author_scope_preserved': True, 'Q3_read': False, 'research_goal_achieved': False}, indent=2)+'\n')
    print(json.dumps({'prepared': str(BASE), 'small_evidence_snapshots': len(entries),
                      'old_supplement_prefix_preserved': True, 'documents_built': False}))


if __name__ == '__main__':
    main()
