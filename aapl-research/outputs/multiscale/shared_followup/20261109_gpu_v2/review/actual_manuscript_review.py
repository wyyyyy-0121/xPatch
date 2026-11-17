"""Independent actual-manuscript JSON/text/geometry review; never loads arrays.

Run only after the approved pipeline renders the complete market manuscript.
This file does not import training, evaluation or manuscript builder code.
"""
from pathlib import Path
import hashlib
import json
import math
import os
import re
import subprocess
import time
import xml.etree.ElementTree as ET

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[3]
PRECEDING = REPO/'outputs/multiscale/mechanism_repair/20261109_v3b/manuscript'
PAPER = BASE/'manuscript/new_version'
DEADLINE = 1794216709.0
HORIZONS = (5, 15, 30, 60)
SEEDS = (2026, 2027, 2028)
TITLE = 'Multi-Horizon Forecasting of Apple Stock Prices and Returns Across Temporal Scales'


def guard():
    if time.time() >= DEADLINE:
        raise InterruptedError('Authorized boundary reached; actual review not run')


def read(path):
    guard()
    return json.loads(Path(path).read_text())


def sha(path):
    guard()
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            guard(); digest.update(block)
    return digest.hexdigest()


def gate():
    """Require complete receipts before parsing any new statistics or prose."""
    guard()
    rules = read(BASE/'protocol/RULES.json')
    assert rules['admitted'] is True and rules['fixed_epochs'] == 20
    jobs = rules['jobs']
    assert len(jobs) == len({j['id'] for j in jobs}) == 30
    assert {(j['arm'], j['fold'], j['seed']) for j in jobs} == {
        (a, f, s) for a in ('tied', 'untied') for f in range(1, 6) for s in SEEDS}
    paths = [BASE/'runs'/j['id']/'DONE.json' for j in jobs]
    paths += [BASE/'audit/COMPLETE.json', BASE/'analysis/ANALYSIS_COMPLETE.json',
              BASE/'review/INDEPENDENT_STATISTICS_REVIEW.json', PAPER/'DOCUMENT_BUILD.json']
    assert all(p.is_file() for p in paths), 'Await full completion and formal render'
    rule_hash = sha(BASE/'protocol/RULES.json')
    for job in jobs:
        done = read(BASE/'runs'/job['id']/'DONE.json')
        assert done['status'] == 'completed' and done['job'] == job
        assert done['rules_sha256'] == rule_hash and done['fixed_epochs'] == 20
    complete = read(BASE/'audit/COMPLETE.json')
    assert complete['status'] == 'completed' and complete['bundles'] == 30
    analysis = read(BASE/'analysis/ANALYSIS_COMPLETE.json')
    assert analysis['status'] == 'completed' and analysis['complete_bundles'] == 30
    assert analysis['market_training_complete'] is True and analysis['all_five_folds_three_seeds'] is True
    review = read(BASE/'review/INDEPENDENT_STATISTICS_REVIEW.json')
    assert review['status'] == 'passed'
    for field, relative in (
        ('rules_sha256', 'protocol/RULES.json'),
        ('analysis_complete_sha256', 'analysis/ANALYSIS_COMPLETE.json'),
        ('evaluation_freeze_sha256', 'protocol/EVALUATION_FREEZE.json'),
        ('script_sha256', 'review/independent_statistics.py')):
        assert review[field] == sha(BASE/relative), field
    for receipt in (analysis, review):
        assert receipt['Q3_values_read'] is False and receipt['independent_confirmation'] is False
        assert receipt['research_goal_achieved'] is False
    assert {(r['horizon'], r['fold']) for r in review['fold_checks']} == {
        (h, f) for h in HORIZONS for f in range(1, 6)}
    assert len(review['fold_checks']) == 20 and all(r['status'] == 'passed' for r in review['fold_checks'])
    build = read(PAPER/'DOCUMENT_BUILD.json')
    assert build['status'] == 'rendered_requires_visual_review'
    for relative, expected in analysis['files'].items():
        path = (BASE/relative).resolve()
        assert path.is_relative_to(BASE.resolve())
        assert sha(path) == expected
    return rules, analysis, review, build


def compare(a, b, path='root'):
    if isinstance(a, dict):
        # The evaluator additionally labels interval scope; independently
        # reconstructed numerical fields must all agree without requiring
        # the reviewer to copy that evaluator-only prose field.
        assert set(b).issubset(a), path
        for k in b: compare(a[k], b[k], path+'/'+k)
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)): compare(x, y, path+'/'+str(i))
    elif isinstance(a, float):
        assert math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=2e-9, abs_tol=2e-14), path
    else:
        assert a == b, (path, a, b)


def tables(text):
    output = []; current = []
    for line in text.splitlines()+['']:
        if line.startswith('|') and line.endswith('|'):
            current.append([x.strip() for x in line.strip('|').split('|')])
        elif current:
            assert len(current) >= 2 and all(set(x) <= {'-', ':'} for x in current[1])
            output.append((current[0], current[2:])); current = []
    return output


def pct(value):
    return 'undefined' if value is None else f'{value:+.3f}%'


def bounds(row):
    value = row.get('relative_ci95_pct')
    return row['status'] if value is None else f'[{value[0]:+.3f}, {value[1]:+.3f}]%'


def numbers(stats, main, supplement):
    candidate = 'tied_selected_shrink'
    primary = stats['tasks']['minute_h60_future_log_return']
    mechanism = primary['mechanism']['passed']
    predictive = primary['candidate_performance_gate_passed']
    main_section = main.split('**Table 3.**', 1)[1].split('## 5.', 1)[0]
    _, actual_rows = tables(main_section)[0]
    expected = []
    for h in HORIZONS:
        task = stats['tasks'][f'minute_h{h}_future_log_return']
        expected.append([f'H{h}', pct(task['summary'][candidate]['improvement_pct_over_zero']),
            f"{task['summary'][candidate]['positive_folds_over_zero']}/5",
            'pass' if task['mechanism']['passed'] else 'fail',
            'pass' if task['candidate_performance_gate_passed'] else 'fail'])
    assert actual_rows == expected
    s16 = supplement.split('## S16.', 1)[1].split('## S17.', 1)[0]
    mechanism_table, candidate_table = tables(s16)
    expected_m = []; expected_c = []
    for h in HORIZONS:
        task = stats['tasks'][f'minute_h{h}_future_log_return']
        intervals = [next(r for r in task['intervals'] if r['primary'] == 'tied_last'
            and r['reference'] == 'untied_last' and r['block'] == block) for block in (1, 5)]
        m = task['mechanism']; c = task['candidate_checks']
        expected_m.append([f'H{h}', pct(intervals[0]['relative_improvement_pct']),
            str(sum(m['seed_positive'].values())), f"{m['positive_folds']}/5", bounds(intervals[0]), bounds(intervals[1]),
            'pass' if m['passed'] else 'fail'])
        expected_c.append([f'H{h}', pct(task['summary'][candidate]['improvement_pct_over_zero']),
            str(c['three_positive_seeds_over_zero']), str(c['four_positive_folds_over_zero']),
            str(c['minimum_one_percent_over_zero']), str(c['both_block_intervals_against_every_control']),
            'pass' if task['candidate_performance_gate_passed'] else 'fail'])
    assert mechanism_table[1] == expected_m and candidate_table[1] == expected_c
    summary_count = interval_count = 0
    for index, h in enumerate(HORIZONS, 1):
        section = supplement.split(f'### S17.{index} ', 1)[1]
        section = section.split(f'### S17.{index+1} ', 1)[0] if index < 4 else section.split('## S18.', 1)[0]
        summary_table, interval_table = tables(section)
        task = stats['tasks'][f'minute_h{h}_future_log_return']
        expected_s = [[name.replace('_', ' '), str(row['origins']), f"{row['endpoint_mse']:.9g}",
            pct(row['improvement_pct_over_zero']), f"{row['positive_folds_over_zero']}/5"]
            for name, row in task['summary'].items()]
        expected_i = [[row['primary'].replace('_', ' '), row['reference'].replace('_', ' '), str(row['block']),
            pct(row.get('relative_improvement_pct')), bounds(row), str(row['supports_improvement'])]
            for row in task['intervals']]
        assert summary_table[1] == expected_s and interval_table[1] == expected_i
        assert len(expected_s) == (76 if h == 60 else 43)
        assert len(expected_i) == (82 if h == 60 else 16)
        summary_count += len(expected_s); interval_count += len(expected_i)
    for text in (main, supplement):
        assert f'sharing contrast {"passes" if mechanism else "fails"} its predefined mechanism gate.' in text
        assert f'candidate {"passes" if predictive else "fails"} the predefined minute-H60 development performance gate.' in text
    gain = primary['summary'][candidate]['improvement_pct_over_zero']
    mgain = 100*(1-primary['summary']['tied_last']['endpoint_mse']/primary['summary']['untied_last']['endpoint_mse'])
    assert f'raw final tied versus untied gains {pct(mgain)}' in main
    assert f'candidate gains {pct(gain)} against zero' in main
    return {'main_table3_rows_verified': 4, 'supplement_mechanism_rows_verified': 4,
        'supplement_candidate_rows_verified': 4, 'supplement_summary_rows_verified': summary_count,
        'supplement_interval_rows_verified': interval_count, 'primary_mechanism_gate_passed': mechanism,
        'primary_candidate_performance_gate_passed': predictive, 'primary_candidate_gain_pct': gain,
        'raw_final_tied_vs_untied_gain_pct': mgain}


def geometry(build):
    output = []; ns = {'x': 'http://www.w3.org/1999/xhtml'}
    for doc in build['documents']:
        guard(); name = doc['name']; pdf = PAPER/f'{name}_EN.pdf'
        assert sha(pdf) == doc['pdf_sha256']
        assert sha(PAPER/f'{name}_EN.md') == doc['source_sha256']
        assert sha(PAPER/f'{name}_EN.docx') == doc['docx_sha256']
        previews = Path(doc['preview_directory']); assert previews.resolve().is_relative_to(PAPER.resolve())
        assert previews.name == doc['pdf_sha256'][:16]
        manifest_path = previews/'PREVIEW_SOURCE_MANIFEST.json'
        assert sha(manifest_path) == doc['preview_manifest_sha256']
        manifest = read(manifest_path)
        assert manifest['pdf_sha256'] == sha(pdf) and Path(manifest['pdf']) == pdf
        pngs = sorted(previews.glob('page-*.png'))
        assert len(pngs) == len(manifest['images']) == doc['pages']
        assert {p.name: sha(p) for p in pngs} == manifest['images']
        pages = ET.fromstring(subprocess.check_output(['pdftotext', '-bbox', str(pdf), '-'])).findall('.//x:page', ns)
        assert len(pages) == doc['pages']
        all_text = subprocess.check_output(['pdftotext', '-layout', str(pdf), '-'], text=True)
        assert 'SYNTHETIC RENDER ACCEPTANCE' not in all_text and 'SYNTHETIC — NOT MARKET RESULTS' not in all_text
        checks = []
        for index, page in enumerate(pages, 1):
            guard(); words = page.findall('x:word', ns)
            assert words
            assert all(0 <= float(w.attrib['xMin']) < float(w.attrib['xMax']) <= float(page.attrib['width'])
                and 0 <= float(w.attrib['yMin']) < float(w.attrib['yMax']) <= float(page.attrib['height']) for w in words)
            footer = [w for w in words if float(w.attrib['yMin']) >= 800]
            body = [w for w in words if w not in footer]
            gap = min(float(w.attrib['yMin']) for w in footer)-max(float(w.attrib['yMax']) for w in body) if footer else None
            assert gap is None or gap > 0
            checks.append({'page': index, 'all_text_boxes_within_page': True, 'body_footer_gap_pt': gap})
        assert doc['preservation']['passed'] is True
        assert doc['preservation']['columns'] == (2 if name == 'MAIN_PAPER' else 1)
        assert doc['times_embedded'] is True
        output.append({'name': name, 'pages': len(pages), 'pdf_sha256': sha(pdf),
            'preview_manifest_sha256': sha(manifest_path), 'preview_directory': str(previews),
            'all_preview_hashes_verified': True, 'geometry_pages': checks})
    return output


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    assert set(os.sched_getaffinity(0)).issubset({24, 26})
    rules, analysis, review, build = gate()
    stats = read(BASE/'analysis/STATS.json')
    assert analysis['statistics_sha256'] == sha(BASE/'analysis/STATS.json')
    assert stats['research_goal_achieved'] is False and stats['independent_confirmation'] is False
    for h in HORIZONS:
        task = f'minute_h{h}_future_log_return'
        compare(stats['tasks'][task], review['tasks'][task], task)
    snapshot = read(PAPER/'RESULT_SOURCE_SNAPSHOT.json')
    for key, relative in (('rules_sha256', 'protocol/RULES.json'), ('analysis_complete_sha256', 'analysis/ANALYSIS_COMPLETE.json'),
        ('statistics_sha256', 'analysis/STATS.json'), ('results_sha256', 'analysis/RESULTS.json'),
        ('evaluation_freeze_sha256', 'protocol/EVALUATION_FREEZE.json'),
        ('independent_statistics_review_sha256', 'review/INDEPENDENT_STATISTICS_REVIEW.json')):
        assert snapshot['provenance'][key] == sha(BASE/relative), key
    assert snapshot['manuscript_builder_sha256'] == sha(BASE/'code/build_manuscript.py')
    assert snapshot['reviewed_methods_sha256'] == sha(BASE/'manuscript/SHARED_STUDY_METHODS_EN.md')
    assert snapshot['original_title_preserved'] is True and snapshot['original_twenty_task_table_preserved'] is True
    assert snapshot['research_goal_achieved'] is False and snapshot['independent_confirmation'] is False
    main_text = (PAPER/'MAIN_PAPER_EN.md').read_text(); supplement = (PAPER/'SUPPLEMENT_EN.md').read_text()
    assert main_text.startswith('# '+TITLE+'\n')
    for value in ('Wang Yang', 'High School Affiliated to Shanghai Jiao Tong University',
                  'Shanghai 200439, China', '[Email to be completed by the author]'):
        assert value in main_text
    for name, expected in snapshot['preceding_sources'].items(): assert sha(PRECEDING/name) == expected
    old = (PRECEDING/'MAIN_PAPER_EN.md').read_text()
    assert main_text.split('**Table 1.**', 1)[1].split('**Table 2.**', 1)[0] == old.split('**Table 1.**', 1)[1].split('**Table 2.**', 1)[0]
    assert read(PAPER/'AUTHOR_INFORMATION.json') == read(PRECEDING/'AUTHOR_INFORMATION.json')
    for phrase in ('H60 is primary; other horizons are supporting', 'Mechanism and candidate decisions are separate',
                   'The original hourly-H8 criterion is not retested', 'protected 2026Q3 remains sealed',
                   'All assessment periods had been inspected before this phase', 'forty control entries including known duplicates',
                   'The sole candidate', '472,549 versus 524,797', 'Deterministic local-GPU FP32'):
        assert phrase in main_text, phrase
    for phrase in ('The mechanism and predictive decisions are reported separately',
                   'Forty is the registered entry count, not the number of distinct methods',
                   'No result in this minute-only study completes the original hourly-H8 objective',
                   'full inner-stop re-inference is not claimed',
                   'The overall mechanism-and-improvement research goal is not marked achieved'):
        assert phrase in supplement, phrase
    numerical = numbers(stats, main_text, supplement)
    documents = geometry(build)
    result = {'status': 'numeric_claims_and_geometry_checked_visual_review_pending',
        'scope': 'complete exposed-development manuscript; not independent market confirmation',
        'all_30_done_analysis_and_independent_review_gated_before_scores': True,
        'title_author_original_twenty_task_table_preserved': True, 'numerical_checks': numerical,
        'documents': documents, 'visual_pages_reviewed': {}, 'script_sha256': sha(__file__),
        'builder_sha256': sha(BASE/'code/build_manuscript.py'), 'source_snapshot_sha256': sha(PAPER/'RESULT_SOURCE_SNAPSHOT.json'),
        'statistics_sha256': sha(BASE/'analysis/STATS.json'),
        'independent_statistics_review_sha256': sha(BASE/'review/INDEPENDENT_STATISTICS_REVIEW.json'),
        'market_raw_arrays_loaded': False, 'GPU': False, 'Q3_values_read': False,
        'independent_confirmation': False, 'research_goal_achieved': False, 'affinity': sorted(os.sched_getaffinity(0))}
    guard()
    output = BASE/'review/ACTUAL_MANUSCRIPT_CLAIM_LAYOUT_REVIEW.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': result['status'], 'documents': [{k: d[k] for k in ('name', 'pages', 'preview_directory')} for d in documents], 'numerical_checks': numerical}), flush=True)


if __name__ == '__main__': main()
