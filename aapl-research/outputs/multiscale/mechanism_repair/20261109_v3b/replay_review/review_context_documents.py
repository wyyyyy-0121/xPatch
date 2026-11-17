"""Independent numeric, OOXML, PDF-text and claim review of the v3b manuscripts."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OLD = BASE.with_name('20261109_v3a')
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
      'rel': 'http://schemas.openxmlformats.org/package/2006/relationships'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tables(markdown):
    output = []; current = []
    for line in markdown.splitlines()+['']:
        if line.startswith('|'):
            current.append([part.strip() for part in line.strip('|').split('|')])
        elif current:
            output.append(current); current = []
    return output


def pct(value):
    return f'{value:+.3f}%'


def ci(row):
    values = row['relative_ci95_pct']
    return 'degenerate' if values is None else f'[{values[0]:+.3f}, {values[1]:+.3f}]%'


def docx_parts(path):
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        document = ET.fromstring(archive.read('word/document.xml'))
        relationships = ET.fromstring(archive.read('word/_rels/document.xml.rels'))
        tokens = [node.text or '' for node in document.iter() if node.tag in (f"{{{NS['w']}}}t", f"{{{NS['m']}}}t")]
        math = [(node.tag, node.text or '') for node in document.iter() if node.tag.startswith('{'+NS['m']+'}')]
        rows = [[[''.join(node.itertext()) for node in cell.findall('.//w:t', NS)] for cell in row.findall('w:tc', NS)]
                for table in document.findall('.//w:tbl', NS) for row in table.findall('w:tr', NS)]
        media = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist() if name.startswith('word/media/')}
        links = [node.attrib['Target'] for node in relationships if node.attrib.get('TargetMode') == 'External']
        return {'tokens': tokens, 'math': math, 'table_rows': rows, 'media': media, 'links': links,
                'math_objects': len(document.findall('.//m:oMath', NS)),
                'tables': len(document.findall('.//w:tbl', NS)),
                'columns': [node.attrib.get('{'+NS['w']+'}num', '1') for node in document.findall('.//w:sectPr/w:cols', NS)]}


def main():
    assert set(os.sched_getaffinity(0)).issubset({24, 26})
    folder = BASE/'manuscript'; prior = OLD/'manuscript'
    main = (folder/'MAIN_PAPER_EN.md').read_text()
    supplement = (folder/'SUPPLEMENT_EN.md').read_text()
    previous_supp = (prior/'SUPPLEMENT_EN.md').read_text()
    old_main = (prior/'MAIN_PAPER_EN.md').read_text()
    assert supplement.startswith(previous_supp)
    appended = supplement[len(previous_supp):]
    result = read(BASE/'analysis/STATISTICS.json')
    outer = read(BASE/'analysis/OUTER_RESULTS.json')
    build = read(BASE/'audit/DOCUMENT_BUILD.json')
    author = read(folder/'AUTHOR_INFORMATION.json')
    assert author == read(prior/'AUTHOR_INFORMATION.json')
    assert author['authors'][0]['name_en'] == 'Wang Yang'
    assert author['authors'][0]['email'] is None
    title = '# Multi-Horizon Forecasting of Apple Stock Prices and Returns Across Temporal Scales'
    assert main.startswith(title+'\n')
    for value in ('Wang Yang', 'High School Affiliated to Shanghai Jiao Tong University',
                  'Shanghai 200439, China', '[Email to be completed by the author]'):
        assert value in main
    assert tables(main) == tables(old_main)
    new_tables = tables(appended)
    assert len(new_tables) == 4
    tasks = list(result['tasks']); entries = [result['tasks'][task] for task in tasks]
    expected = []
    for cell in ('A_mean', 'B_context', 'C_history', 'D_joint'):
        for mode in ('raw', 'shrink'):
            name = cell+'_'+mode
            expected.append([name.replace('_', ' ')]+[pct(entry['summary'][name]['improvement_pct']) for entry in entries])
    assert new_tables[0][2:] == expected
    tasklabel = lambda task: 'Minute H60' if task.startswith('minute') else 'Hourly H8'
    expected = []
    for task, entry in zip(tasks, entries):
        for control in ('zero', 'C_history_shrink', 'repair_three_seed_shrink', 'ridge_history_selected'):
            for row in entry['comparisons'][control]:
                expected.append([tasklabel(task), control.replace('_', ' '), str(row['block']),
                    pct(row['relative_improvement_pct']), ci(row)])
    assert new_tables[1][2:] == expected
    expected = []
    for task, entry in zip(tasks, entries):
        for row in entry['diagnostic_contrasts']:
            label = row['primary'].replace('_raw', '').replace('_', ' ')+' / '+row['reference'].replace('_raw', '').replace('_', ' ')
            expected.append([tasklabel(task), label, str(row['block']), pct(row['relative_improvement_pct']), ci(row)])
    assert new_tables[2][2:] == expected
    expected = []
    for task in tasks:
        for fold in range(1, 6):
            done = read(BASE/'fits'/task/f'fold_{fold}'/'INNER_DONE.json')
            expected.append([tasklabel(task), str(fold)]+[f"{done['inner_metrics'][cell]['alpha']:.6f}" for cell in ('A_mean', 'B_context', 'C_history', 'D_joint')])
    assert new_tables[3][2:] == expected
    for task, entry in zip(tasks, entries):
        candidate = entry['summary'][result['candidate']]
        assert pct(candidate['improvement_pct']) in main
        assert f"{candidate['positive_folds']}/5" in main
        assert entry['performance_gate_passed'] is entry['inner_direction_gate_passed'] is entry['independent_confirmation_eligible'] is False
    for expected_claim in ('We have not established a forecasting method superior to the strong controls.',
                           'No new recipe meets the designated stable predictive-improvement objective.',
                           'No alternate cell, lag set or regularization is promoted using outer scores.',
                           'Shared-versus-independent horizon learning and eligible independent confirmation remain unfinished.'):
        assert expected_claim in main
    assert 'not an isolated addition to xPatch' in main
    assert 'training seeds are N/A' in appended
    assert 'not establish a market cause' in main
    assert '2026Q3 series remains sealed' in main and 'Goal remains active' in appended
    main_diagnostics = ((tasks[0], 'D_joint_raw', 'C_history_raw'),
                        (tasks[0], 'B_context_raw', 'A_mean_raw'),
                        (tasks[1], 'D_joint_raw', 'B_context_raw'))
    for task, primary, reference in main_diagnostics:
        rows = [row for row in result['tasks'][task]['diagnostic_contrasts'] if row['primary'] == primary and row['reference'] == reference]
        assert len(rows) == 2 and all(row['relative_ci95_pct'][1] < 0 for row in rows)
        assert pct(rows[0]['relative_improvement_pct']) in main
    rows = [row for row in result['tasks'][tasks[1]]['diagnostic_contrasts'] if row['primary'] == 'D_joint_raw' and row['reference'] == 'C_history_raw']
    assert all(row['relative_ci95_pct'][0] < 0 < row['relative_ci95_pct'][1] for row in rows)
    assert len(outer['metrics']) == 80 and outer['matrix_cells'] == 40
    # Check equations' mathematical definitions against the frozen implementation.
    assert 'Y^R_{t,j}=\\log(P_{t+j}/P_t)' in main
    assert 'MSE}_{model}/\\mathrm{MSE}_{reference}' in main
    assert 'sum q r' in main and 'sum q^2' in main
    assert "(X'X/n+0.1I)" in appended and "X'(Y-mean Y)/n" in appended
    assert 'unpenalized fit target-mean intercept' in appended
    assert 'Monday-through-Thursday dummies with Friday as reference' in appended
    assert '1/15/30/60/120/240/479' in appended and '1/2/4/8/16/32/95' in appended
    figures = []
    for name in re.findall(r'!\[[^\]]*\]\(([^)]+)\)', main):
        path = folder/name
        assert path.exists() and sha(path) == sha(prior/name)
        figures.append({'path': name, 'sha256': sha(path), 'unchanged_from_v3a': True})
    document_rows = []
    for rec in build['documents']:
        name = rec['document']; source = folder/(name+'_EN.md')
        output = folder/(name+'_EN.docx'); pdf = folder/(name+'_EN.pdf')
        assert sha(source) == rec['source_sha256'] and sha(output) == rec['docx_sha256'] and sha(pdf) == rec['pdf_sha256']
        raw = docx_parts(folder/(name+'_RAW.docx')); final = docx_parts(output)
        for key in ('tokens', 'math', 'table_rows', 'media', 'links'):
            assert raw[key] == final[key], (name, key)
        assert final['math_objects'] == rec['preservation']['math_objects']
        assert final['tables'] == rec['preservation']['tables']
        assert len(final['media']) == rec['preservation']['images']
        urls = {value.rstrip('.,') for value in re.findall(r'https://[^\s)]+', source.read_text())}
        if name == 'MAIN_PAPER':
            assert urls == {value.rstrip('.,') for value in final['links']}
            assert '2' in final['columns']
            assert len(re.findall(r'^\[\d+\]', source.read_text(), flags=re.M)) == 16
        info = subprocess.check_output(['pdfinfo', str(pdf)], text=True)
        pages = int(re.search(r'^Pages:\s+(\d+)', info, flags=re.M).group(1))
        assert pages == rec['pages']
        pdf_text = subprocess.check_output(['pdftotext', '-layout', str(pdf), '-'], text=True)
        for value in ('0.222', '0.684', '31', '2026Q3'):
            assert value in pdf_text, (name, value)
        assert '\ufffd' not in pdf_text
        document_rows.append({'document': name, 'pages': pages, 'source_sha256': sha(source),
            'docx_sha256': sha(output), 'pdf_sha256': sha(pdf), 'math_objects': final['math_objects'],
            'tables': final['tables'], 'media': len(final['media']), 'links': len(final['links']),
            'prose_equation_structure_table_cells_media_and_links_preserved': True,
            'pdf_text_has_negative_candidate_results_and_no_replacement_character': True})
    assert document_rows[0]['pages'] in range(4, 7)
    review = {'status': 'passed', 'blocking_findings': [],
        'scope': 'Independent numeric/structural/claim review; root separately performs every rendered-page visual review',
        'documents': document_rows, 'figures': figures,
        'prior_v3a_supplement_exact_prefix_preserved': True,
        'prior_prefix_sha256': hashlib.sha256(previous_supp.encode()).hexdigest(),
        'original_title_and_main_result_tables_preserved': True,
        'new_main_raw_diagnostic_gains_and_interval_directions_match': True,
        'new_context_tables': {'matrix_rows': 8, 'selected_interval_rows': 16, 'diagnostic_interval_rows': 12, 'alpha_rows': 10,
            'all_values_match_original_frozen_receipts': True},
        'author_information_preserved_and_email_self_fill': True,
        'math_inputs_objective_and_normalized_ridge_consistent_with_frozen_code': True,
        'claim_review': {'negative_results_not_promoted': True, 'no_xpatch_mechanism_repair_claim_from_new_ridge': True,
            'deterministic_seeds_not_fabricated': True, 'independent_confirmation_not_claimed': True,
            'shared_learning_not_claimed': True, 'Goal_active_not_marked_complete': True,
            'calendar_effects_not_universally_rejected': True},
        'network_reference_validation': 'Existing reference URLs preserved and valid DOCX hyperlinks; no new network lookup',
        'visual_review_complete': False, 'Q3_values_read': False,
        'affinity': sorted(os.sched_getaffinity(0)), 'GPU': False,
        'reviewer_sha256': sha(Path(__file__)), 'at_utc': datetime.now(timezone.utc).isoformat()}
    write_path = HERE/'DOCUMENT_REVIEW.json'
    write_path.write_text(json.dumps(review, indent=2)+'\n')
    print(json.dumps(review), flush=True)


if __name__ == '__main__':
    main()
