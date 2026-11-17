"""Verify all manuscript cells and the inserted metadata evidence."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from xml.etree import ElementTree as ET
from zipfile import ZipFile

NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path): return json.loads(path.read_text())
def require(ok, message):
    if not ok: raise ValueError(message)
def command(args): return subprocess.run(args, capture_output=True, text=True, check=True).stdout


def document(path):
    with ZipFile(path) as z:
        body = ET.fromstring(z.read('word/document.xml'))
        relations = ET.fromstring(z.read('word/_rels/document.xml.rels'))
        tables = []
        for table in body.findall('.//w:tbl', NS):
            tables.append([[''.join(x.text or '' for x in cell.iter() if x.tag in
                            (f"{{{NS['w']}}}t", f"{{{NS['m']}}}t")).strip()
                            for cell in row.findall('./w:tc', NS)]
                           for row in table.findall('./w:tr', NS)])
        return {'text': [x.text or '' for x in body.findall('.//w:t', NS)],
                'math': [''.join(x.itertext()) for x in body.findall('.//m:oMath', NS)],
                'tables': tables, 'drawings': len(body.findall('.//w:drawing', NS)),
                'links': sorted(x.attrib.get('Target', '') for x in relations
                                if x.attrib.get('Type', '').endswith('/hyperlink')),
                'media': sorted(hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()
                                if n.startswith('word/media/'))}


def markdown_tables(text):
    tables, current = [], []
    for line in text.splitlines():
        if line.startswith('|'):
            if re.match(r'^\|\s*:?-+', line): continue
            current.append([s.strip().replace('\\(', '').replace('\\)', '')
                            for s in line.strip('|').split('|')])
        elif current:
            tables.append(current)
            current = []
    if current: tables.append(current)
    return tables


def main():
    root = Path(sys.argv[1]).resolve()
    md, raw, word, pdf = [root / ('PRIVATE_WORKING_MANUSCRIPT_' + s) for s in
                          ('EN.md', 'RAW.docx', 'EN.docx', 'EN.pdf')]
    insertion = load(root / 'TIMING_INSERTION_AUDIT.json')
    refresh = load(root / 'SNAPSHOT_REFRESH_AUDIT.json') if (root / 'SNAPSHOT_REFRESH_AUDIT.json').exists() else None
    n = refresh['completed_folds'] if refresh else 74
    require(digest(md) == insertion['markdown_sha256'], 'Markdown changed after insertion')
    parent = Path(insertion['parent'])
    require(digest(parent / md.name) == insertion['parent_markdown_sha256'], 'Parent source changed')
    audit = load(root / 'forecast_timing/DESIGN_TIMING_AUDIT.json')
    require(digest(root / 'forecast_timing/DESIGN_TIMING_AUDIT.json') == insertion['audit_sha256'],
            'Timing evidence changed')
    timing_n = audit['completed_saved_fold_metadata_verified']
    require(audit['planned_folds_verified'] == 100 and
            timing_n == insertion['timing_saved_file_snapshot_completed_folds'] and
            audit['paired_price_return_designs_verified'] == 50, 'Unexpected timing scope')
    before, after = document(raw), document(word)
    for field in ('text', 'math', 'tables', 'links', 'media'):
        require(before[field] == after[field], 'Formatting changed ' + field)
    table_count = 37 if refresh else 35
    require((len(after['math']), len(after['links']), len(after['tables']), after['drawings']) ==
            (30, 18, table_count, 2), 'Unexpected document structure')
    body = md.read_text(encoding='utf8')
    require(markdown_tables(body) == after['tables'], 'Markdown and Word table cells differ')
    result = load(root / 'all_task_results/RESULTS.json')
    result_source = refresh if refresh else load(parent / 'MANUSCRIPT_UPDATE_AUDIT.json')
    require(digest(root / 'all_task_results/RESULTS.json') == result_source['results_sha256'],
            'Numerical result source differs from parent')
    for name, expected in result['csv_sha256'].items():
        require(digest(root / 'all_task_results' / name) == expected, 'Result CSV changed')
    appendix = body.split('## Appendix A. Detailed historical development results', 1)[1]
    appendix, timing_appendix = appendix.split('## Appendix B. Forecast timing and original fold design', 1)
    require(len(re.findall(r'^### ', appendix, re.M)) == 20, 'Appendix A task count differs')
    tables = markdown_tables(appendix)
    require(len(tables) == 20, 'Appendix A table count differs')
    order = ('annual', 'quarterly', 'monthly', 'hourly', 'minute')
    tasks = sorted(result['tasks'], key=lambda t: (order.index(t['profile']), t['horizon'], t['target']))
    fields = 0
    for task, table in zip(tasks, tables):
        require(len(table) == 9, 'Task table lacks eight models')
        for model, row in zip(('last_close', 'drift', 'seasonal', 'ridge', 'mlp', 'gru', 'lstm', 'xpatch'), table[1:]):
            metrics = task['pooled']['models'][model]
            expected = [f'{metrics[k]:.6g}' for k in
                        ('endpoint_mse', 'endpoint_rmse', 'endpoint_mae', 'path_mse')]
            expected += ['—' if metrics[k] is None else f'{100 * metrics[k]:.2f}%' for k in
                         ('relative_improvement', 'direction_coverage', 'direction_accuracy_when_active')]
            require(row[1:] == expected, 'Outcome fields differ from recomputed results')
            fields += len(expected)
    timing_tasks = [t for t in sorted(audit['tasks'], key=lambda t:
                    (order.index(t['profile']), t['horizon'], t['target'])) if t['target'] == 'future_close']
    timing_tables = markdown_tables(timing_appendix)
    require(len(timing_tables) == 1 and len(timing_tables[0]) == 11, 'Timing table shape differs')
    timing_fields = 0
    for task, row in zip(timing_tasks, timing_tables[0][1:]):
        timing = task['planned_origin_timing']
        expected = [task['profile'].title() + f" H{task['horizon']}", f"{timing['origins']:,}"]
        expected += [f'{timing[k]:.6g}' for k in ('elapsed_hours_min', 'elapsed_hours_median',
                                                'elapsed_hours_p95', 'elapsed_hours_max')]
        expected += [f"{100 * timing['cross_session_fraction']:.2f}%"]
        require(row == expected, 'Timing fields differ from audit')
        timing_fields += len(row) - 1
    split_fields = 0
    if refresh:
        split_path = root / 'monthly_split_groups/SPLIT_GROUP_AUDIT.json'
        require(digest(split_path) == refresh['split_group_audit_sha256'], 'Split-group evidence changed')
        split = load(split_path)
        split_section = body.split('### Split sensitivity and label definition', 1)[1].split(
            'A subsequent label-only diagnostic', 1)[0]
        split_tables = markdown_tables(split_section)
        require(len(split_tables) == 2, 'Split-group tables missing')
        for target, table in zip(('future_close', 'future_log_return'), split_tables):
            require(len(table) == 9, 'Split-group table lacks eight models')
            for model, row in zip(('last_close', 'drift', 'seasonal', 'ridge', 'mlp', 'gru', 'lstm', 'xpatch'), table[1:]):
                cross = split['groups'][target]['crosses_split']['models'][model]
                other = split['groups'][target]['no_split']['models'][model]
                expected = [f"{cross['endpoint_mse']:.6g}", f"{other['endpoint_mse']:.6g}",
                            f"{100 * cross['relative_improvement']:.2f}%", f"{100 * other['relative_improvement']:.2f}%"]
                require(row[1:] == expected, 'Split-group table differs from recomputed evidence')
                split_fields += len(expected)
    a = command(['pandoc', '--from=markdown+tex_math_single_backslash+autolink_bare_uris',
                 '--to=plain', '--wrap=none', str(md)])
    b = command(['pandoc', '--from=docx', '--to=plain', '--wrap=none', str(word)])
    pattern = r"[A-Za-z]{3,}(?:[’'-][A-Za-z]+)?"
    ca, cb = Counter(re.findall(pattern, a)), Counter(re.findall(pattern, b))
    for token in ('mathrm', 'qquad', 'quad', 'mspace', 'leq'):
        ca.pop(token, None)
        cb.pop(token, None)
    require(ca == cb, 'Prose words changed during conversion')
    info = command(['pdfinfo', str(pdf)])
    count = int(re.search(r'^Pages:\s+(\d+)\s*$', info, re.M)[1])
    rendered = command(['pdftotext', '-layout', '-enc', 'UTF-8', str(pdf), '-'])
    pages = [p for p in rendered.split('\f') if p.strip()]
    require(count == len(pages) and count >= 19 and all('PRIVATE WORKING DRAFT' in p for p in pages),
            'PDF pages or footers differ')
    for phrase in ('400 partition hashes', '119 hours', f'{timing_n} completed', 'Appendix B', f'{n} of 100'):
        require(phrase in rendered, 'Rendered text lacks ' + phrase)
    (root / 'RENDERED_TEXT.txt').write_text(rendered, encoding='utf8')
    receipt = {'status': 'timing_manuscript_content_and_rendered_text_verified',
               'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'markdown_sha256': digest(md), 'raw_docx_sha256': digest(raw),
               'formatted_docx_sha256': digest(word), 'rendered_pdf_sha256': digest(pdf),
               'script_sha256': digest(Path(__file__)),
               'checks': {'editable_equations': 30, 'hyperlinks': 18, 'tables': table_count, 'figures': 2,
                          'appendix_outcome_fields_verified': fields, 'timing_appendix_numeric_fields_verified': timing_fields,
                          'split_group_numeric_fields_verified': split_fields,
                          'all_markdown_table_cells_match_word': True, 'prose_word_counts_match': True,
                          'formatting_preserves_text_equations_links_media_tables': True,
                          'pdf_pages': count, 'all_pages_have_draft_footer': True},
               'outcome_snapshot_completed_folds': n, 'timing_saved_file_snapshot_completed_folds': timing_n,
               'planned_design_folds': 100, 'visual_review_pending': True, 'Q3_values_read': False}
    with (root / 'WORD_CONTENT_VERIFICATION.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(receipt['checks']))


if __name__ == '__main__': main()
