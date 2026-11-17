"""Check the revised working manuscript against its data and source evidence."""
from collections import Counter
from datetime import datetime, timezone
import csv
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
from xml.etree import ElementTree as ET
from zipfile import ZipFile


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def output(command):
    return subprocess.run(command, text=True, capture_output=True, check=True).stdout


def doc_data(path):
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
          'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}
    with ZipFile(path) as z:
        root = ET.fromstring(z.read('word/document.xml'))
        rel = ET.fromstring(z.read('word/_rels/document.xml.rels'))
        tables = []
        for table in root.findall('.//w:tbl', ns):
            tables.append([[''.join(x.text or '' for x in cell.iter() if x.tag in
                           (f"{{{ns['w']}}}t", f"{{{ns['m']}}}t")).strip()
                            for cell in row.findall('./w:tc', ns)]
                           for row in table.findall('./w:tr', ns)])
        return {'text': [x.text or '' for x in root.findall('.//w:t', ns)],
                'math': [''.join(x.itertext()) for x in root.findall('.//m:oMath', ns)],
                'links': sorted(x.attrib.get('Target', '') for x in rel
                                if x.attrib.get('Type', '').endswith('/hyperlink')),
                'media': sorted(digest_bytes(z.read(n)) for n in z.namelist() if n.startswith('word/media/')),
                'tables': tables, 'drawings': len(root.findall('.//w:drawing', ns))}


def digest_bytes(value):
    return sha256(value).hexdigest()


def md_tables(text):
    tables, current = [], []
    for line in text.splitlines():
        if line.startswith('|'):
            if re.match(r'^\|\s*:?-+', line):
                continue
            current.append([c.strip().replace('\\(', '').replace('\\)', '')
                            for c in line.strip('|').split('|')])
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    return tables


def main():
    root = Path(sys.argv[1]).resolve()
    md, raw, word, pdf = (root / name for name in ('PRIVATE_WORKING_MANUSCRIPT_EN.md',
                          'PRIVATE_WORKING_MANUSCRIPT_RAW.docx', 'PRIVATE_WORKING_MANUSCRIPT_EN.docx',
                          'PRIVATE_WORKING_MANUSCRIPT_EN.pdf'))
    audit = load(root / 'MANUSCRIPT_UPDATE_AUDIT.json')
    require(digest(md) == audit['manuscript_sha256'], 'Markdown identity changed')
    for filename, sha in audit['csv_sha256'].items():
        require(digest(root / filename) == sha, 'A result CSV changed: ' + filename)
    permission = root / 'DATA_PERMISSION_CONFIRMATION.json'
    provenance = root / 'DATA_PROVENANCE_MANIFEST.json'
    minute = root / 'MINUTE_RECOMPUTATION.json'
    require(digest(permission) == audit['permission_sha256'] and
            digest(provenance) == audit['source_manifest_sha256'] and
            digest(minute) == audit['minute_recomputation_sha256'], 'Copied evidence differs')
    restore = load(audit['restoration_audit'])
    require(digest(audit['restoration_audit']) == audit['restoration_audit_sha256'] and
            restore['all_saved_predictions_exactly_reproduced'] is True and
            restore['model_restorations_checked'] == 240 and
            restore['origin_model_paths_checked'] == 179288, 'Restoration evidence differs')
    reference = load(root / 'reference_evidence_20261108/REFERENCE_EVIDENCE_VERIFICATION.json')
    require(reference['manuscript_sha256'] == digest(md) and reference['checks']['entries'] == 17,
            'Reference map does not cover this revision')
    before, after = doc_data(raw), doc_data(word)
    require(before == after, 'Word formatting changed body text, formulas, tables, links or images')
    require(len(after['math']) == 30 and len(after['links']) == 18 and len(after['tables']) == 25 and
            len(after['media']) == after['drawings'] == 2, 'Word asset counts differ')
    require(after['media'] == sorted(digest(root / p) for p in
            ('paper_artifacts/chronological_split_schematic.png', 'figures/completed_task_error_ratios.png')),
            'Embedded figures differ')
    text = md.read_text(encoding='utf8')
    require(md_tables(text) == after['tables'], 'Markdown and Word table cells differ')
    original = output(['pandoc', '--from=markdown+tex_math_single_backslash+autolink_bare_uris',
                       '--to=plain', '--wrap=none', str(md)])
    converted = output(['pandoc', '--from=docx', '--to=plain', '--wrap=none', str(word)])
    pattern = r"[A-Za-z]{3,}(?:[’'-][A-Za-z]+)?"
    a, b = Counter(re.findall(pattern, original)), Counter(re.findall(pattern, converted))
    for command in ('mathrm', 'qquad', 'quad', 'mspace', 'leq'):
        a.pop(command, None)
        b.pop(command, None)
    require(a == b, 'Word prose differs from Markdown: ' + str((a - b, b - a)))
    layout = output(['pdftotext', '-layout', '-enc', 'UTF-8', str(pdf), '-'])
    pages = [p for p in layout.split('\f') if p.strip()]
    info = output(['pdfinfo', str(pdf)])
    count = re.search(r'^Pages:\s+(\d+)\s*$', info, re.MULTILINE)
    require(count and int(count.group(1)) == len(pages) and
            all('WORKING DRAFT' in p and 'EXPERIMENTS IN PROGRESS' in p for p in pages),
            'PDF page count or working-draft footers differ')
    for phrase in ('179,288', '240 saved neural', 'maximum absolute difference zero', 'Alpaca and Kibot',
                   '535,620', '125 complete months', 'Figure 1.', 'Figure 2.', '68 of 100',
                   '-303817.03%', '-24.67%', '64.53%', '-0.0303697', '0.00529228', 'References'):
        require(phrase in layout, 'Rendered evidence is missing: ' + phrase)
    tasks, in_appendix = [], False
    task_pattern = r'(?:Annual H12 \(short history\)|Quarterly H3|Monthly H21|Hourly H[148]): (?:nominal close|cumulative log return)'
    for page in pages:
        if 'Appendix A.' in page:
            in_appendix = True
        if not in_appendix:
            continue
        matches = list(re.finditer(task_pattern, page))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(page)
            content = page[match.end():end]
            require('Five folds;' in content and 'End MSE' in content and 'xPatch' in content,
                    'Detailed results table split from its task label')
            tasks.append(match.group(0))
    require(len(tasks) == len(set(tasks)) == 12, 'Detailed task tables are missing or duplicated')
    previews = root / 'preview_verified'
    images = {str(i): digest(previews / f'page-{i:02}.png') for i in range(1, len(pages) + 1)}
    result = {'status': 'revised_working_manuscript_content_and_render_checks_passed',
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'markdown_sha256': digest(md), 'raw_docx_sha256': digest(raw),
              'formatted_docx_sha256': digest(word), 'rendered_pdf_sha256': digest(pdf),
              'script_sha256': digest(Path(__file__)), 'preview_sha256': images,
              'checks': {'equations': 30, 'tables': 25, 'hyperlinks': 18, 'embedded_figures': 2,
                         'all_markdown_table_cells_match_word': True,
                         'body_and_equations_preserved_by_formatting': True,
                         'prose_word_counts_match': True, 'references_rechecked': 17,
                         'result_csvs_unchanged': True, 'complete_detailed_task_tables': 12,
                         'rendered_pages': len(pages), 'working_footers_all_pages': True,
                         'permission_and_restoration_evidence_bound': True},
              'visual_review_recorded_separately': True,
              'limitations': ['The minute matrix remains incomplete and the snapshot is historical development.',
                              'Content checks do not establish forecast superiority or point-in-time data availability.']}
    out = root / 'WORD_CONTENT_VERIFICATION.json'
    with out.open('x', encoding='utf8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(json.dumps({'status': result['status'], 'path': str(out), 'checks': result['checks']}))


if __name__ == '__main__':
    main()
