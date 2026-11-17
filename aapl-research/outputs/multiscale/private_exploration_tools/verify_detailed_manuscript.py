"""Verify one private AAPL manuscript snapshot and its rendered Word document."""
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

root = Path(sys.argv[1]).resolve()
md = root / 'PRIVATE_WORKING_MANUSCRIPT_EN.md'
raw = root / 'PRIVATE_WORKING_MANUSCRIPT_RAW.docx'
docx = root / 'PRIVATE_WORKING_MANUSCRIPT_EN.docx'
pdf = root / 'PRIVATE_WORKING_MANUSCRIPT_EN.pdf'
source_audit = root / 'PRIVATE_DETAILED_RESULTS_AUDIT.json'
pool_audit = root / 'PRIVATE_POOL_VERIFICATION.json'
out = root / 'PRIVATE_WORD_VERIFICATION.json'
ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def require(ok, message):
    if not ok:
        raise ValueError(message)

def document_data(path):
    with ZipFile(path) as z:
        d = ET.fromstring(z.read('word/document.xml'))
        rel = ET.fromstring(z.read('word/_rels/document.xml.rels'))
        text = [x.text or '' for x in d.findall('.//w:t', ns)]
        math = [''.join(x.itertext()) for x in d.findall('.//m:oMath', ns)]
        links = sorted(x.attrib.get('Target', '') for x in rel if x.attrib.get('Type', '').endswith('/hyperlink'))
        media = sorted(hashlib.sha256(z.read(name)).hexdigest()
                       for name in z.namelist() if name.startswith('word/media/'))
        tables = []
        for t in d.findall('.//w:tbl', ns):
            rows = []
            for row in t.findall('./w:tr', ns):
                cells = []
                for cell in row.findall('./w:tc', ns):
                    parts = [x.text or '' for x in cell.iter()
                             if x.tag in (f"{{{ns['w']}}}t", f"{{{ns['m']}}}t")]
                    cells.append(''.join(parts).strip())
                rows.append(cells)
            tables.append(rows)
        return {'text': text, 'math': math, 'links': links, 'media': media, 'tables': tables,
                'drawing_count': len(d.findall('.//w:drawing', ns)),
                'footers': [name for name in z.namelist() if re.fullmatch(r'word/footer\d+\.xml', name)]}

def markdown_tables(body):
    result, current = [], []
    for line in body.splitlines():
        if line.startswith('|'):
            if re.match(r'^\|\s*:?-+', line):
                continue
            cells = [s.strip().replace('\\(', '').replace('\\)', '') for s in line.strip('|').split('|')]
            current.append(cells)
        elif current:
            result.append(current)
            current = []
    if current:
        result.append(current)
    return result

audit = json.loads(source_audit.read_text(encoding='utf8'))
require(digest(md) == audit['manuscript_sha256'], 'Evidence-checked Markdown changed')
before = document_data(raw)
after = document_data(docx)
for key in ('text', 'math', 'links', 'media', 'tables'):
    require(before[key] == after[key], 'Formatting changed ' + key)
require(len(after['math']) == 30 and len(after['links']) == 18 and len(after['tables']) == 23,
        'Unexpected formula, hyperlink, or table count')
require(after['drawing_count'] == 1 and len(after['footers']) == 1, 'Figure or footer missing')
require(markdown_tables(md.read_text(encoding='utf8')) == after['tables'], 'Markdown-to-Word table content differs')

def plain(args):
    proc = subprocess.run(args, capture_output=True, text=True, check=True)
    return proc.stdout

a = plain(['pandoc', '--from=markdown+tex_math_single_backslash+autolink_bare_uris',
           '--to=plain', '--wrap=none', str(md)])
b = plain(['pandoc', '--from=docx', '--to=plain', '--wrap=none', str(docx)])
word_re = r"[A-Za-z]{3,}(?:[’'-][A-Za-z]+)?"
counts_a, counts_b = Counter(re.findall(word_re, a)), Counter(re.findall(word_re, b))
for latex_command in ('mathrm', 'qquad', 'quad', 'mspace', 'leq'):
    counts_a.pop(latex_command, None)
    counts_b.pop(latex_command, None)
require(counts_a == counts_b, 'Text-word counts differ after excluding equation syntax')

info = plain(['pdfinfo', str(pdf)])
page_match = re.search(r'^Pages:\s+(\d+)\s*$', info, re.MULTILINE)
require(page_match is not None and int(page_match.group(1)) == 14, 'Unexpected rendered PDF page count')
layout = plain(['pdftotext', '-layout', '-enc', 'UTF-8', str(pdf), '-'])
pages = [p for p in layout.split('\f') if p.strip()]
require(len(pages) == 14 and all('PRIVATE WORKING DRAFT' in p for p in pages),
        'A rendered page or private footer is missing')
for phrase in ('PRIVATE WORKING MANUSCRIPT', '68 of 100', '240 neural', '1473.81',
               '0.00004971', '-303817.03%', '-24.67%', 'References'):
    require(phrase in layout, 'Rendered PDF omits ' + phrase)
pool = json.loads(pool_audit.read_text(encoding='utf8'))
require(pool['status'] == 'private_pooled_metrics_and_appendix_verified' and
        pool['manuscript_sha256'] == digest(md) and pool['checks']['appendix_numeric_cells_match_csv'] == 672,
        'Independent pooled-results audit differs')
started = False
table_titles = []
pattern = r'(?:Annual H12 \(short history\)|Quarterly H3|Monthly H21|Hourly H[148]): (?:nominal close|cumulative log return)'
for page in pages:
    if 'Appendix A.' in page:
        started = True
    if not started:
        continue
    matches = list(re.finditer(pattern, page))
    for i, match in enumerate(matches):
        stop = matches[i+1].start() if i+1 < len(matches) else len(page)
        segment = page[match.end():stop]
        require('End MSE' in segment and 'xPatch' in segment, 'An appendix table was split from its heading')
        table_titles.append(match.group(0))
require(len(table_titles) == 12 and len(set(table_titles)) == 12, 'Appendix task pages are incomplete')
previews = Path('/tmp/aapl_manuscript_review_20261108')
for page in range(1, 11):
    require(digest(previews / f'page_{page}.png') == digest(previews / f'current_page_{page}.png'),
            'A previously reviewed main page changed')
for page in range(11, 15):
    require((previews / f'paginated_page_{page}.png').is_file(), 'A reviewed appendix page image is missing')
receipt = {
    'status': 'verified_private_working_draft_not_submission_ready',
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'markdown_sha256': digest(md), 'raw_docx_sha256': digest(raw),
    'formatted_docx_sha256': digest(docx), 'rendered_pdf_sha256': digest(pdf),
    'evidence_audit_sha256': digest(source_audit), 'pool_audit_sha256': digest(pool_audit), 'verification_script_sha256': digest(Path(__file__).resolve()),
    'checks': {
        'editable_equations': len(after['math']), 'hyperlinks': len(after['links']),
        'tables': len(after['tables']), 'body_drawings': after['drawing_count'],
        'body_text_math_links_media_and_tables_preserved_by_formatting': True,
        'every_markdown_table_cell_matches_word': True,
        'prose_word_counts_match_after_excluding_math_syntax': True,
        'rendered_pages': len(pages), 'all_pages_contain_private_footer': True,
        'visual_pages_reviewed': list(range(1, 15)),
        'appendix_tables_complete_with_heading': len(table_titles),
        'main_pages_pixel_identical_to_previous_review': 10,
    },
    'limitations': [
        'Visual review and format checks do not establish publication rights or independent market confirmation.',
        'The document is a private development snapshot; some experiments and submission requirements remain incomplete.',
    ],
}
require(not out.exists(), 'Verification receipt already exists')
out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
print(json.dumps({'status': receipt['status'], 'path': str(out), 'checks': receipt['checks']}, ensure_ascii=False))
