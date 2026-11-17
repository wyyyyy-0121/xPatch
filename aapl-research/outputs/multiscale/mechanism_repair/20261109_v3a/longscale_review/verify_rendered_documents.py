"""Independent source-to-DOCX and rendered-number acceptance after parent build.

Visual inspection is explicitly separate. The source conversion is repeated in
this reviewer directory; the parent's RAW.docx is not trusted as source truth.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
from xml.etree import ElementTree as ET
from zipfile import ZipFile
import hashlib
import json
import os
import re
import subprocess

B = Path(__file__).resolve().parents[1]
D = B/'manuscript'
R = Path(__file__).resolve().parent
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
M='http://schemas.openxmlformats.org/officeDocument/2006/math'
NS={'w':W,'m':M}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def compact(s):
    for c in ('\u2212','\u2010','\u2011','\u2012','\u2013'):
        s=s.replace(c,'-')
    return re.sub(r'\s+','',s.replace('\u00ad',''))

def inspect(p):
    with ZipFile(p) as z:
        root=ET.fromstring(z.read('word/document.xml'))
        rels=ET.fromstring(z.read('word/_rels/document.xml.rels'))
        signature={
            'ordered_prose_nodes':[n.text or '' for n in root.findall('.//w:t',NS)],
            'equations':[''.join(n.itertext()) for n in root.findall('.//m:oMath',NS)],
            'table_cells':[[[''.join(n.itertext()) for n in cell.findall('.//w:p',NS)] for cell in row.findall('w:tc',NS)] for table in root.findall('.//w:tbl',NS) for row in table.findall('w:tr',NS)],
            'media_sha256':sorted(sha_bytes(z.read(n)) for n in z.namelist() if n.startswith('word/media/')),
            'hyperlinks':sorted(r.attrib['Target'] for r in rels if r.attrib.get('Type','').endswith('/hyperlink'))}
        return signature, root

def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()

def numeric_cells(md):
    values=[]
    for line in md.splitlines():
        if not line.startswith('|'):continue
        for value in line.strip('|').split('|'):
            value=value.strip()
            if re.fullmatch(r'[+−\-\d.,%/\[\]– ]+(?:to[+−\-\d.,%/\[\]– ]+)?',value) and any(c.isdigit() for c in value):
                # Ranges may line-wrap; validate each endpoint independently.
                values += re.findall(r'[+−\-]?\d+(?:\.\d+)?%?(?:/\d+)?',value)
    return Counter(compact(v) for v in values)

def main():
    assert len(os.sched_getaffinity(0))<=2
    reviewed=json.loads((R/'MANUSCRIPT_REVIEW.json').read_text())
    build=json.loads((B/'audit/DOCUMENT_BUILD.json').read_text())
    checks,docs={},[]
    def check(key, ok, detail=None):
        checks[key]={'passed':bool(ok),'detail':detail}
    for name in ('MAIN_PAPER','SUPPLEMENT'):
        source=D/f'{name}_EN.md';docx=D/f'{name}_EN.docx';pdf=D/f'{name}_EN.pdf'
        check(f'{name}_reviewed_source_unchanged',reviewed['source_sha256'].get(str(source))==sha(source))
        independent=R/f'{name}_INDEPENDENT_FROM_SOURCE.docx'
        result=subprocess.run(['pandoc','--from=markdown+tex_math_single_backslash+autolink_bare_uris','--to=docx','--standalone',f'--resource-path={D}',f'--output={independent}',str(source)],text=True,capture_output=True,check=True)
        (R/f'{name}_INDEPENDENT_CONVERSION.log').write_text(result.stdout+result.stderr)
        expected,_=inspect(independent);actual,root=inspect(docx)
        for key in expected:
            check(f'{name}_source_to_DOCX_{key}',expected[key]==actual[key],{'expected_count':len(expected[key]),'actual_count':len(actual[key])})
        columns=[n.attrib.get('{'+W+'}num','1') for n in root.findall('.//w:cols',NS)]
        check(f'{name}_column_layout','2' in columns and '1' in columns if name=='MAIN_PAPER' else all(n=='1' for n in columns))
        tables=root.findall('.//w:tbl',NS)
        check(f'{name}_repeated_table_headers',all(t.find('w:tr/w:trPr/w:tblHeader',NS) is not None for t in tables))
        check(f'{name}_rows_keep_together',all(row.find('w:trPr/w:cantSplit',NS) is not None for t in tables for row in t.findall('w:tr',NS)))
        text=subprocess.check_output(['pdftotext','-layout',str(pdf),'-'],text=True)
        (R/f'{name}_INDEPENDENT_PDF_TEXT.txt').write_text(text)
        ct=compact(text);needed=numeric_cells(source.read_text())
        missing={value:{'minimum_table_occurrences':count,'PDF_occurrences':ct.count(value)} for value,count in needed.items() if ct.count(value)<count}
        check(f'{name}_all_table_numeric_tokens_in_PDF',not missing,missing)
        if name=='MAIN_PAPER':
            for value in ['Multi-Horizon Forecasting of Apple Stock Prices and Returns Across Temporal Scales','Wang Yang','High School Affiliated to Shanghai Jiao Tong University','Shanghai 200439, China','Email to be completed by the author']:
                check(f'{name}_PDF_identity_{value}',compact(value) in ct)
        fonts=subprocess.check_output(['pdffonts',str(pdf)],text=True)
        check(f'{name}_TimesNewRoman_embedded','TimesNewRoman' in fonts)
        info=subprocess.check_output(['pdfinfo',str(pdf)],text=True)
        pages=int(next(line.split(':',1)[1] for line in info.splitlines() if line.startswith('Pages:')))
        if name=='MAIN_PAPER':check('main_page_target_4_to_6',4<=pages<=6)
        record=next(x for x in build['documents'] if x['document']==name)
        check(f'{name}_parent_build_hashes_match',record['source_sha256']==sha(source) and record['docx_sha256']==sha(docx) and record['pdf_sha256']==sha(pdf) and record['pages']==pages)
        docs.append({'name':name,'pages':pages,'source_sha256':sha(source),'docx_sha256':sha(docx),'pdf_sha256':sha(pdf),'source_prose_nodes':len(expected['ordered_prose_nodes']),'math_objects':len(expected['equations']),'table_count':len(tables),'table_numeric_tokens':sum(needed.values()),'table_distinct_numeric_tokens':len(needed)})
    passed=all(x['passed'] for x in checks.values())
    receipt={'status':'passed' if passed else 'failed','checks':checks,'documents':docs,
        'scope':'Independent Markdown conversion compared with formatted DOCX: ordered prose nodes, equations, complete table-cell structure, figure bytes and hyperlink targets; PDF contains every table numeric token with required minimum multiplicity. Parent performs page-level visual inspection.',
        'visual_acceptance':'separate_parent_review_required','BN_claim_acceptance':'separate_numerical_gate_review_required',
        'research_goal_achieved':False,'independent_confirmation':False,'Q3_values_read':False,
        'verifier_sha256':sha(Path(__file__)),'at_utc':datetime.now(timezone.utc).isoformat()}
    (R/'RENDERED_DOCUMENTS_REVIEW.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'status':receipt['status'],'checks':len(checks),'failed':[k for k,v in checks.items() if not v['passed']],'documents':docs},indent=2))
    if not passed:raise SystemExit(1)

if __name__=='__main__':main()
