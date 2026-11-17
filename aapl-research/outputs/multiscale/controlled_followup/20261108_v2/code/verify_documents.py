from common import *
from zipfile import ZipFile
import xml.etree.ElementTree as ET
import re

def normalized(s):return re.sub(r'\s+','',s).replace('−','-').replace('–','-')

def main():
    result=read(BASE/'analysis/RESULTS.json');build=read(BASE/'audit/DOCUMENT_BUILD.json')
    folder=BASE/'manuscript';text=(folder/'MAIN_PAPER_EN.md').read_text();pdf=(folder/'MAIN_PAPER_PDF_TEXT.txt').read_text()
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
    with ZipFile(folder/'MAIN_PAPER_EN.docx') as z:
        root=ET.fromstring(z.read('word/document.xml'))
        prose=' '.join(x.text or '' for x in root.findall('.//w:t',ns))
        columns=[x.attrib.get('{'+ns['w']+'}num','1') for x in root.findall('.//w:cols',ns)]
        equations=len(root.findall('.//m:oMath',ns))
    values=[f"{x['improvement_pct']:+.3f}%" for x in result['pooled'] if x['variant']=='long_aligned' or
        x['task'] in [key(*a) for a in MECHANISM]]
    required=['Multi-Horizon Forecasting of Apple Stock Prices and Returns Across Temporal Scales','Wang Yang',
        'High School Affiliated to Shanghai Jiao Tong University','Shanghai 200439, China',
        'Email to be completed by the author']
    checks={
        'twenty_primary_tasks':len([x for x in result['pooled'] if x['variant']=='long_aligned'])==20,
        'all_selected_new_fits_complete':result['completed_new_fits']==len(read(BASE/'protocol/SELECTED_JOBS.json')['jobs']),
        'author_and_original_title_in_docx':all(normalized(a) in normalized(prose) for a in required),
        'author_and_original_title_in_pdf':all(normalized(a) in normalized(pdf) for a in required),
        'all_primary_and_factorial_percentages_in_source':all(v in text for v in values),
        'all_primary_and_factorial_percentages_in_pdf':all(normalized(v).lstrip('+') in normalized(pdf) for v in values),
        'two_column_body': '2' in columns,
        'editable_equations_present':equations>=5,
        'main_page_count_4_to_6':build['main_page_target_4_to_6'],
        'all_document_build_preservations_passed':all(x['preservation']['passed'] for x in build['documents']),
        'pdf_previews_cover_all_pages':all(x['preview_pages']==x['pages'] for x in build['documents']),
        'rendered_document_identities_current':all(sha(folder/(x['document']+'_EN.pdf'))==x['pdf_sha256'] and
            sha(folder/(x['document']+'_EN.docx'))==x['docx_sha256'] and sha(folder/(x['document']+'_EN.md'))==x['source_sha256']
            for x in build['documents']),
        'frozen_training_source_unchanged':code_hashes()==read(BASE/'protocol/CODE_FREEZE.json')['hashes'],
        'no_independent_confirmation_claim':result['independent_confirmation'] is False and result['Q3_values_read'] is False,
        'unrun_shared_experiment_disclosed':'remains unrun' in text and 'shared' in text}
    atomic(BASE/'audit/NUMERIC_AND_DOCUMENT_CHECK.json',{'status':'passed' if all(checks.values()) else 'requires_repair',
        'checks':checks,'equations':equations,'column_sections':columns,'percentage_values_checked':len(values),
        'manual_visual_review_still_required':True})
    print(json.dumps({'checks':checks,'failed':[k for k,v in checks.items() if not v]}),flush=True)
    if not all(checks.values()):raise SystemExit(1)

if __name__=='__main__':main()
