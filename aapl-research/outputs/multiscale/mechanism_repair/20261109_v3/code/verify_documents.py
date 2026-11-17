"""Check generated paper values, retained title and text/media-preserving layout."""
from study import *
from zipfile import ZipFile
from xml.etree import ElementTree as ET
import re


def main():
    result = read(BASE/'analysis/CANDIDATE_RESULTS.json')
    folder = BASE/'manuscript'; text = (folder/'MAIN_PAPER_EN.md').read_text()
    pdf_text = (folder/'MAIN_PAPER_PDF_TEXT.txt').read_text()
    title = 'Multi-Horizon Forecasting of Apple Stock Prices and Returns Across Temporal Scales'
    flat = re.sub(r'\s+', ' ', pdf_text)
    assert title in flat and 'Wang Yang' in flat
    assert 'High School Affiliated to Shanghai Jiao Tong University' in flat
    assert '200439' in flat and 'Email to be completed by the author' in flat
    checks = []
    for profile, (_, _, hs) in PROFILES.items():
        for h in hs:
            rows = [next(r for r in result['pooled'] if r['task'] == task_key(profile, h, t) and
                         r['method'] == 'candidate') for t in ('future_close', 'future_log_return')]
            expected = '| '+' | '.join([f'{profile.capitalize()} H{h}',
                f"{rows[0]['improvement_pct']:+.3f}%", f"{rows[1]['improvement_pct']:+.3f}%",
                f"{rows[0]['seed_count']}/{rows[1]['seed_count']}"])+ ' |'
            assert expected in text, expected
            for row in rows:
                value = f"{row['improvement_pct']:+.3f}%"
                normalized_pdf = flat.replace('\u2212', '-').replace('\u2010', '-').replace('\u2011', '-')
                assert value in normalized_pdf, value
            checks.append({'profile': profile, 'horizon': h, 'both_targets_checked': True})
    assert 'Neither designated return task passes the predictive gate' in text
    assert not result['primary_tasks_both_pass']
    assert title == text.splitlines()[0].removeprefix('# ')
    namespace = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    with ZipFile(folder/'MAIN_PAPER_EN.docx') as z:
        body = ET.fromstring(z.read('word/document.xml'))
        columns = [node.attrib.get('{'+namespace['w']+'}num', '1') for node in body.findall('.//w:cols', namespace)]
    assert '2' in columns and '1' in columns
    documents = []
    for name in ('MAIN_PAPER', 'SUPPLEMENT'):
        preserved = read(str(folder/f'{name}_EN.docx')+'.preservation.json')
        assert preserved['passed'] and preserved['prose_equations_media_links_tables_preserved']
        documents.append({'name': name, 'preservation_passed': True,
                          'pdf_sha256': sha(folder/f'{name}_EN.pdf'),
                          'source_sha256': sha(folder/f'{name}_EN.md')})
    atomic(BASE/'audit/NUMERIC_AND_DOCUMENT_CHECK.json', {'status': 'passed', 'twenty_targets': checks,
           'title_and_author_verified': True, 'main_body_columns': 2, 'documents': documents,
           'visual_review_still_required': True, 'Q3_values_read': False, 'at_utc': now()})
    print(json.dumps({'status': 'passed', 'task_horizon_rows': len(checks), 'documents': 2}), flush=True)


if __name__ == '__main__':
    main()
