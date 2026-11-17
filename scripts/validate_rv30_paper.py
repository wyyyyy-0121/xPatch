"""Audit manuscript tables and document conversion against saved evidence.

Uses only the standard library plus pdfinfo. Does not train, score, access the
supplier, or inspect observation scores. An optional raw Pandoc DOCX allows a
content, embedded-image, and editable-equation preservation check.
"""
import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'research/paper'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}
MODELS = {'Previous RV': 'last_rv', 'HAR': 'har', 'HAR-X': 'har',
          'HAR/HAR-X': 'har', 'Ridge': 'ridge', 'LightGBM': 'lightgbm',
          'xPatch': 'xpatch', 'Chronos-2': 'chronos2'}
PANELS = {'AAPL': 'aapl', 'Market': 'market'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def docx_parts(path):
    with ZipFile(path) as archive:
        xml = ET.fromstring(archive.read('word/document.xml'))
        media = sorted(hashlib.sha256(archive.read(p)).hexdigest()
                       for p in archive.namelist() if p.startswith('word/media/'))
        styles = ET.fromstring(archive.read('word/styles.xml'))
    return xml, media, styles


def math_tree(element):
    return (element.tag, tuple(sorted(element.attrib.items())),
            element.text or '', tuple(math_tree(child) for child in element))


def paragraph_texts(xml, styles, formatted=False):
    style_names = {s.get('{'+NS['w']+'}styleId'):
                   s.find('w:name', NS).get('{'+NS['w']+'}val')
                   for s in styles.findall('w:style', NS)
                   if s.find('w:name', NS) is not None}
    result = []
    for p in xml.findall('.//w:body//w:p', NS):
        value = ''.join(n.text or '' for n in p.iter()
                        if n.tag in ('{'+NS['w']+'}t', '{'+NS['m']+'}t'))
        style = p.find('w:pPr/w:pStyle', NS)
        name = style_names.get(style.get('{'+NS['w']+'}val'), '') if style is not None else ''
        if name == 'Image Caption' or (name == 'Caption' and
                not value.startswith(('Figure ', 'Table '))):
            continue
        if formatted and re.match(r'^Table [1-7]\. ', value):
            continue
        if value.startswith("*Corresponding author") or value.startswith('Corresponding author'):
            continue
        if not formatted:
            if value == 'Wang Yang': value += '1,*'
            if value == 'High School Affiliated to Shanghai Jiao Tong University': value = '1' + value
            if value == 'Configurations': value = 'Configs.'
        value = ' '.join(value.split())
        if value: result.append(value)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-docx', type=Path)
    parser.add_argument('--docx', type=Path, default=PAPER/'AAPL_RV30_Wang_Yang.docx')
    parser.add_argument('--output', type=Path, default=PAPER/'validation.json')
    args = parser.parse_args()
    checks = 0

    def check(condition, label):
        nonlocal checks
        checks += 1
        if not condition: raise ValueError(label)

    def number(display, actual, label):
        shown = Decimal(display.replace('−', '-').replace(',', '').rstrip('%'))
        tolerance = Decimal('0.5') * Decimal(1).scaleb(shown.as_tuple().exponent)
        check(abs(shown - Decimal(str(actual))) <= tolerance + Decimal('1e-12'), label)

    source = (PAPER/'MANUSCRIPT_EN.md').read_text()
    tables = []
    in_table = False
    for line in source.splitlines():
        if line.startswith('|'):
            if not in_table: tables.append([])
            tables[-1].append([s.strip() for s in line.strip('|').split('|')])
            in_table = True
        else:
            in_table = False
    check(len(tables) == 7, 'Seven manuscript tables')
    tables = [table[2:] for table in tables]
    evidence = json.loads((PAPER/'evidence.json').read_text())
    for entry in evidence['inputs'].values():
        check(digest(ROOT/entry['path']) == entry['sha256'], 'Original evidence identity: '+entry['path'])
    for name, fingerprint in evidence['tables'].items():
        check(digest(PAPER/name) == fingerprint, 'Original CSV identity: '+name)
    diagnostics = json.loads((PAPER/'diagnostics/evidence.json').read_text())
    for path, entry in diagnostics['tables'].items():
        check(digest(ROOT/path) == entry['sha256'], 'Diagnostic identity: '+path)
        check(len(rows(ROOT/path)) == entry['rows'], 'Diagnostic row count: '+path)
    dev = {(r['information_set'], r['model']): r for r in rows(PAPER/'development.csv')}
    conf = {(r['information_set'], r['model']): r for r in rows(PAPER/'confirmation.csv')}
    annual = {(r['phase'], r['information_set'], r['model'], r['segment']): r
              for r in rows(PAPER/'diagnostics/annual.csv')}
    mz = {(r['phase'], r['information_set'], r['model']): r
          for r in rows(PAPER/'diagnostics/calibration.csv')}
    data = json.loads((ROOT/'delivery/data.json').read_text())
    check(len(tables[0]) == len(data['assets']), 'Complete asset table')
    for asset, expected, observed, missing, role in tables[0]:
        item = data['assets'][asset]
        for shown, actual in [(expected,item['expected_rows']), (observed,item['regular_rows']),
                              (missing,item['expected_rows']-item['regular_rows'])]:
            number(shown, actual, asset+' bar counts')
    search = json.loads((ROOT/'configs/rv30_search.json').read_text())['models']
    check(len(tables[1]) == len(search), 'Complete search table')
    for model, count, description in tables[1]:
        number(count.replace(' fixed',''),len(search[MODELS[model]]),model+' configurations')
    check(len(tables[2]) == len(dev), 'All development model/panel rows')
    check(len({(r[0],r[1]) for r in tables[2]}) == len(dev), 'No duplicated development rows')
    for panel, model, loss, mse, gain, pvalue, passed in tables[2]:
        r = dev[(PANELS[panel],MODELS[model])]
        number(loss,r['qlike'],model+' development loss')
        number(mse,r['log_rv_mse'],model+' development MSE')
        if model.startswith('HAR'):
            check((gain,pvalue,passed) == ('Reference','—','Reference'), 'HAR reference status')
        else:
            number(gain,r['qlike_improvement_vs_same_panel_HAR_pct'],model+' development gain')
            number(pvalue,r['holm_p'],model+' development Holm p')
            check(passed == ('Pass' if r['passed_all_registered_gates']=='True' else 'Fail'),model+' gate status')
    for panel, model, gain23, gain24, gain25, quarters, seed_rule in tables[3]:
        key = (PANELS[panel],MODELS[model])
        for segment, gain in [('2023',gain23),('2024',gain24),('2025H1',gain25)]:
            number(gain,annual[('development',*key,segment)]['qlike_gain_vs_same_panel_HAR_pct'],model+' annual gain '+segment)
        r = dev[key]
        check(quarters == r['improving_quarters']+'/'+r['total_quarters'],model+' quarter stability')
        expected = 'Fail' if 'all_seed_pooled_and_segments_improve' in r['failed_gate_checks'] else 'Pass'
        check(seed_rule == expected,model+' seed rule')
    check(len(tables[4]) == len(dev), 'Complete coverage table')
    for panel, model, coverage80, width80, coverage90, width90 in tables[4]:
        r = dev[(PANELS[panel],MODELS[model])]
        for shown, name, scale in [(coverage80,'coverage_80',100),(width80,'width_80',1e5),
                                   (coverage90,'coverage_90',100),(width90,'width_90',1e5)]:
            number(shown,float(r[name])*scale,model+' '+name)
    for phase, panel, model, intercept, slope, se in tables[5]:
        r = mz[(phase.lower(),PANELS[panel],MODELS[model])]
        number(intercept,float(r['intercept'])*1e5,model+' MZ intercept')
        number(slope,r['slope'],model+' MZ slope')
        number(se,r['slope_hac_se'],model+' MZ standard error')
    check(len(tables[6])*2 == len(conf), 'Complete confirmation table')
    for panel, baseline, loss, gain, pvalue, interval in tables[6]:
        base = conf[(PANELS[panel],'har')]; r = conf[(PANELS[panel],'ridge')]
        for shown, actual in [(baseline,base['qlike']),(loss,r['qlike']),
                              (gain,r['qlike_improvement_vs_same_panel_HAR_pct']),(pvalue,r['holm_p'])]:
            number(shown,actual,panel+' confirmation result')
        lower,upper=interval.strip('[]').split(', ')
        number(lower,r['ci95_five_day_lower'],panel+' confirmation CI lower')
        number(upper,r['ci95_five_day_upper'],panel+' confirmation CI upper')
    body, references = source.split('## References\n',1)
    check([int(i) for i in re.findall(r'^\[(\d+)\]',references,re.M)] == list(range(1,16)), 'Complete reference list')
    check(set(map(int,re.findall(r'\[(\d+)\]',body))) == set(range(1,16)), 'All numbered references cited')
    check(not re.search(r'\b(TODO|TBD|FIXME)\b',source), 'No editorial result placeholders')

    xml,media,styles=docx_parts(args.docx)
    check(len(xml.findall('.//w:tbl',NS))==7,'Seven Word tables')
    check(len(xml.findall('.//w:drawing',NS))==3,'Three Word figures')
    check('2' in [n.get('{'+NS['w']+'}num') for n in xml.findall('.//w:cols',NS)],'Word two-column body')
    figures=re.findall(r'!\[[^\]]+\]\((figures/[^)]+\.png)\)',source)
    check(media == sorted(digest(PAPER/f) for f in figures), 'Word embedded image bytes match figures')
    equations=xml.findall('.//m:oMath',NS)
    preservation = None
    if args.raw_docx:
        raw,raw_media,raw_styles=docx_parts(args.raw_docx)
        check(paragraph_texts(raw,raw_styles) == paragraph_texts(xml,styles,formatted=True),
              'Word conversion preserved all prose and table values apart from documented formatting')
        check(media==raw_media,'Word conversion preserved all image bytes')
        check([math_tree(m) for m in equations] == [math_tree(m) for m in raw.findall('.//m:oMath',NS)],
              'Word conversion preserved all editable equation trees')
        preservation={'raw_docx':str(args.raw_docx),'raw_sha256':digest(args.raw_docx),
                      'prose_and_table_values_preserved':True,'editable_math_preserved':True,
                      'image_bytes_preserved':True}
    build=json.loads((PAPER/'latex/build_manifest.json').read_text())
    check(digest(PAPER/'MANUSCRIPT_EN.md')==build['source_sha256'],'LaTeX manuscript identity')
    for path,expected in build['files'].items(): check(digest(ROOT/path)==expected,'LaTeX source identity: '+path)
    latex=(PAPER/'latex/body.tex').read_text()
    check(len(re.findall(r'\\begin\{table\*?\}',latex))==7,'Seven LaTeX tables')
    check(latex.count('\\begin{figure*}')==3,'Three LaTeX figures')
    log=(PAPER/'latex/main.log').read_text(errors='replace')
    check('Overfull' not in log,'No LaTeX overfull boxes')
    check('Undefined control sequence' not in log and not re.search(r'^!',log,re.M),'No LaTeX compile errors')
    pages=int(re.search(r'^Pages:\s+(\d+)',subprocess.check_output(
        ['pdfinfo',str(PAPER/'latex/main.pdf')],text=True),re.M).group(1))
    report={'checked_utc':datetime.now(timezone.utc).isoformat(),'passed':True,
            'scope':'historical_manuscript_numeric_provenance_and_document_conversion_not_new_inference',
            'checks':checks,'english_source_sha256':digest(PAPER/'MANUSCRIPT_EN.md'),
            'english_references':15,'chinese_version':'checking_digest_not_full_translation',
            'manuscript_tables':7,'manuscript_figures':3,'latex_pages':pages,
            'latex_overfull_boxes':0,'word_editable_math_objects':len(equations),
            'word_document':{'path':str(args.docx),'sha256':digest(args.docx)},
            'conversion_preservation':preservation,'new_training_or_test_statistics':False,
            'observer_scores_read':False,'final_research_acceptance':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__ == '__main__':
    main()
