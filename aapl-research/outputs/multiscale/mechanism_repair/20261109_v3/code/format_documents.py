"""Set paper layout while preserving every source text, equation and media item."""
from pathlib import Path
from copy import deepcopy
from zipfile import ZipFile
from xml.etree import ElementTree as ET
import hashlib, json, sys
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Mm, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH


def inspect(path):
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
        'm':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
    with ZipFile(path) as z:
        root=ET.fromstring(z.read('word/document.xml'))
        rels=ET.fromstring(z.read('word/_rels/document.xml.rels'))
        return {'text':[x.text or '' for x in root.findall('.//w:t',ns)],
            'math':[''.join(x.itertext()) for x in root.findall('.//m:oMath',ns)],
            'media':sorted(hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith('word/media/')),
            'tables':len(root.findall('.//w:tbl',ns)),
            'links':sorted(x.attrib['Target'] for x in rels if x.attrib.get('Type','').endswith('/hyperlink'))}


def columns(pr, n):
    old=pr.find(qn('w:cols'))
    if old is not None: pr.remove(old)
    node=OxmlElement('w:cols'); node.set(qn('w:num'),str(n)); node.set(qn('w:space'),'288'); pr.append(node)


def format_doc(raw, out, two_columns=True):
    before=inspect(raw)
    doc=Document(raw); section=doc.sections[0]
    section.page_width,section.page_height=Mm(210),Mm(297)
    section.top_margin=section.bottom_margin=section.left_margin=section.right_margin=Mm(16)
    section.footer_distance=Mm(7)
    for name in ('Normal','Body Text','First Paragraph','Compact'):
        if name in doc.styles:
            style=doc.styles[name]; style.font.name='Times New Roman'; style.font.size=Pt(10)
            p=style.paragraph_format; p.space_before=Pt(0); p.space_after=Pt(3); p.line_spacing=1.
    for name in ('Heading 1','Heading 2','Heading 3'):
        style=doc.styles[name]; style.font.name='Times New Roman'; style.font.size=Pt(10.5)
        style.font.bold=True; style.font.color.rgb=RGBColor(0,0,0)
        style.paragraph_format.space_before=Pt(7); style.paragraph_format.space_after=Pt(3)
        style.paragraph_format.keep_with_next=True
    for name in ('Caption','Image Caption'):
        if name in doc.styles:
            doc.styles[name].font.name='Times New Roman'; doc.styles[name].font.size=Pt(8.5)
    abstract = next((i for i,p in enumerate(doc.paragraphs) if p.text=='Abstract'), None)
    if abstract is None: abstract=1
    refs=False
    for i,p in enumerate(doc.paragraphs):
        p.paragraph_format.widow_control=True
        p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
        if p.style.name.startswith('Heading'):p.alignment=WD_ALIGN_PARAGRAPH.LEFT
        if i<abstract:
            p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after=Pt(2); p.paragraph_format.keep_with_next=True
            for run in p.runs: run.font.size=Pt(15 if i==0 else 10); run.bold=(i==0)
        if p.text=='References': refs=True
        elif refs:
            p.alignment=WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.left_indent=Mm(3); p.paragraph_format.first_line_indent=Mm(-3)
            for run in p.runs: run.font.size=Pt(8.5)
        if p.text.startswith(('Table 1.','Table 2.')):
            p.paragraph_format.keep_with_next=True
            p.paragraph_format.keep_together=True
    for run in doc.element.body.iter(qn('w:r')):
        rpr=run.find(qn('w:rPr'))
        if rpr is None: rpr=OxmlElement('w:rPr'); run.insert(0,rpr)
        fonts=rpr.find(qn('w:rFonts'))
        if fonts is None: fonts=OxmlElement('w:rFonts'); rpr.insert(0,fonts)
        fonts.set(qn('w:ascii'),'Times New Roman'); fonts.set(qn('w:hAnsi'),'Times New Roman')
    width=(178-5.08)/2 if two_columns else 178
    for table in doc.tables:
        table.autofit=False
        if 'Table Grid' in doc.styles: table.style='Table Grid'
        n=len(table.columns)
        ratios=([.36,.22,.27,.15] if n==4 and two_columns else
            ([.40]+[.60/(n-1)]*(n-1)) if n>1 else [1.])
        heads=[c.text.lower() for c in table.rows[0].cells]
        if not two_columns:
            if n==2:ratios=[.72,.28]
            elif n==3:
                ratios=([.13,.72,.15] if 'implemented' in heads[1] else
                    [.10,.20,.70] if 'alpha' in heads[1] else [.50,.25,.25])
            elif n==4:ratios=[.08,.36,.36,.20] if 'trend-retained' in heads[1] else [.40,.20,.22,.18]
            elif n==5:
                ratios=([.06,.22,.22,.22,.28] if 'fit dates' in heads[1] else
                    [.25,.08,.17,.25,.25] if heads[0]=='reference' else [.15,.22,.22,.22,.19])
            elif n==6:ratios=[.06,.25,.12,.13,.22,.22]
            elif n==7:
                ratios=([.06,.12,.19,.12,.14,.18,.19] if heads[2]=='state' else
                    [.06,.22,.08,.17,.15,.16,.16] if heads[2]=='seed' else [.19]+[.135]*6)
            elif n==8:
                ratios=([.05,.18,.06,.07,.09,.23,.16,.16] if 'local variant' in heads[1] else
                    [.06,.24,.08,.09,.08,.16,.17,.12])
            # Follow-up tables begin with long task identities, unlike the older
            # fold-index-first tables for which the generic widths were written.
            if heads[0]=='task / variant' and n==6:ratios=[.42,.07,.07,.16,.12,.16]
            elif heads[0]=='task / variant' and n==4:ratios=[.49,.17,.17,.17]
            elif heads[0]=='task / comparison' and n==4:ratios=[.43,.07,.13,.37]
            elif heads[0]=='task / model' and n==5:ratios=[.40,.07,.18,.18,.17]
            elif heads[0]=='task / model' and n==4:ratios=[.43,.19,.20,.18]
            elif heads[0]=='task' and n==6:ratios=[.32,.12,.15,.13,.16,.12]
            elif heads[0]=='task' and n==5:ratios=[.33,.07,.17,.31,.12]
            elif heads[0]=='method' and n==4:ratios=[.30,.14,.30,.26]
            elif heads[0]=='gate versus' and n==4:ratios=[.18,.08,.17,.57]
            elif heads[0]=='scale' and n==6:ratios=[.14,.12,.12,.23,.23,.16]
            elif heads[0]=='task' and n==5 and heads[1]=='origins':ratios=[.22,.12,.22,.22,.22]
            elif heads[0]=='model' and n==5 and heads[1]=='implementation':ratios=[.10,.48,.10,.16,.16]
        for j,col in enumerate(table.columns): col.width=Mm(width*ratios[j])
        for i,row in enumerate(table.rows):
            row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
            if i==0: row._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
            for j,cell in enumerate(row.cells):
                cell.width=Mm(width*ratios[j])
                margins=OxmlElement('w:tcMar')
                for side in ('top','bottom','left','right'):
                    edge=OxmlElement('w:'+side);edge.set(qn('w:w'),'32' if side in ('top','bottom') else '40')
                    edge.set(qn('w:type'),'dxa');margins.append(edge)
                cell._tc.get_or_add_tcPr().append(margins)
                for p in cell.paragraphs:
                    p.paragraph_format.space_after=p.paragraph_format.space_before=Pt(0)
                    p.paragraph_format.keep_with_next=(i<len(table.rows)-1) if len(table.rows)<14 else i==0
                    textual=any(x in heads[j] for x in ('method','model','state','structure','variant','implementation'))
                    p.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 or textual else WD_ALIGN_PARAGRAPH.RIGHT
                    for run in p.runs: run.font.name='Times New Roman'; run.font.size=Pt(8); run.bold=i==0
    footer=section.footer.paragraphs[0]; footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    field=OxmlElement('w:fldSimple'); field.set(qn('w:instr'),'PAGE'); footer._p.append(field)
    if two_columns:
        final=doc.element.body.find(qn('w:sectPr'))
        initial=deepcopy(final); columns(initial,1)
        kind=initial.find(qn('w:type'))
        if kind is None: kind=OxmlElement('w:type'); initial.insert(0,kind)
        kind.set(qn('w:val'),'continuous')
        boundary=OxmlElement('w:p'); ppr=OxmlElement('w:pPr'); boundary.append(ppr); ppr.append(initial)
        doc.paragraphs[abstract]._p.addprevious(boundary)
        columns(final,2)
        kind=final.find(qn('w:type'))
        if kind is None: kind=OxmlElement('w:type'); final.insert(0,kind)
        kind.set(qn('w:val'),'continuous')
        # End the two-column body explicitly so LibreOffice balances its final page.
        end=OxmlElement('w:p'); endpr=OxmlElement('w:pPr');end.append(endpr)
        endpr.append(deepcopy(final));doc.element.body.insert(len(doc.element.body)-1,end)
        columns(final,1)
    doc.save(out)
    after=inspect(out)
    assert before==after, {k:before[k]==after[k] for k in before}
    Path(str(out)+'.preservation.json').write_text(json.dumps({'passed':True,
        'prose_equations_media_links_tables_preserved':True, 'math_objects':len(after['math']),
        'tables':after['tables'], 'images':len(after['media']), 'columns':2 if two_columns else 1},indent=2))


if __name__=='__main__': format_doc(Path(sys.argv[1]),Path(sys.argv[2]),len(sys.argv)<4 or sys.argv[3]!='single')
