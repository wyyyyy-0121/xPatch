"""Format a private Word draft and verify its text, equations, links and media."""
import hashlib,json,sys
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches,Pt


def inspect(path):
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
        'm':'http://schemas.openxmlformats.org/officeDocument/2006/math',
        'r':'http://schemas.openxmlformats.org/package/2006/relationships'}
    with ZipFile(path) as z:
        root=ET.fromstring(z.read('word/document.xml'))
        rel=ET.fromstring(z.read('word/_rels/document.xml.rels'))
        text=[x.text or '' for x in root.findall('.//w:t',ns)]
        math=[''.join(x.itertext()) for x in root.findall('.//m:oMath',ns)]
        links=sorted(x.attrib.get('Target','') for x in rel if x.attrib.get('Type','').endswith('/hyperlink'))
        media=sorted(hashlib.sha256(z.read(name)).hexdigest() for name in z.namelist() if name.startswith('word/media/'))
        tables=len(root.findall('.//w:tbl',ns))
    return {'text':text,'math':math,'links':links,'media':media,'tables':tables}


def main():
    raw,dest=(Path(x) for x in sys.argv[1:3])
    if dest.exists():raise ValueError('Destination already exists')
    before=inspect(raw)
    if len(before['math'])<30 or len(before['links'])<17 or not before['media'] or before['tables']<8:
        raise ValueError('The source document is missing expected equations, references, figure or tables')
    doc=Document(raw)
    for name in ('Normal','Body Text','First Paragraph','Compact'):
        if name in doc.styles:
            style=doc.styles[name];style.font.name='Arial';style.font.size=Pt(10)
            style.paragraph_format.space_before=Pt(0);style.paragraph_format.space_after=Pt(3)
            style.paragraph_format.line_spacing=1.05
    for name in ('Heading 1','Heading 2','Heading 3'):
        if name in doc.styles:
            style=doc.styles[name];style.paragraph_format.space_before=Pt(6);style.paragraph_format.space_after=Pt(3)
            style.paragraph_format.keep_with_next=True
    for section in doc.sections:
        section.orientation=WD_ORIENT.LANDSCAPE;section.page_width=Inches(11.69);section.page_height=Inches(8.27)
        section.left_margin=section.right_margin=Inches(0.6);section.top_margin=section.bottom_margin=Inches(0.55)
        section.footer_distance=Inches(0.25)
        footer=section.footer.paragraphs[0];footer.text='PRIVATE WORKING DRAFT — NOT FOR SUBMISSION'
        for run in footer.runs:run.font.size=Pt(8);run.bold=True
    for table in doc.tables:
        table.autofit=False
        if 'Table Grid' in doc.styles:table.style='Table Grid'
        count=len(table.rows[0].cells)
        widths=([1.55,0.8]+[8.14/(count-2)]*(count-2)) if count>=5 else [10.49/count]*count
        for index,column in enumerate(table.columns):column.width=Inches(widths[index])
        for row_number,row in enumerate(table.rows):
            trpr=row._tr.get_or_add_trPr();trpr.append(OxmlElement('w:cantSplit'))
            if row_number==0:
                header=OxmlElement('w:tblHeader');header.set(qn('w:val'),'true');trpr.append(header)
            for col,cell in enumerate(row.cells):
                cell.width=Inches(widths[min(col,len(widths)-1)])
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.space_before=Pt(0);paragraph.paragraph_format.space_after=Pt(0)
                    for run in paragraph.runs:
                        run.font.name='Arial';run.font.size=Pt(8)
                        if row_number==0:run.bold=True
    doc.save(dest)
    after=inspect(dest)
    for field in ('text','math','links','media','tables'):
        if before[field]!=after[field]:raise ValueError('Document content changed during formatting: '+field)
    print(json.dumps({'status':'private_working_manuscript_formatted','body_text_preserved':True,
                      'editable_equations':len(after['math']),'hyperlinks':len(after['links']),
                      'embedded_media':len(after['media']),'tables':after['tables'],
                      'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}))


if __name__=='__main__':main()
