"""Format a generated private appendix without changing its body text."""
import hashlib,json,sys
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches,Pt


def body_text(path):
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    with ZipFile(path) as z:root=ET.fromstring(z.read('word/document.xml'))
    return [x.text or '' for x in root.findall('.//w:t',ns)]


def main():
    raw,dest=(Path(x) for x in sys.argv[1:3])
    if dest.exists():raise ValueError('Destination already exists')
    doc=Document(raw)
    for name in ('Normal','Body Text','First Paragraph','Compact'):
        if name in doc.styles:
            style=doc.styles[name];style.font.name='Arial';style.font.size=Pt(9)
            style.paragraph_format.space_before=Pt(0);style.paragraph_format.space_after=Pt(2)
            style.paragraph_format.line_spacing=1.0
    for name in ('Heading 1','Heading 2'):
        if name in doc.styles:
            style=doc.styles[name];style.paragraph_format.space_before=Pt(4);style.paragraph_format.space_after=Pt(2)
    for section in doc.sections:
        section.orientation=WD_ORIENT.LANDSCAPE;section.page_width=Inches(11.69);section.page_height=Inches(8.27)
        section.left_margin=section.right_margin=Inches(0.5);section.top_margin=section.bottom_margin=Inches(0.55)
        footer=section.footer.paragraphs[0];footer.text='PRIVATE DRAFT — NOT FOR SUBMISSION'
        for run in footer.runs:run.font.size=Pt(8);run.bold=True
    for table in doc.tables:
        table.autofit=False
        if 'Table Grid' in doc.styles:table.style='Table Grid'
        columns=len(table.rows[0].cells)
        if columns>=3:
            widths=[1.5,0.8]+[(10.55-2.3)/(columns-2)]*(columns-2)
        else:widths=[3.8,6.75]
        for row_index,row in enumerate(table.rows):
            trpr=row._tr.get_or_add_trPr();cant=OxmlElement('w:cantSplit');trpr.append(cant)
            if row_index==0:
                header=OxmlElement('w:tblHeader');header.set(qn('w:val'),'true');trpr.append(header)
            for col,cell in enumerate(row.cells):
                cell.width=Inches(widths[min(col,len(widths)-1)])
                for para in cell.paragraphs:
                    para.paragraph_format.space_before=Pt(0);para.paragraph_format.space_after=Pt(0)
                    for run in para.runs:
                        run.font.name='Arial';run.font.size=Pt(8)
                        if row_index==0:run.bold=True
    doc.save(dest)
    if body_text(raw)!=body_text(dest):raise ValueError('Body text changed during formatting')
    print(json.dumps({'status':'private_appendix_formatted','tables':len(doc.tables),
                      'body_text_preserved':True,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}))


if __name__=='__main__':main()
