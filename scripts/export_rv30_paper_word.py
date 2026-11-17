"""Format a Pandoc DOCX export as the project's two-column Word manuscript.

Requires python-docx. Takes a saved content conversion; no model or data calls.
The accompanying author.json supplies the verified school and blank email.
"""
import argparse
import hashlib
import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt


ROOT = Path(__file__).resolve().parents[1]
CAPTIONS = (
    'Table 1. Historical regular-session bar coverage.',
    'Table 2. Registered search sizes per information set and outer fold. Configurations have model-specific computing costs.',
    'Table 3. Development results, 2023Q1–2025Q2. Improvements are relative to the corresponding HAR. All gates refers to the joint qualification rule.',
    'Table 4. Annual-segment relative QLIKE improvements versus matching HAR; improving quarter counts and the prescribed seed rule.',
    'Table 5. Development empirical coverage and mean width. Widths are on the RV scale, multiplied by 10⁵; quantities use equal session weights.',
    'Table 6. Daily-mean calibration regressions. Intercepts are multiplied by 10⁵; slope standard errors use the saved five-lag HAC covariance. No new joint test is performed.',
    'Table 7. Historical confirmation, 2025H2–2026H1. Intervals are five-session block intervals for Ridge minus HAR.',
)


def set_value(parent, tag, **attrs):
    item = parent.find(qn('w:' + tag))
    if item is None:
        item = OxmlElement('w:' + tag)
        parent.append(item)
    for name, value in attrs.items():
        item.set(qn('w:' + name), str(value))
    return item


def set_run(run, size=10, bold=None):
    run.font.name = 'Times New Roman'
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    set_value(run._r.get_or_add_rPr(), 'rFonts', ascii='Times New Roman', hAnsi='Times New Roman', eastAsia='Times New Roman')


def section_properties(columns):
    sect = OxmlElement('w:sectPr')
    set_value(sect, 'type', val='continuous')
    set_value(sect, 'pgSz', w=11906, h=16838)
    set_value(sect, 'pgMar', top=1020, right=1020, bottom=1020, left=1020, header=450, footer=450, gutter=0)
    set_value(sect, 'cols', num=columns, space=340, equalWidth=1)
    return sect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    doc = Document(args.input)
    # Pandoc adds an implicit caption from each Markdown image's alt text.
    # Keep the explicit numbered captions supplied by the manuscript instead.
    for paragraph in list(doc.paragraphs):
        if paragraph.style.name == 'Image Caption' or (
            paragraph.style.name == 'Caption' and
            not paragraph.text.startswith(('Figure ', 'Table '))
        ):
            paragraph._element.getparent().remove(paragraph._element)
    author = json.loads((ROOT/'research/paper/author.json').read_text(encoding='utf-8'))
    doc.core_properties.author = author['name']
    doc.core_properties.last_modified_by = author['name']
    doc.core_properties.title = doc.paragraphs[0].text
    normal = doc.styles['Normal']
    normal.font.name = 'Times New Roman'
    normal.font.size = Pt(10)
    normal.paragraph_format.space_after = Pt(3)
    normal.paragraph_format.line_spacing = Pt(11.5)
    for name in ('Body Text', 'First Paragraph'):
        if name in doc.styles:
            doc.styles[name].base_style = normal
            doc.styles[name].font.name = 'Times New Roman'
            doc.styles[name].font.size = Pt(10)
            doc.styles[name].paragraph_format.line_spacing = Pt(11.5)
            doc.styles[name].paragraph_format.space_after = Pt(3)
    for name, size in (('Heading 1',11),('Heading 2',10.5),('Heading 3',10)):
        style = doc.styles[name]
        style.font.name = 'Times New Roman'
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = None
        style.paragraph_format.space_before = Pt(7)
        style.paragraph_format.space_after = Pt(4)
        style.paragraph_format.keep_with_next = True
    if 'Caption' in doc.styles:
        doc.styles['Caption'].font.name = 'Times New Roman'
        doc.styles['Caption'].font.size = Pt(8)

    introduction = next(p for p in doc.paragraphs if p.text.startswith('1. Introduction'))
    head = True
    references = False
    captions = []
    for index, p in enumerate(doc.paragraphs):
        if p._p is introduction._p:
            head = False
        p.paragraph_format.widow_control = True
        p.paragraph_format.line_spacing = Pt(11.5)
        p.paragraph_format.space_after = Pt(3)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        if head:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.first_line_indent = Mm(0)
            if index == 0:
                p.style = doc.styles['Normal']
                for run in p.runs:
                    set_run(run, 15, True)
                p.paragraph_format.line_spacing = Pt(17)
                p.paragraph_format.space_after = Pt(6)
            elif p.text == author['name']:
                for run in p.runs:
                    set_run(run, 11, True)
                marker=p.add_run('1,*')
                marker.font.superscript=True
                set_run(marker,8)
            elif p.text == author['school']:
                for run in p.runs:
                    set_run(run, 10)
                marker=p.add_run('1')
                marker.font.superscript=True
                set_run(marker,8)
                p._p.insert(1,marker._r)
            elif p.text.startswith('Corresponding author'):
                p.clear()
                run=p.add_run("*Corresponding author's e-mail: ")
                set_run(run,9)
                blank=p.add_run('\u00a0' * 28)
                set_run(blank,9)
                blank.underline=True
            elif p.text == 'Abstract':
                p.paragraph_format.space_before=Pt(5)
                for run in p.runs:
                    set_run(run,10,True)
            else:
                for run in p.runs:
                    set_run(run,9)
            if p.text.startswith(('Short-horizon variance forecasts', 'Keywords:')):
                p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
        elif p.style.name.startswith('Heading'):
            p.alignment=WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.keep_with_next=True
        else:
            p.paragraph_format.first_line_indent=Mm(3)
        if p.text == 'References':
            references=True
        elif references:
            p.paragraph_format.first_line_indent=Mm(0)
            p.paragraph_format.space_after=Pt(3)
            p.paragraph_format.line_spacing=Pt(9.5)
            for run in p.runs:
                set_run(run,8.5)
        if p._p.find('.//' + qn('w:drawing')) is not None:
            p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.first_line_indent=Mm(0)
            # Exact text line heights clip tall inline figures in Word/Writer.
            p.paragraph_format.line_spacing=1.0
            p.paragraph_format.keep_with_next=True
        if p.text.startswith(('Figure 1.', 'Figure 2.', 'Figure 3.')):
            captions.append(p._p)
            p.paragraph_format.first_line_indent=Mm(0)
            p.paragraph_format.line_spacing=Pt(9.5)
            p.paragraph_format.space_after=Pt(5)
            for run in p.runs:
                set_run(run,8)

    # Use the official school, without inventing a department for a high school.
    assert sum(p.text.lstrip('1')==author['school'] for p in doc.paragraphs)==1
    assert any(p.text=='Shanghai 200439, China' for p in doc.paragraphs)
    for shape in doc.inline_shapes:
        ratio=shape.height/shape.width
        shape.width=Mm(169)
        shape.height=int(shape.width*ratio)

    table_elements=[]
    widths=((13,16,16,14,24),(20,14,49),(16,23,18,20,46,21,25),
        (17,24,26,26,29,26,21),(18,24,26,38,26,37),(28,20,23,38,26,34),
        (21,23,23,23,22,57))
    for number, table in enumerate(doc.tables):
        table_elements.append(table._tbl)
        table.alignment=WD_TABLE_ALIGNMENT.CENTER
        table.autofit=False
        desired=widths[number]
        if number==1:
            table.rows[0].cells[1].text='Configs.'
        for col,width in zip(table.columns,desired):col.width=Mm(width)
        for row_index,row in enumerate(table.rows):
            set_value(row._tr.get_or_add_trPr(),'cantSplit')
            if row_index==0:
                set_value(row._tr.get_or_add_trPr(),'tblHeader')
            for cell,width in zip(row.cells,desired):
                cell.width=Mm(width)
                cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                tcpr=cell._tc.get_or_add_tcPr()
                margins=set_value(tcpr,'tcMar')
                for side in ('top','left','bottom','right'):set_value(margins,side,w=45,type='dxa')
                for p in cell.paragraphs:
                    p.paragraph_format.first_line_indent=Mm(0)
                    p.paragraph_format.space_after=Pt(1)
                    p.paragraph_format.space_before=Pt(1)
                    p.paragraph_format.line_spacing=Pt(9.2)
                    p.paragraph_format.keep_with_next=(row_index<len(table.rows)-1)
                    p.alignment=WD_ALIGN_PARAGRAPH.LEFT
                    for run in p.runs:set_run(run,8,bold=row_index==0)
        # Add a clear caption without changing table cell values.
        caption=OxmlElement('w:p')
        props=OxmlElement('w:pPr');caption.append(props)
        set_value(props,'keepNext')
        set_value(props,'spacing',before=60,after=60,line=190,lineRule='exact')
        run=OxmlElement('w:r');caption.append(run)
        rpr=OxmlElement('w:rPr');run.append(rpr)
        set_value(rpr,'rFonts',ascii='Times New Roman',hAnsi='Times New Roman')
        set_value(rpr,'sz',val=16)
        text=OxmlElement('w:t');text.text=CAPTIONS[number];run.append(text)
        table._tbl.addprevious(caption)
        if number>=2:captions.append(caption)

    # Continuous sections retain a two-column body and full-width wide items.
    body=doc._element.body
    old_section=body.find(qn('w:sectPr'))
    if old_section is not None:body.remove(old_section)
    mode=1
    heading_seen=False
    wide_tables=set(table_elements[2:])
    for element in list(body):
        if element is introduction._p:heading_seen=True
        wanted=2 if heading_seen else 1
        if element in wide_tables or element in captions or element.find('.//'+qn('w:drawing')) is not None:
            wanted=1
        if wanted!=mode:
            boundary=OxmlElement('w:p')
            props=OxmlElement('w:pPr');boundary.append(props)
            set_value(props,'spacing',before=0,after=0,line=20,lineRule='exact')
            props.append(section_properties(mode))
            element.addprevious(boundary)
            mode=wanted
    body.append(section_properties(mode))
    footer=doc.sections[0].footer.paragraphs[0]
    footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    run=footer.add_run()
    set_run(run,8)
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE')
    run._r.addnext(field)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    doc.save(args.output)
    print(json.dumps({'passed':True,'output':str(args.output),'tables':len(doc.tables),
        'images':len(doc.inline_shapes),'body_columns':2,'email_intentionally_blank':True,
        'bytes':args.output.stat().st_size,'sha256':hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__=='__main__':
    main()
