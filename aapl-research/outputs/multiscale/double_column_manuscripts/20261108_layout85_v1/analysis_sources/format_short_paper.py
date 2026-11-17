"""Format a compact Word paper, preserving its prose, math, tables and figure."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import sys
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Mm, Pt


def inspect(path):
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
          "m": "http://schemas.openxmlformats.org/officeDocument/2006/math"}
    with ZipFile(path) as archive:
        body = ET.fromstring(archive.read("word/document.xml"))
        rels = ET.fromstring(archive.read("word/_rels/document.xml.rels"))
        return {"text": [x.text or "" for x in body.findall(".//w:t", ns)],
                "math": ["".join(x.itertext()) for x in body.findall(".//m:oMath", ns)],
                "links": sorted(x.attrib.get("Target", "") for x in rels if x.attrib.get("Type", "").endswith("/hyperlink")),
                "media": sorted(hashlib.sha256(archive.read(n)).hexdigest() for n in archive.namelist() if n.startswith("word/media/")),
                "tables": len(body.findall(".//w:tbl", ns))}


def columns(properties, number):
    old = properties.find(qn("w:cols"))
    if old is not None:
        properties.remove(old)
    element = OxmlElement("w:cols")
    element.set(qn("w:num"), str(number))
    element.set(qn("w:space"), "288")  # 0.2 inches
    element.set(qn("w:equalWidth"), "1")
    properties.append(element)


def continuous(properties):
    element = properties.find(qn("w:type"))
    if element is None:
        element = OxmlElement("w:type")
        properties.insert(0, element)
    element.set(qn("w:val"), "continuous")


def section_before(element, properties):
    paragraph = OxmlElement("w:p")
    ppr = OxmlElement("w:pPr")
    spacing = OxmlElement("w:spacing")
    spacing.set(qn("w:before"), "0")
    spacing.set(qn("w:after"), "0")
    spacing.set(qn("w:line"), "20")
    spacing.set(qn("w:lineRule"), "exact")
    ppr.append(spacing)
    ppr.append(deepcopy(properties))
    paragraph.append(ppr)
    element.addprevious(paragraph)


def main():
    raw, output = (Path(p) for p in sys.argv[1:3])
    if output.exists():
        raise ValueError("Preserve previous compact Word releases")
    before = inspect(raw)
    if before["tables"] != 2 or len(before["media"]) != 1 or len(before["math"]) < 12:
        raise ValueError("Unexpected compact-paper structure")
    info = json.loads((raw.parent / "MAIN_PAPER_SOURCE_AUDIT.json").read_text())
    doc = Document(raw)
    section = doc.sections[0]
    section.page_width, section.page_height = Mm(210), Mm(297)
    section.left_margin = section.right_margin = Mm(16)
    section.top_margin = Mm(16)
    section.bottom_margin = Mm(16)
    section.footer_distance = Mm(7)
    section.header_distance = Mm(6)
    for name in ("Normal", "Body Text", "First Paragraph", "Compact"):
        if name in doc.styles:
            style = doc.styles[name]
            style.font.name = "Times New Roman"
            style.font.size = Pt(10)
            style.paragraph_format.space_before = Pt(0)
            style.paragraph_format.space_after = Pt(3)
            style.paragraph_format.line_spacing = 1.0
            style.paragraph_format.widow_control = True
    for name, size in (("Heading 1", 16), ("Heading 2", 10.5), ("Heading 3", 10)):
        style = doc.styles[name]
        style.font.name, style.font.size = "Times New Roman", Pt(size)
        style.font.bold = True
        style.paragraph_format.space_before = Pt(7)
        style.paragraph_format.space_after = Pt(3)
        style.paragraph_format.keep_with_next = True
    for name in ("Image Caption", "Caption"):
        if name in doc.styles:
            doc.styles[name].font.name = "Times New Roman"
            doc.styles[name].font.size = Pt(9)
            doc.styles[name].paragraph_format.space_after = Pt(4)
    abstract_index = next(i for i, p in enumerate(doc.paragraphs) if p.text == "Abstract")
    for i, p in enumerate(doc.paragraphs):
        p.paragraph_format.widow_control = True
        if i < abstract_index:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.keep_with_next = True
            if i == 0:
                for run in p.runs:
                    run.font.size = Pt(16)
                    run.bold = True
            elif p.text.startswith("LAYOUT PREVIEW") or p.text.startswith("Historical development study"):
                for run in p.runs:
                    run.font.size = Pt(8)
                    run.italic = True
            else:
                for run in p.runs:
                    run.font.size = Pt(10)
        elif p.style.name not in ("Heading 1", "Heading 2", "Heading 3"):
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    in_references = False
    for p in doc.paragraphs:
        if p.text == "References":
            in_references = True
            continue
        if in_references:
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.left_indent = Mm(3)
            p.paragraph_format.first_line_indent = Mm(-3)
            for run in p.runs:
                run.font.name, run.font.size = "Times New Roman", Pt(9)
    # Format hyperlink runs too, as python-docx does not include them in p.runs.
    for run in doc.element.body.iter(qn("w:r")):
        rpr = run.find(qn("w:rPr"))
        if rpr is None:
            rpr = OxmlElement("w:rPr")
            run.insert(0, rpr)
        fonts = rpr.find(qn("w:rFonts"))
        if fonts is None:
            fonts = OxmlElement("w:rFonts")
            rpr.insert(0, fonts)
        fonts.set(qn("w:ascii"), "Times New Roman")
        fonts.set(qn("w:hAnsi"), "Times New Roman")
    width = (210 - 32 - 5.08)/2/25.4
    for table in doc.tables:
        table.autofit = False
        if "Table Grid" in doc.styles:
            table.style = "Table Grid"
        count = len(table.columns)
        proportions = [0.32, 0.24, 0.22, 0.22] if count == 4 else [0.31, 0.34, 0.35]
        for index, column in enumerate(table.columns):
            column.width = Inches(width*proportions[index])
        for number, row in enumerate(table.rows):
            properties = row._tr.get_or_add_trPr()
            properties.append(OxmlElement("w:cantSplit"))
            if number == 0:
                properties.append(OxmlElement("w:tblHeader"))
            for index, cell in enumerate(row.cells):
                cell.width = Inches(width*proportions[index])
                margins = cell._tc.get_or_add_tcPr().find(qn("w:tcMar"))
                if margins is None:
                    margins = OxmlElement("w:tcMar")
                    cell._tc.get_or_add_tcPr().append(margins)
                for side in ("top", "bottom", "left", "right"):
                    edge = OxmlElement("w:"+side)
                    edge.set(qn("w:w"), "32")
                    edge.set(qn("w:type"), "dxa")
                    margins.append(edge)
                for p in cell.paragraphs:
                    p.paragraph_format.space_before = p.paragraph_format.space_after = Pt(0)
                    p.paragraph_format.line_spacing = 1.0
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT if index == 0 else WD_ALIGN_PARAGRAPH.RIGHT
                    for run in p.runs:
                        run.font.name, run.font.size = "Times New Roman", Pt(8)
                        run.bold = number == 0
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.text = (f"LAYOUT PREVIEW | {info['completed_folds']}/100 FOLDS | " if info["layout_preview"] else
                   "HISTORICAL DEVELOPMENT | ")
    if not info["authors_ready"]:
        footer.add_run("AUTHOR DETAILS PENDING | ")
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    for run in footer.runs:
        run.font.name, run.font.size = "Times New Roman", Pt(7)
    # The byline spans the page; narrative text uses two equal columns.
    tail = doc.element.body.sectPr
    single = deepcopy(tail)
    columns(single, 1)
    continuous(single)
    double = deepcopy(single)
    columns(double, 2)
    section_before(doc.paragraphs[abstract_index]._p, single)
    # The compact scientific figure spans both columns, at its native width.
    fig = next(p for p in doc.paragraphs if p._p.xpath('.//w:drawing'))
    # Paragraph wrappers are recreated by python-docx, so locate by XML identity.
    paragraphs = list(doc.paragraphs)
    index = next(i for i, p in enumerate(paragraphs) if p._p is fig._p)
    caption = paragraphs[index+1] if index+1 < len(paragraphs) and paragraphs[index+1].text.startswith("Figure 1.") else fig
    fig.paragraph_format.keep_with_next = caption is not fig
    fig.paragraph_format.space_before = Pt(2)
    fig.paragraph_format.space_after = Pt(2)
    fig.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.paragraph_format.keep_together = True
    caption.paragraph_format.space_after = Pt(5)
    section_before(fig._p, double)
    after = caption._p.getnext()
    if after is None:
        raise ValueError("Missing narrative after figure")
    section_before(after, single)
    doc.element.body.replace(tail, deepcopy(double))
    if len(doc.inline_shapes) != 1:
        raise ValueError("Expected one scientific figure")
    shape = doc.inline_shapes[0]
    old_width = shape.width
    shape.width = Inches(7.0)
    shape.height = int(shape.height*shape.width/old_width)
    doc.save(output)
    after_content = inspect(output)
    if before != after_content:
        raise ValueError("Layout changed prose, mathematics, tables, links or figure bytes")
    doc = Document(output)
    cols = [int(s._sectPr.find(qn("w:cols")).get(qn("w:num"), "1")) for s in doc.sections]
    if cols != [1, 2, 1, 2]:
        raise ValueError("Title / body / figure / body column structure differs")
    result = {"status": "compact_double_column_word_formatted", "body_content_preserved": True,
              "columns": cols, "paper": "A4 portrait", "body_font": "Times New Roman 10 pt",
              "editable_math_objects": len(after_content["math"]), "hyperlinks": len(after_content["links"]),
              "tables": after_content["tables"], "figures": len(after_content["media"]),
              "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
              "formatted_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output.parent / "DOUBLE_COLUMN_LAYOUT_AUDIT.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
