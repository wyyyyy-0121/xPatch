"""Format the complete Word manuscript without changing body content."""
from pathlib import Path
import hashlib
import json
import sys
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt


def inspect(path):
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
          "m": "http://schemas.openxmlformats.org/officeDocument/2006/math"}
    with ZipFile(path) as archive:
        body = ET.fromstring(archive.read("word/document.xml"))
        relations = ET.fromstring(archive.read("word/_rels/document.xml.rels"))
        return {"text": [x.text or "" for x in body.findall(".//w:t", ns)],
                "math": ["".join(x.itertext()) for x in body.findall(".//m:oMath", ns)],
                "links": sorted(x.attrib.get("Target", "") for x in relations if x.attrib.get("Type", "").endswith("/hyperlink")),
                "media": sorted(hashlib.sha256(archive.read(n)).hexdigest() for n in archive.namelist() if n.startswith("word/media/")),
                "tables": len(body.findall(".//w:tbl", ns))}


def main():
    raw, destination = (Path(p) for p in sys.argv[1:3])
    if destination.exists():
        raise ValueError("Preserve previous Word releases")
    before = inspect(raw)
    if (len(before["math"]), len(before["links"]), len(before["media"]), before["tables"]) != (30, 18, 2, 37):
        raise ValueError("Unexpected complete-manuscript structure")
    document = Document(raw)
    for name in ("Normal", "Body Text", "First Paragraph", "Compact"):
        if name in document.styles:
            style = document.styles[name]
            style.font.name = "Arial"
            style.font.size = Pt(10)
            style.paragraph_format.space_before = Pt(0)
            style.paragraph_format.space_after = Pt(3)
            style.paragraph_format.line_spacing = 1.05
    for name in ("Heading 1", "Heading 2", "Heading 3"):
        if name in document.styles:
            style = document.styles[name]
            style.font.name = "Arial"
            style.paragraph_format.space_before = Pt(6)
            style.paragraph_format.space_after = Pt(3)
            style.paragraph_format.keep_with_next = True
    for section in document.sections:
        section.orientation = WD_ORIENT.LANDSCAPE
        section.page_width, section.page_height = Inches(11.69), Inches(8.27)
        section.left_margin = section.right_margin = Inches(0.6)
        section.top_margin = section.bottom_margin = Inches(0.55)
        section.footer_distance = Inches(0.25)
        footer = section.footer.paragraphs[0]
        footer.text = "HISTORICAL DEVELOPMENT STUDY | "
        for run in footer.runs:
            run.font.name, run.font.size = "Arial", Pt(8)
        field = OxmlElement("w:fldSimple")
        field.set(qn("w:instr"), "PAGE")
        footer._p.append(field)
    for table in document.tables:
        table.autofit = False
        if "Table Grid" in document.styles:
            table.style = "Table Grid"
        columns = len(table.rows[0].cells)
        widths = ([1.55, 0.8] + [8.14 / (columns - 2)] * (columns - 2)) if columns >= 5 else [10.49 / columns] * columns
        if columns == 5 and table.rows[0].cells[1].text == "Mean seed endpoint MSE":
            widths = [1.55, 2.2, 2.0, 2.4, 2.34]
        for index, column in enumerate(table.columns):
            column.width = Inches(widths[index])
        for number, row in enumerate(table.rows):
            properties = row._tr.get_or_add_trPr()
            properties.append(OxmlElement("w:cantSplit"))
            if number == 0:
                header = OxmlElement("w:tblHeader")
                header.set(qn("w:val"), "true")
                properties.append(header)
            for index, cell in enumerate(row.cells):
                cell.width = Inches(widths[index])
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.space_before = Pt(0)
                    paragraph.paragraph_format.space_after = Pt(0)
                    for run in paragraph.runs:
                        run.font.name, run.font.size = "Arial", Pt(8)
                        if number == 0:
                            run.bold = True
    document.save(destination)
    after = inspect(destination)
    if after != before:
        raise ValueError("Formatting changed body text, mathematics, links, images or table count")
    print(json.dumps({"status": "complete_manuscript_word_formatted", "body_content_preserved": True,
                      "editable_equations": len(after["math"]), "hyperlinks": len(after["links"]),
                      "embedded_media": len(after["media"]), "tables": after["tables"],
                      "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
