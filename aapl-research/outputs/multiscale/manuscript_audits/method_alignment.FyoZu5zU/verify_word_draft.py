"""Verify the editable working draft without opening market data."""
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import subprocess
from xml.etree import ElementTree as ET
from zipfile import ZipFile


HERE = Path(__file__).resolve().parent
OUT = HERE.parent
SOURCE = OUT / "PAPER_DRAFT_EN.md"
DOCX = OUT / "AAPL_MULTISCALE_WORKING_DRAFT.docx"
RAW = HERE / "RAW_PANDOC_WORKING_DRAFT.docx"
FIGURE = OUT / "paper_artifacts/chronological_split_schematic.png"
PDF = OUT / "paper_artifacts/word_render.dBpbeHyI/AAPL_MULTISCALE_WORKING_DRAFT.pdf"
RECEIPT = HERE / "WORD_DRAFT_VERIFICATION.json"
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
      "m": "http://schemas.openxmlformats.org/officeDocument/2006/math"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package(path):
    with ZipFile(path) as archive:
        document = ET.fromstring(archive.read("word/document.xml"))
        relations = ET.fromstring(archive.read("word/_rels/document.xml.rels"))
        media = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()
                 if name.startswith("word/media/") and not name.endswith("/")}
    return document, relations, media


def math_contents(document):
    return [[(element.tag, tuple(sorted(element.attrib.items())), element.text or "")
             for element in equation.iter()] for equation in document.findall(".//m:oMath", NS)]


def main():
    source = SOURCE.read_text(encoding="utf8")
    body, references = source.split("## References\n", 1)
    cited = list(dict.fromkeys(map(int, re.findall(r"\[(\d+)\]", body))))
    listed = [int(number) for number in re.findall(r"(?m)^(\d+)\. ", references)]
    require(cited == listed == list(range(1, 18)), "Citation and bibliography order differs")
    urls = {url.rstrip(".,;") for url in re.findall(r"https?://\S+", references)}
    require(len(urls) == 18, "Reference URL count differs")
    ast = json.loads(subprocess.run(["pandoc", "-f", "markdown+tex_math_single_backslash+autolink_bare_uris",
                                     "-t", "json", str(SOURCE)], check=True, capture_output=True, text=True).stdout)
    counts = {}
    def walk(node):
        if isinstance(node, dict):
            if "t" in node:
                counts[node["t"]] = counts.get(node["t"], 0) + 1
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
    walk(ast)
    document, relations, media = package(DOCX)
    raw, _, _ = package(RAW)
    word_text = [element.text or "" for element in document.findall(".//w:t", NS)]
    raw_text = [element.text or "" for element in raw.findall(".//w:t", NS)]
    require(word_text == raw_text and math_contents(document) == math_contents(raw),
            "Word formatting changed text or equations")
    require(len(math_contents(document)) == counts.get("Math") == 30, "Equation count differs")
    require(len(document.findall(".//w:tbl", NS)) == counts.get("Table") == 1, "Task table is missing")
    require(len(document.findall(".//w:drawing", NS)) == counts.get("Image") == 1 and
            digest(FIGURE) in media.values(), "Embedded figure differs from the source")
    links = {item.attrib["Target"] for item in relations if item.attrib.get("TargetMode") == "External"}
    require(urls <= links, "A reference URL is not linked in the Word document")
    matching = []
    for paragraph in document.findall(".//w:body/w:p", NS):
        text = "".join(element.text or "" for element in paragraph.findall(".//w:t", NS))
        if "its step weight is" in text:
            matching.append(paragraph)
    require(len(matching) == 1 and matching[0].find("./w:pPr/w:keepNext", NS) is not None,
            "Shared-loss equation is not kept with its introduction")
    weights = [sum((Fraction(1, h) for h in (5, 15, 30, 60) if j <= h), Fraction()) / 4
               for j in range(1, 61)]
    require(weights[0] == Fraction(19, 240) and weights[30] == Fraction(1, 240) and
            sum(weights) == 1, "Shared-prefix weights differ")
    text = subprocess.run(["pdftotext", "-layout", str(PDF), "-"], check=True,
                          capture_output=True, text=True).stdout
    pages = [page for page in text.split("\f") if page.strip()]
    for marker in ("Working manuscript: not ready for submission.", "Figure 1.",
                   "Experimental results", "Not yet run", "References"):
        require(marker in text, "Rendered draft is missing: " + marker)
    require(len(pages) == 13 and all(len(page.strip()) > 100 for page in pages), "Unexpected rendered pages")
    receipt = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "working_draft_checked",
               "ready_for_submission": False, "market_data_read": False,
               "source_sha256": digest(SOURCE), "docx_sha256": digest(DOCX), "pdf_sha256": digest(PDF),
               "figure_sha256": digest(FIGURE), "raw_pandoc_docx_sha256": digest(RAW),
               "checks": {"references_in_first_citation_order": len(listed), "reference_urls_linked": len(urls),
                          "editable_equations": len(math_contents(document)), "task_tables": counts["Table"],
                          "embedded_figures": counts["Image"], "rendered_pages": len(pages),
                          "word_text_and_equations_preserved_after_formatting": True,
                          "shared_prefix_weights_verified": True, "draft_and_unrun_results_labeled": True},
               "limitations": ["The PDF is a LibreOffice rendering of a provisional Word draft, not a journal layout.",
                               "Selected pages were visually reviewed; final venue and full citation verification remain open.",
                               "No formal market experiment, data rights, or scientific findings are established by this check."]}
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"status": receipt["status"], "receipt": str(RECEIPT), "checks": receipt["checks"]}))


if __name__ == "__main__":
    main()
