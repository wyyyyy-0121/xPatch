"""Check a revised working manuscript without opening any market data."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from xml.etree import ElementTree as ET
from zipfile import ZipFile


here = Path(__file__).resolve().parent
project = here.parents[3]
source = here / "PAPER_DRAFT_EN_REVISED.md"
previous = here / "PAPER_DRAFT_EN.md"
raw = here / "REVISED_RAW.docx"
formatted = here / "REVISED_FORMATTED.docx"
old_raw = here / "RAW_PANDOC_WORKING_DRAFT.docx"
pdf = here / "render/REVISED_FORMATTED.pdf"
figure = project / "outputs/multiscale/paper_artifacts/chronological_split_schematic.png"
ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
      "m": "http://schemas.openxmlformats.org/officeDocument/2006/math"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def package(path):
    with ZipFile(path) as archive:
        document = ET.fromstring(archive.read("word/document.xml"))
        relations = ET.fromstring(archive.read("word/_rels/document.xml.rels"))
        media = {hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()
                 if name.startswith("word/media/") and not name.endswith("/")}
    return document, relations, media


def maths(document):
    return [[(item.tag, tuple(sorted(item.attrib.items())), item.text or "") for item in equation.iter()]
            for equation in document.findall(".//m:oMath", ns)]


text = source.read_text(encoding="utf8")
old = previous.read_text(encoding="utf8")
body, bibliography = text.split("## References\n", 1)
require(bibliography == old.split("## References\n", 1)[1], "Bibliography changed unexpectedly")
citations = list(dict.fromkeys(map(int, re.findall(r"\[(\d+)\]", body))))
references = [int(n) for n in re.findall(r"(?m)^(\d+)\. ", bibliography)]
require(citations == references == list(range(1, 18)), "Reference order or citation coverage differs")
urls = {url.rstrip(".,;") for url in re.findall(r"https?://\S+", bibliography)}
ast = json.loads(subprocess.run(["pandoc", "-f", "markdown+tex_math_single_backslash+autolink_bare_uris",
                                 "-t", "json", str(source)], capture_output=True, text=True, check=True).stdout)
counts = {}


def walk(value):
    if isinstance(value, dict):
        if "t" in value:
            counts[value["t"]] = counts.get(value["t"], 0) + 1
        for item in value.values():
            walk(item)
    elif isinstance(value, list):
        for item in value:
            walk(item)


walk(ast)
doc, relations, media = package(formatted)
raw_doc, _, _ = package(raw)
old_doc, _, _ = package(old_raw)
doc_text = [item.text or "" for item in doc.findall(".//w:t", ns)]
raw_text = [item.text or "" for item in raw_doc.findall(".//w:t", ns)]
require(doc_text == raw_text and maths(doc) == maths(raw_doc), "Word formatting altered text or equations")
require(maths(doc) == maths(old_doc) and len(maths(doc)) == counts.get("Math") == 30,
        "An original editable equation changed or was lost")
require(len(doc.findall(".//w:tbl", ns)) == counts.get("Table") == 1, "Task table differs")
require(len(doc.findall(".//w:drawing", ns)) == counts.get("Image") == 1 and digest(figure) in media,
        "Figure is missing or differs from the original")
links = {item.attrib["Target"] for item in relations if item.attrib.get("TargetMode") == "External"}
require(len(urls) == 18 and urls <= links, "A reference hyperlink is missing")
paragraphs = doc.findall(".//w:body/w:p", ns)
matches = [p for p in paragraphs if "its step weight is" in "".join(x.text or "" for x in p.findall(".//w:t", ns))]
require(len(matches) == 1 and matches[0].find("./w:pPr/w:keepNext", ns) is not None,
        "Shared-loss introduction no longer stays with its equation")
pdf_text = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True,
                          text=True, check=True).stdout
pages = [page for page in pdf_text.split("\f") if page.strip()]
for marker in ("Working manuscript: not ready for submission.", "Data and code availability",
               "Not yet run", "References", "first and last target sessions"):
    require(marker in pdf_text, "Rendered draft is missing: " + marker)
require(len(pages) >= 1 and all(len(page.strip()) > 100 for page in pages), "A rendered page is empty")
sources = ("multiscale/features.py", "multiscale/development.py", "multiscale/training.py",
           "multiscale/evaluation.py", "multiscale/aggregate_development.py", "multiscale/report_development.py")
receipt = {"status": "revised_working_draft_checked", "created_at_utc": datetime.now(timezone.utc).isoformat(),
           "ready_for_submission": False, "market_data_read": False, "formal_results_added": False,
           "source_sha256": digest(source), "docx_sha256": digest(formatted), "pdf_sha256": digest(pdf),
           "previous_source_sha256": digest(previous), "figure_sha256": digest(figure),
           "raw_docx_sha256": digest(raw), "source_code_sha256": {name: digest(project / name) for name in sources},
           "checks": {"references_in_first_citation_order": len(references), "reference_urls_linked": len(urls),
                      "editable_equations_unchanged": len(maths(doc)), "task_tables": counts["Table"],
                      "embedded_figures": counts["Image"], "rendered_pages": len(pages),
                      "word_text_and_equations_preserved_after_formatting": True,
                      "bibliography_unchanged": True, "draft_and_unrun_results_labeled": True},
           "limitations": ["This is a provisional working draft, not a journal-formatted submission.",
                           "No approved market data, actual results, or publication rights are established.",
                           "The final title, authors, venue, citations, results and availability statement remain open."]}
with (here / "STAGED_MANUSCRIPT_VERIFICATION.json").open("x", encoding="utf8") as output:
    json.dump(receipt, output, indent=2, allow_nan=False)
print(json.dumps({"status": receipt["status"], "checks": receipt["checks"]}))
