"""Verify the rendered private manuscript after the uncertainty insertion."""
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
from xml.etree import ElementTree as ET
from zipfile import ZipFile


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def document_data(path):
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
          "m": "http://schemas.openxmlformats.org/officeDocument/2006/math"}
    with ZipFile(path) as z:
        root = ET.fromstring(z.read("word/document.xml"))
        rel = ET.fromstring(z.read("word/_rels/document.xml.rels"))
        text = [x.text or "" for x in root.findall(".//w:t", ns)]
        math = ["".join(x.itertext()) for x in root.findall(".//m:oMath", ns)]
        links = sorted(x.attrib.get("Target", "") for x in rel
                       if x.attrib.get("Type", "").endswith("/hyperlink"))
        media = sorted(sha256(z.read(name)).hexdigest() for name in z.namelist()
                       if name.startswith("word/media/"))
        tables = []
        for table in root.findall(".//w:tbl", ns):
            rows = []
            for row in table.findall("./w:tr", ns):
                cells = []
                for cell in row.findall("./w:tc", ns):
                    parts = [x.text or "" for x in cell.iter()
                             if x.tag in (f"{{{ns['w']}}}t", f"{{{ns['m']}}}t")]
                    cells.append("".join(parts).strip())
                rows.append(cells)
            tables.append(rows)
        return {"text": text, "math": math, "links": links, "media": media, "tables": tables,
                "drawings": len(root.findall(".//w:drawing", ns)),
                "footers": [name for name in z.namelist() if re.fullmatch(r"word/footer\d+\.xml", name)]}


def markdown_tables(text):
    result, current = [], []
    for line in text.splitlines():
        if line.startswith("|"):
            if re.match(r"^\|\s*:?-+", line):
                continue
            current.append([cell.strip().replace("\\(", "").replace("\\)", "")
                            for cell in line.strip("|").split("|")])
        elif current:
            result.append(current)
            current = []
    if current:
        result.append(current)
    return result


def output(args):
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout


root = Path(sys.argv[1]).resolve()
preview_dir = Path(sys.argv[2]).resolve()
prior_previews = Path("/tmp/aapl_manuscript_review_20261108")
md = root / "PRIVATE_WORKING_MANUSCRIPT_EN.md"
raw = root / "PRIVATE_WORKING_MANUSCRIPT_RAW.docx"
word = root / "PRIVATE_WORKING_MANUSCRIPT_EN.docx"
pdf = root / "PRIVATE_WORKING_MANUSCRIPT_EN.pdf"
audit = json.loads((root / "PRIVATE_ANNUAL_MANUSCRIPT_AUDIT.json").read_text(encoding="utf8"))
interval_check = json.loads((root / "PRIVATE_INTERVAL_INSERTION_VERIFICATION.json").read_text(encoding="utf8"))
reference_check = json.loads((root / "reference_evidence_20261108" /
                              "REFERENCE_EVIDENCE_VERIFICATION.json").read_text(encoding="utf8"))
require(digest(md) == audit["manuscript_sha256"] == interval_check["manuscript_sha256"] ==
        reference_check["manuscript_sha256"], "The manuscript evidence identity differs")
require(interval_check["checks"]["rows"] == 192 and reference_check["checks"]["entries"] == 17,
        "The interval or reference checks are incomplete")

before, after = document_data(raw), document_data(word)
for name in ("text", "math", "links", "media", "tables"):
    require(before[name] == after[name], "Formatting changed " + name)
require(len(after["math"]) == 30 and len(after["links"]) == 18 and
        len(after["tables"]) == 25 and len(after["media"]) == 1 and
        after["drawings"] == 1 and len(after["footers"]) == 1,
        "Formula, hyperlink, table, image or footer counts differ")
figure = Path("outputs/multiscale/paper_artifacts/chronological_split_schematic.png").resolve()
require(after["media"] == [digest(figure)], "Embedded figure differs from the source")
require(markdown_tables(md.read_text(encoding="utf8")) == after["tables"],
        "Markdown and Word table cells differ")

source_text = output(["pandoc", "--from=markdown+tex_math_single_backslash+autolink_bare_uris",
                      "--to=plain", "--wrap=none", str(md)])
word_text = output(["pandoc", "--from=docx", "--to=plain", "--wrap=none", str(word)])
word_re = r"[A-Za-z]{3,}(?:[’'-][A-Za-z]+)?"
source_words, word_words = Counter(re.findall(word_re, source_text)), Counter(re.findall(word_re, word_text))
for command in ("mathrm", "qquad", "quad", "mspace", "leq"):
    source_words.pop(command, None)
    word_words.pop(command, None)
require(source_words == word_words, "Word prose differs after excluding formula syntax")

info = output(["pdfinfo", str(pdf)])
match = re.search(r"^Pages:\s+(\d+)\s*$", info, re.MULTILINE)
layout = output(["pdftotext", "-layout", "-enc", "UTF-8", str(pdf), "-"])
pages = [p for p in layout.split("\f") if p.strip()]
require(match and int(match.group(1)) == len(pages) == 15 and
        all("PRIVATE WORKING DRAFT" in page for page in pages), "Rendered pages or private footers differ")
for phrase in ("68 of 100", "240 neural", "1473.81", "0.00004971", "-303817.03%", "-24.67%",
               "Descriptive uncertainty across completed tasks", "-0.0303697", "0.00529228",
               "-0.0258706", "-0.00546571", "References"):
    require(phrase in layout, "Rendered PDF omits " + phrase)

started = False
titles = []
pattern = r"(?:Annual H12 \(short history\)|Quarterly H3|Monthly H21|Hourly H[148]): (?:nominal close|cumulative log return)"
for page in pages:
    if "Appendix A." in page:
        started = True
    if not started:
        continue
    matches = list(re.finditer(pattern, page))
    for index, item in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(page)
        segment = page[item.end():end]
        require("End MSE" in segment and "xPatch" in segment, "An appendix table is split from its title")
        titles.append(item.group(0))
require(len(titles) == len(set(titles)) == 12, "The detailed appendix task tables are incomplete")
annual_pages = [p for p in pages if "Annual sensitivity: nonoverlapping forecast windows" in p]
require(len(annual_pages) == 1, "Annual sensitivity heading is missing or duplicated")
annual_section = annual_pages[0].split("Annual sensitivity: nonoverlapping forecast windows", 1)[1].split(
    "Descriptive uncertainty across completed tasks", 1)[0]
for phrase in ("Price endpoint MSE", "Return endpoint MSE", "64.53%", "xPatch", "2021-07-30", "2025-07-31"):
    require(phrase in annual_section, "Annual sensitivity table was split or omitted: " + phrase)

preview_hashes = {}
for page in range(1, 16):
    image = preview_dir / f"page-{page:02}.png"
    require(image.is_file(), "Rendered page preview is missing")
    preview_hashes[str(page)] = digest(image)
    if page <= 8:
        require(digest(image) == digest(prior_previews / f"fixed_page_{page}.png"),
                "A previously reviewed page changed")

out = root / "PRIVATE_WORD_VERIFICATION.json"
require(not out.exists(), "Word verification receipt already exists")
receipt = {
    "status": "verified_private_working_draft_not_submission_ready",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "markdown_sha256": digest(md), "raw_docx_sha256": digest(raw),
    "formatted_docx_sha256": digest(word), "rendered_pdf_sha256": digest(pdf),
    "script_sha256": digest(Path(__file__)), "preview_sha256": preview_hashes,
    "checks": {"editable_equations": len(after["math"]), "hyperlinks": len(after["links"]),
               "tables": len(after["tables"]), "body_drawings": after["drawings"],
               "body_text_math_links_media_and_tables_preserved_by_formatting": True,
               "every_markdown_table_cell_matches_word": True,
               "prose_word_counts_match_after_excluding_math_syntax": True,
               "rendered_pages": len(pages), "all_pages_contain_private_footer": True,
               "visual_pages_reviewed": list(range(1, 16)),
               "pages_identical_to_previous_review": list(range(1, 9)),
               "changed_pages_visually_reviewed": list(range(9, 16)),
               "appendix_tables_complete_with_heading": len(titles),
               "annual_sensitivity_table_complete_with_heading": True,
               "interval_insertion_and_reference_checks_passed": True},
    "limitations": ["Visual and format checks do not establish publication rights or independent confirmation.",
                    "The minute experiments and submission requirements remain incomplete."]
}
out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
print(json.dumps({"status": receipt["status"], "path": str(out), "checks": receipt["checks"]}, ensure_ascii=False))
