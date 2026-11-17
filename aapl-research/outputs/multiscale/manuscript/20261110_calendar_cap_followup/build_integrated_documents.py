"""Build the separate integrated manuscript without loading market data."""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
ROOT = next(p for p in BASE.parents if (p / "MULTISCALE_CURRENT_PLAN_ZH.md").is_file())
FORMATTER = ROOT / "outputs/multiscale/mechanism_repair/20261109_v3b/code"
PREVIOUS = ROOT / "outputs/multiscale/manuscript/20261110_readout_followup"
sys.path.insert(0, str(FORMATTER))
from format_context_documents import format_doc, inspect
from docx import Document
from docx.shared import Mm


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args, **kwargs):
    return subprocess.run(args, check=True, timeout=120, **kwargs)


def table_layout(path, double):
    before = inspect(path)
    doc = Document(path)
    changed = []
    for p in doc.paragraphs:
        if re.match(r"Table\s+\d+[.]", p.text):
            p.paragraph_format.keep_with_next = True
            p.paragraph_format.keep_together = True
    for table in doc.tables:
        heads = [c.text for c in table.rows[0].cells]
        ratios = None
        if heads == ["H", "Candidate gain", "Folds", "Mechanism", "Candidate"]:
            ratios = [.08, .27, .13, .26, .26]
        elif heads == ["Primary", "Reference", "Block days", "Gain", "95% interval", "Supports gain"]:
            ratios = [.22, .31, .07, .11, .19, .10]
        elif heads == ["State", "Path change", "Endpoint change", "Eligibility"]:
            ratios = [.27, .23, .27, .23]
        elif heads == ["Loss", "Reference", "Block sessions", "Gain (%)", "95% interval (%)", "Supports gain"]:
            ratios = [.13, .13, .13, .14, .32, .15]
        if ratios is None:
            continue
        width = (178 - 5.08) / 2 if double else 178
        for j, col in enumerate(table.columns):
            col.width = Mm(width * ratios[j])
        for row in table.rows:
            for j, cell in enumerate(row.cells):
                cell.width = Mm(width * ratios[j])
        changed.append(heads)
    doc.save(path)
    assert inspect(path) == before, "Layout changed text, equations, media, tables or links"
    return changed


def main():
    start = time.monotonic()
    review = BASE / "review"
    review.mkdir(exist_ok=True)
    config = review / "fontconfig.xml"
    config.write_text('<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd">'
                      '<fontconfig><include ignore_missing="yes">/etc/fonts/fonts.conf</include>'
                      '<dir>/mnt/c/Windows/Fonts</dir></fontconfig>')
    env = dict(os.environ, FONTCONFIG_FILE=str(config), OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="-1")
    for directory in ("figures",):
        if (PREVIOUS / directory).is_dir() and not (BASE / directory).exists():
            shutil.copytree(PREVIOUS / directory, BASE / directory)
    author = PREVIOUS / "AUTHOR_INFORMATION.json"
    if author.is_file() and not (BASE / author.name).exists():
        shutil.copy2(author, BASE / author.name)
    docs = []
    for name, double in (("MAIN_PAPER", True), ("SUPPLEMENT", False)):
        source = BASE / f"{name}_EN.md"
        raw = BASE / f"{name}_RAW.docx"
        docx = BASE / f"{name}_EN.docx"
        pdf = docx.with_suffix(".pdf")
        source_sha = sha(source)
        command(["pandoc", "--from=markdown+tex_math_single_backslash+autolink_bare_uris",
                 "--to=docx", "--standalone", f"--resource-path={BASE}",
                 f"--output={raw}", str(source)], env=env)
        format_doc(raw, docx, double)
        changes = table_layout(docx, double)
        preservation = json.loads(Path(str(docx) + ".preservation.json").read_text())
        assert inspect(raw) == inspect(docx)
        preservation["final_layout_preservation_passed"] = True
        preservation["adjusted_table_headers"] = changes
        Path(str(docx) + ".preservation.json").write_text(json.dumps(preservation, indent=2) + "\n")
        profile = review / f"lo_{name.lower()}"
        command(["libreoffice", "-env:UserInstallation=" + profile.as_uri(), "--headless",
                 "--convert-to", "pdf", "--outdir", str(BASE), str(docx)], env=env)
        info = command(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
        pages = int(re.search(r"^Pages:\s+(\d+)", info, re.M).group(1))
        pdf_hash = sha(pdf)
        preview = BASE / "page_previews" / name.lower() / pdf_hash[:16]
        preview.mkdir(parents=True, exist_ok=False)
        command(["pdftotext", "-layout", str(pdf), str(BASE / f"{name}_PDF_TEXT.txt")])
        command(["pdftoppm", "-scale-to", "1600", "-png", str(pdf), str(preview / "page")], env=env)
        images = sorted(preview.glob("page-*.png"))
        assert len(images) == pages
        fonts = command(["pdffonts", str(pdf)], capture_output=True, text=True).stdout
        (review / f"{name}_FONTS.txt").write_text(fonts)
        assert source_sha == sha(source), "Source changed during build"
        docs.append({"name": name, "source_sha256": source_sha, "docx_sha256": sha(docx),
                     "pdf_sha256": pdf_hash, "pages": pages, "columns": 2 if double else 1,
                     "preservation": preservation, "preview_directory": str(preview),
                     "preview_sha256": {p.name: sha(p) for p in images}})
    report = {"status": "rendered_pending_independent_claim_and_actual_visual_acceptance",
              "documents": docs, "elapsed_seconds": time.monotonic() - start,
              "formatter_sha256": sha(FORMATTER / "format_context_documents.py"),
              "build_script_sha256": sha(Path(__file__)), "GPU": False,
              "market_data_loaded_or_scored": False, "Q3_values_read": False,
              "research_goal_achieved": False}
    (review / "DOCUMENT_BUILD.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
