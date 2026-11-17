"""Render the separate addendum, retaining the original reviewed main manuscript."""
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import sys

RUN = Path(__file__).resolve().parents[1]
ROOT = next(p for p in RUN.parents if (p / "MULTISCALE_CURRENT_PLAN_ZH.md").is_file())
FORMATTER = ROOT / "outputs/multiscale/mechanism_repair/20261109_v3b/code"
sys.path.insert(0, str(FORMATTER))
from format_context_documents import format_doc


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args, **kwargs):
    return subprocess.run(args, check=True, timeout=90, **kwargs)


source = RUN / "MANUSCRIPT_ADDENDUM_EN.md"
raw = RUN / "MANUSCRIPT_ADDENDUM_RAW.docx"
docx = RUN / "MANUSCRIPT_ADDENDUM_EN.docx"
pdf = RUN / "MANUSCRIPT_ADDENDUM_EN.pdf"
review = RUN / "review"
run(["pandoc", str(source), "-o", str(raw)])
format_doc(raw, docx, False)
env = os.environ.copy()
env["FONTCONFIG_FILE"] = str(review / "fontconfig.xml")
profile = review / "lo_addendum_profile"
run(["libreoffice", "-env:UserInstallation=" + profile.as_uri(), "--headless",
     "--convert-to", "pdf", "--outdir", str(RUN), str(docx)], env=env)
info = run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
pages = int(re.search(r"^Pages:\s+(\d+)", info, re.M).group(1))
run(["pdftotext", "-layout", str(pdf), str(RUN / "MANUSCRIPT_ADDENDUM_PDF_TEXT.txt")])
fonts = run(["pdffonts", str(pdf)], capture_output=True, text=True).stdout
(review / "ADDENDUM_FONTS.txt").write_text(fonts)
preview = RUN / "page_previews"
preview.mkdir(exist_ok=True)
run(["pdftoppm", "-r", "110", "-png", str(pdf), str(preview / "page")])
receipt = {
    "status": "rendered_pending_final_visual_and_claim_acceptance",
    "pages": pages,
    "source_sha256": sha(source),
    "docx_sha256": sha(docx),
    "pdf_sha256": sha(pdf),
    "formatter_sha256": sha(FORMATTER / "format_context_documents.py"),
    "build_script_sha256": sha(Path(__file__)),
    "editable_source": str(source),
    "editable_docx": str(docx),
    "PDF": str(pdf),
    "body_layout": "single_column_supplementary_addendum; original main double-column unchanged",
    "preservation": json.loads(Path(str(docx) + ".preservation.json").read_text()),
    "email_pending_author": True,
    "research_goal_achieved": False,
}
(review / "ADDENDUM_BUILD.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
