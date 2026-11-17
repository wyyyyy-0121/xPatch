"""Install the verified draft only if the prior user files have not changed."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile


here = Path(__file__).resolve().parent
out = here.parents[1]
staged_receipt = json.loads((here / "STAGED_MANUSCRIPT_VERIFICATION.json").read_text(encoding="utf8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


originals = {out / "PAPER_DRAFT_EN.md": here / "PAPER_DRAFT_EN.md",
             out / "AAPL_MULTISCALE_WORKING_DRAFT.docx": here / "AAPL_MULTISCALE_WORKING_DRAFT.docx",
             out / "manuscript_audits/RAW_PANDOC_WORKING_DRAFT.docx": here / "RAW_PANDOC_WORKING_DRAFT.docx",
             out / "manuscript_audits/verify_word_draft.py": here / "verify_word_draft.py",
             out / "manuscript_audits/WORD_DRAFT_VERIFICATION.json": here / "WORD_DRAFT_VERIFICATION.json",
             out / "paper_artifacts/word_format_receipt.json": here / "word_format_receipt.json"}
if any(digest(current) != digest(snapshot) for current, snapshot in originals.items()):
    raise ValueError("A prior manuscript file changed after its backup; installation stopped")
checks = {here / "PAPER_DRAFT_EN_REVISED.md": "source_sha256",
          here / "REVISED_FORMATTED.docx": "docx_sha256", here / "REVISED_RAW.docx": "raw_docx_sha256",
          here / "render/REVISED_FORMATTED.pdf": "pdf_sha256"}
if staged_receipt.get("status") != "revised_working_draft_checked" or any(
        digest(path) != staged_receipt[key] for path, key in checks.items()):
    raise ValueError("A staged manuscript artifact changed after verification")
pdf_target = out / "AAPL_MULTISCALE_WORKING_DRAFT.pdf"
if pdf_target.exists():
    raise ValueError("A PDF already exists at the intended new path; preserve it and stop")
verifier_path = out / "manuscript_audits/verify_word_draft.py"
verifier = verifier_path.read_text(encoding="utf8")
old_pdf = 'PDF = OUT / "paper_artifacts/word_render.dBpbeHyI/AAPL_MULTISCALE_WORKING_DRAFT.pdf"'
new_pdf = 'PDF = OUT / "AAPL_MULTISCALE_WORKING_DRAFT.pdf"'
if verifier.count(old_pdf) != 1:
    raise ValueError("The existing Word verifier is not in its expected state")
format_receipt = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "format_checked",
                  "before_docx_sha256": digest(here / "REVISED_RAW.docx"),
                  "after_docx_sha256": digest(here / "REVISED_FORMATTED.docx"),
                  "source_sha256": digest(here / "PAPER_DRAFT_EN_REVISED.md"),
                  "formatting": "Keep the shared-loss introduction with the following equation.",
                  "paragraph_text_unchanged": True, "formal_results_added": False}
updates = {out / "PAPER_DRAFT_EN.md": (here / "PAPER_DRAFT_EN_REVISED.md").read_bytes(),
           out / "AAPL_MULTISCALE_WORKING_DRAFT.docx": (here / "REVISED_FORMATTED.docx").read_bytes(),
           out / "manuscript_audits/RAW_PANDOC_WORKING_DRAFT.docx": (here / "REVISED_RAW.docx").read_bytes(),
           pdf_target: (here / "render/REVISED_FORMATTED.pdf").read_bytes(),
           verifier_path: verifier.replace(old_pdf, new_pdf).encode("utf8"),
           out / "paper_artifacts/word_format_receipt.json":
               (json.dumps(format_receipt, indent=2) + "\n").encode("utf8")}
temporary = {}
for target, content in updates.items():
    with tempfile.NamedTemporaryFile(prefix=".manuscript-update-", dir=target.parent, delete=False) as stream:
        mode = stat.S_IMODE(originals[target].stat().st_mode) if target in originals else 0o644
        os.fchmod(stream.fileno(), mode)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
        temporary[target] = Path(stream.name)
if any(digest(current) != digest(snapshot) for current, snapshot in originals.items()) or pdf_target.exists():
    raise ValueError("A manuscript file changed during staging; installation stopped")
for target, path in temporary.items():
    os.replace(path, target)
receipt = {"status": "installed_revised_working_draft", "created_at_utc": datetime.now(timezone.utc).isoformat(),
           "ready_for_submission": False, "formal_results_added": False,
           "backups": {str(current): str(snapshot) for current, snapshot in originals.items()},
           "installed_sha256": {str(target): digest(target) for target in updates},
           "staged_verification": str(here / "STAGED_MANUSCRIPT_VERIFICATION.json")}
with (here / "INSTALL_RECEIPT.json").open("x", encoding="utf8") as stream:
    json.dump(receipt, stream, indent=2, allow_nan=False)
print(json.dumps({"status": receipt["status"], "docx": str(out / "AAPL_MULTISCALE_WORKING_DRAFT.docx"),
                  "pdf": str(pdf_target)}))
