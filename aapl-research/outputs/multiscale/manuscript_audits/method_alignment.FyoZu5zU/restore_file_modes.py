"""Restore pre-existing manuscript file modes after atomic replacement."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat


here = Path(__file__).resolve().parent
out = here.parents[1]
receipt = json.loads((here / "INSTALL_RECEIPT.json").read_text(encoding="utf8"))
targets = {out / "PAPER_DRAFT_EN.md": here / "PAPER_DRAFT_EN.md",
           out / "AAPL_MULTISCALE_WORKING_DRAFT.docx": here / "AAPL_MULTISCALE_WORKING_DRAFT.docx",
           out / "manuscript_audits/RAW_PANDOC_WORKING_DRAFT.docx": here / "RAW_PANDOC_WORKING_DRAFT.docx",
           out / "manuscript_audits/verify_word_draft.py": here / "verify_word_draft.py",
           out / "paper_artifacts/word_format_receipt.json": here / "word_format_receipt.json",
           out / "AAPL_MULTISCALE_WORKING_DRAFT.pdf": here / "AAPL_MULTISCALE_WORKING_DRAFT.pdf"}
for target in targets:
    actual = hashlib.sha256(target.read_bytes()).hexdigest()
    if actual != receipt["installed_sha256"][str(target)]:
        raise ValueError(f"Installed file changed; leave its mode untouched: {target}")
changes = []
for target, snapshot in targets.items():
    before, expected = stat.S_IMODE(target.stat().st_mode), stat.S_IMODE(snapshot.stat().st_mode)
    if before != expected:
        os.chmod(target, expected)
    changes.append({"path": str(target), "before": oct(before), "restored": oct(expected)})
result = {"status": "file_modes_restored", "created_at_utc": datetime.now(timezone.utc).isoformat(),
          "changes": changes, "file_bytes_changed": False}
with (here / "MODE_RESTORATION.json").open("x", encoding="utf8") as stream:
    json.dump(result, stream, indent=2)
print(json.dumps({"status": result["status"], "files": len(changes)}))
