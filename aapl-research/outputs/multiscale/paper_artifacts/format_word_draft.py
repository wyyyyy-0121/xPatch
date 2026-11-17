"""Keep the shared-loss equation with its introduction in the working DOCX."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from docx import Document


HERE = Path(__file__).resolve().parent
TARGET = HERE.parent / "AAPL_MULTISCALE_WORKING_DRAFT.docx"
RECEIPT = HERE / "word_format_receipt.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    before = digest(TARGET)
    document = Document(TARGET)
    texts = [paragraph.text for paragraph in document.paragraphs]
    matches = [paragraph for paragraph in document.paragraphs if "its step weight is" in paragraph.text]
    if len(matches) != 1:
        raise ValueError("Expected exactly one shared-loss introduction")
    matches[0].paragraph_format.keep_with_next = True
    document.save(TARGET)
    restored = Document(TARGET)
    if [paragraph.text for paragraph in restored.paragraphs] != texts:
        raise ValueError("Word paragraph text changed during formatting")
    match = [paragraph for paragraph in restored.paragraphs if "its step weight is" in paragraph.text]
    if len(match) != 1 or match[0].paragraph_format.keep_with_next is not True:
        raise ValueError("The paragraph keep-with-next setting was not preserved")
    receipt = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "format_checked",
               "before_docx_sha256": before, "after_docx_sha256": digest(TARGET),
               "source_sha256": digest(HERE.parent / "PAPER_DRAFT_EN.md"),
               "formatting": "Keep the shared-loss introduction with the following equation.",
               "paragraph_text_unchanged": True, "formal_results_added": False}
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"status": receipt["status"], "receipt": str(RECEIPT),
                      "docx_sha256": receipt["after_docx_sha256"]}))


if __name__ == "__main__":
    main()
