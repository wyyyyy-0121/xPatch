"""Apply the existing equation pagination rule to a staged working draft."""
from pathlib import Path
import json
import sys

from docx import Document


source, destination = (Path(arg) for arg in sys.argv[1:3])
if destination.exists():
    raise ValueError("The staged Word output already exists")
document = Document(source)
before = [paragraph.text for paragraph in document.paragraphs]
matches = [p for p in document.paragraphs if "its step weight is" in p.text]
if len(matches) != 1:
    raise ValueError("Expected one shared-loss equation introduction")
matches[0].paragraph_format.keep_with_next = True
document.save(destination)
after = Document(destination)
matches = [p for p in after.paragraphs if "its step weight is" in p.text]
if ([p.text for p in after.paragraphs] != before or len(matches) != 1 or
        matches[0].paragraph_format.keep_with_next is not True):
    raise ValueError("The equation pagination changed the document text or did not persist")
print(json.dumps({"status": "staged_word_formatted", "paragraphs": len(before)}))
