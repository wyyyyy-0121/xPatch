"""Keep each detailed-results table with its task label in a private Word draft."""
import json
from pathlib import Path
import sys
from docx import Document
from format_private_manuscript import inspect

source, destination = (Path(v) for v in sys.argv[1:3])
if destination.exists():
    raise ValueError('Destination already exists')
before = inspect(source)
doc = Document(source)
if len(doc.tables) != 24:
    raise ValueError('Expected twelve main and twelve detailed-result tables')
paragraphs = [p for p in doc.paragraphs if p.text.startswith('Five folds; ')]
if len(paragraphs) != 12:
    raise ValueError('Expected twelve detailed-result descriptions')
for paragraph in paragraphs:
    paragraph.paragraph_format.keep_with_next = True
for table in doc.tables[12:]:
    if len(table.columns) != 8 or len(table.rows) != 9 or table.rows[0].cells[0].text.strip() != 'Model':
        raise ValueError('Unexpected detailed-result table shape')
    for index, row in enumerate(table.rows):
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.keep_with_next = index < len(table.rows) - 1
doc.save(destination)
after = inspect(destination)
for field in ('text', 'math', 'links', 'media', 'tables'):
    if before[field] != after[field]:
        raise ValueError('Content changed during pagination: ' + field)
print(json.dumps({'status':'detailed_tables_kept_with_labels',
                  'table_groups':12,'body_text_math_links_media_tables_preserved':True}))
