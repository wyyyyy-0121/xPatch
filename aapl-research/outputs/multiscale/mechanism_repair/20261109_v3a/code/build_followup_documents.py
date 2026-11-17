"""Render isolated updated documents with the immutable text-preserving v3 formatter."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

BASE = Path(__file__).resolve().parents[1]
V3 = BASE.parent / '20261109_v3'
sys.path.insert(0, str(V3 / 'code'))
from format_documents import format_doc


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert len(os.sched_getaffinity(0)) <= 2
    folder = BASE / 'manuscript'
    audit = BASE / 'audit'; audit.mkdir(exist_ok=True)
    config = audit / 'fontconfig.xml'
    config.write_text('<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd"><fontconfig><include ignore_missing="yes">/etc/fonts/fonts.conf</include><dir>/mnt/c/Windows/Fonts</dir></fontconfig>')
    env = dict(os.environ, FONTCONFIG_FILE=str(config), OPENBLAS_NUM_THREADS='1')
    font = subprocess.check_output(['fc-match', 'Times New Roman'], env=env, text=True).strip()
    assert 'times' in font.lower(), font
    documents = []
    for name, double in [('MAIN_PAPER', True), ('SUPPLEMENT', False)]:
        source = folder / f'{name}_EN.md'; raw = folder / f'{name}_RAW.docx'; output = folder / f'{name}_EN.docx'
        subprocess.run(['pandoc', '--from=markdown+tex_math_single_backslash+autolink_bare_uris', '--to=docx',
            '--standalone', f'--resource-path={folder}', f'--output={raw}', str(source)], check=True)
        format_doc(raw, output, double)
        profile = audit / f'lo_profile_{name.lower()}'
        subprocess.run(['libreoffice', f'-env:UserInstallation={profile.as_uri()}', '--headless',
            '--convert-to', 'pdf', '--outdir', str(folder), str(output)], env=env, check=True)
        pdf = output.with_suffix('.pdf')
        info = subprocess.check_output(['pdfinfo', str(pdf)], text=True)
        pages = int(next(x.split(':', 1)[1] for x in info.splitlines() if x.startswith('Pages:')))
        subprocess.run(['pdftotext', '-layout', str(pdf), str(folder / f'{name}_PDF_TEXT.txt')], check=True)
        previews = folder / 'page_previews' / name.lower(); previews.mkdir(parents=True, exist_ok=True)
        # Earlier generated previews only; no user sources are removed.
        for path in previews.glob('page-*.png'):
            path.unlink()
        subprocess.run(['pdftoppm', '-scale-to', '1600', '-png', str(pdf), str(previews / 'page')], check=True)
        fonts = subprocess.check_output(['pdffonts', str(pdf)], text=True)
        (folder / f'{name}_PDF_FONTS.txt').write_text(fonts)
        documents.append({'document': name, 'pages': pages, 'source_sha256': sha(source),
            'pdf_sha256': sha(pdf), 'docx_sha256': sha(output), 'preview_pages': len(list(previews.glob('page-*.png'))),
            'times_embedded': 'TimesNewRoman' in fonts,
            'preservation': json.loads(Path(str(output) + '.preservation.json').read_text())})
    (audit / 'DOCUMENT_BUILD.json').write_text(json.dumps({'status': 'rendered_requires_visual_review',
        'formatter_sha256': sha(V3 / 'code/format_documents.py'), 'documents': documents,
        'main_page_target_4_to_6': 4 <= documents[0]['pages'] <= 6,
        'visual_review_complete': False}, indent=2))
    print(json.dumps(documents), flush=True)


if __name__ == '__main__':
    main()
