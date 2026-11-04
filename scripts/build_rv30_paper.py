"""Build two-column LaTeX sources from the English working manuscript.

Requires Pandoc; add --compile for an optional pdfLaTeX check. This command does not load models,
query market data, or alter experimental results. The LaTeX project is also
editable and compilable separately in the desktop app.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'research/paper'
PROJECT = PAPER / 'latex'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compile', action='store_true', help='Also compile a local preview with pdfLaTeX')
    args = parser.parse_args()
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    if not (PROJECT / 'main.tex').exists():
        raise RuntimeError('Create the dedicated LaTeX project first')
    for executable in (('pandoc', 'pdflatex') if args.compile else ('pandoc',)):
        if not shutil.which(executable):
            raise RuntimeError(f'{executable} is required')
    text = (PAPER / 'MANUSCRIPT_EN.md').read_text()
    title = text.splitlines()[0].removeprefix('# ')
    abstract = text.split('## Abstract\n\n', 1)[1].split('\n\n**Keywords:**', 1)[0]
    keywords = text.split('**Keywords:** ', 1)[1].split('\n', 1)[0]
    body = '## 1. Introduction' + text.split('## 1. Introduction', 1)[1]
    body = body.split('## Editorial notes for the submission version', 1)[0].rstrip()

    def latex_fragment(value):
        return subprocess.run(['pandoc', '-f', 'markdown+tex_math_single_backslash',
            '-t', 'latex'], input=value, text=True, capture_output=True, check=True).stdout.strip()

    titleblock = '\\twocolumn[\n\\begin{center}\n{\\Large\\bfseries ' + latex_fragment(title) + '}\\par\n'
    titleblock += '\\vspace{5pt}\n{\\normalsize \\paperauthor\\textsuperscript{1,*}}\\par\n'
    titleblock += '{\\small \\textsuperscript{1}\\paperaffiliation}\\par\n'
    titleblock += '{\\small \\paperlocation}\\par\n'
    titleblock += "{\\small \\textsuperscript{*}Corresponding author's e-mail: \\underline{\\hspace{3.5cm}}}\\par\n"
    titleblock += '{\\small Working manuscript -- 4 November 2026}\\par\n\\end{center}\n'
    titleblock += '\\begin{center}\\begin{minipage}{0.96\\textwidth}\\small\n'
    titleblock += '\\textbf{Abstract.} ' + latex_fragment(abstract) + '\n\\par\\vspace{4pt}\n'
    titleblock += '\\textbf{Keywords:} ' + latex_fragment(keywords) + '\n'
    titleblock += '\\end{minipage}\\end{center}\n\\vspace{5pt}\n]\n'
    (PROJECT / 'titleblock.tex').write_text(titleblock)

    images = PROJECT / 'images'
    images.mkdir(exist_ok=True)
    image_pattern = r'!\[[^\]]+\]\(figures/([^\)]+)\.png\)\n\n\*\*Figure ([12])\.\*\* ([^\n]+)'
    count = 0
    def replace_image(match):
        nonlocal count
        name, number, caption = match.groups()
        source = PAPER / 'figures' / (name + '.pdf')
        shutil.copyfile(source, images / source.name)
        count += 1
        return ('\\begin{figure*}[!t]\n\\centering\n'
            '\\includegraphics[width=0.98\\textwidth]{images/' + source.name + '}\n'
            '\\caption{' + latex_fragment(caption) + '}\n'
            '\\label{fig:result-' + number + '}\n\\end{figure*}')
    body = re.sub(image_pattern, replace_image, body)
    if count != 2:
        raise RuntimeError('Both manuscript figures and their captions must be retained')
    # Preserve the working reference list while keeping editorial tasks out of the article.
    body += '\n\n\\endgroup\n'
    intermediate = PROJECT / 'article_body.md'
    intermediate.write_text(body)
    subprocess.run(['pandoc', str(intermediate), '-f', 'markdown+tex_math_single_backslash',
        '-t', 'latex', '--lua-filter', str(PROJECT / 'tables.lua'), '-o', str(PROJECT / 'body.tex')], check=True)
    intermediate.unlink()
    if args.compile:
        for run in (1, 2):
            with (PROJECT / f'compile-{run}.log').open('w') as log:
                subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
                    '-file-line-error', 'main.tex'], cwd=PROJECT, stdout=log,
                    stderr=subprocess.STDOUT, check=True)
    files = [PROJECT / name for name in ('main.tex','body.tex','titleblock.tex','tables.lua','README.md','.gitignore')]
    files += sorted(images.glob('*.pdf'))
    versions = {'pandoc':subprocess.run(['pandoc','--version'],capture_output=True,text=True,
        check=True).stdout.splitlines()[0]}
    if args.compile:
        versions['pdflatex']=subprocess.run(['pdflatex','--version'],capture_output=True,text=True,
            check=True).stdout.splitlines()[0]
    manifest = {'source':str((PAPER / 'MANUSCRIPT_EN.md').relative_to(ROOT)),
        'source_sha256':digest(PAPER / 'MANUSCRIPT_EN.md'),
        'builder':str(Path(__file__).relative_to(ROOT)), 'builder_sha256':digest(Path(__file__)),
        'purpose':'two_column_review_manuscript_not_final_submission_or_model_acceptance',
        'typesetting_versions':versions, 'figures':count, 'delivered_as':'latex_sources_only',
        'files':{str(path.relative_to(ROOT)):digest(path) for path in files}}
    (PROJECT / 'build_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'passed':True,'main_tex':str(PROJECT / 'main.tex'),
        'compiled_preview':args.compile,'figures':count}))


if __name__ == '__main__':
    main()
