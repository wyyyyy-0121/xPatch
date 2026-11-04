# Two-column LaTeX manuscript

This directory is a self-contained LaTeX source project. Copy the **whole directory**, including `images/`, when compiling elsewhere. No market credentials, model environment, Python, or Pandoc is needed to compile the saved `.tex` files.

## Compile

Use **pdfLaTeX** with a current TeX Live or MiKTeX installation. In Overleaf, upload the directory contents, select `main.tex` as the main document, and choose pdfLaTeX.

From a terminal in this directory:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Two passes resolve table and figure references. The project uses standard packages: geometry, fontenc, inputenc, mathptmx, amsmath, amssymb, graphicx, booktabs, array, microtype, caption, hyperref, url, and titlesec. It requires neither BibTeX nor shell escape. References are included directly in the saved source.

## Edit

| File | Content |
|---|---|
| `main.tex` | Two-column page layout, package settings, PDF metadata, and document entry point |
| `titleblock.tex` | Title, author placeholder, abstract, and keywords spanning both columns |
| `body.tex` | English article, equations, tables, captions, and the working reference list |
| `images/development_loss_differences.pdf` | Vector figure for development comparisons |
| `images/quarterly_loss_differences.pdf` | Vector figure for all ten development quarters |
| `tables.lua` | Optional Markdown-to-LaTeX conversion filter; not needed to compile |
| `build_manifest.json` | Fingerprints of saved source and vector figures |

The style is a general two-column research-article layout. It is not an official IEEE, ACM, or journal template. Once a venue is selected, its official class and author instructions should replace the general layout. Figure and table references and the bibliography must then be checked again.

Author details remain placeholders. Some references still require original-source checking; the status is recorded in `../SOURCES.md`. The article distinguishes historical results from the uncompleted delayed and timely observation studies. Formatting does not change those evidence limits.

The longer editable Markdown draft is `../MANUSCRIPT_EN.md`. To regenerate the `.tex` files from that draft, use the optional repository script `scripts/build_rv30_paper.py` with Pandoc installed. **Regeneration overwrites `body.tex` and `titleblock.tex`**, so preserve direct LaTeX edits first. Ordinary editing and compilation can use this directory alone.

Build auxiliaries (`.aux`, `.log`, `.out`, `.toc`, `.synctex.gz`) and the local preview PDF were removed from the delivered source tree. They will be recreated by your compiler.
