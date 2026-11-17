# Two-column LaTeX manuscript

This directory is a self-contained LaTeX source project. Copy the **whole directory**, including `images/`, when compiling elsewhere. No market credentials, model environment, Python, or Pandoc is needed to compile the saved `.tex` files.

The current historical empirical manuscript follows three research questions about aggregate error, temporal stability, and calibration. It contains **seven tables and three figures** and compiles to **eight pages** with the supplied general two-column layout. The accompanying `../SUPPLEMENTARY.md` records the historical evidence, diagnostic exports, reproduction steps, and unfinished studies. Simulations and timely prospective evaluation are future work, not reported experimental results.

The author information has also been added to the editable Word copy `../AAPL_RV30_Wang_Yang.docx`. Its tables, figures, and equations were checked after conversion; the email line remains blank as requested. Conversion provenance is recorded in `../word_export.json`, and the verified author details are in `../author.json`.

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
| `titleblock.tex` | Title, shared author setting, abstract, and keywords spanning both columns |
| `body.tex` | English article, equations, tables, captions, and the working reference list |
| `images/task_and_validation.pdf` | Vector diagram of the information cutoff, target, and temporal partitions |
| `images/development_loss_differences.pdf` | Vector figure for development comparisons |
| `images/quarterly_loss_differences.pdf` | Vector figure for all ten development quarters |
| `tables.lua` | Optional Markdown-to-LaTeX conversion filter; not needed to compile |
| `build_manifest.json` | Fingerprints of saved source and vector figures |

The style is a general two-column research-article layout. It is not an official IEEE, ACM, or journal template. Once a venue is selected, its official class and author instructions should replace the general layout. Figure and table references and the bibliography must then be checked again.

The author name is **Wang Yang**. Edit `\newcommand{\paperauthor}{Wang Yang}` in `main.tex` to change it once for both the visible byline and PDF author metadata. The affiliation is **High School Affiliated to Shanghai Jiao Tong University, Shanghai 200439, China**, verified against the official school logo and main-campus contact page. The corresponding-author email line is intentionally blank for the author to fill in. Some references still require original-source checking; the status is recorded in `../SOURCES.md`. The article distinguishes historical results from the uncompleted delayed and timely observation studies. Formatting does not change those evidence limits.

The editable source manuscript is `../MANUSCRIPT_EN.md`; `../MANUSCRIPT_ZH.md` is a Chinese checking digest rather than a full word-for-word translation. To regenerate the `.tex` files from the English draft, use the optional repository script `scripts/build_rv30_paper.py` with Pandoc installed. **Regeneration overwrites `body.tex` and `titleblock.tex`**, so preserve direct LaTeX edits first. Ordinary editing and compilation can use this directory alone.

The standalone source archive excludes build auxiliaries (`.aux`, `.log`, `.out`, `.toc`, `.synctex.gz`) and the local preview PDF. Your compiler will recreate them. A local preview may also remain beside these sources for document validation.
