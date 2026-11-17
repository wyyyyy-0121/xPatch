#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
if command -v pdfinfo >/dev/null 2>&1; then
    article_pages=$(pdfinfo main.pdf | awk '/^Pages:/ {print $2}')
    if [ "$article_pages" != "6" ]; then
        printf 'Expected 6 pages; got %s. Review layout before submission.\n' "$article_pages" >&2
        exit 1
    fi
fi
printf 'Compiled main.pdf successfully.\n'
