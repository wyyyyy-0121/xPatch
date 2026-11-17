# Article and figure reproduction

Compile `main.tex` twice with pdfLaTeX, or run `bash build.sh`. The main article has six A4 pages, including its bibliography, in a generic two-column layout with 10-point body text. This is not a venue-specific official template. The project uses relative paths and requires neither shell escape nor BibTeX. Figures are already supplied as vector PDFs; generating them is optional for document compilation.

This edition rewrites the preceding six-page article at `outputs/multiscale/manuscript/20261111_six_page_latex/`. Its title, author, school, task results and scientific conclusions are retained. Methods now introduce the comparisons needed to interpret the figures; internal stage accounting stays in the supplement. Mathematical explanations of calibration and nominal split returns do not represent new empirical tests.

## Four figures, four purposes

`make_figures.py` requires Python 3, NumPy and Matplotlib. It reads only the bundled small CSV/JSON files, writes PDF, SVG and PNG for all four figures, and does not read licensed market bars, weights or holdout data. No model fitting, inference or resampling is performed.

- `return_factorial`: initialization by nominal-price-anchor matrix for minute H60 and hourly H8. `return_factorial.csv` retains the exact saved pooled MSE and zero MSE for each cell and the separately calibrated observed-anchor cell. Normalization is MSE / zero MSE. Its source is `outputs/multiscale/mechanism_repair/20261109_v3/analysis/REPLICATION_POOLED.csv`, xPatch rows with the recorded method names. The grid uses trained losses; the values below it add shrinkage to the initialized, observed-anchor cell.
- `cap_comparison`: hourly-H1 price MSE from `cap_values.csv`, identical to the accepted preceding table. The up-to-sixty-epoch arm is continued training with preserved optimizer/RNG/batch state, not a restart. Persistence MSE is 1.8985 USD squared. The logarithmic axis and direct labels retain the close initialization/reference comparison.
- `annual_intervals`: the eight saved annual rows for `repair_shrink` against persistence/zero and `fit_mean_shrink`, with 12- and 24-month blocks. `annual_intervals.csv` is an exact row subset of `outputs/multiscale/mechanism_repair/20261109_v3a/longscale_replication/analysis/INTERVALS.csv`. Each interval uses its named denominator. No interval is recalculated or truncated. They are descriptive nominal 95% intervals on exposed history. `annual_values.csv` additionally retains the accepted pooled and original four-path summaries discussed in the article.
- `split_event`: dates, reference equations and pooled monthly endpoint reductions from `split_event.json`. Supplement S32 documents the July 30, 2020 announcement, August 1 00:00 New York eligibility boundary, August 31 effective date, and 21 eligible overlapping paths among 1,970. Time positions are proportional calendar annotations. No price trajectory or additional observation is generated. This reference adds public announcement information to the original close-only comparison.

`review/SOURCE_LOCKS.json` records hashes of the specific local source summaries and parent article used for this edition. Its repository-relative provenance paths do not participate in compilation or figure generation. The acceptance record distinguishes document checks from a new scientific review.

## Supplement and experimental evidence

The 41-page `supplement/S1_S38_previous.pdf` and its Markdown source are byte-identical to the preceding accepted supplement. The standalone two-page probability addendum and its score/decision receipts are also inherited without modification. Their historical stage descriptions and section labels retain their original meaning.

The probability screen uses the existing results at `outputs/multiscale/probability_direction_screen/20261111_fast_screen/`. The main article reports the same two-fold scores: minute passes, hourly fails, and the overall screen stops. The screen does not demonstrate a new conditional-mean forecasting method. The addendum retains full scores and the source-access qualification for the HAR-type motivation.

Raw bars, per-origin arrays, checkpoints and the original experiment environment remain external dependencies for full scientific replication. The evidence map and original reproduction guide in `outputs/multiscale/manuscript/20261111_contribution_revision/` identify those records. The ZIP reproduces document rendering and aggregate plots. It does not reproduce independent market confirmation, which has not been completed. No 2026Q3 data were opened for this revision. Fill in the author's email before submission.
