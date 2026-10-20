# Verification record

Verified UTC: 2026-10-20T06:11:03.184438+00:00

Core regression suite: `31 passed in 4.88s` using project `.venv/bin/python -m pytest tests/test_experiment.py -q`.
Final `git diff --check` passed.

- Calendar/session coverage includes sparse sessions, whole missing days, holidays, early closes and DST.
- Incomplete aggregates are discarded; input and future labels cross closures while rejecting missing trading bars.
- Empty and impossible configurations skip cleanly; 480-minute windows and 8-hour horizons generate valid cross-session samples.
- Same origins, fold membership and raw-history MASE scales across candidate windows.
- Feature/target scalers use training only; explicit final evaluation requires matching freeze and shared exposure ledger.
- All five learned model types round-trip through saved artifacts; neural tests used CUDA.
- Vectorized EMA matches recurrence values and gradients at 12/252/480 steps; unfolding preserves patch order.
- Legacy CSV reproduced from pinned public mirror with matching raw and canonical SHA-256; execution uses project data paths.
- Six selected deployment artifacts loaded and produced correctly counted future timestamped predictions.

Training/preprocessing code SHA-256: `44758e133723fe2a07c1a1017a4695b99f79cb3af602bde267d9d5380ba6727e`.

Compute: NVIDIA GeForce RTX 5070 Ti Laptop GPU; PyTorch 2.13.0+cu130. CPU training uses 24 threads / 24 workers; GPU loaders use zero workers.

| Profile | Configurations / folds | Naive median MASE (same for all windows) | Accepted |
|---|---:|---:|---:|
| annual | 64 / 5 | 6.13292215 | 0 |
| quarterly | 64 / 5 | 3.49480257 | 0 |
| monthly | 64 / 5 | 8.56455972 | 0 |

All 192 long-horizon configurations completed five validation folds. New final tests were not evaluated.
Historical final reports retain their original scores but are annotated as exposed/exploratory and superseded.
Real hourly/minute experimental acceptance remains unmet: the bundled minute file has unverified timezone/provenance/adjustment, and the supplied ZIP is synthetic interpolation.
The long-term dataset ends in 2022 and remains unverified. Yahoo refresh returned HTTP 403. No current-market or predictive-advantage claim is made.
