# Historical development verification

These commands verify the local AAPL multiscale cohort started on 7 November
2026. They preserve the original source, protocol, checkpoint and exposure
identities. They do not train models or access the sealed 2026Q3 observations.
The permission update is recorded separately in `DATA_PERMISSION_STATUS.md`.

The fixed inventory has 20 tasks and 100 chronological task-folds. Its paths
identify the originally approved runs, including the separate first H60-return
fold. There is no automatic search for a more favorable result or a replacement
source. The inventory is relative to the repository directory; copied output
trees must preserve the recorded paths in their original evidence records.

## Required local artifacts

Run from `aapl-research` with its existing environment. Keep the original
`outputs/multiscale/private_*` protocols, queue states, fold receipts,
independent checks, predictions and checkpoints, and the Alpaca admission at
`outputs/multiscale/admission/alpaca_20261107T052228573242Z`.

The Git repository contains the verification code. Market arrays, weights,
manuscripts, supplier responses and machine receipts remain in ignored output
directories. A Git checkout alone cannot reproduce the numerical evidence;
the matching licensed local artifacts are also required. Do not edit original
receipts to relocate or relabel a run.

## Commands

All examples use at most two logical CPUs and hide CUDA. Set the CPU IDs to two
available processors on the execution machine. On the current workstation,
the shared set `24,26` is used for the night restriction. Use a new output name
for each run; verification commands do not overwrite earlier receipts.

```bash
AAPL_CPU_IDS=24,26 AAPL_CPU_THREADS=2 CUDA_VISIBLE_DEVICES=-1 \
  bash scripts/python.sh -m multiscale.historical_coverage \
  --output outputs/multiscale/reproduction/coverage_run1.json

AAPL_CPU_IDS=24,26 AAPL_CPU_THREADS=2 CUDA_VISIBLE_DEVICES=-1 \
  bash scripts/python.sh -m multiscale.historical_minute_check \
  --coverage outputs/multiscale/reproduction/coverage_run1.json \
  --output outputs/multiscale/reproduction/minute_metrics_run1.json

AAPL_CPU_IDS=24,26 AAPL_CPU_THREADS=2 CUDA_VISIBLE_DEVICES=-1 \
  bash scripts/python.sh -m multiscale.historical_restore_check \
  --coverage outputs/multiscale/reproduction/coverage_run1.json \
  --output outputs/multiscale/reproduction/model_restore_run1
```

`historical_coverage` requires a unique record for every planned task-fold. A
completed status must match the protocol, origins, all eight models, current
training source hashes and an independent verification. Resumed folds may use
their explicitly recorded replacement verification. A stale original check
does not override a matching resumed check. A queue change during inventory
assembly triggers a bounded retry.

`historical_minute_check` requires a coverage snapshot matching the live queue
state and current inventory code. It independently recalculates saved
per-step, full-path and endpoint errors, directional coverage and accuracy.
Pooled losses weight folds by origin count. Partial tasks list their missing
folds. `--task 60:future_log_return` restricts the check to one task without
changing its inventory. If training advances, take a new coverage snapshot
before rerunning.

`historical_restore_check` starts a fresh CPU process for every fold of each
task that has all five completed folds. It loads all four neural predictors,
checks source and checkpoint hashes, and recreates every saved forecast from
input windows ending at their own decisions. Saved target arrays are not
opened. The original prediction batch size of 512 is retained because a
different batch shape can change floating-point arithmetic. The fixed
tolerances are reported together with exact equality and maximum differences.
`--admission` can select the original matching admission directory explicitly.

These checks measure different things: inventory integrity, arithmetic
recalculation, and predictor restoration. None changes the historical
development status, establishes convergence, or completes the final inference
package or an interrupted training recovery test.

## Focused tests

```bash
AAPL_CPU_IDS=24,26 AAPL_CPU_THREADS=2 CUDA_VISIBLE_DEVICES=-1 \
  bash scripts/python.sh -m pytest -q \
  --basetemp tmp/historical-tools-pytest-new-run \
  multiscale/test_historical_coverage.py \
  multiscale/test_historical_minute_check.py \
  multiscale/test_historical_restore_check.py
```

The completion checks exercise mismatched receipts, origins, protocols,
protected-period flags, changed source files and stale verification selection.
Metric checks cover directional abstentions, weighting, overlap and the strict
cutoff. Source-path checks reject traversal and changed bytes. The current
inventory check is skipped when its local research outputs are absent.
