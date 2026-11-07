# Historical development verification

These commands verify the local AAPL multiscale cohort started on 7 November
2026. They preserve the original source, protocol, checkpoint and exposure
identities. Inventory, scoring and prediction checks do not train models.
The separate restart check performs eight bounded optimizer updates on copies
of archived training checkpoints. None accesses sealed 2026Q3 observations.
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
the shared set `24,26` is reserved for these checks. Use a new output name
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
For the original H60-return first fold, the audit retains the separate control
and neural-stage check identities instead of requiring a schema-2 receipt that
the original run did not create. Its fresh process restores all four neural
predictors on all original validation origins using that layout.

These checks measure different things: inventory integrity, arithmetic
recalculation, and predictor restoration. None changes the historical
development status, establishes convergence, or completes the final inference
package or an interrupted training recovery test.

## All-task result compilation

```bash
AAPL_CPU_IDS=24,26 AAPL_CPU_THREADS=2 CUDA_VISIBLE_DEVICES=-1 \
  bash scripts/python.sh -m multiscale.historical_results \
  --coverage outputs/multiscale/reproduction/coverage_run1.json \
  --output outputs/multiscale/reproduction/full_results_run1
```

The default requires all 100 verified folds. Add `--allow-partial` to create an
explicit interim compilation. All 20 tasks, 160 pooled model slots, 800 model
folds, 400 planned neural fits and 320 model/block interval slots remain in the
tables. Pending cells stay empty. Completed evidence must still match the
verified inventory; later completions do not alter the saved snapshot.

The compiler recalculates endpoint and path metrics from the original arrays,
including every path step. It checks paired origin identities and weights by
the number of origins when pooling. It reproduces the existing descriptive
analysis: 5,000 non-circular moving-block draws, seed 2026, nominal 95% percentile
intervals on the full original source session grid. Annual, quarterly, monthly,
hourly and minute block pairs are 12/24 months, 3/6 months, 21/42 sessions,
2/5 sessions and 1/5 sessions. It reports no development p-values; degenerate or
insufficient intervals remain undefined. Annual nonoverlapping label windows
are a separate descriptive subset, not an effective independent sample size.

`TRAINING_MODELS.csv` records actual epoch caps, early stopping, optimizer steps
and hashed learning histories. `FOLD_ELAPSED.csv` records each completed stage
once, rather than repeating its cost for all eight models. The first H60-return
fold has separate control and neural stages. Original raw elapsed receipts are
unchanged. Live workers suspended during the night include that suspension in
their monotonic wall envelope, despite the original receipt's active-time
label. A separate estimate subtracts the recorded pause envelope for those
workers; it is not exact CPU time. Parallel stage sums are not the study's
elapsed duration, and original queue budgets are accounted separately.

## Real archived-checkpoint process restart

```bash
AAPL_CPU_IDS=24,26 AAPL_CPU_THREADS=2 CUDA_VISIBLE_DEVICES=-1 \
  bash scripts/python.sh -m multiscale.historical_training_restart \
  --output outputs/multiscale/historical_training_recovery/new_run
```

This checks two immutable real-data mid-epoch snapshots: GRU H60-return fold 4
and xPatch H60-return fold 5. Each case compares two continuous updates with one
update, exit code 73, and a second update in a fresh process. It reconstructs
the original training identity, purged inner split and scaler. All checkpoint
fields, including optimizer, permutation, cursor and RNG, must agree exactly.
The source snapshots remain unchanged; original workers and deadlines are not
modified. The check uses six fresh processes and eight updates in total, with
a 180-second overall admission and 60-second per-process cap. It reads training
labels but scores no validation targets. This is a bounded CPU engineering
check, separate from full-fold real-data recovery and synthetic CUDA recovery.

## Original split design and actual clock horizons

```bash
AAPL_CPU_IDS=24,26 AAPL_CPU_THREADS=2 CUDA_VISIBLE_DEVICES=-1 \
  bash scripts/python.sh -m multiscale.historical_design_audit \
  --coverage outputs/multiscale/reproduction/coverage_run1.json \
  --destination outputs/multiscale/reproduction/design_timing_run1
```

This metadata audit reconstructs every original fit, stopping, outer-training
and validation partition independently of the training splitter. All 400
partition hashes must match the original protocols, and each of the 50 paired
price/return designs must agree. It retains the original rolling and count-only
warmup rules, including strict label completion before the next decision. It
checks observed positions across each input/target window and the protected
cutoff. For completed folds, saved origins, decisions and target completions
must equal the admitted bar metadata; the separate first H60-return fold keeps
its original control/neural layout.

Only the timestamp, observed-mask, session and origin members of the admitted
NPZ files are materialized. No close, actual-target or prediction-value member
is loaded, and no training is performed. Full archive hashes still bind the
original files. This checks metadata consistency, not a new price-quality
audit or historical vendor-availability guarantee.

`DESIGN_TIMING_AUDIT.json` retains all 100 fold designs, partition counts and
purge gaps, as well as 20 task summaries. `FOLD_TIMING.csv` lists endpoint
elapsed-hour quantiles and cross-session fractions. Planned timing for an
unfinished fold is explicitly distinguished from completed saved-file checks
and experimental results. In particular, complete trading-hour H4/H8 paths
can cross overnight, weekend and holiday gaps; H does not specify elapsed
clock hours. Minute endpoints must remain in one session and last exactly H
minutes.

## Focused tests

```bash
AAPL_CPU_IDS=24,26 AAPL_CPU_THREADS=2 CUDA_VISIBLE_DEVICES=-1 \
  bash scripts/python.sh -m pytest -q \
  --basetemp tmp/historical-tools-pytest-new-run \
  multiscale/test_historical_coverage.py \
  multiscale/test_historical_minute_check.py \
  multiscale/test_historical_restore_check.py \
  multiscale/test_historical_results.py \
  multiscale/test_historical_design_audit.py
```

The completion checks exercise mismatched receipts, origins, protocols,
protected-period flags, changed source files and stale verification selection.
Metric checks cover directional abstentions, weighting, overlap and the strict
cutoff. Source-path checks reject traversal and changed bytes. The current
inventory check is skipped when its local research outputs are absent.
Compilation checks also cover duplicate or missing cells, changed completed
evidence, unequal fold sizes, missing session groups, degenerate intervals and
the distinction between raw elapsed envelopes and pause-adjusted estimates.
Design checks cover missing positions, cross-session clock gaps, strict purge
equality, protected completion timestamps and endian-stable index identities.
