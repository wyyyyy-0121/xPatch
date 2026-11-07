# Saved historical study predictors

`study_package.py` provides the five-scale prediction interface for the actual
historical study. It retains all twenty task slots and all eight models. Each
available task uses its latest completed chronological fold; scores do not
choose a fold or model. A saved evaluation model is not a full-history
production refit.

The original exploration identity is preserved, including the separate minute
queue prefixes. A current packaging record binds the original admission and
protocol, a verified coverage inventory, the source manifest and the user's
permission confirmation. This enables the existing historical models to be
used without rewriting earlier receipts. It does not reclassify a study as a
prospectively admitted matrix or authorize additional training. The existing
`market:` export gate is retained.

## Export

Use a newly verified coverage snapshot and a new output directory. The
following paths identify the local study cohort; the output directory is an
example and must not already exist.

```bash
AAPL_CPU_IDS=24,26 AAPL_CPU_THREADS=2 CUDA_VISIBLE_DEVICES=-1 \
  bash scripts/python.sh -m multiscale.study_package export \
  --coverage outputs/multiscale/private_coverage_verified/20261108_package20_snapshot.json \
  --permission outputs/multiscale/permission_confirmations/20261108_ALPACA_KIBOT_USER_CONFIRMED.json \
  --source-manifest outputs/multiscale/data_provenance/20261108_CURRENT_DEVELOPMENT_SOURCE_MANIFEST.json \
  --admission outputs/multiscale/admission/alpaca_20261107T052228573242Z \
  --output outputs/multiscale/historical_inference_packages/new_package_run \
  --allow-partial
```

Omit `--allow-partial` for the final core comparison: export then requires all
100 task-folds to be complete. A partial package still has twenty registry
entries, with missing folds and unavailable tasks listed. Twenty callable
tasks and twenty complete five-fold comparisons are separate quantities.

`REGISTRY.json` contains task coverage and bundle hashes. Each available
bundle contains the four neural parameter files, Ridge coefficients and its
scaler, and an inference manifest. Raw price, saved target and saved forecast
arrays are not copied. Model and source identities are checked at load and
prediction. Keep the matching repository code and Python environment; a
parameter folder alone is insufficient.

## Forecast

The Python interface selects a task directly:

```python
from multiscale.study_package import load_study_package

study = load_study_package(package_directory,
                           expected_registry_sha256=registry_sha256,
                           device="cpu")
result = study.predict(history_bars, profile="minute", horizon=60,
                       target="future_log_return", model="xpatch",
                       as_of_ns=int(history_bars.end_ns[-1]),
                       history_identity=caller_data_version)
```

Profiles are `annual`, `quarterly`, `monthly`, `hourly` and `minute`; valid
horizons are defined in `spec.py`. Targets are `future_close` and
`future_log_return`. Models are `last_close`, `drift`, `seasonal`, `ridge`,
`mlp`, `gru`, `lstm` and `xpatch`.

Supply only completed historical bars ending at `as_of_ns`. The input contract
and XNYS calendar checks are the same as `inference.py`, documented in the
README. Incomplete inputs, skipped bars, incompatible tasks and a target
schedule crossing the protected cutoff are rejected. Minute targets remain in
the same regular session. Future observations and actual completion times are
reported as unknown; scheduled times are calendar expectations.

Forecasts carry `historical_exploration_forecast`, the selected fold and the
task's completed folds. They retain the source's prior-exposure status. Neural
single-origin inference can differ slightly from the original batch of 512
because floating-point operations depend on batch shape. Price predictions
are not silently clipped.

The equivalent CLI is:

```bash
AAPL_CPU_IDS=24,26 AAPL_CPU_THREADS=2 CUDA_VISIBLE_DEVICES=-1 \
  bash scripts/python.sh -m multiscale.study_package forecast \
  --package '<package directory>' --registry-sha256 '<exported SHA256>' \
  --bars '<past-only bars NPZ>' --profile minute --horizon 60 \
  --target future_log_return --model xpatch \
  --as-of-ns '<last completed bar UTC nanoseconds>' \
  --history-identity '<caller data version>'
```

## Verify the saved package

```bash
AAPL_CPU_IDS=24,26 AAPL_CPU_THREADS=2 CUDA_VISIBLE_DEVICES=-1 \
  bash scripts/python.sh -m multiscale.verify_study_package \
  --package '<package directory>' --registry-sha256 '<exported SHA256>' \
  --admission outputs/multiscale/admission/alpaca_20261107T052228573242Z \
  --output outputs/multiscale/historical_inference_packages/new_package_audit
```

The verifier starts a fresh process for every available task and compares all
eight models with their saved forecasts at one original validation origin.
It checks past-only inputs, returned scope, caller RNG preservation, future
schedule metadata and rejection of wrong horizons, model names, decision
times, unobserved inputs and omitted bars. It never opens saved target arrays
or starts training. Neural tolerance is the existing float32 API contract
(`rtol=1e-5`, `atol=1e-6`); baseline and Ridge checks use `rtol=1e-10`,
`atol=1e-12`. Differences are recorded per model and task in their own target
units. This bounded interface check complements the separate full-array
restoration audit and independent metric recalculation.
