# AAPL Multi-Timescale Forecast Experiments

This directory contains a separate, reproducible research pipeline for annual, quarterly, monthly, hourly, and minute-scale AAPL forecasts. It compares future close and future log-return targets with last-close, drift, seasonal, Ridge, MLP, GRU, LSTM, and xPatch models. It does not run a trading strategy or estimate transaction costs.

## Install and run

From this directory, install the experiment dependencies:

```bash
python -m pip install -r requirements.txt
```

Download the longest history currently exposed by Yahoo Finance for daily or monthly bars:

```bash
python download_data.py --symbol AAPL --interval 1d
python download_data.py --symbol AAPL --interval 1mo
```

Run a profile using the matching CSV. The hourly and minute profiles use the local minute CSV by default:

```bash
python run_experiment.py --profile annual --csv data/AAPL_1mo.csv
python run_experiment.py --profile quarterly --csv data/AAPL_1mo.csv
python run_experiment.py --profile monthly --csv data/AAPL_1d.csv
python run_experiment.py --profile hourly --csv data/AAPL_1min.csv
python run_experiment.py --profile minute --csv data/AAPL_1min.csv
```

The minute profile defaults to one-minute bars. To aggregate to five-minute bars, pass `--bar-interval 5min`; its horizons become 1, 3, 6, and 12 bars (5, 15, 30, and 60 minutes).

Reports are written below `results/<profile>/`. A profile that misses its minimum history or data-verification requirement writes an `insufficient_data` report and returns exit code 2 without ranking models. Validation selects a configuration; the final chronological 20% is evaluated once afterward and kept separate in the report. A completed output directory cannot be reused for another final test. Use a new `--output` directory for another experiment. Use `--lookbacks` or `--models` for a smaller diagnostic run.

## Profiles and validation

The full lookback, horizon, and coverage settings are in `profiles.json`. Annual and quarterly tasks use monthly bars; monthly tasks use daily bars; hourly and minute tasks aggregate the supplied intraday CSV by session. Annual/quarterly/monthly require 15/10/5 years, hourly requires one year, and minute requires six months plus verifiable intraday provenance and timezone. Yahoo Finance intraday retention does not provide this project's full-history requirement; this pipeline does not fabricate missing minutes or fill overnight gaps.

The input contains causal OHLCV-derived features. The target is either the future close or the future log return from the forecast origin. All methods share the same chronological folds and horizon-sized purge gap. Results retain fold-level MAE, RMSE, sMAPE, MASE, direction accuracy, normalized log-return error, per-step errors, and baseline improvements. Final test results do not affect selection.

Intraday local CSV timestamps must have a known timezone and verifiable source to pass the data gate. The included `AAPL_1min.csv` has naive timestamps and its source/adjustment policy are undocumented, so hourly and minute model rankings are intentionally withheld until that metadata is verified. It is not safe to infer regular-session boundaries from these timestamps.

For a documented local intraday file, add a sibling `<file>.manifest.json` with `source_verified: true`, a descriptive `source`, the file's `sha256`, and an IANA `timezone` (for example `America/New_York`) when CSV timestamps are naive. The hash must match. This is an explicit provenance assertion and should only be set when the provider and file can be checked. Samples crossing missing bars or session boundaries are dropped; the pipeline never forward-fills across them.

## Compute

PyTorch models use CUDA when it is available and log the selected GPU. On the user's RTX 5070 Ti Laptop GPU this selects CUDA device 0. CPU runs set PyTorch to 24 threads and use 24 DataLoader workers; CUDA runs use zero DataLoader workers and transfer batches to the GPU. If CUDA is unavailable, the run reports CPU use rather than claiming GPU execution.

## Provenance and scope

The downloader writes a sidecar JSON manifest with provider URL, interval, retrieval time, coverage, timezone, adjustment setting, and SHA-256. The downloaded Yahoo Finance data can change as the provider revises history. Local minute files remain user-supplied and are hashed in each result report.

The previous `run.py`, `train.py`, and `data_processor.py` are retained for historical reference only. Their old README performance claims and automatic install/train behavior are not part of this pipeline. Use `run_experiment.py` for new research.
