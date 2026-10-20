# AAPL Clean 1-minute data audit

Status: `exploratory_only_quality_gate_failed`

File SHA-256: `54d27ebb6ab2ed686ae38f1fcba5f1ff4dd7e48573b7fd2a7064b59ab40e4bb4`. Rows: 436,906, from 2022-03-07T09:30:00-05:00 to 2026-10-16T15:59:00-04:00. No duplicates. The data are strongly non-linear and pass the interpolation heuristic.

HF Data Library's [download page](https://hfdatalibrary.com/pages/download?) describes one-minute IEX HIST data. Its [methodology](https://hfdatalibrary.com/pages/docs) specifies Eastern Time, bars labelled 09:30 through 15:59, IEX-only volume, split adjustment, and the Clean pipeline. The [known-issues page](https://hfdatalibrary.com/pages/issues) records that 2024-12-23 is cut short. The source field in this CSV is `iex`. The exact local file bytes were not matched to a published release checksum.

Observed regular-session coverage is 97.00%, below the project's 98% gate; median session coverage is 97.18%. 790 sessions are below 98%. The publisher says bars with no IEX trades and zero-volume bars are absent; we do not fill those gaps. Samples intersecting gaps are discarded. The unusually short Dec. 23, 2024 session has 59 rows (2024-12-23 09:30:00-05:00 through 2024-12-23 13:33:00-05:00).

The Clean pipeline also removes outliers using a centered 50-bar window, so future neighboring bars can affect which historical bars remain. Results from this file are exploratory and cannot establish a causal minute forecasting advantage. It is IEX-only data, not consolidated-market OHLCV. The provider's published price-basis notice changed on 2026-10-06; this file's release version is not matched to a checksum.

No prices or missing minutes are synthesized. Formal model rankings remain withheld under the 98% requirement. See `report.json` for hashes, session distributions and source-page hashes.
