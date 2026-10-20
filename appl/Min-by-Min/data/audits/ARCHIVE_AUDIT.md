# Supplied archive audit

Decision: `rejected_synthetic_interpolation`

Nearly all OHLCV triplets are linear; weekend/off-session timestamps and fractional volumes indicate daily-to-minute interpolation. Future daily endpoints contaminate intermediate timestamps.

Rows: 8689184; range: 2006-01-03T00:00:00 to 2024-05-13T00:00:00

Linear triplet fraction: 0.9994862577397965; weekend rows: 2376000

The original ZIP is preserved. Filtering to market hours cannot remove future endpoint interpolation. Do not use it for minute forecasting. Full counts, SHA-256 and tolerances are in archive_audit.json.
