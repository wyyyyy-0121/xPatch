"""Download AAPL daily or monthly OHLCV and write a provenance manifest."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--interval", choices=["1d", "1mo"], required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        import yfinance as yf
    except ImportError as exc:
        raise SystemExit("Install yfinance with: python -m pip install -r requirements.txt") from exc
    output = args.output or Path(__file__).parent / "data" / f"{args.symbol}_{args.interval}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = yf.download(args.symbol, period="max", interval=args.interval, auto_adjust=False,
                        actions=False, progress=False, multi_level_index=False)
    if frame.empty:
        raise SystemExit("Yahoo Finance returned no rows")
    frame = frame.rename(columns={c: c.lower() for c in frame.columns})
    frame.index.name = "timestamp"
    frame = frame[["open", "high", "low", "close", "volume"]]
    frame.to_csv(output)
    manifest = {"symbol": args.symbol, "interval": args.interval, "provider": "Yahoo Finance chart API via yfinance",
                "provider_url": "https://query1.finance.yahoo.com/v8/finance/chart/" + args.symbol,
                "downloaded_utc": datetime.now(timezone.utc).isoformat(), "start": frame.index.min().isoformat(),
                "end": frame.index.max().isoformat(), "timezone": str(frame.index.tz), "rows": len(frame),
                "adjustment": "auto_adjust=False; unadjusted OHLC as returned", "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_suffix(output.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"csv": str(output), "rows": len(frame), "manifest": str(output.with_suffix(output.suffix + '.manifest.json'))}, indent=2))


if __name__ == "__main__":
    main()
