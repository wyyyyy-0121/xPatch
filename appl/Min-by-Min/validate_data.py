"""Validate OHLCV, provenance, trading coverage and complete aggregation."""
import argparse
import json
from pathlib import Path
from data_quality import aggregate_bars, data_gate, load_ohlcv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',required=True,type=Path)
    parser.add_argument('--profile',required=True,choices=['annual','quarterly','monthly','hourly','minute'])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    profiles=json.loads((Path(__file__).parent/'profiles.json').read_text())
    raw,meta=load_ohlcv(args.csv)
    interval=profiles['profiles'][args.profile]['bar_interval']
    bars=aggregate_bars(raw,interval)
    reasons=data_gate(args.profile,bars,meta,profiles['requirements'])
    result=dict(status='insufficient_data' if reasons else 'eligible',reasons=reasons,
                metadata=meta,complete_bars=len(bars),bar_interval=interval)
    content=json.dumps(result,indent=2)+'\n'
    if args.output:
        args.output.write_text(content)
    print(content)
    return 2 if reasons else 0


if __name__=='__main__':
    raise SystemExit(main())
