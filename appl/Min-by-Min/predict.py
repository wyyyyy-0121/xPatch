"""Predict from an explicitly selected, trusted local experiment artifact."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from data_quality import INTRADAY, aggregate_bars, expected_index, load_ohlcv, schedule
from experiment import code_hash, eligible_origins, features, price_space
from forecast_models import BASELINES, FEATURE_ORDER, baseline, predict_model


def future_timestamps(last, horizon, interval):
    if interval in INTRADAY:
        span = pd.DatetimeIndex([last, last+pd.Timedelta(days=max(30,horizon))])
        grid = expected_index(span, interval)
        return grid[grid > last][:horizon]
    if interval == '1mo':
        return pd.date_range(last+pd.offsets.MonthBegin(1), periods=horizon, freq='MS')
    days = schedule(str(last.date()), str((last+pd.Timedelta(days=max(60,3*horizon))).date())).index
    days = days.tz_localize(None) if days.tz else days
    if last.tz:
        days = days.tz_localize(last.tz)
    return days[days > last][:horizon]


def predict_artifact(artifact, csv):
    bundle = joblib.load(artifact)  # Only load trusted local pickle/joblib artifacts.
    if bundle.get('schema_version') != 1 or bundle.get('feature_order') != FEATURE_ORDER:
        raise ValueError('Unsupported artifact schema or feature order')
    if bundle['code_sha256'] != code_hash():
        raise ValueError('Artifact preprocessing/model code hash differs from current code')
    raw, meta = load_ohlcv(Path(csv))
    bars = aggregate_bars(raw, bundle['bar_interval'])
    if bundle['bar_interval'] in INTRADAY:
        from data_quality import data_gate
        issues = data_gate('minute', bars, meta, {'minute':{'minimum_session_coverage':.98}})
        if issues:
            raise ValueError('; '.join(issues))
    window, horizon = bundle['lookback'], bundle['horizon']
    valid = eligible_origins(bars, window, 0, bundle['bar_interval'])
    if not len(valid) or valid[-1] != len(bars)-1:
        raise ValueError('Latest input window is incomplete or has missing trading bars')
    closes = bars.close.to_numpy(dtype=float)
    origin = np.array([len(bars)-1])
    if bundle['name'] in BASELINES:
        price = baseline(bundle['name'],closes,origin,horizon,window,bundle['bar_interval'],bundle['target_mode'])[0]
    else:
        price = price_space(predict_model(bundle,features(bars)[-window:][None]),bundle['target_mode'],closes[-1:])[0]
    timestamps = future_timestamps(bars.index[-1],horizon,bundle['bar_interval'])
    return dict(symbol='AAPL', origin=bars.index[-1].isoformat(), model=bundle['name'],
                training_label_end=bundle['training_label_end'],
                timestamp_convention='start of target bar; predicted close occurs at bar end',
                predictions=[dict(step=i+1,timestamp=ts.isoformat(),future_close=float(p),
                                  future_log_return=float(np.log(p/closes[-1])))
                             for i,(ts,p) in enumerate(zip(timestamps,price))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--csv',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    content = json.dumps(predict_artifact(args.artifact,args.csv),indent=2,allow_nan=False)+'\n'
    if args.output:
        args.output.write_text(content)
    print(content)


if __name__ == '__main__':
    main()
