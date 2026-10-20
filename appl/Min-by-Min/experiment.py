"""AAPL multi-scale validation with common origins and explicit held-out evaluation."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data_quality import (INTRADAY, aggregate_bars, as_market, data_gate,
                          expected_index, load_ohlcv, sha256)
from forecast_models import (BASELINES, FEATURE_ORDER, MODEL_ORDER, baseline,
                             compute_manifest, predict_model, train_model)

HERE = Path(__file__).resolve().parent
PROFILE_PATH = HERE / 'profiles.json'


@dataclass
class ExperimentProfile:
    symbol: str
    bar_interval: str
    lookbacks: list
    horizons: list
    target_modes: list = field(default_factory=lambda: ['future_close', 'future_log_return'])
    data_source: str = 'local CSV with manifest'


@dataclass
class DataManifest:
    path: str
    sha256: str
    metadata: dict


@dataclass
class EvaluationResult:
    profile: str
    status: str
    rankings: list
    reasons: list


def features(bars):
    close = bars.close.to_numpy(dtype=float)
    if not len(close):
        return np.empty((0, 5), dtype='float32')
    return np.column_stack([np.log(bars.open), np.log(bars.high),
                            np.log(bars.low), np.log(close),
                            np.log1p(bars.volume)]).astype('float32')


def sequence_ids(bars, interval):
    if interval in INTRADAY:
        cached = bars.attrs.get('sequence_ids')
        if cached is not None and len(cached) == len(bars):
            return np.asarray(cached, dtype=np.int64)
        grid = expected_index(bars.index, interval)
        return grid.get_indexer(as_market(bars.index))
    if interval == '1mo':
        return bars.index.year.to_numpy()*12 + bars.index.month.to_numpy()
    # Daily data cannot compress missing exchange sessions either.
    from data_quality import schedule
    if not len(bars):
        return np.array([], dtype=int)
    dates = bars.index.tz_localize(None).normalize() if bars.index.tz else bars.index.normalize()
    calendar = schedule(str(dates.min().date()), str(dates.max().date())).index
    calendar = calendar.tz_localize(None).normalize() if calendar.tz else calendar.normalize()
    return calendar.get_indexer(dates)


def eligible_origins(bars, lookback, horizon, interval):
    if lookback <= 0 or horizon < 0:
        raise ValueError('lookback must be positive and horizon non-negative')
    n = len(bars)
    if n < lookback + horizon:
        return np.array([], dtype=np.int64)
    ids = sequence_ids(bars, interval)
    bad = (np.diff(ids) != 1) | (ids[:-1] < 0) | (ids[1:] < 0)
    cumulative = np.r_[0, np.cumsum(bad)]
    origins = np.arange(lookback-1, n-horizon, dtype=np.int64)
    return origins[(cumulative[origins+horizon]-cumulative[origins-lookback+1] == 0)
                   & (ids[origins] >= 0)]


def make_samples(bars, lookback, horizon, interval, target_mode, allowed_origins=None):
    origins = eligible_origins(bars, lookback, horizon, interval)
    if allowed_origins is not None:
        origins = np.intersect1d(origins, allowed_origins)
    close = bars.close.to_numpy(dtype=float)
    if not len(origins):
        return (np.empty((0, lookback, 5), dtype='float32'),
                np.empty((0, horizon)), origins, close)
    f = features(bars)
    xids = origins[:, None] + np.arange(-lookback+1, 1)
    yids = origins[:, None] + np.arange(1, horizon+1)
    y = close[yids]
    if target_mode == 'future_log_return':
        y = np.log(y / close[origins, None])
    elif target_mode != 'future_close':
        raise ValueError(f'Unknown target mode {target_mode}')
    return f[xids], y, origins, close


def rolling_splits(origins, dates=None, folds=5, purge=1, validation_ranges=None):
    order = np.argsort(origins)
    if validation_ranges is None:
        chunks = np.array_split(order, folds+1)
        validation_ranges = [(int(origins[c[0]]), int(origins[c[-1]])+1) for c in chunks[1:] if len(c)]
    splits = []
    for start, end in validation_ranges:
        tr = order[origins[order]+purge < start]
        va = order[(origins[order] >= start) & (origins[order] < end)]
        if len(tr) and len(va):
            splits.append((tr, va))
    return splits


def holdout_indices(origins, horizon, test_fraction=.2, cutoff_origin=None):
    empty = np.array([], dtype=np.int64)
    if not len(origins):
        return empty, empty
    if cutoff_origin is None:
        cut = int(len(origins)*(1-test_fraction))
        if cut <= 0 or cut >= len(origins):
            return empty, empty
        cutoff_origin = origins[cut]
    return np.flatnonzero(origins+horizon < cutoff_origin), np.flatnonzero(origins >= cutoff_origin)


def scale_fold(X_train, X_eval, y_train):
    from sklearn.preprocessing import StandardScaler
    feature = StandardScaler().fit(X_train.reshape(-1, 5))
    target = StandardScaler().fit(y_train)
    return (feature.transform(X_train.reshape(-1, 5)).reshape(X_train.shape),
            feature.transform(X_eval.reshape(-1, 5)).reshape(X_eval.shape),
            target.transform(y_train), target)


def years_covered(index):
    return (index[-1]-index[0]).total_seconds()/31556952 if len(index) else 0.


def price_space(y, target_mode, last_close):
    return np.maximum(y, 1e-12) if target_mode == 'future_close' else last_close[:, None]*np.exp(np.clip(y, -20, 20))


def mase_denominator(bars, cutoff, interval):
    # Raw training history ends before validation; independent of input window.
    values = bars.close.to_numpy(dtype=float)[:cutoff]
    ids = sequence_ids(bars, interval)[:cutoff]
    differences = np.abs(np.diff(values))[np.diff(ids) == 1]
    return float(differences.mean()) if len(differences) else 0.


def metrics(actual, predicted, last_close, mase_scale):
    if not np.isfinite(predicted).all():
        raise ValueError('Non-finite prediction')
    err = predicted-actual
    returns = np.log(actual/last_close[:, None])
    pred_returns = np.log(np.maximum(predicted, 1e-12)/last_close[:, None])
    return dict(MAE=float(np.abs(err).mean()), RMSE=float(np.sqrt((err**2).mean())),
                sMAPE=float((np.abs(err)/np.maximum((np.abs(actual)+np.abs(predicted))/2, 1e-12)).mean()*100),
                MASE=float(np.abs(err).mean()/max(mase_scale, 1e-12)),
                mase_scale=float(mase_scale),
                direction_accuracy=float((np.sign(returns) == np.sign(pred_returns)).mean()),
                log_return_RMSE=float(np.sqrt(((returns-pred_returns)**2).mean())),
                per_step=[dict(step=i+1, MAE=float(np.abs(err[:,i]).mean()),
                               RMSE=float(np.sqrt((err[:,i]**2).mean()))) for i in range(err.shape[1])])


def acceptance(record):
    return dict(median_MASE_below_1=record['median_MASE'] < 1,
                baseline_improvement_at_least_2_percent=record['median_baseline_RMSE_gain'] >= .02,
                at_least_3_of_5_folds_improve=sum(f.get('baseline_rmse_gain', 0) > 0 for f in record['folds']) >= 3)


def selection_key(record):
    return (record['median_normalized_return_error'], record['return_error_std'],
            MODEL_ORDER.index(record['model']), record['lookback'])


def code_hash():
    import hashlib
    digest = hashlib.sha256()
    for name in ('experiment.py', 'data_quality.py', 'forecast_models.py', 'test_ledger.py', 'predict.py', 'profiles.json'):
        digest.update((HERE/name).read_bytes())
    for path in sorted((HERE/'models').glob('*.py')):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_profile(name, csv_path, out_dir, lookbacks=None, requested_models=None,
                bar_interval=None, horizons=None, final=False, freeze_path=None, ledger_path=None):
    csv_path, out_dir = Path(csv_path), Path(out_dir)
    config = json.loads(PROFILE_PATH.read_text())
    p = dict(config['profiles'][name])
    interval = bar_interval or p['bar_interval']
    allowed = {'annual':{'1mo'}, 'quarterly':{'1mo'}, 'monthly':{'1d'}, 'hourly':{'1h'}, 'minute':{'1min','5min'}}
    if interval not in allowed[name]:
        raise ValueError(f'{name} does not support interval {interval}')
    if name == 'minute' and interval == '5min':
        p.update(lookbacks=[6,12,24,48,96], horizons=[1,3,6,12])
    windows = sorted(set(lookbacks or p['lookbacks']))
    spans = sorted(set(horizons or p['horizons']))
    if min(windows+spans) <= 0:
        raise ValueError('Lookbacks and horizons must be positive')
    models = requested_models or MODEL_ORDER
    if set(models)-set(MODEL_ORDER):
        raise ValueError('Unknown model')
    models = ['last_close'] + [m for m in MODEL_ORDER[1:] if m in models]
    raw, meta = load_ohlcv(csv_path)
    bars = aggregate_bars(raw, interval)
    profile = ExperimentProfile('AAPL', interval, windows, spans)
    cutoff = int(len(bars)*.8)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = dict(profile=name, symbol='AAPL', status='validation_complete',
                  generated_utc=datetime.now(timezone.utc).isoformat(),
                  data=meta, source_sha256=sha256(csv_path), code_sha256=code_hash(),
                  data_manifest=asdict(DataManifest(str(csv_path.resolve()), sha256(csv_path), meta)),
                  bars=len(bars), history_years=years_covered(bars.index), bar_interval=interval,
                  experiment_config={**asdict(profile), 'models':models, 'rolling_folds':5,
                                     'purge_gap_bars':'equal to horizon',
                                     'holdout_boundary':bars.index[cutoff].isoformat() if len(bars) else None,
                                     'validation':'common origin intersection; expanding purged five-fold; final 20% excluded',
                                     'selection':'acceptance gates then normalized return RMSE, variability, simplicity',
                                     'neural':{'maximum_epochs':20,'early_stopping_patience':3,
                                               'stopping_partition':'purged final 15% of outer training only',
                                               'seed':2026,'batch_size':128,'learning_rate':.001,
                                               'weight_decay':.0001,'gradient_clip':1.,'width':32,
                                               'layers':1,'dropout':.1,'mlp_width':64},
                                     'hourly_semantics':'complete 60-minute bars from exchange open; terminal partial bar dropped; horizons count valid bars across sessions'},
                  compute=compute_manifest(), rankings=[], skipped=[], selections=[],
                  hybrid={'status':'not_enabled', 'reason':'validation OOF complementarity and stable gains unproven'},
                  final_test={'status':'not_requested', 'results':[]},
                  data_limitations=[] if meta.get('source_verified') else ['Source remains unverified: results are exploratory/regression evidence.'])
    issues = data_gate(name, bars, meta, config['requirements'])
    if issues:
        result.update(status='insufficient_data', reasons=issues)
        return write_result(out_dir, result)
    if final:
        if freeze_path is None:
            raise ValueError('--final requires a frozen validation configuration (--freeze)')
        frozen = json.loads(Path(freeze_path).read_text())
        for key in ('source_sha256', 'code_sha256', 'experiment_config'):
            if frozen[key] != result[key]:
                raise ValueError(f'Frozen configuration mismatch: {key}')
        if not meta.get('source_verified'):
            raise ValueError('Final evaluation requires verified provenance')
        from test_ledger import claim_period
        claim_period(ledger_path or HERE/'results'/'test_use_ledger.json', 'AAPL',
                     bars.index[cutoff].isoformat(), bars.index[-1].isoformat(),
                     str(out_dir.resolve()), sha256(Path(freeze_path)))
        result['selections'] = frozen['selections']
        result['rankings'] = frozen['rankings']
    else:
        for horizon in spans:
            feasible = []
            for window in windows:
                candidate = eligible_origins(bars, window, horizon, interval)
                candidate = candidate[candidate+horizon < cutoff]
                if len(candidate) >= 12 and len(rolling_splits(candidate, folds=5, purge=horizon)) == 5:
                    feasible.append(window)
                else:
                    result['skipped'].append(dict(horizon=horizon,lookback=window,
                                                   reason='window cannot support five purged development folds'))
            if not feasible:
                continue
            # Use the same maximal-window origins for ALL candidate windows, including training.
            common = eligible_origins(bars, max(feasible), horizon, interval)
            development, _ = holdout_indices(common, horizon, cutoff_origin=cutoff)
            if len(development) < 12:
                result['skipped'].append(dict(horizon=horizon, reason='too few common purged development origins'))
                continue
            common_dev = common[development]
            splits = rolling_splits(common_dev, folds=5, purge=horizon)
            if len(splits) != 5:
                result['skipped'].append(dict(horizon=horizon, reason='cannot form five nonempty purged folds'))
                continue
            for mode in profile.target_modes:
                for window in feasible:
                    X, y, origins, closes = make_samples(bars, window, horizon, interval, mode, common_dev)
                    for model_name in models:
                        scores = []
                        for fold, (tr, va) in enumerate(splits, 1):
                            if model_name in BASELINES:
                                pred = baseline(model_name, closes, origins[va], horizon, window, interval, mode)
                                train_pred = baseline(model_name, closes, origins[tr], horizon, window, interval, mode)
                            else:
                                bundle = train_model(model_name, X[tr], y[tr], horizon)
                                pred = price_space(predict_model(bundle, X[va]), mode, closes[origins[va]])
                                train_pred = price_space(predict_model(bundle, X[tr]), mode, closes[origins[tr]])
                            actual = closes[origins[va,None]+np.arange(1,horizon+1)]
                            # Training raw history ends at the last TRAIN label, strictly before validation.
                            scale_cut = int(origins[tr].max()+horizon+1)
                            scale = mase_denominator(bars, scale_cut, interval)
                            score = metrics(actual, pred, closes[origins[va]], scale)
                            naive = metrics(actual, baseline('last_close', closes, origins[va], horizon, window, interval, mode), closes[origins[va]], scale)
                            train_actual = closes[origins[tr,None]+np.arange(1,horizon+1)]
                            train_score = metrics(train_actual, train_pred, closes[origins[tr]], scale)
                            train_naive = metrics(train_actual, baseline('last_close', closes, origins[tr], horizon,
                                                  window, interval, mode), closes[origins[tr]], scale)
                            train_score['normalized_return_error'] = train_score['log_return_RMSE']/max(train_naive['log_return_RMSE'],1e-12)
                            gain = 1-score['RMSE']/max(naive['RMSE'],1e-12)
                            score['normalized_return_error'] = score['log_return_RMSE']/max(naive['log_return_RMSE'],1e-12)
                            scores.append(dict(fold=fold, metrics=score, train_metrics=train_score,
                                               baseline_rmse_gain=float(gain),
                                               train_samples=len(tr), validation_samples=len(va),
                                               origin_sha256=__import__('hashlib').sha256(origins[va].tobytes()).hexdigest(),
                                               train_label_end=bars.index[scale_cut-1].isoformat(),
                                               validation_start=bars.index[origins[va[0]]].isoformat(),
                                               validation_end=bars.index[origins[va[-1]]+horizon].isoformat(),
                                               price_examples=[dict(timestamp=bars.index[origins[va[j]]+horizon].isoformat(),
                                                                    actual=float(actual[j,-1]),predicted=float(pred[j,-1]))
                                                               for j in np.unique(np.linspace(0,len(va)-1,min(100,len(va)),dtype=int))]))
                        row = dict(model=model_name, target_mode=mode, lookback=window, horizon=horizon, folds=scores,
                                   comparison_lookback=max(feasible),
                                   median_MASE=float(np.median([f['metrics']['MASE'] for f in scores])),
                                   median_RMSE=float(np.median([f['metrics']['RMSE'] for f in scores])),
                                   median_normalized_return_error=float(np.median([f['metrics']['normalized_return_error'] for f in scores])),
                                   return_error_std=float(np.std([f['metrics']['normalized_return_error'] for f in scores])),
                                   median_baseline_RMSE_gain=float(np.median([f['baseline_rmse_gain'] for f in scores])),
                                   baseline_improvement_fraction=float(np.mean([f['baseline_rmse_gain'] > 0 for f in scores])))
                        row['acceptance_criteria'] = acceptance(row)
                        row['accepted'] = model_name != 'last_close' and all(row['acceptance_criteria'].values())
                        row['diagnostic'] = diagnose(row)
                        result['rankings'].append(row)
        result['rankings'].sort(key=selection_key)
        for horizon in spans:
            for mode in profile.target_modes:
                rows = [r for r in result['rankings'] if r['horizon']==horizon and r['target_mode']==mode]
                passing = [r for r in rows if r['accepted']]
                selected = min(passing, key=selection_key) if passing else next((r for r in rows if r['model']=='last_close'), None)
                if selected:
                    result['selections'].append({k:selected[k] for k in ('model','target_mode','lookback','horizon','accepted','comparison_lookback')})
        result['acceptance_criteria'] = dict(accepted_run_exists=any(r['accepted'] for r in result['rankings']))
        if not result['rankings']:
            result.update(status='insufficient_data', reasons=['No configuration supports five purged folds; see skipped'])
            return write_result(out_dir, result)
    # Export deployment artifacts using development labels only, also for baseline fallback.
    for selected in result['selections']:
        horizon, window, mode, model_name = (selected[k] for k in ('horizon','lookback','target_mode','model'))
        common = eligible_origins(bars, selected['comparison_lookback'], horizon, interval)
        if not final:
            common = common[common+horizon < cutoff]
        X, y, origins, closes = make_samples(bars, window, horizon, interval, mode, common)
        tr, te = holdout_indices(origins, horizon, cutoff_origin=cutoff)
        bundle = (dict(name=model_name, lookback=window, horizon=horizon) if model_name in BASELINES
                  else train_model(model_name, X[tr], y[tr], horizon))
        bundle.update(schema_version=1, target_mode=mode, bar_interval=interval, symbol='AAPL',
                      feature_order=FEATURE_ORDER, source_sha256=result['source_sha256'], code_sha256=result['code_sha256'],
                      training_label_end=bars.index[origins[tr[-1]]+horizon].isoformat(),
                      split_settings=result['experiment_config'], data_metadata=meta)
        artifact = out_dir / f'{mode}_h{horizon}.joblib'
        joblib.dump(bundle, artifact)
        selected['artifact'] = str(artifact.resolve())
        if final:
            if not len(te):
                raise ValueError('No final samples')
            pred = (baseline(model_name, closes, origins[te], horizon, window, interval, mode) if model_name in BASELINES
                    else price_space(predict_model(bundle, X[te]), mode, closes[origins[te]]))
            actual = closes[origins[te,None]+np.arange(1,horizon+1)]
            result['final_test']['results'].append({**selected,
                'metrics':metrics(actual,pred,closes[origins[te]],mase_denominator(bars,cutoff,interval)),
                'test_start':bars.index[origins[te[0]]+1].isoformat(),
                'test_end':bars.index[origins[te[-1]]+horizon].isoformat()})
    if final:
        result.update(status='final_complete')
        result['final_test']['status'] = 'claimed_and_evaluated_once'
    else:
        save_validation_plots(result, out_dir)
        # Keep plot examples in their own CSVs; all metrics still use every origin.
        for row in result['rankings']:
            for fold in row['folds']:
                fold.pop('price_examples', None)
        result['freeze_sha256'] = None
        frozen_path = out_dir/'frozen_config.json'
        frozen_path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        result['freeze_sha256'] = sha256(frozen_path)
    return write_result(out_dir, result)


def save_validation_plots(result, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for selected in result['selections']:
        row = next(r for r in result['rankings'] if all(r[k] == selected[k]
                   for k in ('model','target_mode','lookback','horizon')))
        points = [p for fold in row['folds'] for p in fold['price_examples']]
        df = pd.DataFrame(points)
        stem = f"{selected['target_mode']}_h{selected['horizon']}_validation_prices"
        df.to_csv(out_dir/(stem+'.csv'), index=False)
        fig, ax = plt.subplots(figsize=(10,4))
        dates = pd.to_datetime(df.timestamp, utc=True)
        ax.plot(dates,df.actual,label='Actual close',lw=1)
        ax.plot(dates,df.predicted,label=f"{selected['model']} forecast",lw=1)
        ax.set(ylabel='USD',title=f"AAPL {result['profile']}: validation, lead {selected['horizon']} bars")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out_dir/(stem+'.png'),dpi=140)
        plt.close(fig)


def diagnose(row):
    train_error = np.median([f['train_metrics']['normalized_return_error'] for f in row['folds']])
    ratios = [f['metrics']['log_return_RMSE']/max(f['train_metrics']['log_return_RMSE'],1e-12) for f in row['folds']]
    if row['return_error_std'] > .3:
        return 'unstable_across_folds: inspect market regimes and data quality'
    if np.median(ratios) > 2:
        return 'possible_overfit_or_regime_change: reduce features/capacity, regularize, rolling retrain'
    if train_error >= 1 and row['median_normalized_return_error'] >= 1:
        return 'possible_underfit_or_target_data_issue: audit labels/features/frequency/history before capacity'
    if row['median_normalized_return_error'] >= 1:
        return 'no_baseline_advantage: inspect data/target/alignment before increasing capacity'
    return 'candidate_advantage_requires_all_acceptance_gates'


def write_result(out_dir, result):
    (out_dir/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (out_dir/'REPORT.md').write_text(render_report(result))
    return result


def render_report(result):
    lines = [f"# {result['profile']} AAPL", '', f"Status: `{result['status']}`", '',
             f"Final test: `{result['final_test']['status']}`. Historical test periods are recorded in the shared exposure ledger.", '',
             '## Data limitations', '', *['- '+r for r in result.get('data_limitations',[])],
             *['- '+r for r in result.get('reasons',[])], '',
             '## Validation selection', '',
             'Candidates must pass MASE < 1, median baseline RMSE gain >= 2%, and improvement in at least 3/5 folds.',
             'Passing candidates are ordered by normalized log-return RMSE, fold variability, then simplicity. Otherwise select last close.', '',
             '| Target | Model | Window | Horizon | Accepted |', '|---|---|---:|---:|---|']
    for r in result.get('selections',[]):
        lines.append(f"| {r['target_mode']} | {r['model']} | {r['lookback']} | {r['horizon']} | {r['accepted']} |")
    lines += ['', '## Window and model comparison', '',
              '| Target | Model | Window | Horizon | MASE | Return error / baseline | RMSE gain | Fold std |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in result['rankings']:
        lines.append(f"| {r['target_mode']} | {r['model']} | {r['lookback']} | {r['horizon']} | {r['median_MASE']:.4f} | {r['median_normalized_return_error']:.4f} | {r['median_baseline_RMSE_gain']:.2%} | {r['return_error_std']:.4f} |")
    lines += ['', 'Full per-fold, per-step and training errors, skips, hashes and diagnostic flags: `report.json`.',
              'Price charts/CSVs show at most 100 final-lead examples per fold; metrics use all validation samples.',
              'Diagnostic actions and hybrid/stopping rules: `DIAGNOSTICS.md` in the project root.',
              'Hourly bars exclude the incomplete terminal 30 minutes; horizons count complete trading bars across sessions.',
              'No predictive advantage is claimed unless every acceptance gate passes.', '']
    return '\n'.join(lines)


_compute_manifest = compute_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True, choices=['annual','quarterly','monthly','hourly','minute'])
    parser.add_argument('--csv', type=Path)
    parser.add_argument('--output', type=Path, default=HERE/'results'/'validation')
    parser.add_argument('--lookbacks', type=int, nargs='+')
    parser.add_argument('--horizons', type=int, nargs='+')
    parser.add_argument('--bar-interval', choices=['1min','5min','1h','1d','1mo'])
    parser.add_argument('--models', nargs='+', choices=MODEL_ORDER)
    parser.add_argument('--final', action='store_true')
    parser.add_argument('--freeze', type=Path)
    args = parser.parse_args(argv)
    source = args.csv or HERE/'data'/('AAPL_1d.csv' if args.profile in ('annual','quarterly','monthly') else 'AAPL_1min.csv')
    result = run_profile(args.profile,source,args.output/args.profile,args.lookbacks,args.models,
                         args.bar_interval,args.horizons,args.final,args.freeze)
    print(json.dumps(dict(status=result['status'],reasons=result.get('reasons',[]), report=str(args.output/args.profile/'REPORT.md')),indent=2))
    return 2 if result['status']=='insufficient_data' else 0


if __name__ == '__main__':
    raise SystemExit(main())
