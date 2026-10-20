"""Regression coverage for leakage, calendars, provenance and deployment artifacts."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_quality import expected_index, schedule, sha256
from experiment import (aggregate_bars, data_gate, eligible_origins, holdout_indices,
                        load_ohlcv, make_samples, metrics, rolling_splits, run_profile,
                        scale_fold, acceptance, selection_key)
from forecast_models import baseline, train_model, predict_model
from predict import predict_artifact, future_timestamps
from test_ledger import claim_period, migrate_history


def frame(index):
    close = 100*np.exp(np.arange(len(index))*.00001 + .001*np.sin(np.arange(len(index))))
    return pd.DataFrame(dict(open=close, high=close+1,low=close-1,close=close,
                             volume=np.full(len(index),100)),index=index)


def trading(start='2024-01-02',end='2024-01-05',interval='1min'):
    return expected_index(pd.DatetimeIndex([start,end]),interval)


def save(df, path, manifest=None):
    df.rename_axis('timestamp').to_csv(path)
    if manifest is not None:
        manifest = {**manifest,'sha256':sha256(path)}
        path.with_suffix('.csv.manifest.json').write_text(json.dumps(manifest))
    return path


def test_csv_sort_duplicate_and_empty(tmp_path):
    path=tmp_path/'bars.csv'
    path.write_text('timestamp,open,high,low,close,volume\n2024-01-02 09:32,11,12,10,11,3\n2024-01-02 09:31,10,11,9,10,2\n2024-01-02 09:32,11,13,10,12,4\n')
    df,meta=load_ohlcv(path)
    assert df.index.is_monotonic_increasing and len(df)==2 and df.iloc[-1].close==12
    assert meta['duplicate_timestamps_removed']==1
    path.write_text('timestamp,open,high,low,close,volume\n')
    with pytest.raises(ValueError,match='no rows'):
        load_ohlcv(path)


def test_provenance_boolean_and_source_string_cannot_bypass(tmp_path):
    path=save(frame(trading()[:2]),tmp_path/'bars.csv',
              dict(source_verified=True, source='trusted provider',timezone='America/New_York'))
    bars,meta=load_ohlcv(path)
    assert not meta['source_verified']
    reasons=data_gate('minute',bars,meta,{'minute':{}})
    assert any('provenance' in r for r in reasons)
    path.write_text(path.read_text()+'2024-01-02 09:32-05:00,100,101,99,100,1\n')
    with pytest.raises(ValueError,match='SHA-256'):
        load_ohlcv(path)


def test_two_minutes_are_not_full_coverage_and_missing_whole_session(tmp_path):
    index=pd.DatetimeIndex(['2024-01-02 09:30','2024-01-02 09:31',
                            '2024-01-04 09:30','2024-01-04 09:31'])
    _,meta=load_ohlcv(save(frame(index),tmp_path/'two.csv'))
    assert meta['calendar_minute_coverage']==pytest.approx(4/1170)
    assert meta['minimum_session_coverage']==0
    assert meta['missing_sessions']==1


def test_calendar_holiday_early_close_and_dst():
    july=trading('2024-07-03','2024-07-05')
    assert len(july)==210+390
    assert not any(july.day==4)
    dst=trading('2024-03-08','2024-03-11')
    assert dst[0].utcoffset()!=dst[-1].utcoffset()
    assert len(dst)==780
    # A weekend-only file has no eligible bars, without calendar construction errors.
    assert len(trading('2024-01-06','2024-01-07'))==0


def test_incomplete_aggregation_and_session_anchoring():
    df=frame(trading('2024-01-02','2024-01-02'))
    complete=aggregate_bars(df,'5min')
    assert len(complete)==78 and complete.iloc[0].volume==500
    hourly=aggregate_bars(df,'1h')
    assert len(hourly)==6 and hourly.index[0].minute==30
    damaged=aggregate_bars(df.drop(df.index[2]),'5min')
    assert len(damaged)==77 and df.index[0] not in damaged.index
    assert len(aggregate_bars(df.drop(df.index[2]),'1h'))==5
    _,_,origins,_=make_samples(damaged,2,2,'5min','future_close')
    assert all(i>=1 for i in origins)


def test_cross_day_labels_and_480_window_and_hour8():
    df=frame(trading())
    X,y,origins,closes=make_samples(df,480,60,'1min','future_log_return')
    assert X.shape[1:]==(480,5) and y.shape[1]==60 and len(X)>0
    assert np.allclose(y[0],np.log(closes[origins[0]+1:origins[0]+61]/closes[origins[0]]))
    assert any(df.index[i].date()!=df.index[i+60].date() for i in origins)
    hourly=aggregate_bars(df,'1h')
    X,y,_,_=make_samples(hourly,12,8,'1h','future_close')
    assert len(X)>0 and y.shape[1]==8


def test_gaps_cannot_be_compressed_even_across_missing_day():
    df=frame(trading())
    df=df.loc[df.index.date!=pd.Timestamp('2024-01-03').date()]
    _,_,origins,_=make_samples(df,30,15,'1min','future_close')
    for i in origins:
        assert not (df.index[i-29].date()==pd.Timestamp('2024-01-02').date()
                    and df.index[i+15].date()>=pd.Timestamp('2024-01-04').date())
    damaged=frame(trading()[:100]).drop(trading()[50])
    _,_,origins,_=make_samples(damaged,10,5,'1min','future_close')
    assert all(i+5<50 or i-9>=50 for i in origins)


def test_empty_sample_shapes_and_holdout():
    X,y,o,_=make_samples(frame(trading()[:10]),96,8,'1h','future_close')
    assert X.shape==(0,96,5) and y.shape==(0,8)
    assert all(len(a)==0 for a in holdout_indices(o,8,cutoff_origin=8))


@pytest.mark.parametrize('name',['annual','quarterly','monthly','hourly','minute'])
def test_five_profile_shapes(name):
    profiles=json.loads((Path(__file__).resolve().parents[1]/'profiles.json').read_text())
    p=profiles['profiles'][name]
    interval=p['bar_interval']; lookback=p['lookbacks'][0]; horizon=p['horizons'][0]
    if interval=='1mo':
        idx=pd.date_range('2000-01-01',periods=100,freq='MS')
    elif interval=='1d':
        idx=schedule('2020-01-01','2021-01-01').index
    else:
        idx=trading(interval=interval)
    X,y,_,_=make_samples(frame(idx),lookback,horizon,interval,'future_close')
    assert len(X)>0 and X.shape[1:]==(lookback,5) and y.shape[1]==horizon


def test_purge_and_scaler_isolation():
    origins=np.arange(200)
    tr,te=holdout_indices(origins,8)
    assert origins[tr].max()+8<origins[te].min()
    for train,valid in rolling_splits(origins[tr],purge=8):
        assert origins[tr[train]].max()+8<origins[tr[valid]].min()
    rng=np.random.default_rng(0)
    X=rng.normal(size=(20,4,5)); y=rng.normal(size=(20,3))
    a,_,b,c=scale_fold(X,X[:5],y)
    d,_,e,f=scale_fold(X,X[:5]+1000,y)
    assert np.array_equal(a,d) and np.array_equal(b,e) and np.array_equal(c.mean_,f.mean_)


def test_seasonal_never_uses_future_labels():
    closes=np.arange(1.,101.)
    origins=np.array([20])
    before=baseline('seasonal',closes,origins,30,12,'1mo','future_close')
    closes[21:]=1e9
    after=baseline('seasonal',closes,origins,30,12,'1mo','future_close')
    assert np.array_equal(before,after)


@pytest.fixture
def monthly_csv(tmp_path):
    idx=pd.date_range('2000-01-01',periods=360,freq='MS')
    df=frame(idx).astype(float)
    return save(df,tmp_path/'monthly.csv')


def test_common_origins_baseline_mase_invariance_and_default_no_test(monthly_csv,tmp_path):
    result=run_profile('annual',monthly_csv,tmp_path/'out',lookbacks=[12,60],requested_models=['last_close','ridge'])
    assert result['status']=='validation_complete' and len(result['rankings'])==8
    assert result['final_test']=={'status':'not_requested','results':[]}
    for mode in ('future_close','future_log_return'):
        rows=[r for r in result['rankings'] if r['model']=='last_close' and r['target_mode']==mode]
        assert rows[0]['median_MASE']==rows[1]['median_MASE']
        assert [f['origin_sha256'] for f in rows[0]['folds']]==[f['origin_sha256'] for f in rows[1]['folds']]
        assert [f['train_samples'] for f in rows[0]['folds']]==[f['train_samples'] for f in rows[1]['folds']]
    assert (tmp_path/'out'/'frozen_config.json').exists()
    for selected in result['selections']:
        if not selected['accepted']:
            assert selected['model']=='last_close'
    prediction=predict_artifact(result['selections'][0]['artifact'],monthly_csv)
    assert len(prediction['predictions'])==12
    assert prediction['predictions'][0]['timestamp'].startswith('2030-01-01')
    with pytest.raises(ValueError,match='requires a frozen'):
        run_profile('annual',monthly_csv,tmp_path/'final',lookbacks=[12],requested_models=['last_close'],final=True)


@pytest.mark.parametrize('name',['ridge','mlp','gru','lstm','xpatch'])
def test_model_artifact_roundtrip(name,tmp_path):
    rng=np.random.default_rng(10)
    x=rng.normal(size=(24,6,5)).astype('float32')
    y=rng.normal(size=(24,2))
    bundle=train_model(name,x,y,2)
    a=predict_model(bundle,x[:3])
    path=tmp_path/'model.joblib'; joblib.dump(bundle,path)
    b=predict_model(joblib.load(path),x[:3])
    assert np.allclose(a,b) and a.shape==(3,2)
    if name!='ridge':
        import torch
        if torch.cuda.is_available():
            assert bundle['trained_device']=='cuda'


def test_cross_directory_test_reuse_and_history_migration(tmp_path):
    history=tmp_path/'old'/'report.json'; history.parent.mkdir()
    history.write_text(json.dumps({'symbol':'AAPL','final_test':{'results':[{'test_start':'2014-01-01','test_end':'2022-06-17'}]}}))
    ledger=tmp_path/'test_use_ledger.json'
    with pytest.raises(ValueError,match='overlaps'):
        claim_period(ledger,'AAPL','2020-01-01','2021-01-01','different_folder','abc')
    assert json.loads(ledger.read_text())[0]['status']=='historically_exposed_exploratory_regression'
    claim_period(ledger,'AAPL','2023-01-01','2024-01-01','new','abc')
    with pytest.raises(ValueError,match='overlaps'):
        claim_period(ledger,'AAPL','2023-06-01','2024-06-01','another_folder','def')


def test_future_timestamps_skip_holidays_and_hourly_partial_bin():
    last=pd.Timestamp('2024-07-03 11:30',tz='America/New_York')
    future=future_timestamps(last,2,'1h')
    assert future[0]==pd.Timestamp('2024-07-05 09:30',tz='America/New_York')


def test_synthetic_interpolation_gate(tmp_path):
    idx=pd.date_range('2024-01-02',periods=2000,freq='min')
    df=frame(idx).astype(float)
    df.loc[:,:]=np.linspace([100,101,99,100,1000000],[102,103,101,102,2000000],len(df))
    _,meta=load_ohlcv(save(df,tmp_path/'synthetic.csv'))
    assert meta['suspected_interpolation']
    reasons=data_gate('minute',df,meta,{'minute':{}})
    assert any('synthetic' in r for r in reasons)
