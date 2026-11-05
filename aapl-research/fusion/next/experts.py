"""Balanced same-information CPU fits and auditable quarterly OOF expert storage."""
from pathlib import Path
import time

import numpy as np
from scipy.linalg import solve
import torch

from .common import OUT, digest, read_json, write_json, utc_now
from .expert_models import tabular
from .hardware import ResourceGuard


NAMES=('ridge','lightgbm','xpatch','patchtst')


def scope_key(information='ohlcv',training_scope='multistock',market_information=True):
    if information not in ('ohlcv','close') or training_scope not in ('multistock','aapl'):
        raise ValueError('Expert scope outside prescribed comparison')
    return information+'_'+training_scope+('_market' if market_information else '_no_market')


def balanced_rows(panel,fit_end,seed,per_day=16,only_aapl=False):
    rng=np.random.default_rng(seed);stocks=[];rows=[];weights=[]
    chosen=[(i,s) for i,s in enumerate(panel.stocks) if not only_aapl or s=='AAPL']
    for sid,name in chosen:
        available=panel.origins[name];available=available[panel.dates[available]<=fit_end]
        days=panel.dates[available];groups=np.split(available,np.flatnonzero(days[1:]!=days[:-1])+1)
        for group in groups:
            if not len(group):continue
            selected=np.sort(rng.choice(group,min(per_day,len(group)),replace=False))
            stocks.extend([sid]*len(selected));rows.extend(selected.tolist())
            weights.extend([1/(len(chosen)*len(groups)*len(selected))]*len(selected))
    weights=np.asarray(weights,dtype=np.float64);weights*=len(weights)/weights.sum()
    return np.asarray(stocks),np.asarray(rows),weights


def matrix(panel,stocks,rows,config,scalers,batch_size=256,with_labels=True):
    # Aggregate windows in bounded batches instead of retaining an N*L*C tensor.
    x=None;y=np.empty(len(rows),dtype=np.float64) if with_labels else None
    for first in range(0,len(rows),batch_size):
        stop=first+batch_size
        batch=panel.windows(stocks[first:stop],rows[first:stop],config,scalers,include_labels=with_labels)
        values=tabular(batch,len(panel.stocks))
        if x is None:x=np.empty((len(rows),values.shape[1]),dtype=np.float32)
        x[first:stop]=values
        if with_labels:y[first:stop]=batch['target'].numpy()
    if x is None:raise ValueError('Empty supervised design')
    return x,y


def fit_cpu(panel,config,fit_end,directory,seed=2026):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    scalers=panel.scaler(fit_end);unit=scalers['AAPL']['target_std']
    stock,rows,weight=balanced_rows(panel,fit_end,seed,config.get('rows_per_day',16),
                                    config.get('training_scope')=='aapl')
    start=time.perf_counter();x,y=matrix(panel,stock,rows,config,scalers);features_seconds=time.perf_counter()-start
    y=y/unit;guard=ResourceGuard();guard.check();start=time.perf_counter()
    if config['family']=='ridge':
        # No implicit intercept: stock indicators already provide stock-specific offsets.
        z=x.astype(np.float64);weighted=z*weight[:,None]
        gram=z.T@weighted;rhs=weighted.T@y
        gram.flat[::gram.shape[0]+1]+=config.get('alpha',1000.)
        coefficient=solve(gram,rhs,assume_a='pos')
        np.savez(directory/'ridge.npz',coefficient=coefficient,unit=unit)
        model_path=directory/'ridge.npz'
        predict=lambda values:values@coefficient*unit
    elif config['family']=='lightgbm':
        import lightgbm as lgb
        params={key:config[key] for key in ('num_leaves','max_depth','min_child_samples','learning_rate','reg_lambda','colsample_bytree')}
        model=lgb.LGBMRegressor(**params,n_estimators=config.get('trees',1000),n_jobs=config.get('cpu_threads',8),
                                verbosity=-1,deterministic=True,force_col_wise=True,random_state=seed)
        model.fit(x,y,sample_weight=weight);model_path=directory/'lightgbm.txt';model.booster_.save_model(str(model_path))
        predict=lambda values:model.predict(values,num_threads=config.get('cpu_threads',8))*unit
    else:raise ValueError('CPU family must be Ridge or LightGBM')
    report={'family':config['family'],'config':config,'fit_end':fit_end,'fit_start':str(panel.dates[rows].min()),
            'seed':seed,'training_rows':len(rows),'stock_count':len(np.unique(stock)),
            'balanced_stock_day_weights':True,'feature_seconds':features_seconds,
            'fit_seconds':time.perf_counter()-start,'model_sha256':digest(model_path),'target_unit':unit,
            'resources':guard.check(),'created_at_utc':utc_now()}
    write_json(directory/'summary.json',report)
    write_json(directory/'scalers.json',{name:{key:value.tolist() if isinstance(value,np.ndarray) else value for key,value in row.items()}
                                        for name,row in scalers.items()})
    return predict,scalers,report


def predict_cpu(predict,panel,stock,rows,config,scalers,batch_size=1024):
    result=[]
    for first in range(0,len(rows),batch_size):
        x,_=matrix(panel,stock[first:first+batch_size],rows[first:first+batch_size],config,scalers,with_labels=False)
        result.append(np.asarray(predict(x),dtype=np.float64))
    values=np.concatenate(result)
    if not np.isfinite(values).all():raise FloatingPointError('Nonfinite supervised expert predictions')
    return values


def write_oof(panel,information,fold,predictions,configurations,training_scope='multistock',market_information=True):
    """Predictions keyed by stock, with four columns in NAMES order; never in-sample."""
    if fold['quarter']=='2026Q2':raise ValueError('Q2 confirmation must not enter development OOF storage')
    required=panel.stocks if training_scope=='multistock' else ['AAPL']
    if set(predictions)!=set(required):raise ValueError('OOF quarter missing an admitted training stock')
    key=scope_key(information,training_scope,market_information)
    root=OUT/'oof'/key;root.mkdir(parents=True,exist_ok=True)
    receipt_path=root/(fold['quarter']+'.json')
    if receipt_path.exists():raise ValueError('Completed OOF quarter is immutable')
    files={}
    for name,(rows,values) in predictions.items():
        if name not in panel.stocks:raise ValueError('OOF stock outside admitted set')
        rows=np.asarray(rows);values=np.asarray(values,dtype=np.float32)
        dates=panel.dates[rows]
        if values.shape!=(len(rows),4) or not np.isfinite(values).all():raise ValueError('Incomplete OOF expert columns')
        if np.any((dates<fold['score_start'])|(dates>fold['score_end'])):raise ValueError('OOF row outside its quarter')
        if not fold['fit_end']<fold['purged_full_session']<fold['score_start']:raise ValueError('OOF fit/purge boundary invalid')
        path=root/(name+'.npy')
        if not path.exists():
            array=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(panel.dates),4));array[:]=np.nan;array.flush()
        array=np.lib.format.open_memmap(path,mode='r+')
        existing=np.asarray(array[rows]);already=np.isfinite(existing).all(1)
        # Retry after an interrupted write can replay identical values only.
        if np.any(already & np.any(existing!=values,axis=1)):raise ValueError('Attempt to rewrite existing OOF predictions')
        array[rows]=values;array.flush()
        quarter_path=root/(fold['quarter']+'_'+name+'.npz')
        np.savez_compressed(quarter_path,rows=rows,predictions=values)
        files[name]={'rows':len(rows),'quarter_file':quarter_path.name,'sha256':digest(quarter_path)}
    write_json(receipt_path,{'fold':fold,'expert_columns':list(NAMES),'configurations':configurations,
                            'stocks':files,'completed_at_utc':utc_now(),'labels_after_fit_used':False})
