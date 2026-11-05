"""Manifest-verified local inference from completed one-minute OHLCV bars."""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aapl.common import ROOT, digest, read_json, utc_now, write_json
from aapl.data import Market, read_source
from aapl.models import chronos_pipeline, chronos_predict
from .config import OUT, EXPERTS, source_hash
from .experts import load_expert
from .features import price_features, representation
from . import head


def market_from_frame(frame, as_of):
    """Build an inference-only Market without requesting or fabricating a label."""
    as_of=pd.Timestamp(as_of)
    if as_of.tzinfo is None:
        raise ValueError('as_of must include an explicit timezone')
    frame=frame.loc[frame.index+pd.Timedelta(minutes=1)<=as_of].tail(241).copy()
    if len(frame)!=241 or not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError('Need 241 distinct completed minute bars')
    if not np.all(np.diff(frame.index.as_unit('ns').asi8)==60_000_000_000):
        raise ValueError('Inference bars contain a gap')
    dates=frame.index.tz_convert('America/New_York').strftime('%Y-%m-%d').to_numpy(dtype='U10')
    if len(set(dates))!=1:
        raise ValueError('Inference context crosses an overnight boundary')
    local=frame.index[-1].tz_convert('America/New_York')
    if local.hour*60+local.minute not in range(810,948):
        raise ValueError('This model accepts origins from 13:30 through 15:47 ET')
    prev=np.r_[np.nan,frame.close.to_numpy()[:-1]]
    returns=np.log(frame.close.to_numpy()/prev)
    features=np.column_stack([np.log(frame[c].to_numpy()/prev) for c in ('open','high','low','close')]+
                             [np.log1p(frame.volume.to_numpy())])
    return Market(frame,dates,returns,features,np.array([240]),{'inference_only':True})


class FusionPredictor:
    def __init__(self, bundle=OUT/'bundle'):
        from .pipeline import assert_protocol,load_market
        protocol=assert_protocol()
        self.directory=Path(bundle)
        self.manifest=read_json(self.directory/'manifest.json')
        if self.manifest['source_sha256']!=source_hash():
            raise ValueError('Serving implementation differs from bundled version')
        for path,checksum in self.manifest['files'].items():
            if digest(ROOT/path)!=checksum:raise ValueError(f'Serving artifact changed: {path}')
        market=load_market()
        quarter=protocol['confirmation_quarter']
        self.experts={name:load_expert(name,protocol['expert_selections'][name],OUT/'oof'/quarter/name,market)
                      for name in EXPERTS[:-1]}
        self.pipe,_=chronos_pipeline(device='cuda')
        self.chronos_config=protocol['expert_selections']['chronos2_frozen']['trial']
        self.projection=np.load(OUT/'projection.npy')
        self.heads=[head.load(OUT/'confirmation'/f'seed{seed}'/'head.pt') for seed in (2026,2027,2028)]

    def predict(self, frame, as_of):
        started=time.perf_counter()
        market=market_from_frame(frame,as_of);origins=np.array([240])
        experts=[self.experts[name](market,origins) for name in EXPERTS[:-1]]
        experts.append(chronos_predict(self.pipe,market,origins,self.chronos_config,batch_size=1))
        features,volatility,_=price_features(market,origins)
        embedding,_=representation(self.pipe,market,origins,projection=self.projection)
        data={'experts':np.column_stack(experts),'features':features,'volatility':volatility,'embedding':embedding}
        values=[float(head.predict(state,data)[0]) for state in self.heads]
        # Ensemble is an explicitly identified serving output; research primary is per-seed loss.
        value=float(np.mean(values));price=float(market.bars.close.iloc[-1]);stamp=market.bars.index[-1]
        return {'research_id':self.manifest['research_id'],'model_status':self.manifest['model_status'],
                'symbol':'AAPL','origin_bar_start':stamp.isoformat(),'input_completed_at':(stamp+pd.Timedelta(minutes=1)).isoformat(),
                'target_bar_start':(stamp+pd.Timedelta(minutes=12)).isoformat(),
                'horizon_minutes':12,'predicted_log_return':value,'derived_price':price*float(np.exp(value)),
                'per_seed_predictions':values,'expert_predictions':dict(zip(EXPERTS,map(float,data['experts'][0]))),
                'training_data_end':self.manifest['training_data_end'],'latency_seconds':time.perf_counter()-started,
                'serving_aggregation':'mean of three fusion head predictions',
                'interval_available':False}


def package(protocol,selection,result):
    directory=OUT/'bundle';directory.mkdir(parents=True,exist_ok=True)
    files={}
    for seed in (2026,2027,2028):
        p=OUT/'confirmation'/f'seed{seed}'/'head.pt';files[str(p.relative_to(ROOT))]=digest(p)
    for name in EXPERTS[:-1]:
        p=OUT/'oof'/protocol['confirmation_quarter']/name
        names=['summary.json','ridge.npz'] if name=='ridge' else ['summary.json','lightgbm.txt'] if name=='lightgbm' else ['summary.json','best.pt','best.receipt.json']
        for name2 in names:
            f=p/name2;files[str(f.relative_to(ROOT))]=digest(f)
    for p in (OUT/'projection.npy',OUT/'selection.json',OUT/'protocol.json'):
        files[str(p.relative_to(ROOT))]=digest(p)
    write_json(directory/'manifest.json',{'research_id':protocol['research_id'],'created_at_utc':utc_now(),
               'source_sha256':source_hash(),'files':files,'selected_config':selection['main']['config'],
               'model_status':'development_qualified_pending_final_test' if result['qualified'] else 'experimental_not_qualified',
               'training_data_end':result['training_max_date'],'foundation_weight_reference':
               read_json(OUT/'admission.json')['chronos_weight_identity'],'standalone_weights_copied':False})
    # Verify the full raw-bar inference path against the saved feature/prediction path.
    from .pipeline import load_market,load_oof
    market=load_market();data=load_oof([protocol['confirmation_quarter']])
    predictor=FusionPredictor(directory);samples=[];latencies=[];checks=[]
    for index in np.linspace(0,len(data['origins'])-1,12,dtype=int):
        origin=int(data['origins'][index]);frame=market.bars.iloc[origin-240:origin+1]
        output=predictor.predict(frame,frame.index[-1]+pd.Timedelta(minutes=1))
        expected=[head.predict(state,{k:data[k][index:index+1] for k in ['experts','features','embedding','volatility']})[0]
                  for state in predictor.heads]
        checks.append(bool(np.allclose(output['per_seed_predictions'],expected,rtol=1e-4,atol=1e-7)))
        latencies.append(output['latency_seconds']);samples.append(output)
    report={'passed':all(checks) and max(latencies)<60,'complete_raw_bar_path_matches_saved_predictions':all(checks),
            'samples':len(samples),'latency_p50_seconds':float(np.median(latencies)),
            'latency_p95_seconds':float(np.quantile(latencies,.95)),'latency_max_seconds':max(latencies),
            'first_prediction':samples[0],'bundle_manifest_sha256':digest(directory/'manifest.json'),
            'scope':'Warm local replay inference, no brokerage orders or live data connection'}
    write_json(OUT/'serving_verification.json',report)
    del predictor
    gc.collect();torch.cuda.empty_cache()
    if not report['passed']:raise RuntimeError('Full local inference verification failed')


if __name__=='__main__':
    from aapl.common import runtime
    runtime()
    parser=argparse.ArgumentParser(description='Predict from a local Alpaca-format one-minute OHLCV CSV')
    parser.add_argument('--csv',required=True);parser.add_argument('--as-of',required=True)
    args=parser.parse_args();frame=read_source(args.csv)
    print(json.dumps(FusionPredictor().predict(frame,args.as_of),ensure_ascii=False,indent=2))
