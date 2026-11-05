"""Local F8 replay, recovery, latency measurement and independent saved-score recomputation."""
import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .common import OUT,digest,read_json,write_json,utc_now
from .inference import FrozenSystem
from .stream import Stream
from .panel import Panel
from .data import pages_frame,FIELDS
from aapl.data import read_source
from .evaluation import point_scores
from .hardware import runtime,ResourceGuard,inspect_hardware


def score_journal(path):
    saved=np.load(path);required={'target','prediction','rows','dates'}
    if not required<=set(saved.files):raise ValueError('Saved result lacks raw aligned predictions and targets')
    if len(np.unique(saved['rows']))!=len(saved['rows']):raise ValueError('Repeated origins in saved score file')
    return {'file_sha256':digest(path),'recomputed_raw_metrics':point_scores(saved['target'],saved['prediction']),
            'days':len(np.unique(saved['dates']))}


def replay(deployment,maximum_origins=256):
    device=runtime(2026)
    if device!='cpu':raise RuntimeError('The final local inference acceptance must run on the local host')
    panel=Panel.load();frozen=read_json(OUT/'system_freeze.json');config=frozen['selected_system']['config']
    start=time.perf_counter();predictor=FrozenSystem(deployment,device='cpu');cold=time.perf_counter()-start
    directory=OUT/'deployment_acceptance'/predictor.model_identity;directory.mkdir(parents=True,exist_ok=True)
    if (directory/'state.json').exists():raise ValueError('Acceptance replay already has state; do not overwrite its prediction journal')
    # Use only already known development data, and explicitly delayed receive times.
    origins=panel.origins['AAPL'];origins=origins[(panel.dates[origins]>='2026-01-01')&(panel.dates[origins]<='2026-03-31')]
    origins=origins[:maximum_origins]
    if not len(origins):raise ValueError('No admitted development replay origins')
    names=list(dict.fromkeys(['AAPL']+panel.background));frames={}
    for name in names:
        if name=='AAPL':
            original=read_json(OUT/'data/AAPL/manifest.json')['source'];frames[name]=read_source(Path(original['source_csv_read_only']))
        else:frames[name]=pages_frame(name)[0]
    scalers=predictor.models[0][1]
    stream=Stream(panel.stamps,panel.stocks,panel.background,config,scalers,predictor,predictor.model_identity,directory/'state.json')
    latencies=[];outputs=[];rejected=[];guard=ResourceGuard();first=int(origins[0])-240;last_row=first-1
    resumed=False
    for index,row in enumerate(origins):
        for current in range(last_row+1,int(row)+1):
            stamp=panel.stamps[current]
            for name in names:
                if stamp not in frames[name].index:continue
                values=frames[name].loc[stamp,list(FIELDS)].to_dict()
                if any(not np.isfinite(value) for value in values.values()):continue
                try:stream.ingest(name,stamp,values,stamp+pd.Timedelta(minutes=16))
                except ValueError as exc:rejected.append({'symbol':name,'bar_start':stamp.isoformat(),'reason':str(exc)})
        last_row=int(row);stamp=panel.stamps[row]
        start=time.perf_counter();output=stream.predict(stamp,stamp+pd.Timedelta(minutes=16));latencies.append(time.perf_counter()-start)
        outputs.append(output);guard.check()
        if not resumed and index>=len(origins)//2:
            stream.save();restored=Stream(panel.stamps,panel.stocks,panel.background,config,scalers,predictor,predictor.model_identity,directory/'state.json')
            restored.resume()
            if restored.predict(stamp)['status']!='already_generated':raise ValueError('Replay recovery repeated a prediction')
            stream=restored;resumed=True
    failed=[i for i,result in enumerate(outputs) if result['status']!='ok']
    if failed:
        write_json(directory/'failure.json',{'failed_origins':failed,'outputs':outputs,'missing_predictions_not_discarded':True})
        raise RuntimeError('Replay contains failed origins; no success score is computed')
    prediction=np.asarray([r['predicted_log_return'] for r in outputs]);target=panel.targets['AAPL'][origins]
    prediction_path=directory/'predictions.npz'
    np.savez_compressed(prediction_path,rows=origins,dates=panel.dates[origins],target=target,prediction=prediction)
    # Independently rerun the same origins as bounded batches using the saved weights.
    batch=[]
    for first in range(0,len(origins),config['batch_size']):batch.append(predictor.predict_origin(panel,origins[first:first+config['batch_size']])['prediction'])
    batch=np.concatenate(batch)
    difference=float(np.max(np.abs(batch-prediction)))
    tolerance=max(1e-8,float(np.sqrt(np.mean(target**2)))*1e-4)
    report={'model_identity':predictor.model_identity,'origins':len(origins),'mode':'delayed_historical_replay',
            'receive_timestamp_policy':'simulated bar-start +16min; not claimed as actual historical receipt',
            'generation_timestamp_policy':'simulated replay clock; actual wall clock also saved per output',
            'hardware':inspect_hardware(False),'rejected_input_bars':rejected,
            'cold_load_seconds':cold,'warm_single_origin_p95_seconds':float(np.quantile(latencies,.95)),
            'warm_single_origin_median_seconds':float(np.median(latencies)),
            'latency_includes_stream_input_preparation_model_and_durable_state_write':True,
            'resume_without_duplicate_prediction':resumed,'batch_stream_max_prediction_difference':difference,
            'batch_stream_tolerance':tolerance,'recomputed_scores':score_journal(prediction_path),
            'resources':guard.check(),'Q3_accessed':False,'completed_at_utc':utc_now()}
    report['engineering_acceptance_passed']=bool(cold<=60 and report['warm_single_origin_p95_seconds']<=1 and resumed and difference<=tolerance)
    write_json(directory/'report.json',report);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--deployment',type=Path);parser.add_argument('--score-file',type=Path)
    parser.add_argument('--origins',type=int,default=256);args=parser.parse_args()
    if (args.deployment is None)==(args.score_file is None):parser.error('Choose exactly one of --deployment or --score-file')
    print(score_journal(args.score_file) if args.score_file else replay(args.deployment,args.origins))
