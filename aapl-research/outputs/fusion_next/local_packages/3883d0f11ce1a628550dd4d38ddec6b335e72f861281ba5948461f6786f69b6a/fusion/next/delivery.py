"""Local F8 replay, recovery, latency measurement and independent saved-score recomputation."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from .common import OUT,digest,identity,read_json,write_json,utc_now,source_identity
from .inference import FrozenSystem
from .stream import Stream
from .panel import Panel
from .data import pages_frame,FIELDS
from aapl.data import read_source
from .evaluation import point_scores
from .hardware import runtime,ResourceGuard,inspect_hardware


def score_journal(path):
    with np.load(path,allow_pickle=False) as saved:
        required={'target','prediction','rows','dates'}
        if not required<=set(saved.files):raise ValueError('Saved result lacks raw aligned predictions and targets')
        rows=saved['rows'];dates=saved['dates']
        if rows.ndim!=1 or dates.shape!=rows.shape or len(saved['target'])!=len(rows):raise ValueError('Saved origin/date/target alignment differs')
        if len(np.unique(rows))!=len(rows):raise ValueError('Repeated origins in saved score file')
        if rows.dtype.kind not in 'iu' or np.any(np.diff(rows)<=0):raise ValueError('Saved origins must be ordered integer rows')
        return {'file_sha256':digest(path),'recomputed_raw_metrics':point_scores(saved['target'],saved['prediction']),
                'days':len(np.unique(dates))}


def replay(deployment,maximum_origins=256,resume=False,stop_after=None):
    replay_started=time.perf_counter();cpu_started=time.process_time()
    if maximum_origins<2:raise ValueError('Recovery acceptance requires at least two origins')
    device=runtime(2026)
    if device!='cpu':raise RuntimeError('The final local inference acceptance must run on the local host')
    panel=Panel.load();frozen=read_json(OUT/'system_freeze.json');config=frozen['selected_system']['config']
    start=time.perf_counter();predictor=FrozenSystem(deployment,device='cpu');cold=time.perf_counter()-start
    directory=OUT/'deployment_acceptance'/predictor.model_identity;directory.mkdir(parents=True,exist_ok=True)
    if (directory/'report.json').exists():raise ValueError('Acceptance replay already completed; do not overwrite its evidence')
    if (directory/'state.json').exists() and not resume:raise ValueError('Acceptance replay already has state; use --resume')
    if resume and not (directory/'state.json').exists():raise ValueError('No committed replay state exists to resume')
    # Use only already known development data, and explicitly delayed receive times.
    origins=panel.origins['AAPL'];origins=origins[(panel.dates[origins]>='2026-01-01')&(panel.dates[origins]<='2026-03-31')]
    origins=origins[:maximum_origins]
    if not len(origins):raise ValueError('No admitted development replay origins')
    if stop_after is not None and not 0<stop_after<len(origins):raise ValueError('Controlled exit must leave at least one origin to resume')
    recipe={'model_identity':predictor.model_identity,'origin_identity':identity(origins.tolist()),
            'system_freeze_sha256':digest(OUT/'system_freeze.json'),'source_identity':source_identity(),
            'mode':'delayed_historical_replay','origins':len(origins)}
    progress_path=directory/'progress.json'
    if resume:
        progress=read_json(progress_path)
        if progress['recipe']!=recipe:raise ValueError('Cannot resume acceptance with changed origins, code or frozen system')
    else:progress={'recipe':recipe,'latencies_seconds':{},'processes':[],'rejected_input_bars':[]}
    progress['processes'].append({'pid':os.getpid(),'started_at_utc':utc_now(),'model_cold_load_seconds':cold})
    names=list(dict.fromkeys(['AAPL']+panel.background));frames={}
    for name in names:
        if name=='AAPL':
            original=read_json(OUT/'data/AAPL/manifest.json')['source'];frames[name]=read_source(Path(original['source_csv_read_only']))
        else:frames[name]=pages_frame(name)[0]
    scalers=predictor.models[0][1]
    stream=Stream(panel.stamps,panel.stocks,panel.background,config,scalers,predictor,predictor.model_identity,directory/'state.json')
    outputs=[];rejected=progress['rejected_input_bars'];guard=ResourceGuard();window_first=int(origins[0])-240;first=window_first
    for name in names:
        known=np.flatnonzero(panel.observed[name][:window_first+1])
        if not len(known):raise ValueError('Replay has no causal close preceding its first window')
        first=min(first,int(known[-1]))
    last_row=first-1
    if resume:
        stream.resume();keys=[panel.stamps[row].isoformat() for row in origins]
        completed=[key for key in keys if key in stream.outputs]
        if completed!=keys[:len(completed)] or set(stream.outputs)!=set(completed):
            raise ValueError('Saved replay outputs are not the exact common-origin prefix')
        if any(stream.outputs[key]['status']!='ok' for key in completed):raise RuntimeError('A failed saved origin cannot be dropped on resume')
        outputs=[stream.outputs[key] for key in completed]
        if completed:last_row=int(origins[len(completed)-1])
    completed_before=len(outputs)
    for index,row in enumerate(origins):
        if index<completed_before:continue
        start=time.perf_counter()
        for current in range(last_row+1,int(row)+1):
            stamp=panel.stamps[current]
            for name in names:
                if stamp not in frames[name].index:continue
                values=frames[name].loc[stamp,list(FIELDS)].to_dict()
                if any(not np.isfinite(value) for value in values.values()):continue
                try:stream.ingest(name,stamp,values,stamp+pd.Timedelta(minutes=16))
                except ValueError as exc:rejected.append({'symbol':name,'bar_start':stamp.isoformat(),'reason':str(exc)})
        last_row=int(row);stamp=panel.stamps[row]
        output=stream.predict(stamp,stamp+pd.Timedelta(minutes=16));latency=time.perf_counter()-start
        outputs.append(output);guard.check()
        progress['processes'][-1].setdefault('first_output_after_start_seconds',time.perf_counter()-replay_started)
        progress['latencies_seconds'][str(int(row))]=latency
        progress['processes'][-1].update(wall_seconds=time.perf_counter()-replay_started,
                    process_cpu_seconds=time.process_time()-cpu_started,resources=guard.check())
        progress.update(completed_origins=len(outputs),last_process_pid=os.getpid(),updated_at_utc=utc_now())
        write_json(progress_path,progress)
        if stop_after is not None and len(outputs)>=stop_after:
            return {'completed':False,'controlled_exit':True,'completed_origins':len(outputs),
                    'model_identity':predictor.model_identity,'pid':os.getpid(),'Q3_accessed':False}
    failed=[i for i,result in enumerate(outputs) if result['status']!='ok']
    if failed:
        write_json(directory/'failure.json',{'failed_origins':failed,'outputs':outputs,'missing_predictions_not_discarded':True})
        raise RuntimeError('Replay contains failed origins; no success score is computed')
    prediction=np.asarray([r['predicted_log_return'] for r in outputs]);target=panel.targets['AAPL'][origins]
    prediction_path=directory/'predictions.npz'
    np.savez_compressed(prediction_path,rows=origins,dates=np.asarray(panel.dates[origins],dtype='U10'),target=target,prediction=prediction)
    # Independently rerun the same origins as bounded batches using the saved weights.
    batch=[]
    for first in range(0,len(origins),config['batch_size']):batch.append(predictor.predict_origin(panel,origins[first:first+config['batch_size']])['prediction'])
    batch=np.concatenate(batch)
    difference=float(np.max(np.abs(batch-prediction)))
    tolerance=max(1e-8,float(np.sqrt(np.mean(target**2)))*1e-4)
    latencies=np.asarray([progress['latencies_seconds'][str(int(row))] for row in origins
                          if str(int(row)) in progress['latencies_seconds']])
    timing_complete=len(latencies)==len(origins)
    distinct_processes=len({p['pid'] for p in progress['processes']})>=2
    resumed=bool(resume and completed_before>0 and distinct_processes)
    if resume and completed_before:
        previous=panel.stamps[origins[completed_before-1]]
        if stream.predict(previous)['status']!='already_generated':raise ValueError('Process recovery repeated a saved prediction')
    report={'model_identity':predictor.model_identity,'origins':len(origins),'mode':'delayed_historical_replay',
            'receive_timestamp_policy':'simulated bar-start +16min; not claimed as actual historical receipt',
            'generation_timestamp_policy':'simulated replay clock; actual wall clock also saved per output',
            'hardware':inspect_hardware(False),'rejected_input_bars':rejected,
            'cold_load_seconds':max(p.get('first_output_after_start_seconds',p.get('wall_seconds',cold)) for p in progress['processes']),
            'cold_model_load_seconds':max(p['model_cold_load_seconds'] for p in progress['processes']),
            'cold_scope':'fresh process startup, frozen model load, historical source preparation, state/warmup and first output',
            'cold_load_measurements_by_process':progress['processes'],
            'warm_single_origin_p95_seconds':float(np.quantile(latencies,.95)) if len(latencies) else None,
            'warm_single_origin_median_seconds':float(np.median(latencies)) if len(latencies) else None,
            'timed_origins':len(latencies),'timing_complete':timing_complete,
            'latency_includes_stream_input_preparation_model_and_durable_state_write':True,
            'resume_without_duplicate_prediction':resumed,'distinct_exit_and_resume_processes':distinct_processes,
            'batch_stream_max_prediction_difference':difference,
            'batch_stream_tolerance':tolerance,'recomputed_scores':score_journal(prediction_path),
            'resources':guard.check(),'Q3_accessed':False,'completed_at_utc':utc_now()}
    report['engineering_acceptance_passed']=bool(report['cold_load_seconds']<=60 and timing_complete and
        report['warm_single_origin_p95_seconds']<=1 and resumed and difference<=tolerance)
    progress['processes'][-1].update(wall_seconds=time.perf_counter()-replay_started,
                process_cpu_seconds=time.process_time()-cpu_started,resources=guard.check())
    report['total_replay_wall_seconds']=sum(p.get('wall_seconds',0.) for p in progress['processes'])
    report['total_replay_process_cpu_seconds']=sum(p.get('process_cpu_seconds',0.) for p in progress['processes'])
    write_json(progress_path,progress)
    write_json(directory/'report.json',report);return report


def run_acceptance(deployment,maximum_origins=256):
    """Exercise a real process exit and fresh model load, sequentially on the local CPU."""
    manifest=read_json(Path(deployment)/'deployment.json');directory=OUT/'deployment_acceptance'/manifest['identity']
    if (directory/'report.json').exists():raise ValueError('Acceptance already completed; preserve its existing measurements')
    base=[sys.executable,'-m','fusion.next.delivery','--deployment',str(Path(deployment).resolve()),'--origins',str(maximum_origins)]
    commands=[]
    if not (directory/'state.json').exists():commands.append(base+['--stop-after',str(maximum_origins//2)])
    commands.append(base+['--resume'])
    for command in commands:
        result=subprocess.run(command,capture_output=True,text=True)
        write_json(directory/'process_logs'/f'{time.time_ns()}.json',{'arguments':command[2:],
                   'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'completed_at_utc':utc_now()})
        if result.returncode:raise RuntimeError('Local acceptance child failed; inspect its saved process log')
    return read_json(directory/'report.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--deployment',type=Path);parser.add_argument('--score-file',type=Path)
    parser.add_argument('--origins',type=int,default=256)
    continuation=parser.add_mutually_exclusive_group();continuation.add_argument('--resume',action='store_true')
    continuation.add_argument('--stop-after',type=int);args=parser.parse_args()
    if (args.deployment is None)==(args.score_file is None):parser.error('Choose exactly one of --deployment or --score-file')
    if args.origins<2:parser.error('--origins must be at least two for a process recovery demonstration')
    if args.score_file and (args.resume or args.stop_after is not None):parser.error('Resume/controlled exit only apply to replay')
    print(score_journal(args.score_file) if args.score_file else
          (replay(args.deployment,args.origins,args.resume,args.stop_after) if args.resume or args.stop_after is not None
           else run_acceptance(args.deployment,args.origins)))
