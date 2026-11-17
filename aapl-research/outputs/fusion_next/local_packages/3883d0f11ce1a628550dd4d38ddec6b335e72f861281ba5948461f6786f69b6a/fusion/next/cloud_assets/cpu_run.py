"""Portable CPU package launcher; integrity and environment checks precede model loading."""
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main():
    root=Path(__file__).resolve().parent;os.chdir(root)
    package=json.loads((root/'package.json').read_text())
    if package['identity']!=identity({k:v for k,v in package.items() if k!='identity'}):raise ValueError('CPU package identity differs')
    for name,row in package['files'].items():
        original=root/name;path=original.resolve()
        if not path.is_relative_to(root) or not path.is_file() or original.is_symlink():raise ValueError('CPU package file is missing or unsafe: '+name)
        with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
        if sha!=row['sha256']:raise ValueError('CPU package file differs: '+name)
    lock=json.loads((root/'runtime_lock.json').read_text())
    if sys.version_info[:2]!=tuple(map(int,lock['python'].split('.')[:2])):raise RuntimeError('Python major/minor differs from acceptance environment')
    for name,version in lock['packages'].items():
        if metadata.version(name)!=version:raise RuntimeError('Runtime distribution differs: '+name)
    if not hasattr(os,'sched_getaffinity'):raise RuntimeError('Use Linux/WSL for the accepted CPU runtime')
    available=set(os.sched_getaffinity(0))
    if os.environ.get('AAPL_CPU_IDS'):
        selected=set()
        for part in os.environ['AAPL_CPU_IDS'].split(','):
            bounds=list(map(int,part.split('-')))
            selected.update(range(bounds[0],bounds[-1]+1))
    else:selected=set(sorted(available&set(range(18)))[:4])
    if not selected or not selected<=available&set(range(18)):raise ValueError('CPU IDs must remain within available research CPUs 0-17')
    threads=int(os.environ.get('AAPL_CPU_THREADS',min(4,len(selected))))
    if not 1<=threads<=len(selected):raise ValueError('Thread count exceeds CPU allocation')
    os.sched_setaffinity(0,selected)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
                 'RAYON_NUM_THREADS','ARROW_NUM_THREADS','POLARS_MAX_THREADS','AAPL_CPU_THREADS'):os.environ[name]=str(threads)
    os.environ.update(FUSION_EXECUTION_PROFILE='local',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                      HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',
                      WANDB_MODE='disabled',COMET_MODE='DISABLED',PYTHONPATH=str(root),CUDA_VISIBLE_DEVICES='')
    for name,folder in (('HF_HOME','hf'),('TORCH_HOME','torch'),('XDG_CACHE_HOME','cache'),('TMPDIR','tmp')):
        path=root/'.cache'/folder;path.mkdir(parents=True,exist_ok=True);os.environ[name]=str(path)
    if (root/'weights/chronos2').exists():os.environ['FUSION_CHRONOS_WEIGHTS']=str(root/'weights/chronos2')
    from fusion.next.common import source_identity
    if source_identity()!=package['source_identity']:raise ValueError('Packaged source identity differs from the frozen code')
    if sys.argv[1:]==['--verify-only']:
        print(json.dumps({'verified':True,'model_identity':package['model_identity'],'files':len(package['files'])}));return
    from fusion.next.serve import run
    if sys.argv[1:]==['--example']:
        example=json.loads((root/'example.json').read_text());state=root/'outputs/example/state.json'
        with (root/'example.jsonl').open() as feed:
            result=run(root/'deployment',example['calendar_start'],example['calendar_end'],state,state.exists(),example['mode'],feed)
        saved=json.loads(state.read_text());output=saved['outputs'][example['origin']]
        if abs(output['predicted_log_return']-example['expected_prediction'])>1e-8:raise RuntimeError('Packaged example prediction differs from accepted output')
    else:
        import argparse
        parser=argparse.ArgumentParser();parser.add_argument('--calendar-start',required=True);parser.add_argument('--calendar-end',required=True)
        parser.add_argument('--state',type=Path,default=root/'outputs/feed/state.json');parser.add_argument('--resume',action='store_true')
        parser.add_argument('--mode',choices=('delayed_historical_replay','timely_forecast'),default='delayed_historical_replay')
        args=parser.parse_args();result=run(root/'deployment',args.calendar_start,args.calendar_end,args.state,args.resume,args.mode)
    print(json.dumps(result),file=sys.stderr)
    if result['rejected_events'] or result['failed_forecasts']:raise SystemExit(1)


if __name__=='__main__':main()
