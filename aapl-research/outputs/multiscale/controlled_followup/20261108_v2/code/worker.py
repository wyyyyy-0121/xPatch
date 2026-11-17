from common import *
import argparse,fcntl,traceback

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True)
    ap.add_argument('--bench-seconds',type=float);ap.add_argument('--steps',type=int)
    args=ap.parse_args();jobs=read(BASE/'protocol/ALL_JOBS.json')['jobs']
    cfg=next(j for j in jobs if j['id']==args.job)
    folder=BASE/'runs'/cfg['id'];folder.mkdir(exist_ok=True)
    with (folder/'LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (folder/'DONE.json').exists():raise SystemExit(0)
        try:
            from engine import run
            print(json.dumps(run(cfg,args.bench_seconds,args.steps)),flush=True)
        except Exception:
            atomic(folder/'ERROR.json',{'job':cfg,'traceback':traceback.format_exc()});raise
