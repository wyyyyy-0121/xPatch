from common import *
import argparse,fcntl,traceback

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);ap.add_argument('--bench-seconds',type=float)
    args=ap.parse_args();jobs=read(BASE/'protocol/JOBS.json')['jobs'];cfg=next(j for j in jobs if j['id']==args.job)
    folder=BASE/'runs'/cfg['id'];folder.mkdir(exist_ok=True)
    with (folder/'LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (folder/'DONE.json').exists():raise SystemExit(0)
        try:
            from engine import run
            result=run(cfg,bench_seconds=args.bench_seconds)
            print(json.dumps({'status':result['status'],'job':cfg['id']}),flush=True)
        except Exception:
            atomic(folder/'ERROR.json',{'traceback':traceback.format_exc(),'job':cfg})
            raise
