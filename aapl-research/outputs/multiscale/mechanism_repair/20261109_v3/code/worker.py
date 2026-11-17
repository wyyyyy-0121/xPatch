from study import *
import argparse, fcntl, traceback

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--steps',type=int)
    p.add_argument('--seconds',type=float);p.add_argument('--folder',type=Path);a=p.parse_args()
    cfg=next(j for j in read(BASE/'protocol/JOBS.json')['jobs'] if j['id']==a.job)
    out=a.folder or BASE/'runs'/cfg['id'];out.mkdir(parents=True,exist_ok=True)
    with (out/'LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            from engine import run
            result=run(cfg,out,a.steps,a.seconds)
            print(json.dumps({'status':result['status'],'job':cfg['id']}),flush=True)
        except Exception:
            atomic(out/'ERROR.json',{'at_utc':now(),'traceback':traceback.format_exc()});raise
