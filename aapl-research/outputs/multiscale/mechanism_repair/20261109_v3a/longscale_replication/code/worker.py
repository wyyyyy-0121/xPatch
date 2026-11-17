"""Use the unchanged v3 engine with an isolated output directory."""
from common import *
import argparse, fcntl, traceback

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',required=True)
    parser.add_argument('--steps-limit',type=int)
    args=parser.parse_args()
    manifest=read(BASE/'protocol/JOBS.json')
    assert implementation_hashes()==manifest['wrapper_hashes']
    assert source_hashes()==manifest['training_hashes']
    assert sha(V3/'protocol/BUDGET.json')==manifest['budget_sha256']
    cfg=next(j for j in manifest['jobs'] if j['id']==args.job)
    out=folder(cfg,manifest)
    assert out.is_relative_to(BASE/'runs'), 'Reused fits must not be relaunched'
    out.mkdir(parents=True,exist_ok=True)
    with (out/'LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            from engine import run
            result=run(cfg,out,steps_limit=args.steps_limit)
            print(result['status'],flush=True)
        except Exception:
            atomic(out/'ERROR.json',{'at_utc':now(),'traceback':traceback.format_exc()})
            raise
