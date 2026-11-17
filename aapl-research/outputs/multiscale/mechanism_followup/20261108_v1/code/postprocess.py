"""Durable sequential analysis after the declared worker matrix finishes."""
from common import *
import subprocess,time,signal
from datetime import datetime,timezone

STOP=False
def stop(*_):
    global STOP
    STOP=True


def main():
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    audit=BASE/'audit/POSTPROCESS_STATE.json'
    while not (BASE/'audit/TRAINING_COMPLETE.json').exists():
        if STOP:return
        if (BASE/'audit/CONTROLLER_ERROR.json').exists():raise RuntimeError('Training controller requires review')
        errors=list((BASE/'runs').glob('*/ERROR.json'))
        if errors:raise RuntimeError('Unreviewed training error')
        atomic(audit,{'phase':'waiting_for_declared_training','at_utc':datetime.now(timezone.utc).isoformat()})
        time.sleep(10)
    steps=[['controls.py','shrink'],['analyze.py'],['verify_resume.py'],['verify_predictors.py'],
           ['inspect_scales.py'],['figures.py'],['write_paper.py'],['build_documents.py']]
    records=[]
    for step in steps:
        if STOP:return
        label=step[0].removesuffix('.py'); start=time.time()
        atomic(audit,{'phase':label,'completed':records,'at_utc':datetime.now(timezone.utc).isoformat()})
        with (BASE/'logs'/f'postprocess_{label}.log').open('a') as log:
            python='/home/ayaka/.venvs/doc-tools/bin/python' if label in ('figures','write_paper','build_documents') else sys.executable
            proc=subprocess.run([python]+step,cwd=BASE/'code',stdout=log,stderr=subprocess.STDOUT)
        record={'step':step,'exit_code':proc.returncode,'wall_seconds':time.time()-start}
        records.append(record)
        if proc.returncode:
            atomic(BASE/'audit/POSTPROCESS_ERROR.json',{'failed':record,'completed':records})
            raise RuntimeError('Postprocess stage failed: '+label)
    atomic(audit,{'phase':'artifacts_built_manual_review_remaining','completed':records,
        'at_utc':datetime.now(timezone.utc).isoformat(),'goal_complete':False})


if __name__=='__main__':main()
