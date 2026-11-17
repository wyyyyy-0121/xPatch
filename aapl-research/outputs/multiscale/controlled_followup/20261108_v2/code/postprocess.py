"""Continue declared work after the training controller; preserve failures."""
from common import *
import time,subprocess,fcntl
from datetime import datetime,timezone

def call(python,script):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    log=BASE/'logs'/f'{script}.log'
    with log.open('a') as f:
        subprocess.run(['taskset','-c','30,31',python,str(BASE/'code'/f'{script}.py')],cwd=ROOT,
            env=env,stdout=f,stderr=subprocess.STDOUT,check=True)

def main():
    with (BASE/'audit/POSTPROCESS_LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (BASE/'audit/TRAINING_COMPLETE.json').exists():
            if (BASE/'audit/CONTROLLER_ERROR.json').exists():raise RuntimeError('Training error requires review; no fabricated complete paper')
            if (BASE/'audit/TRAINING_PAUSED.json').exists():raise RuntimeError('Budget ended with incomplete matrix; preserve checkpoints and report actual scope')
            time.sleep(5)
        for script in ('analyze','write_paper'):
            atomic(BASE/'audit/POSTPROCESS_STATE.json',{'phase':script,'at_utc':datetime.now(timezone.utc).isoformat()})
            call(sys.executable,script)
        docpython='/home/ayaka/.venvs/doc-tools/bin/python'
        for script in ('figures','build_documents'):
            atomic(BASE/'audit/POSTPROCESS_STATE.json',{'phase':script,'at_utc':datetime.now(timezone.utc).isoformat()})
            call(docpython,script)
        atomic(BASE/'audit/POSTPROCESS_STATE.json',{'phase':'predictor_replay','at_utc':datetime.now(timezone.utc).isoformat()})
        call(sys.executable,'replay')
        atomic(BASE/'audit/POSTPROCESS_STATE.json',{'phase':'manual_visual_and_final_audit_required',
            'at_utc':datetime.now(timezone.utc).isoformat(),'paper_delivery_complete':False})
        print('Drafts built and predictors replayed. Actual page inspection and release audit remain.',flush=True)

if __name__=='__main__':main()
