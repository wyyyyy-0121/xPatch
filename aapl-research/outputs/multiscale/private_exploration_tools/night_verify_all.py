"""Recheck every paused checkpoint identity without starting training."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    snapshot=Path(sys.argv[1]).resolve();data=json.loads(snapshot.read_text())
    worker=Path(__file__).with_name('night_resume_worker.py').resolve()
    checks=[]
    for item in data['workers']:
        proc=subprocess.run([sys.executable,str(worker),'--snapshot',str(snapshot),'--pid',str(item['pid']),
                             '--check','--cores','24'],capture_output=True,text=True,timeout=120)
        if proc.returncode!=0:raise ValueError(f"Checkpoint check failed for {item['pid']}: {proc.stderr[-500:]}")
        result=json.loads(proc.stdout.strip().splitlines()[-1])
        if result.get('status')!='checkpoint_resume_identity_verified' or result.get('training_started') is not False:
            raise ValueError('Unexpected verification result')
        checks.append(result)
        print(json.dumps({'checked_pid':item['pid'],'job':item['job'],'completed':len(checks),
                          'total':len(data['workers'])}),flush=True)
    out={'status':'all_checkpoint_resume_identities_verified','created_at_utc':datetime.now(timezone.utc).isoformat(),
         'snapshot_sha256':digest(snapshot),'resume_worker_sha256':digest(worker),
         'workers':checks,'training_started':False,
         'limitations':['Verification checks saved checkpoint identity and data, not future hardware availability.',
                        'A WSL or OS shutdown can lose the batches performed after the latest periodic checkpoint.']}
    path=snapshot.parent/'CHECKPOINT_RESUME_VALIDATION.json'
    with path.open('x',encoding='utf8') as f:json.dump(out,f,indent=2,ensure_ascii=False);f.write('\n')
    print(json.dumps({'status':out['status'],'path':str(path),'workers':len(checks),'sha256':digest(path)}),flush=True)


if __name__=='__main__':main()
