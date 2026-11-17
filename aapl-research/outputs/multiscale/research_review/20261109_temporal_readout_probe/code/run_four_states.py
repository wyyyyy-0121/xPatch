"""Sequential fixed diagnostic queue; preserve one absolute stage deadline."""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time

BASE=Path(__file__).resolve().parents[1]
STOP=False
CHILD=None


def load(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path,value):
    with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False) as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno());temporary=stream.name
    os.replace(temporary,path)


def stop(*_):
    global STOP
    STOP=True
    if CHILD is not None and CHILD.poll() is None:CHILD.send_signal(signal.SIGUSR1)


def main():
    global CHILD
    assert set(os.sched_getaffinity(0))=={24} and os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGUSR1]:signal.signal(sig,stop)
    admission=load(BASE/'MARKET_ADMISSION.json')
    assert admission['status']=='admitted_four_fixed_temporal_development_states'
    assert admission['controller_sha256']==sha(Path(__file__))
    assert admission['code_freeze_sha256']==sha(BASE/'CODE_FREEZE.json')
    assert admission['stage_input_identities_sha256']==sha(BASE/'STAGE_INPUT_IDENTITIES.json')
    states=load(BASE/'STAGE_INPUT_IDENTITIES.json')['states']
    assert [s['id'] for s in states]==admission['state_ids'] and len(states)==4
    deadline=load(BASE/'STAGE_SCOPE.json')['deadline_epoch'];rows=[]
    for state in states:
        if STOP or time.time()>=deadline:break
        directory=BASE/'runs'/state['id'];directory.mkdir(parents=True,exist_ok=True)
        result_path=directory/'RESULT.json'
        if result_path.exists() and (directory/'RESULT_COMMIT.json').exists():
            result=load(result_path)
            binding=result['binding'];commit=load(directory/'RESULT_COMMIT.json')
            assert commit['result_sha256']==sha(result_path)
            assert result['identity']==commit['identity']==hashlib.sha256(
                json.dumps(binding,sort_keys=True,allow_nan=False).encode()).hexdigest()
            assert result['status'] in ['completed_fixed_temporal_diagnostic','completed_numerical_nonqualification']
            assert binding['stage_deadline_epoch']==deadline
            assert binding['state_id']==state['id'] and binding['cfg']==state['cfg']
            assert binding['fixed_epochs']==state['fixed_epochs']
            assert binding['scope_sha256']==sha(BASE/'STAGE_SCOPE.json')
            assert binding['code_freeze_sha256']==sha(BASE/'CODE_FREEZE.json')
            assert binding['stage_input_identities_sha256']==sha(BASE/'STAGE_INPUT_IDENTITIES.json')
            assert not result['Q3_values_read'] and not result['new_candidate_admitted']
            if result['tail_scored']:
                assert result['paired_loss_sha256']==sha(directory/'PAIRED_TAIL_LOSSES.npz')
            rows.append({'state_id':state['id'],'exit_code':0,'result_status':result['status'],
                         'existing_verified_result':True})
            continue
        command=[sys.executable,str(BASE/'code/temporal_worker.py'),'--mode','market',
                 '--state-id',state['id'],'--run-dir',str(directory)]
        started=time.time()
        with (directory/'WORKER_LOG.txt').open('a') as stream:
            CHILD=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
            atomic(BASE/'QUEUE_STATE.json',{'status':'running','controller_pid':os.getpid(),
                'worker_pid':CHILD.pid,'current_state':state['id'],'completed':rows,
                'deadline_epoch':deadline,'GPU':False,'Q3_values_read':False})
            code=CHILD.wait()
        row={'state_id':state['id'],'exit_code':code,'wall_seconds':time.time()-started,
             'checkpoint_exists':(directory/'CHECKPOINT.json').exists(),
             'result_status':load(result_path)['status'] if result_path.exists() else None}
        rows.append(row);CHILD=None
        atomic(BASE/'QUEUE_STATE.json',{'status':'running','completed':rows,'deadline_epoch':deadline})
        if code!=0:break
    complete=len(rows)==4 and all(row['exit_code']==0 and row['result_status'] for row in rows)
    report={'status':'completed_all_four_fixed_states' if complete else 'partial_preserved',
        'completed':rows,'deadline_epoch':deadline,'finished_epoch':time.time(),'GPU':False,
        'original_stop_outer_scored':False,'Q3_values_read':False,
        'new_forecasting_candidate_admitted':False,'independent_market_confirmation':False,
        'research_goal_achieved':False}
    atomic(BASE/'QUEUE_STATE.json',report);print(json.dumps(report,indent=2))
    return 0 if complete else 2


if __name__=='__main__':raise SystemExit(main())
