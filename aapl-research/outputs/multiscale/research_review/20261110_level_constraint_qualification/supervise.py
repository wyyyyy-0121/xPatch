"""Freeze one 180s window; real signal/new-PID recovery and independent replay."""
import os
import signal
import subprocess
import sys
import time
from runtime import BASE, read, write, sha, require, setup

def launch(script, log):
    stream=open(BASE/'run'/log,'w')
    proc=subprocess.Popen([sys.executable,'-B',str(BASE/script)],cwd=BASE,stdout=stream,stderr=subprocess.STDOUT)
    stream.close()
    return proc

def wait(proc,deadline):
    # Cleanup belongs to this same wall-clock window, rather than extending it.
    try: return proc.wait(timeout=max(.001,deadline-time.time()-2))
    except subprocess.TimeoutExpired:
        proc.send_signal(signal.SIGTERM)
        try: proc.wait(timeout=max(.001,deadline-time.time()-.2))
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait(timeout=max(.001,deadline-time.time()))
        raise RuntimeError('Shared deadline expired; committed states retained')

def main():
    setup(24)
    require(not (BASE/'EXECUTION_PROTOCOL.json').exists(),'Refuse another execution window')
    engineering=read(BASE/'ENGINEERING_ACCEPTANCE.json')
    require(engineering['status']=='passed_artificial_only','Engineering required')
    for file,expected in engineering['local_code_hashes'].items():
        require(sha(file)==expected,'Engineering-tested source changed')
    acceptance=read(BASE/'CODE_ACCEPTANCE.json')
    require(acceptance['accepted_before_execution'] is True,'Independent source review required')
    for file,expected in acceptance['reviewed_sources'].items():
        require(sha(file)==expected,'Reviewed source changed')
    require((BASE/'DESIGN_AND_DUPLICATION_REVIEW.md').exists(),'Design review required')
    files=[BASE/name for name in ['DESIGN_ZH.md','DESIGN_SPEC.json','ARTIFACT_AND_CODE_REVIEW.md',
           'DESIGN_AND_DUPLICATION_REVIEW.md','CODE_ACCEPTANCE.json','ENGINEERING_ACCEPTANCE.json',
           'runtime.py','main_worker.py','prepare.py','engineering.py','supervise.py','independent_replay.py']]
    spec=read(BASE/'DESIGN_SPEC.json')
    locks={str(p):sha(p) for p in files}|spec['source_files']|spec['source_metadata']
    t=time.time()
    p={'schema':'level_constraint_pair_v1','execution_admitted':True,'started_epoch':t,'deadline_epoch':t+180,
       'locked_files':locks,'GPU':False,'Q3_allowed':False,'market_training_updates':0,'model_inference':0}
    write(BASE/'EXECUTION_PROTOCOL.json',p)
    output=BASE/'run'; output.mkdir(exist_ok=False)
    first=launch('main_worker.py','FIRST_WORKER.log')
    children=[first]
    lifecycle={'protocol_sha256':sha(BASE/'EXECUTION_PROTOCOL.json'),'initial_pid':first.pid,'deadline_epoch':p['deadline_epoch']}
    try:
        while not (output/'FIRST_COMMIT.json').exists():
            require(first.poll() is None,'Worker exited before signal rendezvous')
            require(time.time()<t+90,'First commit cost exceeded fixed reserve')
            time.sleep(.01)
        checkpoint=read(output/'CHECKPOINT.json'); paused_bytes=(output/'CHECKPOINT.json').read_bytes()
        first.send_signal(signal.SIGUSR1)
        require(wait(first,p['deadline_epoch'])==75,'External pause must preserve then exit75')
        require((output/'CHECKPOINT.json').read_bytes()==paused_bytes,'Checkpoint changed after pause')
        require(read(output/'PAUSED.json')['checkpoint_sha256']==sha(output/'CHECKPOINT.json'),'Pause identity')
        lifecycle.update({'signal':'SIGUSR1','first_exit':75,'pause_completed_states':len(checkpoint['completed']),
                          'paused_checkpoint_sha256':sha(output/'CHECKPOINT.json'),'paused_bytes_equality_checked':True})
        # Preserve the actual historical paused checkpoint, unlike the preceding stage.
        write(output/'PAUSED_CHECKPOINT.json',checkpoint)
        second=launch('main_worker.py','RESUMED_WORKER.log');children.append(second)
        require(second.pid!=first.pid,'Resume must use new PID')
        require(wait(second,p['deadline_epoch'])==0,'Resumed main failed')
        lifecycle.update({'resumed_pid':second.pid,'same_protocol_and_deadline':True,'resumed_exit':0})
        replay=launch('independent_replay.py','INDEPENDENT_REPLAY.log'); children.append(replay)
        require(wait(replay,p['deadline_epoch'])==0,'Independent numerical qualification failed')
        require(read(output/'INDEPENDENT_REPLAY.json')['passed'] is True,'Independent report failed')
        require(time.time()<p['deadline_epoch'],'Acceptance must precede same absolute deadline')
        lifecycle.update({'status':'passed','independent_pid':replay.pid,'actual_shared_seconds':time.time()-t,
                          'all_children_terminal':all(x.poll() is not None for x in children)})
        write(output/'LIFECYCLE_ACCEPTANCE.json',lifecycle)
    except BaseException as exc:
        for child in children:
            if child.poll() is None:
                child.send_signal(signal.SIGTERM)
                try: child.wait(timeout=max(.001,min(2,p['deadline_epoch']-time.time()-.2)))
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait(timeout=max(.001,p['deadline_epoch']-time.time()))
        write(output/'CONTROLLER_FAILURE.json',lifecycle|{'error':repr(exc),'seconds':time.time()-t})
        raise

if __name__=='__main__': main()
