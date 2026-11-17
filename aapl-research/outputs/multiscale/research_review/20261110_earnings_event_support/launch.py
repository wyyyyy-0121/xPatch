"""Run main and independent reconstruction within one absolute 180s window."""
import os
import signal
import subprocess
import sys
import time
from runtime import BASE, setup, read, write, sha, require

def launch():
    setup(24)
    require(not (BASE/'EXECUTION_PROTOCOL.json').exists(),'Do not overwrite frozen execution')
    acceptance=read(BASE/'CODE_ACCEPTANCE.json')
    require(acceptance['accepted'] is True,'No code acceptance')
    locks=acceptance['locked_files']
    for path,h in locks.items():require(sha(path)==h,'Preexecution code/source drift '+path)
    require(str(BASE/'launch.py') in locks,'Supervisor not reviewed')
    started=time.time()
    p={'schema':'earnings_event_support_execution_v1','execution_admitted':True,'started_epoch':started,'deadline_epoch':started+180,'locked_files':{**locks,str(BASE/'CODE_ACCEPTANCE.json'):sha(BASE/'CODE_ACCEPTANCE.json')},'scope':'22event/four original partition metadata support only; no prices/Y or fit','GPU':False,'Q3_allowed':False,'actual_pause_resume_test_claimed':False}
    write(BASE/'EXECUTION_PROTOCOL.json',p)
    children=[]
    for filename,cpu in [('support.py',24),('independent_replay.py',26)]:
        remaining=p['deadline_epoch']-time.time()
        require(remaining>15,'Insufficient shared time; preserve partials')
        with (BASE/(filename+'.stdout')).open('w') as out:
            process=subprocess.Popen([sys.executable,'-B',str(BASE/filename)],stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
            row={'script':filename,'pid':process.pid,'analysis_CPU':cpu,'started_epoch':time.time()}
            timedout=False
            try:
                code=process.wait(timeout=max(.01,p['deadline_epoch']-time.time()-5))
            except subprocess.TimeoutExpired:
                timedout=True
                # At least five seconds remain for committing the current atomic state.
                os.killpg(process.pid,signal.SIGUSR1)
                try:code=process.wait(timeout=max(.01,p['deadline_epoch']-time.time()))
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);code=process.wait(timeout=3)
            row.update(exitcode=code,finished_epoch=time.time(),timeout=timedout,terminal=process.poll() is not None)
            children.append(row)
            write(BASE/'EXECUTION_RECEIPT.json',{'protocol_sha256':sha(BASE/'EXECUTION_PROTOCOL.json'),'children':children,'all_children_terminal':all(x['terminal'] for x in children),'elapsed_seconds':time.time()-started,'scientific_acceptance_not_implied':True})
            require(code==0 and not timedout,'Phase failed; retained checkpoints and diagnostics')
    require(time.time()<=p['deadline_epoch'],'Absolute shared window exceeded')
    print({'exitcode':0,'elapsed_seconds':time.time()-started,'all_children_terminal':True})

if __name__=='__main__':launch()
