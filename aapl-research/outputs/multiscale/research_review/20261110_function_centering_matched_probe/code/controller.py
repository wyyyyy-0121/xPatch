"""Frozen stdlib-only supervisor; actual signals/new processes and fixed deadline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,value):
    temporary=path.with_name(path.name+'.tmp'); temporary.write_text(json.dumps(value,indent=2)+'\n'); os.replace(temporary,path)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--protocol',required=True,type=Path); parser.add_argument('--resume',action='store_true'); args=parser.parse_args()
    protocol_path=args.protocol.resolve(); protocol=json.loads(protocol_path.read_text())
    for group in ['code_files','source_files']:
        for path,digest in protocol[group].items():
            if sha(path)!=digest: raise RuntimeError('Supervisor frozen source changed: '+path)
    if str(Path(__file__).resolve()) not in protocol['code_files']: raise RuntimeError('Supervisor not frozen')
    root=Path(protocol['output_directory']); root.mkdir(parents=True,exist_ok=True)
    deadline=protocol['deadline_epoch']; events=[]
    environment=os.environ.copy(); environment.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    guard=Path(__file__).parent/'guarded_run.py'
    def run(mode,name,resume=False,pause_kind=None):
        output=root/name; output.mkdir(parents=True,exist_ok=True)
        if time.time()>=deadline-15: raise RuntimeError('Insufficient remaining fixed deadline')
        command=[sys.executable,str(guard),'--protocol',str(protocol_path),'--mode',mode,'--output',str(output)]
        if resume: command.append('--resume')
        if pause_kind: command+=['--pause-kind',pause_kind]
        log=output/('resume.log' if resume else 'initial.log')
        with log.open('w') as stream:
            process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=environment)
            event={'mode':mode,'name':name,'resume':resume,'pause_kind':pause_kind,'pid':process.pid,'started_epoch':time.time(),'command':command,'log':str(log),'signals_sent':[]}
            events.append(event); sent=False; time_signal=False
            write(root/'EXECUTION_RECEIPT.json',{'status':'active_child_registered','protocol_sha256':sha(protocol_path),'events':events,'deadline_epoch':deadline,'deadline_reset':False})
            execution_error=None
            try:
                while process.poll() is None:
                    now=time.time()
                    ready=output/'READY_FOR_SIGNAL.json'
                    if pause_kind=='signal' and not resume and ready.is_file() and not sent:
                        marker=json.loads(ready.read_text())
                        if marker['pid']!=process.pid or marker['steps']!=4: raise RuntimeError('Signal marker not from actual worker')
                        os.kill(process.pid,signal.SIGUSR1); sent=True
                        event['signals_sent'].append({'signal':'SIGUSR1','reason':'actual_commonbatch4','epoch':now})
                    if now>=deadline-15 and not time_signal:
                        os.kill(process.pid,signal.SIGUSR1); time_signal=True
                        event['signals_sent'].append({'signal':'SIGUSR1','reason':'globaldeadline_reserve','epoch':now})
                    if now>=deadline:
                        event['forced_deadline_kill']=True
                        break
                    time.sleep(.02)
            except BaseException as exc:
                execution_error=repr(exc)
                event['supervisor_exception']=execution_error
            finally:
                if process.poll() is None:
                    try:
                        process.send_signal(signal.SIGUSR1)
                        reserve=max(0.,min(10.,deadline-time.time()))
                        if reserve>0: process.wait(timeout=reserve)
                    except (ProcessLookupError,subprocess.TimeoutExpired):
                        pass
                    finally:
                        if process.poll() is None:
                            process.kill()
                        process.wait()
            event['exit_code']=process.returncode; event['finished_epoch']=time.time(); event['log_sha256']=sha(log)
        write(root/'EXECUTION_RECEIPT.json',{'status':'running_or_failed_until_final_receipt','protocol_sha256':sha(protocol_path),'events':events,'deadline_epoch':deadline,'deadline_reset':False})
        if execution_error: raise RuntimeError(execution_error)
        return process.returncode
    try:
        if protocol['phase']=='market':
            code=run('market','market',args.resume)
            if code!=0: raise RuntimeError('Market worker stopped with code '+str(code))
        else:
            for mode,name,pause,resume,expected in [('qualify','qualify',None,False,0),('cost','cost',None,False,0),('trajectory','continuous',None,False,0),('trajectory','signal','signal',False,75),('trajectory','signal',None,True,0),('trajectory','deadline','deadline',False,75),('trajectory','deadline',None,True,0),('compare','comparison',None,False,0)]:
                code=run(mode,name,resume,pause)
                if code!=expected: raise RuntimeError(name+' unexpected exit '+str(code)+' wanted '+str(expected))
        status='completed_worker_sequence_pending_independent_acceptance'; error=None
    except Exception as exc:
        status='stopped_or_failed_without_market_promotion'; error=repr(exc)
    write(root/'EXECUTION_RECEIPT.json',{'status':status,'error':error,'protocol_sha256':sha(protocol_path),'events':events,'deadline_epoch':deadline,'deadline_reset':False,'market_read':False if protocol['phase']=='synthetic' else 'see_worker_INPUT_RECEIPT_for_actual_decode','market_access_attempted':protocol['phase']=='market','whole_preQ3_decode_receipt_available':(root/'market'/'INPUT_RECEIPT.json').is_file(),'Q3_read':False,'full_goal_achieved':False})
    if error: return 1
    return 0

if __name__=='__main__': sys.exit(main())
