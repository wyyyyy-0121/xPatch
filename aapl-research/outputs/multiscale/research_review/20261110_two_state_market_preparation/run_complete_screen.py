"""Single bounded queue -> finish -> independent review entry; standard library.

No market reader or training import. The queue creates output_directory. Logs
are retained in a fresh sibling directory. Negative scientific gates may be
qualified completed screens; they are never marked method/Goal success.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

BASE=Path(__file__).resolve().parent
AS_LIMIT=6*1024**3
RSS_LIMIT=3*1024**3
THREAD_KEYS=("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS")
SCRIPTS=("run_market_queue_supervisor.py","market_screen_finish.py","review_market_screen.py")


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    temporary=path.with_name(path.name+".tmp-"+str(os.getpid()))
    with temporary.open("x") as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write("\n")
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def process_identity(pid):
    try:
        raw=Path(f"/proc/{pid}/stat").read_text();fields=raw[raw.rfind(")")+2:].split()
        return {"PID":pid,"start_ticks":int(fields[19]),"state":fields[0],"PGID":int(fields[2])}
    except FileNotFoundError:return None


def descendants():
    pending=[os.getpid()];seen=set()
    while pending:
        parent=pending.pop()
        try:
            for task in Path(f"/proc/{parent}/task").iterdir():
                try:children=(task/"children").read_text().split()
                except FileNotFoundError:continue
                for text in children:
                    pid=int(text)
                    if pid not in seen:seen.add(pid);pending.append(pid)
        except FileNotFoundError:pass
    return seen


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",type=Path,required=True)
    args=parser.parse_args();protocol_path=args.protocol.resolve()
    body=protocol_path.read_bytes();p=json.loads(body);protocol_sha=hashlib.sha256(body).hexdigest()
    start,end=p["started_epoch"],p["deadline_epoch"]
    require(type(start) in (int,float) and type(end) in (int,float)
            and math.isfinite(start) and math.isfinite(end) and end-start==14400
            and start<=time.time()<end-20,"Same original 14400s window required")
    require(p["kind"]=="matched_two_state_market_execution_v1" and p["Q3_allowed"] is False
            and p["GPU"] is False and p["paid_resources"] is False,"Frozen screen scope differs")
    finite_path=BASE/"FINITE_CLOSEOUT_PLAN.json";finite=json.loads(finite_path.read_text())
    require(p["finite_closeout_plan_sha256"]==sha(finite_path)
            and time.time()<finite["engineering_deadline_epoch"] and end<=finite["overall_delivery_deadline_epoch"],
            "Finite engineering admission or delivery deadline reached")
    locks={str(Path(row["path"]).resolve()):row["sha256"] for row in p["locks"]}
    for name in (*SCRIPTS,Path(__file__).name):
        require(locks.get(str((BASE/name).resolve()))==sha(BASE/name),"Pipeline source not frozen: "+name)
    python=Path(p["python_executable"]);require(python.is_absolute() and python.is_file(),"Explicit Python executable required")
    output=Path(p["output_directory"]);require(output.is_absolute() and not output.exists(),"Queue must create a fresh outputroot")
    logs=output.with_name(output.name+"__complete_supervision");require(not logs.exists(),"Refuse retry/overwrite of pipeline logs")
    logs.mkdir(parents=True)
    os.sched_setaffinity(0,{24});resource.setrlimit(resource.RLIMIT_AS,(AS_LIMIT,AS_LIMIT))
    # Linux subreaper adopts detached worker sessions if their queue parent exits.
    libc=ctypes.CDLL(None,use_errno=True)
    require(libc.prctl(36,1,0,0,0)==0,"PR_SET_CHILD_SUBREAPER failed; orphan-safe admission denied")
    env=dict(os.environ,**{name:"1" for name in THREAD_KEYS},CUDA_VISIBLE_DEVICES="-1",PYTHONDONTWRITEBYTECODE="1")
    state={"kind":"bounded_complete_screen_pipeline_v1","protocol_sha256":protocol_sha,
           "started_epoch":start,"deadline_epoch":end,"supervisor_started_epoch":time.time(),
           "supervisor_PID":os.getpid(),"subreaper":True,"status":"running","phases":[],
           "screen_complete":False,"method_success":False,"research_goal_achieved":False,
           "automatic_retry":False,"deadline_reset":False,"GPU":False,"Q3_read":False}
    receipt=logs/"COMPLETE_SCREEN_SUPERVISOR.json";owned={};stop={"signal":None};handlers={};current=None
    for signum in (signal.SIGTERM,signal.SIGINT):
        handlers[signum]=signal.getsignal(signum)
        signal.signal(signum,lambda received,_frame:stop.update(signal=received))

    def refresh():
        for pid in descendants():
            identity=process_identity(pid)
            if identity is not None:
                key=(pid,identity["start_ticks"])
                record=owned.setdefault(key,{**identity,"maximum_RSS_bytes":0,"maximum_VmHWM_bytes":0})
                record["state"]=identity["state"]
                try:
                    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                        for prefix,name in (("VmRSS:","maximum_RSS_bytes"),("VmHWM:","maximum_VmHWM_bytes")):
                            if line.startswith(prefix):record[name]=max(record[name],int(line.split()[1])*1024)
                except FileNotFoundError:pass
        live=[]
        for (pid,ticks),record in owned.items():
            identity=process_identity(pid)
            if identity is None or identity["start_ticks"]!=ticks:continue
            if identity["state"]!="Z":live.append((pid,ticks))
            elif current is None or pid!=current.pid:
                try:os.waitpid(pid,os.WNOHANG)
                except ChildProcessError:pass
        return live

    def signal_owned(signum):
        refresh();sent=[]
        # Per-PID starttime binding handles workers with independent sessions.
        for (pid,ticks),record in owned.items():
            identity=process_identity(pid)
            if identity is None or identity["start_ticks"]!=ticks or identity["state"]=="Z":continue
            try:os.kill(pid,signum);sent.append(pid)
            except ProcessLookupError:pass
        return sent

    def run_phase(index,name,cpus):
        nonlocal current
        require(not stop["signal"] and time.time()<end-20,"No remaining original window for next phase")
        require(sha(protocol_path)==protocol_sha,"Protocol changed between phases")
        require(sha(BASE/name)==locks[str((BASE/name).resolve())],"Frozen phase source changed")
        command=[str(python),"-B",str(BASE/name),"--protocol",str(protocol_path)]
        phase={"role":name,"command":command,"CPU_affinity":cpus,"threads":1,
               "started_epoch":time.time(),"deadline_TERM_epoch":None,"maximum_RSS_bytes":0,"maximum_VmHWM_bytes":0}
        state["phases"].append(phase)
        stdout_path=logs/(str(index)+".stdout.txt");stderr_path=logs/(str(index)+".stderr.txt")
        stdout=stdout_path.open("x");stderr=stderr_path.open("x");proc=None
        def restrict():
            os.sched_setaffinity(0,set(cpus));resource.setrlimit(resource.RLIMIT_AS,(AS_LIMIT,AS_LIMIT))
        try:
            proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                                  env=env,start_new_session=True,preexec_fn=restrict)
            current=proc;phase["PID"]=proc.pid;atomic(receipt,state)
            while proc.poll() is None:
                live=refresh();now=time.time()
                for key in live:
                    record=owned[key]
                    require(max(record["maximum_RSS_bytes"],record["maximum_VmHWM_bytes"])<=RSS_LIMIT,
                            "Child/descendant RSS3GiB exceeded")
                require(not stop["signal"],"Pipeline externally stopped")
                if now>=end:
                    phase["hard_deadline_killed_PIDs"]=signal_owned(signal.SIGKILL)
                    raise TimeoutError("Original hard deadline reached")
                if now>=end-20 and phase["deadline_TERM_epoch"] is None:
                    # Queue owns safe worker checkpoints; notify its supervisor first.
                    os.kill(proc.pid,signal.SIGTERM);phase["deadline_TERM_epoch"]=time.time();atomic(receipt,state)
                time.sleep(min(.025,max(.001,end-time.time())))
            proc.wait();refresh()
            require(proc.returncode==0,"Pipeline phase failed; no automatic retry: "+name)
            require(not refresh(),"Phase left a live detached descendant")
            require(time.time()<=end,"Pipeline phase exceeded original deadline")
        finally:
            if proc is not None and proc.poll() is None:
                try:os.kill(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                until=min(end,time.time()+20)
                while proc.poll() is None and time.time()<until:refresh();time.sleep(.025)
                if proc.poll() is None:phase["cleanup_killed_PIDs"]=signal_owned(signal.SIGKILL)
                proc.wait()
            refresh();stdout.close();stderr.close()
            phase.update(exit_code=None if proc is None else proc.returncode,finished_epoch=time.time(),terminal=True,
                         cumulative_child_ru_maxrss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024)
            phase["elapsed_seconds"]=phase["finished_epoch"]-phase["started_epoch"]
            phase["stdout_path"]=stdout_path.name;phase["stderr_path"]=stderr_path.name
            phase["stdout_sha256"]=sha(stdout_path);phase["stderr_sha256"]=sha(stderr_path)
            direct=[record for record in owned.values() if record["PID"]==phase.get("PID")]
            phase["maximum_RSS_bytes"]=max([0]+[record["maximum_RSS_bytes"] for record in direct])
            phase["maximum_VmHWM_bytes"]=max([0]+[record["maximum_VmHWM_bytes"] for record in direct])
            phase["descendants"]=list(owned.values());current=None;atomic(receipt,state)
        require(phase["cumulative_child_ru_maxrss_bytes"]<=RSS_LIMIT,"Final child peak RSS3GiB exceeded")

    failed=False
    try:
        atomic(receipt,state)
        run_phase(1,SCRIPTS[0],[24])
        queue=json.loads((output/"Q_S_QUEUE_SUPERVISOR.json").read_text())
        require(queue["status"]=="Q_S_QUEUE_COMPLETE_AWAITING_POSTPROCESSING"
                and queue["protocol_sha256"]==protocol_sha and queue["screen_complete"] is False,
                "Complete Q/S queue receipt missing")
        run_phase(2,SCRIPTS[1],[24,26])
        main=json.loads((output/"postprocessing/RESULTS.json").read_text())
        require(main["status"]=="main_scoring_complete_independent_review_pending"
                and main["protocol_sha256"]==protocol_sha and main["research_goal_achieved"] is False,
                "Qualified main scoring receipt missing")
        run_phase(3,SCRIPTS[2],[24,26])
        review=json.loads((output/"postprocessing/INDEPENDENT_REVIEW.json").read_text())
        require(review["status"]=="PASS" and review["protocol_sha256"]==protocol_sha
                and review["research_goal_achieved"] is False and type(review["screen_passed"]) is bool
                and type(main["screen_conjunction_passed"]) is bool
                and review["screen_passed"]==main["screen_conjunction_passed"],"Independent screen review mismatch")
        require(time.time()<=end and not refresh(),"Screen outside original deadline or orphan remains")
        state.update(status="qualified_positive_screen" if review["screen_passed"] else "qualified_negative_screen",
                     screen_complete=True,screen_passed=review["screen_passed"],
                     main_results_sha256=sha(output/"postprocessing/RESULTS.json"),
                     independent_review_sha256=sha(output/"postprocessing/INDEPENDENT_REVIEW.json"))
    except BaseException as error:
        failed=True;state.update(status="INCOMPLETE_PRESERVE_ARTIFACTS",screen_complete=False,
                                 error={"type":type(error).__name__,"message":str(error)})
    finally:
        # Any remaining adopted workers are signalled then killed within the same deadline.
        live=refresh()
        if live:
            state["orphan_cleanup_TERM_PIDs"]=signal_owned(signal.SIGTERM)
            until=min(end,time.time()+20)
            while refresh() and time.time()<until:time.sleep(.025)
            if refresh():state["orphan_cleanup_KILL_PIDs"]=signal_owned(signal.SIGKILL)
            refresh()
        state["descendants"]=list(owned.values())
        if refresh() or time.time()>end:
            failed=True;state.update(status="INCOMPLETE_PRESERVE_ARTIFACTS",screen_complete=False)
        state.update(finished_epoch=time.time(),window_elapsed_seconds=time.time()-start,
                     method_success=False,research_goal_achieved=False)
        atomic(receipt,state)
        if output.is_dir():atomic(output/"COMPLETE_SCREEN_SUPERVISOR.json",state)
        for signum,handler in handlers.items():signal.signal(signum,handler)
    print(json.dumps(state,allow_nan=False),flush=True)
    return 1 if failed else 0


if __name__=="__main__":raise SystemExit(main())
