"""One 300s artificial paired-training recovery acceptance; standard library only.

Protocol is frozen by root before execution. Every child is sequential, CPU24,
threads1, AS6GiB, RSS3GiB, GPU hidden, and shares the original absolute deadline.
No retry, source changes, market arrays, or additional experiments are permitted.
"""
from __future__ import annotations
import argparse
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
PREPARATION=BASE.parent
AS_LIMIT=6*1024**3
RSS_LIMIT=3*1024**3
DEADLINE_SIGNAL_LEAD=20.
THREAD_ENV=("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS")


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


def committed_pause_copy(run_directory,output):
    pointer=json.loads((run_directory/"LATEST.json").read_text())
    require(pointer["payload"] in ("latest_0.pt","latest_1.pt"),"Invalid pause latest slot")
    body=(run_directory/pointer["payload"]).read_bytes()
    require(hashlib.sha256(body).hexdigest()==pointer["sha256"],"Pause payload checksum changed")
    target=output/"pause_committed.pt"
    with target.open("xb") as stream:
        stream.write(body);stream.flush();os.fsync(stream.fileno())
    receipt={"payload":target.name,"sha256":pointer["sha256"],
             "identity_sha256":pointer["identity_sha256"],"source_pointer":pointer,
             "copied_after_exit75":True,"observation_only":True}
    atomic(output/"PAUSE_COMMITTED.json",receipt)
    return receipt


def source_paths():
    return [Path(__file__),BASE/"artificial_train_fixture.py",BASE/"train_pair.py",
            BASE/"compare_artificial_recovery.py",PREPARATION/"common_adapter.py",
            PREPARATION/"artificial_adapter_tests.py",PREPARATION/"SCIENCE_RULES.json",
            PREPARATION.parent/"20261110_session_component_source_review/artificial_model/prototype_model.py"]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",type=Path,required=True)
    args=parser.parse_args()
    protocol_path=args.protocol.resolve();body=protocol_path.read_bytes();protocol=json.loads(body)
    start,end=protocol["started_epoch"],protocol["deadline_epoch"]
    require(type(start) in (int,float) and type(end) in (int,float)
            and math.isfinite(start) and math.isfinite(end) and end-start==300
            and start<=time.time()<end-DEADLINE_SIGNAL_LEAD,"Frozen 300s window invalid")
    require(protocol["kind"]=="two_state_training_artificial_recovery_v1"
            and protocol["market_arrays_allowed"] is False
            and protocol["C_cal_T_operations_allowed"] is False
            and protocol["GPU"] is False,"Artificial-only scope invalid")
    require(protocol["CPU_affinity"]==[24] and protocol["threads"]==1
            and protocol["AS_limit_bytes"]==AS_LIMIT and protocol["RSS_limit_bytes"]==RSS_LIMIT,
            "Frozen resource cap invalid")
    require(protocol["seed"]==2026,"Single artificial seed must be 2026")
    require(protocol["supervisor_sha256"]==sha(Path(__file__)),"Supervisor source changed")
    locks={str(Path(row["path"]).resolve()):row["sha256"] for row in protocol["locks"]}
    for path in source_paths():
        require(locks.get(str(path.resolve()))==sha(path),"Required frozen source missing/changed: "+str(path))
    for name,digest in locks.items():require(sha(Path(name))==digest,"Frozen source changed: "+name)
    python=Path(protocol["python_executable"])
    require(python.is_absolute() and python.is_file(),"Explicit Python executable required")
    output=Path(protocol["output_directory"])
    require(output.is_absolute(),"Fresh absolute output_directory required")
    require(not output.exists(),"Refuse to overwrite or retry an existing run")
    output.mkdir(parents=True)
    os.sched_setaffinity(0,{24})
    resource.setrlimit(resource.RLIMIT_AS,(AS_LIMIT,AS_LIMIT))
    env=dict(os.environ,**{name:"1" for name in THREAD_ENV},
             CUDA_VISIBLE_DEVICES="-1",PYTHONDONTWRITEBYTECODE="1")
    state={"kind":protocol["kind"],"protocol_sha256":hashlib.sha256(body).hexdigest(),
           "started_epoch":start,"deadline_epoch":end,"supervisor_started_epoch":time.time(),
           "supervisor_PID":os.getpid(),"CPU_affinity":[24],"threads":1,
           "AS_limit_bytes":AS_LIMIT,"RSS_limit_bytes":RSS_LIMIT,"GPU":False,
           "status":"running","phases":[],"automatic_retry":False,"deadline_reset":False,
           "market_arrays_opened":False,"C_cal_T_operations":False,"research_goal_achieved":False}
    receipt=output/"SUPERVISOR.json"
    stop={"signal":None};active_child=None
    handlers={}
    for signum in (signal.SIGTERM,signal.SIGINT):
        handlers[signum]=signal.getsignal(signum)
        signal.signal(signum,lambda received,_frame:stop.update(signal=received))

    def run_process(role,command,expected_code,*,interrupt=False,training=False):
        nonlocal active_child
        require(not stop["signal"] and time.time()<end-DEADLINE_SIGNAL_LEAD,
                "Original window/supervisor stop before "+role)
        phase={"role":role,"command":command,"started_epoch":time.time(),"PID":None,
               "maximum_observed_RSS_bytes":0,"maximum_observed_VmHWM_bytes":0,
               "SIGUSR1_sent_epoch":None,"SIGUSR1_trigger":None,"exit_code":None,
               "progress_events":[],"termination_reason":None}
        state["phases"].append(phase)
        stdout_path=output/(role+".stdout.txt");stderr_path=output/(role+".stderr.txt")
        stdout=stdout_path.open("x");stderr=stderr_path.open("x");reader=None;proc=None
        try:
            proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                                  env=env,start_new_session=True)
            active_child=proc;phase["PID"]=proc.pid
            atomic(receipt,state)
            reader=stdout_path.open("r");pending=""
            while proc.poll() is None:
                now=time.time()
                try:
                    status=Path(f"/proc/{proc.pid}/status").read_text()
                    for line in status.splitlines():
                        if line.startswith("VmRSS:"):
                            phase["maximum_observed_RSS_bytes"]=max(phase["maximum_observed_RSS_bytes"],int(line.split()[1])*1024)
                        elif line.startswith("VmHWM:"):
                            phase["maximum_observed_VmHWM_bytes"]=max(phase["maximum_observed_VmHWM_bytes"],int(line.split()[1])*1024)
                except FileNotFoundError:pass
                pending+=reader.read()
                lines=pending.split("\n");pending=lines.pop()
                for line in lines:
                    if not line.startswith("PROGRESS "):continue
                    event=json.loads(line[len("PROGRESS "):])
                    phase["progress_events"].append({"observed_epoch":time.time(),**event})
                    if interrupt and event.get("event")=="COMMON_BATCH_COMMITTED" and event.get("common_steps")==1:
                        require(phase["SIGUSR1_sent_epoch"] is None,"Duplicate first-batch interrupt")
                        require((output/"resumed_pair/FIRST_COMMON_BATCH.json").is_file(),"First observation not committed")
                        os.kill(proc.pid,signal.SIGUSR1)
                        phase["SIGUSR1_sent_epoch"]=time.time();phase["SIGUSR1_trigger"]="external_after_first_common_batch"
                        phase["SIGUSR1_target_PID"]=proc.pid;atomic(receipt,state)
                if max(phase["maximum_observed_RSS_bytes"],phase["maximum_observed_VmHWM_bytes"])>RSS_LIMIT:
                    phase["termination_reason"]="RSS_budget";raise RuntimeError("RSS3GiB exceeded")
                if stop["signal"]:
                    phase["termination_reason"]="supervisor_signal_"+str(stop["signal"])
                    raise InterruptedError("Supervisor externally stopped; no retry")
                if now>=end:
                    phase["termination_reason"]="absolute_deadline";raise TimeoutError("Original absolute deadline reached")
                if training and now>=end-DEADLINE_SIGNAL_LEAD and phase["SIGUSR1_sent_epoch"] is None:
                    os.kill(proc.pid,signal.SIGUSR1)
                    phase["SIGUSR1_sent_epoch"]=time.time();phase["SIGUSR1_trigger"]="deadline_grace_20s"
                    phase["SIGUSR1_target_PID"]=proc.pid;atomic(receipt,state)
                time.sleep(min(.02,max(.001,end-time.time())))
            proc.wait()
            require(proc.returncode==expected_code,"Unexpected exit in "+role+": "+str(proc.returncode))
            if interrupt:
                require(phase["SIGUSR1_trigger"]=="external_after_first_common_batch", "No actual first-batch external SIGUSR1")
            require(time.time()<=end,"Phase exceeded original deadline")
        finally:
            if proc is not None and proc.poll() is None:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                proc.wait();phase["forced_kill"]=True
            if reader is not None:reader.close()
            stdout.close();stderr.close()
            phase["exit_code"]=None if proc is None else proc.returncode
            phase["finished_epoch"]=time.time();phase["elapsed_seconds"]=phase["finished_epoch"]-phase["started_epoch"]
            phase["stdout_path"]=stdout_path.name;phase["stderr_path"]=stderr_path.name
            phase["stdout_sha256"]=sha(stdout_path);phase["stderr_sha256"]=sha(stderr_path)
            phase["cumulative_child_ru_maxrss_bytes"]=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024
            phase["terminal"]=True;active_child=None;atomic(receipt,state)
        require(phase["cumulative_child_ru_maxrss_bytes"]<=RSS_LIMIT,"Final cumulative child peak RSS3GiB exceeded")
        return phase

    def fixture_command(directory,*,resume=False,hold=False):
        command=[str(python),"-B",str(BASE/"artificial_train_fixture.py"),
                 "--outputdir",str(directory),"--absolute-deadline",repr(end),"--seed","2026"]
        if resume:command.append("--resume")
        if hold:command.append("--hold-first-progress")
        return command

    failed=False
    try:
        atomic(receipt,state)
        resumed=output/"resumed_pair";uninterrupted=output/"uninterrupted_pair"
        first=run_process("01_external_pause",fixture_command(resumed,hold=True),75,interrupt=True,training=True)
        paused=json.loads((resumed/"PAUSED.json").read_text())
        require(paused["exit_code"]==75 and paused["reason"]=="signal_"+str(signal.SIGUSR1)
                and paused["common_steps"]==1 and paused["absolute_deadline"]==end,
                "Pause was not the external first-batch safe checkpoint")
        state["pause_snapshot"]=committed_pause_copy(resumed,output);atomic(receipt,state)
        second=run_process("02_new_PID_resume",fixture_command(resumed,resume=True),0,training=True)
        require(first["PID"]!=second["PID"],"Resume must use a new PID")
        run_process("03_uninterrupted",fixture_command(uninterrupted),0,training=True)
        command=[str(python),"-B",str(BASE/"compare_artificial_recovery.py"),"--protocol",str(protocol_path),
                 "--run-directory",str(output),"--output",str(output/"COMPARISON.json")]
        run_process("04_independent_compare",command,0)
        comparison=json.loads((output/"COMPARISON.json").read_text())
        require(comparison["status"]=="PASS" and comparison["bitwise_equal"] is True,"Independent comparison failed")
        state["comparison_sha256"]=sha(output/"COMPARISON.json")
        for path in source_paths():require(sha(path)==locks[str(path.resolve())],"Source drift during acceptance")
    except BaseException as error:
        failed=True;state["error"]={"type":type(error).__name__,"message":str(error)}
    finally:
        if active_child is not None and active_child.poll() is None:
            try:os.killpg(active_child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            active_child.wait()
        finished=time.time();failed=failed or finished>end
        state.update(status="FAIL" if failed else "PASS",finished_epoch=finished,
                     window_elapsed_seconds=finished-start,supervisor_elapsed_seconds=finished-state["supervisor_started_epoch"])
        atomic(receipt,state)
        for signum,handler in handlers.items():signal.signal(signum,handler)
    print(json.dumps(state,allow_nan=False),flush=True)
    return 1 if failed else 0


if __name__=="__main__":raise SystemExit(main())
