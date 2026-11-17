"""Bounded Q/S-only queue supervision. This is not a complete market screen.

Root must freeze the complete postprocessing implementation/qualification and
all eight cost bounds before any price work. This standard-library supervisor
runs one first10 cost package then, only if the same 4h window admits all work,
twelve fixed C/R state-seed pairs. It never performs C_cal/T or scoring itself.
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
AS_LIMIT=6*1024**3
RSS_LIMIT=3*1024**3
COST_KEYS=("scaler_io_all_jobs_seconds","ridge_controls_seconds","S_validation_all_epochs_seconds",
           "Ccal_inference_all_models_seconds","T_inference_all_models_seconds","statistics_seconds",
           "independent_review_seconds","closing_reserve_seconds")
THREAD_KEYS=("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS")


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    temp=path.with_name(path.name+".tmp-"+str(os.getpid()))
    with temp.open("x") as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write("\n")
        stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def list_schedule(jobs,counts,slots,first_id):
    order=sorted(jobs,key=lambda row:(row["job_id"]!=first_id,-counts[row["job_id"]],row["job_id"]))
    loads=[0]*len(slots);assigned={}
    for job in order:
        slot=min(range(len(slots)),key=lambda index:(loads[index],index))
        assigned[job["job_id"]]=slot;loads[slot]+=20*counts[job["job_id"]]
        require(job["CPU_affinity"]==slots[slot],"Job CPU affinity differs from frozen list scheduling")
    return order,assigned,loads


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",type=Path,required=True)
    args=parser.parse_args();protocol_path=args.protocol.resolve()
    body=protocol_path.read_bytes();p=json.loads(body);protocol_sha=hashlib.sha256(body).hexdigest()
    start,end=p["started_epoch"],p["deadline_epoch"]
    require(type(start) in (int,float) and type(end) in (int,float)
            and math.isfinite(start) and math.isfinite(end) and end-start==14400
            and start<=time.time()<end,"Original four-hour window invalid")
    require(p["kind"]=="matched_two_state_market_execution_v1" and p["market_price_io_allowed"] is True
            and p["Q3_allowed"] is False and p["GPU"] is False and p["paid_resources"] is False,
            "Frozen market scope differs")
    finite_path=BASE/"FINITE_CLOSEOUT_PLAN.json";finite=json.loads(finite_path.read_text())
    require(end<=finite["overall_delivery_deadline_epoch"]
            and p["finite_closeout_plan_sha256"]==sha(finite_path),"Finite closeout bound differs")
    locks={str(Path(row["path"]).resolve()):row["sha256"] for row in p["locks"]}
    for path,digest in locks.items():require(sha(path)==digest,"Frozen source changed: "+path)
    require(locks.get(str(Path(__file__).resolve()))==sha(__file__)
            and locks.get(str((BASE/"market_pair_worker.py").resolve()))==sha(BASE/"market_pair_worker.py"),
            "Queue/worker source not frozen")
    python=Path(p["python_executable"]);require(python.is_absolute() and python.is_file(),"Explicit Python required")
    # A complete root qualification is mandatory even before the cost package.
    post=p["postprocessing_freeze"]
    require(post["complete"] is True,"Complete postprocessing is not frozen; NOT_ADMITTED")
    require(sha(post["qualification_path"])==post["qualification_sha256"],"Postprocessing qualification changed")
    qualification=json.loads(Path(post["qualification_path"]).read_text())
    require(qualification["accepted"] is True and qualification["complete"] is True,
            "Postprocessing is not independently qualified; NOT_ADMITTED")
    command=post["command"]
    require(isinstance(command,list) and len(command)>=3 and command[0]==str(python) and command[1]=="-B"
            and Path(command[2]).is_absolute() and locks.get(str(Path(command[2]).resolve()))==sha(command[2]),
            "Frozen postprocessing entry point missing")
    bounds=p["cost_bounds"]
    require(set(bounds)==set(COST_KEYS) and all(type(bounds[key]) in (int,float)
            and math.isfinite(bounds[key]) and bounds[key]>0 for key in COST_KEYS),"Complete positive cost bounds required")
    nontraining_seconds=sum(bounds.values())
    post_reserve=sum(bounds[key] for key in COST_KEYS if key not in
                     ("scaler_io_all_jobs_seconds","S_validation_all_epochs_seconds"))
    jobs=p["training_jobs"];expected={(task,fold,seed) for task in
            ("minute_h60_future_log_return","hourly_h8_future_log_return") for fold in (1,2) for seed in (2026,2027,2028)}
    require(len(jobs)==12 and len({job["job_id"] for job in jobs})==12
            and {(job["task"],job["fold"],job["seed"]) for job in jobs}==expected,"Exact twelve jobs required")
    counts=p["Q_batch_counts"];ids={job["job_id"] for job in jobs};first_id=p["first_cost_job_id"]
    require(set(counts)==ids and all(type(counts[key]) is int and counts[key]>0 for key in ids),"Clock-only Q batch counts missing")
    first=next(job for job in jobs if job["job_id"]==first_id)
    require(first["task"]=="minute_h60_future_log_return" and first["fold"]==2 and first["seed"]==2026
            and counts[first_id]>10,"Fixed first work package differs")
    slots=p["training_slots"]
    require(isinstance(slots,list) and 1<=len(slots)<=8 and all(isinstance(slot,list) and len(slot)==2
            and all(type(cpu) is int and cpu>=0 for cpu in slot) and slot==sorted(set(slot)) for slot in slots)
            and len(set(cpu for slot in slots for cpu in slot))==2*len(slots)
            and not ({24,26}&set(cpu for slot in slots for cpu in slot)),"At most eight distinct two-CPU slots required")
    for job in jobs:
        require(job["threads"]==2 and isinstance(job["job_id"],str) and job["job_id"]
                and Path(job["job_id"]).name==job["job_id"],"Frozen worker identity/thread cap differs")
    order,assignment,loads=list_schedule(jobs,counts,slots,first_id)
    output=Path(p["output_directory"])
    require(output.is_absolute() and not output.exists(),"Fresh output required; no automatic retry")
    output.mkdir(parents=True)
    os.sched_setaffinity(0,{24});resource.setrlimit(resource.RLIMIT_AS,(AS_LIMIT,AS_LIMIT))
    env=dict(os.environ,**{name:"2" for name in THREAD_KEYS},CUDA_VISIBLE_DEVICES="-1",PYTHONDONTWRITEBYTECODE="1")
    receipt=output/"Q_S_QUEUE_SUPERVISOR.json"
    state={"kind":"fixed_twelve_pair_Q_S_queue_only_v1","protocol_sha256":protocol_sha,
           "started_epoch":start,"deadline_epoch":end,"supervisor_started_epoch":time.time(),"phases":[],
           "status":"running","screen_complete":False,"postprocessing_executed":False,
           "automatic_retry":False,"deadline_reset":False,"GPU":False,"research_goal_achieved":False}
    active=[];stop={"signal":None};handlers={};grace=60.;queue_cutoff=end-post_reserve
    for signum in (signal.SIGTERM,signal.SIGINT):
        handlers[signum]=signal.getsignal(signum)
        signal.signal(signum,lambda received,_frame:stop.update(signal=received))

    def launch(job,*,cost=False,resume=False):
        require(time.time()<queue_cutoff-grace and not stop["signal"],"No original queue time for another process")
        name=("cost_" if cost else "train_")+job["job_id"]
        stdout_path=output/(name+".stdout.txt");stderr_path=output/(name+".stderr.txt")
        stdout=stdout_path.open("x");stderr=stderr_path.open("x")
        command=[str(python),"-B",str(BASE/"market_pair_worker.py"),"--protocol",str(protocol_path),"--job-id",job["job_id"]]
        if cost:command.append("--cost-only")
        if resume:command.append("--resume")
        def restrict():
            os.sched_setaffinity(0,set(job["CPU_affinity"]));resource.setrlimit(resource.RLIMIT_AS,(AS_LIMIT,AS_LIMIT))
        phase={"job_id":job["job_id"],"cost_only":cost,"resume":resume,"command":command,
               "CPU_affinity":job["CPU_affinity"],"started_epoch":time.time(),"RSS_bytes":0,"VmHWM_bytes":0,
               "stdout_path":stdout_path.name,"stderr_path":stderr_path.name,"pause_signal_epoch":None}
        proc=None
        try:
            proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                                  env=env,start_new_session=True,preexec_fn=restrict)
            phase["PID"]=proc.pid;state["phases"].append(phase)
            record={"proc":proc,"phase":phase,"stdout":stdout,"stderr":stderr,"reader":stdout_path.open("r"),"pending":""}
            active.append(record);atomic(receipt,state);return record
        except BaseException:
            if proc is not None and proc.poll() is None:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            stdout.close();stderr.close();raise

    def finish(record):
        proc=record["proc"];phase=record["phase"];proc.wait()
        record["reader"].close();record["stdout"].close();record["stderr"].close()
        phase.update(exit_code=proc.returncode,finished_epoch=time.time(),terminal=True,
                     cumulative_child_ru_maxrss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024)
        phase["elapsed_seconds"]=phase["finished_epoch"]-phase["started_epoch"]
        phase["stdout_sha256"]=sha(output/phase["stdout_path"]);phase["stderr_sha256"]=sha(output/phase["stderr_path"])
        active.remove(record);atomic(receipt,state)
        require(max(phase["RSS_bytes"],phase["VmHWM_bytes"],phase["cumulative_child_ru_maxrss_bytes"])<=RSS_LIMIT,"RSS budget exceeded")
        require(proc.returncode==(75 if phase["cost_only"] else 0),"Unexpected worker exit; no automatic retry")
        return phase

    def tick():
        require(not stop["signal"],"Supervisor externally stopped")
        now=time.time();require(now<queue_cutoff,"Q/S cutoff reached; preserve committed checkpoints")
        for record in list(active):
            proc=record["proc"];phase=record["phase"]
            try:
                for line in Path(f"/proc/{proc.pid}/status").read_text().splitlines():
                    for source,key in (("VmRSS:","RSS_bytes"),("VmHWM:","VmHWM_bytes")):
                        if line.startswith(source):phase[key]=max(phase[key],int(line.split()[1])*1024)
            except FileNotFoundError:pass
            if max(phase["RSS_bytes"],phase["VmHWM_bytes"])>RSS_LIMIT:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                phase["forced_kill"]=True
                raise RuntimeError("Live worker RSS budget exceeded")
            record["pending"]+=record["reader"].read()
            lines=record["pending"].split("\n");record["pending"]=lines.pop()
            for line in lines:
                if line.startswith("PROGRESS "):
                    event=json.loads(line[9:])
                    require((event["Q_count"]+127)//128==counts[phase["job_id"]],"Actual Q count differs from frozen clock-only batch count")
            if proc.poll() is not None:finish(record)
            elif now>=queue_cutoff-grace and phase["pause_signal_epoch"] is None:
                os.kill(proc.pid,signal.SIGUSR1);phase["pause_signal_epoch"]=time.time();atomic(receipt,state)
        time.sleep(.025)

    failed=False
    try:
        atomic(receipt,state)
        require(time.time()<finite["engineering_deadline_epoch"],"Engineering admission deadline reached; no first price IO")
        cost_started=time.time();launch(first,cost=True)
        while active:tick()
        cost=json.loads((output/(first_id+"_COST_PACKAGE.json")).read_text())
        times=cost["subsequent_common_batch_seconds"];result=cost["result"]
        require(cost["protocol_sha256"]==protocol_sha and cost["absolute_deadline"]==end
                and cost["cost_only"] is True and cost["resume"] is False and len(times)==9
                and all(type(t) in (int,float) and math.isfinite(t) and t>0 for t in times)
                and result["exit_code"]==75 and result["common_steps"]==10
                and result["reason"]=="fixed_first10_cost_package_complete","Invalid first-ten cost qualification")
        common_upper=2*max(times);grace=max(60.,common_upper+10.)
        remaining_training=max(loads)*common_upper
        projected_remaining=remaining_training+nontraining_seconds
        admitted=time.time()+projected_remaining<end
        admission={"accepted":admitted,"protocol_sha256":protocol_sha,"absolute_deadline":end,
                   "first_cost_actual_elapsed_seconds":time.time()-cost_started,"all_window_elapsed_seconds":time.time()-start,
                   "common_batch_upper_seconds":common_upper,"upper_rule":"2*max(subsequent9)",
                   "Q_batch_counts":counts,"epochs":20,"list_scheduling_loads_full_common_batches":loads,
                   "projected_training_wall_seconds":remaining_training,"cost_bounds":bounds,
                   "projected_remaining_seconds":projected_remaining,"earlystop_discount":False,
                   "first10_training_credit_applied":False,"screen_complete":False,"deadline_reset":False}
        atomic(output/"REMAINING_QUEUE_ADMISSION.json",admission);state["cost_admission"]=admission;atomic(receipt,state)
        require(admitted,"Same-window complete projection does not fit; NOT_ADMITTED")
        pending=list(order)
        while pending or active:
            busy={assignment[record["phase"]["job_id"]] for record in active}
            for job in list(pending):
                slot=assignment[job["job_id"]]
                if slot in busy:continue
                launch(job,resume=job["job_id"]==first_id);pending.remove(job);busy.add(slot)
            tick()
        for job in jobs:
            result=json.loads((output/(job["job_id"]+"_PAIR_WORKER_RESULT.json")).read_text())
            require(result["protocol_sha256"]==protocol_sha and result["absolute_deadline"]==end
                    and result["C_cal_T_role_features_labels_constructed"] is False
                    and result["all_preQ3_close_columns_physically_decoded"] is True
                    and result["result"]["exit_code"]==0,
                    "Incomplete paired Q/S output")
        state["status"]="Q_S_QUEUE_COMPLETE_AWAITING_POSTPROCESSING"
    except BaseException as error:
        failed=True;state["status"]="NOT_ADMITTED_OR_INCOMPLETE_PRESERVE_CHECKPOINTS"
        state["error"]={"type":type(error).__name__,"message":str(error)}
    finally:
        # One safe-stop request, bounded by the original cutoff/deadline. Never resume.
        for record in active:
            if record["proc"].poll() is None and record["phase"]["pause_signal_epoch"] is None:
                try:os.kill(record["proc"].pid,signal.SIGUSR1)
                except ProcessLookupError:pass
                record["phase"]["pause_signal_epoch"]=time.time()
        stop_by=min(end,queue_cutoff,time.time()+grace)
        while active and time.time()<stop_by:
            for record in list(active):
                try:
                    for line in Path(f"/proc/{record['proc'].pid}/status").read_text().splitlines():
                        for source,key in (("VmRSS:","RSS_bytes"),("VmHWM:","VmHWM_bytes")):
                            if line.startswith(source):record["phase"][key]=max(record["phase"][key],int(line.split()[1])*1024)
                    if max(record["phase"]["RSS_bytes"],record["phase"]["VmHWM_bytes"])>RSS_LIMIT:
                        os.killpg(record["proc"].pid,signal.SIGKILL);record["phase"]["forced_kill"]=True
                except ProcessLookupError:pass
                except FileNotFoundError:pass
                if record["proc"].poll() is not None:
                    try:finish(record)
                    except Exception:pass
            if active:time.sleep(.025)
        for record in list(active):
            if record["proc"].poll() is None:
                try:os.killpg(record["proc"].pid,signal.SIGKILL)
                except ProcessLookupError:pass
                record["phase"]["forced_kill"]=True
            try:finish(record)
            except Exception:pass
        state.update(finished_epoch=time.time(),screen_complete=False,postprocessing_executed=False)
        atomic(receipt,state)
        for signum,handler in handlers.items():signal.signal(signum,handler)
    print(json.dumps(state,allow_nan=False),flush=True)
    return 1 if failed else 0


if __name__=="__main__":raise SystemExit(main())
