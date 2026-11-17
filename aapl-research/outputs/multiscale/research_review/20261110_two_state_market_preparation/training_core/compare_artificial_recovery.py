"""Independent weights_only, bitwise checkpoint comparison; no training imports.

Only artificial acceptance checkpoints are opened. Tensor storage bytes and
Python FP64 bit patterns are compared recursively; torch.save file bytes may differ.
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
import struct
import time
import torch

SCIENCE_SHA="e9e562d5b6f9d795104389f9d91950a737765566d041524dd8cd27360a1ba12b"
ARMS=("C","R")
THREAD_ENV=("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS")


def require(value,message):
    if not bool(value):raise AssertionError(message)


def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha(value):return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic(path,value):
    temporary=path.with_name(path.name+".tmp-"+str(os.getpid()))
    with temporary.open("x") as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write("\n")
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def same(left,right,path="root",counts=None):
    """Fail at the first unequal leaf without tolerances or numerical imports."""
    counts={} if counts is None else counts
    require(type(left) is type(right),path+": Python type differs")
    if isinstance(left,torch.Tensor):
        require(left.device.type==right.device.type=="cpu" and left.layout==right.layout==torch.strided,
                path+": CPU dense tensors required")
        require(left.dtype==right.dtype and left.shape==right.shape and left.stride()==right.stride(),
                path+": Tensor metadata differs")
        a=left.detach().contiguous().reshape(-1).view(torch.uint8)
        b=right.detach().contiguous().reshape(-1).view(torch.uint8)
        require(torch.equal(a,b),path+": Tensor bytes differ")
        counts["tensor_leaves"]=counts.get("tensor_leaves",0)+1
        counts["tensor_bytes"]=counts.get("tensor_bytes",0)+a.numel()
    elif isinstance(left,dict):
        require(set(left)==set(right),path+": Mapping keys differ")
        for key in sorted(left,key=repr):same(left[key],right[key],path+"["+repr(key)+"]",counts)
    elif isinstance(left,(list,tuple)):
        require(len(left)==len(right),path+": Sequence length differs")
        for index,(a,b) in enumerate(zip(left,right)):same(a,b,path+"["+str(index)+"]",counts)
    elif type(left) is float:
        require(math.isfinite(left) and math.isfinite(right),path+": Nonfinite scalar")
        require(struct.pack("!d",left)==struct.pack("!d",right),path+": FP64 scalar bytes differ")
        counts["scalar_leaves"]=counts.get("scalar_leaves",0)+1
    elif left is None or type(left) in (str,int,bool):
        require(left==right,path+": Scalar differs")
        counts["scalar_leaves"]=counts.get("scalar_leaves",0)+1
    else:raise AssertionError(path+": Unsupported checkpoint type "+type(left).__name__)
    return counts


def load(path):
    # No model construction, imports, optimizer operation, or pickle fallback.
    return torch.load(path,map_location="cpu",weights_only=True)


def latest(directory):
    pointer=json.loads((directory/"LATEST.json").read_text())
    require(pointer["payload"] in ("latest_0.pt","latest_1.pt"),"Invalid committed latest slot")
    path=directory/pointer["payload"]
    require(sha(path)==pointer["sha256"],"Committed latest checksum differs")
    payload=load(path)
    require(pointer["identity_sha256"]==object_sha(payload["identity"]),"Pointer identity differs")
    return payload


def observation(directory,label):
    receipt=json.loads((directory/(label+".json")).read_text())
    require(receipt["payload"]==label+".pt" and receipt["observation_only"] is True,
            "Invalid observation receipt")
    path=directory/receipt["payload"]
    require(sha(path)==receipt["sha256"],"Observation checksum differs")
    payload=load(path)
    require(receipt["identity_sha256"]==object_sha(payload["identity"]),"Observation identity differs")
    if label!="PAUSE_COMMITTED":
        for key in ("phase","epoch","cursor","common_steps"):
            require(receipt[key]==payload["run"][key],"Observation metadata differs: "+key)
    return payload


def validate_latest(payload,protocol):
    require(payload["kind"]=="pair_latest" and payload["schema"]==1 and payload["resumable"] is True,
            "Only complete latest checkpoints admitted")
    identity=payload["identity"];scaler=payload["scaler"]
    require(identity["absolute_deadline"]==protocol["deadline_epoch"] and identity["seed"]==protocol["seed"],
            "Checkpoint deadline/seed differs")
    require(identity["caller"]["spec"]==SCIENCE_SHA,"Frozen science identity differs")
    base=Path(__file__).resolve().parent;preparation=base.parent
    locks={str(Path(row["path"]).resolve()):row["sha256"] for row in protocol["locks"]}
    files={"source":{"prototype":preparation.parent/"20261110_session_component_source_review/artificial_model/prototype_model.py",
                     "adapter":preparation/"common_adapter.py"},
           "code":{"kernel":base/"train_pair.py","CLI":base/"artificial_train_fixture.py",
                   "synthetic_fixture":preparation/"artificial_adapter_tests.py"}}
    for section,entries in files.items():
        for key,path in entries.items():
            require(identity["caller"][section][key]==locks[str(path.resolve())],"Checkpoint source/code lock differs")
    require(identity["scaler_id"]==object_sha(scaler) and scaler["state"]==identity["state"]
            and scaler["roles_id"]==identity["roles_id"],"Checkpoint scaler/state binding differs")
    require(identity["state"]["task"]=="hourly_h8_future_log_return" and scaler["n_q"]==29,
            "Unexpected artificial state/shape")
    runtime=identity["runtime"]
    require(runtime["CPU_affinity"]==[24] and runtime["threads"]==runtime["interop_threads"]==1
            and runtime["numpy"]=="2.5.4" and runtime["deterministic_algorithms"] is True,
            "Checkpoint runtime differs")
    spec=identity["training_spec"]
    require(spec["max_epochs"]==20 and spec["patience"]==3 and spec["batch_size"]==128
            and spec["final_P_refit"] is False and spec["eligible_epochs"]==[1,20],"Training spec differs")
    expected={"zero_scaled":(8,),"head.weight":(8,82),"head.bias":(8,)}
    for cell in ("cell1","cell2"):
        expected.update({cell+".weight_ih":(96,2),cell+".weight_hh":(96,32),
                         cell+".bias_ih":(96,),cell+".bias_hh":(96,)})
    require(set(payload["models"])==set(payload["adam"])==set(ARMS),"Paired arms missing")
    for arm in ARMS:
        model=payload["models"][arm]
        require(set(model)==set(expected),"Unexpected model state keys")
        for name,shape in expected.items():
            tensor=model[name]
            require(isinstance(tensor,torch.Tensor) and tensor.device.type=="cpu" and tensor.dtype==torch.float32
                    and tuple(tensor.shape)==shape and torch.isfinite(tensor).all(),"Invalid model tensor "+arm+"/"+name)
        groups=payload["adam"][arm]["param_groups"]
        require(len(groups)==1 and groups[0]["lr"]==.001 and tuple(groups[0]["betas"])==(.9,.999)
                and groups[0]["eps"]==1e-8 and groups[0]["weight_decay"]==0
                and groups[0]["amsgrad"] is False and groups[0]["foreach"] is False
                and groups[0]["fused"] is False,"Adam frozen settings differ")


def compare_best(directory,final,arm):
    best=load(directory/("best_"+arm+".pt"));record=final["run"]["arms"][arm]
    require(best["kind"]=="pair_best_inference_only" and best["schema"]==1
            and best["inference_only"] is True and best["resumable"] is False
            and best["optimizer_included"] is False and best["RNG_included"] is False,
            "Best must be inference-only and nonresumable")
    require(not any(key in best for key in ("adam","optimizer","torch_cpu_rng","shuffle_rng","outputdir","path")),
            "Best contains resume state or output location")
    same(best["identity"],final["identity"],"best.identity")
    same(best["scaler"],final["scaler"],"best.scaler")
    require(best["arm"]==arm and best["seed"]==final["identity"]["seed"]
            and best["state"]==final["identity"]["state"] and best["scaler_id"]==final["identity"]["scaler_id"]
            and best["best_epoch"]==record["best_epoch"] and 1<=best["best_epoch"]<=final["run"]["epoch"],
            "Best metadata/state/eligible epoch differs")
    same(best["best_stop_mse"],record["best_loss"],"best.stop_metric")
    same(best["model"],final["run"]["best_models"][arm],"best.selected_weights")
    return best


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--run-directory",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    require(args.output.is_absolute() and not args.output.exists(),"Fresh absolute comparison output required; no retry/overwrite")
    began=time.time();report={"status":"FAIL","bitwise_equal":False,
        "started_epoch":began,"PID":os.getpid(),"weights_only":True,"training_module_imported":False,
        "market_arrays_opened":False,"C_cal_T_operations":False,"research_goal_achieved":False,"checks":[]}
    failed=False
    try:
        require(sorted(os.sched_getaffinity(0))==[24] and all(os.environ.get(n)=="1" for n in THREAD_ENV)
                and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1","Comparator resource environment differs")
        require(resource.getrlimit(resource.RLIMIT_AS)==(6*1024**3,6*1024**3),"Comparator AS6GiB limit differs")
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        body=args.protocol.read_bytes();protocol=json.loads(body)
        report["protocol_sha256"]=hashlib.sha256(body).hexdigest()
        require(protocol["kind"]=="two_state_training_artificial_recovery_v1"
                and protocol["deadline_epoch"]-protocol["started_epoch"]==300
                and time.time()<protocol["deadline_epoch"],"Original comparison deadline expired")
        require(args.run_directory.resolve()==Path(protocol["output_directory"]).resolve()
                and args.output.resolve()==(args.run_directory/"COMPARISON.json").resolve()
                and not args.output.exists(),"Comparison destination differs or exists")
        # Popen precedes parent receipt publication: bounded handshake only,
        # never a retry of training, process creation, or numerical comparison.
        handshake_until=time.monotonic()+min(1.,max(0.,protocol["deadline_epoch"]-time.time()))
        while True:
            supervision=json.loads((args.run_directory/"SUPERVISOR.json").read_text())
            stages=supervision["phases"]
            if len(stages)==4 and stages[3].get("PID")==os.getpid():break
            require(time.monotonic()<handshake_until,"Parent PID manifest handshake timed out")
            time.sleep(min(.01,max(0.,handshake_until-time.monotonic())))
        require(len(stages)==4 and [stage["role"] for stage in stages[:3]]==
                ["01_external_pause","02_new_PID_resume","03_uninterrupted"],"Expected external process sequence missing")
        require([stage["exit_code"] for stage in stages[:3]]==[75,0,0]
                and stages[0]["PID"]!=stages[1]["PID"]
                and stages[0]["SIGUSR1_trigger"]=="external_after_first_common_batch"
                and stages[0]["SIGUSR1_target_PID"]==stages[0]["PID"]
                and stages[0]["SIGUSR1_sent_epoch"] is not None,"External SIGUSR1/new-PID evidence missing")
        for stage in stages[:3]:
            command=stage["command"];index=command.index("--absolute-deadline")
            require(float(command[index+1])==protocol["deadline_epoch"]
                    and stage["termination_reason"] is None and not stage.get("forced_kill",False),
                    "Child deadline or terminal evidence differs")
        require("--resume" not in stages[0]["command"] and "--resume" in stages[1]["command"]
                and "--resume" not in stages[2]["command"],"Resume command sequence differs")
        require(stages[3]["PID"]==os.getpid(),"Comparison must be a separate supervised process")
        report["checks"].append({"check":"external_SIGUSR1_exit75_new_PID_same_deadline","status":"PASS"})
        resumed=args.run_directory/"resumed_pair";control=args.run_directory/"uninterrupted_pair"
        rfinal=latest(resumed);cfinal=latest(control)
        pause_receipt=json.loads((args.run_directory/"PAUSE_COMMITTED.json").read_text())
        require(pause_receipt["payload"]=="pause_committed.pt" and pause_receipt["copied_after_exit75"] is True,
                "Pause copy missing")
        pause_path=args.run_directory/pause_receipt["payload"]
        require(sha(pause_path)==pause_receipt["sha256"],"Pause copy checksum differs")
        paused=load(pause_path)
        paused_result=json.loads((resumed/"PAUSED.json").read_text())
        require(paused_result["exit_code"]==75 and paused_result["reason"]=="signal_"+str(signal.SIGUSR1)
                and paused_result["absolute_deadline"]==protocol["deadline_epoch"]
                and paused_result["common_steps"]==1,"Pause receipt does not confirm actual external safe pause")
        require(pause_receipt["identity_sha256"]==object_sha(paused["identity"]),"Pause identity differs")
        checkpoints=[rfinal,cfinal,paused]
        for label,step in (("FIRST_COMMON_BATCH",1),("SECOND_COMMON_BATCH",2)):
            a=observation(resumed,label);b=observation(control,label);checkpoints.extend((a,b))
            require(a["run"]["common_steps"]==b["run"]["common_steps"]==step
                    and a["run"]["phase"]==b["run"]["phase"]=="train"
                    and a["run"]["cursor"]==b["run"]["cursor"]==29,"Committed-step snapshot differs")
            for arm in ARMS:
                require(a["run"]["arms"][arm]["optimizer_updates"]==step
                        and b["run"]["arms"][arm]["optimizer_updates"]==step,"Arm update count differs")
                for state in a["adam"][arm]["state"].values():
                    require(float(state["step"])==step,"Adam step differs from common commit")
            counts=same(a,b,label)
            report["checks"].append({"check":label+"_recursive_bitwise","status":"PASS",**counts})
            if step==1:
                same(paused,a,"external_pause_equals_first_common_commit");first=a
            else:
                for arm in ARMS:
                    changed=any(not torch.equal(first["models"][arm][key].contiguous().reshape(-1).view(torch.uint8),
                                               a["models"][arm][key].contiguous().reshape(-1).view(torch.uint8))
                                for key in a["models"][arm])
                    require(changed,"Second common update changed no parameter in "+arm)
        for checkpoint in checkpoints:
            validate_latest(checkpoint,protocol)
            same(checkpoint["identity"],rfinal["identity"],"all_checkpoint_identity")
            same(checkpoint["scaler"],rfinal["scaler"],"all_checkpoint_scaler")
        require(rfinal["run"]["phase"]==cfinal["run"]["phase"]=="done"
                and rfinal["run"]["epoch"]>=2 and all(not rfinal["run"]["arms"][arm]["active"] for arm in ARMS),
                "Final paired training incomplete")
        counts=same(rfinal,cfinal,"final_latest_models_Adam_RNG_run_history_best")
        report["checks"].append({"check":"final_latest_recursive_bitwise","status":"PASS",**counts})
        for arm in ARMS:
            a=compare_best(resumed,rfinal,arm);b=compare_best(control,cfinal,arm)
            counts=same(a,b,"best_"+arm)
            report["checks"].append({"check":"best_"+arm+"_inference_only_bitwise","status":"PASS",**counts})
        require(time.time()<=protocol["deadline_epoch"],"Comparison exceeded original deadline")
        report.update(status="PASS",bitwise_equal=True,checkpoint_identity_sha256=object_sha(rfinal["identity"]),
                      original_deadline_epoch=protocol["deadline_epoch"],deadline_reset=False,
                      same_initial_and_next_update=True,pause_equals_first_commit=True,
                      external_signal_and_new_PID_confirmed=True)
    except BaseException as error:
        failed=True;report["error"]={"type":type(error).__name__,"message":str(error)}
    report.update(finished_epoch=time.time(),elapsed_seconds=time.time()-began)
    atomic(args.output,report)
    print(json.dumps(report,allow_nan=False),flush=True)
    return 1 if failed else 0


if __name__=="__main__":raise SystemExit(main())
