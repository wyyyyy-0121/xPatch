"""Small real-shape synthetic Q/S CLI. Root owns hard time/resource supervision."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

PREPARATION=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(PREPARATION))
import numpy as np
import torch
from common_adapter import GlobalPhaseGate,access
from artificial_adapter_tests import fixture,adapter
from train_pair import train_pair,SCIENCE_RULES_SHA256


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observe_committed_payload(outputdir, summary):
    """Copy an already committed payload; never inspect or mutate numerical state."""
    step=summary["common_steps"]
    if step not in (1,2):return
    label="FIRST_COMMON_BATCH" if step==1 else "SECOND_COMMON_BATCH"
    pointer=json.loads((outputdir/"LATEST.json").read_text())
    if pointer["payload"] not in ("latest_0.pt","latest_1.pt"):raise ValueError("Invalid committed slot")
    body=(outputdir/pointer["payload"]).read_bytes()
    if hashlib.sha256(body).hexdigest()!=pointer["sha256"]:raise ValueError("Committed payload changed")
    destination=outputdir/(label+".pt")
    if destination.exists() or destination.with_suffix(".json").exists():raise ValueError("Duplicate observation snapshot")
    temporary=destination.with_suffix(".pt.tmp-"+str(os.getpid()))
    with temporary.open("xb") as stream:
        stream.write(body);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,destination)
    receipt={"payload":destination.name,"sha256":pointer["sha256"],
             "identity_sha256":pointer["identity_sha256"],**summary,
             "observation_only":True,"RNG_or_payload_changed":False}
    temporary=destination.with_suffix(".json.tmp-"+str(os.getpid()))
    with temporary.open("x") as stream:
        json.dump(receipt,stream,sort_keys=True,allow_nan=False);stream.write("\n")
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,destination.with_suffix(".json"))
    fd=os.open(outputdir,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputdir",type=Path,required=True)
    parser.add_argument("--absolute-deadline",type=float,required=True)
    parser.add_argument("--seed",type=int,choices=(2026,2027,2028),default=2026)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--hold-first-progress",action="store_true")
    args=parser.parse_args()
    if not args.outputdir.is_absolute():raise ValueError("outputdir must be absolute")
    if sorted(os.sched_getaffinity(0))!=[24]:raise ValueError("Artificial worker must use CPU24")
    for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        if os.environ.get(name)!="1":raise ValueError(name+" must be 1")
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="-1":raise ValueError("GPU must be hidden")
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if sha(PREPARATION/"SCIENCE_RULES.json")!=SCIENCE_RULES_SHA256:raise ValueError("SCIENCE_RULES changed")
    prototype=PREPARATION.parent/"20261110_session_component_source_review/artificial_model/prototype_model.py"
    spec=importlib.util.spec_from_file_location("frozen_pair_artificial_prototype",prototype)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    f=fixture("hourly_h8_future_log_return","pair_training_artificial_hour_f1")
    gate=GlobalPhaseGate([f["state"]]);a=adapter(f,gate)
    a.fit_scaler(access(a.state,"Q","fit_scaler"))
    identity={"source":{"prototype":sha(prototype),"adapter":sha(PREPARATION/"common_adapter.py")},
              "code":{"kernel":sha(Path(__file__).with_name("train_pair.py")),"CLI":sha(__file__),
                      "synthetic_fixture":sha(PREPARATION/"artificial_adapter_tests.py")},
              "spec":SCIENCE_RULES_SHA256,"data":{"kind":"built_in_artificial_H8_L96_no_Ccal_T_operations",
              "Q_origins_sha256":hashlib.sha256(a.roles["Q"].tobytes()).hexdigest(),
              "S_origins_sha256":hashlib.sha256(a.roles["S"].tobytes()).hexdigest()}}
    held=False
    def progress(event,summary,token):
        nonlocal held
        if event=="COMMON_BATCH_COMMITTED":observe_committed_payload(args.outputdir,summary)
        print("PROGRESS "+json.dumps({"event":event,**summary}),flush=True)
        if args.hold_first_progress and not held and event=="COMMON_BATCH_COMMITTED" and summary["common_steps"]==1:
            held=True;until=min(time.monotonic()+5,time.monotonic()+max(0,args.absolute_deadline-time.time()-10))
            while time.monotonic()<until and not token.requested:time.sleep(.025)
    result=train_pair(a,module.paired_models,seed=args.seed,absolute_deadline=args.absolute_deadline,
                      outputdir=args.outputdir,identity=identity,resume=args.resume,progress_hook=progress)
    print("RESULT "+json.dumps(result,allow_nan=False),flush=True)
    raise SystemExit(result["exit_code"])


if __name__=="__main__":main()
