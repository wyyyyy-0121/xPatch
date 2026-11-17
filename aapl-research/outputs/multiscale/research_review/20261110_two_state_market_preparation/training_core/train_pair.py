"""Q/S-only paired CPU training kernel; no market reader or scheduler.

The supplied adapter/factory/identities must have been qualified by the caller.
Only latest is resumable; best files are explicitly inference-only.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import copy
import hashlib
import json
import math
import os
import signal
import time
import numpy as np
import torch
from common_adapter import access

SCIENCE_RULES_SHA256="e9e562d5b6f9d795104389f9d91950a737765566d041524dd8cd27360a1ba12b"
ARMS=("C","R")
FLAGS={"C":0,"R":1}
SPEC={"max_epochs":20,"patience":3,"batch_size":128,"loss":"standardized_full_path_MSE",
      "lr":.001,"betas":[.9,.999],"eps":1e-8,"weight_decay":0.,"amsgrad":False,"foreach":False,"fused":False,
      "shuffle":"Generator(PCG64(seed))","eligible_epochs":[1,20],"final_P_refit":False}


class TrainingError(RuntimeError):pass


def require(value,message):
    if not bool(value):raise TrainingError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)


class StopToken:
    def __init__(self):self.requested=False;self.reason=None
    def request(self,reason):self.requested=True;self.reason=str(reason)


def _fsync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def _atomic(path,value,tensor=False):
    temporary=path.with_name(path.name+".tmp-"+str(os.getpid()))
    with temporary.open("xb") as stream:
        if tensor:torch.save(value,stream)
        else:stream.write((canonical(value)+"\n").encode())
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path);_fsync_dir(path.parent)


class LatestStore:
    def __init__(self,path,identity,resume):
        self.path=Path(path);self.identity=identity;self.generation=0
        self.path.mkdir(parents=True,exist_ok=True)
        if resume:
            require(not (self.path/"DONE.json").exists(),"Completed run may not resume")
            require((self.path/"LATEST.json").is_file(),"Resume requires committed latest pointer")
        else:require(not any(self.path.iterdir()),"Fresh outputdir must be empty")

    def save(self,payload):
        generation=self.generation+1
        name="latest_"+str(generation%2)+".pt"
        _atomic(self.path/name,payload,tensor=True)
        pointer={"schema":1,"payload":name,"sha256":sha(self.path/name),"generation":generation,
                 "identity_sha256":hashlib.sha256(canonical(self.identity).encode()).hexdigest()}
        _atomic(self.path/"LATEST.json",pointer)
        self.generation=generation
        return pointer

    def load(self):
        pointer=json.loads((self.path/"LATEST.json").read_text())
        require(pointer.get("payload") in ("latest_0.pt","latest_1.pt"),"Invalid latest slot")
        require(pointer["identity_sha256"]==hashlib.sha256(canonical(self.identity).encode()).hexdigest(),"Resume identity/deadline mismatch")
        path=self.path/pointer["payload"]
        require(path.is_file() and sha(path)==pointer["sha256"],"Latest payload checksum mismatch")
        payload=torch.load(path,map_location="cpu",weights_only=True)
        require(payload.get("kind")=="pair_latest" and payload.get("resumable") is True,
                "Only complete latest may resume; best is inference-only")
        require(payload["identity"]==self.identity,"Latest semantic identity mismatch")
        self.generation=pointer["generation"]
        return payload


def _optim(model):
    return torch.optim.Adam(model.parameters(),lr=SPEC["lr"],betas=tuple(SPEC["betas"]),eps=SPEC["eps"],
                            weight_decay=0.,amsgrad=False,foreach=False,fused=False)


def _tensors(adapter,role,origins,*,scaled_target):
    v=adapter.network_inputs(access(adapter.state,role,"inputs"),origins)
    target=adapter.labels(access(adapter.state,role,"labels"),origins,scaled=scaled_target)
    # tensor(copy) deliberately owns writable storage distinct from immutable adapter arrays.
    x=torch.tensor(v["x"].copy(),dtype=torch.float32)
    g=torch.tensor(v["boundary"].copy(),dtype=torch.bool)
    a=torch.tensor(v["anchor"].copy(),dtype=torch.float32)
    c=torch.tensor(v["context"].copy(),dtype=torch.float32)
    y=torch.tensor(target.copy(),dtype=torch.float32) if scaled_target else target
    return (x,g,a,c),y


def _forward(model,inputs,arm):
    x,g,a,c=inputs
    return model(x,g,a,c,reset_flag=FLAGS[arm])["output"]


def _snapshot(identity,adapter,models,optimizers,rng,state):
    return {"kind":"pair_latest","schema":1,"resumable":True,"identity":copy.deepcopy(identity),
            "scaler":asdict(adapter.scaler),"models":{a:copy.deepcopy(models[a].state_dict()) for a in ARMS},
            "adam":{a:copy.deepcopy(optimizers[a].state_dict()) for a in ARMS},
            "torch_cpu_rng":torch.get_rng_state().clone(),
            "shuffle_rng":json.loads(json.dumps(rng.bit_generator.state)),"run":copy.deepcopy(state)}


def _best_payload(identity,adapter,arm,state):
    record=state["arms"][arm]
    require(record["best_epoch"] is not None and state["best_models"][arm] is not None,"No eligible best yet")
    return {"kind":"pair_best_inference_only","schema":1,"resumable":False,"inference_only":True,
            "identity":copy.deepcopy(identity),"state":adapter.state.as_dict(),"arm":arm,
            "seed":identity["seed"],"best_epoch":record["best_epoch"],"best_stop_mse":record["best_loss"],
            "scaler":asdict(adapter.scaler),"scaler_id":adapter.scaler.scaler_id,
            "model":copy.deepcopy(state["best_models"][arm]),"optimizer_included":False,"RNG_included":False}


def _write_bests(store,identity,adapter,state):
    result={}
    for arm in ARMS:
        if state["arms"][arm]["best_epoch"] is not None:
            path=store.path/("best_"+arm+".pt")
            _atomic(path,_best_payload(identity,adapter,arm,state),tensor=True)
            result[arm]={"path":path.name,"sha256":sha(path),"best_epoch":state["arms"][arm]["best_epoch"],
                         "inference_only":True,"resumable":False}
    return result


def train_pair(adapter,paired_models_factory,*,seed,absolute_deadline,outputdir,identity,
               resume=False,stop_token=None,progress_hook=None,install_signal_handlers=True):
    """Train C/R on Q, select each best on S. Return exit_code 0 or safe-pause75.

    progress_hook(event, summary, StopToken) is an optional artificial observer;
    it must not alter models, data, RNG or scientific settings.
    The caller owns hard deadline/resource supervision and all file identities.
    """
    require(type(seed) is int and seed in (2026,2027,2028),"Unregistered seed")
    require(type(absolute_deadline) in (int,float) and math.isfinite(absolute_deadline)
            and time.time()<absolute_deadline,"Absolute deadline already reached")
    require(adapter.gate.phase=="TRAIN","Only TRAIN/Q/S phase is admitted")
    require(isinstance(identity,dict) and all(k in identity for k in ("source","code","data","spec")),"Complete caller source/code/data/spec identity required")
    identity=json.loads(canonical(identity))
    require(identity["spec"]==SCIENCE_RULES_SHA256,"Frozen SCIENCE_RULES identity required")
    require(np.__version__=="2.5.4","Frozen PCG64 NumPy version mismatch")
    require(len(os.sched_getaffinity(0))<=2 and torch.get_num_threads()<=2
            and torch.get_num_interop_threads()<=2,"Training CPU/thread cap exceeded")
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="-1","GPU must be hidden")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    models=paired_models_factory(adapter.H,torch.tensor(adapter.scaler.zero_scaled.copy(),dtype=torch.float32))
    require(set(models)==set(ARMS),"Exact C/R models required")
    for arm in ARMS:
        require(all(p.device.type=="cpu" and p.dtype==torch.float32 for p in models[arm].parameters()),"CPU FP32 prototype required")
    for name,value in models["C"].state_dict().items():
        require(torch.equal(value,models["R"].state_dict()[name]),"Initial C/R parameter/buffer mismatch")
    require(not ({p.untyped_storage().data_ptr() for p in models["C"].parameters()}
                 & {p.untyped_storage().data_ptr() for p in models["R"].parameters()}),"C/R mutable parameter alias")
    optimizers={a:_optim(models[a]) for a in ARMS}
    rng=np.random.Generator(np.random.PCG64(seed))
    bound={"caller":identity,"state":adapter.state.as_dict(),"roles_id":adapter.roles_id,
           "scaler_id":adapter.scaler.scaler_id,"seed":seed,"absolute_deadline":absolute_deadline,
           "training_spec":SPEC,"runtime":{"numpy":str(np.__version__),"torch":str(torch.__version__),
           "CPU_affinity":sorted(os.sched_getaffinity(0)),"threads":torch.get_num_threads(),
           "interop_threads":torch.get_num_interop_threads(),"deterministic_algorithms":True}}
    store=LatestStore(outputdir,bound,resume)
    token=stop_token if stop_token is not None else StopToken()
    old_handlers={}
    if install_signal_handlers:
        for signum in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT):
            old_handlers[signum]=signal.getsignal(signum)
            signal.signal(signum,lambda received,_frame:token.request("signal_"+str(received)))
    q=adapter.roles["Q"];s=adapter.roles["S"]
    state={"phase":"train","epoch":0,"cursor":0,"permutation":None,"common_steps":0,
           "arms":{a:{"active":True,"stale":0,"best_loss":None,"best_epoch":None,
                       "optimizer_updates":0,"loss_sum":0.,"seen":0} for a in ARMS},
           "best_models":{a:None for a in ARMS},"history":[]}
    def commit():return store.save(_snapshot(bound,adapter,models,optimizers,rng,state))
    def observe(event):
        if progress_hook is not None:
            progress_hook(event,{"phase":state["phase"],"epoch":state["epoch"],"cursor":state["cursor"],
                                "common_steps":state["common_steps"],"Q_count":len(q)},token)
    def should_pause():
        if absolute_deadline-time.time()<10:token.request("deadline_grace_under_10s")
        return token.requested
    def pause():
        pointer=commit()
        result={"status":"paused_safe_complete_common_batch","exit_code":75,"reason":token.reason,
                "phase":state["phase"],"epoch":state["epoch"],"cursor":state["cursor"],
                "common_steps":state["common_steps"],"latest":pointer,"deadline_reset":False,
                "absolute_deadline":absolute_deadline,"best_inference":_write_bests(store,bound,adapter,state)}
        _atomic(store.path/"PAUSED.json",result)
        return result
    try:
        if resume:
            saved=store.load()
            require(saved["scaler"]==asdict(adapter.scaler),"State Q scaler changed on resume")
            state=saved["run"]
            require(state["phase"] in ("train","validate","done") and 0<=state["epoch"]<=20,"Latest run phase/epoch invalid")
            require(0<=state["cursor"]<=len(q) and (state["cursor"]%128==0 or state["cursor"]==len(q)),"Latest cursor invalid")
            require(state["phase"]!="validate" or (state["cursor"]==len(q) and state["permutation"] is not None),"Validation resume must not repeat Q updates")
            require(state["permutation"] is not None or state["cursor"]==0,"Cursor without permutation")
            for arm in ARMS:
                record=state["arms"][arm]
                if record["best_epoch"] is None:
                    require(state["epoch"]==0 and record["best_loss"] is None and state["best_models"][arm] is None,"Invalid epoch0/best semantics")
                else:
                    require(1<=record["best_epoch"]<=state["epoch"] and math.isfinite(record["best_loss"])
                            and state["best_models"][arm] is not None,"Best record not from a completed eligible epoch")
            if state["permutation"] is not None:
                perm=state["permutation"]
                require(perm.dtype==torch.int64 and tuple(perm.shape)==(len(q),)
                        and torch.equal(torch.sort(perm).values,torch.arange(len(q))),"Latest permutation invalid")
            for arm in ARMS:
                models[arm].load_state_dict(saved["models"][arm],strict=True)
                optimizers[arm].load_state_dict(saved["adam"][arm])
            rng.bit_generator.state=saved["shuffle_rng"]
            torch.set_rng_state(saved["torch_cpu_rng"])
        else:commit()  # Epoch0 is resumable but never an inference candidate.
        while state["phase"]!="done":
            if should_pause():return pause()
            if state["phase"]=="train":
                if state["permutation"] is None:
                    state["permutation"]=torch.tensor(rng.permutation(len(q)),dtype=torch.int64)
                    state["cursor"]=0
                if state["cursor"]<len(q):
                    begin=state["cursor"];end=min(begin+128,len(q))
                    batch=q[state["permutation"][begin:end].numpy()]
                    inputs,target=_tensors(adapter,"Q",batch,scaled_target=True)
                    # A signal only sets a flag: both active arms finish before cursor commit.
                    for arm in ARMS:
                        record=state["arms"][arm]
                        if not record["active"]:continue
                        models[arm].train();optimizers[arm].zero_grad(set_to_none=True)
                        pred=_forward(models[arm],inputs,arm)
                        loss=(pred-target).square().mean()
                        require(torch.isfinite(loss),"Nonfinite standardized path loss")
                        loss.backward()
                        require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in models[arm].parameters()),"Nonfinite/missing gradient")
                        optimizers[arm].step()
                        record["optimizer_updates"]+=1;record["seen"]+=len(batch)
                        record["loss_sum"]+=float(loss.detach())*len(batch)
                    state["cursor"]=end;state["common_steps"]+=1
                    commit();observe("COMMON_BATCH_COMMITTED")
                    continue
                state["phase"]="validate";commit();observe("VALIDATE_START")
            if state["phase"]=="validate":
                # No state is changed by a partial S scan; a resumed scan restarts here.
                totals={a:0. for a in ARMS};count=0
                for start in range(0,len(s),128):
                    if should_pause():return pause()
                    origins=s[start:start+128]
                    inputs,truth=_tensors(adapter,"S",origins,scaled_target=False)
                    with torch.inference_mode():
                        for arm in ARMS:
                            if not state["arms"][arm]["active"]:continue
                            models[arm].eval();scaled=_forward(models[arm],inputs,arm).detach().cpu().numpy()
                            raw=adapter.scaler.inverse(adapter.state,scaled)
                            totals[arm]+=float(np.sum((raw[:,-1]-truth[:,-1])**2,dtype=np.float64))
                    count+=len(origins)
                if should_pause():return pause()
                require(count==len(s),"Incomplete S scan")
                epoch=state["epoch"]+1;history={"epoch":epoch,"arms":{}}
                for arm in ARMS:
                    record=state["arms"][arm]
                    if not record["active"]:continue
                    score=totals[arm]/count;require(math.isfinite(score),"Nonfinite S endpoint MSE")
                    if record["best_loss"] is None or score<record["best_loss"]:
                        record["best_loss"]=score;record["best_epoch"]=epoch;record["stale"]=0
                        state["best_models"][arm]=copy.deepcopy(models[arm].state_dict())
                    else:record["stale"]+=1
                    history["arms"][arm]={"S_raw_endpoint_MSE":score,"Q_scaled_path_MSE":record["loss_sum"]/record["seen"],
                                          "best_epoch":record["best_epoch"],"stale":record["stale"]}
                    record["active"]=record["stale"]<3 and epoch<20
                state["history"].append(history);state["epoch"]=epoch
                state["cursor"]=0;state["permutation"]=None
                for arm in ARMS:state["arms"][arm]["seen"]=0;state["arms"][arm]["loss_sum"]=0.
                state["phase"]="train" if any(state["arms"][a]["active"] for a in ARMS) else "done"
                commit();_write_bests(store,bound,adapter,state);observe("EPOCH_VALIDATED")
        bests=_write_bests(store,bound,adapter,state)
        result={"status":"Q_S_pair_complete","exit_code":0,"epoch":state["epoch"],"common_steps":state["common_steps"],
                "state":adapter.state.as_dict(),"seed":seed,"best_inference":bests,
                "absolute_deadline":absolute_deadline,"deadline_reset":False,"C_cal_or_T_constructed":False,
                "alpha_implemented":False,"predictive_success":False}
        _atomic(store.path/"DONE.json",result)
        return result
    except BaseException as error:
        # Never commit half-updated arms on an error. The prior pointer remains valid.
        try:_atomic(store.path/"ERROR.json",{"status":"terminal_error_preserve_last_committed_latest",
                    "type":type(error).__name__,"message":str(error),"automatic_retry":False})
        except BaseException:pass
        raise
    finally:
        for signum,handler in old_handlers.items():signal.signal(signum,handler)
