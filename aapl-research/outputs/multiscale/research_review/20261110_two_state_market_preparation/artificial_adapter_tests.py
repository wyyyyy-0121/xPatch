"""In-memory adapter fixtures only. Root owns the 180s protocol and receipt."""
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from dataclasses import FrozenInstanceError
import argparse
import hashlib
import json
import math
import os
import numpy as np
from common_adapter import (AdapterError, StateKey, GlobalPhaseGate, CommonAdapter,
                            access, context17, feature_names, endpoint_alpha,
                            apply_alpha, object_id, CONTRACT_SHA256)

RTOL, ATOL = 1e-12, 1e-13


def check(condition, name):
    if not bool(condition):
        raise AssertionError(name)


def same(a, b, name):
    check(np.shape(a) == np.shape(b) and np.allclose(a,b,rtol=RTOL,atol=ATOL), name)


def rejects(func, name):
    try:
        func()
    except AdapterError:
        return name
    raise AssertionError("Expected AdapterError: "+name)


def fixture(task, uid):
    minute=task.startswith("minute")
    L,H=(480,60) if minute else (96,8)
    days=[]; day=date(2020,2,3)
    while len(days)<(8 if minute else 50):
        if day.weekday()<5:days.append(day)
        day+=timedelta(days=1)
    ns=[]; sessions=[]
    for d in days:
        for j in range(390 if minute else 6):
            minutes=571+j if minute else 630+60*j
            t=datetime(d.year,d.month,d.day,minutes//60,minutes%60,tzinfo=ZoneInfo("America/New_York"))
            ns.append(int(t.astimezone(timezone.utc).timestamp())*1_000_000_000)
            sessions.append(d.isoformat())
    n=len(ns); i=np.arange(n,dtype=np.float64)
    close=(100*np.exp(.0001*i+.001*np.sin(.03*i))).astype(np.float64)
    if minute:
        roles={"Q":np.r_[np.arange(500,630),780].astype(np.int64),"S":np.arange(1000,1004,dtype=np.int64),
               "C_cal":np.arange(1400,1404,dtype=np.int64),"T":np.arange(1800,1804,dtype=np.int64)}
    else:
        roles={"Q":np.arange(101,130,dtype=np.int64),"S":np.arange(160,164,dtype=np.int64),
               "C_cal":np.arange(200,204,dtype=np.int64),"T":np.arange(240,244,dtype=np.int64)}
    for values in roles.values():values.flags.writeable=False
    return {"state":StateKey(task,1,uid),"close":close,"end_ns":np.asarray(ns,dtype=np.int64),
            "session_date":np.asarray(sessions,dtype="<U10"),"roles":roles,
            "split_actions":({"date":days[2].isoformat(),"ratio":2.},),"days":days,"L":L,"H":H}


def adapter(f,gate,close=None,actions=None):
    return CommonAdapter(f["state"],gate,close=f["close"] if close is None else close,
                         end_ns=f["end_ns"],session_date=f["session_date"],roles=f["roles"],
                         split_actions=f["split_actions"] if actions is None else actions)


def run_tests():
    passed=[]
    fs=[fixture("minute_h60_future_log_return","artificial_minute_f1"),
        fixture("hourly_h8_future_log_return","artificial_hour_f1")]
    gate=GlobalPhaseGate([f["state"] for f in fs]); adapters=[adapter(f,gate) for f in fs]
    for f,a in zip(fs,adapters):
        Q=f["roles"]["Q"]; state=f["state"]; L,H=f["L"],f["H"]
        qi=access(state,"Q","inputs"); qy=access(state,"Q","labels")
        raw=a.raw_inputs(qi,Q)
        for row,o in enumerate(Q):
            idx=np.arange(o-L+1,o+1)
            manual=np.log(f["close"][idx]/f["close"][o])
            for action in f["split_actions"]:
                manual+=(f["session_date"][idx]>=action["date"])*np.log(action["ratio"])
                manual-=(f["session_date"][o]>=action["date"])*np.log(action["ratio"])
            same(raw["history"][row],manual,"manual effective-at-origin split formula")
            expected_g=np.r_[False,f["session_date"][idx[1:]]!=f["session_date"][idx[:-1]]]
            last=int(np.flatnonzero(expected_g)[-1]) if expected_g.any() else 0
            expected_m=np.arange(L)>=last
            check(np.array_equal(raw["boundary"][row],expected_g) and np.array_equal(raw["suffix_mask"][row],expected_m),
                  "full g/m firstslot/lastboundary/suffix inclusion")
        check(not raw["boundary"][:,0].any(),"g first slot false")
        if H==60:
            check(f["session_date"][Q[0]]<f["split_actions"][0]["date"]
                  and f["session_date"][Q[-1]]>=f["split_actions"][0]["date"],"before/after split fixture")
            check(raw["boundary"][-1,-1] and np.count_nonzero(raw["suffix_mask"][-1])==1,"last-slot boundary mask")
        else:
            start=Q[0]-L+1
            check(f["session_date"][start]!=f["session_date"][start-1] and not raw["boundary"][0,0],
                  "window-start global boundary excluded")
        check(len(feature_names(H,L))==17 and len(feature_names(H,L)[:7])==7,"original seven lags retained")
        s=a.fit_scaler(access(state,"Q","fit_scaler"))
        y=a.labels(qy,Q)
        def expected_stats(v,axis=None):
            # Contract oracle: chunk along Q origins, including for flattened history.
            # A single global mean changes FP64 addition order before variance subtraction.
            total=0. if axis is None else np.zeros(v.shape[1:],dtype=np.float64)
            squares=0. if axis is None else np.zeros(v.shape[1:],dtype=np.float64)
            for first in range(0,len(v),128):
                part=v[first:first+128]
                if axis is None:
                    total+=float(np.sum(part,dtype=np.float64))
                    squares+=float(np.sum(part*part,dtype=np.float64))
                else:
                    total+=np.sum(part,axis=axis,dtype=np.float64)
                    squares+=np.sum(part*part,axis=axis,dtype=np.float64)
            count=v.size if axis is None else len(v)
            mean=total/count
            std=np.sqrt(np.maximum(squares/count-mean*mean,0.))
            return mean,np.where(std>1e-12,std,1.)
        xm,xs=expected_stats(raw["history"]); am,astd=expected_stats(raw["anchor"])
        cm,cs=expected_stats(raw["context"],axis=0); ym,ys=expected_stats(y,axis=0)
        same([s.x_mean,s.x_std,s.anchor_mean,s.anchor_std],[xm,xs,am,astd],"Q scalar population moments")
        same(s.context_mean,cm,"Q context means"); same(s.context_std,cs,"Q context stds")
        same(s.y_mean,ym,"Q target means"); same(s.y_std,ys,"Q target stds")
        v=a.network_inputs(qi,Q); repeated=a.network_inputs(qi,Q,scaler=s)
        for key in v:
            check(v[key].tobytes()==repeated[key].tobytes(),"common network bytes "+key)
            check(not v[key].flags.writeable,"immutable network tensor "+key)
        check(v["x"].tobytes()==((raw["history"]-s.x_mean)/s.x_std).astype(np.float32)[...,None].tobytes(),"FP32 x transform")
        check(v["anchor"].tobytes()==((raw["anchor"]-s.anchor_mean)/s.anchor_std).astype(np.float32).tobytes(),"FP32 anchor transform")
        check(v["context"].tobytes()==((raw["context"]-np.asarray(s.context_mean))/np.asarray(s.context_std)).astype(np.float32).tobytes(),"FP32 context transform")
        scaled_y=a.labels(qy,Q,scaled=True)
        check(scaled_y.tobytes()==((y-np.asarray(s.y_mean))/np.asarray(s.y_std)).astype(np.float32).tobytes(),"FP32 targets")
        E=a.ridge_features(qi,Q)
        product=(v["x"][:,:,0]*v["suffix_mask"].astype(np.float32)).astype(np.float32)
        expected=np.concatenate((v["x"][:,:,0].astype(np.float64),v["boundary"].astype(np.float64),
                                 v["suffix_mask"].astype(np.float64),product.astype(np.float64),
                                 v["anchor"].astype(np.float64),v["context"].astype(np.float64)),axis=1)
        check(E.shape==(len(Q),4*L+18) and E.dtype==np.float64 and E.tobytes()==expected.tobytes(),
              "complete E_Ridge dictionary exact FP32 product then FP64")
        zero=(-np.asarray(s.y_mean)/np.asarray(s.y_std)).astype(np.float32)
        check(s.zero_scaled.tobytes()==zero.tobytes(),"zero FP64 division before FP32 cast")
        inv=s.inverse(state,np.tile(zero,(2,1)))
        check(np.all(np.abs(inv)<=s.zero_inverse_bound()),"declared FP32 physical-zero inverse bound")
        reverse=Q[[len(Q)-1,0]]
        check(a.network_inputs(qi,reverse)["origins"].tobytes()==reverse.tobytes(),"shuffled batch retains origin order")
        for role in ("S","C_cal","T"):
            changed=f["close"].copy(); changed[f["roles"][role][0]:]*=1.3
            other=adapter(f,GlobalPhaseGate([state]),close=changed)
            s2=other.fit_scaler(access(state,"Q","fit_scaler"))
            check(s2==s and s2.scaler_id==s.scaler_id,"later "+role+"/future price cannot alter Q scaler")
            check(other.network_inputs(qi,Q)["x"].tobytes()==v["x"].tobytes(),"Q bytes invariant to later "+role)
        future_actions=f["split_actions"]+({"date":f["days"][-1].isoformat(),"ratio":3.},)
        other=adapter(f,GlobalPhaseGate([state]),actions=future_actions)
        check(other.fit_scaler(access(state,"Q","fit_scaler"))==s,"future split cannot alter earlier Q scaler")
        passed.append(state.task+": split/history/g/m/Qstats/FP32/E_Ridge/zero/later-role invariance")
    # Original context formula with manually known NY times on either side of DST.
    for L,H,hour,minute in ((96,8,15,30),(480,60,10,30)):
        history=np.tile(-.01*np.arange(L-1,-1,-1,dtype=np.float64),(2,1))
        stamps=[]
        for d in (date(2020,3,6),date(2020,3,9)):
            local=datetime(d.year,d.month,d.day,hour,minute,tzinfo=ZoneInfo("America/New_York"))
            stamps.append(int(local.timestamp())*1_000_000_000)
        check((stamps[1]-stamps[0])//1_000_000_000==3*86400-3600,"DST UTC shift fixture")
        c=context17(history,np.asarray(stamps,dtype=np.int64),H)
        lags=[int(n.rsplit("_",1)[1]) for n in feature_names(H,L)[:7]]
        same(c[:,:7],np.tile(.01*np.asarray(lags),(2,1)),"manual context lags")
        same(c[:,7:9],np.full((2,2),.01),"manual RMS")
        u=(hour*60+minute-570)/390
        clock=[math.sin(2*math.pi*u),math.cos(2*math.pi*u),math.sin(4*math.pi*u),math.cos(4*math.pi*u)]
        same(c[:,9:13],np.tile(clock,(2,1)),"manual NY completed clock/DST")
        check(np.array_equal(c[:,13:],np.asarray([[0,0,0,0],[1,0,0,0]],dtype=np.float64)),"manual Friday/Monday weekdays")
    passed.append("manual full17 seven-lags/RMS/NYclock/DST/weekday")
    a,b=adapters; f=fs[0]; state=f["state"]; Q=f["roles"]["Q"]
    passed.append(rejects(lambda:a.fit_scaler(access(state,"Q","fit_scaler")),"second scaler fit denied"))
    passed.append(rejects(lambda:a.network_inputs(access(state,"Q","inputs"),Q,scaler=b.scaler),"foreign state scaler denied"))
    passed.append(rejects(lambda:a.raw_inputs(access(fs[1]["state"],"Q","inputs"),Q),"foreign state key denied"))
    passed.append(rejects(lambda:a.labels(access(state,"Q","inputs"),Q),"operation mismatch denied"))
    passed.append(rejects(lambda:a.labels(access(state,"Q","labels"),f["roles"]["S"]),"wrong role membership denied"))
    for role in ("C_cal","T"):
        passed.append(rejects(lambda role=role:a.labels(access(state,role,"labels"),f["roles"][role]),"premature "+role))
    try:
        a.scaler.x_mean=0.
    except FrozenInstanceError:
        passed.append("frozen scaler field mutation denied")
    else:raise AssertionError("scaler mutation accepted")
    try:
        a.roles["Q"][0]=0
    except ValueError:
        passed.append("immutable role-array mutation denied")
    else:raise AssertionError("role mutation accepted")
    models={}
    for ad in adapters:
        for mid in gate.model_ids(ad.state):
            models[mid]={"state":ad.state.as_dict(),"model_sha256":hashlib.sha256(mid.encode()).hexdigest(),
                         "scaler_id":ad.scaler.scaler_id,"roles_id":ad.roles_id}
    missing=dict(models); missing.pop(next(iter(missing)))
    passed.append(rejects(lambda:gate.freeze_models(missing),"missing global model manifest ID"))
    wrong=json.loads(json.dumps(models)); wrong[next(iter(wrong))]["model_sha256"]="z"*64
    passed.append(rejects(lambda:gate.freeze_models(wrong),"invalid model SHA"))
    wrong=json.loads(json.dumps(models)); wrong[next(iter(wrong))]["scaler_id"]=b.scaler.scaler_id
    passed.append(rejects(lambda:gate.freeze_models(wrong),"cross-state manifest scaler binding"))
    check(gate.freeze_models(models)["actual_files_verified"] is False,"structural-only model manifest")
    first_mid=gate.model_ids(state)[0]
    returned=gate.model_entry(state,first_mid); returned["state"]["state_uid"]="mutated_return_copy"
    check(gate.model_entry(state,first_mid)["state"]==state.as_dict(),"model_entry returns deep copy")
    passed.append(rejects(lambda:a.calibrate(access(state,"C_cal","alpha"),gate.model_ids(b.state)[0],np.zeros((4,60))),
                          "foreign state frozen model denied"))
    passed.append(rejects(lambda:a.labels(access(state,"T","labels"),f["roles"]["T"]),"T before global alpha freeze"))
    alpha_records={}
    for ad in adapters:
        role=ad.roles["C_cal"]; truth=ad.labels(access(ad.state,"C_cal","labels"),role)
        for mid in gate.model_ids(ad.state):
            record=ad.calibrate(access(ad.state,"C_cal","alpha"),mid,2*truth)
            same(record["alpha"],.5,"per-model alpha half")
            same(apply_alpha(record,ad.state,mid,2*truth),truth,"alpha whole raw path")
            alpha_records[mid]=record
    missing=dict(alpha_records); missing.pop(next(iter(missing)))
    passed.append(rejects(lambda:gate.freeze_alphas(missing),"missing global alpha ID"))
    check(gate.freeze_alphas(alpha_records)["actual_files_verified"] is False,"structural-only alpha manifest")
    check(a.network_inputs(access(state,"T","inputs"),f["roles"]["T"])["x"].shape==(4,480,1),"T released after global freeze")
    check(a.labels(access(state,"T","labels"),f["roles"]["T"]).shape==(4,60),"T labels released")
    p=np.ones((3,8),dtype=np.float64)
    same(endpoint_alpha(np.zeros_like(p),p),0.,"alpha exact denominator0")
    same(endpoint_alpha(p,2*p),1.,"alpha upper clip")
    same(endpoint_alpha(p,-p),0.,"alpha lower clip")
    passed.append("exact eight-ID model/scaler/alpha global barriers and state binding")
    passed.append("alpha exactzero/lower/upper clipping/wholepath")
    return {"status":"artificial_adapter_tests_complete","contract_sha256":CONTRACT_SHA256,
            "checks":passed,"rtol":RTOL,"atol":ATOL,"explicit_data_or_manifest_file_IO":False,
            "actual_manifest_files_verified":False,"market_arrays_read":False,"training_admitted":False,
            "raw_future_array_presence_implies_global_blindness":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artificial-tests",action="store_true",required=True)
    parser.parse_args()
    check(sorted(os.sched_getaffinity(0))==[24],"CPU24 required")
    for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        check(os.environ.get(name)=="1",name+"=1 required")
    print(json.dumps(run_tests(),ensure_ascii=False,allow_nan=False))


if __name__=="__main__":main()
