"""Independent same-state reconstruction of the frozen temporal readout probe.

Original forwards + head pre-hooks, independently accumulated prefix scalers,
direct original-X SVD, and plain metrics. Never imports the primary worker, LS
kernel or old engines. No new training or predictor serialization is performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import time

BASE = Path(__file__).resolve().parents[1]
STOP = False


def stop(*_):
    global STOP
    STOP = True


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic(path, value):
    with tempfile.NamedTemporaryFile("w", dir=Path(path).parent, delete=False) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)


def boundary(deadline):
    if STOP or time.time() >= deadline:
        raise InterruptedError("Original absolute stage deadline or explicit stop reached")


def verify(files, deadline):
    for path, expected in files.items():
        boundary(deadline)
        if sha(path) != expected:
            raise ValueError("Frozen bytes changed: "+path)


def array_sha(np, values):
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def difference(left, right):
    return abs(float(left)-float(right))/max(1.0, abs(float(left)), abs(float(right)))


def independent_scaler(np, windows, update, h, targets, deadline):
    # Independent implementation of the frozen 4096-origin accumulation order;
    # no fit_scaler call and no use of any full-fit/old checkpoint scaler.
    sx = sx2 = sa = sa2 = 0.0
    sy, sy2 = np.zeros(h), np.zeros(h)
    for start in range(0, len(update), 4096):
        boundary(deadline)
        origins = update[start:start+4096]
        x, a = windows.arrays(origins)
        y = targets(windows.bars, origins, h, "future_log_return")
        sx += float(np.sum(x, dtype=np.float64)); sx2 += float(np.sum(x*x, dtype=np.float64))
        sa += float(np.sum(a, dtype=np.float64)); sa2 += float(np.sum(a*a, dtype=np.float64))
        sy += np.sum(y, axis=0, dtype=np.float64); sy2 += np.sum(y*y, axis=0, dtype=np.float64)
    n = len(update)
    xm, am, ym = sx/(n*windows.lookback), sa/n, sy/n
    xs = float(np.sqrt(max(sx2/(n*windows.lookback)-xm*xm, 0)))
    aas = float(np.sqrt(max(sa2/n-am*am, 0)))
    ys = np.sqrt(np.maximum(sy2/n-ym*ym, 0))
    return {"x_mean": float(xm), "x_std": xs if xs > 1e-12 else 1.0,
            "anchor_mean": float(am), "anchor_std": aas if aas > 1e-12 else 1.0,
            "y_mean": ym.tolist(), "y_std": np.where(ys > 1e-12, ys, 1.0).tolist(),
            "fitted_origin_count": n}


def assemble(np, torch, trained, initialized, windows, bars, origins, scaler, h,
             targets, deadline, progress):
    x0, xd = np.empty((len(origins), 34)), np.empty((len(origins), 34))
    y, native, raw = (np.empty((len(origins), h)) for _ in range(3))
    ym, ys = np.asarray(scaler["y_mean"]), np.asarray(scaler["y_std"])
    captures = {"trained": [], "initial": []}

    def hook(name):
        def capture(_module, args):
            if len(args) != 1 or args[0].ndim != 2 or args[0].shape[1] != 33:
                raise ValueError("Original head input shape changed")
            captures[name].append(args[0].detach().numpy().astype(np.float64, copy=True))
        return capture

    handles = [trained.head.register_forward_pre_hook(hook("trained")),
               initialized.head.register_forward_pre_hook(hook("initial"))]
    try:
        for start in range(0, len(origins), 128):
            boundary(deadline)
            end = min(start+128, len(origins)); ids = origins[start:end]
            xx, aa = windows.arrays(ids)
            xx = ((xx-scaler["x_mean"])/scaler["x_std"]).astype(np.float32)
            aa = ((aa-scaler["anchor_mean"])/scaler["anchor_std"]).astype(np.float32)
            raw[start:end] = targets(bars, ids, h, "future_log_return")
            y[start:end] = ((raw[start:end]-ym)/ys).astype(np.float32).astype(np.float64)
            captures["trained"].clear(); captures["initial"].clear()
            with torch.inference_mode():
                native[start:end] = trained(torch.from_numpy(xx[:, :, None]), torch.from_numpy(aa)).numpy().astype(np.float64)
                initialized(torch.from_numpy(xx[:, :, None]), torch.from_numpy(aa))
            if len(captures["trained"]) != 1 or len(captures["initial"]) != 1:
                raise ValueError("Original forward head invocation count changed")
            x0[start:end, :33] = captures["trained"][0]
            xd[start:end, :33] = captures["initial"][0]
            x0[start:end, 33] = xd[start:end, 33] = 1.0
            if not np.array_equal(x0[start:end, 32], xd[start:end, 32]):
                raise ValueError("Trained and initial heads do not use the same observed anchor")
            progress(end, len(origins))
    finally:
        for handle in handles:
            handle.remove()
    return x0, xd, y, native, raw


def direct_svd(np, x, y, deadline):
    boundary(deadline)
    n, p = x.shape
    u, s, vh = np.linalg.svd(x, full_matrices=False)
    boundary(deadline)
    tau = float(np.finfo(float).eps*max(n, p)*s[0])
    retained = s > tau
    band = (s >= 0.1*tau) & (s <= 10*tau)
    rank = int(np.count_nonzero(retained))
    coefficient = vh[retained].T@((u[:, retained].T@y)/s[retained, None])
    residual = x@coefficient-y
    reconstruction = float(np.linalg.norm(x-(u*s)@vh, ord="fro"))
    orthogonality = max(float(np.linalg.norm(u.T@u-np.eye(p), ord="fro")),
                        float(np.linalg.norm(vh@vh.T-np.eye(p), ord="fro")))
    stationarity = float(np.linalg.norm(x.T@residual, ord="fro")/max(
        1.0, float(s[0])*float(np.linalg.norm(y, ord="fro")),
        float(s[0])*float(np.linalg.norm(residual, ord="fro"))))
    gamma_p = p*np.finfo(float).eps/(1-p*np.finfo(float).eps)
    # Independent direct-SVD stability diagnostic; no first-order prediction
    # margin is imported or recomputed from the primary kernel.
    estimated_delta_x = reconstruction+float(s[0])*orthogonality+32*gamma_p*float(np.linalg.norm(x, ord="fro"))
    eta = estimated_delta_x/float(s[retained][-1]) if rank else 1.0
    gates = {"full_rank": rank == p, "noambiguity": not bool(np.any(band)),
             "direct_reconstruction": reconstruction/max(1.0,float(np.linalg.norm(x,ord="fro"))) <= 1e-10,
             "direct_orthogonality": orthogonality <= 1e-10,
             "stationarity": stationarity <= 1e-10, "perturbation_stable": eta < 0.1}
    gates = {key: bool(value) for key, value in gates.items()}
    return coefficient, {"rank": rank, "rank_ambiguous": bool(np.any(band)),
        "ambiguous_singular_indices": np.flatnonzero(band).tolist(), "singular_values": s.tolist(),
        "solver_qualified": all(gates.values()), "direct_numerical_gates": gates,
        "rank_cutoff": tau, "stationarity_normalized": stationarity,
        "direct_reconstruction_Frobenius": reconstruction, "direct_orthogonality": orthogonality,
        "direct_estimated_delta_X_to_sigma_min": eta,
        "residual_MSE": float(np.mean(residual*residual)),
        "primary_first_order_margin_called": False}


def replay(np, torch, source, primary, saved_path, partition, protocol, stage_inputs,
           inputs, deadline, progress):
    from multiscale.bars import Bars, targets
    from multiscale.features import CloseWindows
    from multiscale.models import make_model

    cfg, h = source["cfg"], source["cfg"]["horizon"]
    with np.load(source["split_file"], allow_pickle=False) as archive:
        fit = archive["fit"]
    if len(fit) != source["fit_count"] or array_sha(np, fit) != source["fit_array_sha256"]:
        raise ValueError("Original fit origins changed")
    with np.load(source["source_bars"], allow_pickle=False) as archive:
        if set(archive.files) != {"start_ns", "end_ns", "close", "observed", "session_date"}:
            raise ValueError("Unexpected admitted source bar fields")
        arrays = {name: archive[name] for name in archive.files}
    if not np.all(arrays["session_date"] <= "2026-06-30"):
        raise ValueError("Q3 source boundary failed")
    cutoff = np.datetime64("2026-07-01T04:00:00", "ns").astype("int64")
    if not np.all(arrays["end_ns"][arrays["observed"]] < cutoff):
        raise ValueError("Q3 timestamp source boundary failed")
    bars = Bars(cfg["frequency"], **arrays)
    cut = (4*len(fit))//5
    tail = fit[cut:]
    before_purge = fit[:cut]
    update = before_purge[bars.end_ns[before_purge+h] < bars.end_ns[tail[0]]]
    if len(update)<128 or len(tail)<32 or np.intersect1d(update,tail).size:
        raise ValueError("Temporal split eligibility failed")
    binding = primary["binding"]
    row_checks = {"original_fit": array_sha(np,fit)==binding["original_fit_sha256"],
        "update": array_sha(np,update)==binding["update_sha256"]==partition["update_sha256"],
        "tail": array_sha(np,tail)==binding["tail_sha256"]==partition["tail_sha256"],
        "counts": len(update)==binding["update_count"]==partition["update_count"] and len(tail)==binding["tail_count"]==partition["tail_count"],
        "purge": len(before_purge)-len(update)==binding["purged_count"] and int(bars.end_ns[update[-1]+h])==partition["last_update_label_end_ns"]<partition["first_tail_decision_ns"]==int(bars.end_ns[tail[0]])}
    if not all(row_checks.values()):
        raise ValueError("Independent chronological split/purge differs")
    windows = CloseWindows(bars,cfg["lookback"],load(inputs["source_admission_path"])["split_actions"])
    scaler = independent_scaler(np,windows,update,h,targets,deadline)
    if scaler != binding["scaler"]:
        raise ValueError("Independent update-only scaler differs from saved exact scaler")
    boundary(deadline)
    saved = torch.load(saved_path,map_location="cpu",weights_only=True)
    if saved["identity"] != primary["identity"] or saved["binding"] != binding:
        raise ValueError("New temporal checkpoint identity changed")
    if saved["state"]["epoch"] != source["fixed_epochs"] or saved["state"]["phase"] not in ("completed", "completed_nonqualified"):
        raise ValueError("Checkpoint is not the committed fixed-epoch final state")
    torch.manual_seed(cfg["seed"])
    trained=make_model("lstm",cfg["lookback"],h)
    expected_initial={key:value.detach().clone() for key,value in trained.state_dict().items()}
    expected_initial["head.weight"].zero_()
    expected_initial["head.bias"].copy_(torch.as_tensor(-np.asarray(scaler["y_mean"])/np.asarray(scaler["y_std"]),dtype=torch.float32))
    initial_matches=(set(expected_initial)==set(saved["initial_model"]) and
        all(torch.equal(expected_initial[key],saved["initial_model"][key]) for key in expected_initial))
    if not initial_matches:
        raise ValueError("Saved initial clone does not match same-seeded prefix-only initialization")
    trained.load_state_dict(saved["model"],strict=True)
    initialized=make_model("lstm",cfg["lookback"],h)
    initialized.load_state_dict(saved["initial_model"],strict=True)
    for model in (trained,initialized):
        model.eval()
        for param in model.parameters():param.requires_grad_(False)
    ux,ud,uy,ua,_ = assemble(np,torch,trained,initialized,windows,bars,update,scaler,h,
                            targets,deadline,lambda cursor,n:progress("update",cursor,n))
    designs={"B":ux,"C":ux[:,32:],"D":ud}
    fits, direct, numerical_checks = {}, {}, {}
    for arm in ("B","C","D"):
        fits[arm],direct[arm]=direct_svd(np,designs[arm],uy,deadline)
        previous=primary["readouts"][arm]
        left,right=np.asarray(direct[arm]["singular_values"]),np.asarray(previous["singular_values"])
        singular_difference=float(np.linalg.norm(left-right)/max(1.0,float(np.linalg.norm(left)),float(np.linalg.norm(right))))
        exact={key:direct[arm][key]==previous[key] for key in ("rank","rank_ambiguous","ambiguous_singular_indices","solver_qualified")}
        residual_difference=difference(direct[arm]["residual_MSE"],previous["residual_MSE"])
        numerical_checks[arm]={"normalized_singular_value_difference":singular_difference,
            "normalized_update_MSE_difference":residual_difference,"exact_qualification_comparisons":exact,
            "passed":singular_difference<=protocol["singular_normalized_tolerance"] and residual_difference<=protocol["metric_normalized_tolerance"] and all(exact.values())}
    update_metrics={"A":float(np.mean((ua-uy)**2)),"zero":float(np.mean((-np.asarray(scaler["y_mean"])/np.asarray(scaler["y_std"])-uy)**2)),"mean":float(np.mean(uy**2)),**{arm:value["residual_MSE"] for arm,value in direct.items()}}
    result={"state_id":source["id"],"row_hash_and_purge_checks":row_checks,"exact_fresh_scaler_matches":True,
            "same_seed_initial_clone_exact":True,"new_fixed_epochs":source["fixed_epochs"],
            "update_count":len(update),"tail_count":len(tail),"direct_readouts":direct,
            "readout_comparisons":numerical_checks,"direct_update_metrics":update_metrics,
            "tail_replayed":False,"only_original_fit_used":True,"whole_preQ3_source_loaded":True,
            "original_stop_outer_or_Q3_scored":False,"temporary_coefficients_saved":False}
    if not primary["tail_scored"]:
        result["status"]="prefix_reconstruction_only_primary_nonqualification_tail_skipped"
        result["passed"]=all(item["passed"] for item in numerical_checks.values()) and not all(item["solver_qualified"] for item in direct.values())
        return result
    if not all(item["passed"] and direct[arm]["solver_qualified"] for arm,item in numerical_checks.items()):
        result["status"]="independent_prefix_qualification_disagrees_tail_not_replayed"
        result["passed"]=False
        return result
    tx,td,ty,ta,tr=assemble(np,torch,trained,initialized,windows,bars,tail,scaler,h,
                           targets,deadline,lambda cursor,n:progress("tail",cursor,n))
    prediction={"A":ta,"B":tx@fits["B"],"C":tx[:,32:]@fits["C"],"D":td@fits["D"],
                "zero":np.broadcast_to(-np.asarray(scaler["y_mean"])/np.asarray(scaler["y_std"]),ty.shape),"mean":np.zeros_like(ty)}
    tail_metrics={};metric_comparisons={}
    for arm,pred in prediction.items():
        raw=pred*np.asarray(scaler["y_std"])+np.asarray(scaler["y_mean"])
        metrics={"path_standardized_MSE":float(np.mean((pred-ty)**2)),
                 "endpoint_raw_MSE":float(np.mean((raw[:,-1]-tr[:,-1])**2))}
        tail_metrics[arm]=metrics
        metric_comparisons[arm]={}
        for metric,value in metrics.items():
            expected=primary["tail_metrics"][arm][metric]
            which="path" if metric=="path_standardized_MSE" else "endpoint"
            margin=primary["tail_loss_numerical_sensitivity"][arm][which]["estimated_MSE_error_margin"]
            error=difference(value,expected)
            metric_comparisons[arm][metric]={"normalized_difference":error,
                "absolute_difference":abs(value-expected),
                "relative_to_primary_metric_difference":abs(value-expected)/max(np.finfo(float).tiny,abs(expected)),
                "primary_estimated_MSE_margin_disclosed_not_used_as_comparison_tolerance":margin,
                "passed":error<=protocol["metric_normalized_tolerance"]}
    primary_update=primary["update_path_standardized_MSE"]
    update_differences={arm:difference(value,primary_update[arm]) for arm,value in update_metrics.items()}
    result.update({"status":"completed_independent_same_state_temporal_reconstruction","tail_replayed":True,
        "direct_tail_metrics":tail_metrics,"metric_comparisons":metric_comparisons,
        "normalized_update_metric_differences":update_differences})
    result["passed"]=(all(item["passed"] for item in numerical_checks.values()) and
        all(value<=protocol["metric_normalized_tolerance"] for value in update_differences.values()) and
        all(item["passed"] for group in metric_comparisons.values() for item in group.values()))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,default=BASE/"INDEPENDENT_TEMPORAL_REPLAY.json")
    args=parser.parse_args();destination=args.report.resolve();destination.relative_to(BASE)
    if set(os.sched_getaffinity(0))!={26}:raise ValueError("CPU26 required")
    for key,want in {"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","CUDA_VISIBLE_DEVICES":"-1"}.items():
        if os.environ.get(key)!=want:raise ValueError("Frozen execution environment changed: "+key)
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGUSR1):signal.signal(sig,stop)
    protocol_path=BASE/"INDEPENDENT_REPLAY_PROTOCOL.json";protocol=load(protocol_path)
    if sha(Path(__file__))!=protocol["code_sha256"]:raise ValueError("Independent source changed")
    for name,expected in protocol["metadata_sha256"].items():
        if sha(BASE/name)!=expected:raise ValueError("Bound stage metadata changed: "+name)
    scope=load(BASE/"STAGE_SCOPE.json");deadline=scope["deadline_epoch"]
    if deadline!=protocol["deadline_epoch"]:raise ValueError("Deadline changed")
    boundary(deadline)
    admission=load(BASE/"INDEPENDENT_EXECUTION_ADMISSION.json")
    if admission["status"]!="admitted_independent_temporal_replay_after_four_primary_commits":raise ValueError("Independent execution not admitted")
    wanted={"code_sha256":sha(Path(__file__)),"protocol_sha256":sha(protocol_path),
        "code_freeze_sha256":sha(BASE/"CODE_FREEZE.json"),"scope_sha256":sha(BASE/"STAGE_SCOPE.json"),
        "stage_input_identities_sha256":sha(BASE/"STAGE_INPUT_IDENTITIES.json"),
        "deadline_epoch":deadline,"state_ids":protocol["state_ids"]}
    if any(admission[key]!=value for key,value in wanted.items()):raise ValueError("Execution admission binding changed")
    market=load(BASE/"MARKET_ADMISSION.json")
    if (market["status"]!="admitted_four_fixed_temporal_development_states" or
        market["code_freeze_sha256"]!=wanted["code_freeze_sha256"] or market["scope_sha256"]!=wanted["scope_sha256"]):
        raise ValueError("Primary market admission absent or changed")
    stage_inputs=load(BASE/"STAGE_INPUT_IDENTITIES.json")
    inputs=load(Path(scope["parent_directory"])/"INPUT_IDENTITIES.json")
    states=stage_inputs["states"]
    if [state["id"] for state in states]!=protocol["state_ids"]:raise ValueError("Four-state list changed")
    primaries={};sources={};source_hashes={}
    for source in states:
        directory=BASE/"runs"/source["id"]
        primary=load(directory/"RESULT.json");commit=load(directory/"RESULT_COMMIT.json")
        manifest=load(directory/"CHECKPOINT.json");payload=directory/manifest["payload_file"]
        if payload.parent!=directory or payload.suffix!=".pt":raise ValueError("Checkpoint payload escaped state directory")
        for path in (directory/"RESULT.json",directory/"RESULT_COMMIT.json",directory/"CHECKPOINT.json",payload):
            if str(path) not in admission["verified_files"]:raise ValueError("Admission does not bind every final source")
            source_hashes[str(path)]=admission["verified_files"][str(path)]
        if (commit["result_sha256"]!=sha(directory/"RESULT.json") or commit["identity"]!=primary["identity"]
            or manifest["identity"]!=primary["identity"] or manifest["payload_sha256"]!=admission["verified_files"][str(payload)]
            or primary["identity"]!=digest(primary["binding"]) or commit["stage_deadline_epoch"]!=deadline
            or manifest["stage_deadline_epoch"]!=deadline or manifest["epoch"]!=source["fixed_epochs"]
            or primary["status"] not in ("completed_fixed_temporal_diagnostic","completed_numerical_nonqualification")
            or primary["binding"]["state_id"]!=source["id"] or primary["binding"]["cfg"]!=source["cfg"]
            or primary["binding"]["scope_sha256"]!=wanted["scope_sha256"]
            or primary["binding"]["code_freeze_sha256"]!=wanted["code_freeze_sha256"]
            or primary["binding"]["stage_input_identities_sha256"]!=wanted["stage_input_identities_sha256"]
            or primary["binding"]["stage_deadline_epoch"]!=deadline):
            raise ValueError("Every fixed-epoch primary must have a valid final commit before replay")
        primaries[source["id"]]=primary;sources[source["id"]]=payload
    verify(source_hashes,deadline)
    verify(load(BASE/"CODE_FREEZE.json")["files"],deadline)
    verify(stage_inputs["verified_files"],deadline)
    verify(stage_inputs["inherited_data_container_identities_requiring_recheck_before_open"],deadline)
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    sys.path.insert(0,stage_inputs["reference_root"])
    from multiscale import bars,features,models
    for module in (bars,features,models):Path(module.__file__).resolve().relative_to(Path(stage_inputs["reference_root"]).resolve())
    binding={**wanted,"protocol_sha256":sha(protocol_path),"execution_admission_sha256":sha(BASE/"INDEPENDENT_EXECUTION_ADMISSION.json"),"primary_source_sha256":source_hashes}
    rows=[]
    if destination.exists():
        old=load(destination)
        if old["binding"]!=binding:raise ValueError("Resume binding changed")
        rows=old["states"]
        if [row["state_id"] for row in rows]!=protocol["state_ids"][:len(rows)]:raise ValueError("Completed state cursor changed")
    report={"status":"running","binding":binding,"states":rows,"started_at_UTC":datetime.now(timezone.utc).isoformat(),
        "actual_python_executable":sys.executable,"CPU_affinity":[26],"NumPy_version":np.__version__,
        "GPU":False,"new_training":False,"new_predictor_saved":False,"stop_outer_Q3_scored":False,
        "independent_market_confirmation":False,"research_goal_achieved":False}
    def progress(state_id,part,cursor,total):
        report.update(current_state=state_id,current_partition=part,transient_cursor=cursor,partition_rows=total,
                      partial_in_memory_arrays_reassembled_from_zero_on_resume=True)
        atomic(destination,report)
    atomic(destination,report)
    try:
        for source in states[len(rows):]:
            boundary(deadline);state_id=source["id"];started=time.perf_counter()
            row=replay(np,torch,source,primaries[state_id],sources[state_id],
                load(BASE/"runs"/state_id/"PARTITION.json"),protocol,stage_inputs,inputs,deadline,
                lambda part,cursor,total:progress(state_id,part,cursor,total))
            row["wall_seconds"]=time.perf_counter()-started
            rows.append(row);report["states"]=rows;atomic(destination,report)
        report["status"]="completed_all_four_reconstructions_passed" if all(row["passed"] for row in rows) else "completed_reconstruction_disagreement_retained"
    except InterruptedError as error:
        report["status"]="partial_stop_completed_scalar_states_preserved";report["stop_reason"]=str(error)
    except Exception as error:
        report["status"]="error_completed_scalar_states_preserved";report["error_type"]=type(error).__name__;report["error"]=str(error)
        atomic(destination,report);raise
    report["finished_at_UTC"]=datetime.now(timezone.utc).isoformat();atomic(destination,report)
    print(json.dumps({"status":report["status"],"completed_states":len(rows),"report":str(destination)},indent=2))
    return 0 if report["status"]=="completed_all_four_reconstructions_passed" else 2


if __name__=="__main__":raise SystemExit(main())
