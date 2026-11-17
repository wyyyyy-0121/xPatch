"""Post-hoc, local price-anchor experiment on the already exposed folds."""
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from anchored_price import AnchoredPrice, fit, predict
from multiscale.bars import file_digest
from multiscale.development import atomic_json, validate_split
from multiscale.features import CloseWindows, fit_scaler
from multiscale.matrix_executor import read_bars
from multiscale.spec import OUT, PROFILES, ROOT
from multiscale.splits import rolling_folds, warmup_folds
from multiscale.training import TrainConfig, digest_indices


NEURAL = ("mlp", "gru", "lstm", "xpatch")


def write_new(path, value):
    with path.open("x",encoding="utf8") as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write("\n")


def main():
    start = time.monotonic()
    previous = OUT / "private_three_scale_development/20261107T134128870773Z"
    prior_plan_path = previous / "PRIVATE_PROTOCOL.json"
    prior_plan = json.loads(prior_plan_path.read_text(encoding="utf8"))
    prior_state = json.loads((previous / "PRIVATE_QUEUE_STATE.json").read_text(encoding="utf8"))
    if prior_state.get("status") != "completed_private_development_only" or prior_state.get("completed") != 50:
        raise ValueError("The paired development matrix is not complete")
    source = OUT / "admission/alpaca_20261107T052228573242Z"
    admission_path = source / "admission.json"
    admission = json.loads(admission_path.read_text(encoding="utf8"))
    if file_digest(admission_path) != prior_plan["admission_sha256"] or admission.get("Q3_values_read") is not False:
        raise ValueError("Candidate identity or cutoff status differs")
    jobs = [job for job in prior_plan["jobs"] if job["target"] == "future_close"]
    if len(jobs) != 25:
        raise ValueError("Expected exactly five complete price tasks")
    profile_map = {p.name:p for p in PROFILES}; cached = {}
    for job in jobs:
        key = (job["profile"],job["horizon"])
        if key not in cached:
            profile = profile_map[job["profile"]]
            bars = read_bars(source,admission,profile.frequency)
            path = source / f"origins_{job['profile']}_h{job['horizon']}.npy"
            if file_digest(path) != admission["files"][path.name]:raise ValueError("Origin identity changed")
            origins = np.load(path,allow_pickle=False)
            if job["profile"] == "quarterly":
                folds = warmup_folds(bars,origins,job["horizon"],min_inner_fit=24,min_inner_stop=4,
                                     min_validation_groups=6)
            else:folds = rolling_folds(bars,origins,job["horizon"])
            if len(folds) != 5:raise ValueError("Cannot recreate the five folds")
            cached[key] = (bars,folds)
        bars,folds = cached[key]
        train,fit_indices,stop,validation = validate_split(bars,folds[job["fold"]-1],job["horizon"])
        for name,indices in (("train",train),("fit",fit_indices),("stop",stop),("validation",validation)):
            if digest_indices(indices) != job[name+"_indices_sha256"]:
                raise ValueError("Paired split differs from the prior protocol")
        old = Path(prior_state["jobs"][job["key"]]["fold_directory"]).resolve()
        if not old.is_relative_to(OUT.resolve()) or file_digest(old/"fold_receipt.json") != prior_state["jobs"][job["key"]]["receipt_sha256"]:
            raise ValueError("Prior fold identity differs")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    parent = OUT / "private_anchored_price_exploration";parent.mkdir(parents=True,exist_ok=True)
    directory = parent/stamp;directory.mkdir(exist_ok=False)
    exposure = OUT/"POST_CUTOFF_EXPOSURE.md";before=exposure.read_bytes();(directory/"EXPOSURE_BEFORE.md").write_bytes(before)
    helper = Path(__file__).parent/"anchored_price.py"
    source_files = ("multiscale/models.py","multiscale/features.py","multiscale/training.py",
                    "multiscale/bars.py","multiscale/splits.py","aapl/models.py","aapl/vendor/xpatch_network.py")
    plan = {"status":"private_posthoc_development_not_publication_admitted","created_at_utc":datetime.now(timezone.utc).isoformat(),
            "motivation":"After observing large direct-price errors, test a persistence-anchored residual head.",
            "prior_results_already_seen":True,"original_protocol_sha256":file_digest(prior_plan_path),
            "prior_aggregate_check_sha256":file_digest(previous/"PRIVATE_AGGREGATE_INDEPENDENT_CHECK.json"),
            "admission_sha256":file_digest(admission_path),"jobs":jobs,"models":list(NEURAL),
            "method":"standardized persistence price plus a trainable zero-initialized H-by-H residual head",
            "stopping":"same inner stop endpoint MSE; epoch-zero persistence is also eligible",
            "training":{"max_epochs":20,"patience":3,"batch_size":128,"learning_rate":0.001,"seed":2026},
            "device":"cpu","cpu_threads":2,"maximum_wall_seconds":900,
            "script_sha256":file_digest(Path(__file__)),"helper_sha256":file_digest(helper),
            "source_sha256":{name:file_digest(ROOT/name) for name in source_files},
            "exposure_before_sha256":hashlib.sha256(before).hexdigest(),
            "data_cutoff":"2026-06-30","Q3_values_read":False,"independent_test":False,
            "publication_rights":"not confirmed; do not publish results",
            "limitations":["Post-hoc method choice; all folds are already exposed development data.",
                           "Extra head and epoch-zero stopping change the recipe and capacity.",
                           "One seed; no independent test or trading-cost evaluation.",
                           "Model checkpoints are saved after each fit; incomplete fits are not automatically resumed."]}
    plan_path=directory/"PRIVATE_PROTOCOL.json";write_new(plan_path,plan);protocol_hash=file_digest(plan_path)
    with exposure.open("a",encoding="utf8") as f:
        f.write("\n## 个人内部价格锚点残差：事后实验登记\n\n"
                f"{stamp}：已看到直接价格与收益任务结果后，预先固定当前价格加可训练残差的后续探索，"
                "覆盖小时 H1/H4/H8、月度 H21、季度 H3 的四类神经模型及其五折，"
                "使用同一起点与拟合/早停划分，并将零更新持续性解列为早停候选。"
                "这是事后开发，不能视为预注册或独立确认；账号具体协议和发表权待核实，2026Q3 封存。"
                f"设置和身份见 {plan_path}。\n")
    state={"status":"running","protocol_sha256":protocol_hash,"completed":0,"total":len(jobs),"jobs":{}}
    state_path=directory/"PRIVATE_QUEUE_STATE.json";atomic_json(state_path,state)
    print(json.dumps({"status":"private_posthoc_anchor_queue_started","directory":str(directory),
                      "folds":len(jobs),"models_per_fold":4,"cpu_threads":2,"maximum_seconds":900},ensure_ascii=False),flush=True)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    completed = {};recipe=plan["training"]
    for job in jobs:
        key=job["key"];state["jobs"][key]={"status":"running"};atomic_json(state_path,state)
        bars,folds=cached[(job["profile"],job["horizon"])];fold=folds[job["fold"]-1]
        _,fit_indices,stop_indices,validation=validate_split(bars,fold,job["horizon"])
        windows=CloseWindows(bars,job["lookback"],admission["split_actions"])
        scaler=fit_scaler(windows,fit_indices,job["horizon"],"future_close")
        old=Path(prior_state["jobs"][key]["fold_directory"])
        with np.load(old/"targets.npz",allow_pickle=False) as p:
            if not np.array_equal(p["origins"],validation) or not np.array_equal(p["decision_ns"],bars.end_ns[validation]):
                raise ValueError("Paired validation identities differ")
            anchor=p["anchor"].copy();decisions=p["decision_ns"].copy()
        fold_dir=directory/key;fold_dir.mkdir(exist_ok=False)
        results={};predictions={}
        for name in NEURAL:
            cfg=TrainConfig(name,job["lookback"],job["horizon"],"future_close",recipe["max_epochs"],
                            recipe["patience"],recipe["batch_size"],recipe["learning_rate"],recipe["seed"])
            model,record=fit(name,windows,scaler,fit_indices,stop_indices,cfg)
            prediction=predict(model,windows,scaler,validation,batch_size=recipe["batch_size"])
            if prediction.shape != (len(validation),job["horizon"]):raise ValueError("Invalid predicted path shape")
            model_path=fold_dir/(name+".pt")
            torch.save({"state_dict":model.state_dict(),"config":asdict(cfg),"protocol_sha256":protocol_hash},model_path)
            check=torch.load(model_path,map_location="cpu",weights_only=True)
            restored=AnchoredPrice(name,job["lookback"],job["horizon"],scaler);restored.load_state_dict(check["state_dict"])
            repeated=predict(restored,windows,scaler,validation[:min(32,len(validation))])
            if not np.allclose(repeated,prediction[:len(repeated)],rtol=1e-7,atol=1e-7):
                raise ValueError("Restored model predictions differ")
            pred_path=fold_dir/(name+".npz")
            np.savez_compressed(pred_path,prediction=prediction,origins=validation,decision_ns=decisions)
            record.update(model_sha256=file_digest(model_path),prediction_sha256=file_digest(pred_path),restore_check=True)
            results[name]=record;predictions[name]=prediction
        # Outer prices are read only after all four model fits and stopping choices for the fold.
        with np.load(old/"targets.npz",allow_pickle=False) as p:actual=np.asarray(p["actual"],dtype=np.float64)
        with np.load(old/"prediction_last_close.npz",allow_pickle=False) as p:baseline=np.asarray(p["prediction"],dtype=np.float64)
        base_sse=float(np.square(baseline[:,-1]-actual[:,-1]).sum()); n=len(actual)
        for name in NEURAL:
            with np.load(old/f"prediction_{name}.npz",allow_pickle=False) as p:direct=np.asarray(p["prediction"],dtype=np.float64)
            new=predictions[name]
            results[name].update(endpoint_mse=float(np.square(new[:,-1]-actual[:,-1]).mean()),
                                 path_mse=float(np.square(new-actual).mean()),
                                 direct_endpoint_mse=float(np.square(direct[:,-1]-actual[:,-1]).mean()),
                                 relative_improvement=(1-float(np.square(new[:,-1]-actual[:,-1]).sum())/base_sse) if base_sse else None)
        fold_result={"status":"private_posthoc_fold_completed","job":job,"validation_origins":n,
                     "baseline_endpoint_mse":base_sse/n,"models":results,"independent_test":False}
        fold_result_path=fold_dir/"PRIVATE_FOLD_RESULT.json";write_new(fold_result_path,fold_result)
        completed[key]=fold_result;state["completed"]+=1;state["elapsed_seconds"]=time.monotonic()-start
        state["jobs"][key]={"status":"completed","fold_result_sha256":file_digest(fold_result_path)};atomic_json(state_path,state)
        if job["fold"]==5:
            group=[j for j in jobs if (j["profile"],j["horizon"])==(job["profile"],job["horizon"])]
            total=sum(completed[j["key"]]["validation_origins"] for j in group)
            baseline=sum(completed[j["key"]]["baseline_endpoint_mse"]*completed[j["key"]]["validation_origins"] for j in group)/total
            metrics={}
            for name in NEURAL:
                pooled=sum(completed[j["key"]]["models"][name]["endpoint_mse"]*completed[j["key"]]["validation_origins"] for j in group)/total
                direct=sum(completed[j["key"]]["models"][name]["direct_endpoint_mse"]*completed[j["key"]]["validation_origins"] for j in group)/total
                metrics[name]={"anchored_endpoint_mse":pooled,"direct_endpoint_mse":direct,
                               "anchored_improvement_vs_persistence":1-pooled/baseline if baseline else None,
                               "direct_improvement_vs_persistence":1-direct/baseline if baseline else None}
            task={"status":"private_posthoc_five_fold_comparison", "profile":job["profile"],"horizon":job["horizon"],
                  "validation_origins":total,"baseline_endpoint_mse":baseline,"models":metrics,"independent_test":False}
            task_path=directory/f"PRIVATE_TASK_{job['profile']}_h{job['horizon']}.json";write_new(task_path,task)
            print(json.dumps({"status":"private_anchor_task_complete","profile":job["profile"],"horizon":job["horizon"],
                              "completed_folds":state["completed"],"total_folds":len(jobs),"task_result":str(task_path)},ensure_ascii=False),flush=True)
    state["status"]="completed_private_posthoc_only";state["elapsed_seconds"]=time.monotonic()-start;atomic_json(state_path,state)
    rows=[]
    for key in dict.fromkeys((job["profile"],job["horizon"]) for job in jobs):
        task=json.loads((directory/f"PRIVATE_TASK_{key[0]}_h{key[1]}.json").read_text())
        for name,v in task["models"].items():rows.append({"profile":key[0],"horizon":key[1],"model":name,
                                                         "folds":5,"validation_origins":task["validation_origins"],**v})
    with (directory/"PRIVATE_ANCHORED_RESULTS.csv").open("x",encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with exposure.open("a",encoding="utf8") as f:
        f.write(f"\n{stamp}：上述价格锚点事后探索已完成 {state['completed']} 折，结果保存在 {directory}；"
                "该开发结果不构成独立确认。\n")
    print(json.dumps({"status":state["status"],"directory":str(directory),"folds":state["completed"],
                      "elapsed_seconds":state["elapsed_seconds"]},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    try:main()
    except Exception as exc:
        print(json.dumps({"status":"private_anchor_queue_failed","error":f"{type(exc).__name__}: {exc}"},ensure_ascii=False),
              file=sys.stderr,flush=True);raise
