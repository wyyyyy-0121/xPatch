"""Core nested HPO/refit implementation. This module is not run during admission."""
import argparse
import gc
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from .common import ROOT, digest, identity, read_json, runtime, utc_now, write_json
from .data import Market
from .models import chronos_pipeline, chronos_predict, neural
from .training import Trainer, neural_predict, source_identity
from . import baselines
from .resources import ResourceMonitor


def score(y, pred):
    return float(np.mean((np.asarray(y) - np.asarray(pred)) ** 2))


def neural_job(config, seed, fit_dates, val_dates, budget, directory, market, lora=False, refit=False):
    directory = Path(directory)
    try:
        return _neural_job(config, seed, fit_dates, val_dates, budget, directory, market, lora, refit)
    except Exception as exc:
        write_json(directory / 'failure.json', {'failed_at_utc': utc_now(), 'error': repr(exc),
                   'config': config, 'seed': seed, 'requested_budget': budget,
                   'implementation_sha256': source_identity(), 'checkpoints_preserved': True})
        raise


def _neural_job(config, seed, fit_dates, val_dates, budget, directory, market, lora=False, refit=False):
    """A rung resumes latest; best selection checkpoint remains separate from latest."""
    device = runtime(seed)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "checkpoints_pruned.json").exists():
        cached = read_json(directory / "summary.json")
        pruned = read_json(directory / "checkpoints_pruned.json")
        if refit or cached["completed_budget"] != budget or pruned["implementation_sha256"] != source_identity():
            raise ValueError("Cannot resume a pruned/nonmatching training rung")
        return cached
    train, val = market.select(fit_dates), market.select(val_dates)
    scale = market.scale(fit_dates)
    if lora:
        pipe, receipt = chronos_pipeline(True, config["rank"], device)
        model = pipe.model
        predict = lambda origins: chronos_predict(pipe, market, origins, config)
    else:
        model = neural(config).to(device)
        predict = lambda origins: neural_predict(model, config, market, origins, scale, device)
    trainer = Trainer(model, config, market, train, scale, device, seed, 900 if lora else 27, lora=lora)
    latest = directory / "latest.pt"
    best_file = directory / "best.pt"
    summary_file = directory / "summary.json"
    summary = read_json(summary_file) if summary_file.exists() else {"best_score": None, "best_budget": None, "evaluations": []}
    if latest.exists():
        trainer.restore(latest)
    current = trainer.updates if lora else trainer.epoch
    if current > budget:
        raise ValueError("Cannot reverse a training rung")
    t = time.perf_counter()
    while current < budget:
        if lora:
            next_budget = min(budget, (current // 100 + 1) * 100)
            trainer.train_updates(next_budget)
            current = trainer.updates
        else:
            trainer.train_epoch()
            current = trainer.epoch
        trainer.save(latest)
        if not refit:
            value = score(market.y(val), predict(val))
            summary["evaluations"].append({"budget": current, "mse": value})
            if summary["best_score"] is None or value < summary["best_score"]:
                summary.update(best_score=value, best_budget=current)
                trainer.save(best_file)
        write_json(summary_file, {**summary, "completed_budget": current,
                                 "signature": trainer.signature, "fit_dates": fit_dates, "validation_dates": val_dates})
    if refit:
        trainer.save(best_file)
        summary["best_budget"] = budget
    else:
        trainer.restore(best_file)
    pred = predict(val) if len(val) else np.empty(0)
    if len(val):
        np.savez_compressed(directory / "inner_predictions.npz", origins=val, prediction=pred)
    summary.update(completed_budget=budget, signature=trainer.signature, wall_seconds=time.perf_counter() - t,
                   fit_dates=fit_dates, validation_dates=val_dates, config=config, seed=seed,
                   prediction_sha256=digest(directory / 'inner_predictions.npz') if len(val) else None)
    write_json(summary_file, summary)
    # Return model for immediate calibration/outer prediction only in final refit.
    if refit:
        return summary, predict, {'trainer': trainer, 'model': model, 'pipe': pipe if lora else None}
    del trainer, model
    if lora:
        del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def baseline_job(config, seed, fit_dates, val_dates, directory, market, fixed_trees=None):
    directory.mkdir(parents=True, exist_ok=True)
    cfg = {**config, "seed": seed}
    scale = market.scale(fit_dates)
    train, val = market.select(fit_dates), market.select(val_dates)
    signature = identity({'config': config, 'seed': seed, 'fit_dates': fit_dates,
                          'validation_dates': val_dates, 'fixed_trees': fixed_trees, 'source': source_identity()})
    if fixed_trees is not None:
        cfg["n_estimators"] = fixed_trees
    if (directory / 'summary.json').exists():
        old = read_json(directory / 'summary.json')
        model_file = directory / ('ridge.npz' if cfg['model'] == 'ridge' else 'lightgbm.txt')
        if old.get('signature') == signature and old.get('prediction_sha256') == digest(directory / 'inner_predictions.npz') and old.get('model_sha256') == digest(model_file):
            if cfg['model'] == 'ridge':
                saved = np.load(directory / 'ridge.npz')
                coef, intercept = saved['coef'].copy(), saved['intercept'].copy()
                saved.close()
                for_market = lambda data, origins: (baselines.design(data, origins, cfg, scale) @ coef + intercept) * scale['target_std']
            else:
                import lightgbm as lgb
                model = lgb.Booster(model_file=str(directory / 'lightgbm.txt'))
                for_market = lambda data, origins: model.predict(baselines.design(data, origins, cfg, scale), num_threads=int(os.environ.get('AAPL_CPU_THREADS', '4'))) * scale['target_std']
            predict = lambda origins: for_market(market, origins)
            predict.for_market = for_market
            return old, predict
    if cfg["model"] == "ridge":
        model = baselines.fit_ridge(market, train, cfg, scale)
        np.savez(directory / "ridge.npz", coef=model.coef_, intercept=model.intercept_)
        chosen = 1
    else:
        model = baselines.fit_lightgbm(market, train, cfg, scale, val if fixed_trees is None else None)
        model.booster_.save_model(str(directory / "lightgbm.txt"))
        chosen = model.best_iteration_ or cfg["n_estimators"]
    predict = lambda origins: baselines.predict(model, market, origins, cfg, scale)
    predict.for_market = lambda data, origins: baselines.predict(model, data, origins, cfg, scale)
    pred = predict(val)
    summary = {"best_score": score(market.y(val), pred) if fixed_trees is None else None, "best_budget": chosen,
               "fit_dates": fit_dates, "validation_dates": val_dates, "config": cfg, "seed": seed, 'signature': signature}
    np.savez_compressed(directory / "inner_predictions.npz", origins=val, prediction=pred)
    summary["prediction_sha256"] = digest(directory / "inner_predictions.npz")
    summary['model_sha256'] = digest(directory / ('ridge.npz' if cfg['model'] == 'ridge' else 'lightgbm.txt'))
    write_json(directory / "summary.json", summary)
    return summary, predict


def execute_spec(spec, market):
    cfg, seed = spec['config'], spec['seed']
    d = Path(spec['directory'])
    try:
        if cfg['model'] in ('xpatch', 'patchtst', 'chronos2_lora'):
            result = neural_job(cfg, seed, spec['fit_dates'], spec['validation_dates'], spec['budget'],
                                d, market, cfg['model'] == 'chronos2_lora')
        elif cfg['model'] == 'chronos2_frozen':
            device = runtime(seed)
            pipe, _ = chronos_pipeline(device=device)
            val = market.select(spec['validation_dates'])
            pred = chronos_predict(pipe, market, val, cfg)
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / 'inner_predictions.npz', origins=val, prediction=pred)
            result = {'best_score': score(market.y(val), pred), 'best_budget': 0,
                      'prediction_sha256': digest(d / 'inner_predictions.npz'), 'config': cfg, 'seed': seed,
                      'fit_dates': spec['fit_dates'], 'validation_dates': spec['validation_dates']}
            write_json(d / 'summary.json', result)
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
        else:
            result, _ = baseline_job(cfg, seed, spec['fit_dates'], spec['validation_dates'], d, market)
        return result
    except Exception as exc:
        write_json(d / 'failure.json', {'failed_at_utc': utc_now(), 'error': repr(exc), 'spec': spec,
                   'implementation_sha256': source_identity(), 'artifacts_preserved': True})
        raise


def run_inner(name, trials, split, market, root, dispatch=None, execution=None):
    trials = [{**cfg, **(execution or {})} for cfg in trials]
    root.mkdir(parents=True, exist_ok=True)
    run_id = identity({"model": name, "trials": trials, "split": split, "source": source_identity(), 'runner': digest(__file__)})
    idfile = root / "run_identity.json"
    if idfile.exists() and read_json(idfile)["identity"] != run_id:
        raise ValueError("Inner search identity changed")
    write_json(idfile, {"identity": run_id})
    if (root / "selection.json").exists():
        return read_json(root / "selection.json")[0]
    candidates = list(trials)
    neural_type = name in ("xpatch", "patchtst", "chronos2_lora")
    lora = name == "chronos2_lora"
    budgets, promoted = ([100, 300, 900], [4, 2]) if lora else ([3, 9, 27], [11, 4])
    rung_budgets = budgets if neural_type else [1]
    rankings = []
    for rung, budget in enumerate(rung_budgets):
        ranking_file = root / f"ranking_rung{rung}.json"
        if ranking_file.exists():
            ranking = read_json(ranking_file)
            rankings.append(ranking)
            if rung < len(promoted) and neural_type:
                candidates = [r["trial"] for r in ranking[:promoted[rung]]]
            continue
        ranking = []
        specifications = []
        for cfg in candidates:
            for j, inner in enumerate(split["inner"]):
                d = root / cfg["trial_id"] / f"inner{j}" / "seed2026"
                specifications.append(dict(config=cfg, seed=2026, fit_dates=inner['fit_dates'],
                    validation_dates=inner['validation_dates'], budget=budget, directory=str(d)))
        completed = dispatch(specifications) if dispatch else [execute_spec(s, market) for s in specifications]
        cursor = 0
        for cfg in candidates:
            total, count, details = 0.0, 0, []
            for j, inner in enumerate(split['inner']):
                val = market.select(inner["validation_dates"])
                s = completed[cursor]
                cursor += 1
                details.append(s)
                total += s["best_score"] * len(val)
                count += len(val)
            ranking.append({"trial": cfg, "mse": total / count, "inner": details})
        ranking.sort(key=lambda x: (x["mse"], x["trial"]["trial_id"]))
        rankings.append(ranking)
        write_json(ranking_file, ranking)
        if rung < len(promoted) and neural_type:
            candidates = [r["trial"] for r in ranking[:promoted[rung]]]
            for loser in ranking[promoted[rung]:]:
                prune_successful_checkpoints(root / loser["trial"]["trial_id"])
    final = rankings[-1]
    if name not in ("ridge", "chronos2_frozen"):
        reviews = []
        specifications = []
        for candidate in final[:2]:
            for seed in (2026, 2027, 2028):
                for j, inner in enumerate(split['inner']):
                    specifications.append(dict(config=candidate['trial'], seed=seed,
                        fit_dates=inner['fit_dates'], validation_dates=inner['validation_dates'],
                        budget=budgets[-1] if neural_type else 1,
                        directory=str(root / candidate['trial']['trial_id'] / f'inner{j}' / f'seed{seed}')))
        completed = dispatch(specifications) if dispatch else [execute_spec(s, market) for s in specifications]
        cursor = 0
        for candidate in final[:2]:
            cfg = candidate["trial"]
            scores, chosen_budgets = [], []
            for seed in (2026, 2027, 2028):
                total, count = 0.0, 0
                for j, inner in enumerate(split["inner"]):
                    s = completed[cursor]
                    cursor += 1
                    n = len(market.select(inner["validation_dates"]))
                    total += n * s["best_score"]
                    count += n
                    chosen_budgets.append(s["best_budget"])
                scores.append(total / count)
            reviews.append({"trial": cfg, "mse": float(np.mean(scores)), "per_seed_mse": scores,
                            "final_budget": int(math.floor(float(np.median(chosen_budgets)) + 0.5))})
        reviews.sort(key=lambda x: (x["mse"], x["trial"]["trial_id"]))
    else:
        reviews = [{**final[0], "final_budget": 1}]
    write_json(root / "selection.json", reviews)
    if neural_type:
        for trial in trials:
            if trial["trial_id"] != reviews[0]["trial"]["trial_id"]:
                prune_successful_checkpoints(root / trial["trial_id"])
    return reviews[0]


def prune_successful_checkpoints(trial_directory):
    """Only new, completed losing trials; metadata and all failure evidence survive."""
    directory = Path(trial_directory).resolve()
    if not any(directory.is_relative_to(ROOT / p) for p in ('outputs/formal', 'outputs/executor_smoke')):
        raise ValueError("Pruning is confined to new formal artifacts")
    for summary in directory.glob("inner*/seed*/summary.json"):
        job = summary.parent
        if (job / "checkpoints_pruned.json").exists() or (job / "failure.json").exists():
            continue
        saved = read_json(summary)
        if not saved.get("completed_budget") or saved.get("best_score") is None:
            continue
        removed = []
        for path in job.glob("*.pt"):
            receipt = path.with_suffix(".receipt.json")
            if not receipt.exists() or read_json(receipt)["sha256"] != digest(path):
                raise ValueError("Cannot prune a checkpoint whose integrity is uncertain")
            removed.append({"file": path.name, "bytes": path.stat().st_size, "sha256": digest(path)})
        write_json(job / "checkpoints_pruned.json", {"reason": "successful_trial_eliminated_by_locked_rule",
                   "files": removed, "implementation_sha256": source_identity(), "pruned_at_utc": utc_now()})
        for row in removed:
            (job / row["file"]).unlink()


def require_lock():
    locked = read_json(ROOT / "protocol_locked.json")
    receipt = read_json(ROOT / "protocol_lock_receipt.json")
    if digest(ROOT / "protocol_locked.json") != receipt["protocol_sha256"]:
        raise ValueError("Locked protocol changed")
    if source_identity() != locked["implementation_sha256"]:
        raise ValueError("Training implementation changed since lock")
    if digest(ROOT / locked["trial_list_file"]) != locked["trial_list_sha256"]:
        raise ValueError("Trial list changed since lock")
    if digest(ROOT / 'IMPLEMENTATION_PLAN.md') != locked['authoritative_plan']['sha256']:
        raise ValueError('The user plan changed after this protocol was locked')
    if digest(ROOT / 'environment_lock.json') != receipt['environment_sha256']:
        raise ValueError('Environment manifest changed since lock')
    if any(digest(ROOT / path) != checksum for path, checksum in receipt['implementation_files'].items()):
        raise ValueError('Implementation changed since lock')
    if any(digest(ROOT / path) != checksum for path, checksum in receipt['admission_files'].items()):
        raise ValueError('Admission evidence changed since lock')
    import importlib.metadata as metadata
    environment = read_json(ROOT / "environment_lock.json")
    for name, version in environment["packages"].items():
        if metadata.version(name) != version:
            raise ValueError(f"Dependency changed since lock: {name}")
    for path, checksum in environment["critical_api_sources"].items():
        if digest(path) != checksum:
            raise ValueError("Critical installed API source changed since lock")
    if digest(ROOT / "scripts/python.sh") != environment["launch_script_sha256"]:
        raise ValueError("Resource/offline launch policy changed since lock")
    if not all(locked["admission"][k] for k in ("data_passed", "adapters_passed", "performance_passed")):
        raise ValueError("Admission incomplete")
    return locked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['search', 'confirm', 'final', 'evaluate', 'all'], default='all')
    parser.add_argument('--execute-formal-research', action='store_true')
    args = parser.parse_args()
    if not args.execute_formal_research:
        raise SystemExit('Formal research requires --execute-formal-research and a passed protocol lock')
    from .pipeline import run
    run(args.stage)


if __name__ == '__main__':
    main()
