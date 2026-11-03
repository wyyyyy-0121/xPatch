"""Generate the complete immutable search order before observing outer scores."""
from itertools import product
import math
import random

from .io import stable_hash


def configurations(config):
    search = config["search"]
    rng = random.Random(search["seed"])
    representations = search["representations"]
    result = {"last_rv": [{"representation": "log_seasonal_ratio"}]}
    result["har"] = [dict(alpha=a, semivariance=s, representation=r)
        for a, s, r in product(search["har"]["alpha"], search["har"]["semivariance"], search["har"]["representations"])]
    result["ridge"] = [dict(alpha=a, feature_profile=f, representation=r)
        for a, f, r in product(search["ridge"]["alpha"], search["ridge"]["feature_profile"], representations)]

    def log_uniform(bounds):
        return round(math.exp(rng.uniform(*map(math.log, bounds))), 10)

    for model in ("lightgbm", "xpatch"):
        settings, trials, identities = search[model], [], set()
        while len(trials) < search["counts"][model]:
            if model == "lightgbm":
                params = {key: rng.choice(settings[key]) for key in
                    ("num_leaves", "max_depth", "min_child_samples", "feature_profile")}
                if params["max_depth"] > 0 and params["num_leaves"] > 2 ** params["max_depth"]:
                    continue
                for key in ("learning_rate", "lambda_l1", "lambda_l2"):
                    params[key] = log_uniform(settings[key + "_log_range"])
                for key in ("feature_fraction", "bagging_fraction"):
                    params[key] = round(rng.uniform(*settings[key + "_range"]), 8)
                params["max_estimators"] = settings["max_estimators"]
            else:
                params = {key: rng.choice(settings[key]) for key in
                    ("seq_len", "patch_len", "ma_type", "alpha", "revin", "batch_size")}
                params["beta"] = rng.choice(settings["beta"]) if params["ma_type"] == "dema" else params["alpha"]
                params["stride"] = params["patch_len"] // 2
                params["pred_len"] = settings["pred_len"]
                for key in ("learning_rate", "weight_decay"):
                    params[key] = log_uniform(settings[key + "_log_range"])
            params["representation"] = rng.choice(representations)
            identity = stable_hash(params)
            if identity not in identities:
                trials.append(params)
                identities.add(identity)
        result[model] = trials
    result["chronos2"] = [dict(context_length=c, representation=r)
        for c, r in product(search["chronos2"]["context_length"], representations)]
    for name, trials in result.items():
        if len({stable_hash(p) for p in trials}) != len(trials):
            raise ValueError(f"Nonunique search list: {name}")
        if name != "last_rv" and len(trials) != search["counts"][name]:
            raise ValueError(f"Incomplete search list: {name}")
    return {"schema_version": 1, "seed": search["seed"], "models": {
        name: [{"id": f"{name}_{i:03d}_{stable_hash(params)[:10]}", "params": params}
            for i, params in enumerate(trials)] for name, trials in result.items()}}


def rank_trials(records, expected_folds=3, expected_seeds=(2026,)):
    """Missing/failed fold or seed cannot win against a complete configuration."""
    grouped = {}
    for row in records:
        grouped.setdefault(row["config_id"], []).append(row)
    ranked = []
    for identity, rows in grouped.items():
        pairs = {(row["fold"], row["seed"]) for row in rows}
        if len(rows) != expected_folds * len(expected_seeds) or len(pairs) != len(rows):
            continue
        if {row["seed"] for row in rows} != set(expected_seeds):
            continue
        folds = {row["fold"] for row in rows}
        if len(folds) != expected_folds or any({row["fold"] for row in rows if row["seed"] == seed} != folds for seed in expected_seeds):
            continue
        if any(row["status"] != "complete" or not math.isfinite(row["score"]["qlike"]) for row in rows):
            continue
        ranked.append({"config_id": identity, "inner_qlike": sum(row["score"]["qlike"] for row in rows) / len(rows)})
    return sorted(ranked, key=lambda row: (row["inner_qlike"], row["config_id"]))
