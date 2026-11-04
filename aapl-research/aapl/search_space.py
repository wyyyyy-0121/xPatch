"""Generate pre-scoring trial lists exactly once, with paired supervised losses."""
import itertools
import numpy as np

from .common import identity


def _draw(grid, n, rng):
    keys = list(grid)
    values = [dict(zip(keys, v)) for v in itertools.product(*grid.values())]
    return [values[i] for i in rng.choice(len(values), size=n, replace=False)]


def generate():
    rng = np.random.default_rng(20261104)
    result = {}
    result["ridge"] = [dict(model="ridge", lookback=l, alpha=a) for l in (48, 96, 240)
                       for a in (1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000)]
    result["lightgbm"] = _draw(dict(model=["lightgbm"], lookback=[48, 96, 240],
        num_leaves=[7, 15, 31], max_depth=[3, 5, -1], min_child_samples=[100, 500, 2000],
        learning_rate=[0.01, 0.03, 0.1], reg_lambda=[0, 1, 10], colsample_bytree=[0.7, 1.0],
        n_estimators=[1000]), 48, rng)
    for model in ("xpatch", "patchtst"):
        grid = dict(model=[model], lookback=[48, 96, 240], patch=[8, 12, 16], stride_ratio=[0.5, 1.0],
                    lr=[1e-4, 3e-4, 1e-3], weight_decay=[0, 1e-4, 1e-2], batch_size=[64, 128],
                    revin=[False, True])
        if model == "xpatch":
            grid["alpha"] = [0.01, 0.05, 0.1, 0.3, 0.7]
        else:
            grid.update(d_model=[32, 64], layers=[1, 2, 3], dropout=[0, 0.1, 0.2])
        geoms = _draw(grid, 16, rng)
        trials = []
        for geom in geoms:
            geom["stride"] = int(geom["patch"] * geom.pop("stride_ratio"))
            trials.extend([{**geom, "loss": loss} for loss in ("step_mse", "cumulative12_mse")])
        result[model] = trials
    result["chronos2_frozen"] = [dict(model="chronos2_frozen", lookback=l, representation=r, point=p)
                                for l in (48, 96, 240)
                                for r in ("minute_return", "anchored_log_price")
                                for p in ("native_median", "quantile_integral_tailflat")]
    result["chronos2_lora"] = _draw(dict(model=["chronos2_lora"], lookback=[48, 96, 240],
        lr=[1e-5, 3e-5, 1e-4], rank=[4, 8, 16], point=["native_median", "quantile_integral_tailflat"],
        representation=["minute_return"], weight_decay=[0.0], batch_size=[16]), 12, rng)
    for name, trials in result.items():
        for config in trials:
            config["trial_id"] = identity(config)[:16]
        if len(set(t["trial_id"] for t in trials)) != len(trials):
            raise ValueError("Duplicate trial list")
    return result
