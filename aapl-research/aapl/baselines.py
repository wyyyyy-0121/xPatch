from pathlib import Path
import os
import time

import numpy as np


def design(market, origins, config, scale):
    return market.x(origins, config["lookback"], scale).reshape(len(origins), -1).astype(np.float64)


def fit_ridge(market, origins, config, scale):
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=config["alpha"], fit_intercept=True, solver="cholesky")
    model.fit(design(market, origins, config, scale), market.y(origins) / scale["target_std"])
    return model


def fit_lightgbm(market, origins, config, scale, validation=None):
    import lightgbm as lgb
    params = {k: config[k] for k in ("num_leaves", "max_depth", "min_child_samples", "learning_rate",
                                    "reg_lambda", "colsample_bytree") if k in config}
    model = lgb.LGBMRegressor(**params, n_estimators=config.get("n_estimators", 1000),
                             n_jobs=int(os.environ.get('AAPL_CPU_THREADS', '4')), random_state=config.get("seed", 2026), deterministic=True,
                             force_col_wise=True, verbosity=-1, objective="regression_l2")
    kwargs = {}
    if validation is not None:
        kwargs.update(eval_set=[(design(market, validation, config, scale), market.y(validation) / scale["target_std"])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
    model.fit(design(market, origins, config, scale), market.y(origins) / scale["target_std"], **kwargs)
    return model


def predict(model, market, origins, config, scale):
    return model.predict(design(market, origins, config, scale)) * scale["target_std"]
