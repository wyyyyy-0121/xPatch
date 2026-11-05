"""Fixed-budget causal expert fits, independent of the old evaluation pipeline."""
import gc
from pathlib import Path

import numpy as np
import torch

from aapl.baselines import design
from aapl.common import ROOT, digest, read_json, utc_now, write_json
from aapl.models import chronos_pipeline, chronos_predict, neural
from aapl.runner import baseline_job, neural_job
from aapl.training import neural_predict


def expert_prediction(name, selection, market, fit_dates, evaluation_dates, directory):
    directory = Path(directory)
    config, budget = selection['trial'], selection['final_budget']
    if name in ('ridge', 'lightgbm'):
        summary, predict = baseline_job(config, 2026, fit_dates, evaluation_dates,
                                       directory, market, fixed_trees=budget)
        pred = predict(market.select(evaluation_dates))
    elif name in ('xpatch', 'patchtst'):
        summary, predict, hold = neural_job(config, 2026, fit_dates, evaluation_dates,
                                           budget, directory, market, refit=True)
        pred = predict(market.select(evaluation_dates))
        del hold
    else:
        raise ValueError(name)
    del predict
    gc.collect()
    torch.cuda.empty_cache()
    return pred


def load_expert(name, selection, directory, market, device='cuda'):
    directory = Path(directory)
    config = selection['trial']
    summary = read_json(directory / 'summary.json')
    if summary['config'].get('trial_id') != config['trial_id']:
        raise ValueError('Expert configuration identity changed')
    scale = market.scale(summary['fit_dates'])
    if name == 'ridge':
        if digest(directory / 'ridge.npz') != summary['model_sha256']:
            raise ValueError('Ridge artifact changed')
        with np.load(directory / 'ridge.npz') as saved:
            coef, intercept = saved['coef'].copy(), saved['intercept'].copy()
        return lambda m, origins: (design(m, origins, config, scale) @ coef + intercept) * scale['target_std']
    if name == 'lightgbm':
        import lightgbm as lgb
        if digest(directory / 'lightgbm.txt') != summary['model_sha256']:
            raise ValueError('LightGBM artifact changed')
        model = lgb.Booster(model_file=str(directory / 'lightgbm.txt'))
        return lambda m, origins: model.predict(design(m, origins, config, scale), num_threads=4) * scale['target_std']
    path = directory / 'best.pt'
    if read_json(path.with_suffix('.receipt.json'))['sha256'] != digest(path):
        raise ValueError('Neural expert artifact changed')
    saved = torch.load(path, map_location=device, weights_only=False)
    model = neural(config).to(device)
    model.load_state_dict(saved['model'])
    model.eval()
    return lambda m, origins: neural_predict(model, config, m, origins, scale, device)
