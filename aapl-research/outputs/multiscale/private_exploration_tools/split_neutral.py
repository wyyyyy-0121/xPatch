"""Causal split-neutral price series and independent label validation."""
from dataclasses import replace
from datetime import date

import numpy as np

from multiscale.bars import Bars,targets


def normalized_actions(actions):
    result=[]
    previous=None
    for action in actions:
        day=str(action['date'])
        if date.fromisoformat(day).isoformat()!=day:
            raise ValueError('Split date must be an ISO calendar date')
        ratio=float(action['ratio'])
        if not np.isfinite(ratio) or ratio<=0 or (previous is not None and day<=previous):
            raise ValueError('Split actions must have positive ratios and increasing distinct dates')
        result.append({'date':day,'ratio':ratio})
        previous=day
    return result


def share_adjusted_bars(bars:Bars,actions):
    """Multiply each bar by the share count due to splits effective by that bar.

    Each factor uses actions no later than its own date. A future action cannot
    alter an earlier value. Cash dividends and other actions are not included.
    """
    actions=normalized_actions(actions)
    factors=np.ones(len(bars.close),dtype=np.float64)
    for action in actions:
        factors[bars.session_date>=action['date']]*=action['ratio']
    if not np.isfinite(factors).all() or (factors<=0).any():
        raise ValueError('Invalid cumulative split factor')
    adjusted=bars.close.copy()*factors
    return replace(bars,close=adjusted),factors


def independent_neutral_return(bars:Bars,origins,horizon,actions):
    """Compute raw log returns plus realized split ratios in each target path."""
    actions=normalized_actions(actions)
    idx=np.asarray(origins,dtype=np.int64)
    result=targets(bars,idx,horizon,'future_log_return')
    future=idx[:,None]+np.arange(1,horizon+1)
    origin_dates=bars.session_date[idx,None]
    future_dates=bars.session_date[future]
    for action in actions:
        crossed=(origin_dates<action['date']) & (future_dates>=action['date'])
        result+=crossed*np.log(action['ratio'])
    return result
