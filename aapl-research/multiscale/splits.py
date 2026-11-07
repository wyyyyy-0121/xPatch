"""Chronological development splits purged by actual label completion time."""
from __future__ import annotations

import numpy as np

from .bars import Bars


def inner_split(bars: Bars, outer_train, horizon: int, stop_fraction: float = 0.15):
    train = np.asarray(outer_train, dtype=np.int64)
    if not 0 < stop_fraction < 1 or not len(train):
        raise ValueError("A nonempty train partition and valid stop fraction are required")
    groups = np.unique(bars.session_date[train])
    tail_count = max(1, int(np.ceil(len(groups) * stop_fraction)))
    if len(groups) <= tail_count:
        return {"fit": np.array([], dtype=np.int64), "stop": train.copy(), "status": "insufficient_groups"}
    stop = train[np.isin(bars.session_date[train], groups[-tail_count:])]
    stop_start = bars.end_ns[stop[0]]
    fit = train[bars.end_ns[train + horizon] < stop_start]
    return {"fit": fit, "stop": stop, "status": "ok" if len(fit) and len(stop) else "insufficient_fit"}


def rolling_folds(bars: Bars, origins, horizon: int, folds: int = 5):
    """Split origin dates forward, retaining empty/insufficient folds explicitly.

    Both inner and outer purges use actual end timestamps, not positions in a
    filtered origin array. Returned indices are bar indices, never scores.
    """
    origins = np.asarray(origins, dtype=np.int64)
    if horizon < 1 or folds < 1 or origins.ndim != 1:
        raise ValueError("Invalid folds, origins or horizon")
    if len(origins) and ((np.diff(origins) <= 0).any() or origins[0] < 0
                         or origins[-1] + horizon >= len(bars.close)):
        raise ValueError("Origins must be sorted, unique and have complete future positions")
    groups = np.unique(bars.session_date[origins])
    if len(groups) < folds + 1:
        return []
    chunks = np.array_split(groups, folds + 1)
    result = []
    for number, group in enumerate(chunks[1:], start=1):
        validation = origins[np.isin(bars.session_date[origins], group)]
        if not len(validation):
            result.append({"fold": number, "train": np.array([], dtype=np.int64),
                           "validation": validation, "inner": None, "status": "empty_validation"})
            continue
        first_decision = bars.end_ns[validation[0]]
        train = origins[bars.end_ns[origins + horizon] < first_decision]
        inner = inner_split(bars, train, horizon) if len(train) else None
        status = "ok" if inner is not None and inner["status"] == "ok" else "insufficient_training"
        result.append({"fold": number, "train": train, "validation": validation,
                       "inner": inner, "status": status})
    return result


def warmup_folds(bars: Bars, origins, horizon: int, *, folds: int = 5,
                 min_inner_fit: int = 24, min_inner_stop: int = 4, min_validation_groups: int = 6):
    """Propose five folds after a prespecified minimum warm-up; never inspect scores."""
    origins = np.asarray(origins, dtype=np.int64)
    if (horizon < 1 or folds < 1 or min_inner_fit < 1 or min_inner_stop < 1 or
            min_validation_groups < 1 or origins.ndim != 1 or
            (len(origins) and ((np.diff(origins) <= 0).any() or origins[0] < 0 or
                               origins[-1] + horizon >= len(bars.close)))):
        raise ValueError("Invalid warm-up split inputs")
    groups = np.unique(bars.session_date[origins])
    last_warmup = len(groups) - folds * min_validation_groups
    for warmup in range(1, last_warmup + 1):
        first_validation = origins[bars.session_date[origins] == groups[warmup]][0]
        train = origins[bars.end_ns[origins + horizon] < bars.end_ns[first_validation]]
        inner = inner_split(bars, train, horizon) if len(train) else None
        if inner and len(inner["fit"]) >= min_inner_fit and len(inner["stop"]) >= min_inner_stop:
            break
    else:
        return []
    result = []
    for number, chunk in enumerate(np.array_split(groups[warmup:], folds), start=1):
        validation = origins[np.isin(bars.session_date[origins], chunk)]
        train = origins[bars.end_ns[origins + horizon] < bars.end_ns[validation[0]]]
        inner = inner_split(bars, train, horizon)
        if len(inner["fit"]) < min_inner_fit or len(inner["stop"]) < min_inner_stop:
            raise ValueError("A later fold violates the minimum warm-up requirements")
        result.append({"fold": number, "train": train, "validation": validation, "inner": inner,
                       "status": "ok", "validation_groups": len(chunk), "warmup_groups": warmup})
    return result
