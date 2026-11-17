"""Clock/availability operations only. No prices, targets, predictions or fits."""
from __future__ import annotations

import numpy as np


def delayed_history_available(observed, origins, lookback):
    """At original origin i require source indices [i-L, i-1], without filling."""
    observed = np.asarray(observed)
    origins = np.asarray(origins)
    if (observed.ndim != 1 or observed.dtype != np.bool_
            or origins.ndim != 1 or origins.dtype != np.dtype('int64')
            or type(lookback) is not int or lookback < 1
            or (origins < 0).any() or (origins >= len(observed)).any()):
        raise ValueError('Invalid availability inputs')
    result = np.zeros(len(origins), dtype=bool)
    valid = origins >= lookback
    positions = origins[valid]
    missing = np.concatenate((np.zeros(1, dtype=np.int64),
                              np.cumsum(~observed, dtype=np.int64)))
    result[valid] = missing[positions] == missing[positions-lookback]
    return result


def partition_original_fit(fit, bar_end_ns, horizon):
    """Preserve original 80/20 boundary, then purge prefix by actual label clock."""
    fit = np.asarray(fit)
    ends = np.asarray(bar_end_ns)
    if (fit.dtype != np.dtype('int64') or fit.ndim != 1 or len(fit) < 2
            or ends.dtype != np.dtype('int64') or ends.ndim != 1
            or (np.diff(fit) <= 0).any() or (np.diff(ends) <= 0).any()
            or type(horizon) is not int or horizon < 1
            or fit[0] < 0 or fit[-1]+horizon >= len(ends)):
        raise ValueError('Invalid original fit/clock inputs')
    cut = 4*len(fit)//5
    tail = fit[cut:].copy()
    first_tail_decision = ends[tail[0]]
    candidates = fit[:cut]
    prefix = candidates[ends[candidates+horizon] < first_tail_decision].copy()
    return cut, prefix, tail


def original_hour_groups(minute_start_ns, minute_end_ns, session_date):
    """Full sixty-minute original-grid blocks from each session open; no residual."""
    start = np.asarray(minute_start_ns)
    end = np.asarray(minute_end_ns)
    session = np.asarray(session_date)
    minute = 60_000_000_000
    if (start.ndim != 1 or end.shape != start.shape or session.shape != start.shape
            or len(start) == 0 or start.dtype != np.dtype('int64')
            or end.dtype != np.dtype('int64') or session.dtype.kind != 'U'
            or (np.diff(start) <= 0).any() or (end-start != minute).any()
            or (session[1:] < session[:-1]).any()):
        raise ValueError('Invalid original minute grid')
    first = np.flatnonzero(np.r_[True, session[1:] != session[:-1]])
    stop = np.r_[first[1:], len(start)]
    groups = []
    for a, b in zip(first, stop):
        if (np.diff(start[a:b]) != minute).any():
            raise ValueError('Original within-session clock has a gap')
        groups.extend(range(int(a), int(b)-59, 60))
    a = np.asarray(groups, dtype=np.int64)
    return a, a+60


def hour_source_available(minute_observed, group_start, group_stop):
    observed = np.asarray(minute_observed)
    a = np.asarray(group_start)
    b = np.asarray(group_stop)
    if (observed.ndim != 1 or observed.dtype != np.bool_
            or a.ndim != 1 or a.shape != b.shape or a.dtype != np.dtype('int64')
            or b.dtype != np.dtype('int64') or (a < 0).any()
            or (b > len(observed)).any() or (b-a != 60).any()):
        raise ValueError('Invalid original-hour blocks')
    missing = np.r_[0, np.cumsum(~observed, dtype=np.int64)]
    return missing[b] == missing[a]
