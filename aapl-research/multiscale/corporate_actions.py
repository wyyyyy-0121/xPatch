"""Predeclared descriptive grouping for future split events in label paths."""
from __future__ import annotations

from datetime import date
import math

import numpy as np

from .spec import DEV_END


SPLIT_SENSITIVITY_RULE = {
    "grouping": "origin_session_lt_effective_date_lte_target_session",
    "metrics": ["endpoint_mse", "endpoint_mae", "path_mse"],
    "use": "descriptive_development_only_no_model_selection",
}


def split_event_dates(actions):
    """Validate the reviewed effective dates; never accept protected-period events."""
    if not isinstance(actions, list):
        raise ValueError("A reviewed split-action list is required")
    dates = []
    for action in actions:
        if not isinstance(action, dict) or not isinstance(action.get("date"), str):
            raise ValueError("Invalid split action")
        raw, ratio = action["date"], action.get("ratio")
        try:
            day = date.fromisoformat(raw)
        except ValueError as exc:
            raise ValueError("Invalid split effective date") from exc
        if (day.isoformat() != raw or raw > DEV_END or isinstance(ratio, bool) or
                not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or ratio <= 0 or ratio == 1):
            raise ValueError("Invalid or out-of-scope split action")
        dates.append(raw)
    if dates != sorted(set(dates)):
        raise ValueError("Split actions must have unique increasing effective dates")
    return np.asarray(dates, dtype="datetime64[D]")


def _calendar_days(values):
    source = np.asarray(values)
    if source.ndim != 1 or source.dtype.kind != "U":
        raise ValueError("Session dates must be a one-dimensional Unicode array")
    try:
        days = source.astype("datetime64[D]")
    except ValueError as exc:
        raise ValueError("Invalid session date") from exc
    if (np.isnat(days).any() or (np.datetime_as_string(days, unit="D") != source).any() or
            (source > DEV_END).any()):
        raise ValueError("Invalid or post-cutoff session date")
    return days


def split_event_flags(origin_sessions, target_sessions, event_dates):
    """Flag an event after the origin session and through the final target session.

    Flags are used only for a post-score descriptive breakdown, never as model
    inputs or a reason to select a model. An event on the origin session is
    already effective at that origin and is not counted as a future event.
    """
    origin, target = _calendar_days(origin_sessions), _calendar_days(target_sessions)
    events = np.asarray(event_dates)
    if origin.shape != target.shape or (target < origin).any():
        raise ValueError("Origin and target sessions are not aligned")
    if (events.ndim != 1 or events.dtype != np.dtype("datetime64[D]") or np.isnat(events).any() or
            (events[1:] <= events[:-1]).any() or (events > np.datetime64(DEV_END)).any()):
        raise ValueError("Invalid reviewed split-event dates")
    return np.searchsorted(events, target, side="right") > np.searchsorted(events, origin, side="right")
