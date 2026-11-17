"""Test split grouping with fictional calendar dates, never market prices."""
import numpy as np
import pytest

from .corporate_actions import split_event_dates, split_event_flags


def test_future_split_group_respects_origin_and_target_boundaries():
    events = split_event_dates([{"date": "2020-08-31", "ratio": 4.0},
                                {"date": "2021-06-01", "ratio": 2.0}])
    origins = np.array(["2020-08-28", "2020-08-31", "2020-08-28", "2021-05-31", "2021-06-01"])
    targets = np.array(["2020-08-31", "2020-09-01", "2020-08-28", "2021-06-01", "2021-06-01"])
    assert split_event_flags(origins, targets, events).tolist() == [True, False, False, True, False]
    assert not split_event_flags(origins, targets, split_event_dates([])).any()


@pytest.mark.parametrize("actions", [
    [{"date": "2026-07-01", "ratio": 2.0}],
    [{"date": "2020-08-31", "ratio": 1.0}],
    [{"date": "2020-08-31", "ratio": 4.0}, {"date": "2020-08-31", "ratio": 2.0}],
    [{"date": "2021-01-01", "ratio": 2.0}, {"date": "2020-01-01", "ratio": 2.0}],
])
def test_invalid_or_post_cutoff_actions_fail_closed(actions):
    with pytest.raises(ValueError):
        split_event_dates(actions)


def test_invalid_session_order_and_dates_fail_closed():
    events = split_event_dates([])
    with pytest.raises(ValueError):
        split_event_flags(np.array(["2020-02-02"]), np.array(["2020-02-01"]), events)
    with pytest.raises(ValueError):
        split_event_flags(np.array(["2026-06-30"]), np.array(["2026-07-01"]), events)
