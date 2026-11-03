"""Prevent the previous experiment's restored-purge issue from recurring."""
import pytest

from finance.splits import inner_partitions, outer_partitions, partition
from finance_v2.data import calendar_grid


def test_final_fit_retains_both_excluded_purge_dates():
    days = tuple(calendar_grid("2021-01-04", "2025-06-30"))
    folds = outer_partitions(days)
    assert len(folds) == 10
    for fold in folds:
        groups = [set(fold.fit_dates), set(fold.calibration_dates), set(fold.evaluation_dates),
                  set(fold.purge_before_calibration), set(fold.purge_before_evaluation)]
        assert sum(len(group) for group in groups) == len(set.union(*groups))
        assert len(fold.calibration_dates) == 20
        assert len(fold.purge_before_calibration) == len(fold.purge_before_evaluation) == 1
        assert fold.fit_dates[-1] < fold.purge_before_calibration[0] < fold.calibration_dates[0]
        assert fold.calibration_dates[-1] < fold.purge_before_evaluation[0] < fold.evaluation_dates[0]
        # No final-refit union with either purge group is permitted by this object.
        assert not set(fold.fit_dates) & set(fold.purge_before_calibration + fold.purge_before_evaluation)


def test_three_inner_folds_expand_without_using_outer_calibration():
    days = tuple(calendar_grid("2021-01-04", "2025-06-30"))
    outer = outer_partitions(days)[0]
    inner = inner_partitions(outer)
    assert len(inner) == 3
    assert [len(p.evaluation_dates) for p in inner] == [20, 20, 20]
    for first, later in zip(inner, inner[1:]):
        assert set(first.fit_dates) < set(later.fit_dates)
        assert first.evaluation_dates[-1] < later.evaluation_dates[0]
    for p in inner:
        for group in p.dictionary().values():
            if isinstance(group, tuple):
                assert set(group).issubset(outer.fit_dates)
                assert not set(group) & set(outer.calibration_dates + outer.evaluation_dates)


def test_partition_rejects_overlapping_or_unordered_dates():
    with pytest.raises(ValueError):
        partition("bad", ["2023-01-02", "2023-01-01"], ["2023-01-03"], 1, 1)
    with pytest.raises(ValueError):
        partition("bad", ["2023-01-01"] * 25, ["2023-01-01"], 20, 1)
    with pytest.raises(ValueError):
        outer_partitions(tuple(calendar_grid("2024-01-01", "2025-06-30")))
