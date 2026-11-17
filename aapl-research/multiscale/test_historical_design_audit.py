"""Check clock gaps, missing positions, cutoff and strict label purging."""
import numpy as np
import pytest

from .historical_coverage import EvidenceError
from .historical_design_audit import (HOUR_NS, Metadata, clock_summary, complete_origins,
                                      index_digest, load_metadata, partitions)
from .historical_minute_check import CUTOFF_NS


def example(hours, sessions=None, missing=()):
    end = np.asarray(hours, dtype=np.int64) * HOUR_NS
    observed = np.ones(len(end), dtype=bool)
    observed[list(missing)] = False
    sessions = np.asarray(sessions or [f"2026-01-{i + 1:02d}" for i in range(len(end))])
    return Metadata(end - HOUR_NS, end, observed, sessions)


def test_clock_horizon_includes_weekend_and_overnight():
    meta = example([1, 2, 3, 70, 71, 72], ["2026-01-02"] * 3 + ["2026-01-05"] * 3)
    result = clock_summary(meta, [0, 1, 2], 1)
    assert result["elapsed_hours_min"] == result["elapsed_hours_median"] == 1
    assert result["elapsed_hours_max"] == 67
    assert result["cross_session_origins"] == 1
    assert result["cross_session_fraction"] == pytest.approx(1 / 3)


def test_missing_position_invalidates_input_and_every_target_step():
    meta = example(list(range(1, 13)), ["2026-01-02"] * 6 + ["2026-01-05"] * 6, (4,))
    assert complete_origins(meta, 3, 2, False).tolist() == [7, 8, 9]
    assert complete_origins(meta, 3, 2, True).tolist() == [7, 8, 9]
    clean = example(list(range(1, 13)), ["2026-01-02"] * 6 + ["2026-01-05"] * 6)
    assert 4 in complete_origins(clean, 3, 2, False)
    assert 4 not in complete_origins(clean, 3, 2, True)


def test_purge_rejects_labels_finishing_exactly_on_next_decision():
    meta = example(list(range(1, 21)))
    part = partitions(meta, np.arange(1, 18), 2, np.array([15, 16, 17]))
    assert part["train"].tolist() == list(range(1, 13))
    assert part["stop"].tolist() == [11, 12]
    assert part["fit"].tolist() == list(range(1, 9))
    assert not set(part["fit"]) & set(part["stop"])


def test_cutoff_checks_observed_completion_and_all_session_dates():
    meta = example([1, 2, 3])
    meta.validate()
    end = meta.end.copy()
    end[-1] = CUTOFF_NS
    with pytest.raises(EvidenceError, match="protected cutoff"):
        Metadata(meta.start, end, meta.observed, meta.sessions).validate()
    with pytest.raises(EvidenceError, match="protected cutoff"):
        Metadata(meta.start, meta.end, meta.observed,
                 np.array(["2026-06-28", "2026-06-29", "2026-07-01"])).validate()


def test_split_hash_is_order_sensitive_and_endian_stable():
    assert index_digest(np.array([1, 2, 3], dtype=">i8")) == index_digest([1, 2, 3])
    assert index_digest([1, 2, 3]) != index_digest([1, 3, 2])


def test_metadata_loading_does_not_materialize_price_member(tmp_path, monkeypatch):
    from . import historical_design_audit as audit_module
    from .historical_coverage import digest

    meta = example([1, 2, 3])
    path = tmp_path / "bars_hour.npz"
    # Loading this member with allow_pickle=False would fail immediately.
    np.savez_compressed(path, start_ns=meta.start, end_ns=meta.end, observed=meta.observed,
                        session_date=meta.sessions, close=np.array([object()], dtype=object))
    monkeypatch.setattr(audit_module, "in_tree", lambda value: value)
    restored, evidence = load_metadata(tmp_path, {"files": {path.name: digest(path)}}, "hour")
    assert np.array_equal(restored.end, meta.end)
    assert "close" not in evidence["members_read"]
