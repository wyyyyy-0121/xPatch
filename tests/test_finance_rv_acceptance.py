"""Unfrozen or incomplete work cannot satisfy the final delivery goal."""
import pytest

from finance.acceptance import accept
from finance.io import ROOT, read_json
from finance import provenance


def test_draft_protocol_and_null_budgets_cannot_start_formal_delivery(monkeypatch, tmp_path):
    monkeypatch.setattr(provenance, "FREEZE_PATH", tmp_path / "not_frozen.json")
    config = read_json(ROOT / "configs/rv30.json")
    config["status"] = "draft_pending_pre_outer_smoke"
    config["runtime"]["total_gpu_hours"] = None
    with pytest.raises(ValueError, match="budgets"):
        provenance.freeze(config, "irrelevant", "irrelevant")
    result = accept(config, tmp_path / "incomplete_delivery")
    assert not result["passed"] and not result["goal_can_complete"]
