"""Budget regression checks use fabricated timings; no research outputs or GPU."""
from datetime import datetime
import pytest
from scripts import fusion_9h as focus


def values():
    parent={'deployment_fit_policy':{'total_updates':23424},
        'folds':{q:{'total_updates':n,'score_origins':100} for q,n in zip(focus.QUARTERS,(17884,20057,22337))},
        'original_inputs':{'deployment/pre_Q2/toy':{'bytes':1000}}}
    probe={'seconds_per_update_upper':.06,'prediction_seconds_per_origin_upper':.001,
        'initialization_seconds':5.,'checkpoint_copy_bytes':1000,'checkpoint_copy_seconds':.1}
    return parent,probe


def test_full_scope_counts_every_seed_quarter_and_engineering_fit():
    parent,probe=values();result=focus.project(parent,probe)
    assert result['updates']==251106 and result['full_fits']==12
    assert result['prediction_origins']==900
    assert result['required_seconds']==pytest.approx((251106*.06+900*.001+12*5+.1)*1.5+60)
    final=focus.project(parent,probe,True)
    assert final['updates']==70272 and final['full_fits']==3 and final['prediction_origins']==0


def test_elapsed_waiting_cannot_reset_shortened_window(monkeypatch):
    parent,probe=values();deadline=datetime.fromisoformat(focus.guard.DEADLINE).timestamp()
    monkeypatch.setattr(focus.time,'time',lambda:deadline-8*3600)
    focus.budget(parent,probe)
    monkeypatch.setattr(focus.time,'time',lambda:deadline-3600)
    with pytest.raises(TimeoutError):focus.budget(parent,probe)
    probe['seconds_per_update_upper']=float('nan')
    with pytest.raises(TimeoutError):focus.budget(parent,probe)


def test_completed_artifact_conflict_is_preserved(tmp_path):
    path=tmp_path/'receipt.json';focus.immutable(path,{'value':1})
    focus.immutable(path,{'value':1})
    with pytest.raises(ValueError):focus.immutable(path,{'value':2})
    assert focus.read_json(path)=={'value':1}
