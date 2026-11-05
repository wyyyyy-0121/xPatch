from pathlib import Path
from types import SimpleNamespace

import pytest

from fusion.next.common import read_json, write_json
from scripts import arm_future_speed as arm


@pytest.fixture
def setup_hook(tmp_path, monkeypatch):
    monkeypatch.setattr(arm, 'OUT', tmp_path/'outputs')
    monkeypatch.setattr(arm, 'ROOT', tmp_path)
    monkeypatch.setattr(arm, 'checked_policy', lambda: {'identity': 'speed'})
    monkeypatch.setattr(arm.psutil, 'Process', lambda pid: SimpleNamespace(create_time=lambda: 7.0))
    directory = arm.OUT/'cloud/local_completion'
    # write_json accepts only research-root paths, so test in a real allowed directory.
    from fusion.next.common import ROOT
    allowed = ROOT/'tmp'/tmp_path.name
    monkeypatch.setattr(arm, 'OUT', allowed/'outputs')
    monkeypatch.setattr(arm, 'ROOT', allowed)
    directory = arm.OUT/'cloud/local_completion'
    write_json(directory/'dispatch.json', {'pid': 123, 'created_unix': 7.0})
    write_json(directory/'state.json', {'stage': 'waiting_for_cloud'})
    hook = arm.ROOT/'tmp/portable_inference_bootstrap/sitecustomize.py'
    hook.parent.mkdir(parents=True)
    hook.write_text('from scripts.portable_inference_runtime import install\ninstall()\n')
    yield directory, hook
    import shutil
    shutil.rmtree(allowed)


def test_only_future_children_are_armed_and_rearming_is_idempotent(setup_hook):
    directory, hook = setup_hook
    value = arm.arm()
    assert value['current_cloud_training_modified'] is False
    assert value['active_parent_modified'] is False
    assert 'if sys.argv[0] == "-m"' in hook.read_text()
    assert arm.arm()['hook_sha256'] == value['hook_sha256']


def test_never_attach_after_local_work_has_started(setup_hook):
    directory, hook = setup_hook
    original = hook.read_bytes()
    write_json(directory/'state.json', {'stage': 'F8_local_replay'})
    with pytest.raises(ValueError, match='entered a task'):
        arm.arm()
    assert hook.read_bytes() == original


def test_reused_pid_cannot_arm_the_hook(setup_hook):
    directory, _ = setup_hook
    write_json(directory/'dispatch.json', {'pid': 123, 'created_unix': 8.0})
    with pytest.raises(ValueError, match='identity changed'):
        arm.arm()
