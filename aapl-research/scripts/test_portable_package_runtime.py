"""Package copy repair and future-child hook tests; never affect real training."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest

from fusion.next.common import ROOT,OUT,digest,read_json,write_json
from scripts import portable_package_runtime as package
from scripts import arm_portable_package as arm


def test_only_root_entry_is_substituted_and_engine_source_bytes_are_preserved(monkeypatch):
    from fusion.next import local_package
    from scripts import portable_inference_runtime as numerical
    monkeypatch.setattr(package,'_installed',False);monkeypatch.setattr(numerical,'_installed',True)
    monkeypatch.setattr(package,'checked_policy',lambda:{'engine_source':'fusion/next/cloud_assets/cpu_run.py','entry_destination':'run.py'})
    import shutil
    class Prior:
        def __getattr__(self,name):return getattr(shutil,name)
        def copyfile(self,source,target,*args,**kwargs):return shutil.copyfile(ROOT/'scripts/portable_cpu_run.py',target,*args,**kwargs)
    monkeypatch.setattr(local_package,'shutil',Prior())
    monkeypatch.setattr(local_package,'source_files',local_package.source_files)
    with TemporaryDirectory(dir=ROOT/'tmp',prefix='package_copy_test_') as temporary:
        root=Path(temporary);engine=root/'cpu_run.py';entry=root/'run.py';source=ROOT/'fusion/next/cloud_assets/cpu_run.py'
        package.install();local_package.shutil.copyfile(source,engine);local_package.shutil.copyfile(source,entry)
        assert digest(engine)==digest(source) and digest(entry)==digest(ROOT/'scripts/portable_cpu_run.py')
        assert digest(engine)!=digest(entry)


@pytest.fixture
def hook(monkeypatch):
    with TemporaryDirectory(dir=ROOT/'tmp',prefix='package_hook_test_') as temporary:
        root=Path(temporary);directory=root/'outputs/cloud/local_completion'
        monkeypatch.setattr(arm,'ROOT',root);monkeypatch.setattr(arm,'OUT',root/'outputs')
        monkeypatch.setattr(arm,'specification',lambda:{'identity':'synthetic_package_policy','runtime_sha256':'synthetic_runtime'})
        monkeypatch.setattr(arm.psutil,'Process',lambda pid:SimpleNamespace(pid=pid,create_time=lambda:7.,is_running=lambda:True,status=lambda:'sleeping'))
        write_json(directory/'dispatch.json',{'pid':123,'created_unix':7.})
        write_json(directory/'state.json',{'stage':'waiting_for_cloud'})
        path=root/'tmp/portable_inference_bootstrap/sitecustomize.py';path.parent.mkdir(parents=True)
        path.write_text('from scripts.portable_inference_runtime import install\ninstall()\nimport sys\n'
            'if sys.argv[0] == "-m":\n    from scripts.future_speed_runtime import install as speed\n    speed()\n')
        write_json(directory/'future_speed_hook.json',{'hook_sha256':digest(path),'speed_policy_identity':'synthetic_speed'})
        yield directory,path


def test_waiting_receiver_arms_only_future_children_and_preserves_previous_hook(hook):
    directory,path=hook;original=path.read_bytes()
    value=arm.arm()
    assert not value['active_receiver_modified'] and not value['current_cloud_training_modified']
    assert (directory/'before_package_sitecustomize.py').read_bytes()==original
    assert value['hook_sha256']==digest(path)==read_json(directory/'future_speed_hook.json')['hook_sha256']
    assert arm.arm()==value and 'package_copy()' in path.read_text()


def test_active_phase_or_reused_receiver_identity_refuses_hook_changes(hook):
    directory,path=hook;before=path.read_bytes()
    write_json(directory/'state.json',{'stage':'F8_local_replay'})
    with pytest.raises(RuntimeError,match='already entered'):arm.arm()
    assert path.read_bytes()==before
    write_json(directory/'state.json',{'stage':'waiting_for_cloud'})
    write_json(directory/'dispatch.json',{'pid':123,'created_unix':8.})
    with pytest.raises(ValueError,match='identity changed'):arm.arm()
    assert path.read_bytes()==before
