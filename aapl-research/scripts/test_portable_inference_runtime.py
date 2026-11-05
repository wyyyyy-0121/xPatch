from types import SimpleNamespace
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import pytest


@pytest.fixture
def installed(monkeypatch):
    from scripts import portable_inference_runtime as adapter
    from fusion.next import common, experiment, trainer, inference, local_package, experts
    for obj, name in [(trainer.Trainer,'predict'),(inference.FrozenSystem,'predict_origin'),
                      (experiment,'save_prediction'),(experiment,'read_prediction'),
                      (experiment,'write_oof'),(experts,'write_oof'),(common,'write_json'),
                      (local_package,'source_files'),(local_package,'shutil')]:
        monkeypatch.setattr(obj,name,getattr(obj,name))
    monkeypatch.setattr(adapter,'_installed',False)
    monkeypatch.setattr(adapter,'checked_policy',lambda:{'identity':'policy'})
    return adapter, trainer, inference


def test_prediction_precision_does_not_mutate_training_recipe(installed,monkeypatch):
    adapter,trainer,_=installed
    run_until=trainer.Trainer.run_until;optimizer=trainer.optimizer_for
    config={'precision':'bf16','lr':.001};recipe={'config':config}
    def original(self):
        assert self.config['precision']=='fp32'
        assert self.recipe['config'] is config and config['precision']=='bf16'
        return 'predicted'
    monkeypatch.setattr(trainer.Trainer,'predict',original);adapter.install()
    fake=SimpleNamespace(config=config,recipe=recipe)
    assert trainer.Trainer.predict(fake)=='predicted'
    assert fake.config is config and fake.recipe is recipe
    assert trainer.Trainer.run_until is run_until and trainer.optimizer_for is optimizer


def test_failed_prediction_restores_original_config(installed,monkeypatch):
    adapter,trainer,_=installed
    def failed(self):raise RuntimeError('controlled inference failure')
    monkeypatch.setattr(trainer.Trainer,'predict',failed);adapter.install()
    config={'precision':'bf16'};fake=SimpleNamespace(config=config)
    with pytest.raises(RuntimeError):trainer.Trainer.predict(fake)
    assert fake.config is config


def test_system_precision_restores_expert_and_seed_configs(installed,monkeypatch):
    adapter,_,inference=installed
    cfg={'precision':'bf16'};models=[('model','scaler',cfg)];experts={'xpatch':('expert','scaler',cfg)}
    def original(self):
        assert all(c['precision']=='fp32' for _,_,c in self.models)
        assert all(c['precision']=='fp32' for _,_,c in self.experts.values())
        assert cfg['precision']=='bf16'
        return 'predicted'
    monkeypatch.setattr(inference.FrozenSystem,'predict_origin',original);adapter.install()
    fake=SimpleNamespace(models=models,experts=experts)
    assert inference.FrozenSystem.predict_origin(fake)=='predicted'
    assert fake.models is models and fake.experts is experts and cfg['precision']=='bf16'


def test_actual_isolated_CPU_entry_uses_package_root():
    from fusion.next.common import ROOT,OUT,source_files,source_identity,digest,identity,read_json
    with TemporaryDirectory(prefix='portable_entry_test_',dir=ROOT/'tmp') as temporary:
        directory=Path(temporary)
        for name,path in source_files().items():
            target=directory/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        shutil.copyfile(ROOT/'scripts/portable_cpu_run.py',directory/'run.py')
        shutil.copyfile(ROOT/'scripts/portable_inference_runtime.py',directory/'portable_inference_runtime.py')
        policy={'base_engine_source_identity':source_identity(),'runtime_sha256':digest(directory/'portable_inference_runtime.py'),
            'CPU_entrypoint_sha256':digest(directory/'run.py'),'inference_precision':'float32','training_precision_changed':False}
        policy['identity']=identity(policy)
        (directory/'portable_inference_policy.json').write_text(json.dumps(policy))
        (directory/'runtime_lock.json').write_text(json.dumps({'python':platform.python_version(),'packages':{}}))
        files={str(p.relative_to(directory)):{'sha256':digest(p)} for p in directory.rglob('*') if p.is_file()}
        package={'source_identity':source_identity(),'model_identity':'entry-only-check-no-model','files':files};package['identity']=identity(package)
        (directory/'package.json').write_text(json.dumps(package))
        result=subprocess.run([sys.executable,'run.py','--verify-only'],cwd=directory,
            env={**os.environ,'PYTHONPATH':str(directory),'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,timeout=30)
        assert result.returncode==0,result.stderr
        assert json.loads(result.stdout)['verified'] is True
