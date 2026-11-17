"""Explicit inference-only adapter for the admitted engine; training stays unchanged.

All replacements are scoped to prediction/export entry points. The original
source identity remains the base engine identity; the extra policy and this
file's hash must accompany every new result and the CPU package.
"""
from contextlib import contextmanager
import functools
from pathlib import Path
import shutil


POLICY_NAME = 'portable_inference_policy.json'
_installed = False


def policy_path():
    from fusion.next.common import ROOT, OUT
    packaged = ROOT/POLICY_NAME
    return packaged if packaged.exists() else OUT/'performance'/POLICY_NAME


def checked_policy():
    from fusion.next.common import read_json, identity, digest, source_identity
    path = policy_path();value = read_json(path)
    if value['identity'] != identity({k:v for k,v in value.items() if k != 'identity'}):
        raise ValueError('Portable inference policy changed')
    if value['base_engine_source_identity'] != source_identity():
        raise ValueError('Portable inference uses a different admitted engine')
    if value['runtime_sha256'] != digest(Path(__file__)):
        raise ValueError('Portable inference adapter changed')
    from fusion.next.common import ROOT
    entry = ROOT/'run.py' if (ROOT/POLICY_NAME).exists() else ROOT/'scripts/portable_cpu_run.py'
    if digest(entry) != value['CPU_entrypoint_sha256']:
        raise ValueError('Portable CPU entry point changed')
    if value['inference_precision'] != 'float32' or value['training_precision_changed']:
        raise ValueError('Inference amendment cannot change training')
    return value


def install():
    global _installed
    if _installed:return
    policy = checked_policy()
    from fusion.next import common, experiment, trainer, inference, local_package, experts
    from fusion.next.common import ROOT, OUT

    original_predict = trainer.Trainer.predict
    @functools.wraps(original_predict)
    def predict(self, *args, **kwargs):
        previous = self.config
        self.config = {**previous, 'precision':'fp32'}
        try:return original_predict(self, *args, **kwargs)
        finally:self.config = previous
    trainer.Trainer.predict = predict

    original_origin = inference.FrozenSystem.predict_origin
    @functools.wraps(original_origin)
    def predict_origin(self, *args, **kwargs):
        models, experts = self.models, self.experts
        self.models = [(model,scalers,{**cfg,'precision':'fp32'}) for model,scalers,cfg in models]
        self.experts = {name:(model,scalers,{**cfg,'precision':'fp32'}) for name,(model,scalers,cfg) in experts.items()}
        try:return original_origin(self, *args, **kwargs)
        finally:self.models, self.experts = models, experts
    inference.FrozenSystem.predict_origin = predict_origin

    original_save = experiment.save_prediction
    @functools.wraps(original_save)
    def save_prediction(directory, rows, prediction, panel, details):
        if details['config']['family'] in ('xpatch','patchtst','fusion'):
            details = {**details, 'evaluation_precision':'float32', 'numerical_policy_identity':policy['identity']}
        return original_save(directory, rows, prediction, panel, details)
    experiment.save_prediction = save_prediction

    original_read = experiment.read_prediction
    @functools.wraps(original_read)
    def read_prediction(directory, rows, details):
        from fusion.next.common import read_json
        stored = read_json(Path(directory)/'result.json')['details']
        if 'evaluation_precision' in stored:
            if stored['evaluation_precision'] != 'float32' or stored['numerical_policy_identity'] != policy['identity']:
                raise ValueError('Prediction belongs to another numerical policy')
            details = {**details, 'evaluation_precision':'float32', 'numerical_policy_identity':policy['identity']}
        return original_read(directory, rows, details)
    experiment.read_prediction = read_prediction

    original_oof = experts.write_oof
    @functools.wraps(original_oof)
    def write_oof(panel, information, fold, predictions, configurations, training_scope='multistock', market_information=True):
        result = original_oof(panel, information, fold, predictions, configurations, training_scope, market_information)
        path = OUT/'oof'/experts.scope_key(information,training_scope,market_information)/(fold['quarter']+'.json')
        receipt = common.read_json(path)
        receipt.update(evaluation_precision='float32', numerical_policy_identity=policy['identity'])
        common.write_json(path,receipt)
        return result
    experts.write_oof = write_oof;experiment.write_oof = write_oof

    original_write = common.write_json
    @functools.wraps(original_write)
    def write_json(path, value):
        if Path(path).name in ('F3_complete.json','F4_complete.json','F5_complete.json','confirmation.json'):
            value = {**value,'evaluation_precision':'float32','numerical_policy_identity':policy['identity']}
        return original_write(path,value)
    common.write_json = write_json

    # Package the adapter and its own immutable policy. The base model source
    # identity continues to cover exactly the original admitted engine files.
    original_files = local_package.source_files
    def package_files():
        return {**original_files(), 'portable_inference_runtime.py':Path(__file__),
                POLICY_NAME:policy_path(),
                'evidence/PORTABLE_INFERENCE_AMENDMENT.md':ROOT/'fusion/next/PORTABLE_INFERENCE_AMENDMENT.md'}
    local_package.source_files = package_files
    original_copy = shutil.copyfile
    class PackageCopy:
        def __getattr__(self, name):return getattr(shutil, name)
        def copyfile(self, source, target, *args, **kwargs):
            if Path(source).resolve() == (ROOT/'fusion/next/cloud_assets/cpu_run.py').resolve():
                return original_copy(ROOT/'scripts/portable_cpu_run.py', target, *args, **kwargs)
            return original_copy(source, target, *args, **kwargs)
    local_package.shutil = PackageCopy()
    _installed = True

