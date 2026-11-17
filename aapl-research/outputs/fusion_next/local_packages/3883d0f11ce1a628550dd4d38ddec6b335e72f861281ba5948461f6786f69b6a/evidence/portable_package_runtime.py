"""Restrict the existing FP32 launch-file substitution to package run.py only.

The unchanged adapter also intercepts the same source while copying the frozen
engine tree. Preserve that engine file byte for byte so its identity verifies.
"""
from pathlib import Path
import shutil

from fusion.next.common import ROOT,OUT,digest,identity,read_json,source_identity


_installed = False
POLICY = 'portable_package_policy.json'


def specification():
    from scripts.portable_inference_runtime import checked_policy
    numerical=checked_policy()
    value={'version':'portable-package-engine-copy-v1','source_identity':source_identity(),
        'numerical_policy_identity':numerical['identity'],'runtime_sha256':digest(Path(__file__)),
        'engine_source':'fusion/next/cloud_assets/cpu_run.py','entry_destination':'run.py',
        'model_bytes_changed':False,'inference_precision_changed':False,'research_gates_changed':False,
        'absolute_deadline_changed':False}
    value['identity']=identity(value)
    return value


def checked_policy():
    from fusion.next.common import OUT
    value=read_json(OUT/'performance'/POLICY)
    if value!=specification():raise ValueError('Portable package copy policy or runtime differs')
    return value


def install():
    global _installed
    if _installed:return
    from scripts import portable_inference_runtime as numerical
    if not numerical._installed:raise RuntimeError('Install the admitted FP32 adapter before package copy repair')
    policy=checked_policy()
    from fusion.next import local_package
    from fusion.next.common import OUT
    previous=local_package.shutil
    class Copy:
        def __getattr__(self,name):return getattr(previous,name)
        def copyfile(self,source,target,*args,**kwargs):
            if (Path(source).resolve()==(ROOT/policy['engine_source']).resolve()
                    and Path(target).name!=policy['entry_destination']):
                return shutil.copyfile(source,target,*args,**kwargs)
            return previous.copyfile(source,target,*args,**kwargs)
    local_package.shutil=Copy()
    original_files=local_package.source_files
    def files():
        return {**original_files(),'evidence/portable_package_policy.json':OUT/'performance'/POLICY,
                'evidence/portable_package_runtime.py':Path(__file__),
                'evidence/PORTABLE_PACKAGE_AMENDMENT.md':ROOT/'fusion/next/PORTABLE_PACKAGE_AMENDMENT.md'}
    local_package.source_files=files
    _installed=True
