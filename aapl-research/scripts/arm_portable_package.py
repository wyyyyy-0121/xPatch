"""Attach a verified package-only repair to future waiting-receiver children."""
import os
from pathlib import Path

import psutil

from fusion.next.common import ROOT,OUT,digest,read_json,write_json,utc_now
from scripts.portable_package_runtime import specification


def arm():
    directory=OUT/'cloud/local_completion';dispatch=read_json(directory/'dispatch.json')
    process=psutil.Process(dispatch['pid'])
    if (abs(process.create_time()-dispatch['created_unix'])>=.01 or not process.is_running()
            or process.status()==psutil.STATUS_ZOMBIE):raise ValueError('Waiting local receiver identity changed')
    if read_json(directory/'state.json')['stage'] not in ('waiting_for_cloud','cloud_connection_retry'):
        raise RuntimeError('Local receiver already entered a task; do not alter its future bootstrap')
    policy=specification();policy_path=OUT/'performance/portable_package_policy.json'
    if policy_path.exists() and read_json(policy_path)!=policy:raise ValueError('Preserve the existing package repair policy')
    hook=ROOT/'tmp/portable_inference_bootstrap/sitecustomize.py'
    suffix=('\n# Verified package copy repair; future local children only.\n'
        'if sys.argv[0] == "-m" or sys.argv[0].endswith("/scripts/finish_fusion_100h.py"):\n'
        '    from scripts.portable_package_runtime import install as package_copy\n    package_copy()\n')
    receipt_path=directory/'portable_package_hook.json'
    if receipt_path.exists():
        saved=read_json(receipt_path)
        if (saved['hook_sha256']!=digest(hook) or saved['policy_identity']!=policy['identity']
                or saved['local_receiver']!={'pid':process.pid,'created_unix':process.create_time()}):
            raise ValueError('Previously armed package bootstrap identity differs')
        return saved
    speed_hook=read_json(directory/'future_speed_hook.json')
    if speed_hook['hook_sha256']!=digest(hook) or 'from scripts.future_speed_runtime import install as speed' not in hook.read_text():
        raise ValueError('Future-child bootstrap is not the verified speed hook')
    source=hook.read_text()
    if 'package_copy' in source:raise ValueError('Unexpected unreceipted package hook')
    before=directory/'before_package_sitecustomize.py'
    if before.exists() and before.read_text()!=source:raise ValueError('Preserve previous bootstrap evidence')
    before.write_text(source)
    previous=directory/'before_package_speed_hook.json'
    if previous.exists() and read_json(previous)!=speed_hook:raise ValueError('Previous speed-hook receipt differs')
    write_json(previous,speed_hook);write_json(policy_path,policy)
    # Revalidate immediately before the atomic change; the receiver itself and
    # its imported code remain untouched, as does the running cloud executor.
    if (abs(psutil.Process(process.pid).create_time()-dispatch['created_unix'])>=.01
            or read_json(directory/'state.json')['stage'] not in ('waiting_for_cloud','cloud_connection_retry')):
        raise RuntimeError('Receiver changed phase before arming future package children')
    pending=hook.with_suffix('.package_pending');pending.write_text(source+suffix);os.replace(pending,hook)
    value={'policy_identity':policy['identity'],'policy_sha256':digest(policy_path),'runtime_sha256':policy['runtime_sha256'],
        'local_receiver':{'pid':process.pid,'created_unix':process.create_time()},'hook_sha256':digest(hook),
        'prior_hook_sha256':digest(before),'active_receiver_modified':False,'current_cloud_training_modified':False,
        'activation':'future local completion child interpreters only','armed_at_utc':utc_now()}
    write_json(receipt_path,value)
    write_json(directory/'future_speed_hook.json',{**speed_hook,'hook_sha256':digest(hook),
        'package_policy_identity':policy['identity'],'package_repair_hook_receipt':'portable_package_hook.json'})
    return value


if __name__=='__main__':print(arm())
