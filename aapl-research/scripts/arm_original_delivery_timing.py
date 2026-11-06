"""Arm only future original CPU children while the verified receiver waits."""
import os
from pathlib import Path

import psutil

from fusion.next.common import ROOT, OUT, PLAN, digest, read_json, source_identity, utc_now, write_json
from scripts.original_delivery_timing_runtime import POLICY_NAME, process_birth, specification


def verified_receiver():
    directory = OUT/'cloud/local_completion'
    review = read_json(directory/'receiver_identity_clock_review.json')
    dispatch = read_json(directory/'dispatch.json')
    binding = read_json(directory/'binding.json')
    prior = review['current_observation']
    process = psutil.Process(dispatch['pid'])
    birth = process_birth(process.pid)
    if (birth != {k:prior[k] for k in ('pid','boot_id','start_ticks','ticks_per_second')}
            or not process.is_running() or process.status() == psutil.STATUS_ZOMBIE
            or Path(f'/proc/{process.pid}').stat().st_ino != prior['proc_dir_inode']
            or process.cmdline() != prior['command'] or process.cwd() != str(ROOT)
            or dispatch != review['original_receiver']
            or digest(directory/'dispatch.json') != review['original_dispatch_sha256']
            or binding['source_identity'] != source_identity() or binding['plan_sha256'] != digest(PLAN)
            or binding['script_sha256'] != digest(ROOT/'scripts/finish_fusion_100h.py')):
        raise ValueError('Original waiting receiver kernel identity, binding or code differs')
    if (read_json(directory/'state.json')['stage'] not in ('waiting_for_cloud','cloud_connection_retry')
            or process.children(recursive=True)):
        raise RuntimeError('Receiver is not in an idle waiting boundary; preserve its bootstrap')
    return birth


def arm():
    directory = OUT/'cloud/local_completion'
    birth = verified_receiver()
    policy = specification()
    path = OUT/'performance'/POLICY_NAME
    hook = ROOT/'tmp/portable_inference_bootstrap/sitecustomize.py'
    receipt_path = directory/'original_delivery_timing_hook.json'
    speed = read_json(directory/'future_speed_hook.json')
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if (receipt['policy_identity'] != policy['identity'] or read_json(path) != policy
                or receipt['hook_sha256'] != digest(hook) or receipt['receiver_kernel_identity'] != birth
                or speed['hook_sha256'] != digest(hook)):
            raise ValueError('Previously armed full-process timing hook differs')
        return receipt
    if path.exists() and read_json(path) != policy:
        raise ValueError('Preserve the previous full-process timing policy')
    package = read_json(directory/'portable_package_hook.json')
    source = hook.read_text()
    if (digest(hook) != speed['hook_sha256'] or digest(hook) != package['hook_sha256']
            or package['policy_identity'] != policy['package_policy_identity']
            or speed['speed_policy_identity'] != policy['speed_policy_identity']
            or 'package_copy()' not in source or 'speed()' not in source
            or 'original_delivery_timing_runtime' in source):
        raise ValueError('Future bootstrap is not the verified package and speed chain')
    before = directory/'before_cold_timing_sitecustomize.py'
    if before.exists() and before.read_text() != source:
        raise ValueError('Preserve the previous future-child bootstrap')
    before.write_text(source)
    prior_speed = directory/'before_cold_timing_speed_hook.json'
    if prior_speed.exists() and read_json(prior_speed) != speed:
        raise ValueError('Previous speed-hook receipt differs')
    write_json(prior_speed, speed)
    write_json(path, policy)
    if verified_receiver() != birth or digest(hook) != speed['hook_sha256']:
        raise RuntimeError('Receiver or bootstrap changed before timing installation')
    suffix = ('\n# Verified full-process cold timing; future local children only.\n'
              'if sys.argv[0] == "-m" or sys.argv[0].endswith("/scripts/finish_fusion_100h.py"):\n'
              '    try:\n'
              '        from scripts.original_delivery_timing_runtime import install as cold_timing\n'
              '        cold_timing()\n'
              '    except Exception as cold_error:\n'
              '        raise SystemExit("Required full-process cold timing failed: " + str(cold_error)) from cold_error\n')
    pending = hook.with_suffix('.cold_timing_pending')
    pending.write_text(source+suffix)
    os.replace(pending, hook)
    receipt = {'policy_identity':policy['identity'], 'policy_sha256':digest(path),
               'runtime_sha256':policy['runtime_sha256'], 'installer_sha256':digest(Path(__file__)),
               'receiver_kernel_identity':birth, 'hook_sha256':digest(hook),
               'prior_hook_sha256':digest(before), 'prior_speed_receipt_sha256':digest(prior_speed),
               'legacy_creation_epoch_not_used_as_exit_evidence':True,
               'active_receiver_modified':False, 'current_cloud_training_modified':False,
               'activation':'future original local completion child interpreters only', 'armed_at_utc':utc_now()}
    write_json(receipt_path, receipt)
    write_json(directory/'future_speed_hook.json', {**speed, 'hook_sha256':digest(hook),
               'cold_timing_policy_identity':policy['identity'],
               'cold_timing_hook_receipt':receipt_path.name})
    return receipt


if __name__ == '__main__':
    print(arm())
