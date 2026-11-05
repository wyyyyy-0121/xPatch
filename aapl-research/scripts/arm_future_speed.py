"""Attach verified acceleration to future local children, leaving cloud training live."""
import os
from pathlib import Path

import psutil

from fusion.next.common import OUT, ROOT, digest, read_json, utc_now, write_json
from scripts.future_speed_runtime import checked_policy


def arm():
    policy = checked_policy()
    directory = OUT/'cloud/local_completion'
    dispatch = read_json(directory/'dispatch.json')
    process = psutil.Process(dispatch['pid'])
    if abs(process.create_time()-dispatch['created_unix']) >= .01:
        raise ValueError('Local completion process identity changed')
    state = read_json(directory/'state.json')
    if state['stage'] not in ('waiting_for_cloud', 'cloud_connection_retry'):
        raise ValueError('Local completion has already entered a task; do not change it')
    bootstrap = ROOT/'tmp/portable_inference_bootstrap/sitecustomize.py'
    original = 'from scripts.portable_inference_runtime import install\ninstall()\n'
    optimized = original + ('import sys\n'
        'if sys.argv[0] == "-m" or sys.argv[0].endswith("/scripts/finish_fusion_100h.py"):\n'
        '    from scripts.future_speed_runtime import install as speed\n    speed()\n')
    if bootstrap.read_text() not in (original, optimized):
        raise ValueError('Unexpected existing child bootstrap')
    pending = bootstrap.with_suffix('.pending')
    pending.write_text(optimized)
    os.replace(pending, bootstrap)
    receipt = {'at_utc': utc_now(), 'speed_policy_identity': policy['identity'],
               'local_observer_pid': dispatch['pid'], 'local_observer_created_unix': dispatch['created_unix'],
               'hook_sha256': digest(bootstrap), 'active_parent_modified': False,
               'current_cloud_training_modified': False,
               'activation': 'future local completion child interpreters only',
               'future_cloud_entry': ['bash', 'cloud/run.sh', 'scripts/start_fast_fusion.py',
                                      'scripts/run_fusion_100h.py'],
               'cloud_entry_must_not_run_while_existing_runner_is_alive': True}
    write_json(directory/'future_speed_hook.json', receipt)
    return receipt


if __name__ == '__main__':
    print(arm())
