"""Verify each proposed fallback in a fresh process; never train or resume."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from multiscale.historical_continuation import digest, matched, preflight, read, save

DEST = Path(__file__).resolve().parent
PROJECT = Path('/home/ayaka/wslcode/aapl-research')
MANIFEST = DEST / 'MANIFEST.json'


def main():
    manifest = read(MANIFEST)
    modes = preflight(manifest)
    started = time.monotonic()
    records = []
    for job in manifest['jobs']:
        if not job['initial_fold_state_sha256']:
            continue
        output = DEST / (job['key'] + '_CHECKPOINT_PREFLIGHT.json')
        command = [sys.executable, '-m', 'multiscale.historical_continuation', 'checkpoint-check',
                   '--manifest', str(MANIFEST), '--job', job['key'], '--output', str(output)]
        checked = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True, timeout=120)
        log = DEST / (job['key'] + '_CHECKPOINT_PREFLIGHT.log')
        log.write_text(checked.stdout + checked.stderr)
        if checked.returncode:
            raise RuntimeError(f'{job["key"]} failed read-only checkpoint verification; see {log}')
        value = read(output)
        assert value['training_started'] is False and value['processes_resumed'] is False
        assert value['Q3_values_read'] is False and value['manifest_sha256'] == digest(MANIFEST)
        records.append({'job': job['key'], 'verification': str(output), 'sha256': digest(output),
                        'active_checkpoints': value['checkpoint_details']})
        print(json.dumps({'verified': job['key'], 'active_checkpoints': value['checkpoint_details']}), flush=True)
    processes = []
    for job in manifest['jobs']:
        original = job['original_process']
        if not original:
            continue
        current = matched(original)
        assert current and current['state'] in ('T', 't')
        processes.append(current)
    result = {'status': 'all_seven_original_fallback_identities_verified_without_updates',
              'at_utc': datetime.now(timezone.utc).isoformat(), 'manifest_sha256': digest(MANIFEST),
              'jobs': records, 'processes_still_frozen': processes, 'modes': modes,
              'wall_seconds': time.monotonic() - started, 'training_updates': 0,
              'processes_resumed': False, 'GPU_training': False, 'Q3_values_read': False,
              'cores': sorted(os.sched_getaffinity(0)), 'script_sha256': digest(__file__)}
    assert len(records) == 7 and len(processes) == 7
    save(DEST / 'CHECKPOINT_FALLBACK_VERIFICATION.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('jobs', 'modes', 'processes_still_frozen')}), flush=True)


if __name__ == '__main__':
    main()
