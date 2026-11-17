"""Exercise actual systemd isolation with one idle fictional job, never AAPL."""
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from multiscale.historical_continuation import (boot_id, boot_time, digest, guard_ready, install_watchdog,
                                              locked, matched, process, read, safe_signal, save)

DEST = Path(__file__).resolve().parent
PROJECT = Path('/home/ayaka/wslcode/aapl-research')


def until(predicate, seconds=8):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.02)
    raise RuntimeError('Fictional service test timed out')


def fictional_controller(identity_path):
    child = subprocess.Popen([sys.executable, '-c',
        'import os,signal,time;os.kill(os.getpid(),signal.SIGSTOP);time.sleep(60)'], start_new_session=True)
    until(lambda: (p := process(child.pid)) and p['state'] == 'T')
    save(identity_path, {'controller': process(os.getpid()), 'worker': process(child.pid)})
    time.sleep(60)


def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ').lower()
    run = DEST / ('synthetic-service-probe-' + stamp)
    run.mkdir()
    fold = run / 'fictional_fold'
    fold.mkdir()
    save(fold / 'fold_state.json', {'fictional': True, 'steps': 123})
    (fold / 'latest.pt').write_bytes(b'fictional checkpoint; not model weights')
    manifest_path = run / 'FICTIONAL_MANIFEST.json'
    manifest = {'jobs': [{'key': 'fictional', 'folder': str(fold)}], 'training': False}
    save(manifest_path, manifest)
    identity_path = run / 'FICTIONAL_IDENTITIES.json'
    unit = 'aapl-synthetic-controller-' + stamp
    args = ['systemd-run', '--user', '--unit=' + unit, '--property=KillMode=process',
            '--property=WorkingDirectory=' + str(PROJECT), '--setenv=PYTHONPATH=' + str(PROJECT),
            '/usr/bin/taskset', '-c', '24,26', sys.executable, str(Path(__file__).resolve()),
            '--fictional-controller', str(identity_path)]
    result = subprocess.run(args, capture_output=True, text=True, timeout=10)
    (run / 'REGISTER.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0
    identities = None
    try:
        until(identity_path.exists)
        identities = read(identity_path)
        properties = subprocess.run(['systemctl', '--user', 'show', unit + '.service',
            '--property=MainPID', '--property=KillMode', '--property=ActiveState'],
            capture_output=True, text=True, timeout=5)
        values = dict(line.split('=', 1) for line in properties.stdout.splitlines() if '=' in line)
        assert values == {'MainPID': str(identities['controller']['pid']), 'KillMode': 'process', 'ActiveState': 'active'}
        state = {'phase': 'armed', 'controller': identities['controller'], 'boot_id': boot_id(),
                 'manifest': str(manifest_path), 'manifest_sha256': digest(manifest_path),
                 'heartbeat_boottime': boot_time(), 'deadline_epoch': time.time() + 30,
                 'deadline_boottime': boot_time() + 30, 'watchdog': None,
                 'jobs': {'fictional': {'phase': 'running', 'process': identities['worker'], 'slot': [24,26]}}}
        save(run / 'STATE.json', state)
        install_watchdog(run)
        state = read(run / 'STATE.json')
        guard_ready(state)
        assert os.sched_getaffinity(state['watchdog']['pid']) == {24,26}
        assert safe_signal(identities['worker'], signal.SIGCONT)
        until(lambda: (p := matched(identities['worker'])) and p['state'] == 'S')
        stopped = subprocess.run(['systemctl', '--user', 'stop', unit + '.service'],
                                 capture_output=True, text=True, timeout=10)
        assert stopped.returncode == 0
        until(lambda: (run / 'STOP_RECEIPT.json').exists())
        receipt = read(run / 'STOP_RECEIPT.json')
        assert receipt['reason'] == 'controller_exited'
        assert matched(identities['worker'])['state'] == 'T'
        assert len(receipt['files']) == 2
        assert all(digest(f['source']) == digest(f['copy']) == f['sha256'] for f in receipt['files'])
        report = {'status': 'real_systemd_isolation_verified_with_idle_fictional_job',
                  'at_utc': datetime.now(timezone.utc).isoformat(), 'controller_properties': values,
                  'independent_watchdog_unit': state['watchdog_unit'], 'watchdog_verified_before_release': True,
                  'controller_exit_left_worker_alive_and_frozen': True,
                  'stop_receipt': str(run / 'STOP_RECEIPT.json'), 'stop_receipt_sha256': digest(run / 'STOP_RECEIPT.json'),
                  'files_verified': len(receipt['files']), 'training_updates': 0,
                  'AAPL_processes_signalled': False, 'Q3_values_read': False, 'cores': [24,26],
                  'script_sha256': digest(__file__)}
        save(DEST / 'SYNTHETIC_SERVICE_VERIFICATION.json', report)
        print(report, flush=True)
    finally:
        if identities:
            safe_signal(identities['worker'], signal.SIGKILL)
        subprocess.run(['systemctl', '--user', 'stop', unit + '.service'], capture_output=True, timeout=5)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--fictional-controller':
        fictional_controller(Path(sys.argv[2]))
    else:
        main()
