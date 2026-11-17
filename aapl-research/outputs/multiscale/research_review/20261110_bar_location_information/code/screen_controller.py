"""Run a frozen screen through one real external pause and same-deadline resume."""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time

BASE = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, value):
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno()); temporary = f.name
    os.replace(temporary, path)


def main():
    protocol = read(BASE/'MARKET_PROTOCOL.json')
    identity = sha(BASE/'MARKET_PROTOCOL.json')
    deadline = protocol['deadline_epoch']
    output = BASE/'screen'; output.mkdir(exist_ok=True)
    receipt = output/'LIFECYCLE_ACCEPTANCE.json'
    if receipt.exists():
        raise RuntimeError('A lifecycle run already has a receipt; do not overwrite/restart it')
    if not protocol['external_signal_pause_after_first_saved_batch']:
        raise RuntimeError('This controller requires the predeclared single pause')
    command = [sys.executable, str(BASE/'code/screen_worker.py'), '--run']
    child = None
    state = dict(protocol_sha256=identity, deadline_epoch=deadline, same_deadline_for_resume=True)
    try:
        if os.sched_getaffinity(0) != {24}:
            raise RuntimeError('Controller and worker must remain on CPU24')
        ready = output/'PAUSE_REQUEST_READY.json'
        if ready.exists():
            raise RuntimeError('Do not restart the single-pause controller over existing progress')
        with (output/'initial_worker.log').open('x') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            wait_limit = min(deadline, time.time()+180)
            while not ready.exists() and child.poll() is None and time.time() < wait_limit:
                time.sleep(.02)
            if not ready.exists() or child.poll() is not None:
                raise RuntimeError('Worker never reached its saved-batch pause handshake')
            checkpoint = read(ready)
            if checkpoint['protocol_sha256'] != identity or checkpoint['pid'] != child.pid:
                raise RuntimeError('Unexpected worker/checkpoint identity')
            os.kill(child.pid, signal.SIGUSR1)
            initial_code = child.wait(timeout=min(30, max(.1, deadline-time.time())))
        paused = read(output/'RUN_STATE.json')
        if initial_code == 0 or paused['status'] != 'paused':
            raise RuntimeError('External signal did not produce the declared checkpoint pause')
        if sha(Path(checkpoint['checkpoint'])) != checkpoint['checkpoint_sha256']:
            raise RuntimeError('Checkpoint bytes changed while handling pause')
        state.update(external_signal='SIGUSR1', first_return_code=initial_code,
            checkpoint=checkpoint, checkpoint_bytes_preserved_at_pause=True,
            original_pause_state=paused, status='paused_then_resuming')
        atomic(receipt, state)
        with (output/'resumed_worker.log').open('x') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            final_code = child.wait(timeout=max(.1, deadline-time.time()))
        final_state = read(output/'RUN_STATE.json')
        state.update(status='passed' if final_code == 0 else 'resume_failed',
                     resumed_return_code=final_code, final_worker_state=final_state,
                     resumed_from_completed_cursor=checkpoint['cursor'])
        atomic(receipt, state)
        if final_code:
            raise RuntimeError('Same-deadline resumed worker failed')
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
        state.update(status='failed', error_type=type(error).__name__, reason=str(error),
                     completed_checkpoints_retained=True)
        atomic(receipt, state)
        raise


if __name__ == '__main__':
    if sys.argv[1:] != ['--run']:
        raise SystemExit('Use --run only with the existing frozen market protocol/admission')
    main()
