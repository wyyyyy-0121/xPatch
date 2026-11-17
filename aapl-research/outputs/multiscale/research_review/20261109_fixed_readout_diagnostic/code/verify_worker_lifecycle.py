"""Actual fresh-process synthetic pause/signal/deadline/recovery checks."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parents[1]
WORKER = BASE/'code/readout_worker.py'


def load(path):
    return json.loads(path.read_text())


def main():
    assert set(os.sched_getaffinity(0)) == {24}
    import numpy as np
    runs = BASE/'engineering'
    rows = []
    wall_times = []

    def command(name, batches, *extra):
        return [sys.executable, str(WORKER), '--mode', 'synthetic', '--run-dir',
                str(runs/name), '--synthetic-batches', str(batches), *extra]

    def invoke(name, batches, *extra, expected=0):
        start = time.perf_counter()
        result = subprocess.run(command(name, batches, *extra), capture_output=True, text=True)
        wall_times.append(time.perf_counter()-start)
        (runs/f'{name}_exit{result.returncode}.log').write_text(result.stdout+result.stderr)
        if result.returncode != expected:
            raise RuntimeError(f'{name}: expected exit{expected},actual exit{result.returncode}: {result.stderr}')
        return result

    def compare(left, right):
        a, b = load(runs/left/'CHECKPOINT.json'), load(runs/right/'CHECKPOINT.json')
        assert a['identity'] == b['identity'] and a['binding'] == b['binding']
        assert a['cursor'] == b['cursor'] == a['qr']['expected_n'] == b['qr']['expected_n']
        ra, rb = np.asarray(a['qr']['augmented_R']), np.asarray(b['qr']['augmented_R'])
        errors = []
        for x, y in [(ra.T@ra, rb.T@rb)]:
            errors.append(float(np.linalg.norm(x-y)/max(1.0, np.linalg.norm(x), np.linalg.norm(y))))
        ar, br = load(runs/left/'RESULT.json'), load(runs/right/'RESULT.json')
        for key in ['SSE_original', 'SSE_projected', 'SSE_head_gap', 'SSE_zero_baseline']:
            x, y = ar['solution'][key], br['solution'][key]
            errors.append(abs(x-y)/max(1.0, abs(x), abs(y)))
        assert max(errors) <= 1e-10
        assert a['binding']['stage_deadline_epoch'] == load(BASE/'STAGE_SCOPE.json')['deadline_epoch']
        return {'compared': [left, right], 'committed_rows': a['cursor'],
                'maximum_normalized_invariant_or_loss_difference': max(errors),
                'original_absolute_stage_deadline_preserved': True}

    # A complete independent8 run already establishes maximum-shape integration.
    assert load(runs/'continuous8/RESULT.json')['status'] == 'completed'
    invoke('pause8', 8, '--pause-after-batches', '3', expected=2)
    assert load(runs/'pause8/CHECKPOINT.json')['cursor'] == 384
    invoke('pause8', 8)
    rows.append({'name': 'fresh_process_batch_boundary_restore', **compare('continuous8', 'pause8')})

    invoke('timeout8', 8, '--synthetic-segment-seconds', '0.005', expected=2)
    timed = load(runs/'timeout8/CHECKPOINT.json')
    assert 0 < timed['cursor'] < 1024
    invoke('timeout8', 8)
    rows.append({'name': 'real_elapsed_segment_timeout_and_restore', 'paused_cursor': timed['cursor'],
                 'segment_limit_only_synthetic_test_not_absolute_stage_extension': True,
                 **compare('continuous8', 'timeout8')})

    child = subprocess.Popen(command('signal100', 100), stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
    until = time.monotonic()+20
    signal_cursor = None
    while time.monotonic() < until:
        path = runs/'signal100/STATUS.json'
        if path.exists():
            state = load(path)
            if 0 < state['cursor'] < state['rows']:
                signal_cursor = state['cursor']
                child.send_signal(signal.SIGUSR1)
                break
        if child.poll() is not None:
            break
        time.sleep(0.01)
    out, err = child.communicate(timeout=20)
    (runs/'signal100_interrupted.log').write_text(out+err)
    assert signal_cursor is not None and child.returncode == 2
    paused = load(runs/'signal100/CHECKPOINT.json')
    assert 0 < paused['cursor'] < 12800
    invoke('signal100', 100)
    invoke('continuous100', 100)
    rows.append({'name': 'actual_SIGUSR1_during_native_model_replay_and_fresh_process_restore',
                 'signal_observation_cursor': signal_cursor, 'saved_pause_cursor': paused['cursor'],
                 **compare('continuous100', 'signal100')})

    # Exercise actual worker rejection, retaining each failed fixture and log.
    for name, kind in [('damaged8', 'checksum'), ('wrongidentity8', 'identity')]:
        directory = runs/name
        directory.mkdir()
        value = load(runs/'continuous8/CHECKPOINT.json')
        if kind == 'checksum':
            value['cursor'] -= 1
        else:
            value['identity'] = 'wrong_identity'
            value.pop('payload_sha256')
            value['payload_sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True,
                allow_nan=False, separators=(',', ':')).encode()).hexdigest()
        (directory/'CHECKPOINT.json').write_text(json.dumps(value))
        invoke(name, 8, expected=1)
        assert not (directory/'RESULT.json').exists()
        rows.append({'name': f'actual_worker_rejects_{kind}', 'rejected': True})

    spec = importlib.util.spec_from_file_location('readout_worker_helpers', WORKER)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    fixture = runs/'source_identity_fixture.txt'
    fixture.write_text('before')
    expected = hashlib.sha256(fixture.read_bytes()).hexdigest()
    helpers.validate_files({str(fixture): expected})
    fixture.write_text('after')
    rejected = False
    try:
        helpers.validate_files({str(fixture): expected})
    except ValueError:
        rejected = True
    assert rejected
    rows.append({'name': 'source_file_byte_tamper_refused', 'rejected': True,
                 'original_project_source_unchanged': True})
    accepted = load(runs/'continuous100/RESULT.json')
    result = {'status': 'passed_actual_synthetic_worker_lifecycle_and_integration',
              'checks': rows, 'maximum_shape': {'batch':128,'lookback':480,'hidden':32,'H':60},
              'continuous100_processing_seconds': accepted['totals']['processing_seconds'],
              'continuous100_batches':100, 'subprocess_wall_seconds':wall_times,
              'source_sha256': {'worker':hashlib.sha256(WORKER.read_bytes()).hexdigest(),
                                'lifecycle_verifier':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              'code_freeze_sha256':hashlib.sha256((BASE/'CODE_FREEZE.json').read_bytes()).hexdigest(),
              'market_arrays_or_checkpoint_weights_loaded':False,'GPU':False,'Q3_values_read':False,
              'new_neural_training':False,'predictive_success':False}
    (BASE/'WORKER_RUNTIME_ACCEPTANCE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
