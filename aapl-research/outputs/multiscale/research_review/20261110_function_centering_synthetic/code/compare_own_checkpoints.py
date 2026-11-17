"""Independently compare only this stage's three synthetic checkpoint payloads."""
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time

STAGE = Path(__file__).resolve().parents[1]


def main():
    protocol_path = STAGE / 'COMPARISON_PROTOCOL.json'
    protocol = json.loads(protocol_path.read_text())
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == protocol['script_sha256']
    assert time.time() < protocol['deadline_epoch']
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    assert set(os.sched_getaffinity(0)) <= {24}
    repository = next(p for p in STAGE.parents if (p / 'MULTISCALE_CURRENT_PLAN_ZH.md').exists())
    runtime = repository / '.venv'

    def audit(event, values):
        if event.startswith('socket.') and event in ('socket.connect', 'socket.bind', 'socket.getaddrinfo'):
            raise PermissionError('No network')
        if event != 'open' or not values or isinstance(values[0], int):
            return
        path = Path(os.fsdecode(values[0])).resolve()
        if path.is_relative_to(repository) and not (path.is_relative_to(STAGE) or path.is_relative_to(runtime)):
            raise PermissionError('Comparator may only open own synthetic inputs in repository')

    sys.addaudithook(audit)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    payloads = {}
    checkpoints = {}
    for name in ('continuous', 'signal', 'deadline'):
        path = STAGE / 'runs' / name / 'checkpoint.pt'
        checkpoints[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        payloads[name] = torch.load(path, map_location='cpu', weights_only=True)
    checks = []

    def compare(left, right, path):
        assert type(left) is type(right), path + ': type'
        if isinstance(left, torch.Tensor):
            assert left.dtype == right.dtype and left.shape == right.shape, path + ': dtype/shape'
            # Compare bytes, preserving signed zero and every stored floating bit.
            a = left.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            b = right.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            assert torch.equal(a, b), path + ': tensor bytes'
            checks.append({'path': path, 'kind': 'tensor_raw_bytes', 'dtype': str(left.dtype),
                           'shape': list(left.shape), 'bytes': a.numel(), 'passed': True})
        elif isinstance(left, dict):
            assert left.keys() == right.keys(), path + ': dictionary keys'
            for key in left:
                compare(left[key], right[key], path + '/' + repr(key))
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right), path + ': sequence length'
            for index, (a, b) in enumerate(zip(left, right)):
                compare(a, b, path + '/' + str(index))
        elif isinstance(left, float):
            assert struct.pack('!d', left) == struct.pack('!d', right), path + ': float bits'
            checks.append({'path': path, 'kind': 'float_bits', 'passed': True})
        else:
            assert left == right, path + ': scalar'
            checks.append({'path': path, 'kind': 'typed_scalar', 'passed': True})

    for name in ('signal', 'deadline'):
        compare(payloads['continuous'], payloads[name], 'continuous_vs_' + name)
    result = {'status': 'passed_independent_full_synthetic_payload_byte_comparison',
              'checkpoint_files': checkpoints, 'checks_passed': len(checks), 'checks': checks,
              'comparison_algorithm_imports_worker': False,
              'all_recursive_payload_members_compared': True,
              'serialized_container_SHA_required_identical': False,
              'only_own_new_synthetic_checkpoints_loaded': True,
              'market_arrays_or_archived_weights_loaded': False, 'Q3_values_read': False,
              'independent_market_confirmation': False, 'full_goal_achieved': False}
    destination = STAGE / 'review/INDEPENDENT_CHECKPOINT_COMPARISON.json'
    temporary = destination.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    os.replace(temporary, destination)
    print(json.dumps({k: v for k, v in result.items() if k != 'checks'}, indent=2))


if __name__ == '__main__':
    main()
