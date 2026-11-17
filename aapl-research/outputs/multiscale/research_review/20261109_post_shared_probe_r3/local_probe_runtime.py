"""Explicitly authorized, resumable fit-only diagnostic; never starts by import.

Preparation needs only the standard library. Torch, original market loaders and
checkpoints are imported/opened only AFTER a valid new authorization is checked.
The old 17:31:49 permission is never reused as a new window.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import signal
import sys
import tempfile
import time

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
STUDY = ROOT/'outputs/multiscale/shared_followup/20261109_gpu_v2'
OLD_DEADLINE = 1794216709.0
STOP = False
FOLDS, SEEDS, FRACTIONS = (1, 2), (2026, 2027, 2028), (0.25, 0.5, 0.75)
SOURCE_FILES = ('local_probe_runtime.py', 'torch_local_probe.py', 'local_update_probe.py',
                'local_probe_supervisor.py', 'NEXT_LOCAL_PROBE_PLAN_ZH.md',
                'NUMERICAL_CONTRACT_REVISION.json')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_manifest():
    """Bind original dependencies and complete model+optimizer checkpoint bytes.

    Hashing opens existing bytes without deserializing or evaluating models.
    No prediction/loss arrays or protected period values are inspected.
    """
    result = {name: sha(BASE/name) for name in SOURCE_FILES}
    rules_path = STUDY/'protocol/RULES.json'
    rules = read(rules_path)
    result[str(rules_path)] = sha(rules_path)
    v3 = ROOT/'outputs/multiscale/mechanism_repair/20261109_v3'
    prior = ROOT/'outputs/multiscale/controlled_followup/20261108_v2'
    frozen = {STUDY/'code'/name: digest for name, digest in rules['sources'].items()}
    frozen.update({v3/name: digest for name, digest in
                   rules['preceding_sources']['new_training'].items()})
    frozen.update({prior/'code'/name: digest for name, digest in
                   rules['preceding_sources']['original_frozen'].items()})
    for path, expected in frozen.items():
        actual = sha(path)
        if actual != expected:
            raise ValueError('An original training/model/loader dependency changed: '+str(path))
        result[str(path)] = actual
    # Bind protocol metadata consumed by the unchanged loader before importing it.
    for path in [prior/'protocol/DATA.json', prior/'protocol/CODE_FREEZE.json',
                 STUDY/'protocol/EVALUATION_FREEZE.json']:
        result[str(path)] = sha(path)
    # Bind the actual admitted pre-Q3 file consumed by Data. The provider raw
    # download is not opened here; its lineage remains in this admission receipt.
    data_protocol = read(prior/'protocol/DATA.json')
    admission_dir = ROOT/'outputs/multiscale/admission/alpaca_20261107T052228573242Z'
    if Path(data_protocol['admission']).resolve() != admission_dir.resolve():
        raise ValueError('The admitted dataset path changed')
    admission_path = admission_dir/'admission.json'
    result[str(admission_path)] = sha(admission_path)
    if result[str(admission_path)] != data_protocol['admission_sha256']:
        raise ValueError('The original dataset admission receipt changed')
    admission = read(admission_path)
    if admission.get('development_cutoff_new_york') != '2026-06-30':
        raise ValueError('The diagnostic only permits the original pre-Q3 dataset')
    bars_path = admission_dir/'bars_minute.npz'
    result[str(bars_path)] = sha(bars_path)
    if result[str(bars_path)] != admission['files']['bars_minute.npz']:
        raise ValueError('The pre-Q3 minute bars changed')
    meta_path = prior/'cache/minute_h60_future_log_return/TASK.json'
    result[str(meta_path)] = sha(meta_path)
    if result[str(meta_path)] != data_protocol['task_files']['minute_h60_future_log_return']:
        raise ValueError('The original task manifest changed')
    metadata = read(meta_path)
    for fold in FOLDS:
        row = metadata['folds'][str(fold)]
        for key in ('split_file', 'scaler_file'):
            path = Path(row[key]); result[str(path)] = sha(path)
        for seed in SEEDS:
            folder = STUDY/'runs'/f'tied_f{fold}_s{seed}'
            done = read(folder/'DONE.json')
            for name in ('DONE.json', 'latest.pt', 'last.pt'):
                path = folder/name; result[str(path)] = sha(path)
            if result[str(folder/'last.pt')] != done['checkpoint_hashes']['last.pt']:
                raise ValueError('Archived final model differs from the completed receipt')
    return result


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
        temporary = handle.name
    os.replace(temporary, path)


def stop(*_):
    global STOP
    STOP = True


def authorization(path, output):
    """Receipt is written only after a direct human approval in this chat."""
    auth = read(path)
    if not (auth.get('authorized') is True and auth.get('direct_human_approval') is True
            and auth.get('thread_id') == '01a1b438-df6f-7f00-9f94-f7d72e9d59f3'
            and auth.get('scope') == '18_fit_only_local_direction_scale_units'
            and auth.get('human_reply_text')):
        raise ValueError('Explicit new human authorization is missing')
    received, deadline = float(auth['received_epoch']), float(auth['deadline_epoch'])
    if not (auth.get('new_window_explicitly_approved') is True
            and deadline != OLD_DEADLINE and 0 < deadline-received <= 1800
            and received <= time.time() < deadline):
        raise ValueError('The original or an expired/reset runtime window is not valid')
    if auth.get('GPU') is not False or auth.get('maximum_processes') != 1 or (
            auth.get('maximum_logical_CPUs') != 2) or auth.get('paid_resources') is not False:
        raise ValueError('Resource authorization differs from the proposed diagnostic')
    if Path(auth['output_directory']).resolve() != output.resolve():
        raise ValueError('Output directory differs from the frozen authorization')
    if auth.get('source_sha256') != source_manifest():
        raise ValueError('Source changed after new authorization was frozen')
    return auth


def prepare():
    return {'status': 'source_prepared_not_model_tested_or_market_admitted',
            'source_sha256': source_manifest(),
            'planned_units': 18, 'folds': list(FOLDS), 'seeds': list(SEEDS),
            'fit_positions': list(FRACTIONS), 'GPU': False,
            'new_market_probe_started': False, 'research_goal_achieved': False}


def require_supervisor():
    parent = os.getppid()
    if os.environ.get('AAPL_LOCAL_PROBE_SUPERVISOR_PID') != str(parent):
        raise ValueError('Use the external deadline supervisor; direct execution is refused')
    command = Path(f'/proc/{parent}/cmdline').read_bytes().split(b'\0')
    if str(BASE/'local_probe_supervisor.py').encode() not in command:
        raise ValueError('The claimed parent is not the frozen deadline supervisor')


def synthetic(auth_path, output):
    """Real native-model fixtures; must run only in the newly approved window."""
    require_supervisor()
    auth = authorization(auth_path, output)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1' or not set(
            os.sched_getaffinity(0)).issubset({24, 26}):
        raise ValueError('Synthetic acceptance must use the authorized CPU allocation')
    import numpy as np
    import torch
    from copy import deepcopy
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0, str(STUDY/'code'))
    import experiment as original
    from torch_local_probe import local_unit, make_cpu_adam, state_fingerprint
    def guard():
        if STOP or time.time() >= float(auth['deadline_epoch']):
            raise InterruptedError('Synthetic acceptance reached the authorized boundary')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, stop)
    torch.manual_seed(2026)
    scaler = {'y_mean': [0.0]*60, 'y_std': [1.0]*60}
    net = original.Bundle(480, scaler, 2026, True)
    args = (torch.randn(128, 480, 1), torch.randn(128, 1))
    truth = torch.randn(128, 60)
    diag_args = (torch.randn(128, 480, 1), torch.randn(128, 1))
    diag_truth = torch.randn(128, 60)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, fused=False, foreach=False)
    zero = local_unit(net, optimizer.state_dict(), args, truth, diag_args, diag_truth, guard)
    if zero['eligibility'] != 'degenerate_both_updates_zero':
        raise ValueError('Zero-head initial encoder null was not reproduced')
    net.train(); optimizer.zero_grad()
    prediction = net(*args)
    loss = sum((prediction[h]-truth[:, :h]).square().mean() for h in original.HORIZONS)
    loss.backward(); optimizer.step()
    saved = deepcopy(optimizer.state_dict())
    result = local_unit(net, saved, args, truth, diag_args, diag_truth, guard)
    if result['eligibility'] not in ('eligible',
            'ineligible_projection_effect_not_separated_from_numerical_null') or any(
            result['actual_update_norms'][key] <= 0 for key in ('A', 'B', 'C')):
        raise ValueError('The native nonzero fixture did not cover eligible A/B/C updates')
    json.dumps(result, allow_nan=False)  # The entire unit must be JSON serializable.
    # Independent ordinary total-backward A route, on a fresh clone.
    manual = deepcopy(net); manual.train()
    manual_opt = make_cpu_adam(manual, saved)
    manual_opt.zero_grad(); p = manual(*args)
    sum((p[h]-truth[:, :h]).square().mean() for h in original.HORIZONS).backward()
    manual_opt.step()
    if result['A_model_state_sha256'] != state_fingerprint(manual.state_dict()) or (
            result['A_optimizer_state_sha256'] != state_fingerprint(manual_opt.state_dict())):
        raise ValueError('Synthetic A differs from the ordinary backward parameter/Adam state')
    manual.eval()
    with torch.no_grad():
        p = manual(*diag_args)
        for h in original.HORIZONS:
            e = (p[h]-diag_truth[:, :h]).double()
            actual = result['metrics']['A'][str(h)]
            if not np.isclose(actual['standardized_path_mse'], float(e.square().mean()),
                              rtol=2e-5, atol=1e-7) or not np.isclose(
                    actual['standardized_endpoint_mse'], float(e[:, -1].square().mean()),
                    rtol=2e-5, atol=1e-7):
                raise ValueError('Synthetic A does not restore the ordinary total-backward route')
    guard()
    atomic(output/'TORCH_SYNTHETIC_ACCEPTANCE.json', {
        'status': 'passed', 'synthetic_only': True, 'source_sha256': auth['source_sha256'],
        'authorization_sha256': sha(auth_path), 'market_data_loaded': False,
        'original_market_checkpoints_deserialized': False, 'GPU': False,
        'native_batch_shape': [128, 480, 1],
        'checks': ['zero_head_encoder_null', 'nonzero_head_model_unit',
                   'A_total_backward_model_and_Adam_state_exact',
                   'A_path_and_endpoint_metrics', 'common_BN_and_nonpersistent_EMA',
                   'head_update_equality', 'nonzero_float32_C_norm_direction_eligibility'],
        'not_covered': ['all_float32_degenerate_boundary_fixtures',
                        'corrupt_checkpoint_rejection_fixtures', 'fused_GPU_arithmetic'],
        'nonzero_case_eligibility': result['eligibility'],
        'synthetic_projection_effect_identifiable': result['eligibility'] == 'eligible',
        'fused_GPU_bitwise_equivalence_claimed': False})


def run(auth_path, output):
    # No heavyweight import or data/checkpoint access can precede this gate.
    worker_stage_started = time.monotonic()
    require_supervisor()
    auth = authorization(auth_path, output)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1' or not set(
            os.sched_getaffinity(0)).issubset({24, 26}):
        raise ValueError('Require CUDA disabled and CPU affinity within24,26')
    deadline = float(auth['deadline_epoch'])
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, stop)

    def guard():
        if STOP or time.time() >= deadline:
            raise InterruptedError('Stop at the newly authorized boundary')

    guard()
    output.mkdir(parents=True, exist_ok=True)
    # This readiness receipt must come from a separately executed synthetic
    # REAL-MODEL acceptance within the same newly approved window, not static review.
    acceptance_path = output/'TORCH_SYNTHETIC_ACCEPTANCE.json'
    acceptance = read(acceptance_path)
    if acceptance.get('status') != 'passed' or acceptance.get('synthetic_only') is not True:
        raise ValueError('Real-model synthetic acceptance has not passed')
    if acceptance.get('source_sha256') != auth['source_sha256']:
        raise ValueError('Model acceptance is bound to different code')
    if acceptance.get('authorization_sha256') != sha(auth_path):
        raise ValueError('Model acceptance belongs to a different authorization')
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0, str(STUDY/'code'))
    import experiment as original
    from torch_local_probe import local_unit
    done_rows, durations = [], []
    source_identity = dict(auth['source_sha256'])
    auth_hash = sha(auth_path)

    def progress(status, **extra):
        atomic(output/'STATE.json', {'status': status, 'completed_units': len(done_rows),
            'planned_units': 18, 'completed': done_rows, 'deadline_epoch': deadline,
            'authorization_sha256': auth_hash, 'source_sha256': source_identity,
            'Q3_values_read': False, 'stop_or_outer_scored': False,
            'research_goal_achieved': False, 'independent_confirmation': False, **extra})

    # Original loader reads previously exposed complete pre-Q3 bars and split
    # metadata, including validation indices. Only FIT data get targets/forwards.
    # Access is reported honestly; this is not an assertion of fit-only disk reads.
    try:
        for fold in FOLDS:
            guard()
            data = original.data_for(fold)
            if str(data.bars.session_date[-1]) > '2026-06-30':
                raise ValueError('Unexpected post-June market bars')
            count = len(data.splits['fit'])
            if count < 256:
                raise ValueError('Insufficient frozen fit origins')
            for seed in SEEDS:
                folder = STUDY/'runs'/f'tied_f{fold}_s{seed}'
                receipt = read(folder/'DONE.json')
                checkpoint = torch.load(folder/'latest.pt', map_location='cpu', weights_only=True)
                last = torch.load(folder/'last.pt', map_location='cpu', weights_only=True)
                if checkpoint['identity'] != receipt['identity'] or (
                        checkpoint['state']['epoch'] != 20 or receipt['fixed_epochs'] != 20
                        or checkpoint['state']['steps'] != receipt['steps']
                        or checkpoint['scaler'] != data.scaler):
                    raise ValueError('Model, optimizer and data identities do not match')
                if checkpoint['model'].keys() != last['model'].keys() or any(
                        not torch.equal(checkpoint['model'][name], last['model'][name])
                        for name in last['model']):
                    raise ValueError('Latest paired optimizer state does not carry the frozen last model')
                net = original.Bundle(480, data.scaler, seed, True)
                net.load_state_dict(checkpoint['model'])
                names = list(dict(net.named_parameters()))
                if names != [name for name in checkpoint['model'] if name in set(names)]:
                    raise ValueError('Model parameter name registration order changed')
                moments = checkpoint['optimizer']['state']
                if set(moments) != set(range(len(names))):
                    raise ValueError('The final market checkpoint requires every Adam history')
                for state in moments.values():
                    step = state['step']
                    if not isinstance(step, torch.Tensor) or step.numel() != 1 or (
                            not torch.isfinite(step).all() or float(step) != receipt['steps']):
                        raise ValueError('Adam moment steps do not exactly match the model')
                checkpoint_hash = sha(folder/'latest.pt')
                for fraction in FRACTIONS:
                    guard()
                    identifier = f'f{fold}_s{seed}_q{int(100*fraction)}'
                    path = output/'units'/(identifier+'.json')
                    if path.exists():
                        previous = read(path)
                        if previous['authorization_sha256'] != auth_hash or (
                                previous['source_sha256'] != source_identity or
                                previous['checkpoint_sha256'] != checkpoint_hash):
                            raise ValueError('Cannot resume a different probe identity')
                        done_rows.append(identifier)
                        if len(done_rows) == 1:
                            admission = read(output/'ADMISSION.json')
                            if admission.get('status') != 'admitted' or (
                                    admission.get('authorization_sha256') != auth_hash or
                                    admission.get('source_sha256') != source_identity):
                                raise ValueError('Resume requires the same passed complete-matrix admission')
                        continue
                    start = int(np.floor((count-256)*fraction))
                    update_positions = np.arange(start, start+128)
                    diagnostic_positions = np.arange(start+128, start+256)
                    args, truth = data.batch('fit', update_positions)
                    diag_args, diag_truth = data.batch('fit', diagnostic_positions)
                    before = time.monotonic()
                    def unit_guard():
                        guard()
                        if not done_rows and time.monotonic()-before >= 120:
                            raise InterruptedError('Initial fixed-shape120second admission limit')
                    unit = local_unit(net, checkpoint['optimizer'], args, truth,
                                      diag_args, diag_truth, unit_guard)
                    elapsed = time.monotonic()-before
                    # First unit is also the fixed-shape admission short test.
                    # Its result is retained whether or not the full matrix fits.
                    durations.append(elapsed)
                    unit.update(id=identifier, fold=fold, seed=seed, fraction=fraction,
                        start=start, checkpoint_sha256=checkpoint_hash,
                        authorization_sha256=auth_hash, source_sha256=source_identity,
                        wall_seconds=elapsed, new_candidate_weights_saved=False)
                    atomic(path, unit)
                    done_rows.append(identifier)
                    progress('running', raw_bar_access='previously_exposed_complete_pre_Q3_file',
                             evaluation_parts=['fit'], optimizer='CPU nonfused FP32')
                    if len(done_rows) == 1:
                        # Conservatively repeat the WHOLE observed first stage,
                        # including imports, provenance, data/checkpoint setup.
                        # This bounds unmeasured setup for later folds/seeds;
                        # the external first-stage watchdog starts before Popen.
                        stage_elapsed = time.monotonic()-worker_stage_started
                        remaining_projection = stage_elapsed*17*1.25+120
                        admitted = stage_elapsed <= 120 and remaining_projection <= deadline-time.time()
                        atomic(output/'ADMISSION.json', {
                            'status': 'admitted' if admitted else 'not_admitted',
                            'authorization_sha256': auth_hash,
                            'source_sha256': source_identity,
                            'first_unit_seconds': elapsed,
                            'first_stage_seconds_including_setup': stage_elapsed,
                            'projection_basis': '17 times complete first-stage elapsed, 25percent margin, 120second reserve',
                            'projected_remaining_seconds': remaining_projection})
                        if not admitted:
                            progress('not_admitted_for_complete18',
                                     projected_remaining_seconds=remaining_projection)
                            return
                del net, checkpoint
            del data
        progress('completed_all18_local_diagnostic_units',
                 model_weights_saved=False, predictive_candidate_selected=False,
                 computational_diagnostic_not_market_confirmation=True)
    except InterruptedError as exc:
        progress('partial_at_authorized_boundary', reason=str(exc))
    except BaseException as exc:
        progress('failed_existing_units_preserved', error=repr(exc))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--worker', choices=('synthetic', 'market'))
    parser.add_argument('--authorization', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare(), indent=2))
        return
    if args.authorization is None or args.output is None:
        parser.error('No market operation is allowed without explicit new authorization and output')
    if args.worker == 'synthetic':
        synthetic(args.authorization, args.output)
    elif args.worker == 'market':
        run(args.authorization, args.output)
    else:
        parser.error('Use local_probe_supervisor.py; no unsupervised market execution')


if __name__ == '__main__':
    main()
