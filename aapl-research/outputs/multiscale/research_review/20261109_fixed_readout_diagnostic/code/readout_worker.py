"""Fixed LSTM fit-only diagnostic with atomic QR state; no optimization or holdout."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import sys
import tempfile
import time

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]
STOP = False


def stop(*_):
    global STOP
    STOP = True


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def state_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
        name = stream.name
    os.replace(name, path)


def validate_files(files):
    for path, expected in files.items():
        if sha(path) != expected:
            raise ValueError('Frozen input identity changed: '+path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['synthetic', 'market'], required=True)
    parser.add_argument('--state-id')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--synthetic-batches', type=int, default=8)
    parser.add_argument('--pause-after-batches', type=int)
    parser.add_argument('--synthetic-segment-seconds', type=float)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    assert set(os.sched_getaffinity(0)) == {24}
    for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGUSR1]:
        signal.signal(sig, stop)
    scope = load(BASE/'STAGE_SCOPE.json')
    freeze = load(BASE/'CODE_FREEZE.json')
    validate_files(freeze['files'])
    deadline = scope['deadline_epoch']
    assert time.time() < deadline
    run_dir = args.run_dir.resolve()
    run_dir.relative_to(BASE)
    run_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from readout_qr import ReadoutQR
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    source = None
    if args.mode == 'market':
        assert args.synthetic_segment_seconds is None and args.pause_after_batches is None
        admission = load(BASE/'MARKET_ADMISSION.json')
        assert admission['status'] == 'admitted_four_fit_only_states'
        assert admission['code_freeze_sha256'] == sha(BASE/'CODE_FREEZE.json')
        assert admission['scope_sha256'] == sha(BASE/'STAGE_SCOPE.json')
        inputs = load(BASE/'INPUT_IDENTITIES.json')
        assert admission['input_identities_sha256'] == sha(BASE/'INPUT_IDENTITIES.json')
        validate_files(inputs['verified_files'])
        source = next(s for s in inputs['states'] if s['id'] == args.state_id)
        # All frozen source/data/checkpoint identities are checked before deserialization.
        sys.path.insert(0, inputs['reference_root'])
        from multiscale.bars import Bars, targets
        from multiscale.features import CloseWindows
        from multiscale.models import make_model
        for module in ['multiscale.bars', 'multiscale.features', 'multiscale.models']:
            Path(sys.modules[module].__file__).resolve().relative_to(Path(inputs['reference_root']).resolve())
        with np.load(source['split_file'], allow_pickle=False) as archive:
            fit = archive['fit']
        assert fit.dtype.kind == 'i' and fit.ndim == 1 and len(fit) == source['fit_count']
        assert (np.diff(fit) > 0).all()
        assert hashlib.sha256(np.asarray(fit, dtype='<i8').tobytes()).hexdigest() == source['fit_array_sha256']
        with np.load(source['source_bars'], allow_pickle=False) as archive:
            assert set(archive.files) == {'start_ns', 'end_ns', 'close', 'observed', 'session_date'}
            arrays = {name: archive[name] for name in archive.files}
        assert arrays['session_date'].max() <= '2026-06-30'
        cutoff = np.datetime64('2026-07-01T04:00:00', 'ns').astype('int64')
        assert (arrays['end_ns'][arrays['observed']] < cutoff).all()
        bars = Bars(source['cfg']['frequency'], **arrays)
        source_admission = load(inputs['source_admission_path'])
        windows = CloseWindows(bars, source['cfg']['lookback'], source_admission['split_actions'])
        cfg = source['cfg']
        assert cfg['model'] == 'lstm' and cfg['variant'] == 'anchor_baseline'
        assert cfg['anchor'] == 'observed'
        assert cfg['output'] == 'return' and cfg['target'] == 'future_log_return'
        assert cfg['loss'] == 'path' and cfg['seed'] == 2026 and cfg['fold'] in [1, 2]
        assert fit[0] >= cfg['lookback']-1 and fit[-1]+cfg['horizon'] < len(bars.close)
        saved = torch.load(source['checkpoint'], map_location='cpu', weights_only=True)
        assert saved['identity'] == source['training_identity']
        assert saved['cfg'] == cfg and saved['epoch'] == source['best_epoch']
        assert saved['scaler'] == source['scaler_original']
        net = make_model('lstm', cfg['lookback'], cfg['horizon'])
        net.load_state_dict(saved['model'], strict=True)
        assert saved['parameters'] == sum(p.numel() for p in net.parameters()) == load(source['done'])['parameters']
        net.eval()
        scaler = saved['scaler']
        h, lookback, n = cfg['horizon'], cfg['lookback'], len(fit)
        ym, ys = np.asarray(scaler['y_mean']), np.asarray(scaler['y_std'])
        zero_prediction = -ym/ys

        def batch(start, end):
            indices = fit[start:end]
            x, a = windows.arrays(indices)
            x = ((x-scaler['x_mean'])/scaler['x_std']).astype('float32')
            a = ((a-scaler['anchor_mean'])/scaler['anchor_std']).astype('float32')
            y = targets(bars, indices, h, cfg['target'])
            y = ((y-ym)/ys).astype('float32')
            return torch.from_numpy(x[:, :, None]), torch.from_numpy(a), y.astype('float64')

        provenance = {'state_id': source['id'], 'input_identities_sha256': sha(BASE/'INPUT_IDENTITIES.json'),
                      'training_identity': saved['identity'], 'selected_epoch': saved['epoch'],
                      'mode': 'original_fit_only', 'full_preQ3_bars_loaded': True,
                      'split_array_members_loaded': ['fit'], 'stop_outer_targets_or_scores_read': False}
    else:
        assert not args.state_id
        assert 1 <= args.synthetic_batches <= 100
        reference = ROOT/'outputs/multiscale/controlled_followup/20261108_v2/code/reference_source'
        sys.path.insert(0, str(reference))
        from multiscale.models import make_model
        torch.manual_seed(2026)
        h, lookback = 60, 480
        net = make_model('lstm', lookback, h).eval()
        n = args.synthetic_batches*128
        zero_prediction = np.zeros(h)

        def batch(start, end):
            generator = torch.Generator().manual_seed(152611+start)
            x = torch.randn(end-start, lookback, 1, generator=generator)
            a = torch.randn(end-start, 1, generator=generator)
            y = torch.randn(end-start, h, generator=generator).numpy().astype('float64')
            return x, a, y

        provenance = {'mode': 'synthetic_maximum_original_shape', 'synthetic_rows': n,
                      'market_arrays_or_weights_loaded': False}

    for param in net.parameters():
        param.requires_grad_(False)
    assert not net.training and not any(p.requires_grad for p in net.parameters())
    b0 = np.concatenate([net.head.weight.detach().numpy().T,
                         net.head.bias.detach().numpy()[None, :]], axis=0).astype('float64')
    assert b0.shape == (34, h)
    binding = {'provenance': provenance, 'scope_sha256': sha(BASE/'STAGE_SCOPE.json'),
               'code_freeze_sha256': sha(BASE/'CODE_FREEZE.json'), 'rows': n,
               'horizon': h, 'lookback': lookback, 'batch_size': 128,
               'stage_deadline_epoch': deadline}
    identity = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    checkpoint = run_dir/'CHECKPOINT.json'
    if checkpoint.exists():
        state = load(checkpoint)
        checksum = state.pop('payload_sha256', None)
        if checksum != state_digest(state):
            raise ValueError('Corrupt diagnostic checkpoint payload')
        if state['identity'] != identity or state['binding'] != binding:
            raise ValueError('Diagnostic checkpoint identity changed')
        qr = ReadoutQR.from_state(state['qr'], expected_identity=identity, expected_n=n)
        cursor = state['cursor']
        assert state['qr']['n_rows'] == cursor and 0 <= cursor <= n
        totals = state['totals']
    else:
        qr = ReadoutQR(h, expected_n=n, identity=identity)
        cursor = 0
        totals = {'maximum_prediction_reconstruction_scaled_error': 0.0,
                  'native_fp32_training_target_SSE': 0.0, 'zero_prediction_SSE': 0.0,
                  'maximum_full_model_feature_route_error': 0.0, 'processed_batches': 0,
                  'processing_seconds': 0.0}

    def save(status):
        payload = {'identity': identity, 'binding': binding, 'cursor': cursor,
                   'qr': qr.state_dict(), 'totals': totals}
        payload['payload_sha256'] = state_digest(payload)
        atomic(checkpoint, payload)
        atomic(run_dir/'STATUS.json', {'status': status, 'identity': identity, 'cursor': cursor,
              'rows': n, 'stage_deadline_epoch': deadline, 'processed_batches': totals['processed_batches'],
              'processing_seconds': totals['processing_seconds'], 'GPU': False,
              'stop_outer_scored': False, 'Q3_values_read': False})

    save('running')
    segment_start = time.time()
    segment_batches = 0
    while cursor < n:
        if (STOP or time.time() >= deadline or
            args.synthetic_segment_seconds is not None and time.time()-segment_start >= args.synthetic_segment_seconds or
            args.pause_after_batches is not None and segment_batches >= args.pause_after_batches):
            save('paused_atomic_state_saved')
            print(json.dumps({'status': 'paused', 'cursor': cursor, 'identity': identity}), flush=True)
            return 2
        batch_start = time.perf_counter()
        end = min(cursor+128, n)
        x, anchor, y = batch(cursor, end)
        with torch.inference_mode():
            hidden = net.rnn(x)[0][:, -1]
            features = torch.cat([hidden, anchor], dim=1)
            native = net.head(features).numpy().astype('float64')
            if cursor == 0:
                full = net(x, anchor).numpy().astype('float64')
                route_error = float(np.max(np.abs(full-native)))
                totals['maximum_full_model_feature_route_error'] = route_error
                if route_error != 0:
                    raise ValueError('Original native full forward differs from frozen feature extraction')
        design = np.concatenate([features.numpy().astype('float64'), np.ones((end-cursor, 1))], axis=1)
        rebuilt = design@b0
        denominator = max(1.0, float(np.abs(native).max()), float((np.abs(design)@np.abs(b0)).max()))
        reconstruction = float(np.abs(rebuilt-native).max())/denominator
        if reconstruction > 64*np.finfo('float32').eps:
            raise ValueError('Original head reconstruction failed')
        totals['maximum_prediction_reconstruction_scaled_error'] = max(
            totals['maximum_prediction_reconstruction_scaled_error'], reconstruction)
        totals['native_fp32_training_target_SSE'] += float(np.square(native-y).sum())
        totals['zero_prediction_SSE'] += float(np.square(y-zero_prediction).sum())
        qr.update(design, y)
        cursor = end
        totals['processed_batches'] += 1
        segment_batches += 1
        totals['processing_seconds'] += time.perf_counter()-batch_start
        save('running')
    assert time.time() < deadline, 'Deadline reached before final diagnostic solve; checkpoint is resumable'
    solved = qr.solve(b0, zero_prediction=zero_prediction)
    zero_sse_delta = abs(solved['SSE_zero_baseline']-totals['zero_prediction_SSE'])/max(
        1.0, solved['SSE_zero_baseline'], totals['zero_prediction_SSE'])
    assert zero_sse_delta <= 1e-10, 'Streamed zero loss differs from compressed-QR loss'
    result = {'status': 'completed', 'identity': identity, 'binding': binding,
              'totals': totals, 'solution': solved,
              'direct_vs_compressed_zero_SSE_relative_difference': zero_sse_delta,
              'native_fp32_vs_FP64_affine_head_MSE_difference':
                  totals['native_fp32_training_target_SSE']/(n*h)-solved['MSE_original'],
              'market_fit_replay': args.mode == 'market', 'new_neural_training': False,
              'candidate_weights_saved': False, 'new_forecasting_candidate_admitted': False,
              'stop_outer_scored': False, 'Q3_values_read': False, 'GPU': False,
              'research_goal_achieved': False, 'independent_market_confirmation': False}
    atomic(run_dir/'RESULT.json', result)
    save('completed')
    print(json.dumps({'status': 'completed', 'rows': n, 'horizon': h,
                      'processing_seconds': totals['processing_seconds'], 'result': str(run_dir/'RESULT.json')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
