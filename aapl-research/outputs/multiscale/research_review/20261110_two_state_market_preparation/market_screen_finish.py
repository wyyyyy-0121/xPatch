"""One frozen post-training controls/calibration/test/statistics pass.

Requires all twelve completed paired Q/S packages. No candidate search, queue
extension, old saved predictions, original stop/outer partitions or Q3 access.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from common_adapter import access, apply_alpha
from fixed_controls import fit_controls, predict_control
from market_reader import MarketInputs
from screen_evaluation import origin_losses, aggregate_primary

BASE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    tmp = path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with tmp.open('x') as s:
        json.dump(value, s, indent=2, allow_nan=False); s.write('\n')
        s.flush(); os.fsync(s.fileno())
    tmp.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def save_arrays(path, arrays):
    tmp = path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with tmp.open('xb') as s:
        np.savez_compressed(s, **arrays); s.flush(); os.fsync(s.fileno())
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args(); began = time.time(); body = args.protocol.read_bytes(); p = json.loads(body)
    protocol_sha = hashlib.sha256(body).hexdigest()
    if sorted(os.sched_getaffinity(0)) != [24, 26] or os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
        raise ValueError('Analysis CPU24/26 and GPU-hidden startup required')
    if any(os.environ.get(k) != '1' for k in
           ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')):
        raise ValueError('Analysis numerical thread environment')
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    for ref in p['locks']:
        if sha(ref['path']) != ref['sha256']: raise ValueError('Source drift: '+ref['path'])
    if not any(Path(r['path']).resolve() == Path(__file__).resolve() for r in p['locks']):
        raise ValueError('Postprocessor must be frozen')
    output = Path(p['output_directory']); out = output/'postprocessing'
    if out.exists(): raise ValueError('No automatic postprocessing repeat or overwrite')
    out.mkdir()
    inputs = MarketInputs(args.protocol)
    admission = json.loads((output/'REMAINING_QUEUE_ADMISSION.json').read_text())
    if admission['protocol_sha256'] != protocol_sha or admission['accepted'] is not True:
        raise ValueError('No same-window remaining queue admission')
    adapters = [inputs.adapter(s.task, s.fold) for s in inputs.states]
    for a in adapters:
        inputs.guard(); a.fit_scaler(access(a.state, 'Q', 'fit_scaler'))
    spec = importlib.util.spec_from_file_location('frozen_postprocessing_prototype', p['prototype_path'])
    prototype = importlib.util.module_from_spec(spec); spec.loader.exec_module(prototype)
    manifest = {}; files = {}; neural = {}; controls = {}
    for a in adapters:
        state_dir = out/a.state.state_uid; state_dir.mkdir()
        atomic(state_dir/'STATE_SCALER.json', {'state': a.state.as_dict(), 'roles_id': a.roles_id,
                                              'scaler_id': a.scaler.scaler_id, 'scaler': asdict(a.scaler)})
        for seed in (2026, 2027, 2028):
            jobs = [j for j in p['training_jobs'] if
                    (j['task'], j['fold'], j['seed']) == (a.state.task, a.state.fold, seed)]
            if len(jobs) != 1: raise ValueError('Exact state/seed job binding')
            directory = output/'pairs'/jobs[0]['job_id']
            done = json.loads((directory/'DONE.json').read_text())
            if done['status'] != 'Q_S_pair_complete' or done['exit_code'] != 0:
                raise ValueError('All paired fits must be complete before calibration')
            for arm in ('C', 'R'):
                path = directory/('best_'+arm+'.pt'); digest = sha(path)
                if done['best_inference'][arm]['sha256'] != digest:
                    raise ValueError('Completed best artifact checksum')
                best = torch.load(path, map_location='cpu', weights_only=True)
                if (best['kind'] != 'pair_best_inference_only' or best['resumable'] is not False
                        or best['state'] != a.state.as_dict() or best['arm'] != arm or best['seed'] != seed
                        or best['scaler_id'] != a.scaler.scaler_id or best['scaler'] != asdict(a.scaler)
                        or best['identity']['roles_id'] != a.roles_id
                        or best['identity']['absolute_deadline'] != p['deadline_epoch']
                        or best['identity']['caller']['data']['protocol_sha256'] != protocol_sha):
                    raise ValueError('Frozen inference model/scaler/source identity')
                mid = a.state.state_uid+'/'+arm+':s'+str(seed)
                model = prototype.TwoStateGRU(a.H, torch.tensor(a.scaler.zero_scaled.copy()))
                model.load_state_dict(best['model'], strict=True); model.eval()
                neural[mid] = model; files[mid] = {'path': str(path), 'sha256': digest}
        inputs.guard(); fitted = fit_controls(a, inputs.guard)
        for name, tag in (('E_Ridge', 'E_Ridge'), ('Q_mean', 'mean')):
            mid = a.state.state_uid+'/'+tag; path = state_dir/(tag+'.npz')
            metadata = {'state': a.state.as_dict(), 'roles_id': a.roles_id, 'scaler_id': a.scaler.scaler_id,
                        'model_id': mid, 'protocol_sha256': protocol_sha}
            save_arrays(path, {**fitted[name], 'identity_json': np.asarray(json.dumps(metadata, sort_keys=True))})
            controls[mid] = (name, fitted[name]); files[mid] = {'path': str(path), 'sha256': sha(path)}
        for mid in inputs.gate.model_ids(a.state):
            manifest[mid] = {'state': a.state.as_dict(), 'roles_id': a.roles_id,
                             'scaler_id': a.scaler.scaler_id, 'model_sha256': files[mid]['sha256']}
    if len(manifest) != 32: raise ValueError('Global 32 fit records incomplete')
    for ref in files.values():
        if sha(ref['path']) != ref['sha256']: raise ValueError('Model changed before global freeze')
    freeze = inputs.gate.freeze_models(manifest)
    atomic(out/'GLOBAL_MODELS_FROZEN.json', {'protocol_sha256': protocol_sha, 'manifest': manifest,
           'files': files, 'API_structural_receipt': freeze, 'actual_file_hashes_verified': True})

    def predict(a, mid, role):
        inputs.guard()
        if sha(files[mid]['path']) != files[mid]['sha256']:
            raise ValueError('Frozen model changed before inference')
        origins = a.roles[role]; prediction = np.empty((len(origins), a.H), dtype=np.float64)
        for begin in range(0, len(origins), 128):
            inputs.guard(); o = origins[begin:begin+128]
            if mid in neural:
                v = a.network_inputs(access(a.state, role, 'inputs'), o)
                with torch.inference_mode():
                    scaled = neural[mid](torch.tensor(v['x'].copy()), torch.tensor(v['boundary'].copy()),
                        torch.tensor(v['anchor'].copy()), torch.tensor(v['context'].copy()),
                        reset_flag=int('/R:' in mid))['output'].numpy()
                raw = a.scaler.inverse(a.state, scaled)
            else:
                name, model = controls[mid]; raw = predict_control(name, model, a, role, o)
            prediction[begin:begin+len(o)] = raw
        return prediction

    alpha_manifest = {}
    for a in adapters:
        for mid in inputs.gate.model_ids(a.state):
            prediction = predict(a, mid, 'C_cal')
            record = a.calibrate(access(a.state, 'C_cal', 'alpha'), mid, prediction)
            save_arrays(out/a.state.state_uid/('Ccal_'+mid.split('/')[-1].replace(':', '_')+'.npz'),
                        {'origins': a.roles['C_cal'], 'raw_prediction': prediction})
            alpha_manifest[mid] = record
    alpha_receipt = inputs.gate.freeze_alphas(alpha_manifest)
    atomic(out/'GLOBAL_ALPHAS_FROZEN.json', {'protocol_sha256': protocol_sha,
           'manifest': alpha_manifest, 'API_structural_receipt': alpha_receipt})
    scored = []
    for a in adapters:
        predictions = {}; alphas = {}; payload = {'origins': a.roles['T']}
        for mid in inputs.gate.model_ids(a.state):
            tag = mid.split('/')[-1]
            name = ('Q_mean' if tag == 'mean' else tag.replace(':s', '_seed'))
            raw = predict(a, mid, 'T'); predictions[name] = raw; alphas[name] = alpha_manifest[mid]['alpha']
            apply_alpha(alpha_manifest[mid], a.state, mid, raw)  # Validate the frozen alpha identity/state.
            payload['raw_'+name] = raw
        truth = a.labels(access(a.state, 'T', 'labels'), a.roles['T']); payload['truth'] = truth
        losses = origin_losses(truth, predictions, alphas)
        for name, row in losses.items():
            for metric, values in row.items(): payload['loss_'+name+'_'+metric] = values
        save_arrays(out/a.state.state_uid/'T_PREDICTIONS_AND_LOSSES.npz', payload)
        scored.append({'task': a.state.task, 'fold': a.state.fold, 'state_uid': a.state.state_uid,
                       'session_date': inputs.bars('minute' if a.state.task.startswith('minute') else 'hour')['session_date'],
                       'origins': a.roles['T'], 'losses': losses})
        inputs.guard()
    results = []
    for task in ('minute_h60_future_log_return', 'hourly_h8_future_log_return'):
        inputs.guard(); results.append(aggregate_primary([s for s in scored if s['task'] == task]))
    result = {'kind': 'matched_two_state_market_screen_results_v1', 'protocol_sha256': protocol_sha,
              'status': 'main_scoring_complete_independent_review_pending', 'primary_results': results,
              'screen_conjunction_passed': all(r['passed'] for r in results), 'independent_review_complete': False,
              'model_records': 32, 'neural_fits': 24, 'enhanced_Ridge_fits': 4, 'Qmean_fits': 4,
              'all_preQ3_close_columns_physically_decoded': True, 'input_decode_audit': inputs.audit,
              'global_timestamp_blindness': False, 'Q3_read': False, 'GPU': False,
              'began_epoch': began, 'finished_epoch': time.time(), 'deadline_epoch': p['deadline_epoch'],
              'method_success': False, 'research_goal_achieved': False}
    inputs.guard(); atomic(out/'RESULTS.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('primary_results', 'input_decode_audit')}))


if __name__ == '__main__':
    main()
