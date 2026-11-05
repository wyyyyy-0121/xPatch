"""Load completed cloud experts and compare CPU predictions on known quarters.

This diagnostic never restores canonical results or changes a scientific gate.
It is not acceptance of the final three-seed fusion system.
"""
import argparse
import gc
import json
from pathlib import Path, PurePosixPath
import tarfile
import tempfile

import numpy as np
import torch

from fusion.next.common import ROOT, OUT, digest, read_json, source_identity, utc_now, write_json
from fusion.next.hardware import runtime
from fusion.next.panel import Panel, CoverageSampler
from fusion.next.checkpoint import resolve_checkpoint
from fusion.next.inference import checkpoint_model
from fusion.next.trainer import optimizer_for
from fusion.next.experiment import score_rows


def verify(archive, receipt_path, gpu_fp32_reference=None):
    if runtime(2026) != 'cpu':
        raise RuntimeError('Portable expert verification belongs on the local CPU')
    archive = Path(archive).resolve();receipt = read_json(receipt_path)
    if digest(archive) != receipt['sha256'] or archive.stat().st_size != receipt['bytes']:
        raise ValueError('Expert archive differs from its receipt')
    references = None
    if gpu_fp32_reference is not None:
        gpu_fp32_reference = Path(gpu_fp32_reference)
        metadata = read_json(gpu_fp32_reference.with_suffix('.json'))
        if metadata['source_identity'] != source_identity() or metadata['inference_precision'] != 'float32':
            raise ValueError('Float32 reference has another source or precision')
        if digest(gpu_fp32_reference) != metadata['prediction_sha256'] or gpu_fp32_reference.stat().st_size != metadata['bytes']:
            raise ValueError('Float32 reference predictions changed')
        with np.load(gpu_fp32_reference, allow_pickle=False) as values:
            references = {name:values[name].copy() for name in values.files}
    rows_checked = [];panel = Panel.load()
    with tempfile.TemporaryDirectory(prefix='expert_snapshot_', dir=ROOT/'tmp') as temporary:
        staging = Path(temporary)
        with tarfile.open(archive) as stream:
            seen = set()
            for member in stream.getmembers():
                path = PurePosixPath(member.name)
                if not member.isfile() or member.name != str(path) or member.name in seen or path.is_absolute() or '..' in path.parts or '\\' in member.name:
                    raise ValueError('Unexpected snapshot member')
                if not (staging/member.name).resolve().is_relative_to(staging):
                    raise ValueError('Snapshot leaves staging')
                seen.add(member.name)
            stream.extractall(staging, filter='data')
        manifest = read_json(staging/'checkpoint_snapshot.json')
        if seen != {'checkpoint_snapshot.json', *manifest['files']}:
            raise ValueError('Snapshot member set differs')
        if manifest['source_identity'] != source_identity() or manifest['source_identity'] != receipt['source_identity']:
            raise ValueError('Snapshot model engine differs')
        if manifest['scope_identity'] != receipt['scope_identity'] or manifest['Q3_values_included']:
            raise ValueError('Snapshot scope or data boundary differs')
        for name, row in manifest['files'].items():
            path = staging/name
            if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
                raise ValueError('Snapshot member changed: '+name)
        for entry in manifest['portable_checkpoints']:
            if entry['family'] not in ('xpatch', 'patchtst') or entry['quarter'] not in ('2025Q1', '2025Q3', '2026Q1'):
                raise ValueError('Snapshot is outside the known expert scope')
            path, committed = resolve_checkpoint(staging/entry['path'])
            state = torch.load(path, map_location='cpu', weights_only=True)
            recipe = state['recipe'];config = recipe['config']
            if recipe != committed['recipe'] or recipe['source_identity'] != source_identity():
                raise ValueError('Checkpoint recipe differs')
            if state['step'] != committed['step'] or state['step'] != config['total_updates'] or state['step'] != entry['step']:
                raise ValueError('Snapshot contains an incomplete training prefix')
            runtime(recipe['seed'])
            model, scalers, checked = checkpoint_model(panel, staging/entry['path'], config, 'cpu')
            optimizer = optimizer_for(model, config, 'cpu');optimizer.load_state_dict(state['optimizer'])
            if not optimizer.state:
                raise ValueError('Trained expert has no optimizer state')
            for saved in state['samplers'].values():
                sampler = CoverageSampler(panel, recipe['fit_end'], saved['seed'])
                sampler.load_state_dict(saved);_,sampled = sampler.sample(16)
                if np.any(panel.dates[sampled] > recipe['fit_end']):
                    raise ValueError('Restored sampler crosses its fit boundary')
            _, expected_rows = score_rows(panel, entry['quarter'])
            prediction_path = staging/entry['prediction_path']
            result = read_json(prediction_path.with_name('result.json'))
            if digest(prediction_path) != result['prediction_sha256']:
                raise ValueError('Saved reference prediction changed')
            with np.load(prediction_path, allow_pickle=False) as values:
                if not np.array_equal(values['rows'], expected_rows):
                    raise ValueError('Reference predictions changed common origins')
                indices = np.unique(np.linspace(0, len(expected_rows)-1, 64, dtype=np.int64))
                rows = expected_rows[indices];reference = values['prediction'][indices].copy()
            if references is not None:
                prefix = entry['family']+'_'+entry['quarter']
                if not np.array_equal(rows, references[prefix+'_rows']):
                    raise ValueError('Float32 reference changed diagnostic origins')
                reference = references[prefix+'_prediction']
            stocks = np.full(len(rows), panel.stocks.index('AAPL'), dtype=np.int64)
            batch = panel.windows(stocks, rows, config, scalers, device='cpu', include_labels=False)
            with torch.inference_mode(), torch.autocast('cpu', dtype=torch.bfloat16, enabled=references is None and config.get('precision')=='bf16'):
                observed = model(batch)['prediction'].float().numpy()
            target = panel.targets['AAPL'][rows]
            tolerance = max(1e-8, float(np.sqrt(np.mean(target**2)))*1e-4)
            difference = observed-reference
            rows_checked.append({'family':entry['family'], 'candidate_id':entry['candidate_id'],
                'quarter':entry['quarter'], 'step':state['step'], 'sampled_origins':len(rows),
                'model_optimizer_and_sampler_loaded':True,
                'maximum_CPU_GPU_absolute_difference':float(np.max(np.abs(difference))),
                'CPU_GPU_prediction_RMSE':float(np.sqrt(np.mean(difference.astype(np.float64)**2))),
                'diagnostic_absolute_tolerance':tolerance, 'diagnostic_relative_tolerance':1e-4,
                'CPU_GPU_prediction_parity_passed':bool(np.allclose(observed,reference,rtol=1e-4,atol=tolerance))})
            del model, optimizer, state, batch;gc.collect()
    report = {'archive_sha256':receipt['sha256'], 'source_identity':source_identity(),
        'scope_identity':receipt['scope_identity'], 'verified_at_utc':utc_now(),
        'checkpoints':rows_checked, 'all_hashes_and_portable_states_verified':True,
        'all_sampled_CPU_GPU_predictions_within_diagnostic_tolerance':all(r['CPU_GPU_prediction_parity_passed'] for r in rows_checked),
        'canonical_training_files_changed':False, 'Q3_values_read':False,
        'inference_precision':'float32' if references is not None else 'original BF16',
        'separate_GPU_reference_sha256':None if references is None else digest(gpu_fp32_reference),
        'final_fusion_acceptance_performed':False,
        'interpretation':'Known-period expert-only diagnostic; final system parity and engineering gates remain mandatory'}
    if len(rows_checked) != receipt['checkpoints']:
        raise ValueError('Checkpoint count differs from the receipt')
    suffix = '_CPU_FP32_verification.json' if references is not None else '_CPU_verification.json'
    write_json(OUT/'cloud/incremental_backups'/(receipt['sha256']+suffix),report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--gpu-fp32-reference', type=Path)
    args = parser.parse_args();report = verify(args.archive, args.receipt, args.gpu_fp32_reference)
    print(json.dumps({'checkpoints':report['checkpoints'],
        'all_sampled_CPU_GPU_predictions_within_diagnostic_tolerance':report['all_sampled_CPU_GPU_predictions_within_diagnostic_tolerance']}))
