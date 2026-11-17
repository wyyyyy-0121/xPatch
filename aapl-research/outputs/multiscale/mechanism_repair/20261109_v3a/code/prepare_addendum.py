"""Stage portable supplemental evidence without copying the 4.63 GiB prior archive."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import sys
import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
V3 = BASE.parent / '20261109_v3'
sys.path.insert(0, str(V3 / 'code'))
from engine import Data
from portable_predictor import Predictor


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def copytree(src, dest):
    shutil.copytree(src, dest, dirs_exist_ok=True,
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'lo_profile_*', '*.log'))


def main():
    assert set(os.sched_getaffinity(0)).issubset({24, 26})
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    root = BASE / 'release/AAPL_MULTISCALE_V3A_ADDENDUM'
    root.mkdir(parents=True, exist_ok=True)
    audit = BASE / 'audit'; audit.mkdir(exist_ok=True)
    prior = read(V3 / 'audit/RELEASE.json')
    priorarchive = Path(prior['archive'])
    assert sha(priorarchive) == prior['archive_sha256']
    # The prior archive is a checked dependency, not a duplicate nested payload.
    reference = {'archive_name': priorarchive.name, 'archive_sha256': prior['archive_sha256'],
        'archive_bytes': priorarchive.stat().st_size, 'workspace_path_at_build': str(priorarchive),
        'not_included': True, 'required_for': 'preceding full comparison/factorial training and original 20-task inference',
        'addendum_entrypoints_are_self_contained': True}
    (root / 'PRECEDING_ARCHIVE.json').write_text(json.dumps(reference, indent=2))
    copytree(V3 / 'code/reference_source', root / 'code/reference_source')
    for name in ('portable_predictor.py', 'portable_stats.py', 'tiny_architecture.py', 'format_documents.py'):
        out = root / 'code' / name; out.parent.mkdir(exist_ok=True)
        shutil.copy2(V3 / 'code' / name, out)
    for name in ('reproduce_addendum_statistics.py', 'replay_addendum.py'):
        shutil.copy2(BASE / 'code' / name, root / 'code' / name)
    copytree(V3 / 'data', root / 'data')
    copytree(BASE / 'longscale_replication/protocol', root / 'longscale_replication/protocol')
    copytree(BASE / 'longscale_replication/code', root / 'longscale_replication/code')
    copytree(BASE / 'longscale_replication/analysis', root / 'longscale_replication/analysis')
    copytree(BASE / 'longscale_replication/audit', root / 'longscale_replication/audit')
    copytree(BASE / 'longscale_replication/calibration', root / 'longscale_replication/calibration')
    for folder in ('mean_controls', 'history_sensitivity', 'longscale_review'):
        copytree(BASE / folder, root / folder)
    manifest = read(BASE / 'longscale_replication/protocol/JOBS.json')
    locations = {}; examples = []
    for cfg in manifest['jobs']:
        src = Path(manifest['locations'][cfg['id']])
        done = read(src / 'DONE.json')
        assert sha(src / 'best_trained.pt') == done['checkpoint_sha256']
        dest = root / 'longscale_replication/runs' / cfg['id']; dest.mkdir(parents=True, exist_ok=True)
        for name in ('best_trained.pt', 'initial.pt', 'coupled_best.pt', 'predictions.npz', 'DONE.json', 'INITIALIZATION.json'):
            if (src / name).exists():
                shutil.copy2(src / name, dest / name)
        locations[cfg['id']] = str(dest.relative_to(root))
        data = Data(cfg)
        origins = data.splits['validation'][:128]
        x, anchor = data.windows.arrays(origins)
        cal = read(BASE / 'longscale_replication/calibration' / cfg['id'] / 'DONE.json')
        alpha = cal['alpha']
        model = Predictor(dest / 'best_trained.pt')
        raw = model.predict_prepared(x, anchor)
        with np.load(src / 'predictions.npz', allow_pickle=False) as a:
            assert np.array_equal(a['origins'][:len(origins)], origins)
            assert np.array_equal(raw, a['trained'][:len(origins)])
        calibrated = model.predict_prepared(x, anchor, shrink_alpha=alpha)
        with np.load(BASE / 'longscale_replication/calibration' / cfg['id'] / 'predictions.npz', allow_pickle=False) as a:
            assert np.array_equal(calibrated, a['prediction'][:len(origins)])
        arrays = {'history': x, 'current_close': anchor, 'expected_raw': raw,
                  'expected_shrink': calibrated, 'origins': origins}
        interventions = False
        if cfg['initialization'] == 'baseline':
            task = f"{cfg['profile']}_h{cfg['horizon']}_{cfg['target']}"
            with np.load(BASE / 'history_sensitivity/analysis' / task / f"fold_{cfg['fold']}.npz", allow_pickle=False) as a:
                arrays['fit_mean_history'] = a['fit_mean_history']
                for mode in ('actual', 'fit_mean_history', 'flat_history'):
                    arrays['expected_' + mode] = a[f"{mode}_seed_{cfg['seed']}"][:len(origins)]
            interventions = True
        path = root / 'delivery' / f"{cfg['id']}.npz"; path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, **arrays)
        examples.append({'job': cfg['id'], 'checkpoint': str((dest / 'best_trained.pt').relative_to(root)),
                         'example': str(path.relative_to(root)), 'alpha': alpha,
                         'history_interventions': interventions, 'rows': len(origins), 'sha256': sha(path)})
    assert len(examples) == 180 and sum(e['history_interventions'] for e in examples) == 90
    (root / 'PORTABLE_LOCATIONS.json').write_text(json.dumps(locations, indent=2))
    (root / 'delivery/EXAMPLES.json').write_text(json.dumps({'selection':
        'Every long-scale configuration, first canonical inference batch up to128 origins, no score selection',
        'examples': examples, 'targets_loaded_during_example_build': False,
        'expected_existing_predictions_loaded': True, 'Q3_values_read': False}, indent=2))
    # Existing saved predictions are expected outputs, not newly selected labels.
    receipt = {'status': 'portable_supplement_staged_final_BN_and_documents_pending',
        'longscale_examples': 180, 'history_examples': 90, 'original_batch_raw_and_shrink_exact': True,
        'prior_archive_hash_verified': prior['archive_sha256'], 'analysis_affinity': sorted(os.sched_getaffinity(0)),
        'Q3_values_read': False, 'at_utc': datetime.now(timezone.utc).isoformat()}
    (audit / 'STAGING.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
