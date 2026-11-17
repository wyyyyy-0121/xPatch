"""Describe the actual additive anchor path and early gradient flow of all cells."""
from study import *
import torch, csv


def main():
    assert len(os.sched_getaffinity(0)) <= 2
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    manifest = read(BASE/'protocol/REPLICATION_JOBS.json')
    assert read(BASE/'audit/REPLICATION_COMPLETE.json')['jobs'] == len(manifest['jobs'])
    rows = []
    for cfg in manifest['jobs']:
        if cfg['model'] not in ('lstm', 'xpatch'):
            continue
        folder = BASE/'runs'/cfg['id']; done = read(folder/'DONE.json')
        assert sha(folder/'best_trained.pt') == done['checkpoint_sha256']
        ck = torch.load(folder/'best_trained.pt', map_location='cpu', weights_only=True)
        rec = read(PRIOR/'cache'/task_key(cfg['profile'], cfg['horizon'], cfg['target'])/'TASK.json')['folds'][str(cfg['fold'])]
        assert sha(rec['targets']) == rec['targets_sha256']
        with np.load(rec['targets'], allow_pickle=False) as a:
            origins = a['origins']; anchor = a['anchor']; truth = a['actual'][:, -1]
        assert sha(folder/'predictions.npz') == done['prediction_sha256']
        with np.load(folder/'predictions.npz', allow_pickle=False) as a:
            assert np.array_equal(origins, a['origins']); p = a['trained'][:, -1]
        scaler = ck['scaler']; state = ck['model']
        h = cfg['horizon']-1
        w = float(state['anchor_head.weight'][h, 0]) if cfg['model'] == 'xpatch' else float(state['head.weight'][h, -1])
        learned_coefficient = w*scaler['y_std'][h]/scaler['anchor_std']
        full_coefficient = learned_coefficient+(1. if cfg['output'] == 'residual' else 0.)
        incremental_coefficient = full_coefficient-(1. if cfg['target'] == 'future_close' else 0.)
        # This is an algebraic decomposition of one fitted network, not a refit.
        component = (anchor-scaler['anchor_mean'])*incremental_coefficient \
            if cfg['anchor'] == 'observed' else np.zeros_like(anchor)
        baseline = np.zeros_like(anchor) if cfg['target'] == 'future_log_return' else anchor
        residual_prediction = p-baseline; actual_change = truth-baseline
        without_anchor_path = residual_prediction-component
        mse = float(np.mean((residual_prediction-actual_change)**2))
        identity_mse = float(np.mean(actual_change**2)+np.mean(residual_prediction**2)-
                             2*np.mean(actual_change*residual_prediction))
        assert np.isclose(mse, identity_mse, rtol=1e-10, atol=1e-12)
        gradient_checks = []
        for step, values in enumerate(done['gradients_first_two'], 1):
            if cfg['model'] == 'lstm':
                backbone = [value for name, value in values.items() if name.startswith('rnn.')]
                head = [value for name, value in values.items() if name.startswith('head.')]
            else:
                backbone = [value for name, value in values.items() if name.startswith('backbone.') and
                            not name.startswith('backbone.fc8.')]
                head = [value for name, value in values.items() if name.startswith(('backbone.fc8.', 'anchor_head.'))]
            gradient_checks.append({'step': step, 'backbone_norm_sum': sum(backbone),
                                    'head_norm_sum': sum(head), 'backbone_all_zero': all(x == 0 for x in backbone)})
        rows.append({'job': cfg['id'], 'task': task_key(cfg['profile'], cfg['horizon'], cfg['target']),
                     'model': cfg['model'], 'variant': cfg['variant'], 'fold': cfg['fold'], 'seed': cfg['seed'],
                     'best_epoch': done['trained_best_epoch'], 'learned_anchor_raw_coefficient': learned_coefficient,
                     'total_anchor_raw_coefficient_including_known_skip': full_coefficient,
                     'increment_anchor_raw_coefficient': incremental_coefficient,
                     'anchor_effective_input': cfg['anchor'], 'prediction_increment_second_moment': float(np.mean(residual_prediction**2)),
                     'actual_increment_second_moment': float(np.mean(actual_change**2)),
                     'prediction_actual_cross_moment': float(np.mean(residual_prediction*actual_change)),
                     'anchor_component_second_moment': float(np.mean(component**2)),
                     'remainder_second_moment': float(np.mean(without_anchor_path**2)),
                     'component_remainder_cross_moment': float(np.mean(component*without_anchor_path)),
                     'endpoint_mse': mse, 'error_identity_verified': True, 'first_updates': gradient_checks,
                     'interpretation': 'algebraic path attribution; retrained factorial contrast is separate causal evidence'})
    atomic(BASE/'analysis/STRUCTURAL_PROBES.json', {'status': 'completed', 'rows': rows,
           'at_utc': now(), 'Q3_values_read': False, 'outer_labels_used_for_design': False,
           'scope': 'descriptive decomposition, does not establish market mechanism'})
    with (BASE/'analysis/STRUCTURAL_PROBES.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    baseline = [r for r in rows if r['variant'].endswith('baseline')]
    print(json.dumps({'cells': len(rows), 'baseline_initialization_cells': len(baseline),
                      'baseline_first_update_backbone_zero': sum(r['first_updates'][0]['backbone_all_zero'] for r in baseline)}), flush=True)


if __name__ == '__main__':
    main()
