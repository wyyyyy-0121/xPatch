"""Inner-tail scalar shrinkage for every admitted cell, without outer fitting."""
from study import *
import time, torch
from engine import Data
from networks import create
from multiscale.bars import targets


def main():
    assert len(os.sched_getaffinity(0)) <= 2 and os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest = BASE/'protocol/CALIBRATION.json'
    recipe = {'source_sha256': sha(__file__),
              'jobs_sha256': sha(BASE/'protocol/REPLICATION_JOBS.json'),
              'selection': 'alpha=clip(sum(q*r)/sum(q*q),0,1), inner-stop endpoints only',
              'q': 'trained prediction minus exact zero/persistence',
              'r': 'truth minus exact zero/persistence',
              'application': 'same scalar for all path steps',
              'outer_labels_used_for_calibration': False,
              'Q3_values_read': False}
    if manifest.exists():
        assert read(manifest)['recipe'] == recipe
    else:
        atomic(manifest, {'recipe': recipe, 'at_utc': now(), 'new_calibrated_outer_scores_seen': False})
    jobs = read(BASE/'protocol/REPLICATION_JOBS.json')['jobs']
    deadline = read(BASE/'protocol/BUDGET.json')['deadline_epoch']
    while True:
        pending = []
        for cfg in jobs:
            out = BASE/'calibration'/cfg['id']; done_path = BASE/'runs'/cfg['id']/'DONE.json'
            if (out/'DONE.json').exists():
                continue
            pending.append(cfg)
            if not done_path.exists():
                continue
            if time.time() >= deadline:
                atomic(BASE/'audit/CALIBRATION_PAUSED.json', {'at_utc': now(), 'deadline_epoch': deadline})
                return
            done = read(done_path); folder = done_path.parent
            assert sha(folder/'best_trained.pt') == done['checkpoint_sha256']
            assert sha(folder/'predictions.npz') == done['prediction_sha256']
            saved = torch.load(folder/'best_trained.pt', map_location='cpu', weights_only=True)
            data = Data(cfg); model = create(cfg, data.scaler)
            model.load_state_dict(saved['model']); model.eval()
            indices = data.splits['stop']; num = den = truth_sum = 0.
            with torch.inference_mode():
                for i in range(0, len(indices), 128):
                    pos = np.arange(i, min(i+128, len(indices))); args, _ = data.batch('stop', pos)
                    idx = indices[pos]
                    pred = data.invert(model(*args).numpy(), idx)[:, -1]
                    baseline = data.baseline(idx)[:, -1]
                    truth = targets(data.bars, idx, cfg['horizon'], cfg['target'])[:, -1]
                    q = pred-baseline; r = truth-baseline
                    num += float(np.dot(q, r)); den += float(np.dot(q, q)); truth_sum += float(np.dot(r, r))
            alpha = float(np.clip(num/den, 0, 1)) if den else 0.
            optimal = (truth_sum-2*alpha*num+alpha*alpha*den)/len(indices)
            endpoints = min(truth_sum, truth_sum-2*num+den)/len(indices)
            assert optimal <= endpoints+1e-10*max(1., abs(endpoints))
            with np.load(folder/'predictions.npz', allow_pickle=False) as a:
                origins = a['origins']; pred = a['trained']
            assert np.array_equal(origins, data.splits['validation'])
            baseline = data.baseline(origins)
            calibrated = baseline+alpha*(pred-baseline)
            npz(out/'predictions.npz', origins=origins, prediction=calibrated)
            atomic(out/'DONE.json', {'status': 'completed', 'job': cfg['id'], 'alpha': alpha,
                'inner_stop_mse': optimal, 'unshrunk_inner_stop_mse': (truth_sum-2*num+den)/len(indices),
                'baseline_inner_stop_mse': truth_sum/len(indices),
                'source_checkpoint_sha256': done['checkpoint_sha256'], 'recipe_sha256': sha(manifest),
                'prediction_sha256': sha(out/'predictions.npz'), 'outer_labels_used_for_calibration': False,
                'Q3_values_read': False, 'at_utc': now()})
            print(json.dumps({'calibrated': cfg['id'], 'alpha': alpha}), flush=True)
        completed = sum((BASE/'calibration'/j['id']/'DONE.json').exists() for j in jobs)
        atomic(BASE/'audit/CALIBRATION_STATE.json', {'planned': len(jobs), 'completed': completed,
               'at_utc': now(), 'pid': os.getpid(), 'deadline_epoch': deadline})
        if completed == len(jobs):
            atomic(BASE/'audit/CALIBRATION_COMPLETE.json', {'status': 'completed', 'jobs': len(jobs),
                   'at_utc': now(), 'Q3_values_read': False})
            return
        if time.time() >= deadline:
            atomic(BASE/'audit/CALIBRATION_PAUSED.json', {'completed': completed, 'planned': len(jobs),
                   'at_utc': now(), 'deadline_epoch': deadline})
            return
        if not any((BASE/'runs'/j['id']/'DONE.json').exists() for j in pending):
            time.sleep(10)


if __name__ == '__main__':
    main()
