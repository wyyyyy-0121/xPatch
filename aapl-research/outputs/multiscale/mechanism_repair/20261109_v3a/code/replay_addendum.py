"""Input-only, portable exact-batch replay of all supplemental long-scale functions."""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import torch
from portable_predictor import Predictor


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args(); root = args.root.resolve()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    examples = json.loads((root / 'delivery/EXAMPLES.json').read_text())['examples']
    n = history = 0
    for rec in examples:
        file = root / rec['example']; assert sha(file) == rec['sha256']
        predictor = Predictor(root / rec['checkpoint'])
        with np.load(file, allow_pickle=False) as a:
            x, anchor = a['history'], a['current_close']
            raw = predictor.predict_prepared(x, anchor)
            calibrated = predictor.predict_prepared(x, anchor, shrink_alpha=rec['alpha'])
            assert np.array_equal(raw, a['expected_raw']), rec['job']
            assert np.array_equal(calibrated, a['expected_shrink']), rec['job']
            n += 1
            if rec['history_interventions']:
                for mode, window in [('actual', x), ('fit_mean_history', np.broadcast_to(a['fit_mean_history'], x.shape)),
                                     ('flat_history', np.zeros_like(x))]:
                    forecast = predictor.predict_prepared(window, anchor, shrink_alpha=rec['alpha'])
                    assert np.array_equal(forecast, a['expected_' + mode]), (rec['job'], mode)
                    history += 1
    assert n == 180 and history == 270
    print(json.dumps({'status': 'passed', 'longscale_predictors': n, 'history_interventions': history,
                      'canonical_batch_predictions_bitwise_equal': True, 'Q3_values_read': False}), flush=True)


if __name__ == '__main__':
    main()
