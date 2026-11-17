"""Exercise the entire frozen engine across a real process exit using synthetic data."""
from common import *
import argparse, subprocess


def child(root, interrupt):
    import engine
    from multiscale.features import Scaler
    import torch

    class SyntheticData:
        def __init__(self, fold):
            rng = np.random.default_rng(771)
            self.fit = np.arange(513)
            self.stop = np.arange(513, 642)
            self.val = np.arange(642, 703)
            self.rows = {'fit': self.fit, 'stop': self.stop, 'val': self.val}
            self.x = rng.normal(size=(703, 480, 1)).astype('float32')
            self.a = rng.normal(size=(703, 1)).astype('float32')
            self.cache = {'actual': rng.normal(0, .001, size=(703, 60))}
            self.scaler = Scaler(0., 1., 0., 1., tuple([0.]*60), tuple([.001]*60), 513)
            self.feature_stats = {}; self.vol_threshold = .001
            self.bars = type('Bars', (), {'end_ns': np.arange(703)})()

        def batch(self, part, positions, kind):
            rr = self.rows[part][positions]
            return (torch.from_numpy(self.x[rr]), torch.from_numpy(self.a[rr])), torch.from_numpy(
                self.scaler.transform_y(self.cache['actual'][rr]))

    engine.BASE = Path(root)
    engine.Data = SyntheticData
    result = engine.run({'id': 'synthetic_resume_mlp', 'kind': 'mlp_noanchor',
                         'fold': 1, 'seed': 2026, 'phase': 'synthetic_acceptance'},
                        stop_after_steps=2 if interrupt else None)
    print(json.dumps({'status': result['status'], 'synthetic_only': True}), flush=True)


def verify():
    import torch
    root = BASE/'audit/engine_recovery'
    root.mkdir(exist_ok=True)
    for mode in ('continuous', 'resumed'):
        target = root/mode
        (target/'protocol').mkdir(parents=True, exist_ok=True)
        (target/'runs').mkdir(exist_ok=True)
        atomic(target/'protocol/CODE_FREEZE.json', read(BASE/'protocol/CODE_FREEZE.json'))
        atomic(target/'protocol/BUDGET.json', read(BASE/'protocol/BUDGET.json'))
        atomic(target/'protocol/DATA.json', {'synthetic_only': True, 'rng': 771, 'fit': 513, 'stop': 129, 'validation': 61})
    def launch(mode, interrupt=False):
        args = [sys.executable, __file__, '--child', str(root/mode)]
        if interrupt: args.append('--interrupt')
        subprocess.run(args, check=True)
    launch('continuous')
    launch('resumed', True)
    paused = read(root/'resumed/runs/synthetic_resume_mlp/PAUSE.json')
    assert paused['steps'] == 2 and paused['at_epoch'] == 0
    launch('resumed')
    a = root/'continuous/runs/synthetic_resume_mlp'; b = root/'resumed/runs/synthetic_resume_mlp'
    ca = torch.load(a/'latest.pt', map_location='cpu', weights_only=True)
    cb = torch.load(b/'latest.pt', map_location='cpu', weights_only=True)
    def exact(x, y):
        if torch.is_tensor(x): return torch.equal(x, y)
        if isinstance(x, dict): return x.keys()==y.keys() and all(exact(x[k], y[k]) for k in x)
        if isinstance(x, (list, tuple)): return len(x)==len(y) and all(exact(i,j) for i,j in zip(x,y))
        return x == y
    checks = {'model': exact(ca['model'], cb['model']), 'optimizer': exact(ca['optimizer'], cb['optimizer']),
        'rng': exact(ca['rng'], cb['rng']), 'torch_rng': exact(ca['torch_rng'], cb['torch_rng']),
        'history': exact(ca['state']['history'], cb['state']['history']),
        'steps': ca['state']['steps']==cb['state']['steps']}
    with np.load(a/'prediction.npz') as p, np.load(b/'prediction.npz') as q:
        checks['full_prediction'] = np.array_equal(p['prediction'], q['prediction'])
    checks['best_checkpoint_model'] = exact(torch.load(a/'best.pt', weights_only=True)['model'],
                                          torch.load(b/'best.pt', weights_only=True)['model'])
    assert all(checks.values()), checks
    atomic(BASE/'audit/ENGINE_RECOVERY.json', {'status': 'passed', 'checks_exact': checks,
        'process_invocations': 3, 'interruption': 'explicit bound after two minibatches, mid-epoch',
        'stochastic_model': 'MLP probe with dropout and Adam', 'synthetic_only': True,
        'new_market_fits': 0, 'source_hashes': code_hashes(), 'Q3_values_read': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--child'); p.add_argument('--interrupt', action='store_true')
    args = p.parse_args()
    child(args.child, args.interrupt) if args.child else verify()
