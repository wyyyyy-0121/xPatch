"""Continue a copied original-price fit; preserve the exact trained prefix."""
from study import *
import argparse, fcntl, traceback, torch
from multiscale.training import atomic_torch_save


def same(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--job', required=True)
    args = parser.parse_args(); manifest = read(BASE/'protocol/CAP_JOBS.json')
    assert manifest['worker_sha256'] == sha(__file__)
    assert manifest['training_hashes'] == source_hashes()
    item = next(j for j in manifest['jobs'] if j['cfg']['id'] == args.job)
    cfg = item['cfg']; source = BASE/'runs'/item['source_job']; out = BASE/'runs'/cfg['id']
    out.mkdir(parents=True, exist_ok=True)
    with (out/'LOCK').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            prior = read(source/'DONE.json')
            assert prior['job']['max_epochs'] == 20 and cfg['max_epochs'] == 60
            assert {k: v for k, v in cfg.items() if k not in ('id', 'max_epochs')} == \
                {k: v for k, v in prior['job'].items() if k not in ('id', 'max_epochs')}
            assert sha(source/'latest.pt') == item['source_latest_sha256']
            frozen = read(BASE/'protocol/CODE_FREEZE.json')
            identity = hashlib.sha256(json.dumps({'cfg': cfg, 'code': frozen['hashes'],
                    'data': sha(PRIOR/'protocol/DATA.json')}, sort_keys=True).encode()).hexdigest()
            if not (out/'latest.pt').exists():
                checks = {}
                for name in ('initial.pt', 'best_trained.pt', 'coupled_best.pt', 'latest.pt'):
                    if not (source/name).exists():
                        continue
                    saved = torch.load(source/name, map_location='cpu', weights_only=True)
                    assert saved['identity'] == prior['identity']
                    copied = dict(saved, cfg=cfg, identity=identity)
                    atomic_torch_save(copied, out/name)
                    actual = torch.load(out/name, map_location='cpu', weights_only=True)
                    checks[name] = all(same(saved[k], actual[k]) for k in saved if k not in ('identity', 'cfg'))
                assert all(checks.values())
                atomic(out/'PREFIX_COPY.json', {'source_job': item['source_job'], 'checks': checks,
                    'exact_model_optimizer_rng_cursor_history': all(checks.values()),
                    'source_latest_sha256': item['source_latest_sha256'],
                    'source_done_sha256': sha(source/'DONE.json'), 'original_files_modified': False,
                    'only_identity_and_maximum_epoch_changed': True, 'at_utc': now()})
            from engine import run
            result = run(cfg, out)
            if result['status'] == 'completed':
                more_steps = result['steps']-prior['steps']
                original_was_cap_bound = prior['epochs'] == 20
                if not original_was_cap_bound:
                    assert more_steps == 0
                    assert np.isclose(result['metrics']['trained']['endpoint_mse'],
                                      prior['metrics']['trained']['endpoint_mse'], rtol=1e-12)
                atomic(out/'CAP_COMPARISON.json', {'status': 'completed', 'source_job': item['source_job'],
                    'original_was_cap_bound': original_was_cap_bound, 'original_epochs': prior['epochs'],
                    'longer_cap_epochs': result['epochs'], 'new_updates': more_steps,
                    'original_trained_mse': prior['metrics']['trained']['endpoint_mse'],
                    'longer_cap_trained_mse': result['metrics']['trained']['endpoint_mse'],
                    'improvement_over_original_pct': 100*(1-result['metrics']['trained']['endpoint_mse']/
                                                         prior['metrics']['trained']['endpoint_mse']),
                    'causal_scope': 'same original default price recipe and early-stop policy, maximum epoch cap only',
                    'Q3_values_read': False, 'at_utc': now()})
            print(json.dumps({'job': cfg['id'], 'status': result['status']}), flush=True)
        except Exception:
            atomic(out/'ERROR.json', {'at_utc': now(), 'traceback': traceback.format_exc()})
            raise


if __name__ == '__main__':
    main()
