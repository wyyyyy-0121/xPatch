"""Execute the sole frozen candidate using the unchanged matched engine."""
from study import *
import argparse, fcntl, traceback


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--job', required=True)
    args = parser.parse_args(); manifest = read(BASE/'protocol/CANDIDATE_JOBS.json')
    assert manifest['worker_sha256'] == sha(__file__)
    assert manifest['training_hashes'] == source_hashes()
    cfg = next(j for j in manifest['jobs'] if j['id'] == args.job)
    out = BASE/'runs'/cfg['id']; out.mkdir(parents=True, exist_ok=True)
    with (out/'LOCK').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            from engine import run
            result = run(cfg, out)
            print(json.dumps({'job': cfg['id'], 'status': result['status']}), flush=True)
        except Exception:
            atomic(out/'ERROR.json', {'at_utc': now(), 'traceback': traceback.format_exc()})
            raise
