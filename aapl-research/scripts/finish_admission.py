"""Sequential remaining GPU admission gates; no development scoring or formal search."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aapl.common import read_json, utc_now, write_json


if __name__ == '__main__':
    if read_json(ROOT / 'outputs/implementation_performance_queue.json')['status'] != 'completed':
        raise SystemExit('The sole-GPU performance queue must finish first')
    phases = [('compile', ['scripts/admit_compile.py']), ('parallel', ['-m', 'aapl.admit_parallel']),
              ('model_admission', ['-m', 'aapl.admit_models']), ('pipeline_smoke', ['scripts/smoke_pipeline.py'])]
    record = {'started_at_utc': utc_now(), 'completed': [], 'status': 'running'}
    for name, args in phases:
        record['current'] = name
        write_json(ROOT / 'outputs/final_admission_queue.json', record)
        print('START', name, utc_now(), flush=True)
        env = {**os.environ, 'AAPL_CPU_THREADS': '2', 'AAPL_CPU_IDS': '0-17' if name == 'parallel' else '0-1'}
        with (ROOT / f'outputs/{name}_final.log').open('w') as log:
            code = subprocess.run(['bash', str(ROOT / 'scripts/python.sh'), *args], cwd=ROOT, env=env,
                                  stdout=log, stderr=subprocess.STDOUT).returncode
        record['completed'].append({'phase': name, 'exit_code': code})
        print('END', name, code, utc_now(), flush=True)
        if code:
            record['status'] = 'failed'
            break
    else:
        record['status'] = 'completed'
    write_json(ROOT / 'outputs/final_admission_queue.json', record)
