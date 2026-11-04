"""Keep compiler startup bounded and preserve a timeout/failure as a tested fallback."""
from pathlib import Path
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aapl.common import read_json, utc_now, write_json


if __name__ == '__main__':
    started = time.perf_counter()
    env = {**os.environ, 'AAPL_CPU_THREADS': '2', 'AAPL_CPU_IDS': '0-1', 'TORCHINDUCTOR_COMPILE_THREADS': '2'}
    record = {'completed': False, 'adopted': False, 'maximum_probe_seconds': 120}
    with (ROOT / 'outputs/compile_probe.log').open('w') as log:
        process = subprocess.Popen(['bash', str(ROOT / 'scripts/python.sh'), '-m', 'aapl.compile_probe'],
                                   stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT, start_new_session=True)
        try:
            code = process.wait(timeout=120)
            record.update(completed=True, exit_code=code)
            if code == 0:
                result = read_json(ROOT / 'outputs/compile_probe.json')
                record['records'] = result['records']
                record['eligible_for_further_admission'] = [n for n, r in result['records'].items() if r['equivalent'] and r['four_epoch_net_gain']]
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=15)
            record.update(completed=True, timeout=True, exit_code=process.returncode)
    record.update(completed_at_utc=utc_now(), wall_seconds=time.perf_counter() - started,
                  policy='Retain the admitted eager path unless equivalence and net pilot benefit both pass, followed by recovery and sustained resource checks')
    write_json(ROOT / 'outputs/compile_admission.json', record)
    print(record, flush=True)
