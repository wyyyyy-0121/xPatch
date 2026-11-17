"""Passively record this approved run's progress and resource observations."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from multiscale.historical_continuation import digest, matched, read, save

RUN = Path(__file__).resolve().parent


def main():
    assert os.sched_getaffinity(0) == {24, 26}
    started = time.time()
    previous = None
    totals = {}
    polls = 0
    with (RUN / 'OBSERVED_TRANSITIONS.jsonl').open('x', encoding='utf8') as stream:
        while True:
            state = read(RUN / 'STATE.json')
            observed = {}
            for key, entry in state['jobs'].items():
                proc = matched(entry.get('process'))
                row = {'phase': entry['phase'], 'pid': proc['pid'] if proc else None,
                       'process_state': proc['state'] if proc else None, 'slot': entry.get('slot'),
                       'mode': entry.get('mode')}
                if proc:
                    try:
                        raw = Path(f'/proc/{proc["pid"]}/stat').read_text()
                        fields = raw[raw.rfind(')') + 2:].split()
                        cpu = (int(fields[11]) + int(fields[12])) / os.sysconf('SC_CLK_TCK')
                        item = totals.setdefault(key, {'pid': proc['pid'], 'start_ticks': proc['start_ticks'],
                                                      'first_cpu_seconds': cpu, 'last_cpu_seconds': cpu})
                        item['last_cpu_seconds'] = cpu
                    except (FileNotFoundError, ProcessLookupError):
                        pass
                observed[key] = row
            signature = {'phase': state['phase'], 'jobs': observed}
            if signature != previous:
                event = {'at_utc': datetime.now(timezone.utc).isoformat(), 'epoch': time.time(), **signature}
                stream.write(json.dumps(event, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                previous = signature
            polls += 1
            running = [v for v in observed.values() if v['phase'] == 'running']
            summary = {'status': 'passive_progress_observation', 'at_utc': datetime.now(timezone.utc).isoformat(),
                       'run_phase': state['phase'], 'completed': sum(v['phase'] == 'completed' for v in observed.values()),
                       'remaining_run_jobs': 15, 'running': len(running), 'polls': polls,
                       'maximum_workers': state['maximum_workers'], 'cpu_observations': totals,
                       'observer_started_epoch': started, 'script_sha256': digest(__file__),
                       'training_started_by_observer': False, 'signals_sent': False,
                       'limitations': ['Samples bound phase changes; these are not exact release timestamps.',
                                       'CPU increments omit work before the first sample and after the last sample.']}
            save(RUN / 'PROGRESS_OBSERVATION.json', summary)
            if state['phase'] in ('completed', 'paused_preserved'):
                return
            if time.time() > state['deadline_epoch'] + 180:
                summary['status'] = 'observation_deadline_passed_needs_attention'
                save(RUN / 'PROGRESS_OBSERVATION.json', summary)
                return
            time.sleep(2)


if __name__ == '__main__':
    main()
