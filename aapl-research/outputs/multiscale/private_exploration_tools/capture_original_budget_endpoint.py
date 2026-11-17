"""Record frozen workers, measured process CPU and verified checkpoint copies."""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path

import torch

from multiscale.historical_coverage import digest, read_json, require

ROOT = Path('/home/ayaka/wslcode/aapl-research')
OUT = ROOT / 'outputs/multiscale'
NIGHT = OUT / 'private_low_load/20261107T153438623158Z'
ARCHIVES = [OUT / 'budget_preserved_checkpoints' / name for name in
            ('20261108_h60_return_f4', '20261108_minute_matrix_original_budget')]
DESTINATION = OUT / 'budget_endpoint_audits/20261108_original_cpu_budget'


def main():
    state = read_json(NIGHT / 'NIGHT_RESUME_STATE.json')
    require(all(g['status'] in ('completed', 'paused_budget_exhausted') for g in state['groups'].values()),
            'Original queues are still active')
    archives, files, checkpoints, workers = [], [], [], []
    for folder in ARCHIVES:
        path = folder / 'PRESERVATION_RECEIPT.json'
        receipt = read_json(path)
        entries = receipt.get('files', receipt.get('copies', []))
        if not isinstance(entries, list):
            raise ValueError('Unexpected preservation receipt layout')
        archives.append({'path': str(path), 'sha256': digest(path), 'status': receipt['status']})
        for entry in entries:
            source, copy = Path(entry['source']), Path(entry['copy'])
            require(digest(source) == digest(copy) == entry['sha256'], 'Checkpoint copy differs')
            files.append({'source': str(source), 'copy': str(copy), 'sha256': entry['sha256'],
                          'bytes': source.stat().st_size})
            if copy.name != 'latest.pt': continue
            saved = torch.load(copy, map_location='cpu', weights_only=True)
            config = read_json(source.parent.parent / 'fold_state.json')['config']
            progress = saved['state']
            loss = float(progress['best_loss'])
            checkpoints.append({'source': str(source), 'copy': str(copy), 'identity': saved['identity'],
                                'model': source.parent.name, 'target': config['target'],
                                'horizon': config['horizon'],
                                'epoch': progress['epoch'], 'steps': progress['steps'],
                                'cursor': progress['cursor'], 'best_epoch': progress['best_epoch'],
                                'best_loss_if_finite': loss if math.isfinite(loss) else None,
                                'note': 'Periodic disk state; original frozen RAM can contain newer updates.'})
    tick_rate = os.sysconf('SC_CLK_TCK')
    for group in ('787603', '797109'):
        marker = NIGHT / f'BUDGET_STOP_{group}.json'
        stop = read_json(marker)
        require(stop['reason'] == 'original_compute_budget_exhausted', 'Stop reason differs')
        for item in stop['stopped']:
            pid = item['pid']
            raw = Path(f'/proc/{pid}/stat').read_text()
            fields = raw[raw.rfind(')') + 2:].split()
            require(int(fields[19]) == item['start_ticks'] and fields[0] in ('T', 't'),
                    'Recorded original worker is missing, replaced or no longer frozen')
            key = next(k for k, v in state['workers'].items() if v['pid'] == pid)
            workers.append({'group': group, 'job': key, 'pid': pid, 'start_ticks': item['start_ticks'],
                            'state': fields[0], 'affinity': sorted(os.sched_getaffinity(pid)),
                            'process_lifetime_user_cpu_seconds': int(fields[11]) / tick_rate,
                            'process_lifetime_system_cpu_seconds': int(fields[12]) / tick_rate,
                            'stop_marker': str(marker), 'stop_marker_sha256': digest(marker)})
    report = {'status': 'original_budget_endpoint_workers_and_archives_verified',
              'created_at_utc': datetime.now(timezone.utc).isoformat(), 'groups': state['groups'],
              'workers': workers, 'archive_receipts': archives, 'files_verified': len(files),
              'total_preserved_bytes': sum(f['bytes'] for f in files), 'files': files,
              'periodic_checkpoint_progress': checkpoints,
              'measured_frozen_worker_lifetime_cpu_seconds': sum(w['process_lifetime_user_cpu_seconds'] +
                  w['process_lifetime_system_cpu_seconds'] for w in workers),
              'processes_resumed': False, 'budget_extended': False, 'new_training_started': False,
              'Q3_values_read': False, 'script_sha256': digest(Path(__file__)),
              'limitations': ['Process CPU totals cover only the frozen workers still alive, not all historical jobs.',
                              'Per-model CPU and total study CPU were not measured.',
                              'Periodic checkpoints can trail the exact in-memory state; both are retained.']}
    DESTINATION.mkdir(parents=True, exist_ok=False)
    with (DESTINATION / 'ORIGINAL_BUDGET_ENDPOINT_AUDIT.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: report[key] for key in ('status', 'files_verified', 'total_preserved_bytes',
                    'measured_frozen_worker_lifetime_cpu_seconds', 'budget_extended')} |
                    {'frozen_workers': len(workers), 'destination': str(DESTINATION)}))


if __name__ == '__main__': main()
