"""New identity-edge tests only; original clock/worker receipts stay preserved."""
import json
from pathlib import Path
import tempfile
import time

import coverage_worker as worker


def main():
    started = time.time(); checks = []
    def reject(name, call):
        try:
            call()
        except ValueError:
            checks.append(name)
        else:
            raise AssertionError(name)
    with tempfile.TemporaryDirectory(prefix='synthetic-frozen-clock-') as temp:
        path = Path(temp)/'input.json'; path.write_text('{"clock": 1}\n')
        refs = [{'path': str(path), 'sha256': worker.file_sha(path)}]
        table = worker.lock_table(refs)
        worker.require_locked(path, table)
        assert worker.frozen_json(path, table) == {'clock': 1}
        checks.append('matching_frozen_actual_bytes_accepted')
        reject('empty_locks_rejected_at_read', lambda: worker.require_locked(path, {}))
        reject('empty_locks_reject_json_read', lambda: worker.frozen_json(path, {}))
        reject('duplicate_resolved_path_rejected', lambda: worker.lock_table(refs+refs))
        path.write_text('{"clock": 2}\n')
        reject('changed_container_hash_rejected', lambda: worker.require_locked(path, table))
        reject('changed_parsed_JSON_bytes_rejected', lambda: worker.frozen_json(path, table))
    root = Path(__file__).resolve().parent
    report = {'passed': True, 'checks_count': len(checks), 'checks': checks,
        'worker_sha256': worker.file_sha(root/'coverage_worker.py'),
        'test_sha256': worker.file_sha(Path(__file__)), 'seconds': time.time()-started,
        'actual_market_inputs_read': False, 'HTTP': False, 'Q3_read': False,
        'earlier_receipts_not_replaced': True}
    (root/'FROZEN_INPUT_CHECKS.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
