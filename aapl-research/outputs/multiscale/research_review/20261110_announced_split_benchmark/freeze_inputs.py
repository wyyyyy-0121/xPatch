"""Freeze saved development inputs without decoding market-array members."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import hashlib
import json
import os
import time

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/ayaka/wslcode/aapl-research')
OLD = ROOT / 'outputs/multiscale/controlled_followup/20261108_v2'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=int, default=300)
    args = parser.parse_args()
    assert 1 <= args.seconds <= 300
    target = HERE / 'PROTOCOL.json'
    assert not target.exists(), 'Never replace a frozen protocol or reset its deadline'
    qualification = read(HERE / 'SOURCE_QUALIFICATION.json')
    assert qualification['qualified_for_fixed_historical_benchmark'] is True
    known = datetime(2020, 8, 1, tzinfo=ZoneInfo('America/New_York'))
    assert qualification['known_ns'] == int(known.timestamp()) * 1000000000
    assert qualification['event']['effective_trading_date'] == '2020-08-31'
    assert qualification['event']['ratio'] == 4.0
    synthetic = read(HERE / 'SYNTHETIC_ACCEPTANCE.json')
    assert synthetic['status'] == 'passed_synthetic_only_boundary_acceptance'
    assert synthetic['checks_passed'] == 12
    assert synthetic['worker_sha256'] == sha(HERE / 'audit_worker.py')
    inputs = {}

    def lock(p, expected=None):
        p = Path(p).resolve()
        assert p.is_relative_to(ROOT) and 'holdout_sealed' not in p.parts
        h = sha(p)
        if expected is not None:
            assert h == expected, str(p)
        if str(p) in inputs:
            assert inputs[str(p)] == h
        inputs[str(p)] = h
        return str(p)

    for name in ('PLAN_ZH.md', 'SOURCE_QUALIFICATION.json', 'SOURCE_REVIEW.md',
                 'audit_worker.py', 'freeze_inputs.py', 'verify_saved_audit.py',
                 'SOURCE_NUMERIC_REVIEW.md', 'SYNTHETIC_ACCEPTANCE.json'):
        lock(HERE / name)
    for source_path in sorted((HERE / 'sources').iterdir()):
        if source_path.is_file():
            lock(source_path, qualification['source_files'][str(source_path.resolve())])
    data = read(OLD / 'protocol/DATA.json')
    lock(OLD / 'protocol/DATA.json')
    admission = read(data['admission'] + '/admission.json')
    lock(Path(data['admission']) / 'admission.json', data['admission_sha256'])
    bars = {freq: lock(Path(data['admission']) / f'bars_{freq}.npz',
                       admission['files'][f'bars_{freq}.npz']) for freq in ('day', 'month')}
    assert admission['split_actions'] == [{'date': '2020-08-31', 'ratio': 4.0}]
    selection_path = OLD / 'protocol/SELECTED_JOBS.json'
    lock(selection_path)
    jobs = read(selection_path)['jobs']
    results_path = OLD / 'analysis/RESULTS.json'
    lock(results_path)
    old_results = read(results_path)
    records, inherited = [], []
    for task, task_hash in sorted(data['task_files'].items()):
        taskpath = OLD / 'cache' / task / 'TASK.json'
        lock(taskpath, task_hash)
        meta = read(taskpath)
        if meta['frequency'] in ('hour', 'minute'):
            oldrow = next(r for r in old_results['pooled']
                          if r['task'] == task and r['variant'] == 'long_aligned')
            source_task = next(r for r in admission['tasks'] if r['profile'] == meta['profile']
                               and r['horizon'] == meta['horizon'] and r['target'] == meta['target'])
            assert source_task['first_decision'].startswith('2021-')
            inherited.append({**oldrow, 'first_eligible_decision': source_task['first_decision'],
                              'scope': 'inherited saved scalar only; no intraday market arrays read or recomputed',
                              'future_2020_split_possible': False})
            continue
        folds = []
        for fn, fold in sorted(meta['folds'].items(), key=lambda item: int(item[0])):
            targets = lock(fold['targets'], fold['targets_sha256'])
            directory = Path(fold['models']['mlp']['summary']).parent.parent
            receipt_path = directory / 'fold_receipt.json'
            lock(receipt_path)
            receipt = read(receipt_path)
            assert receipt['config']['profile'] == meta['profile']
            assert receipt['config']['horizon'] == meta['horizon']
            assert receipt['config']['target'] == meta['target']
            assert receipt['Q3_values_read'] is False
            models = []
            for model, details in receipt['models'].items():
                models.append({'name': model, 'seed': 2026,
                               'prediction': lock(directory / f'prediction_{model}.npz',
                                                  details['prediction_sha256'])})
            assert len(models) == 8
            selected = [j for j in jobs if j['profile'] == meta['profile']
                        and j['horizon'] == meta['horizon'] and j['target'] == meta['target']
                        and j['fold'] == int(fn) and j['family'] == 'tiny'
                        and j['field'] == 'long' and j['loss'] == 'aligned']
            assert sorted(j['seed'] for j in selected) == [2026, 2027, 2028]
            for job in sorted(selected, key=lambda j: j['seed']):
                done_path = OLD / 'runs' / job['id'] / 'DONE.json'
                lock(done_path)
                done = read(done_path)
                assert done['status'] == 'completed'
                models.append({'name': 'tiny_long_aligned', 'seed': job['seed'],
                               'prediction': lock(done_path.parent / 'prediction.npz',
                                                  done['prediction_sha256'])})
            folds.append({'fold': int(fn), 'targets': targets, 'models': models})
        assert len(folds) == 5
        records.append({k: meta[k] for k in ('task', 'profile', 'frequency', 'horizon', 'target')}
                       | {'folds': folds})
    assert len(records) == 6 and len(inherited) == 14
    protocol = {'schema': 1, 'status': 'frozen_before_market_array_decoding',
                'question': 'Effect of a previously announced future split on the original nominal-target benchmark',
                'inputs': inputs, 'bars': bars, 'task_records': records,
                'old_results': str(results_path), 'inherited_tasks': inherited,
                'known_ns': int(known.timestamp()) * 1000000000,
                'known_time_new_york': known.isoformat(), 'effective_date': '2020-08-31', 'ratio': 4.0,
                'start_epoch': time.time(), 'deadline_epoch': time.time() + args.seconds,
                'maximum_seconds': args.seconds, 'reserve_seconds': 60,
                'training_fits': 0, 'new_model_inferences': 0, 'GPU': False,
                'Q3_values_read_allowed': False, 'independent_market_confirmation': False,
                'original_labels_and_predictions_unchanged': True,
                'additional_announcement_information': True,
                'no_new_predictive_success_gate': True,
                'audit_only_one_event_no_multiple_independent_events_claim': True,
                'numerical_threads': 1, 'analysis_cpu': 24,
                'resume_requires_same_protocol_sha256': True}
    with target.open('x') as f:
        json.dump(protocol, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps({'protocol': str(target), 'sha256': sha(target),
                      'locked_inputs': len(inputs), 'deadline_epoch': protocol['deadline_epoch']}))


if __name__ == '__main__':
    main()
