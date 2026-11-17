"""Tabulate accepted saved losses; no source bars, labels, training or inference.

--check-only verifies the original receipts and saved-loss pooled arithmetic
without writing files. This is reporting, not a second test-data scoring pass.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
POST = BASE / 'market_run/postprocessing'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def csv_file(path, rows):
    require(not path.exists(), 'No report overwrite: ' + str(path))
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    require(set(os.sched_getaffinity(0)) <= {24, 26}, 'Analysis affinity required')
    outer = read(BASE / 'market_run/COMPLETE_SCREEN_SUPERVISOR.json')
    results = read(POST / 'RESULTS.json')
    review = read(POST / 'INDEPENDENT_REVIEW.json')
    protocol = read(BASE / 'EXECUTION_PROTOCOL.json')
    psha = sha(BASE / 'EXECUTION_PROTOCOL.json')
    require(psha == '8c7a9258ffb6852348299a0d2583e98f6f65ef4b456a6ef897bac14aa3f763f8', 'Original protocol')
    require(all(d['protocol_sha256'] == psha for d in (outer, results, review)), 'Receipt identity')
    require(outer['status'] == 'qualified_negative_screen' and outer['screen_complete'] is True,
            'This closeout reports the qualified negative screen only')
    require(review['status'] == 'PASS' and not review['screen_passed']
            and not results['screen_conjunction_passed'] and not outer['screen_passed'], 'Decision agreement')
    require(outer['main_results_sha256'] == sha(POST / 'RESULTS.json')
            and outer['independent_review_sha256'] == sha(POST / 'INDEPENDENT_REVIEW.json'), 'Result hashes')
    require(all(d['finished_epoch'] <= protocol['deadline_epoch'] for d in (outer, results, review)),
            'Same original deadline')
    require(all(p['terminal'] and p['exit_code'] == 0 for p in outer['phases']), 'Pipeline terminal')
    for entry in protocol['locks']:
        require(sha(Path(entry['path'])) == entry['sha256'], 'Frozen source drift')
    fold_rows = []
    pooled = {}
    decoded = []
    for source in sorted(POST.glob('*/STATE_SCALER.json')):
        state = read(source)['state']
        task = state['task']
        with np.load(source.parent / 'T_PREDICTIONS_AND_LOSSES.npz', allow_pickle=False) as archive:
            # NPZ members are decoded lazily. Never request truth/raw predictions.
            keys = [k for k in archive.files if k.startswith('loss_') and k.endswith('_endpoint')]
            require(len(keys) == 21, 'Complete predictor loss matrix')
            zero = archive['loss_zero_endpoint']
            n = len(zero)
            for key in keys:
                name = key[len('loss_'):-len('_endpoint')]
                endpoint = archive[key]
                pathloss = archive['loss_' + name + '_path']
                require(endpoint.shape == pathloss.shape == zero.shape, 'Saved loss shape')
                require(np.isfinite(endpoint).all() and np.isfinite(pathloss).all(), 'Finite saved loss')
                ep = float(endpoint.sum(dtype=np.float64))
                pp = float(pathloss.sum(dtype=np.float64))
                row = dict(task=task, fold=state['fold'], state_uid=state['state_uid'],
                           predictor=name, T_origins=n, endpoint_MSE=ep/n, path_MSE=pp/n,
                           endpoint_gain_overzero_pct=100*(1-ep/float(zero.sum(dtype=np.float64))))
                fold_rows.append(row)
                old = pooled.setdefault((task, name), [0, 0.0, 0.0])
                old[0] += n
                old[1] += ep
                old[2] += pp
                decoded.extend([key, 'loss_' + name + '_path'])
    pooled_rows, gate_rows, ci_rows = [], [], []
    for primary in results['primary_results']:
        task = primary['task']
        for name, expected in primary['pooled_MSE'].items():
            n, ep, pp = pooled[(task, name)]
            require(n == primary['T_origin_occurrences'], 'Pooled sample occurrences')
            for value, target in [(ep/n, expected['endpoint']), (pp/n, expected['path'])]:
                require(math.isclose(value, target, rel_tol=1e-12, abs_tol=1e-18), 'Pooled loss mismatch')
            pooled_rows.append(dict(task=task, predictor=name, T_origin_occurrences=n,
                                    endpoint_MSE=ep/n, path_MSE=pp/n,
                                    endpoint_gain_overzero_pct=100*(1-(ep/n)/primary['pooled_MSE']['zero']['endpoint'])))
        gate_rows.extend(dict(task=task, gate=name, passed=value) for name, value in primary['gates'].items())
        for interval in primary['endpoint_intervals']:
            row = dict(task=task)
            for key, value in interval.items():
                row[key] = json.dumps(value) if isinstance(value, (dict, list)) else value
            ci_rows.append(row)
    alpha_rows = []
    for mid, record in read(POST / 'GLOBAL_ALPHAS_FROZEN.json')['manifest'].items():
        alpha_rows.append(dict(task=record['state']['task'], fold=record['state']['fold'],
                               model_id=mid, alpha=record['alpha'], model_sha256=record['model_sha256']))
    training_rows = []
    for source in sorted((BASE / 'market_run/pairs').glob('*/DONE.json')):
        done = read(source)
        for arm, best in done['best_inference'].items():
            training_rows.append(dict(job=source.parent.name, task=done['state']['task'],
                                      fold=done['state']['fold'], seed=done['seed'], arm=arm,
                                      epochs_executed=done['epoch'], common_steps=done['common_steps'],
                                      best_epoch=best['best_epoch'], checkpoint_sha256=best['sha256']))
    receipt = dict(status='accepted_saved_loss_reporting', protocol_sha256=psha,
                   asof_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                   source_bars_read=False, labels_decoded=False, raw_predictions_decoded=False,
                   inference_or_training=False, bootstrap_rerun=False, Q3_read=False,
                   per_state_loss_rows=len(fold_rows), pooled_loss_rows=len(pooled_rows),
                   interval_rows=len(ci_rows), gate_rows=len(gate_rows), alpha_rows=len(alpha_rows),
                   training_rows=len(training_rows), original_screen_passed=False,
                   research_goal_achieved=False)
    if not args.check_only:
        out = BASE / 'reporting'
        out.mkdir(exist_ok=False)
        for name, rows in [('STATE_LOSSES.csv', fold_rows), ('POOLED_LOSSES.csv', pooled_rows),
                           ('ENDPOINT_INTERVALS.csv', ci_rows), ('GATES.csv', gate_rows),
                           ('CALIBRATION_ALPHAS.csv', alpha_rows), ('TRAINING_SELECTION.csv', training_rows)]:
            csv_file(out / name, rows)
        receipt['report_files_sha256'] = {p.name: sha(p) for p in out.glob('*.csv')}
        receipt['summary_script_sha256'] = sha(Path(__file__))
        (out / 'REPORTING_ACCEPTANCE.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
