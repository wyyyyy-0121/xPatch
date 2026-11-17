"""Create an evidence-checked local manuscript snapshot without running training.

This script uses only previously exposed development artifacts. It does not
read protected market data, modify experiments, or establish publication rights.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path('/home/ayaka/wslcode/aapl-research')
OUT = ROOT / 'outputs/multiscale'
SRC = OUT / 'private_working_manuscripts/20261107T153012343263Z'
THREE = OUT / 'private_three_scale_development/20261107T134128870773Z'
ANNUAL = OUT / 'private_annual_short_development/20261107T143734746737Z'
ANCHOR = OUT / 'private_anchored_price_exploration/20261107T135350353680Z'
H60 = OUT / 'private_minute_h60_aggregate/20261107T144910918784Z/PRIVATE_PARTIAL_AGGREGATE.json'
MINUTE = OUT / 'private_minute_first_fold_reports/20261107T152537878961Z/PRIVATE_FIRST_FOLDS.json'
SPLIT = OUT / 'private_split_neutral_monthly/20261107T145708694676Z/PRIVATE_RESULTS.json'
SEEDS = OUT / 'private_split_neutral_seed_replication/20261107T150345851174Z/PRIVATE_SEED_COMPARISON.json'
COVERAGE = OUT / 'private_coverage/20261107T160455479741Z.csv'
MODELS = ('drift', 'seasonal', 'ridge', 'mlp', 'gru', 'lstm', 'xpatch')
NEURAL = ('mlp', 'gru', 'lstm', 'xpatch')


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def near(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-13)


def section(text, start, end):
    require(text.count(start) == 1 and text.count(end) == 1, 'Missing or ambiguous manuscript section')
    return text.split(start, 1)[1].split(end, 1)[0]


def table(text, header):
    lines = text.splitlines()
    require(lines.count(header) == 1, 'Missing or repeated table header: ' + header)
    idx = lines.index(header)
    require(idx + 1 < len(lines) and lines[idx + 1].startswith('| ---'), 'Missing table separator')
    result = []
    for line in lines[idx + 2:]:
        if not line.startswith('|'):
            break
        result.append([part.strip() for part in line.strip().strip('|').split('|')])
    return result


def pct(value):
    return f'{100 * float(value):.2f}%'


def compare_pct(actual, expected, label):
    require(actual == pct(expected), f'{label}: manuscript {actual} vs source {pct(expected)}')


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


source_path = SRC / 'PRIVATE_WORKING_MANUSCRIPT_EN.md'
source_audit_path = SRC / 'PRIVATE_MANUSCRIPT_AUDIT.json'
source_audit = load(source_audit_path)
require(digest(source_path) == source_audit['markdown_sha256'], 'Source manuscript hash changed')
require(digest(Path(source_audit['source_appendix'])) == source_audit['source_appendix_sha256'],
        'Source appendix hash changed')
text = source_path.read_text(encoding='utf-8')
old_data = ('For local exploratory work, the user accepted an approximately ten-year history and a twelve-month '
            'annual lookback; this is a deviation from the original design.')
new_data = ('For local exploratory work, the user accepted approximately ten years of daily history. The annual '
            'variant uses a twelve-month lookback, which also differs from the original design.')
require(text.count(old_data) == 1, 'Annual data statement changed')
text = text.replace(old_data, new_data, 1)
require('PRIVATE WORKING MANUSCRIPT' in text and 'NOT FOR SUBMISSION OR DISTRIBUTION' in text,
        'Required private notice is absent')

coverage_rows = read_csv(COVERAGE)
coverage = {(r['profile'], int(r['horizon']), r['target']): r for r in coverage_rows}
require(len(coverage_rows) == 20 and len(coverage) == 20, 'Coverage inventory is not 20 unique tasks')
complete_tasks = sum(int(r['completed_full_folds']) == 5 for r in coverage_rows)
complete_folds = sum(int(r['completed_full_folds']) for r in coverage_rows)
expected_folds = sum(int(r['expected_folds']) for r in coverage_rows)
require((complete_tasks, complete_folds, expected_folds) == (12, 68, 100), 'Coverage snapshot changed')

three_csv = THREE / 'PRIVATE_THREE_SCALE_RESULTS.csv'
annual_csv = ANNUAL / 'PRIVATE_ANNUAL_RESULTS.csv'
three_check = load(THREE / 'PRIVATE_AGGREGATE_INDEPENDENT_CHECK.json')
annual_check = load(ANNUAL / 'PRIVATE_AGGREGATE_INDEPENDENT_CHECK.json')
require(digest(three_csv) == three_check['results_csv_sha256'], 'Three-scale results identity changed')
require(three_check['folds'] == 50 and three_check['tasks'] == 10 and three_check['independent_test'] is False,
        'Three-scale verification scope differs')

all_rows = read_csv(three_csv) + read_csv(annual_csv)
by_key = {(r['profile'], int(r['horizon']), r['target'], r['model']): r for r in all_rows}
require(len(all_rows) == 96 and len(by_key) == 96, 'Expected 96 unique model rows')
for check in three_check['checks']:
    path = THREE / f"PRIVATE_TASK_{check['profile']}_h{check['horizon']}_{check['target']}.json"
    require(digest(path) == check['task_sha256'], 'Task aggregate identity changed: ' + str(path))
    data = load(path)
    for model in ('last_close',) + MODELS:
        row = by_key[(check['profile'], check['horizon'], check['target'], model)]
        require(int(row['validation_origins']) == data['validation_origins'], 'Origin count mismatch')
        for field in ('endpoint_mse', 'relative_improvement'):
            require(near(row[field], data['metrics'][model][field]), 'Task aggregate mismatch')
for target, item in annual_check['targets'].items():
    for model, value in item['direct_endpoint_mse'].items():
        row = by_key[('annual', 12, target, model)]
        require(near(row['endpoint_mse'], value), 'Annual independent recalculation mismatch')
        require(int(row['validation_origins']) == item['validation_origins'], 'Annual origin mismatch')
for profile, horizon, target, model in by_key:
    row = by_key[(profile, horizon, target, model)]
    baseline = by_key[(profile, horizon, target, 'last_close')]
    require(float(baseline['endpoint_mse']) > 0, 'Undefined persistence comparison')
    require(near(float(row['relative_improvement']),
                 1 - float(row['endpoint_mse']) / float(baseline['endpoint_mse'])),
            'Relative improvement arithmetic mismatch')

label_keys = {
    'Hourly H8': ('hourly', 8), 'Hourly H1': ('hourly', 1),
    'Hourly H4': ('hourly', 4), 'Monthly H21': ('monthly', 21),
    'Quarterly H3': ('quarterly', 3), 'Annual H12 (short history)': ('annual', 12),
}
main_header = '| Task | Origins | Drift | Seasonal | Ridge | MLP | GRU | LSTM | xPatch |'
main_cells = 0
for title, next_title, target in (
    ('### Cumulative log-return target', '### Nominal close-price target', 'future_log_return'),
    ('### Nominal close-price target', '### Additional minute tasks: first-fold snapshot', 'future_close'),
):
    rows = table(section(text, title, next_title), main_header)
    require(len(rows) == 6 and set(row[0] for row in rows) == set(label_keys), 'Main table task coverage mismatch')
    for row in rows:
        profile, horizon = label_keys[row[0]]
        require(len(row) == 9, 'Unexpected main table width')
        base = by_key[(profile, horizon, target, 'last_close')]
        require(int(row[1]) == int(base['validation_origins']), 'Main table origin mismatch')
        for model, cell in zip(MODELS, row[2:]):
            compare_pct(cell, by_key[(profile, horizon, target, model)]['relative_improvement'], row[0] + ' ' + model)
            main_cells += 1

first = load(MINUTE)
require(first['status'] == 'private_first_fold_development_snapshot' and len(first['tasks']) == 5,
        'Minute first-fold snapshot scope differs')
minute_section = section(text, '### Additional minute tasks: first-fold snapshot',
                         '### Minute H60: 3 of 5 full-origin folds')
minute_rows = table(minute_section, main_header)
minute_by_label = {f"Minute H{x['horizon']} {'price' if x['target'] == 'future_close' else 'return'}": x
                   for x in first['tasks']}
require(len(minute_rows) == 5 and set(r[0] for r in minute_rows) == set(minute_by_label),
        'Minute first-fold table coverage mismatch')
minute_cells = 0
for row in minute_rows:
    item = minute_by_label[row[0]]
    verification = load(Path(item['verification']))
    receipt = Path(verification['fold_directory']) / 'fold_receipt.json'
    require(verification['fold_receipt_sha256'] == item['receipt_sha256'] == digest(receipt),
            'Minute first-fold receipt changed')
    require(int(row[1]) == item['validation_origins'], 'Minute origins mismatch')
    for model, cell in zip(MODELS, row[2:]):
        compare_pct(cell, item['relative_improvement'][model], row[0] + ' ' + model)
        minute_cells += 1

h60 = load(H60)
require(h60['completed_folds'] == [1, 2, 3] and h60['validation_origins'] == 226650,
        'H60 partial snapshot differs')
h60_header = '| Method | Relative endpoint MSE improvement |'
h60_rows = table(section(text, '### Minute H60: 3 of 5 full-origin folds',
                         '### Split sensitivity and label definition'), h60_header)
names = {'Zero return': 'last_close', 'Drift': 'drift', 'Seasonal': 'seasonal',
         'Ridge': 'ridge', 'MLP': 'mlp', 'GRU': 'gru', 'LSTM': 'lstm', 'xPatch': 'xpatch'}
require(len(h60_rows) == 8 and set(r[0] for r in h60_rows) == set(names), 'H60 table coverage mismatch')
for row in h60_rows:
    model = names[row[0]]
    pooled = sum(f['validation_origins'] * f['endpoint_mse'][model] for f in h60['folds']) / h60['validation_origins']
    require(near(pooled, h60['models'][model]['endpoint_mse']), 'H60 pooling mismatch')
    compare_pct(row[1], h60['models'][model]['relative_improvement'], 'H60 ' + model)

split_data = load(SPLIT)
seed_data = load(SEEDS)
require(seed_data['same_targets_and_splits'] is True and seed_data['independent_test'] is False,
        'Seed comparison scope changed')
for seed, folder in seed_data['sources'].items():
    require(digest(Path(folder) / 'PRIVATE_RESULTS.json') == seed_data['source_result_sha256'][seed],
            'Seed result identity changed')
split_section = section(text, '### Split sensitivity and label definition', '### Post-hoc price anchoring')
split_rows = table(split_section, h60_header)
require(len(split_rows) == 8 and set(r[0] for r in split_rows) == set(names), 'Split table coverage mismatch')
for row in split_rows:
    compare_pct(row[1], split_data['models'][names[row[0]]]['relative_improvement'], 'split ' + row[0])
seed_header = '| Method | Seed 2026 | Seed 2027 | Seed 2028 |'
seed_rows = table(split_section, seed_header)
require(len(seed_rows) == 4 and set(r[0] for r in seed_rows) == {'MLP', 'GRU', 'LSTM', 'xPatch'},
        'Seed table coverage mismatch')
for row in seed_rows:
    model = names[row[0]]
    for seed, cell in zip(('2026', '2027', '2028'), row[1:]):
        compare_pct(cell, seed_data['relative_mse_improvement_vs_zero_return'][model][seed], 'seed ' + model)

anchor_section = section(text, '### Post-hoc price anchoring', '## 6. Limitations and conclusion')
anchor_header = '| Task | Origins | MLP | GRU | LSTM | xPatch |'
anchor_rows = table(anchor_section, anchor_header)
require(len(anchor_rows) == 5, 'Anchored task count mismatch')
for row in anchor_rows:
    profile, horizon = label_keys[row[0]]
    data = load(ANCHOR / f'PRIVATE_TASK_{profile}_h{horizon}.json')
    require(int(row[1]) == data['validation_origins'], 'Anchored origin count mismatch')
    for model, cell in zip(NEURAL, row[2:]):
        compare_pct(cell, data['models'][model]['anchored_improvement_vs_persistence'], 'anchor ' + model)

history = []
for root in (THREE, ANNUAL):
    queue = load(root / 'PRIVATE_QUEUE_STATE.json')
    for key, job in queue['jobs'].items():
        require(job['status'] == 'completed', 'A referenced development fold is incomplete')
        folder = Path(job['fold_directory'])
        receipt_path = folder / 'fold_receipt.json'
        require(digest(receipt_path) == job['receipt_sha256'], 'Fold receipt hash mismatch: ' + key)
        receipt = load(receipt_path)
        config = receipt['config']
        require(receipt['status'] == 'completed' and receipt['independent_test'] is False and
                receipt['Q3_values_read'] is False, 'Fold scope differs: ' + key)
        require(config['seed'] == 2026 and config['max_epochs'] == 20, 'Unexpected training recipe')
        match = re.search(r'_f(\d+)$', key)
        require(match is not None, 'Cannot identify fold number: ' + key)
        for model in NEURAL:
            record = receipt['models'][model]
            summary_path = folder / model / 'training_summary.json'
            require(digest(summary_path) == record['training_summary_sha256'], 'Training summary hash mismatch')
            summary = load(summary_path)
            require(summary['status'] == 'completed' and summary['identity'] == record['training_identity'],
                    'Training identity mismatch')
            for field in ('epochs', 'steps', 'best_epoch', 'stopped_early'):
                require(summary[field] == record[field], 'Training summary field mismatch')
            history.append({
                'profile': config['profile'], 'horizon': config['horizon'], 'target': config['target'],
                'fold': int(match.group(1)), 'model': model, 'seed': config['seed'],
                'epochs': summary['epochs'], 'steps': summary['steps'], 'best_epoch': summary['best_epoch'],
                'stopped_early': summary['stopped_early'], 'cap_reached': summary['epochs'] >= config['max_epochs'],
                'receipt_sha256': job['receipt_sha256'], 'training_summary_sha256': record['training_summary_sha256'],
                'fold_directory': str(folder),
            })
require(len(history) == 240, 'Expected 240 neural model-fold histories')
require(len({(x['profile'], x['horizon'], x['target'], x['fold'], x['model']) for x in history}) == 240,
        'Duplicate neural model-fold history')
require(all(float(by_key[(p, h, 'future_close', model)]['relative_improvement']) < 0
            for p, h in set(label_keys.values()) for model in NEURAL),
        'The proposed persistence summary does not match all 24 price comparisons')

coverage_order = [
    ('annual', 12, 'Annual H12 (short history)'),
    ('quarterly', 3, 'Quarterly H3'), ('monthly', 21, 'Monthly H21'),
    ('hourly', 1, 'Hourly H1'), ('hourly', 4, 'Hourly H4'), ('hourly', 8, 'Hourly H8'),
    ('minute', 5, 'Minute H5'), ('minute', 15, 'Minute H15'),
    ('minute', 30, 'Minute H30'), ('minute', 60, 'Minute H60'),
]
coverage_table = ['| Task | Price folds | Return folds |', '| --- | ---: | ---: |']
for profile, horizon, label in coverage_order:
    p = coverage[(profile, horizon, 'future_close')]
    r = coverage[(profile, horizon, 'future_log_return')]
    coverage_table.append(f"| {label} | {p['completed_full_folds']}/{p['expected_folds']} | "
                          f"{r['completed_full_folds']}/{r['expected_folds']} |")

baseline_table = ['| Task | Origins per target | Price MSE (USD per share)² | Log-return MSE |',
                  '| --- | ---: | ---: | ---: |']
for profile, horizon, label in coverage_order[:6]:
    p = by_key[(profile, horizon, 'future_close', 'last_close')]
    r = by_key[(profile, horizon, 'future_log_return', 'last_close')]
    require(p['validation_origins'] == r['validation_origins'], 'Paired baseline origins differ')
    baseline_table.append(f"| {label} | {int(p['validation_origins']):,} | "
                          f"{float(p['endpoint_mse']):.6g} | {float(r['endpoint_mse']):.8f} |")

def stats_for(rows):
    return {
        'runs': len(rows),
        'cap': sum(r['cap_reached'] for r in rows),
        'early': sum(r['stopped_early'] for r in rows),
        'both': sum(r['cap_reached'] and r['stopped_early'] for r in rows),
        'median_steps': statistics.median(r['steps'] for r in rows),
    }

training_table = ['| Scale | Neural fits | At 20 epochs | Early-stop flag | Median updates |',
                  '| --- | ---: | ---: | ---: | ---: |']
for profile in ('annual', 'quarterly', 'monthly', 'hourly'):
    item = stats_for([r for r in history if r['profile'] == profile])
    training_table.append(f"| {profile.capitalize()} | {item['runs']} | {item['cap']} | "
                          f"{item['early']} | {item['median_steps']:g} |")
total = stats_for(history)
training_table.append(f"| Total | {total['runs']} | {total['cap']} | {total['early']} | {total['median_steps']:g} |")

old_intro = ('**Private development snapshot:** 12 of 20 tasks have five completed development folds, '
             'including two short-history annual tasks. The minute matrix and independent confirmation are incomplete. '
             'The optional shared-versus-separate multi-horizon comparison has not been run. '
             'All reported percentages compare endpoint MSE with the matched persistence or zero-return baseline; '
             'they are not investment returns.')
new_intro = ('**Private development snapshot:** as of 8 November 2026 at 00:04 Beijing time, '
             f'{complete_tasks} of 20 tasks have five completed development folds, including two short-history annual tasks; '
             f'{complete_folds} of {expected_folds} planned task-folds are complete. The remaining minute folds and independent '
             'confirmation are incomplete. The optional shared-versus-separate multi-horizon comparison has not been run. '
             'All reported percentages compare endpoint MSE with the matched persistence or zero-return baseline; '
             'they are not investment returns.\n\n'
             '### Development coverage and baseline magnitudes\n\n'
             'Only complete full-origin folds are counted. The user accepted approximately ten years of daily history; '
             'the short-history annual configuration uses a twelve-month lookback and differs from the original design.\n\n'
             + '\n'.join(coverage_table) + '\n\n'
             'The following persistence errors provide the denominator for the relative comparisons of the 12 complete tasks. '
             'Each row pools all validation origins across five folds; price and return errors have different units and '
             'are not averaged across targets. Nominal-price results include the effect of stock splits.\n\n'
             + '\n'.join(baseline_table))
require(text.count(old_intro) == 1, 'Development introduction changed')
text = text.replace(old_intro, new_intro, 1)

old_methods = ('The primary runs use seed 2026; the post-hoc split-neutral monthly comparison additionally uses seeds '
               '2027 and 2028. Realized update counts and final training settings remain to be documented.')
new_methods = ('The primary runs use seed 2026; the post-hoc split-neutral monthly comparison additionally uses seeds '
               '2027 and 2028. The realized update counts for the 12 completed tasks are summarized in Section 5; '
               'the remaining tasks and any final training settings still require documentation.')
require(text.count(old_methods) == 1, 'Training methods paragraph changed')
text = text.replace(old_methods, new_methods, 1)

training_text = ('For the six completed price tasks, none of the four unmodified neural recipes achieved a lower pooled '
                 'endpoint MSE than persistence: this is 0 of 24 task-by-model comparisons in this exposed development '
                 'snapshot. It does not establish an architecture-wide limit or a result for the incomplete minute tasks.\n\n'
                 '### Realized training budget\n\n'
                 'The table covers 240 neural model-fold fits from the 12 completed tasks. The cap and early-stop flag '
                 'can overlap when a stopping criterion is met at the final allowed epoch. Counts do not include the minute '
                 'snapshot or the post-hoc anchoring and split-neutral experiments. The number of optimizer updates, not '
                 'merely epochs, varies with the size of the inner-fit sample.\n\n'
                 + '\n'.join(training_table) + '\n\n'
                 'Reaching the epoch cap or setting an early-stop flag does not establish convergence. In particular, '
                 'small annual and quarterly fitting partitions limit what can be inferred about architecture performance.\n\n')
marker = '### Additional minute tasks: first-fold snapshot'
require(text.count(marker) == 1, 'Minute heading changed')
text = text.replace(marker, training_text + marker, 1)

history.sort(key=lambda r: (r['profile'], r['horizon'], r['target'], r['fold'], r['model']))
fields = ['profile', 'horizon', 'target', 'fold', 'model', 'seed', 'epochs', 'steps', 'best_epoch',
          'stopped_early', 'cap_reached', 'receipt_sha256', 'training_summary_sha256', 'fold_directory']
stream = io.StringIO(newline='')
writer = csv.DictWriter(stream, fieldnames=fields)
writer.writeheader()
writer.writerows(history)
history_text = stream.getvalue()

stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
directory = OUT / 'private_working_manuscripts' / stamp
directory.mkdir(parents=True, exist_ok=False)
manuscript = directory / 'PRIVATE_WORKING_MANUSCRIPT_EN.md'
hist_path = directory / 'TRAINING_HISTORY_AUDIT.csv'
manuscript.write_text(text, encoding='utf-8')
hist_path.write_text(history_text, encoding='utf-8', newline='')
evidence_files = {
    'source_manuscript': source_path, 'source_manuscript_audit': source_audit_path,
    'coverage_snapshot': COVERAGE, 'three_scale_results': three_csv,
    'three_scale_independent_check': THREE / 'PRIVATE_AGGREGATE_INDEPENDENT_CHECK.json',
    'annual_results': annual_csv, 'annual_independent_check': ANNUAL / 'PRIVATE_AGGREGATE_INDEPENDENT_CHECK.json',
    'minute_first_fold_snapshot': MINUTE, 'minute_h60_partial_aggregate': H60,
    'split_neutral_results': SPLIT, 'seed_comparison': SEEDS,
}
audit = {
    'status': 'private_working_manuscript_evidence_checked_not_for_submission',
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'source_sha256': {name: {'path': str(path), 'sha256': digest(path)} for name, path in evidence_files.items()},
    'script_sha256': digest(Path(__file__).resolve()),
    'manuscript_sha256': digest(manuscript),
    'training_history_csv_sha256': digest(hist_path),
    'checks': {
        'completed_tasks': complete_tasks, 'completed_task_folds': complete_folds,
        'planned_task_folds': expected_folds, 'original_model_rows_matched': len(all_rows),
        'main_table_percentage_cells': main_cells, 'minute_first_fold_percentage_cells': minute_cells,
        'minute_h60_percentage_cells': len(h60_rows), 'split_neutral_percentage_cells': len(split_rows),
        'seed_percentage_cells': len(seed_rows) * 3, 'anchor_percentage_cells': len(anchor_rows) * 4,
        'fold_receipt_hashes_verified': len(history) // 4,
        'training_summary_hashes_and_fields_verified': len(history),
        'neural_training_summary': total,
        'price_neural_vs_persistence_comparisons_better': 0,
    },
    'independent_test': False, 'publication_rights': 'not confirmed',
    'training_started': False, 'Q3_values_read': False,
    'limitations': [
        'Private already-exposed development snapshot, not submission-ready.',
        'Minute matrix, data and publication rights, protocol freeze, and independent confirmation remain incomplete.',
        'Document checks do not establish source accuracy, point-in-time availability, model convergence, or profitability.',
    ],
}
audit_path = directory / 'PRIVATE_MANUSCRIPT_EVIDENCE_AUDIT.json'
audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
print(json.dumps({'status': audit['status'], 'directory': str(directory), 'manuscript': str(manuscript),
                  'training_history': str(hist_path), 'audit': str(audit_path),
                  'training_summary': total, 'coverage': [complete_folds, expected_folds]}, ensure_ascii=False))
