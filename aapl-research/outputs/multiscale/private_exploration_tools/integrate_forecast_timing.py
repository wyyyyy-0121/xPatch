"""Add metadata-verified clock horizons to a new, immutable working draft."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path('/home/ayaka/wslcode/aapl-research')
OUT = ROOT / 'outputs/multiscale'
PARENT = OUT / 'private_working_manuscripts/20261107T200639663038Z'
AUDIT = OUT / 'historical_design_audits/20261108_all100_planned_metadata'


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def table(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |',
                       '| ' + ' | '.join(['---'] + ['---:'] * (len(header) - 1)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def number(value): return f'{value:.6g}'


def main():
    audit_path = AUDIT / 'DESIGN_TIMING_AUDIT.json'
    audit = json.loads(audit_path.read_text())
    assert audit['planned_folds_verified'] == 100
    assert audit['completed_saved_fold_metadata_verified'] == 81
    assert audit['paired_price_return_designs_verified'] == 50
    assert audit['price_target_prediction_values_materialized'] is False
    assert audit['Q3_values_read'] is False
    source = PARENT / 'PRIVATE_WORKING_MANUSCRIPT_EN.md'
    text = source.read_text(encoding='utf8')
    before = text
    hourly = [t for t in audit['tasks'] if t['profile'] == 'hourly' and t['target'] == 'future_close']
    rows = []
    for task in sorted(hourly, key=lambda t: t['horizon']):
        timing = task['planned_origin_timing']
        assert task['completed_folds_at_snapshot'] == 5
        assert timing == task['completed_origin_timing']
        rows.append([f"H{task['horizon']}", f"{timing['origins']:,}"] +
                    [number(timing[k]) for k in ('elapsed_hours_min', 'elapsed_hours_median',
                                                'elapsed_hours_p95', 'elapsed_hours_max')] +
                    [f"{100 * timing['cross_session_fraction']:.2f}%"])
    addition = (
        'An independent metadata audit reconstructed the original fit, stopping, outer-training and validation '
        'partitions for all 100 planned task-folds. All 400 partition hashes matched the recorded protocols, '
        'and all 50 paired price/return designs agreed. At the audit snapshot, saved origins and decision/target '
        'completion timestamps matched the admitted bar positions in each of 81 completed folds. The audit '
        'checked input and target observed masks, strict inner and outer label purges, and the protected cutoff '
        'without materializing close, target-value or prediction-value arrays. This checks design and metadata '
        'consistency; it supplies no additional completed experimental fold.\n\n'
        'Complete trading-hour bars are not consecutive clock hours. The following elapsed-time distribution '
        'uses every validation origin from all five completed folds of each hourly task. Price and return '
        'tasks have identical origins. Overnight, weekend and holiday gaps are retained.\n\n' +
        table(['Hourly horizon', 'Origins', 'Minimum h', 'Median h', '95th percentile h', 'Maximum h',
               'Cross-session paths'], rows) + '\n\n'
        'For H4 the median endpoint delay is 22 clock hours; for H8 it is 26 hours, and every H8 path crosses '
        'a session boundary. The longest H8 delay is 119 hours. These are elapsed durations between recorded '
        'bar completions, not guarantees of historical vendor delivery times. All minute paths remain within '
        'one regular session and have an endpoint delay of exactly H minutes. Appendix B and the supplement '
        'retain the full task design and fold-level timing records.\n\n')
    boundary = '### 4.3 Optional shared-horizon comparison'
    assert text.count(boundary) == 1
    text = text.replace(boundary, addition + boundary, 1)
    order = {'annual': 0, 'quarterly': 1, 'monthly': 2, 'hourly': 3, 'minute': 4}
    rows = []
    for task in sorted(audit['tasks'], key=lambda t: (order[t['profile']], t['horizon'], t['target'])):
        if task['target'] != 'future_close': continue
        timing = task['planned_origin_timing']
        rows.append([task['profile'].title() + f" H{task['horizon']}", f"{timing['origins']:,}"] +
                    [number(timing[k]) for k in ('elapsed_hours_min', 'elapsed_hours_median',
                                                'elapsed_hours_p95', 'elapsed_hours_max')] +
                    [f"{100 * timing['cross_session_fraction']:.2f}%"])
    appendix = (
        '## Appendix B. Forecast timing and original fold design\n\n'
        'This table describes all five original validation folds of each task, including unfinished minute '
        'folds. Price and return tasks share the same design, so each horizon appears once. Counts in this '
        'table are planned valid origins and must not be read as completed predictions or independent market '
        'observations. Numerical results in Section 5 and Appendix A remain at the separately recorded '
        '74-fold snapshot.\n\n' +
        table(['Task', 'Planned origins', 'Minimum h', 'Median h', '95th percentile h', 'Maximum h',
               'Cross-session paths'], rows) + '\n\n'
        'Durations are measured in clock hours between recorded decision and endpoint completion timestamps. '
        'Percentiles use linear interpolation over individual origins. Monthly and daily bar completions '
        'follow the conservative next-New-York-midnight convention described in Section 3; daylight-saving '
        'changes therefore enter the elapsed durations. A cross-session path means that its endpoint session '
        'label differs from the origin session label. The annual 47-window design uses short history '
        'and overlapping labels.\n\n'
        'The supplement provides all 100 fold designs, 400 partition hashes, fit/stop/train/validation counts, '
        'strict label-purge gaps, decision ranges and endpoint ranges. It distinguishes 81 completed saved-file '
        'checks at the metadata-audit snapshot from the original timing design of unfinished folds. It does '
        'not change the outcome snapshot, training settings, compute budget or sealed-period status.\n')
    text = text.rstrip() + '\n\n' + appendix
    # Removing only the inserted blocks must recreate every original character.
    assert text.replace(addition, '', 1).removesuffix('\n\n' + appendix).rstrip() == before.rstrip()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = OUT / 'private_working_manuscripts' / stamp
    destination.mkdir(parents=True, exist_ok=False)
    for name in ('figures', 'paper_artifacts', 'reference_evidence_20261108', 'all_task_results'):
        if (PARENT / name).is_dir(): shutil.copytree(PARENT / name, destination / name)
    for name in ('REAL_TRAINING_RESTART_AUDIT.json', 'STUDY_PACKAGE_AUDIT.json',
                 'COMBINED_78_FOLD_RESTORATION.json', 'ADDITIONAL_MINUTE_RESTORATION_AUDIT.json',
                 'FINAL_EDITORIAL_NOTES.md', 'REMAINING_REQUIREMENTS.md',
                 'DATA_PERMISSION_CONFIRMATION.json', 'DATA_PROVENANCE_MANIFEST.json',
                 'ACTUAL_NIGHT_RESUME_VERIFICATION.json', 'COVERAGE_SNAPSHOT.json',
                 'FIRST_H60_FULL_RESTORATION_AUDIT.json', 'RESTORATION_AUDIT.json',
                 'PRIVATE_ANNUAL_NONOVERLAP_RESULTS.csv'):
        if (PARENT / name).is_file(): shutil.copy2(PARENT / name, destination / name)
    shutil.copytree(AUDIT, destination / 'forecast_timing')
    manuscript = destination / source.name
    manuscript.write_text(text, encoding='utf8')
    receipt = {'status': 'forecast_timing_inserted_original_manuscript_preserved',
               'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'parent': str(PARENT), 'parent_markdown_sha256': digest(source),
               'audit': str(audit_path), 'audit_sha256': digest(audit_path),
               'numerical_result_snapshot_completed_folds': 74,
               'timing_saved_file_snapshot_completed_folds': 81, 'planned_fold_designs': 100,
               'new_tables': 2, 'prior_content_preserved': True,
               'source_sha256': digest(Path(__file__)), 'markdown_sha256': digest(manuscript)}
    (destination / 'TIMING_INSERTION_AUDIT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (destination / 'README.md').write_text(
        '# English working manuscript with verified clock horizons\n\n'
        'The numerical outcome snapshot remains 74/100 folds. This revision adds an independent metadata '
        'audit of all 100 original fold designs and the saved timing fields of 81 completed folds. All prior '
        'manuscript text, equations and numerical tables were preserved. Two timing tables distinguish '
        'completed hourly evidence from planned timing for unfinished minute folds.\n\n'
        'The separate predictor restoration snapshot covers 78 folds and the partial inference package '
        'covers 71 folds. These engineering snapshots do not replace the outcome snapshot. The final '
        '100-fold results, complete inference package and final editorial review are still pending.\n\n'
        'No training was started or resumed, no compute budget was increased, and no protected 2026Q3 '
        'observations were accessed by this revision. Permission for all Alpaca and Kibot data follows the '
        'researcher confirmation; the experiments use Alpaca.\n')
    print(json.dumps({'destination': str(destination), **receipt}, indent=2))


if __name__ == '__main__': main()
