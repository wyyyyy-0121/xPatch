"""Bind the plan's deliverables to current evidence; never fit or resume."""
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import subprocess

from multiscale.historical_continuation import digest, matched, preflight, read, save
from multiscale.historical_coverage import snapshot

ROOT = Path('/home/ayaka/wslcode/aapl-research')
OUT = ROOT / 'outputs/multiscale'
DEST = Path(__file__).resolve().parent
MANUSCRIPT = OUT / 'private_working_manuscripts/20261107T211544188288Z'
PREPARED = OUT / 'remaining_compute_proposals/20261108_remaining15_controller_v2'
PACKAGE = OUT / 'historical_inference_packages/20261108_all20_tasks'
ADMISSION = OUT / 'admission/alpaca_20261107T052228573242Z'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    assert os.sched_getaffinity(0) == {24, 26}
    checked = {}

    def bind(path, expected=None):
        path = Path(path).resolve()
        actual = digest(path)
        assert expected is None or actual == expected, str(path)
        checked[str(path)] = actual
        return read(path) if path.suffix == '.json' else actual

    plan = ROOT / 'MULTISCALE_RESEARCH_PLAN.md'
    bind(plan)
    current = snapshot()
    assert current['completed_folds'] == 85 and current['planned_folds'] == 100
    assert current['five_fold_tasks'] == 12 and current['planned_tasks'] == 20
    assert current['status_counts'] == {'completed': 85, 'paused_budget_exhausted': 7, 'pending': 8}
    release = bind(MANUSCRIPT / 'INTERIM_RELEASE_RECEIPT.json')
    for relative, expected in release['supplement_files_sha256'].items():
        bind(MANUSCRIPT / relative, expected)
    results = read(MANUSCRIPT / 'all_task_results/RESULTS.json')
    assert results['status'] == 'partial_historical_development_results'
    assert results['completed_folds'] == 85 and results['complete_tasks'] == 12
    assert len(results['tasks']) == 20 and not results['independent_test'] and not results['p_values_reported']
    assert sum(len(t['completed_folds']) for t in results['tasks']) == 85
    assert all(len(t['completed_folds']) + len(t['missing_folds']) == 5 for t in results['tasks'])
    counts, tables = {}, {}
    for name, expected in results['csv_sha256'].items():
        path = MANUSCRIPT / 'all_task_results' / name
        bind(path, expected)
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        counts[name] = len(rows)
        tables[name] = rows
    assert counts['METRICS_POOLED.csv'] == 160 and counts['METRICS_BY_FOLD.csv'] == 800
    assert counts['METRICS_PER_STEP.csv'] == 2544 and counts['DESCRIPTIVE_INTERVALS.csv'] == 320
    assert counts['TRAINING_MODELS.csv'] == 400
    assert sum(r['status'] == 'completed' for r in tables['TRAINING_MODELS.csv']) == 340
    compilation = read(MANUSCRIPT / 'all_task_results/COMPILATION_PROTOCOL.json')
    assert digest(MANUSCRIPT / 'all_task_results/COMPILATION_PROTOCOL.json') == results['protocol_sha256']
    original_coverage = bind(compilation['coverage'], compilation['coverage_sha256'])
    assert original_coverage['queue_state_sha256'] == current['queue_state_sha256']
    restoration = read(MANUSCRIPT / 'COMBINED_ENDPOINT_RESTORATION.json')
    assert restoration['completed_folds'] == 85 and restoration['neural_models'] == 340
    assert restoration['all_arrays_exact'] and restoration['max_absolute_difference'] == 0
    assert len(restoration['checks']) == 85
    for item in restoration['checks']:
        bind(item['check'], item['sha256'])
    design = read(MANUSCRIPT / 'forecast_timing/DESIGN_TIMING_AUDIT.json')
    assert design['planned_folds_verified'] == 100 and design['completed_saved_fold_metadata_verified'] == 85
    assert design['paired_price_return_designs_verified'] == 50 and not design['Q3_values_read']
    restart = bind(MANUSCRIPT / 'REAL_TRAINING_RESTART_AUDIT.json')
    assert restart['total_optimizer_updates_in_audit'] == 8 and restart['fresh_processes'] == 6
    assert len(restart['checks']) == 2 and all(c['all_checkpoint_fields_exactly_equal'] for c in restart['checks'])
    for item in restart['checks']:
        bind(item['source_checkpoint'], item['source_checkpoint_sha256'])
    audit = bind(MANUSCRIPT / 'STUDY_PACKAGE_AUDIT.json')
    registry = bind(PACKAGE / 'REGISTRY.json', audit['registry_sha256'])
    assert sum(len(t['completed_folds']) for t in registry['tasks']) == 71
    assert audit['tasks_verified'] == 20 and audit['models_verified'] == 160
    assert not audit['core_comparison_complete'] and not registry['core_comparison_complete']
    for item in audit['results']:
        bind(item['check'], item['check_sha256'])
    for item in registry['tasks']:
        bind(PACKAGE / item['bundle'] / 'manifest.json', item['manifest_sha256'])
    permission = bind(OUT / 'permission_confirmations/20261108_ALPACA_KIBOT_USER_CONFIRMED.json')
    assert permission['evidence_type'] == 'direct_user_confirmation'
    assert set(permission['providers']) == {'Alpaca', 'Kibot'} and permission['training_allowed_as_confirmed']
    admission = bind(ADMISSION / 'admission.json')
    assert admission['Q3_values_read'] is False and admission['development_cutoff_new_york'] == '2026-06-30'
    for relative, expected in admission['files'].items():
        bind(ADMISSION / relative, expected)
    word = read(MANUSCRIPT / 'WORD_CONTENT_VERIFICATION.json')
    assert word['outcome_snapshot_completed_folds'] == word['timing_saved_file_snapshot_completed_folds'] == 85
    assert word['checks']['all_markdown_table_cells_match_word'] and not word['visual_review_pending']
    visual = read(MANUSCRIPT / 'VISUAL_REVIEW.json')
    assert visual['pages'] == 20
    for relative, expected in (visual['contact_sheets'] | visual['preview_page_sha256']).items():
        bind(MANUSCRIPT / relative, expected)
    references = bind(MANUSCRIPT / 'reference_evidence_20261108/REFERENCE_EVIDENCE_VERIFICATION.json')
    assert references['checks']['entries'] == 17 and references['checks']['source_hashes_match']
    assert references['checks']['identifiers_titles_and_author_names_match']
    prepared = bind(PREPARED / 'PREPARATION_RECEIPT.json')
    manifest = bind(PREPARED / 'MANIFEST.json', prepared['manifest_sha256'])
    preflight(manifest)
    for name, expected in prepared['verification_receipts'].items():
        bind(PREPARED / name, expected)
    assert not prepared['additional_budget_approved'] and not prepared['training_started']
    for job in manifest['jobs']:
        if job['original_process']:
            assert matched(job['original_process'])['state'] == 'T'
    local = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    remote = subprocess.run(['git', 'ls-remote', '--exit-code', 'origin', 'refs/heads/codex/aapl-return12-implementation'],
                            cwd=ROOT, capture_output=True, text=True, check=True, timeout=30).stdout.split()[0]
    assert local == remote == 'f70b62e3dd4036614998d2499b2c1d554a08338a'
    requirements = [
        {'plan': '1,2,10.tasks', 'requirement': 'All 20 tasks and 100 original fold slots retained',
         'status': 'verified', 'actual': [20, 100]},
        {'plan': '3', 'requirement': 'Admitted historical data, user permission and accepted annual design',
         'status': 'verified_with_disclosed_limits', 'actual': 'Original Alpaca; about ten years; direct user permission confirmation'},
        {'plan': '4,5,10.protocol', 'requirement': 'Original configuration, chronology and shared origin identities',
         'status': 'verified', 'actual': '100 designs; 400 partition hashes; 50 paired designs; Q3 remains sealed'},
        {'plan': '4,8,10.results', 'requirement': 'All original eight-model, five-fold comparisons',
         'status': 'incomplete', 'actual': '85/100 folds; 340/400 neural fits; seven frozen, eight unstarted'},
        {'plan': '5,10.reproducibility', 'requirement': 'Full arithmetic and metadata verification',
         'status': 'incomplete', 'actual': 'All 85 completed folds audited; final 100-fold compiler still required'},
        {'plan': '4,5.9,6,8,10.mechanism', 'requirement': 'Additional seeds and shared-horizon mechanism',
         'status': 'conditional_not_admitted_not_claimed', 'actual': 'Core seed 2026; post-hoc monthly seeds separate; mechanism not run'},
        {'plan': '5.8,5.10', 'requirement': 'Independent confirmation and any Holm-adjusted confirmatory claims',
         'status': 'conditional_not_claimed', 'actual': 'Exposed historical development; no development p-values; Q3 sealed'},
        {'plan': '10.package', 'requirement': 'All-task prediction interface and completed package',
         'status': 'incomplete', 'actual': '20 callable interfaces and 160 model calls audited, but registry is the earlier 71-fold partial snapshot'},
        {'plan': '10.package', 'requirement': 'Saved predictor reproduction and genuine process restart',
         'status': 'verified_for_completed_scope', 'actual': '340 predictors; 8,449,488 exact paths; real GRU/xPatch restart compared all checkpoint fields'},
        {'plan': '10.report', 'requirement': 'Final English manuscript and rendered contents',
         'status': 'incomplete', 'actual': 'Checked 85-fold working manuscript: 20 pages, 37 tables, 30 equations, 17 references'},
        {'plan': '9,10.git', 'requirement': 'Code and instructions committed and pushed; private artifacts retained locally',
         'status': 'verified', 'actual': local},
        {'plan': '7,8,9', 'requirement': 'Budget and job protection', 'status': 'verified',
         'actual': 'Original limits exhausted; seven workers stopped; no additional training authorization or activation'},
    ]
    result = {'status': 'full_plan_audit_confirms_study_incomplete_at_compute_approval_boundary',
              'at_utc': datetime.now(timezone.utc).isoformat(), 'plan_sha256': digest(plan),
              'requirements': requirements, 'artifact_files_hash_checked': len(checked), 'artifacts': checked,
              'csv_rows': counts, 'completed_folds': 85, 'required_folds': 100,
              'git_commit': local, 'remote_verified_commit': remote,
              'new_training_updates': 0, 'processes_resumed': False, 'Q3_values_read': False,
              'core_study_complete': False, 'pending_budget_is_not_data_permission': True,
              'next_required_actions': ['Obtain reply to the existing additional-compute request.',
                  'Finish exactly the fifteen missing original folds under the approved cap.',
                  'Recompute all 100 folds, restore the new predictors, and audit all completed metadata.',
                  'Export a new complete inference registry without --allow-partial and audit it.',
                  'Refresh the full English manuscript and figures from final evidence, then check content and rendering.'],
              'script_sha256': digest(__file__)}
    assert not (DEST / 'PLAN_COMPLETION_AUDIT.json').exists()
    save(DEST / 'PLAN_COMPLETION_AUDIT.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'completed_folds', 'required_folds',
                      'artifact_files_hash_checked', 'remote_verified_commit', 'new_training_updates')}))


if __name__ == '__main__':
    main()
