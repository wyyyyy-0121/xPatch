"""Requirement ledger: distinguish completed artifacts from the unmet research goal."""
from study import *
from datetime import datetime


def main():
    assert source_hashes() == read(BASE/'protocol/CODE_FREEZE.json')['hashes']
    for filename in ('ANALYSIS_FREEZE.json', 'CANDIDATE_ANALYSIS_FREEZE.json'):
        frozen = read(BASE/'protocol'/filename)
        assert all(sha(BASE/'code'/name) == digest for name, digest in frozen['files'].items()), filename
    assert read(BASE/'audit/CANDIDATE_RULE_TIMING.json')['rules_precede_full_replication_analysis']
    required_status = {
        'SCREENING_COMPLETE.json': 'all_screening_jobs_complete',
        'REPLICATION_COMPLETE.json': 'all_admitted_replication_cells_complete',
        'CALIBRATION_COMPLETE.json': 'completed', 'CAP_COMPLETE.json': 'completed',
        'CANDIDATE_COMPLETE.json': 'all_candidate_cells_complete',
        'CANDIDATE_CALIBRATION_COMPLETE.json': 'completed',
        'INFORMATION_CONTROLS_COMPLETE.json': 'completed',
        'CROSS_SCALE_CONTROLS_COMPLETE.json': 'completed',
        'ORIGINAL_SHRINK_COMPLETE.json': 'completed',
        'RECOVERY_CHECK.json': 'passed', 'STATISTICS_CHECK.json': 'passed',
        'STATISTICS_REPRODUCED.json': 'passed', 'INPUT_ONLY_REPLAY.json': 'passed',
        'NUMERIC_AND_DOCUMENT_CHECK.json': 'passed', 'VISUAL_REVIEW.json': 'passed', 'RELEASE.json': 'passed'}
    for filename, expected in required_status.items():
        assert read(BASE/'audit'/filename)['status'] == expected, filename
    visual = read(BASE/'audit/VISUAL_REVIEW.json')
    for document in visual['documents']:
        for suffix, key in (('.pdf', 'pdf_sha256'), ('.docx', 'docx_sha256'), ('.md', 'source_sha256')):
            assert sha(BASE/'manuscript'/(document['document']+'_EN'+suffix)) == document[key]
    screen = read(BASE/'analysis/SCREENING_RESULTS.json')
    rep = read(BASE/'analysis/REPLICATION_RESULTS.json'); result = read(BASE/'analysis/CANDIDATE_RESULTS.json')
    assert screen['matched_interventions_verified'] and len(rep['original_restorations']) == 20
    assert all(r['model_and_stopping_exact'] for r in rep['original_restorations'])
    assert result['task_count'] == 20 and len(result['exact_stage_c_overlap']) == 45
    assert not list((BASE/'runs').glob('*/ERROR.json'))
    budget = read(BASE/'protocol/BUDGET.json')
    all_done = [read(path) for path in (BASE/'runs').glob('*/DONE.json')]
    assert len(all_done) == 375
    for done in all_done:
        assert done['status'] == 'completed' and done['deadline_epoch'] == budget['deadline_epoch']
        assert len(done['affinity']) <= 2 and not done['Q3_values_read']
        assert datetime.fromisoformat(done['at_utc']).timestamp() < budget['deadline_epoch']
    assert not result['independent_confirmation'] and not result['method_success_claimed']
    release = read(BASE/'audit/RELEASE.json')
    assert sha(release['archive']) == release['archive_sha256']
    checks = [
        {'requirement': 'original title and five-scale twenty-task scope', 'status': 'verified',
         'evidence': ['manuscript/MAIN_PAPER_EN.md', 'analysis/CANDIDATE_RESULTS.json']},
        {'requirement': 'matched audit and original restoration', 'status': 'verified',
         'evidence': ['analysis/SCREENING_RESULTS.json', 'analysis/REPLICATION_RESULTS.json']},
        {'requirement': 'mechanism explanation', 'status': 'partly_supported_with_limits',
         'evidence': ['analysis/REPLICATION_CONTRASTS.csv', 'analysis/STRUCTURAL_PROBES.json'],
         'limit': 'tested LSTM/xPatch implementations and specific hypotheses; not all architectures or market mechanisms'},
        {'requirement': 'training cap confound', 'status': 'verified_for_hourly_H1_only',
         'evidence': ['analysis/CAP_RESULTS.json']},
        {'requirement': 'single evidence-motivated candidate tested across original tasks', 'status': 'verified',
         'evidence': ['protocol/CANDIDATE_RULES.json', 'analysis/CANDIDATE_RESULTS.json']},
        {'requirement': 'stable predictive superiority over strong controls', 'status': 'not_achieved',
         'evidence': ['analysis/CANDIDATE_RESULTS.json:primary_performance_gates']},
        {'requirement': 'eligible independent forecasting confirmation', 'status': 'not_executed',
         'evidence': ['protocol/CONFIRMATION_DESIGN_ZH.md'], 'limit': 'Q3 remains sealed, exposed history cannot be relabeled'},
        {'requirement': 'shared-versus-independent horizon market comparison', 'status': 'not_executed_separate_scope',
         'evidence': ['PLAN_ZH.md', 'protocol/CONFIRMATION_DESIGN_ZH.md']},
        {'requirement': 'paper, rendered layout, saved-function and detached arithmetic reproduction', 'status': 'verified',
         'evidence': ['audit/DOCUMENT_BUILD.json', 'audit/VISUAL_REVIEW.json',
                      'audit/INPUT_ONLY_REPLAY.json', 'audit/RELEASE.json']},
        {'requirement': 'resource ceiling, checkpoints and unchanged absolute deadline', 'status': 'verified',
         'evidence': ['protocol/BUDGET.json', 'audit/RECOVERY_CHECK.json', 'audit/RUNTIME_RESOURCE_CHECK.json'],
         'limit': 'CPU implementation, eight-slot locked queues and per-fit affinity receipts; no GPU training'}]
    final = {'status': 'bounded_phase_delivered_research_goal_not_achieved', 'requirements': checks,
             'fresh_configurations': 345, 'cap_checks': 30, 'all_registered_configurations': 375,
             'predictive_success': False, 'independent_confirmation': False,
             'Q3_values_read': False, 'research_goal_achieved': False, 'goal_should_remain_active': True,
             'training_budget_deadline_epoch': budget['deadline_epoch'],
             'latest_fit_completion_epoch': max(datetime.fromisoformat(r['at_utc']).timestamp() for r in all_done),
             'archive': release['archive'], 'archive_sha256': release['archive_sha256'], 'at_utc': now()}
    atomic(BASE/'audit/FINAL_AUDIT.json', final)
    ledger = read(BASE/'analysis/GOAL_LEDGER.json')
    assert next(row for row in ledger['requirements'] if row['requirement'] == '论文增补与计算复现')['status'] == '已验证交付'
    print(json.dumps({'status': final['status'], 'research_goal_achieved': False,
                      'independent_confirmation': False, 'archive': final['archive']}), flush=True)


if __name__ == '__main__':
    main()
