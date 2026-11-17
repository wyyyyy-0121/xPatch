from common import *
from datetime import datetime,timezone

def main():
    results=read(BASE/'analysis/RESULTS.json');jobs=read(BASE/'protocol/SELECTED_JOBS.json')['jobs']
    replay=read(BASE/'audit/PREDICTOR_REPLAY.json');build=read(BASE/'audit/DOCUMENT_BUILD.json')
    released=read(BASE/'audit/RELEASE.json');visual=read(BASE/'audit/VISUAL_REVIEW.json')
    counts=read(BASE/'diagnostics/DIAGNOSTICS_20_TASKS.json')
    budget=read(BASE/'protocol/BUDGET.json')
    complete=read(BASE/'audit/TRAINING_COMPLETE.json')
    checks={
        'all_twenty_original_tasks_retained':len(read(BASE/'protocol/DATA.json')['task_files'])==20,
        'all_twenty_new_primary_tasks_complete':results['new_primary_tasks']==20,
        'all_selected_new_fits_complete':results['completed_new_fits']==len(jobs)==complete['jobs'],
        'four_original_models_diagnosed_all_twenty_tasks':len(counts['rows'])==400,
        'original_loss_probes_complete':len(results['original_loss_probes'])==16,
        'two_factor_tasks_complete':len(results['mechanism_performance'])==2,
        'fusion_replay_complete':len(results['fusion'])==2 and read(BASE/'fusion/COMPLETE.json')['gate_fits']==18,
        'checkpoint_actual_exit_and_restore_passed':read(BASE/'audit/RECOVERY_CHECK.json')['status']=='passed',
        'all_new_predictors_replayed':replay['status']=='passed' and replay['predictors']==len(jobs),
        'frozen_training_source_unchanged':code_hashes()==read(BASE/'protocol/CODE_FREEZE.json')['hashes'],
        'paper_numeric_and_document_check_passed':read(BASE/'audit/NUMERIC_AND_DOCUMENT_CHECK.json')['status']=='passed',
        'two_column_main_and_page_count_valid':build['main_page_target_4_to_6'],
        'all_pages_actually_reviewed':visual['status']=='passed' and visual['pages_reviewed']==sum(d['pages'] for d in build['documents']),
        'detached_package_inference_passed':released['status']=='passed' and released['detached_five_scale_inference_executed'],
        'detached_new_statistics_recomputed':released['detached_statistics_recomputed']['status']=='passed',
        'archive_hash_matches':sha(released['archive'])==released['archive_sha256'],
        'training_finished_by_absolute_deadline':datetime.fromisoformat(complete['at_utc']).timestamp()<=budget['deadline_epoch']+60,
        'budget_not_extended_on_resume':budget['deadline_epoch']<=read(BASE/'audit/RECOVERY_CHECK.json')['deadline_preserved'],
        'Q3_unread_and_unscored':results['Q3_values_read'] is False and released['Q3_values_read'] is False,
        'independent_validation_not_claimed':results['independent_confirmation'] is False}
    status='paper_and_reproducibility_delivery_complete' if all(checks.values()) else 'incomplete'
    atomic(BASE/'audit/FINAL_COMPLETION_AUDIT.json',{'status':status,'checks':checks,
        'paper_delivery_complete':all(checks.values()),'predictive_performance_goal_passed':results['both_mechanism_tasks_pass'],
        'independent_confirmation':False,'shared_horizon_market_experiment_completed':False,
        'email_pending_author':True,'Q3_values_read':False,'at_utc':datetime.now(timezone.utc).isoformat(),
        'original_title':'Multi-Horizon Forecasting of Apple Stock Prices and Returns Across Temporal Scales'})
    print(json.dumps({'status':status,'failed':[k for k,v in checks.items() if not v],
        'predictive_performance_goal_passed':results['both_mechanism_tasks_pass']}),flush=True)
    if not all(checks.values()):raise SystemExit(1)

if __name__=='__main__':main()
