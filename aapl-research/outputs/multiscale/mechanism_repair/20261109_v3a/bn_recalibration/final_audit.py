"""Audit actual intervention, resources and limits without equating delivery to success."""
from pathlib import Path
from datetime import datetime
import os,json,time
import numpy as np
import torch
import diagnostic_v2 as d

HERE=Path(__file__).resolve().parent

def main():
    assert sorted(os.sched_getaffinity(0))==[24,26] and os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    revision=d.read(HERE/'NUMERICAL_ACCEPTANCE_REVISION.json')
    original=(HERE/'diagnostic.py').read_text();revised=(HERE/'diagnostic_v2.py').read_text()
    diff=revision['literal_diff'];assert revised==original.replace(diff['old'],diff['new'])
    assert d.sha(HERE/'diagnostic.py')==revision['original_source_sha256']
    assert d.sha(HERE/'diagnostic_v2.py')==revision['revised_source_sha256']
    frozen=d.read(HERE/'SOURCE_FREEZE.json');assert d.sha(HERE/'BN_RULES.json')==frozen['identity']['BN_RULES.json']
    assert d.source_hashes()==frozen['identity']['original_source_hashes']
    for name,digest in d.read(HERE/'AUXILIARY_SOURCE_FREEZE.json')['files'].items():assert d.sha(HERE/name)==digest
    gate=d.read(HERE/'GATE_AND_ADMISSION.json');start=d.read(HERE/'OUTER_START.json')
    assert gate['deadline_epoch']==d.DEADLINE and gate['at_epoch']<start['at_epoch']
    assert revision['at_epoch']<start['at_epoch']
    assert start['gate_sha256']==d.sha(HERE/'GATE_AND_ADMISSION.json')
    assert gate['full_tasks']==['minute_h60_future_log_return']
    assert gate['tasks']['hourly_h8_future_log_return']['screen_raw_directions']==[False,True]
    expected=[d.cfg_for('minute',60,fold,seed) for fold in range(1,6) for seed in (2026,2027,2028)]
    expected += [d.cfg_for('hourly',8,fold,2026) for fold in (1,2)]
    records=[]
    for cfg in expected:
        folder=HERE/'cells'/cfg['id'];inner=d.read(folder/'INNER_DONE.json');outer=d.read(folder/'OUTER_DONE.json')
        numeric=d.read(folder/'NUMERIC_ACCEPTANCE_V2.json');moments=d.read(folder/'MOMENTS_DONE.json')
        assert inner['status']==outer['status']=='completed'
        assert numeric['status']=='passed_revised_tolerance'
        assert numeric['source_outer_prediction_file_hash_verified']
        assert moments['all_parameters_byte_equal'] and moments['all_other_buffers_byte_equal'] and not moments['moment_labels_read']
        assert d.sha(folder/'adapted.pt')==inner['adapted_checkpoint_sha256']==moments['adapted_checkpoint_sha256']
        assert d.sha(folder/'outer_losses.npz')==outer['losses_sha256']
        _,done,saved=d.source(cfg);adapted=torch.load(folder/'adapted.pt',map_location='cpu',weights_only=True)
        changed=d.compare_states(saved['model'],adapted['model'])
        assert changed==moments['changed_buffers']
        assert all(r['revised_allclose'] for name in numeric['records'].values() for r in name['checks'])
        assert datetime.fromisoformat(outer['at_utc']).timestamp()<d.DEADLINE
        records.append({'job':cfg['id'],'only_running_mean_var_changed':True,'outer_before_deadline':True,
            'adapted_checkpoint_sha256':inner['adapted_checkpoint_sha256'],
            'max_numeric_abs_error':max(r['max_abs_error'] for item in numeric['records'].values() for r in item['checks'])})
    recovery=d.read(HERE/'RECOVERY_CHECK.json');assert recovery['status']=='passed'
    assert recovery['actual_SIGUSR1_exit_and_resume'] and recovery['independent_uninterrupted_BN3_recompute_float64_exact']
    controls=d.read(HERE/'CONTROL_AUDIT.json');assert controls['status']=='completed_all_registered_controls_for_admitted_tasks'
    assert set(controls['tasks'])==set(gate['full_tasks'])
    for source,digest in controls['source_file_hashes'].items():assert d.sha(source)==digest
    results=d.read(HERE/'RESULTS.json');assert not results['Q3_values_read'] and not results['method_success_claimed']
    final={'status':'passed','research_status':'bn_diagnostic_complete_research_goal_not_achieved','registered_cells':len(expected),
        'full_five_fold_three_seed_tasks':gate['full_tasks'],'hourly_screen_only_cells':2,
        'weights_retrained':False,'only_six_BN_running_buffers_changed':True,
        'moment_estimation_fit_inputs_only':True,'batch_acceptance':'documented_float32_tolerance_revision',
        'original_tolerance_failure_preserved':True,'recovery_verified':True,'affinity':[24,26],
        'GPU':False,'deadline_epoch':d.DEADLINE,'all_outer_predictions_completed_before_deadline':True,
        'method_success_claimed':False,'independent_confirmation':False,'Q3_values_read':False,
        'research_goal_achieved':False,'goal_should_remain_active':True,'cells':records,
        'results_sha256':d.sha(HERE/'RESULTS.json'),'control_audit_sha256':d.sha(HERE/'CONTROL_AUDIT.json'),
        'at_utc':d.now(),'at_epoch':time.time()}
    d.atomic(HERE/'FINAL_AUDIT.json',final)
    print(json.dumps({k:v for k,v in final.items() if k!='cells'}),flush=True)

if __name__=='__main__':main()
