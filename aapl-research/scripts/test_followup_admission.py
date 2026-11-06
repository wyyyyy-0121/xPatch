"""Receipt-chain fixtures test admission, not a claim of real model delivery."""
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pytest

from fusion.next.common import ROOT,PLAN,digest,identity,read_json,write_json
from fusion.next.evaluation import point_scores
from fusion.next.recipe import for_fit
from fusion.next.splits import fold
from scripts import followup_admission as admit
from scripts.followup_guard import process_identity


def seal(value):
    value['identity'] = identity(value)
    return value


@pytest.fixture
def evidence(monkeypatch):
    admit.FOLLOW.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=admit.FOLLOW,prefix='admission_fixture_') as temporary:
        root = Path(temporary)
        def save(name,value):
            write_json(root/name,value)
            return digest(root/name)
        scope = seal({'source_identity':admit.ENGINE,'seeds':[2026,2027,2028],
                      'validation':list(admit.QUARTERS),'training_budget_policy':{'total_coverage_equivalent_epochs':2}})
        numerical = seal({'fixture':'synthetic numerical policy'})
        speed = seal({'fixture':'synthetic speed policy'})
        monkeypatch.setattr(admit,'SCOPE',scope['identity'])
        monkeypatch.setattr(admit,'NUMERICAL',numerical['identity'])
        monkeypatch.setattr(admit,'SPEED',speed['identity'])
        limit = {'total_wall_hours':100,'deadline_utc':admit.DEADLINE}
        state = {'stage':'delivery_evidence_ready_and_summary_pushed','engineering_acceptance_passed':True,
                 'source_identity':admit.ENGINE,'scope_identity':admit.SCOPE}
        save('cloud/local_completion/state.json',state)
        save('cloud/local_completion/binding.json',{'source_identity':admit.ENGINE,'scope_identity':admit.SCOPE,
             'plan_sha256':digest(PLAN),'numerical_policy_identity':admit.NUMERICAL,
             'speed_policy_identity':admit.SPEED,'limit':limit})
        # The current PID with an intentionally different creation identity is
        # not a live receiver. The exact identity is tested separately below.
        save('cloud/local_completion/handoff_receiver_dispatch.json',
             {**process_identity(),'created_unix':process_identity()['created_unix']-100})
        scope_sha = save('performance/scope_100h.json',scope)
        save('performance/portable_inference_policy.json',numerical)
        save('performance/future_speed_policy.json',speed)
        sources = {}
        for stage in ('F3','F4','F5'):
            sources[stage+'_complete.json'] = save(stage+'_complete.json',{
                'source_identity':admit.ENGINE,'scope_identity':admit.SCOPE,'scope_amendment_sha256':scope_sha,
                'original_complete_scope_claimed':False,'numerical_policy_identity':admit.NUMERICAL,
                'evaluation_precision':'float32'})
        frozen = seal({'source_identity':admit.ENGINE,'plan_sha256':digest(PLAN),
                       'F5_sha256':sources['F5_complete.json'],'development_gate_passed':False,
                       'final_fit_policy':{'artifact':'deployment/pre_Q2'}})
        frozen_sha = save('system_freeze.json',frozen)
        sources['system_freeze.json'] = frozen_sha
        sources['confirmation.json'] = save('confirmation.json',{
            'system_freeze_sha256':frozen_sha,'evaluated':False,'proceed_to_final_test':False,'Q3_opened':False})
        weights = root/'deployment/pre_Q2/synthetic_weights.bin'
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b'small test fixture, never an actual fitted model')
        deployment = seal({'freeze_sha256':frozen_sha,'files':{
            weights.name:{'sha256':digest(weights),'bytes':weights.stat().st_size}}})
        save('deployment/pre_Q2/deployment.json',deployment)
        model = deployment['identity']
        accept = {'model_identity':model,'engineering_acceptance_passed':True,'origins':256,
                  'timing_complete':True,'timed_origins':256,'resume_without_duplicate_prediction':True,
                  'distinct_exit_and_resume_processes':True,'cold_load_seconds':12.,
                  'warm_single_origin_p95_seconds':.4,'batch_stream_max_prediction_difference':0.,
                  'batch_stream_tolerance':1e-8,'Q3_accessed':False}
        progress = {'recipe':{'source_identity':admit.ENGINE,'system_freeze_sha256':frozen_sha,
                             'model_identity':model},'processes':[{'pid':111},{'pid':222}]}
        save('deployment_acceptance/'+model+'/report.json',accept)
        save('deployment_acceptance/'+model+'/progress.json',progress)
        save('research_report.json',{'source_identity':admit.ENGINE,'plan_sha256':digest(PLAN),
             'violations':[],'stages':dict.fromkeys(('F0','F1','F2','F3','F4','F5','F6','F8_engineering_passed'),True)|{'F7_evaluated':False},
             'research_status':'stopped_before_final_by_prespecified_rule','source_artifacts_sha256':sources,
             'Q3_preserved_on_negative_path':True,'deployment_identity':model,'engineering_acceptance':accept})
        package = root/'local_packages'/model
        package.parent.mkdir()
        package.with_suffix('.tar').write_bytes(b'synthetic test archive receipt; not executed')
        archive = package.with_suffix('.tar')
        save('local_packages/'+model+'.receipt.json',{'archive_sha256':digest(archive),
             'archive_bytes':archive.stat().st_size,'model_identity':model,'package_identity':'synthetic_package'})
        save('local_packages/'+model+'.verification.json',{'archive_sha256':digest(archive),
             'model_identity':model,'package_identity':'synthetic_package','Q3_accessed':False,
             'isolated_example_and_fresh_process_resume_passed':True,
             'processes':[{'arguments':args} for args in [['--verify-only'],['--example'],['--example']]]})
        save('cloud/local_completion/return_receipt.json',{'sha256':'synthetic_return'})
        save('cloud/restores/synthetic_return/receipt.json',{'archive_sha256':'synthetic_return',
             'all_file_hashes_verified':True,'source_identity':admit.ENGINE,'plan_sha256':digest(PLAN)})
        yield SimpleNamespace(root=root,save=save,model=model)


def test_complete_receipt_chain_is_required_even_for_a_negative_research_result(evidence):
    result = admit.verify_delivery(evidence.root)
    assert result['engineering_acceptance_passed'] is True
    assert result['original_final_evaluated'] is False
    assert result['new_training_launched'] is False
    assert result['Q2_Q3_prediction_arrays_loaded'] is False
    assert result['identity'] == identity({k:v for k,v in result.items() if k != 'identity'})
    assert 'local_packages/'+evidence.model+'.tar' in result['files']


@pytest.mark.parametrize('mutation',[
    'running','live_receiver','late_deadline','stage_scope','freeze_bytes','model_bytes',
    'missing_package_check','same_replay_process','short_replay','unopened_final_violation',
    'changed_package','return_not_verified','source_missing',
])
def test_partial_stale_or_inconsistent_original_delivery_is_never_admitted(evidence,mutation):
    root,save,model = evidence.root,evidence.save,evidence.model
    if mutation == 'running':
        row = read_json(root/'cloud/local_completion/state.json');row['stage'] = 'F8_local_replay'
        save('cloud/local_completion/state.json',row)
    elif mutation == 'live_receiver':
        save('cloud/local_completion/handoff_receiver_dispatch.json',process_identity())
    elif mutation == 'late_deadline':
        row = read_json(root/'cloud/local_completion/binding.json');row['limit']['total_wall_hours'] = 101
        save('cloud/local_completion/binding.json',row)
    elif mutation == 'stage_scope':
        row = read_json(root/'F5_complete.json');row['scope_identity'] = 'another_run'
        save('F5_complete.json',row)
    elif mutation == 'freeze_bytes':
        row = read_json(root/'system_freeze.json');row['F5_sha256'] = 'wrong'
        save('system_freeze.json',row)
    elif mutation == 'model_bytes':
        (root/'deployment/pre_Q2/synthetic_weights.bin').write_bytes(b'changed')
    elif mutation == 'missing_package_check':
        (root/('local_packages/'+model+'.verification.json')).unlink()
    elif mutation in ('same_replay_process','short_replay'):
        name = 'deployment_acceptance/'+model+'/progress.json'
        row = read_json(root/name);row['processes'] = [{'pid':111},{'pid':111}]
        if mutation == 'short_replay':
            name = 'deployment_acceptance/'+model+'/report.json'
            row = read_json(root/name);row['origins'] = 255
        save(name,row)
    elif mutation == 'unopened_final_violation':
        save('final_test/entry.json',{'unexpected':True})
    elif mutation == 'changed_package':
        (root/('local_packages/'+model+'.tar')).write_bytes(b'changed')
    elif mutation == 'return_not_verified':
        row = read_json(root/'cloud/restores/synthetic_return/receipt.json');row['all_file_hashes_verified'] = False
        save('cloud/restores/synthetic_return/receipt.json',row)
    elif mutation == 'source_missing':
        row = read_json(root/'research_report.json');del row['source_artifacts_sha256']['F4_complete.json']
        save('research_report.json',row)
    with pytest.raises((ValueError,RuntimeError,FileNotFoundError)):
        admit.verify_delivery(root)


def test_exact_reference_requires_all_eligible_rows_and_recomputed_raw_scores(evidence):
    root = evidence.root
    dates = np.array(['2024-12-27','2024-12-30','2024-12-31','2025-01-02',
                      '2025-06-26','2025-06-27','2025-06-30','2025-07-01',
                      '2025-12-29','2025-12-30','2025-12-31','2026-01-02'])
    panel = SimpleNamespace(dates=dates,origins={'AAPL':np.array([3,7,11])},
                            targets={'AAPL':np.linspace(.0001,.0012,len(dates))})
    reference = {'candidate_id':'synthetic_reference','family':'fusion','foundation':True,
                 'foundation_mode':'frozen','loss':'scaled','representation_cache':True,
                 'sampler_policy':'balanced_complete_coverage_v1','total_updates':6,
                 'updates_by_fit_end':{'2024-12-30':2,'2025-06-27':4,'2025-12-30':6}}
    evidence.save('selections/fusion.json',{'selected':reference})
    for quarter,index in zip(admit.QUARTERS,[3,7,11]):
        f = fold(dates,quarter);config = for_fit(reference,f['fit_end'])
        directory = root/'trials/fusion/synthetic_reference'/quarter/'seed_2026'/('updates_'+str(config['total_updates']))
        directory.mkdir(parents=True)
        prediction = np.array([.00001]);rows = np.array([index])
        path = directory/'prediction.npz';np.savez_compressed(path,rows=rows,prediction=prediction)
        write_json(directory/'result.json',{'details':{'config':config,'seed':2026,'fold':f,
                   'source':admit.ENGINE,'updates':config['total_updates'],'evaluation_precision':'float32',
                   'numerical_policy_identity':admit.NUMERICAL},'prediction_sha256':digest(path),
                   'scores':point_scores(panel.targets['AAPL'][rows],prediction)})
    selected,configs,quarters,files = admit.fixed_reference(root,panel)
    assert selected == reference and len(configs) == 3 and set(quarters) == set(admit.QUARTERS)
    assert len(files) == 7
    path = root/quarters['2026Q1']['reference_result']
    row = read_json(path);row['scores']['mse'] *= .9;write_json(path,row)
    with pytest.raises(ValueError,match='local development targets'):
        admit.fixed_reference(root,panel)


def test_portable_frozen_manifest_binds_inputs_recipe_code_and_the_original_protocol(evidence):
    root = evidence.root
    (root/'data').mkdir()
    dates = np.array(['2024-12-27','2024-12-30','2024-12-31','2025-01-02',
                      '2025-06-26','2025-06-27','2025-06-30','2025-07-01',
                      '2025-12-29','2025-12-30','2025-12-31','2026-01-02'])
    np.save(root/'data/dates.npy',dates)
    reference = {'candidate_id':'synthetic_reference','family':'fusion','foundation':True,
                 'foundation_mode':'frozen','loss':'scaled','representation_cache':True,
                 'sampler_policy':'balanced_complete_coverage_v1','total_updates':6,
                 'updates_by_fit_end':{'2024-12-30':2,'2025-06-27':4,'2025-12-30':6}}
    evidence.save('selections/fusion.json',{'selected':reference})
    directory = admit.FOLLOW/'preparation'
    directory.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=directory,prefix='frozen_fixture_') as temporary:
        path = Path(temporary)/'manifest.json'
        files = {name:{'sha256':digest(root/name),'bytes':(root/name).stat().st_size}
                 for name in ('selections/fusion.json','data/dates.npy')}
        value = {'schema':1,'status':'matrix_frozen_benchmark_and_launch_admission_pending',
                 'source_identity':admit.ENGINE,'scope_identity':admit.SCOPE,'plan_sha256':digest(PLAN),
                 'followup_plan_sha256':digest(ROOT/'FUSION_FOLLOWUP_PLAN.md'),'delivery':admit.verify_delivery(root),
                 'reference':reference,'candidates':admit.candidates(reference),'original_inputs':files,
                 'implementation':{name:digest(ROOT/'scripts'/name) for name in admit.IMPLEMENTATIONS},
                 'earliest_large_run':admit.EARLIEST,'limit':{'total_wall_hours':100,'deadline_utc':admit.DEADLINE},
                 'validation':list(admit.QUARTERS),'pilot_seeds':[2026],'expanded_seeds':[2026,2027,2028],
                 'total_coverage_equivalent_epochs':2,'pilot_full_fits':9,'pilot_GPU_hours_ceiling':6,
                 'extension_GPU_hours_ceiling':6,'review_policy':admit.REVIEW_POLICY,
                 'training_admitted':False,'new_training_launched':False,
                 'folds':{q:{'fit_end':fold(dates,q)['fit_end'],
                             'score_origins':1,
                             'total_updates':for_fit(reference,fold(dates,q)['fit_end'])['total_updates']} for q in admit.QUARTERS}}
        write_json(path,seal(copy.deepcopy(value)))
        checked = admit.check_frozen(path,root)
        assert checked['training_admitted'] is False
        mutations = [lambda v:v.update(total_coverage_equivalent_epochs=1),
                     lambda v:v['limit'].update(total_wall_hours=101),
                     lambda v:v.update(earliest_large_run='2026-11-07T09:00:00+08:00'),
                     lambda v:v['implementation'].update({'followup_guard.py':'changed'}),
                     lambda v:v['folds']['2026Q1'].update(total_updates=1),
                     lambda v:v.update(folds={k:r for k,r in v['folds'].items() if k != '2026Q1'}),
                     lambda v:v['delivery'].update(engineering_acceptance_passed=False),
                     lambda v:v.update(review_policy={**v['review_policy'],'remove_top_positive_gain_days':0}),
                     lambda v:v['candidates'].pop()]
        for mutate in mutations:
            altered = copy.deepcopy(value);mutate(altered);write_json(path,seal(altered))
            with pytest.raises(ValueError):
                admit.check_frozen(path,root)
        write_json(path,seal(copy.deepcopy(value)))
        evidence.save('selections/fusion.json',{'selected':{**reference,'loss':'raw'}})
        with pytest.raises(ValueError,match='fixed input'):
            admit.check_frozen(path,root)
