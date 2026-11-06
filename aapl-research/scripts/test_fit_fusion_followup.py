"""CPU engineering integration on fabricated inputs; no cloud/GPU admission."""
import copy
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from fusion.next.common import ROOT,PLAN,digest,identity,read_json,source_identity,write_json
from fusion.next.experiment import model_for
from fusion.next.trainer import Trainer
from fusion.next.inference import export,FrozenSystem
from fusion.next.experts import fit_cpu
from fusion.next.tests_data import two_sessions
from fusion.next.tests_training import make_panel
from scripts import fit_fusion_followup as fit
from scripts import followup_runtime as runtime
from scripts import followup_guard as guard
from scripts.test_run_fusion_followup import NOW,SIGNATURE,synthetic_probe


def seal(value):
    value=copy.deepcopy(value);value.pop('identity',None);value['identity']=identity(value)
    return value


@pytest.fixture
def evidence(monkeypatch):
    frame,starts,ends=two_sessions();panel=make_panel(frame,starts,ends)
    (runtime.FOLLOW/'tests').mkdir(exist_ok=True)
    (runtime.FOLLOW/'runs').mkdir(exist_ok=True)
    with TemporaryDirectory(dir=runtime.FOLLOW/'tests',prefix='fit_inputs_') as inputs, TemporaryDirectory(dir=runtime.FOLLOW/'runs',prefix='fit_fixture_') as temporary:
        inputs=Path(inputs);(inputs/'data').mkdir();root=Path(temporary);original=root/'original'
        references={};selections={}
        with runtime.isolated_outputs(original,inputs):
            for family in ('ridge','lightgbm'):
                cfg={'family':family,'lookback':48,'alpha':1000.,'rows_per_day':8,'trees':4,'num_leaves':7,
                     'max_depth':3,'min_child_samples':4,'learning_rate':.03,'reg_lambda':10.,
                     'colsample_bytree':1.,'cpu_threads':2,'information':'ohlcv','training_scope':'multistock','market_information':True}
                directory=original/'fitted'/family;fit_cpu(panel,cfg,'2021-01-04',directory)
                references[family]={'config':cfg,'directory':str(directory)};selections[family]=cfg
            for family in ('xpatch','patchtst'):
                cfg={'family':family,'lookback':48,'lr':.001,'batch_size':4,'total_updates':2,'loss':'raw',
                     'information':'ohlcv','training_scope':'multistock','market_information':True}
                trainer=Trainer(panel,model_for(panel,cfg,'cpu'),cfg,'2021-01-04','original/'+family,device='cpu')
                trainer.run_until(2)
                references[family]={'config':cfg,'checkpoint':str(trainer.path)};selections[family]=cfg
            # Only the expert subset is needed for this unit fixture. Production
            # matrices require a separately verified real original three-seed F8.
            export(original/'deployment/pre_Q2',panel.stocks,panel.background,references,[],'synthetic_original_freeze')
        for name,cfg in selections.items():write_json(original/('selections/single_'+name+'.json'),{'selected':cfg})
        config={'family':'fusion','candidate_id':'toy_fixed_followup','lookback':48,'hidden':32,'projection':64,
                'market_encoder':'tcn','residual_cap':.25,'foundation':False,'market':True,'lr':.001,
                'batch_size':4,'total_updates':2,'loss':'raw','precision':'fp32','information':'ohlcv',
                'training_scope':'multistock','market_information':True,
                'updates_by_fit_end':{'2021-01-04':2}}
        class ToyOOF:
            available={0:np.zeros(len(panel.dates),dtype=bool)}
            sources={'fixture':'auxiliary objective only; no in-sample expert predictions'}
        candidates=[{'name':name,'config':{**config,'candidate_id':name}} for name in
                    ('A_foundation_raw','B_no_foundation_scaled','C_no_foundation_raw')]
        files={str(p.relative_to(original)):{'sha256':digest(p),'bytes':p.stat().st_size}
               for p in original.rglob('*') if p.is_file()}
        policy={'fit_end':'2021-01-04','total_updates':2,'system_seeds':[2026,2027,2028],
                'total_coverage_equivalent_epochs':2,'GPU_hours_ceiling':6,'local_delivery_reserve_hours':2}
        manifest={'identity':'synthetic_fit_matrix','source_identity':source_identity(),'plan_sha256':digest(PLAN),
                  'reference':config,'original_inputs':files,'deployment_fit_policy':policy,'candidates':candidates,
                  'expert_sources':ToyOOF.sources,'expert_scope_key':'synthetic_oof',
                  'limit':{'total_wall_hours':100,'deadline_utc':fit.DEADLINE},
                  'folds':{q:{'fit_end':'2021-01-04','total_updates':2,'score_origins':3} for q in ('2025Q1','2025Q3','2026Q1')}}
        reviewed={'selected_candidate':'C_no_foundation_raw','extension_plan':{'selected_candidate':'C_no_foundation_raw'},
                  'seeds':[2026,2027,2028],'total_coverage_equivalent_epochs':2,
                  'Q2_Q3_prediction_arrays_loaded':False,'independent_research_success_claimed':False,
                  'development_checks_authorize_final_data_access':False,'development_qualification_passed':False}
        saved={'report':'synthetic_test_only','identity':'synthetic_test_only','decision_identity':'synthetic_test_only'}
        monkeypatch.setattr(fit,'BASE',original);monkeypatch.setattr(fit,'FOLLOW',root/'followup')
        monkeypatch.setattr(fit,'checked_review',lambda path,manifest:(saved,reviewed))
        yield SimpleNamespace(root=root,original=original,panel=panel,manifest=manifest,review=reviewed,
                store=ToyOOF(),path=root/'synthetic_manifest.json',follow=root/'followup')


def freeze(e):
    path,value=fit.freeze_delivery(e.path,e.manifest)
    run=e.follow/'runs'/e.manifest['identity']
    return path,value,run


def test_reuse_requires_exact_bound_expert_bytes_configuration_and_fit_end(evidence):
    result=fit.original_experts(evidence.manifest)
    assert len(result['experts'])==4 and len(result['files'])==10
    changed=copy.deepcopy(evidence.manifest);changed['deployment_fit_policy']['fit_end']='2021-01-05'
    with pytest.raises(ValueError,match='fit boundary|another recipe'):
        fit.original_experts(changed)
    changed=copy.deepcopy(evidence.manifest);changed['original_inputs'].pop(next(iter(result['files'])))
    with pytest.raises(ValueError,match='not bound'):fit.original_experts(changed)
    artifact=evidence.original/next(iter(result['files']))
    artifact.write_bytes(b'changed original expert bytes')
    with pytest.raises(ValueError):fit.original_experts(evidence.manifest)


def test_delivery_freeze_is_idempotent_and_keeps_a_negative_three_seed_result(evidence):
    path,value,_=freeze(evidence)
    assert not value['development_qualification_passed'] and value['engineering_fit_only']
    assert not value['Q2_Q3_scoring_authorized'] and not value['actual_CPU_acceptance_passed']
    assert fit.freeze_delivery(evidence.path,evidence.manifest)==(path,value)
    evidence.review['selected_candidate']='A_foundation_raw'
    with pytest.raises(ValueError,match='fixed complete'):fit.freeze_delivery(evidence.path,evidence.manifest)
    evidence.review['selected_candidate']='C_no_foundation_raw'
    evidence.review['development_qualification_passed']=True
    with pytest.raises(ValueError,match='Preserve'):fit.freeze_delivery(evidence.path,evidence.manifest)


def probes(e,run):
    for i in range(3):
        attempt=synthetic_probe(run,e.manifest,i)
        path=attempt/'probe_result.json';value=read_json(path)
        value.update(lookback=48,batch_size=4,checkpoint_copy_bytes=2**20,checkpoint_copy_seconds_upper=.02)
        write_json(path,seal(value))


def test_delivery_budget_uses_real_probe_fields_and_keeps_date_stage_and_CPU_limits(evidence):
    _,frozen,run=freeze(evidence);probes(evidence,run)
    budget=fit.delivery_budget(evidence.manifest,run,frozen,SIGNATURE,NOW)
    assert budget['projection']['pending_updates']==6 and budget['local_delivery_reserve_seconds']==2*3600
    with pytest.raises(RuntimeError,match='November 8'):
        fit.delivery_budget(evidence.manifest,run,frozen,SIGNATURE,NOW.replace(day=7))
    with pytest.raises(RuntimeError,match='two-hour local delivery reserve'):
        fit.delivery_budget(evidence.manifest,run,frozen,SIGNATURE,datetime.fromisoformat(fit.DEADLINE).replace(hour=8))
    path=run/'attempts/probe_03/probe_result.json';probe=read_json(path);probe.pop('checkpoint_copy_bytes')
    write_json(path,seal(probe))
    with pytest.raises(ValueError,match='actual checkpoint copy'):
        fit.delivery_budget(evidence.manifest,run,frozen,SIGNATURE,NOW)
    probes(evidence,run)
    write_json(run/'attempts/fit_attempt_01/guard_request.json',seal({'requested_at_unix':NOW.timestamp()-10}))
    with pytest.raises(RuntimeError,match='cumulative six-hour cap'):
        fit.delivery_budget(evidence.manifest,run,frozen,SIGNATURE,NOW)


def test_repeat_probes_cannot_replace_slow_checkpoint_copy_measurement(evidence):
    _,frozen,run=freeze(evidence);probes(evidence,run)
    baseline=fit.delivery_budget(evidence.manifest,run,frozen,SIGNATURE,NOW)['projection']
    import shutil
    repeat=run/'attempts/probe_04';shutil.copytree(run/'attempts/probe_03',repeat)
    report=read_json(repeat/'probe_result.json');report['checkpoint_copy_seconds_upper']=.001
    write_json(repeat/'probe_result.json',seal(report))
    assert fit.delivery_budget(evidence.manifest,run,frozen,SIGNATURE,NOW)['projection']==baseline
    report['checkpoint_copy_seconds_upper']=.04;write_json(repeat/'probe_result.json',seal(report))
    slower=fit.delivery_budget(evidence.manifest,run,frozen,SIGNATURE,NOW)['projection']
    assert slower['export_copy_seconds']==2*baseline['export_copy_seconds']


def test_worker_budget_recheck_excludes_only_its_bound_active_reservation(evidence):
    _,frozen,run=freeze(evidence);probes(evidence,run)
    attempt=run/'attempts/fit_attempt_01';request=seal({'requested_at_unix':NOW.timestamp()-4,
        'purpose':'delivery_fit','limit':evidence.manifest['limit'],
        'delivery_reserve_seconds':guard.CPU_DELIVERY_RESERVE_SECONDS,'stage_seconds':guard.MAX_STAGE_SECONDS-10})
    write_json(attempt/'guard_request.json',request)
    write_json(attempt/'guard_dispatch.json',{'guard':guard.process_identity()})
    with pytest.raises(RuntimeError,match='guard is still alive'):
        fit.delivery_budget(evidence.manifest,run,frozen,SIGNATURE,NOW)
    budget=fit.delivery_budget(evidence.manifest,run,frozen,SIGNATURE,NOW,current_attempt=attempt)
    assert budget['stage_seconds']==guard.MAX_STAGE_SECONDS-14
    request['purpose']='exploration';write_json(attempt/'guard_request.json',seal(request))
    with pytest.raises(ValueError,match='changes purpose'):
        fit.delivery_budget(evidence.manifest,run,frozen,SIGNATURE,NOW,current_attempt=attempt)
    with pytest.raises(ValueError,match='outside its fixed'):
        fit.delivery_budget(evidence.manifest,run,frozen,SIGNATURE,NOW,current_attempt=evidence.root)


@pytest.mark.parametrize('loss',['raw','scaled'])
def test_actual_CPU_partial_fit_resumes_exports_three_fresh_seeds_and_preserves_experts(evidence,monkeypatch,loss):
    selected=next(r for r in evidence.manifest['candidates'] if r['name']==evidence.review['selected_candidate'])
    selected['config']['loss']=loss
    path,frozen,run=freeze(evidence);destination=run/'delivery_fit';probes(evidence,run)
    original_hashes={name:digest(evidence.original/name) for name in frozen['original_experts']['files']}
    original_writer=fit.write_json
    def interrupted(path,value):
        original_writer(path,value)
        if Path(path).name=='fit_progress.json' and value['complete_seeds']==[2026]:
            raise RuntimeError('controlled exit after first seed')
    monkeypatch.setattr(fit,'write_json',interrupted)
    with runtime.isolated_outputs(destination,evidence.original):
        with pytest.raises(RuntimeError,match='controlled exit'):
            fit.fit_and_export(evidence.panel,evidence.store,frozen,path,destination,device='cpu')
        cfg=fit.system_config(frozen,2027)
        partial=Trainer(evidence.panel,model_for(evidence.panel,cfg,'cpu'),cfg,frozen['fit_policy']['fit_end'],
                        'delivery_final/'+cfg['candidate_id'],2027,evidence.store,'cpu')
        partial.run_until(1)
    progress=fit.checkpoint_progress(destination,frozen)
    assert [r['step'] for r in progress.values()]==[2,1,0]
    first_hash=progress['2026']['checkpoint_sha256']
    assert fit.delivery_budget(evidence.manifest,run,frozen,SIGNATURE,NOW)['projection']['pending_updates']==3
    incomplete=destination/'deployment/pre_Q2';incomplete.mkdir(parents=True)
    (incomplete/'retained.bin').write_bytes(b'interrupted export remains available')
    monkeypatch.setattr(fit,'write_json',original_writer)
    with runtime.isolated_outputs(destination,evidence.original):
        result=fit.fit_and_export(evidence.panel,evidence.store,frozen,path,destination,device='cpu')
    assert fit.checkpoint_progress(destination,frozen)['2026']['checkpoint_sha256']==first_hash
    assert result['fresh_system_seeds']==[2026,2027,2028] and not result['Q2_Q3_scored']
    assert not result['actual_CPU_acceptance_passed'] and not result['independent_research_success_claimed']
    assert len(list((destination/'deployment').glob('incomplete_export_*/retained.bin')))==1
    assert {name:digest(evidence.original/name) for name in original_hashes}==original_hashes
    loaded=FrozenSystem(ROOT/result['deployment_directory'],'cpu')
    actual=loaded.predict_origin(evidence.panel,np.array([430,500,700]))
    np.testing.assert_array_equal(actual['prediction'],actual['seed_predictions'].astype(np.float64).mean(0))
    with pytest.raises(RuntimeError,match='already completed'):
        fit.delivery_budget(evidence.manifest,run,frozen,SIGNATURE,NOW)
    # Neither a changed fit boundary nor a partial prefix can masquerade as a
    # completed engineering seed, even if its own JSON is internally resealed.
    receipt=fit.checkpoint_path(destination,frozen,2028).with_suffix('.pt.json')
    value=read_json(receipt);value['recipe']['fit_end']='2026-06-29';write_json(receipt,value)
    with pytest.raises(ValueError,match='Engineering checkpoint changes'):
        fit.checkpoint_progress(destination,frozen)


@pytest.mark.parametrize('purpose',['exploration','delivery_fit'])
def test_real_delivery_worker_binds_only_the_delivery_guard_before_loading_a_matrix(evidence,purpose):
    attempt=evidence.root/'identity_attempt'
    missing=runtime.FOLLOW/'preparation'/('missing_'+evidence.root.name+'.json')
    command=[sys.executable,str(fit.ENTRY),'--worker','--manifest',str(missing),'--attempt',str(attempt)]
    process,_=guard.start(attempt,command,evidence.manifest['limit'],10,checkpoint_grace=2,termination_grace=.2,purpose=purpose)
    try:
        assert process.wait(timeout=12)==1
        log=(attempt/'worker.log').read_text()
        if purpose=='exploration':
            assert 'Delivery worker requires its exact delivery-fit guard purpose' in log
            assert 'FileNotFoundError' not in log
        else:
            assert 'FileNotFoundError' in log and missing.name in log
            assert 'Worker is not bound' not in log
    finally:fit.stop_guard(process,attempt)
