"""Exercise delayed interpreter startup and actual separate-process recovery."""
import copy
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from tempfile import TemporaryDirectory
import textwrap

import psutil
import pytest

from fusion.next.common import ROOT, OUT, PLAN, digest, read_json, source_identity, write_json
from scripts import original_delivery_timing_runtime as timing
from scripts import arm_original_delivery_timing as arm


POLICY = {'identity':'synthetic_cold_policy', 'measurement_scope':timing.SCOPE, 'cold_seconds_ceiling':60}


@pytest.fixture(scope='module')
def actual_replay():
    with TemporaryDirectory(dir=OUT, prefix='full_process_cold_test_') as temporary:
        directory = Path(temporary)
        program = '''import time
time.sleep(.3)  # Deliberate startup work before importing the engine/adapter.
import os
from pathlib import Path
from fusion.next.tests_delivery import configure_toy
from fusion.next import common,delivery,inference,data
from aapl import data as raw_data
from scripts import original_delivery_timing_runtime as timing, portable_package_runtime as package
deployment = configure_toy(Path(os.environ['AAPL_TOY_REPLAY_DIRECTORY']))
common.OUT = delivery.OUT
inference.FrozenSystem = delivery.FrozenSystem
raw_data.read_source = delivery.read_source
data.pages_frame = delivery.pages_frame
common.write_json(deployment/'deployment.json',{'identity':'toy_process_recovery'})
package._installed = True
timing.checked_policy = lambda: {'identity':'synthetic_cold_policy','measurement_scope':timing.SCOPE,'cold_seconds_ceiling':60}
timing.install()
'''
        bootstrap = directory/'bootstrap';bootstrap.mkdir()
        (bootstrap/'sitecustomize.py').write_text('try:\n'+textwrap.indent(program,'    ')+
            '\nexcept Exception as error:\n    raise SystemExit("Synthetic CLI setup failed: "+str(error)) from error\n')
        result = subprocess.run([sys.executable,'-m','fusion.next.delivery',
            '--deployment',str(directory/'deployment'),'--origins','6'],cwd=ROOT,
            env=dict(os.environ,PYTHONPATH=str(bootstrap)+os.pathsep+str(ROOT),
                     AAPL_TOY_REPLAY_DIRECTORY=str(directory),CUDA_VISIBLE_DEVICES=''),
            capture_output=True,text=True,timeout=60)
        assert result.returncode == 0, result.stderr
        saved = directory/'deployment_acceptance/toy_process_recovery'
        logs = [read_json(path) for path in sorted((saved/'process_logs').glob('*.json'))]
        assert len(logs) == 2 and all(row['returncode'] == 0 for row in logs)
        assert '--stop-after' in logs[0]['arguments'] and '--resume' in logs[1]['arguments']
        assert all(row['arguments'][0] == 'fusion.next.delivery' for row in logs)
        yield saved, read_json(saved/'report.json'), read_json(saved/'progress.json')


def test_actual_startup_includes_pre_import_delay_and_durable_fresh_process_recovery(actual_replay):
    saved, report, progress = actual_replay
    full = timing.validate_measurements(report,progress,POLICY)
    assert full > report['engine_cold_load_seconds'] + .25
    assert report['resume_without_duplicate_prediction'] and report['engineering_acceptance_passed']
    assert report['origins'] == report['timed_origins'] == len(read_json(saved/'state.json')['outputs']) == 6
    assert len({row['kernel_process_birth']['pid'] for row in progress['processes']}) == 2
    assert report['batch_stream_max_prediction_difference'] == 0


@pytest.mark.parametrize('mutation',['missing','reused','policy','maximum','inconsistent','wall_clock_only'])
def test_missing_legacy_or_inconsistent_full_process_measurements_are_rejected(actual_replay,mutation):
    _, original, progress = actual_replay
    report, progress = copy.deepcopy(original), copy.deepcopy(progress)
    if mutation == 'missing':
        report['cold_load_measurements_by_process'][0].pop('full_process_to_first_output_seconds')
    elif mutation == 'reused':
        report['cold_load_measurements_by_process'][1] = copy.deepcopy(report['cold_load_measurements_by_process'][0])
    elif mutation == 'policy':
        report['cold_timing_policy_identity'] = 'tampered'
    elif mutation == 'maximum':
        report['cold_load_seconds'] = report['engine_cold_load_seconds']
    elif mutation == 'inconsistent':
        report['cold_load_measurements_by_process'][0]['first_durable_output_boottime_seconds'] += 1
    else:
        report['cold_load_measurements_by_process'][0].pop('kernel_process_birth')
        report['cold_load_measurements_by_process'][0]['created_unix'] = 1.
    progress['processes'] = copy.deepcopy(report['cold_load_measurements_by_process'])
    with pytest.raises(ValueError):
        timing.validate_measurements(report,progress,POLICY)


def test_elapsed_time_ignores_wall_clock_corrections(monkeypatch):
    birth = timing.process_birth()
    monkeypatch.setattr(timing.time,'time',lambda: (_ for _ in ()).throw(AssertionError('wall clock used')))
    elapsed, observed = timing.elapsed_from_birth(birth)
    assert elapsed >= 0 and observed > elapsed


@pytest.mark.parametrize('full,previous_pass,expected',[(12.,True,True),(61.,True,False),(12.,False,False)])
def test_complete_cold_start_gate_cannot_be_relaxed(monkeypatch,full,previous_pass,expected):
    from fusion.next import common,stream,delivery,local_package
    from scripts import portable_package_runtime as package
    for module,name in ((common,'write_json'),(stream.Stream,'predict'),(delivery,'write_json'),
                        (local_package,'source_files'),(runpy,'_run_module_as_main')):
        monkeypatch.setattr(module,name,getattr(module,name))
    monkeypatch.setattr(timing,'_installed',False)
    monkeypatch.setattr(package,'_installed',True)
    monkeypatch.setattr(timing,'checked_policy',lambda:POLICY)
    with TemporaryDirectory(dir=OUT,prefix='full_cold_gate_test_') as temporary:
        directory = Path(temporary)/'deployment_acceptance/synthetic'
        rows = [{'pid':pid,'kernel_process_birth':{'pid':pid,'boot_id':'synthetic_boot',
                    'start_ticks':100*pid,'ticks_per_second':100},
                 'cold_timing_policy_identity':POLICY['identity'],
                 'full_process_to_first_output_seconds':full,
                 'first_durable_output_boottime_seconds':pid+full} for pid in (111,222)]
        write_json(directory/'progress.json',{'processes':rows})
        report = {'cold_load_seconds':10.,'cold_load_measurements_by_process':rows,
                  'distinct_exit_and_resume_processes':True,'warm_single_origin_p95_seconds':.1,
                  'engineering_acceptance_passed':previous_pass}
        timing.install()
        common.write_json(directory/'report.json',report)
        assert read_json(directory/'report.json')['engineering_acceptance_passed'] is expected
        assert report['engine_cold_load_seconds'] == 10. and report['cold_load_seconds'] == full


def test_real_python_module_package_entry_retains_adapters_and_evidence():
    with TemporaryDirectory(dir=ROOT/'tmp',prefix='actual_package_entry_test_') as temporary:
        directory = Path(temporary)
        receipt = directory/'entry_receipt.json'
        bootstrap = '''from pathlib import Path
import json
from scripts import portable_inference_runtime as numerical, future_speed_runtime as speed, portable_package_runtime as package
numerical.install(); speed.install(); package.install()
from scripts import original_delivery_timing_runtime as timing
timing.checked_policy = lambda: {'identity':'synthetic_cold_policy','measurement_scope':timing.SCOPE,'cold_seconds_ceiling':60}
timing.install()
from fusion.next import local_package
from fusion.next.common import ROOT,digest
def build(deployment):
    directory = Path(RECEIPT).parent
    source = ROOT/'fusion/next/cloud_assets/cpu_run.py'
    entry = directory/'run.py'; engine = directory/'cpu_run.py'
    local_package.shutil.copyfile(source,entry); local_package.shutil.copyfile(source,engine)
    import torch
    result = {'deployment':str(deployment),'FP32_entry':digest(entry)==digest(ROOT/'scripts/portable_cpu_run.py'),
        'frozen_engine_preserved':digest(engine)==digest(source),'files':sorted(local_package.source_files()),
        'CUDA_initialized':torch.cuda.is_initialized(),'verified':False}
    Path(RECEIPT).write_text(json.dumps(result))
    return directory/'synthetic_package.tar'
def verify(archive):
    value=json.loads(Path(RECEIPT).read_text());value['verified']=True
    Path(RECEIPT).write_text(json.dumps(value));return value
local_package.build=build;local_package.verify=verify
'''.replace('RECEIPT',repr(str(receipt)))
        (directory/'sitecustomize.py').write_text(bootstrap)
        result = subprocess.run([sys.executable,'-m','fusion.next.local_package','--deployment','synthetic','--verify'],
            cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(directory)+os.pathsep+str(ROOT),CUDA_VISIBLE_DEVICES=''),
            capture_output=True,text=True,timeout=60)
        assert result.returncode == 0, result.stderr
        value = read_json(receipt)
        assert value['verified'] and value['FP32_entry'] and value['frozen_engine_preserved']
        assert value['deployment'] == 'synthetic' and not value['CUDA_initialized']
        assert {'portable_inference_policy.json','evidence/portable_package_policy.json',
                'evidence/'+timing.POLICY_NAME,'evidence/original_cpu_package_entry.py'} <= set(value['files'])


def test_kernel_receiver_identity_survives_legacy_epoch_mismatch_but_refuses_active_phase(monkeypatch):
    with TemporaryDirectory(dir=OUT,prefix='cold_arm_receiver_test_') as temporary:
        directory = Path(temporary)/'cloud/local_completion'
        monkeypatch.setattr(arm,'OUT',Path(temporary))
        process = psutil.Process()
        dispatch = {'pid':process.pid,'created_unix':process.create_time()-1491}
        write_json(directory/'dispatch.json',dispatch)
        birth = timing.process_birth()
        write_json(directory/'receiver_identity_clock_review.json',{
            'original_receiver':dispatch,'original_dispatch_sha256':digest(directory/'dispatch.json'),
            'current_observation':{**birth,'proc_dir_inode':Path(f'/proc/{process.pid}').stat().st_ino,
                                   'command':process.cmdline()}})
        write_json(directory/'binding.json',{'source_identity':source_identity(),'plan_sha256':digest(PLAN),
                   'script_sha256':digest(ROOT/'scripts/finish_fusion_100h.py')})
        write_json(directory/'state.json',{'stage':'waiting_for_cloud'})
        assert arm.verified_receiver() == birth
        write_json(directory/'state.json',{'stage':'F8_local_replay'})
        with pytest.raises(RuntimeError,match='idle waiting'):
            arm.verified_receiver()
