"""Real CPU replay/package on fabricated prices; no actual research admission."""
import copy
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest

from fusion.next.common import ROOT,OUT,digest,identity,read_json,write_json
from fusion.next.tests_data import two_sessions
from fusion.next.tests_training import make_panel
from scripts import deliver_fusion_followup as deliver
from scripts import return_fusion_followup as returned
from scripts import followup_guard as guard
from scripts import followup_runtime as runtime
from scripts.test_followup_return import evidence,prepared,fitted


def seal(value):
    value=copy.deepcopy(value);value.pop('identity',None);value['identity']=identity(value)
    return value


@pytest.fixture
def local_model(fitted,monkeypatch):
    e=fitted
    monkeypatch.setattr(deliver,'BASE',e.original);monkeypatch.setattr(deliver,'FOLLOW',e.follow)
    frame,starts,ends=two_sessions()
    frame.index=frame.index+(pd.Timestamp('2026-01-05')-pd.Timestamp('2021-01-04'))
    panel=make_panel(frame,starts,ends)
    write_json(e.original/'data_admission.json',{'passed':True,'admitted_training_stocks':panel.stocks,
        'background':panel.background,'market_peers':[]})
    np.save(e.original/'data/dates.npy',panel.dates.astype('U10'))
    np.save(e.original/'data/timestamps_ns.npy',panel.stamps.as_unit('ns').asi8)
    for name in ('AAPL','SPY'):
        root=e.original/'data'/name;root.mkdir(parents=True)
        for key,values in (('features',panel.features[name]),('observed',panel.observed[name]),
                           ('origins',panel.origins[name]),('target',panel.targets[name])):np.save(root/(key+'.npy'),values)
    csv=e.original/'fabricated_prices.csv';frame.to_csv(csv,index_label='timestamp')
    write_json(e.original/'data/AAPL/manifest.json',{'source':{'source_csv_read_only':str(csv),'sha256':digest(csv)}})
    raw=e.original/'raw/SPY';raw.mkdir(parents=True)
    bars=[{'t':stamp.isoformat(),'o':row.open,'h':row.high,'l':row.low,'c':row.close,'v':row.volume,
           'n':row.trade_count,'vw':row.vwap} for stamp,row in frame.iterrows()]
    page=raw/'page.json.gz'
    with gzip.open(page,'wt') as stream:json.dump({'bars':{'SPY':bars}},stream)
    write_json(raw/'receipt.json',{'download_complete':True,'rows':len(bars),'pages':[
        {'file':page.name,'sha256':digest(page),'rows':len(bars)}]})
    write_json(e.original/'data/SPY/manifest.json',{'source':{'receipt_sha256':digest(raw/'receipt.json')}})
    for name in ('portable_inference_policy.json','future_speed_policy.json'):
        target=e.original/'performance'/name;target.parent.mkdir(exist_ok=True);shutil.copyfile(OUT/'performance'/name,target)
    from scripts.portable_package_runtime import specification
    write_json(e.original/'performance/portable_package_policy.json',specification())
    # Existing synthetic admission is explicitly supplied only to the low-level
    # test helper; public launch/worker cannot load this matrix as admitted work.
    archive,receipt=returned.build(e.path,e.manifest)
    returned.restore(archive,archive.parent/'receipt.json',e.path,e.manifest)
    monkeypatch.setattr(deliver,'verify_delivery',lambda root:{'files':e.manifest['delivery']['files']})
    monkeypatch.setattr(deliver,'check_frozen',lambda path:e.manifest)
    binding=seal(deliver.checked_return(e.path,e.manifest))
    e.local,e.frozen,e.deployment=deliver.prepare_local(e.manifest,binding)
    e.binding=binding;e.panel=panel;e.raw=csv
    from fusion.next.panel import Panel
    monkeypatch.setattr(Panel,'load',classmethod(lambda cls:panel))
    return e


PROGRAM = '''
import sys
from pathlib import Path
from fusion.next import panel,data
from fusion.next.common import read_json
from scripts import deliver_fusion_followup as d
base,follow,local,freeze,deployment,phase,resume,stop=sys.argv[1:]
d.BASE=Path(base);d.FOLLOW=Path(follow);panel.OUT=Path(base);data.OUT=Path(base)
local=Path(local)
result=d.run_phase(phase,local,read_json(freeze),Path(deployment),read_json(local/'binding.json'),
    resume=resume=='yes',stop_after=None if stop=='none' else int(stop))
print({'synthetic_CPU_test_completed':True,'phase':phase})
'''


def phase(e,name,resume=False,stop=None):
    command=[sys.executable,'-c',PROGRAM,str(e.original),str(e.follow),str(e.local),
        str(e.run/'delivery/engineering_freeze.json'),str(e.deployment),name,'yes' if resume else 'no',
        'none' if stop is None else str(stop)]
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=90,
        env={**os.environ,'PYTHONPATH':str(ROOT),'FUSION_EXECUTION_PROFILE':'local','CUDA_VISIBLE_DEVICES':''})
    assert result.returncode==0,result.stderr[-5000:]
    return result


def test_actual_256_origin_two_process_replay_input_contracts_and_standalone_package(local_model):
    e=local_model;before={str(p.relative_to(e.original)):digest(p) for p in e.original.rglob('*') if p.is_file()}
    phase(e,'replay',stop=128)
    root=e.local/'deployment_acceptance'/e.binding['outcome']['deployment_identity']
    first=read_json(root/'progress.json');state=read_json(root/'state.json')
    assert first['completed_origins']==128 and len(state['outputs'])==128 and not (root/'report.json').exists()
    phase(e,'replay',resume=True)
    report=deliver.checked_replay(e.local,e.binding)
    assert report['origins']==report['timed_origins']==256 and report['timing_complete']
    assert report['resume_without_duplicate_prediction'] and report['distinct_exit_and_resume_processes']
    assert report['evaluation_precision']=='float32' and report['batch_stream_max_prediction_difference']<=report['batch_stream_tolerance']
    progress=read_json(root/'progress.json')
    assert len({(p['pid'],p['created_unix']) for p in progress['processes']})==2
    assert all(p['full_process_to_first_output_seconds']>=p['model_cold_load_seconds'] for p in progress['processes'])
    assert len(read_json(root/'state.json')['outputs'])==256
    phase(e,'package')
    checked=deliver.checked_replay(e.local,e.binding);contracts=read_json(e.local/'input_contracts.json')
    assert not checked['actual_input_contract_checks_pending'] and contracts['all_passed']
    assert all(contracts['checks'].values()) and contracts['controlled_failures_are_not_research_scores']
    archive=e.local/'local_packages'/(checked['model_identity']+'.tar')
    verified=read_json(archive.with_suffix('.verification.json'))
    assert verified['isolated_example_and_fresh_process_resume_passed']
    assert verified['followup_binding_identity']==e.binding['identity'] and len(verified['actual_processes'])==3
    package=read_json(archive.with_suffix('')/'package.json')
    assert package['engineering_acceptance_passed']==checked['engineering_acceptance_passed']
    assert '开发版本' in (archive.with_suffix('')/'README.md').read_text()
    assert not read_json(archive.with_suffix('')/'evidence/confirmation.json')['independent_research_success_claimed']
    assert digest(archive.with_suffix('')/'fusion/next/cloud_assets/cpu_run.py')==digest(ROOT/'fusion/next/cloud_assets/cpu_run.py')
    assert digest(archive.with_suffix('')/'run.py')==digest(ROOT/'scripts/portable_cpu_run.py')
    assert 'evidence/portable_package_policy.json' in package['files']
    assert {str(p.relative_to(e.original)):digest(p) for p in e.original.rglob('*') if p.is_file()}==before
    # Identical weights may be reverified without rebuilding the immutable package.
    original_hash=digest(archive);phase(e,'package')
    assert digest(archive)==original_hash
    original=read_json(root/'report.json');write_json(root/'report.json',{**original,'warm_single_origin_p95_seconds':.000001})
    with pytest.raises(ValueError,match='actual measurements'):deliver.checked_replay(e.local,e.binding)


def test_failed_replay_origin_cannot_be_dropped_on_resume(local_model):
    e=local_model;phase(e,'replay',stop=128)
    root=e.local/'deployment_acceptance'/e.binding['outcome']['deployment_identity']
    state=read_json(root/'state.json');first=min(state['outputs'])
    state['outputs'][first].update(status='input_or_model_failure',reason='controlled retained failed origin')
    state['integrity']=identity({k:v for k,v in state.items() if k!='integrity'});write_json(root/'state.json',state)
    command=[sys.executable,'-c',PROGRAM,str(e.original),str(e.follow),str(e.local),
        str(e.run/'delivery/engineering_freeze.json'),str(e.deployment),'replay','yes','none']
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=25)
    assert result.returncode!=0 and 'failed saved origin cannot be dropped' in result.stderr
    assert len(read_json(root/'state.json')['outputs'])==128 and not (root/'report.json').exists()


@pytest.mark.parametrize('symbol',['AAPL','SPY'])
def test_actual_replay_refuses_changed_raw_price_source_without_scoring(local_model,symbol):
    e=local_model
    path=e.raw if symbol=='AAPL' else e.original/'raw/SPY/receipt.json'
    path.write_text(path.read_text()+'\n')
    command=[sys.executable,'-c',PROGRAM,str(e.original),str(e.follow),str(e.local),
        str(e.run/'delivery/engineering_freeze.json'),str(e.deployment),'replay','no','128']
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=20)
    assert result.returncode!=0 and ('raw source differs' if symbol=='AAPL' else 'background raw receipt differs') in result.stderr
    assert not list(e.local.rglob('predictions.npz'))


def test_return_binding_tampering_and_different_local_freeze_copies_are_refused(local_model):
    e=local_model;before=digest(e.local/'binding.json')
    changed=seal({**e.binding,'return_archive_sha256':'altered'})
    with pytest.raises(ValueError,match='existing local delivery binding'):deliver.prepare_local(e.manifest,changed)
    assert digest(e.local/'binding.json')==before
    (e.local/'system_freeze.json').write_text('{}')
    with pytest.raises(ValueError,match='copy differs'):deliver.prepare_local(e.manifest,e.binding)
    receipt=e.follow/'returns'/e.manifest['identity']/'restored_receipt.json'
    row=read_json(receipt);row['outcome']['deployment_identity']='changed';write_json(receipt,seal(row))
    with pytest.raises(ValueError,match='exact verified formal return'):deliver.checked_return(e.path,e.manifest)


def test_latency_and_origin_constraints_cannot_be_replaced_by_good_predictions():
    good={'cold_load_seconds':59.,'warm_single_origin_p95_seconds':.9,
        'batch_stream_max_prediction_difference':0.,'batch_stream_tolerance':1e-8,
        'origins':256,'timed_origins':256,'timing_complete':True,
        'resume_without_duplicate_prediction':True,'distinct_exit_and_resume_processes':True}
    assert deliver.qualified(good)
    for change in ({'cold_load_seconds':61.},{'warm_single_origin_p95_seconds':1.01},{'origins':128},
        {'timed_origins':255},{'resume_without_duplicate_prediction':False},
        {'batch_stream_max_prediction_difference':float('nan')}, {'batch_stream_max_prediction_difference':1e-7}):
        assert not deliver.qualified({**good,**change})
    assert not deliver.qualified(good,contracts=False)


@pytest.mark.parametrize('purpose',['delivery_fit','delivery_cpu'])
def test_real_local_delivery_worker_requires_CPU_purpose_before_loading_matrix(local_model,purpose):
    e=local_model;attempt=e.root/'local_identity_attempt'
    missing=runtime.FOLLOW/'preparation'/('missing_delivery_'+e.root.name+'.json')
    command=[sys.executable,str(deliver.ENTRY),'--manifest',str(missing),'--worker','--attempt',str(attempt)]
    process,_=guard.start(attempt,command,e.manifest['limit'],10,checkpoint_grace=2,termination_grace=.2,purpose=purpose)
    try:
        assert process.wait(timeout=12)==1
        log=(attempt/'worker.log').read_text()
        if purpose=='delivery_fit':assert 'Delivery CPU worker requires its exact delivery CPU guard purpose' in log
        else:assert 'FileNotFoundError' in log and missing.name in log
        assert 'Worker is not bound' not in log
    finally:deliver.stop_guard(process,attempt)


def test_unguarded_child_request_cannot_start_replay_or_package(local_model):
    e=local_model;path=e.local/'process_requests/fake.json'
    write_json(path,seal({'guard_attempt':str(e.root/'absent_attempt')}))
    result=subprocess.run([sys.executable,str(deliver.ENTRY),'--child-request',str(path)],
        cwd=ROOT,capture_output=True,text=True,timeout=15)
    assert result.returncode!=0 and not list(e.local.rglob('predictions.npz'))
