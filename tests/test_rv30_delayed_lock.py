"""Appending delayed execution must preserve all original signed predecessors."""
import hashlib
import subprocess

import pytest

from finance import provenance
from finance.io import sha, stable_hash, write_json


@pytest.fixture
def replay_lock(monkeypatch,tmp_path):
    monkeypatch.setattr(provenance,'ROOT',tmp_path)
    monkeypatch.setattr(provenance,'FREEZE_PATH',tmp_path/'delivery/rv30_freeze.json')
    name='finance/provenance.py';before=b'original-provenance'
    frozen={'signature':'research','payload':{'source_files':{name:hashlib.sha256(before).hexdigest()}}}
    write_json(provenance.FREEZE_PATH,frozen)
    for path in ('delivery/rv30_completion_lock.json','delivery/prospective_lock.json'):
        write_json(tmp_path/path,{'original':path})
    added=('finance/delayed.py','finance/delayed_service.py','finance/delayed_validation.py')
    support=('tests/test_rv30_delayed.py','tests/test_rv30_delayed_lock.py',
             'configs/rv30_delayed_sip.json','delivery/DELAYED_REPLAY.md')
    for path in (name,*added,*support):
        p=tmp_path/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('installed:'+path)
    summary='evidence/validation.json'
    write_json(tmp_path/summary,dict.fromkeys(('passed','same_model_outputs','timing_and_no_backfill_passed',
        'budget_and_recovery_passed','original_full_workload_evidence_reused'),True))
    payload={'base_signature':frozen['signature'],'base_freeze_sha256':sha(provenance.FREEZE_PATH),
        'prior_completion_lock_sha256':sha(tmp_path/'delivery/rv30_completion_lock.json'),
        'original_plan_sha256':sha(tmp_path/'delivery/prospective_lock.json'),
        'prior_source_snapshot_commit':'original','original_prospective_plan_preserved':True,
        'publication_delay_minutes':15,'timely_forward_validation':False,
        'research_definitions_changed':False,'model_fitting_changed':False,'thresholds_changed':False,
        'market_origin_or_target_changed':False,
        'changes':{name:{'before_sha256':frozen['payload']['source_files'][name],'after_sha256':sha(tmp_path/name)}},
        'added_sources':{p:sha(tmp_path/p) for p in added},'support_sources':{p:sha(tmp_path/p) for p in support},
        'validation_summary':summary,'validation_files':{summary:sha(tmp_path/summary)}}
    path=tmp_path/'delivery/rv30_delayed_execution_lock.json'
    write_json(path,{'payload':payload,'signature':stable_hash(payload)})
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**kw:before)
    def predecessor(freeze,source_view=None,excluded_additions=()):
        assert freeze is frozen and source_view==frozen['payload']['source_files']
        assert set(excluded_additions)==set(added)
        return source_view,{'signature':'original-statistics'}
    monkeypatch.setattr(provenance,'_completion_sources',predecessor)
    return tmp_path,frozen,payload,path


def test_replay_extends_original_completion_without_replacing_it(replay_lock):
    root,frozen,payload,path=replay_lock
    sources,predecessor=provenance.effective_sources(frozen)
    assert sources['finance/delayed.py']==sha(root/'finance/delayed.py')
    assert predecessor['signature']=='original-statistics'


@pytest.mark.parametrize('name',['finance/provenance.py','finance/delayed.py','finance/delayed_validation.py',
    'finance/delayed_service.py','configs/rv30_delayed_sip.json','delivery/DELAYED_REPLAY.md',
    'tests/test_rv30_delayed.py','evidence/validation.json','delivery/rv30_completion_lock.json','delivery/prospective_lock.json'])
def test_altered_replay_source_or_original_predecessor_is_refused(replay_lock,name):
    root,frozen,payload,path=replay_lock
    with (root/name).open('a') as stream:stream.write('tampered')
    with pytest.raises(ValueError):provenance.effective_sources(frozen)


@pytest.mark.parametrize('field',['research_definitions_changed','model_fitting_changed','thresholds_changed',
    'market_origin_or_target_changed','timely_forward_validation'])
def test_replay_cannot_change_original_research_or_claim_timely_validation(replay_lock,field):
    root,frozen,payload,path=replay_lock;payload[field]=True
    write_json(path,{'payload':payload,'signature':stable_hash(payload)})
    with pytest.raises(ValueError,match='cannot revise'):provenance.effective_sources(frozen)
