"""The forward collector is an additive lock, never a historical research rewrite."""
import hashlib
import subprocess

import pytest

from finance import provenance
from finance.io import read_json,sha,stable_hash,write_json


@pytest.fixture
def forward_lock(monkeypatch,tmp_path):
    monkeypatch.setattr(provenance,'ROOT',tmp_path)
    monkeypatch.setattr(provenance,'FREEZE_PATH',tmp_path/'delivery/rv30_freeze.json')
    changes=('finance/cli.py','finance/inference.py','finance/provenance.py')
    additions=('finance/prospective.py','finance/observation.py')
    originals={name:('previous:'+name).encode() for name in changes}
    before={name:hashlib.sha256(value).hexdigest() for name,value in originals.items()}
    for name in (*changes,*additions,'tests/test_rv30_prospective.py'):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('installed:'+name)
    frozen={'signature':'research','payload':{'source_files':before}}
    write_json(provenance.FREEZE_PATH,frozen)
    write_json(tmp_path/'delivery/rv30_replay_lock.json',{'signature':'preserved_replay'})
    write_json(tmp_path/'evidence/correctness.json',{'passed':True,'timing_and_no_backfill_checks_passed':True})
    performance={'passed':True,'complete_session_origins':346,'models':4,'prospective_observations_created':False,
        'source_files':{name:sha(tmp_path/name) for name in additions}}
    write_json(tmp_path/'evidence/performance.json',performance)
    payload={'base_signature':'research','base_freeze_sha256':sha(provenance.FREEZE_PATH),
        'prior_replay_lock_sha256':sha(tmp_path/'delivery/rv30_replay_lock.json'),'prior_source_snapshot_commit':'preserved',
        'historical_research_changed':False,'model_search_space_changed':False,'historical_thresholds_changed':False,
        'confirmed_weights_changed':False,'preregistered_before_live_observation':True,
        'changes':{name:{'before_sha256':before[name],'after_sha256':sha(tmp_path/name)} for name in changes},
        'added_sources':{name:sha(tmp_path/name) for name in additions},
        'support_sources':{'tests/test_rv30_prospective.py':sha(tmp_path/'tests/test_rv30_prospective.py')},
        'validation_summary':'evidence/correctness.json','performance_summary':'evidence/performance.json',
        'validation_files':{name:sha(tmp_path/name) for name in ('evidence/correctness.json','evidence/performance.json')}}
    lock=tmp_path/'delivery/rv30_prospective_execution_lock.json';write_json(lock,{'signature':stable_hash(payload),'payload':payload})
    monkeypatch.setattr(subprocess,'check_output',lambda args,**kw:originals[args[-1].split(':',1)[1]])
    def previous(freeze,source_view=None,excluded_additions=()):
        assert freeze is frozen and source_view==before and set(excluded_additions)==set(additions)
        return before,{'signature':'preserved_statistics'}
    monkeypatch.setattr(provenance,'_replay_sources',previous)
    return tmp_path,frozen,payload,lock


def test_prospective_extends_historical_chain_exactly(forward_lock):
    root,frozen,payload,lock=forward_lock
    sources,previous=provenance.effective_sources(frozen)
    assert previous['signature']=='preserved_statistics'
    assert sources['finance/prospective.py']==sha(root/'finance/prospective.py')


@pytest.mark.parametrize('target',['finance/cli.py','finance/prospective.py','evidence/correctness.json',
    'evidence/performance.json','delivery/rv30_replay_lock.json'])
def test_prospective_rejects_altered_sources_or_proofs(forward_lock,target):
    root,frozen,payload,lock=forward_lock
    with (root/target).open('a') as f:f.write('tamper')
    with pytest.raises(ValueError):provenance.effective_sources(frozen)


@pytest.mark.parametrize('field',['historical_research_changed','model_search_space_changed','historical_thresholds_changed','confirmed_weights_changed'])
def test_forward_lock_cannot_revise_completed_research(forward_lock,field):
    root,frozen,payload,lock=forward_lock;payload[field]=True
    write_json(lock,{'signature':stable_hash(payload),'payload':payload})
    with pytest.raises(ValueError,match='cannot change'):provenance.effective_sources(frozen)


def test_forward_needs_exact_full_session_workload(forward_lock):
    root,frozen,payload,lock=forward_lock
    p=root/payload['performance_summary'];d=read_json(p);d['complete_session_origins']=345;write_json(p,d)
    payload['validation_files'][payload['performance_summary']]=sha(p)
    write_json(lock,{'signature':stable_hash(payload),'payload':payload})
    with pytest.raises(ValueError,match='complete-workload'):provenance.effective_sources(frozen)
