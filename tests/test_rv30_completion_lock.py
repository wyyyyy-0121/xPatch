"""The completion amendment preserves the signed historical and forward plans."""
import hashlib
import subprocess

import pytest

from finance import provenance
from finance.io import read_json, sha, stable_hash, write_json


@pytest.fixture
def completion_lock(monkeypatch, tmp_path):
    monkeypatch.setattr(provenance, 'ROOT', tmp_path)
    monkeypatch.setattr(provenance, 'FREEZE_PATH', tmp_path / 'delivery/rv30_freeze.json')
    changed = ('finance/acceptance.py', 'finance/delivery.py', 'finance/provenance.py')
    added = ('finance/completion.py', 'finance/forward_runtime.py')
    support = ('tests/test_rv30_completion.py', 'tests/test_rv30_completion_lock.py', 'tests/test_rv30_forward_runtime.py')
    originals = {name: ('original:' + name).encode() for name in changed}
    before = {name: hashlib.sha256(value).hexdigest() for name, value in originals.items()}
    frozen = {'signature': 'original-research', 'payload': {'source_files': before}}
    write_json(provenance.FREEZE_PATH, frozen)
    write_json(tmp_path / 'delivery/rv30_prospective_execution_lock.json', {'signature': 'original-execution'})
    write_json(tmp_path / 'delivery/prospective_lock.json', {'signature': 'original-plan'})
    for name in (*changed, *added, *support):
        p = tmp_path / name;p.parent.mkdir(parents=True, exist_ok=True);p.write_text('new:' + name)
    write_json(tmp_path / 'evidence/validation.json', {'passed': True, 'early_acceptance_refused': True,
                                                   'positive_and_negative_selection_tested': True})
    payload = {'base_signature': frozen['signature'], 'base_freeze_sha256': sha(provenance.FREEZE_PATH),
        'prior_prospective_execution_lock_sha256': sha(tmp_path / 'delivery/rv30_prospective_execution_lock.json'),
        'prospective_plan_sha256': sha(tmp_path / 'delivery/prospective_lock.json'),
        'prior_source_snapshot_commit': 'original', 'research_definitions_changed': False,
        'thresholds_changed': False, 'model_fitting_changed': False, 'original_prospective_plan_preserved': True,
        'changes': {name: {'before_sha256': before[name], 'after_sha256': sha(tmp_path / name)} for name in changed},
        'added_sources': {name: sha(tmp_path / name) for name in added},
        'support_sources': {name: sha(tmp_path / name) for name in support},
        'validation_summary': 'evidence/validation.json',
        'validation_files': {'evidence/validation.json': sha(tmp_path / 'evidence/validation.json')}}
    path = tmp_path / 'delivery/rv30_completion_lock.json'
    write_json(path, {'payload': payload, 'signature': stable_hash(payload)})
    monkeypatch.setattr(subprocess, 'check_output', lambda args, **kw: originals[args[-1].split(':', 1)[1]])
    def predecessor(freeze, source_view=None, excluded_additions=()):
        assert freeze is frozen and source_view == before and set(excluded_additions) == set(added)
        return before, {'signature': 'original-statistics'}
    monkeypatch.setattr(provenance, '_prospective_sources', predecessor)
    return tmp_path, frozen, payload, path


def test_completion_extends_all_predecessors_without_rewriting(completion_lock):
    root, frozen, payload, path = completion_lock
    sources, predecessor = provenance.effective_sources(frozen)
    assert predecessor['signature'] == 'original-statistics'
    assert sources['finance/completion.py'] == sha(root / 'finance/completion.py')
    assert sources['finance/forward_runtime.py'] == sha(root / 'finance/forward_runtime.py')


@pytest.mark.parametrize('name', ['finance/acceptance.py', 'finance/completion.py', 'finance/forward_runtime.py',
    'tests/test_rv30_completion.py', 'evidence/validation.json', 'delivery/prospective_lock.json',
    'delivery/rv30_prospective_execution_lock.json'])
def test_any_altered_source_or_predecessor_is_rejected(completion_lock, name):
    root, frozen, payload, path = completion_lock
    with (root / name).open('a') as stream: stream.write('tampered')
    with pytest.raises(ValueError): provenance.effective_sources(frozen)


@pytest.mark.parametrize('field', ['research_definitions_changed', 'thresholds_changed', 'model_fitting_changed'])
def test_completion_cannot_change_the_research(completion_lock, field):
    root, frozen, payload, path = completion_lock
    payload[field] = True
    write_json(path, {'payload': payload, 'signature': stable_hash(payload)})
    with pytest.raises(ValueError, match='cannot revise'): provenance.effective_sources(frozen)


def test_only_the_exact_committed_previous_sources_are_allowed(completion_lock):
    root, frozen, payload, path = completion_lock
    payload['changes']['finance/acceptance.py']['before_sha256'] = 'another-version'
    write_json(path, {'payload': payload, 'signature': stable_hash(payload)})
    with pytest.raises(ValueError, match='Unregistered completion'): provenance.effective_sources(frozen)
