"""Keep recorded-source paths inside the project during predictor audits."""
import pytest

from . import historical_restore_check as audit


def test_audit_rejects_source_escape_even_when_digest_matches(tmp_path, monkeypatch):
    project = tmp_path / 'project'
    project.mkdir()
    outside = tmp_path / 'outside.py'
    outside.write_text('outside source\n')
    monkeypatch.setattr(audit, 'ROOT', project)
    with pytest.raises(ValueError, match='outside the project'):
        audit.checked_source('../outside.py', audit.digest(outside))


def test_audit_rejects_modified_recorded_source(tmp_path, monkeypatch):
    source = tmp_path / 'source.py'
    source.write_text('original\n')
    expected = audit.digest(source)
    source.write_text('changed\n')
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='source changed'):
        audit.checked_source('source.py', expected)
