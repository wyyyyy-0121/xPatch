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


def test_worker_keeps_standard_fold_receipt_identity(tmp_path):
    row = {'evidence': {'fold_directory': str(tmp_path / 'original'), 'receipt_sha256': 'receipt'}}
    command = audit.fold_worker_command(row, tmp_path / 'check.json', tmp_path / 'admission')
    assert command[command.index('--fold') + 1] == str(tmp_path / 'original')
    assert command[command.index('--receipt-sha256') + 1] == 'receipt'
    assert '--first-minute-fold' not in command


def test_worker_uses_original_split_stage_checks_for_first_h60_fold(tmp_path):
    row = {'profile': 'minute', 'horizon': 60, 'target': 'future_log_return', 'fold': 1,
           'evidence': {'controls_check_sha256': 'controls', 'neural_check_sha256': 'neural'}}
    command = audit.fold_worker_command(row, tmp_path / 'check.json', tmp_path / 'admission')
    assert '--first-minute-fold' in command and '--receipt-sha256' not in command
    assert command[command.index('--controls-check-sha256') + 1] == 'controls'
    assert command[command.index('--neural-check-sha256') + 1] == 'neural'


def test_worker_rejects_unknown_split_stage_layout(tmp_path):
    row = {'profile': 'minute', 'horizon': 60, 'target': 'future_log_return', 'fold': 2,
           'evidence': {'controls_check_sha256': 'controls', 'neural_check_sha256': 'neural'}}
    with pytest.raises(ValueError, match='Unexpected nonstandard'):
        audit.fold_worker_command(row, tmp_path / 'check.json', tmp_path / 'admission')
