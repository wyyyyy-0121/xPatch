"""Commit post-acceptance status reports while preserving tested payload hashes."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

BASE = Path(__file__).resolve().parents[1]


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    assert set(os.sched_getaffinity(0)).issubset({24, 26})
    release = read(BASE / 'audit/RELEASE.json')
    archive = Path(release['archive']); root = archive.with_suffix('')
    assert release['status'] == 'passed' and sha(archive) == release['archive_sha256']
    old_manifest = read(root / 'MANIFEST.json')['files']
    assert all(sha(root / name) == digest for name, digest in old_manifest.items())
    subprocess.run([sys.executable, str(BASE / 'code/write_status.py')], check=True)
    assert read(BASE / 'analysis/GOAL_LEDGER.json')['goal_status'] == 'active'
    assert not read(BASE / 'analysis/GOAL_LEDGER.json')['research_goal_achieved']
    checks = {
        'longscale180_actual_predictor_and_statistical_replay': read(BASE / 'longscale_review/REVIEW.json')['status'],
        'mean100_fold_replay': read(BASE / 'mean_controls/audit/REPLAY.json')['status'],
        'history90_fixed_weight_replay': read(BASE / 'longscale_review/HISTORY_REVIEW.json')['status'],
        'BN17_independent_buffers_controls_and_gate': read(BASE / 'longscale_review/BN_REVIEW.json')['status'],
        'rendered_source_numeric_acceptance': read(BASE / 'longscale_review/RENDERED_DOCUMENTS_REVIEW.json')['status'],
        'all9_rendered_pages_visually_reviewed': read(BASE / 'audit/VISUAL_REVIEW.json')['status'],
        'actual_extraction_and_four_detached_entrypoints': release['status'],
    }
    assert all(value == 'passed' for value in checks.values()), checks
    final = {'status': 'bounded_addendum_delivered_research_goal_not_achieved',
        'goal_status': 'active', 'research_goal_achieved': False,
        'predictive_superiority_achieved': False, 'independent_confirmation': False,
        'shared_vs_independent_horizon_market_comparison': 'not_executed',
        'title_unchanged': True, 'all20_tasks_retained': True,
        'prior_v3_archive_unchanged': release['preceding_archive_sha256_verified_unchanged'],
        'checks': checks, 'detached_checks': release['detached_checks'],
        'training_cells_new': 150, 'training_cells_exact_reused': 30,
        'BN_fixed_weight_checkpoints': 17, 'history_fixed_weight_checkpoints': 90,
        'unconditional_mean_folds': 100, 'main_paper_pages': 5, 'supplement_pages': 4,
        'original_absolute_deadline_epoch': 1794205909.6228158,
        'all_experiment_cells_completed_before_deadline': True,
        'GPU_used': False, 'analysis_CPU_affinity': [24, 26], 'Q3_values_read': False,
        'next_work_rule': 'No change to original performance gates or opening of protected history; any new hypothesis must be distinguishable, frozen and within authorized remaining compute',
        'at_utc': datetime.now(timezone.utc).isoformat()}
    (BASE / 'audit/FINAL_AUDIT.json').write_text(json.dumps(final, indent=2))
    # Only these metadata reports are updated after successful execution. Their
    # wording may assert delivery now, without changing any tested function.
    changed = ['analysis/GOAL_LEDGER.json', 'PROGRESS_ZH.md', 'audit/FINAL_AUDIT.json']
    for name in changed:
        shutil.copy2(BASE / name, root / name)
    acceptance = {k: v for k, v in release.items() if k not in ('archive_sha256', 'bytes', 'manifest_files')}
    acceptance['executed_payload_archive_sha256'] = release['archive_sha256']
    acceptance['final_metadata_update'] = 'Goal/progress/acceptance reports only; tested code/data/models unchanged'
    (root / 'audit/DETACHED_ACCEPTANCE.json').write_text(json.dumps(acceptance, indent=2))
    allowed = set(changed + ['audit/DETACHED_ACCEPTANCE.json'])
    current = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'MANIFEST.json'}
    assert all(name in allowed or digest == old_manifest.get(name) for name, digest in current.items())
    assert set(old_manifest).issubset(current)
    (root / 'MANIFEST.json').write_text(json.dumps({'files': current, 'manifest_excludes_itself': True,
        'post_acceptance_changed_files': sorted(allowed), 'tested_code_data_and_models_unchanged': True,
        'research_goal_achieved': False, 'Q3_values_read': False}, indent=2))
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=4) as z:
        for path in sorted(root.rglob('*')):
            if path.is_file():
                z.write(path, str(path.relative_to(root.parent)))
    with tempfile.TemporaryDirectory(prefix='aapl-v3a-final-manifest-') as temporary:
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            z.extractall(temporary)
        extracted = Path(temporary) / root.name
        assert all(sha(extracted / name) == digest for name, digest in current.items())
    release['executed_payload_archive_sha256'] = release['archive_sha256']
    release['archive_sha256'] = sha(archive)
    release['bytes'] = archive.stat().st_size
    release['manifest_files'] = len(current)
    release['post_acceptance_changes'] = sorted(allowed)
    release['tested_code_data_models_unchanged_after_acceptance'] = True
    release['final_archive_reextracted_and_manifest_verified'] = True
    release['at_utc'] = datetime.now(timezone.utc).isoformat()
    (BASE / 'audit/RELEASE.json').write_text(json.dumps(release, indent=2))
    print(json.dumps(release), flush=True)


if __name__ == '__main__':
    main()
