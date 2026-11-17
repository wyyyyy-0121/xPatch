"""Finalize and actually extract/replay the isolated v3a review addendum."""
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
V3 = BASE.parent / '20261109_v3'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def copytree(src, dest):
    shutil.copytree(src, dest, dirs_exist_ok=True,
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'lo_profile_*', '*.log', 'page_previews', '*_RAW.docx'))


def main():
    assert set(os.sched_getaffinity(0)).issubset({24, 26})
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    build = read(BASE / 'audit/DOCUMENT_BUILD.json')
    assert build['main_page_target_4_to_6']
    assert read(BASE / 'audit/VISUAL_REVIEW.json')['status'] == 'passed'
    assert read(BASE / 'longscale_review/RENDERED_DOCUMENTS_REVIEW.json')['status'] == 'passed'
    assert read(BASE / 'bn_recalibration/FINAL_AUDIT.json')['status'] == 'passed'
    root = BASE / 'release/AAPL_MULTISCALE_V3A_ADDENDUM'
    assert read(BASE / 'audit/STAGING.json')['longscale_examples'] == 180
    for folder in ('code', 'manuscript', 'longscale_review', 'audit', 'analysis'):
        copytree(BASE / folder, root / folder)
    for name in ('PLAN_ZH.md', 'PLAN_TO_COMPLETION_ZH.md', 'PROGRESS_ZH.md'):
        shutil.copy2(BASE / name, root / name)
    bn = root / 'bn_recalibration'; bn.mkdir(exist_ok=True)
    source = BASE / 'bn_recalibration'
    for pattern in ('*.py', '*.json', '*.md'):
        for path in source.glob(pattern):
            shutil.copy2(path, bn / path.name)
    copytree(source / 'replay_capsule', bn / 'replay_capsule')
    copytree(source / 'all_control_losses', bn / 'all_control_losses')
    # The exact portable inference modules originated in unchanged v3.
    for name in ('portable_predictor.py', 'portable_stats.py', 'tiny_architecture.py', 'format_documents.py'):
        shutil.copy2(V3 / 'code' / name, root / 'code' / name)
    receipt = read(V3 / 'audit/RELEASE.json')
    assert sha(Path(receipt['archive'])) == receipt['archive_sha256']
    # Keep only generated source/data payloads, not Python import caches.
    for path in sorted(root.rglob('__pycache__'), reverse=True):
        shutil.rmtree(path)
    (root / 'README.md').write_text('''# AAPL multi-scale review addendum v3a

The paper title remains **Multi-Horizon Forecasting of Apple Stock Prices and Returns Across Temporal Scales**. Read `manuscript/MAIN_PAPER_EN.pdf` and `SUPPLEMENT_EN.pdf`; editable Word and Markdown are included. Author: Wang Yang, High School Affiliated to Shanghai Jiao Tong University, Shanghai 200439, China; the author fills the email.

This addendum follows inspection of v3. It is exposed development evidence. It adds 150 long-scale training configurations to thirty exact reused configurations, fit-only unconditional means for all twenty tasks, ninety fixed-weight historical-input probes, and a separately gated fixed-weight BatchNorm diagnostic. See the frozen plans, source identities and final receipts for the full accounting.

**Research status:** the original stable predictive-improvement Goal remains unachieved and active. Independent confirmation and the separate shared-versus-independent horizon market experiment remain unfinished. No protected 2026Q3 values are opened or included. Successful restoration, document rendering, or detached replay cannot establish predictive superiority. No external submission, public deposit, DOI or trading-profit claim is made.

## Dependency and portable entrypoints

`PRECEDING_ARCHIVE.json` identifies the unchanged v3 archive by SHA-256 and size. It is not duplicated inside this addendum. That archive retains the complete original comparison, preceding small-model/fusion study, controlled factorials and original twenty-task inference/statistics acceptance. The addendum entrypoints below execute from this extracted folder with the recorded NumPy/PyTorch CPU environment, without the prior archive or original workspace:

```
python code/replay_addendum.py --root .
python code/reproduce_addendum_statistics.py --root .
```

The first entrypoint replays the first canonical batch for all 180 long-scale configurations, plus the three historical-input conditions for ninety baseline-initialized configurations. Inputs contain completed past log-relative closes and each real current-price anchor. Frozen model scalers/parameters and stopping alphas are used; the entrypoint never fits a scaler/alpha or constructs a future target. Expected forecasts come from the archived outputs. Exact replay is tied to the recorded CPU environment and canonical batch shape; it is not a cross-batch bitwise portability claim.

The statistical entrypoint recalculates 66 long-scale pooled rows, 360 seed/mode rows, 120 paired intervals, 130 phase/year/event subsets, 280 mean-control pooled rows and 280 intervals, plus ninety history rows, 24 pooled history conditions and 24 paired intervals. New mean-control path/MAE values use per-origin arrays. Preceding v3 auxiliary path/MAE values use weighted fold receipts here; their independent per-origin reproduction remains in v3. These are exposed-development descriptive intervals with fixed blocks and no multiplicity adjustment.

`bn_recalibration/replay_capsule` contains its own usage and frozen 17-model input-only replay/statistics entrypoints. Its complete original predictions and resumable intermediate states remain in the originating workspace. The capsule and all-control losses retain the evidence needed for the stated detached checks. Read its original tolerance failure and pre-outer versioned numerical revision; no relaxed gate is described as bitwise cross-batch equality.

## Reproduction boundaries

`PORTABLE_LOCATIONS.json` maps all 180 long-scale identities to the included checkpoint/prediction folders. The full raw pre-cutoff bar capsule is 2026Q3-free and hash identified. Long-scale best/initial/relevant coupled checkpoints, calibrated forecasts and receipts are included. Training wrappers preserve historical absolute paths for provenance; full retraining requires registered source-data identities, partitions, environment and path remapping. The included portable inference/statistics entrypoints do not make those orchestration scripts directly portable.

The six preceding manuscript files are unchanged under `manuscript/preceding_version`. `MANIFEST.json` hashes all included files except itself and the post-test external release receipt. The release acceptance records actual extraction, hash checks, blocked original-workspace/network access and executed entrypoints. This package is a local review artifact, not evidence of independent market validation.
''')
    files = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'MANIFEST.json'}
    (root / 'MANIFEST.json').write_text(json.dumps({'files': files, 'manifest_excludes_itself': True,
        'research_goal_achieved': False, 'Q3_values_read': False}, indent=2))
    archive = root.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=4) as z:
        for path in sorted(root.rglob('*')):
            if path.is_file():
                z.write(path, str(path.relative_to(root.parent)))
    audit_hook = '''from pathlib import Path
import sys,os
forbidden=Path('/home/ayaka/wslcode/aapl-research').resolve()
runtime=Path(sys.prefix).resolve()
def boundary(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        path=Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(forbidden) and not path.is_relative_to(runtime):
            raise RuntimeError('Detached original workspace access: '+str(path))
    if event in ('socket.connect','socket.connect_ex','socket.getaddrinfo'):
        raise RuntimeError('Detached network access')
sys.addaudithook(boundary)
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    checks = []
    with tempfile.TemporaryDirectory(prefix='aapl-v3a-detached-') as temporary:
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            z.extractall(temporary)
        extracted = Path(temporary) / root.name
        assert all(sha(extracted / name) == digest for name, digest in read(extracted / 'MANIFEST.json')['files'].items())
        # The BN owner supplies the final capsule commands, which are frozen
        # in a JSON contract rather than shell-interpolated strings.
        commands = [['code/replay_addendum.py', '--root', '.'],
                    ['code/reproduce_addendum_statistics.py', '--root', '.']]
        contract = read(extracted / 'bn_recalibration/replay_capsule/ENTRYPOINTS.json')
        commands += contract['commands_from_addendum_root']
        for arguments in commands:
            target = arguments[0]
            program = audit_hook + '\nimport runpy,json\n' + \
                'sys.path.insert(0,str(Path("code").resolve()))\n' + \
                'sys.path.insert(0,str(Path(' + repr(str(Path(target).parent)) + ').resolve()))\n' + \
                'sys.argv=' + repr(arguments) + '\nrunpy.run_path(' + repr(target) + ',run_name="__main__")\n'
            process = subprocess.run([sys.executable, '-c', program], cwd=extracted, env=env,
                                     capture_output=True, text=True, check=True)
            final = json.loads(process.stdout.strip().splitlines()[-1])
            assert final['status'] == 'passed', final
            checks.append({'arguments': arguments, 'receipt': final})
            print(json.dumps(checks[-1]), flush=True)
    final = {'status': 'passed', 'archive': str(archive), 'bytes': archive.stat().st_size,
        'archive_sha256': sha(archive), 'manifest_files': len(files),
        'manifest_verified_after_extraction': True, 'zip_integrity_passed': True,
        'original_workspace_and_network_access_blocked': True,
        'runtime_exception': str(Path(sys.prefix).resolve()), 'detached_checks': checks,
        'preceding_archive_sha256_verified_unchanged': receipt['archive_sha256'],
        'research_goal_achieved': False, 'goal_status': 'active', 'Q3_values_read': False,
        'at_utc': datetime.now(timezone.utc).isoformat()}
    (BASE / 'audit/RELEASE.json').write_text(json.dumps(final, indent=2))
    print(json.dumps(final), flush=True)


if __name__ == '__main__':
    main()
