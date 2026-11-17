"""Stage, manifest, extract and actually replay a self-contained v3b capsule."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

DEADLINE = 1794205909.6228158
TOOLS = Path(__file__).resolve().parent
FORBIDDEN_WORKSPACE = '/home/ayaka/wslcode/aapl-research'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def guard_deadline():
    if time.time() >= DEADLINE:
        raise TimeoutError('Original absolute compute deadline reached; staged artifacts retained')


def copytree(source, destination):
    shutil.copytree(source, destination, dirs_exist_ok=True,
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.log', 'page_previews', '*_RAW.docx', 'lo_profile_*'))


def stage(source, package, registered_stats, *, include_manuscript=True):
    guard_deadline()
    rules = read(source/'protocol/RULES.json')
    stats = read(source/'analysis/STATISTICS.json')
    outer = read(source/'analysis/OUTER_RESULTS.json')
    assert outer['matrix_cells'] == 40 and len(outer['metrics']) == 80
    assert stats['rules_sha256'] == sha(source/'protocol/RULES.json')
    assert outer['statistics_sha256'] == sha(source/'analysis/STATISTICS.json')
    package.mkdir(parents=True, exist_ok=True)
    (package/'code').mkdir(exist_ok=True)
    for path in (source/'code').glob('*.py'):
        shutil.copy2(path, package/'code'/path.name)
    for filename in ('replay_context.py', 'reproduce_context_statistics.py', 'context_statistics.py'):
        shutil.copy2(TOOLS/filename, package/'code'/filename)
    shutil.copy2(registered_stats, package/'code/portable_stats_registered.py')
    directories = ['protocol', 'fits', 'analysis', 'delivery', 'design_review', 'evaluation_review', 'replay_review']
    if include_manuscript:
        directories.append('manuscript')
    for directory in directories:
        guard_deadline()
        if (source/directory).exists():
            copytree(source/directory, package/directory)
    for path in source.glob('*.md'):
        shutil.copy2(path, package/path.name)
    if (source/'audit').exists():
        (package/'audit').mkdir(exist_ok=True)
        for path in (source/'audit').glob('*.json'):
            shutil.copy2(path, package/'audit'/path.name)
    preceding = []
    for version in ('20261109_v3', '20261109_v3a'):
        receipt_path = source.with_name(version)/'audit/RELEASE.json'
        if receipt_path.exists():
            receipt = read(receipt_path)
            archive = Path(receipt['archive'])
            assert receipt['status'] == 'passed' and archive.exists()
            assert archive.stat().st_size == receipt['bytes']
            preceding.append({'version': version, 'archive_name': archive.name,
                'archive_sha256': receipt['archive_sha256'], 'bytes': receipt['bytes'],
                'release_receipt_sha256': sha(receipt_path),
                'previous_release_verified_after_extraction': receipt['manifest_verified_after_extraction'],
                'archive_bytes_rehashed_during_this_staging': False,
                'dependency_for_historical_evidence': True, 'needed_for_current_portable_entrypoints': False})
    if preceding:
        rehash_receipt = source/'audit/PRECEDING_ARCHIVES_REHASHED.json'
        if rehash_receipt.exists():
            assert read(rehash_receipt)['status'] == 'passed'
        (package/'PRECEDING_ARCHIVES.json').write_text(json.dumps({'archives': preceding,
            'prior_archives_not_duplicated_or_modified': True,
            'separate_current_archive_rehash_receipt': 'audit/PRECEDING_ARCHIVES_REHASHED.json' if rehash_receipt.exists() else None,
            'sha256_source': 'Previously successful immutable release receipts; file size rechecked during staging'}, indent=2)+'\n', encoding='utf-8')
    (package/'requirements.txt').write_text('numpy=='+__import__('numpy').__version__+'\npandas=='+__import__('pandas').__version__+'\n', encoding='utf-8')
    (package/'README.md').write_text('''# AAPL deterministic context comparison v3b

The paper title remains **Multi-Horizon Forecasting of Apple Stock Prices and Returns Across Temporal Scales**. This is a post-observation exposed-development addendum. It retains the fixed forty-cell matrix (two tasks, five folds, four deterministic models), matched controls and all negative findings. The sole forecast candidate is D_joint_shrink; raw D and A/B/C are prespecified diagnostics or controls. No independent confirmation is claimed, and no protected 2026Q3 values are included.

From this extracted folder, using the recorded NumPy/Pandas environment:

```
python code/replay_context.py --root .
python code/reproduce_context_statistics.py --root .
```

The first command verifies frozen source/model/checkpoint identities and replays the canonical first 128 input histories for all forty models in raw and shrunk modes. Canonical inner examples require bitwise equality. Single-row versus 128-row inference and outer examples sliced from the original 2048-row batches use the separately declared rtol=1e-12, atol=1e-15 engineering tolerance; these are not changes to predictive acceptance thresholds. It never fits a feature scaler, model or calibration coefficient and never constructs future labels.

The second command reconstructs all sixty-six pooled loss rows, eighty endpoint MSE/MAE/path-MSE rows, both frozen trading-day block intervals for all thirty-one controls and three diagnostic contrasts per task, and the frozen qualification checks. It uses only saved per-origin losses and complete per-fold day grids. These nominal intervals describe already exposed development history; no multiplicity adjustment or causal-market claim is made.

Fit accumulators preserve sample cursors and sufficient statistics for interrupted work. Historical absolute paths in protocol and orchestration sources are provenance, not portable dependencies. The input-only predictor and saved-loss reconstruction run without the original workspace, raw market labels, previous archives or network. Full retraining requires the separately registered original data identities and path remapping. Deterministic repetition is not a three-seed neural validation. Engineering replay does not establish predictive superiority, shared-horizon learning, trading profit or publication acceptance.

MANIFEST.json records every included file except itself. The external release receipt records actual ZIP extraction, manifest verification and execution with original workspace/network access blocked. Earlier v3/v3a archives are not overwritten.

PRECEDING_ARCHIVES.json identifies those earlier review packages by their verified release SHA-256 values and sizes. They retain the broader twenty-task historical study. They are not duplicated here and are not required by these two portable entrypoints.

The supplement preserves the v3a text in S1–S10 verbatim and adds this v3b study in S11–S14. Earlier sections describe the preceding experiments, source capsules and 180 neural configurations retained in the v3/v3a archives; this capsule includes the forty new deterministic context functions. Manuscript/preceding_version retains the six cited earlier manuscript files. Reproduction claims for the earlier experiments belong to their identified archives.
''', encoding='utf-8')
    for path in sorted(package.rglob('__pycache__'), reverse=True):
        shutil.rmtree(path)
    files = {str(path.relative_to(package)): sha(path) for path in sorted(package.rglob('*'))
             if path.is_file() and path.name != 'MANIFEST.json'}
    (package/'MANIFEST.json').write_text(json.dumps({'files': files, 'manifest_excludes_itself': True,
        'Q3_values_read': False, 'independent_confirmation': False,
        'candidate': rules['candidate']}, indent=2)+'\n', encoding='utf-8')
    return files


def detached_checks(package):
    hook = '''from pathlib import Path
import os,sys
forbidden=Path(FORBIDDEN).resolve()
runtime=Path(sys.prefix).resolve()
def boundary(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        path=Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(forbidden) and not path.is_relative_to(runtime):
            raise RuntimeError('Blocked original workspace access: '+str(path))
    if event in ('socket.connect','socket.connect_ex','socket.getaddrinfo','socket.bind'):
        raise RuntimeError('Blocked detached network access')
sys.addaudithook(boundary)
for kind in ('workspace','network'):
    try:
        if kind=='workspace':
            open(str(forbidden/'protocol_locked.json'),'rb')
        else:
            import socket
            socket.getaddrinfo('example.com',443)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Boundary test failed: '+kind)
'''.replace('FORBIDDEN', repr(FORBIDDEN_WORKSPACE))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    commands = [['code/replay_context.py', '--root', '.'],
                ['code/reproduce_context_statistics.py', '--root', '.']]
    receipts = []
    for command in commands:
        guard_deadline()
        program = hook+'\nimport runpy\nsys.path.insert(0,str(Path("code").resolve()))\nsys.argv='+repr(command)+'\nrunpy.run_path('+repr(command[0])+',run_name="__main__")\n'
        process = subprocess.run([sys.executable, '-I', '-c', program], cwd=package, env=env,
            capture_output=True, text=True, timeout=max(1., min(300., DEADLINE-time.time())))
        assert process.returncode == 0, (command, process.stdout, process.stderr)
        receipt = json.loads(process.stdout.strip().splitlines()[-1])
        assert receipt['status'] == 'passed'
        receipts.append({'arguments': command, 'receipt': receipt})
    return receipts


def archive_and_verify(package, output_receipt):
    guard_deadline()
    if (package/'manuscript').exists():
        assert read(package/'replay_review/DOCUMENT_REVIEW.json')['status'] == 'passed'
        assert read(package/'evaluation_review/REVIEW.json')['status'] == 'passed'
        assert read(package/'audit/VISUAL_REVIEW.json')['status'] == 'passed'
        for document in read(package/'replay_review/DOCUMENT_REVIEW.json')['documents']:
            stem = package/'manuscript'/(document['document']+'_EN')
            assert sha(stem.with_suffix('.md')) == document['source_sha256']
            assert sha(stem.with_suffix('.docx')) == document['docx_sha256']
            assert sha(stem.with_suffix('.pdf')) == document['pdf_sha256']
    archive = package.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=4) as z:
        for path in sorted(package.rglob('*')):
            if path.is_file():
                guard_deadline()
                z.write(path, str(path.relative_to(package.parent)))
    with tempfile.TemporaryDirectory(prefix='aapl-v3b-detached-') as temporary:
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            z.extractall(temporary)
        extracted = Path(temporary)/package.name
        files = read(extracted/'MANIFEST.json')['files']
        for name, expected in files.items():
            target = (extracted/name).resolve()
            assert target.is_relative_to(extracted.resolve())
            assert sha(target) == expected, name
        checks = detached_checks(extracted)
    final = {'status': 'passed', 'archive': str(archive), 'archive_sha256': sha(archive),
        'bytes': archive.stat().st_size, 'manifest_files': len(files),
        'manifest_verified_after_extraction': True, 'zip_integrity_passed': True,
        'original_workspace_and_network_access_blocked': True,
        'blocking_hooks_exercised': True, 'runtime_exception': str(Path(sys.prefix).resolve()),
        'detached_checks': checks, 'Q3_values_read': False, 'independent_confirmation': False,
        'research_goal_achieved': False, 'goal_status': 'active',
        'deadline_epoch': DEADLINE, 'affinity': sorted(os.sched_getaffinity(0)), 'GPU': False,
        'at_utc': datetime.now(timezone.utc).isoformat()}
    output_receipt.parent.mkdir(parents=True, exist_ok=True)
    output_receipt.write_text(json.dumps(final, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(final), flush=True)
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=TOOLS.parent)
    parser.add_argument('--package', type=Path)
    parser.add_argument('--registered-stats', type=Path)
    parser.add_argument('--stage-only', action='store_true')
    parser.add_argument('--without-manuscript', action='store_true')
    args = parser.parse_args()
    assert set(os.sched_getaffinity(0)).issubset({24, 26})
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    source = args.source.resolve()
    package = args.package.resolve() if args.package else source/'release/AAPL_MULTISCALE_V3B_CONTEXT'
    registered_stats = args.registered_stats or source.with_name('20261109_v3')/'code/portable_stats.py'
    files = stage(source, package, registered_stats, include_manuscript=not args.without_manuscript)
    if args.stage_only:
        print(json.dumps({'status': 'staged_not_yet_detached_verified', 'package': str(package), 'manifest_files': len(files)}))
    else:
        archive_and_verify(package, source/'replay_review/RELEASE.json')


if __name__ == '__main__':
    main()
