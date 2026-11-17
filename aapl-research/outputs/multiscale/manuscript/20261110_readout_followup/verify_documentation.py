"""Verify documentation identities only, using the Python standard library.

This program never imports project modules, runs a builder, extracts an archive,
loads market/model arrays, downloads data, trains, predicts, or computes scores.
The manifest is an identity record, not a cryptographic signature or market test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def canonical_section(data):
    # Only trailing CR/LF bytes at an outer section separator may differ.
    return data.rstrip(b'\r\n')


def supplement_sections(data):
    headings = list(re.finditer(rb'^## S(\d+)\. [^\r\n]+', data, re.M))
    sections = {}
    for i, match in enumerate(headings):
        key = 'S'+match.group(1).decode('ascii')
        if key in sections:
            raise ValueError('Duplicate supplement heading: '+key)
        end = headings[i+1].start() if i+1 < len(headings) else len(data)
        sections[key] = data[match.start():end]
    return sections


def table_one(data):
    start = data.index(b'**Table 1.**')
    end = data.index(b'**Table 2.**', start)
    return canonical_section(data[start:end])


def within(root, relative):
    target = (root/relative).resolve()
    target.relative_to(root)
    return target


def verify_previous_edition(root):
    """Strictly preserve preceding S1-S21 and original Table1; no exceptions.

    This lock is additional to the inherited older-baseline/chronology audit.
    Logging a reason cannot permit changes to the immediately preceding text.
    """
    path = within(root, 'review/PREVIOUS_EDITION_CONTENT_MANIFEST.json')
    previous = json.loads(path.read_text())
    main_bytes = within(root, 'MAIN_PAPER_EN.md').read_bytes()
    supplement_bytes = within(root, 'SUPPLEMENT_EN.md').read_bytes()
    expected_table = previous['main_twenty_task_table']['canonical_sha256']
    if hashlib.sha256(table_one(main_bytes)).hexdigest() != expected_table:
        raise ValueError('Previous-edition original Table1 bytes changed')
    sections = supplement_sections(supplement_bytes)
    expected_sections = previous['supplement_sections']
    preserved = []
    for number in range(1, 22):
        key = f'S{number}'
        if key not in expected_sections or key not in sections:
            raise ValueError('Previous-edition section lock missing: '+key)
        actual = hashlib.sha256(canonical_section(sections[key])).hexdigest()
        if actual != expected_sections[key]['canonical_sha256']:
            raise ValueError('Previous-edition section bytes changed: '+key)
        preserved.append(key)
    return {'status': 'passed', 'manifest_sha256': digest(path),
            'original_twenty_task_Table1': 'exact_preservation_passed',
            'exact_S1_S21_sections': preserved,
            'only_trailing_CR_LF_bytes_ignored': True,
            'chronology_exception_cannot_override_this_lock': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--manifest', default='DOCUMENTATION_MANIFEST.json',
                        help='Root-relative documentation path to SHA256 mapping.')
    parser.add_argument('--dependencies', default='review/REPRODUCTION_DEPENDENCIES.json')
    parser.add_argument('--external', choices=('none', 'small', 'all'), default='none',
                        help='Optional external hashes; all includes the 8.17GB immutable ZIP.')
    parser.add_argument('--workspace-root', type=Path,
                        help='Optional relocated project root for external workspace-relative paths.')
    parser.add_argument('--output', type=Path, help='Optional JSON receipt; otherwise stdout only.')
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = within(root, args.manifest)
    documentation = json.loads(manifest_path.read_text())
    if 'files' in documentation and isinstance(documentation['files'], dict):
        documentation = documentation['files']
    dependencies_path = within(root, args.dependencies)
    manifest = json.loads(dependencies_path.read_text())
    report = {'scope': 'documentation_and_dependency_identity_only',
              'manifest_sha256': digest(manifest_path),
              'dependency_map_sha256': digest(dependencies_path), 'external_mode': args.external,
              'checked': [], 'deferred': [], 'errors': [],
              'market_arrays_or_model_weights_loaded': False,
              'numerical_reproduction_performed': False,
              'independent_market_confirmation': False,
              'predictive_success_claimed': False}
    prohibited = {'.npz', '.npy', '.pt', '.pth', '.pkl', '.pickle', '.csv', '.parquet', '.arrow'}
    for relative, expected in documentation.items():
        try:
            path = within(root, relative)
            if path == manifest_path:
                raise ValueError('Documentation manifest must exclude its own recursive hash')
            if path.suffix.lower() in prohibited:
                raise ValueError('Raw market/model files do not belong in the documentation manifest')
            if not isinstance(expected, str) or digest(path) != expected:
                raise ValueError('Documentation SHA256 mismatch')
            report['checked'].append({'id': 'documentation:'+relative, 'sha256': expected})
        except (OSError, ValueError) as exc:
            report['errors'].append({'id': 'documentation:'+relative, 'error': str(exc)})
    for entry in manifest['entries']:
        included = entry['location'] == 'included'
        policy = entry['verification_policy']
        if policy == 'reference_only_never_open':
            report['deferred'].append({'id': entry['id'], 'reason': policy})
            continue
        if not included and (args.external == 'none' or
                             (args.external == 'small' and entry['kind'] == 'immutable_archive')):
            report['deferred'].append({'id': entry['id'], 'reason': 'external dependency not requested'})
            continue
        try:
            if included:
                path = within(root, entry['path'])
            elif args.workspace_root and entry.get('workspace_relative_path'):
                path = within(args.workspace_root.resolve(), entry['workspace_relative_path'])
            else:
                path = (root/entry['path']).resolve()
            if path.suffix.lower() in prohibited:
                raise ValueError('Refuse market/model/raw-data file in documentation verifier')
            actual = digest(path)
            if actual != entry['sha256']:
                raise ValueError('SHA256 mismatch')
            if entry.get('bytes') is not None and path.stat().st_size != entry['bytes']:
                raise ValueError('Byte-size mismatch')
            report['checked'].append({'id': entry['id'], 'sha256': actual})
        except (OSError, ValueError) as exc:
            report['errors'].append({'id': entry['id'], 'error': str(exc)})

    try:
        baseline = json.loads(within(root, 'review/source_snapshot/BASELINE_CONTENT_MANIFEST.json').read_text())
        main_bytes = within(root, 'MAIN_PAPER_EN.md').read_bytes()
        supplement_bytes = within(root, 'SUPPLEMENT_EN.md').read_bytes()
        main_table_hash = hashlib.sha256(table_one(main_bytes)).hexdigest()
        if main_table_hash != baseline['main_twenty_task_table']['canonical_sha256']:
            raise ValueError('Original twenty-task Table1 caption or values changed')
        integrated = supplement_sections(supplement_bytes)
        edit_log_path = root/'review/CHRONOLOGY_EDIT_LOG.json'
        edits = json.loads(edit_log_path.read_text()).get('sections', {}) if edit_log_path.exists() else {}
        preserved, exceptions = [], []
        for number in range(1, 19):
            key = f'S{number}'
            actual = hashlib.sha256(canonical_section(integrated[key])).hexdigest()
            expected = baseline['supplement_sections'][key]['canonical_sha256']
            if actual == expected:
                preserved.append(key)
                continue
            edit = edits.get(key, {})
            if (edit.get('baseline_sha256') != expected or edit.get('integrated_sha256') != actual
                    or not isinstance(edit.get('reason'), str) or not edit['reason'].strip()):
                raise ValueError('Unlogged preceding supplement content change: '+key)
            exceptions.append({'section': key, 'reason': edit['reason'],
                               'requires_separate_claim_review': True})
        report['content_preservation'] = {'twenty_task_table': 'passed',
                                           'exact_S1_S18_sections': preserved,
                                           'logged_chronology_exceptions': exceptions,
                                           'outer_section_separator_newlines_ignored': True}
    except (OSError, ValueError, KeyError) as exc:
        report['errors'].append({'id': 'baseline_content_preservation', 'error': str(exc)})
    try:
        report['previous_edition_content_preservation'] = verify_previous_edition(root)
    except (OSError, ValueError, KeyError) as exc:
        report['errors'].append({'id': 'previous_edition_content_preservation', 'error': str(exc)})
    report['status'] = 'passed' if not report['errors'] else 'failed'
    report['verification_limit'] = ('Only listed hashes and specified preceding Markdown content are checked. '
                                   'Rendered layout, scientific claims, offline numerical replay, full retraining '
                                   'and unseen-market performance are outside this program.')
    text = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end='')
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
