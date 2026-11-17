"""Portable standard-library documentation and dependency identity verifier.

No numerical/source engine import, archive extraction, training, prediction,
market-array/weight opening, or scientific-success inference is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

PROHIBITED = {'.npy','.npz','.pt','.pth','.pkl','.pickle','.csv','.parquet','.arrow','.feather','.h5','.hdf5','.safetensors','.onnx','.bin'}
SMALL_SUFFIXES = {'.py','.md','.txt','.json','.xml','.toml','.yaml','.yml','.sh','.pdf','.docx','.png','.svg'}
MAX_SMALL_BYTES = 32 * 1024 * 1024


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): value.update(block)
    return value.hexdigest()


def within(root,relative):
    relative=Path(relative)
    if relative.is_absolute(): raise ValueError('Included path must be root-relative')
    target=(root/relative).resolve()
    target.relative_to(root)
    return target


def canonical_section(data):
    return data.rstrip(b'\r\n')


def supplement_sections(data):
    headings=list(re.finditer(rb'^## S(\d+)\. [^\r\n]+',data,re.M))
    result={}
    for i,match in enumerate(headings):
        key='S'+match.group(1).decode('ascii')
        if key in result: raise ValueError('Duplicate supplement section '+key)
        end=headings[i+1].start() if i+1<len(headings) else len(data)
        result[key]=data[match.start():end]
    return result


def table_one(data):
    start=data.index(b'**Table 1.**')
    end=data.index(b'**Table 2.**',start)
    return canonical_section(data[start:end])


def verify_previous_edition(root):
    """Exactly28 locks: original Table1 and previous S1-S27; no exceptions."""
    path=within(root,'review/PREVIOUS_EDITION_CONTENT_MANIFEST.json')
    previous=json.loads(path.read_text())
    current_main=within(root,'MAIN_PAPER_EN.md').read_bytes()
    current_supp=within(root,'SUPPLEMENT_EN.md').read_bytes()
    if hashlib.sha256(table_one(current_main)).hexdigest()!=previous['main_twenty_task_table']['canonical_sha256']:
        raise ValueError('Original twenty-task Table1 changed')
    sections=supplement_sections(current_supp)
    checked=[]
    for number in range(1,28):
        key='S'+str(number)
        if key not in previous['supplement_sections'] or key not in sections:
            raise ValueError('Missing previous section lock '+key)
        expected=previous['supplement_sections'][key]['canonical_sha256']
        actual=hashlib.sha256(canonical_section(sections[key])).hexdigest()
        if actual!=expected: raise ValueError('Previous edition section changed '+key)
        checked.append(key)
    # The independent preceding source copies additionally bind the lock's origin.
    predecessor_main=within(root,'review/previous_edition/MAIN_PAPER_EN.md')
    predecessor_supp=within(root,'review/previous_edition/SUPPLEMENT_EN.md')
    if digest(predecessor_main)!=previous['source_MAIN_sha256'] or digest(predecessor_supp)!=previous['source_SUPPLEMENT_sha256']:
        raise ValueError('Previous source copy identity mismatch')
    old_sections=supplement_sections(predecessor_supp.read_bytes())
    if hashlib.sha256(table_one(predecessor_main.read_bytes())).hexdigest()!=previous['main_twenty_task_table']['canonical_sha256']:
        raise ValueError('Lock does not match preceding Table1')
    for key in checked:
        if hashlib.sha256(canonical_section(old_sections[key])).hexdigest()!=previous['supplement_sections'][key]['canonical_sha256']:
            raise ValueError('Lock does not match preceding section '+key)
    return {'status':'passed','lock_count':28,'previous_Table1':'exact','previous_S1_S27':checked,
            'only_trailing_outer_CR_LF_ignored':True,'chronology_exceptions_allowed':False,
            'lock_manifest_sha256':digest(path),'preceding_source_copies_verified':True}


def snapshot_records(data):
    """One explicit normalized records list; no recursive directory discovery."""
    if isinstance(data,dict) and isinstance(data.get('entries'),list): return data['entries']
    if isinstance(data,dict) and isinstance(data.get('files'),list): return data['files']
    raise ValueError('Snapshot identity map requires entries/files list')


def verify_snapshot_map(root,dependencies):
    mapping=dependencies.get('snapshot_map')
    if not mapping: raise ValueError('Missing explicit snapshot identity map')
    path=within(root,mapping['path'])
    if digest(path)!=mapping['sha256']: raise ValueError('Snapshot identity map changed')
    records=snapshot_records(json.loads(path.read_text()))
    checked=[]; seen=set()
    for record in records:
        relative=record.get('included_path',record.get('snapshot_path',record.get('destination_relative')))
        if not relative: raise ValueError('Snapshot record missing included relative path')
        if relative in seen: raise ValueError('Duplicate snapshot destination '+relative)
        seen.add(relative)
        snapshot=within(root,relative)
        if snapshot.suffix.lower() in PROHIBITED or snapshot.suffix.lower() not in SMALL_SUFFIXES:
            raise ValueError('Snapshot is not approved small source/text '+relative)
        if snapshot.stat().st_size>MAX_SMALL_BYTES: raise ValueError('Snapshot exceeds small-file ceiling '+relative)
        expected=record['sha256']
        if digest(snapshot)!=expected: raise ValueError('Snapshot digest mismatch '+relative)
        if record.get('bytes') is not None and snapshot.stat().st_size!=record['bytes']:
            raise ValueError('Snapshot size mismatch '+relative)
        checked.append({'included_path':relative,'sha256':expected,'source_reference':record.get('source_path',record.get('source'))})
    if not checked: raise ValueError('No source/evidence snapshots committed')
    return {'status':'passed','map_sha256':mapping['sha256'],'count':len(checked),'snapshots':checked,
            'original_external_sources_opened_by_this_check':False}


def verify_manifest(root,manifest_path):
    data=json.loads(manifest_path.read_text())
    records=data['files'] if isinstance(data,dict) and 'files' in data else data
    if not isinstance(records,dict) or not records: raise ValueError('Empty/invalid documentation manifest')
    checked=[]
    for relative,expected in records.items():
        path=within(root,relative)
        if path==manifest_path: raise ValueError('Recursive manifest selfhash forbidden')
        if path.suffix.lower() in PROHIBITED or path.suffix.lower()=='.zip':
            raise ValueError('Market/weight/nestedarchive in documentation manifest '+relative)
        if not isinstance(expected,str) or digest(path)!=expected: raise ValueError('Documentation digest mismatch '+relative)
        checked.append({'id':'documentation:'+relative,'sha256':expected})
    return checked


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--manifest',default='DOCUMENTATION_MANIFEST.json')
    parser.add_argument('--dependencies',default='review/REPRODUCTION_DEPENDENCIES.json')
    parser.add_argument('--external',choices=['none','small','all'],default='none')
    parser.add_argument('--workspace-root',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(); root=args.root.resolve()
    report={'scope':'documentation_identity_and_legacy_content_only','external_mode':args.external,
            'checked':[],'deferred':[],'errors':[],'market_arrays_or_model_weights_opened':False,
            'numerical_reproduction_performed':False,'independent_market_confirmation':False,'predictive_success_claimed':False}
    try:
        manifest_path=within(root,args.manifest)
        report['manifest_sha256']=digest(manifest_path)
        report['checked'].extend(verify_manifest(root,manifest_path))
    except (OSError,ValueError,KeyError,TypeError) as exc:
        report['errors'].append({'id':'documentation_manifest','error':str(exc)})
    try:
        dependency_path=within(root,args.dependencies)
        dependencies=json.loads(dependency_path.read_text())
        report['dependency_map_sha256']=digest(dependency_path)
        ids=set()
        for entry in dependencies['entries']:
            identifier=entry['id']
            if identifier in ids: raise ValueError('Duplicate dependency id '+identifier)
            ids.add(identifier)
            policy=entry['verification_policy']; included=entry['location']=='included'
            suffix=Path(entry['path']).suffix.lower()
            if policy=='reference_only_never_open' or suffix in PROHIBITED:
                report['deferred'].append({'id':identifier,'reason':'reference_only_never_open'})
                continue
            if not included and args.external=='none':
                report['deferred'].append({'id':identifier,'reason':'external dependency not requested'}); continue
            archive=entry['kind']=='immutable_archive'
            if archive and (included or suffix!='.zip'): raise ValueError('Immutable archive must remain external ZIP')
            if not included and archive and args.external!='all':
                report['deferred'].append({'id':identifier,'reason':'immutable archive requires --external all'}); continue
            try:
                if included: path=within(root,entry['path'])
                elif args.workspace_root and entry.get('workspace_relative_path'):
                    path=within(args.workspace_root.resolve(),entry['workspace_relative_path'])
                else: path=Path(entry['path']).resolve()
                if path.suffix.lower() in PROHIBITED: raise ValueError('Never open raw data/arrays/weights')
                if not archive:
                    if path.suffix.lower() not in SMALL_SUFFIXES: raise ValueError('External dependency is not approved source/text/document')
                    if path.stat().st_size>MAX_SMALL_BYTES: raise ValueError('Small dependency exceeds32MiB')
                if digest(path)!=entry['sha256']: raise ValueError('Dependency SHA256 mismatch')
                if entry.get('bytes') is not None and path.stat().st_size!=entry['bytes']: raise ValueError('Dependency bytes mismatch')
                report['checked'].append({'id':identifier,'sha256':entry['sha256']})
            except (OSError,ValueError) as exc:
                report['errors'].append({'id':identifier,'error':str(exc)})
        report['snapshot_identity']=verify_snapshot_map(root,dependencies)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        report['errors'].append({'id':'dependencies_or_snapshots','error':str(exc)})
    try:
        report['legacy_content']=verify_previous_edition(root)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        report['errors'].append({'id':'previous_content_preservation','error':str(exc)})
    report['status']='passed' if not report['errors'] else 'failed'
    report['verification_limit']='Checks listed file identities and exact Table1/S1-S27 Markdown locks only. No rendered review, numerical replay, training, source-data availability or research-success claim.'
    text=json.dumps(report,indent=2,allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(text)
    print(text,end='')
    return 0 if not report['errors'] else 1

if __name__=='__main__': sys.exit(main())
