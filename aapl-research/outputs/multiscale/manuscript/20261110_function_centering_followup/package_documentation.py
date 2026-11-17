"""Whitelist a thin accepted-documentation release. No raw data or old release recursion."""
from pathlib import Path
import hashlib
import json
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

from verify_documentation import (verify_previous_edition, verify_snapshot_map,
                                  PROHIBITED, SMALL_SUFFIXES)

BASE=Path(__file__).resolve().parent
PACKAGE='AAPL_FUNCTION_CENTERING_DOCUMENTATION_20261110'
ROOT_NAMES={
    'MAIN_PAPER_EN.md','MAIN_PAPER_EN.docx','MAIN_PAPER_EN.pdf','MAIN_PAPER_PDF_TEXT.txt','MAIN_PAPER_EN.docx.preservation.json',
    'SUPPLEMENT_EN.md','SUPPLEMENT_EN.docx','SUPPLEMENT_EN.pdf','SUPPLEMENT_PDF_TEXT.txt','SUPPLEMENT_EN.docx.preservation.json',
    'AUTHOR_INFORMATION.json','INTEGRATION_PROTOCOL.json','SOURCE_CHANGELOG.json','REPRODUCTION_GUIDE_EN.md',
    'build_integrated_documents.py','integrate_centering_sources.py','verify_documentation.py','package_documentation.py'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): h.update(block)
    return h.hexdigest()


def accepted_files(build):
    current_previews={}
    for document in build['documents']:
        name=document['name']; preview=Path(document['preview_directory']).resolve()
        preview.relative_to(BASE)
        expected=BASE/'page_previews'/name.lower()/document['pdf_sha256'][:16]
        if preview!=expected: raise ValueError('Preview must use current PDF hash directory')
        current_previews[str(preview.relative_to(BASE))]=document['preview_sha256']
        for image,expected_sha in document['preview_sha256'].items():
            if Path(image).name!=image or sha(preview/image)!=expected_sha: raise ValueError('Current preview identity changed')
    result=[]
    for path in sorted(BASE.rglob('*')):
        relative=path.relative_to(BASE)
        if relative.parts[0]=='release': continue
        if path.is_symlink(): raise ValueError('Symlink not admitted in documentation package '+str(relative))
        if not path.is_file(): continue
        if '__pycache__' in relative.parts or any(part.startswith('lo_') for part in relative.parts): continue
        if 'initial_render' in relative.parts: continue
        if path.suffix.lower() in PROHIBITED or path.suffix.lower()=='.zip': raise ValueError('Market/model/raw/nestedarchive file in new stage '+str(relative))
        if len(relative.parts)==1:
            if path.name in ROOT_NAMES: result.append(path)
            continue
        first=relative.parts[0]
        if first=='page_previews':
            directory=str(relative.parent)
            if directory in current_previews and path.name in current_previews[directory]: result.append(path)
            continue
        if first not in {'figures','evidence','review','code'}: continue
        if 'release' in relative.parts or (first=='review' and relative.parts.count('previous_edition')>1):
            raise ValueError('Recursive preceding release/snapshot directory forbidden '+str(relative))
        if path.name.endswith('_RAW.docx'): continue
        if path.suffix.lower() not in SMALL_SUFFIXES: raise ValueError('Non-whitelisted documentation file '+str(relative))
        result.append(path)
    required=ROOT_NAMES-{'SOURCE_CHANGELOG.json'}
    present={p.name for p in result if p.parent==BASE}
    if not required<=present: raise ValueError('Required delivery files missing '+str(sorted(required-present)))
    return result


def main():
    acceptance=json.loads((BASE/'review/DOCUMENT_ACCEPTANCE.json').read_text())
    if not str(acceptance['status']).startswith('passed'): raise ValueError('Independent document acceptance required')
    for key in ['research_goal_achieved','full_goal_achieved','stable_method_success','unseen_confirmation_complete']:
        if acceptance.get(key,False) is not False: raise ValueError('Document delivery must not claim full research success')
    receipt_hashes=acceptance.get('review_receipts',{})
    if receipt_hashes.get('DOCUMENT_BUILD.json') != sha(BASE/'review/DOCUMENT_BUILD.json'):
        raise ValueError('Acceptance does not bind the current document build')
    for relative,expected in receipt_hashes.items():
        target=(BASE/'review'/relative).resolve()
        target.relative_to(BASE/'review')
        if target.suffix.lower() in PROHIBITED or sha(target)!=expected:
            raise ValueError('Acceptance review receipt identity changed '+relative)
    if acceptance.get('source_changelog_sha256') is not None and sha(BASE/'SOURCE_CHANGELOG.json') != acceptance['source_changelog_sha256']:
        raise ValueError('Accepted source changelog changed')
    legacy=verify_previous_edition(BASE)
    dependencies=json.loads((BASE/'review/REPRODUCTION_DEPENDENCIES.json').read_text())
    snapshot=verify_snapshot_map(BASE,dependencies)
    build=json.loads((BASE/'review/DOCUMENT_BUILD.json').read_text())
    if {x['name'] for x in build['documents']}!={'MAIN_PAPER','SUPPLEMENT'} or len(build['documents'])!=2:
        raise ValueError('Both final documents must be built')
    for document in build['documents']:
        for extension,key in [('md','source_sha256'),('docx','docx_sha256'),('pdf','pdf_sha256')]:
            if sha(BASE/(document['name']+'_EN.'+extension))!=document[key]: raise ValueError('Current built document identity changed')
    accepted=accepted_files(build)
    manifest={str(path.relative_to(BASE)):sha(path) for path in accepted}
    manifest_path=BASE/'DOCUMENTATION_MANIFEST.json'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    destination=BASE/'release'/PACKAGE
    destination.mkdir(parents=True,exist_ok=False)
    for source in accepted+[manifest_path]:
        target=destination/source.relative_to(BASE); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)
    archive=destination.with_suffix('.zip')
    with ZipFile(archive,'x',compression=ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(destination.rglob('*')):
            if path.is_file(): z.write(path,str(Path(PACKAGE)/path.relative_to(destination)))
    report={'status':'packaged_pending_actual_detached_validation','archive':str(archive),'archive_sha256':sha(archive),
            'archive_bytes':archive.stat().st_size,'package_directory':str(destination),'manifest_sha256':sha(manifest_path),
            'manifest_file_count':len(manifest),'legacy_content_lock_count':legacy['lock_count'],'snapshot_count':snapshot['count'],
            'raw_market_or_model_files_included':False,'self_contained_full_retraining':False,'numerical_replay_performed':False,
            'independent_market_confirmation':False,'research_goal_achieved':False}
    (BASE/'release/PACKAGE_BUILD.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
