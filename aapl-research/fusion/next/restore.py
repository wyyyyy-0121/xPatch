"""Verify a cloud return archive completely before restoring its fixed research evidence."""
import argparse
import os
from pathlib import Path,PurePosixPath
import shutil
import tarfile
import tempfile

from .common import ROOT,OUT,PLAN,digest,read_json,write_json,utc_now,source_identity


MANIFEST='outputs/fusion_next/cloud/return_manifest.json'
FIXED={'search_manifest.json','budget_projection.json','performance_admission.json','performance_cpu.json',
       'hardware_cloud.json','environment_setup.json','F3_complete.json','F4_complete.json','F5_complete.json',
       'system_freeze.json','confirmation_started.json','confirmation.json','confirmation_predictions.npz','usage_summary.json'}
FOLDERS={'training','trials','selections','seed_verification','oof','frozen_fits','deployment','usage_events',
         'performance','search_progress','environment_setup_events'}


def allowed(name):
    path=PurePosixPath(name)
    if name!=str(path) or path.is_absolute() or '..' in path.parts or '\\' in name:return False
    if path.parts[:2]!=('outputs','fusion_next') or len(path.parts)<3:return False
    relative=path.parts[2:]
    if len(relative)==1:return relative[0] in FIXED
    return relative[0] in FOLDERS and path.suffix in ('.json','.npz','.npy','.pt','.txt')


def restore(archive,expected_sha256):
    """Idempotent for identical files; refuse changed fits, preserve local F2 diagnostics."""
    if os.environ.get('FUSION_EXECUTION_PROFILE','local')!='local':raise RuntimeError('Restore cloud results on the local research host')
    archive=Path(archive).resolve()
    if digest(archive)!=expected_sha256:raise ValueError('Cloud return archive differs from its separate SHA receipt')
    destination=OUT/'cloud/restores'/expected_sha256
    with tempfile.TemporaryDirectory(prefix='return_restore_',dir=ROOT/'tmp') as temporary:
        staging=Path(temporary);seen=set()
        with tarfile.open(archive) as stream:
            for member in stream.getmembers():
                if not member.isfile() or member.name in seen or not (member.name==MANIFEST or allowed(member.name)):
                    raise ValueError('Unexpected or duplicated return member: '+member.name)
                if not (staging/member.name).resolve().is_relative_to(staging):raise ValueError('Return member leaves staging directory')
                seen.add(member.name)
            if MANIFEST not in seen:raise ValueError('Return manifest is missing')
            stream.extractall(staging,filter='data')
        receipt=read_json(staging/MANIFEST)
        if receipt['source_identity']!=source_identity() or receipt['plan_sha256']!=digest(PLAN):
            raise ValueError('Returned results used different frozen code or plan')
        if receipt['Q3_data_included'] or receipt['credentials_included'] or receipt['raw_development_market_data_included']:
            raise ValueError('Returned results declare prohibited data')
        if seen!={MANIFEST,*receipt['files']}:raise ValueError('Return member set differs from its manifest')
        if 'outputs/fusion_next/confirmation.json' not in receipt['files']:raise ValueError('Return has no completed confirmation record')
        operations=[];conflicts=[];already_present=0
        # All hashes and conflicts are checked before the first research file changes.
        for name,row in receipt['files'].items():
            source=staging/name
            if not allowed(name) or source.stat().st_size!=row['bytes'] or digest(source)!=row['sha256']:
                raise ValueError('Returned file identity differs: '+name)
            target=ROOT/name
            if not target.resolve().is_relative_to(OUT.resolve()):raise ValueError('Local result path leaves its output tree')
            preserve=None
            if target.exists():
                old_sha=digest(target)
                if old_sha==row['sha256']:already_present+=1;continue
                # Cloud CPU timing must replace the local-machine preflight for
                # cloud budgeting, while retaining both original measurements.
                if name.removeprefix('outputs/fusion_next/') in ('performance_cpu.json','usage_summary.json'):
                    preserve=OUT/'local_preflight'/f'{target.stem}_{old_sha}.json'
                    if preserve.exists() and digest(preserve)!=old_sha:conflicts.append(str(preserve.relative_to(ROOT)))
                else:conflicts.append(name)
            operations.append((source,target,preserve,row['sha256']))
        if conflicts:
            write_json(destination/'conflict.json',{'archive_sha256':expected_sha256,'conflicts':conflicts,
                       'existing_research_files_changed':False,'checked_at_utc':utc_now()})
            raise ValueError('Existing research results differ; nothing restored: '+', '.join(conflicts[:5]))
        intent={'archive_sha256':expected_sha256,'source_identity':source_identity(),'plan_sha256':digest(PLAN),
                'files_to_restore':[str(target.relative_to(ROOT)) for _,target,_,_ in operations],
                'already_identical_files':already_present,'started_at_utc':utc_now()}
        write_json(destination/'intent.json',intent)
        preserved=[]
        for source,target,backup,sha in operations:
            if backup:
                backup.parent.mkdir(parents=True,exist_ok=True)
                if not backup.exists():shutil.copyfile(target,backup)
                if digest(backup)!=digest(target):raise ValueError('Local diagnostic preservation failed')
                preserved.append(str(backup.relative_to(ROOT)))
            target.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.NamedTemporaryFile(prefix=target.name+'.restore_',dir=target.parent,delete=False) as output:
                pending=Path(output.name)
                with source.open('rb') as incoming:shutil.copyfileobj(incoming,output,1024*1024)
                output.flush();os.fsync(output.fileno())
            if digest(pending)!=sha:pending.unlink();raise ValueError('Restored temporary file differs')
            os.replace(pending,target)
            directory_fd=os.open(target.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(directory_fd)
            finally:os.close(directory_fd)
        write_json(destination/'return_manifest.json',receipt)
        result={**intent,'restored_files':len(operations),'preserved_local_diagnostics':preserved,
                'all_file_hashes_verified':True,'Q3_values_read':False,'completed_at_utc':utc_now()}
        write_json(destination/'receipt.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True);args=parser.parse_args()
    print(restore(args.archive,read_json(args.receipt)['sha256']))
