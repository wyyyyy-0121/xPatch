"""Return immutable follow-up evidence without prices, caches or final data.

Both the negative pilot path and a fully fitted engineering path are supported.
A returned export still requires a separate actual local CPU acceptance.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time

from fusion.next.common import ROOT,OUT as BASE,digest,identity,read_json,utc_now,write_json
from scripts import followup_guard as guard
from scripts import fit_fusion_followup as fit
from scripts import review_fusion_pilot as pilot
from scripts.followup_admission import check_frozen,identified,verify_delivery
from scripts.followup_runtime import FOLLOW,DEADLINE
from scripts.run_fusion_followup import (attempts,spent_seconds,execution_locks,
    original_idle,bind_guarded_worker,stop_guard)


ENTRY = Path(__file__).resolve()
CPU_SECONDS = 2*3600
MANIFEST = 'return_manifest.json'
ROOTS = {'attempts','pilot','extension','reviews','extension_reviews','delivery','delivery_fit'}


def allowed(name):
    path = PurePosixPath(name)
    if (path.is_absolute() or str(path)!=name or '..' in path.parts
            or len(path.parts)<2 or path.parts[0] not in ROOTS
            or any(p in ('data','representation_cache','local_delivery','local_packages','2026Q2','2026Q3') for p in path.parts)):
        return False
    if (path.parts[0]=='attempts' and path.parts[1].startswith('return_')
            and path.name in ('guard_state.json','guard.log','worker.log')):
        # The bundler itself is currently guarded. Its periodically rewritten
        # state/logs cannot be immutable archive inputs; its request is retained.
        return False
    if path.suffix in ('.json','.log','.md'):return True
    if path.suffix=='.pt':
        return 'training' in path.parts or 'deployment' in path.parts
    if 'deployment' in path.parts and path.name in ('ridge.npz','lightgbm.txt'):return True
    return (len(path.parts)==8 and path.parts[0] in ('pilot','extension') and path.parts[1]=='trials'
            and path.parts[2]==path.parts[0] and path.parts[4] in pilot.QUARTERS
            and path.parts[5] in ('seed_2026','seed_2027','seed_2028')
            and path.parts[6].startswith('updates_') and path.name=='prediction.npz')


def stopped_attempts(run):
    """No state file alone can establish that its exact worker group exited."""
    for mode,cap in (('probe',600),('pilot',6*3600),('extend',6*3600),('fit',6*3600)):
        for path in attempts(run,mode):
            if path.exists():spent_seconds(path,cap)


def completion(manifest_path,manifest=None):
    manifest = manifest or check_frozen(manifest_path)
    run = FOLLOW/'runs'/manifest['identity'];stopped_attempts(run)
    saved,review = pilot.checked_report(manifest_path,manifest)
    if review['selected_for_possible_expansion'] is None:
        if any((run/name).exists() for name in ('extension','delivery','delivery_fit')) or attempts(run,'fit') or attempts(run,'extend'):
            raise ValueError('A negative pilot cannot discard an extension or engineering attempt')
        model = manifest['delivery']['deployment_identity']
        archive = manifest['delivery']['files']['local_packages/'+model+'.tar']
        return {'path':'negative_pilot_retaining_original_package','pilot_review':saved,
                'retained_original_model_identity':model,'retained_original_package_sha256':archive['sha256'],
                'new_model_exported':False,'actual_new_model_CPU_acceptance_passed':False,
                'independent_research_success_claimed':False,'Q2_Q3_scored':False}
    freeze_path = run/'delivery/engineering_freeze.json'
    result_path = run/'delivery_fit/fit_result.json'
    if not freeze_path.exists() or not result_path.exists():
        raise RuntimeError('Selected follow-up still needs its complete reviewed three-seed engineering fit')
    actual_path,frozen = fit.freeze_delivery(manifest_path,manifest)
    result = identified(result_path)
    if (actual_path!=freeze_path or result['manifest_identity']!=manifest['identity']
            or result['engineering_freeze_identity']!=frozen['identity']
            or result['engineering_freeze_sha256']!=digest(freeze_path)
            or result['stage']!='engineering_fit_export_complete_local_acceptance_required'
            or result['fresh_system_seeds']!=[2026,2027,2028]
            or result['fit_end']!=frozen['fit_policy']['fit_end']
            or result['total_coverage_equivalent_epochs']!=2
            or result['original_expert_bytes_reused'] is not True
            or result['development_qualification_passed']!=frozen['development_qualification_passed']
            or result['Q2_Q3_scored'] is not False or result['actual_CPU_acceptance_passed'] is not False
            or result['independent_research_success_claimed'] is not False):
        raise ValueError('Engineering export completion differs from its fixed review and freeze')
    destination = run/'delivery_fit'
    if any(r['step']!=frozen['fit_policy']['total_updates'] for r in fit.checkpoint_progress(destination,frozen).values()):
        raise ValueError('Engineering checkpoint is not a complete two-epoch fit')
    systems = [{'config':fit.system_config(frozen,seed),
                'checkpoint':str(fit.checkpoint_path(destination,frozen,seed))} for seed in (2026,2027,2028)]
    deployment = destination/'deployment/pre_Q2'
    model = fit.verify_export(deployment,frozen,digest(freeze_path),systems)
    if (result['model_identity']!=model['identity'] or result['deployment_directory']!=str(deployment.relative_to(ROOT))
            or result['deployment_manifest_sha256']!=digest(deployment/'deployment.json')):
        raise ValueError('Completed export points at another deployment')
    completed_guard = False
    for path in attempts(run,'fit'):
        if not path.exists():continue
        request = identified(path);state = read_json(path.parent/'guard_state.json')
        admission = identified(path.parent/'launch_admission.json')
        if (state['stage']=='completed' and state['worker_returncode']==0
                and state['request_identity']==request['identity'] and request['purpose']=='delivery_fit'
                and request['delivery_reserve_seconds']==guard.CPU_DELIVERY_RESERVE_SECONDS
                and request['limit']==manifest['limit'] and admission['mode']=='delivery_fit'
                and admission['manifest_identity']==manifest['identity'] and admission['entry_sha256']==digest(fit.ENTRY)
                and admission['engineering_freeze_sha256']==digest(freeze_path)):
            completed_guard = True
    if not completed_guard:raise ValueError('No completed exact guarded engineering fit exists')
    return {'path':'new_engineering_export_local_acceptance_required','pilot_review':saved,
            'extension_review':frozen['extension_review'],'engineering_freeze_sha256':digest(freeze_path),
            'fit_result_sha256':digest(result_path),'deployment_identity':model['identity'],
            'deployment_manifest_sha256':digest(deployment/'deployment.json'),'fresh_system_seeds':[2026,2027,2028],
            'development_qualification_passed':frozen['development_qualification_passed'],
            'new_model_exported':True,'actual_new_model_CPU_acceptance_passed':False,
            'independent_research_success_claimed':False,'Q2_Q3_scored':False}


def archive_member(name):
    if name=='matrix.json':return True
    return name.startswith('run/') and allowed(name[4:])


def inspect_archive(archive,receipt,manifest_path,manifest,staging=None):
    """Validate all names, declarations and bytes before any canonical restore."""
    archive = Path(archive)
    if (digest(archive)!=receipt['sha256'] or archive.stat().st_size!=receipt['bytes']
            or receipt['manifest_identity']!=manifest['identity'] or receipt['matrix_sha256']!=digest(manifest_path)):
        raise ValueError('Follow-up return archive or matrix binding differs')
    with tarfile.open(archive,'r:') as stream:
        entries = stream.getmembers();names = [entry.name for entry in entries]
        if (len(set(names))!=len(names) or MANIFEST not in names
                or any(not entry.isfile() or not (entry.name==MANIFEST or archive_member(entry.name)) for entry in entries)):
            raise ValueError('Unsafe, excluded or duplicate follow-up return member')
        entry = stream.getmember(MANIFEST)
        if entry.size>16*2**20:raise ValueError('Return manifest is unexpectedly large')
        value = json.load(stream.extractfile(entry))
        if (value['identity']!=identity({k:v for k,v in value.items() if k!='identity'})
                or value['identity']!=receipt['return_manifest_identity']
                or value['manifest_identity']!=manifest['identity'] or value['matrix_sha256']!=digest(manifest_path)
                or value['source_identity']!=manifest['source_identity'] or value['limit']!=manifest['limit']
                or value['credentials_included'] is not False or value['raw_market_data_included'] is not False
                or value['Q2_Q3_prediction_arrays_included'] is not False
                or value['representation_caches_included'] is not False
                or value['training_guards_observed_exited_on_source_host'] is not True
                or value['CPU_acceptance_proven_by_this_archive'] is not False
                or set(names)!={MANIFEST,*value['files']}):
            raise ValueError('Follow-up return manifest declarations or exact member set differ')
        matrix = value['files'].get('matrix.json')
        if matrix!={'sha256':digest(manifest_path),'bytes':Path(manifest_path).stat().st_size}:
            raise ValueError('Archive carries another frozen matrix')
        for name,row in value['files'].items():
            member = stream.getmember(name)
            if member.size!=row['bytes']:raise ValueError('Follow-up return member size differs: '+name)
            with stream.extractfile(member) as source:
                if staging is None:
                    sha = hashlib.file_digest(source,'sha256').hexdigest()
                else:
                    target = Path(staging)/name;target.parent.mkdir(parents=True,exist_ok=True)
                    with target.open('xb') as output:shutil.copyfileobj(source,output)
                    sha = digest(target)
            if sha!=row['sha256']:raise ValueError('Follow-up return member hash differs: '+name)
    return value


def build(manifest_path,manifest=None):
    """Low-level serializer; the CLI supplies the independent guard and locks."""
    manifest_path = Path(manifest_path);manifest = manifest or check_frozen(manifest_path)
    outcome = completion(manifest_path,manifest);run = FOLLOW/'runs'/manifest['identity']
    directory = FOLLOW/'returns'/manifest['identity'];directory.mkdir(parents=True,exist_ok=True)
    archive,receipt_path = directory/'bundle.tar',directory/'receipt.json'
    if receipt_path.exists():
        receipt = identified(receipt_path);saved = inspect_archive(archive,receipt,manifest_path,manifest)
        if saved['outcome']!=outcome:raise ValueError('Existing return outcome differs; preserve it')
        return archive,receipt
    if archive.exists():archive.rename(directory/('uncommitted_bundle_'+str(time.time_ns())+'.tar'))
    files = {'matrix.json':{'sha256':digest(manifest_path),'bytes':manifest_path.stat().st_size}}
    sources = {'matrix.json':manifest_path}
    for path in sorted(run.rglob('*')):
        name = str(path.relative_to(run))
        if not allowed(name) or not path.is_file():continue
        if not path.resolve().is_relative_to(run.resolve()):raise ValueError('Follow-up artifact link leaves its run')
        sources['run/'+name] = path;files['run/'+name] = {'sha256':digest(path),'bytes':path.stat().st_size}
    if shutil.disk_usage(directory).free<sum(row['bytes'] for row in files.values())+2**30:
        raise RuntimeError('Insufficient space for the immutable follow-up return')
    value = {'schema':1,'artifact_type':'followup_formal_return','manifest_identity':manifest['identity'],
             'matrix_sha256':digest(manifest_path),'source_identity':manifest['source_identity'],'limit':manifest['limit'],
             'outcome':outcome,'files':files,'credentials_included':False,'raw_market_data_included':False,
             'Q2_Q3_prediction_arrays_included':False,'representation_caches_included':False,
             'CPU_acceptance_proven_by_this_archive':False,
             'training_guards_observed_exited_on_source_host':True,
             'source_host_boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
             'created_at_utc':utc_now()}
    value['identity'] = identity(value)
    record = directory/'return_manifest.json';write_json(record,value)
    partial = directory/('bundle_'+str(time.time_ns())+'.partial')
    with tarfile.open(partial,'x',dereference=True) as stream:
        stream.add(record,arcname=MANIFEST,recursive=False)
        for name,path in sources.items():stream.add(path,arcname=name,recursive=False)
    receipt = {'manifest_identity':manifest['identity'],'matrix_sha256':digest(manifest_path),
               'return_manifest_identity':value['identity'],'archive':str(archive.relative_to(ROOT)),
               'sha256':digest(partial),'bytes':partial.stat().st_size,'created_at_utc':utc_now(),
               'actual_new_model_CPU_acceptance_passed':False,'independent_research_success_claimed':False}
    receipt['identity'] = identity(receipt)
    inspect_archive(partial,receipt,manifest_path,manifest)
    os.link(partial,archive);partial.unlink();write_json(receipt_path,receipt)
    return archive,receipt


def restore(archive,receipt_path,manifest_path,manifest=None):
    """Never replace a differing file, including partially restored evidence."""
    manifest = manifest or check_frozen(manifest_path);receipt = identified(receipt_path)
    directory = FOLLOW/'returns'/manifest['identity'];directory.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(directory).free<2*receipt['bytes']+2**30:
        raise RuntimeError('Insufficient local disk for follow-up staging and restore')
    with tempfile.TemporaryDirectory(prefix='staging_',dir=directory) as temporary:
        staging = Path(temporary);value = inspect_archive(archive,receipt,manifest_path,manifest,staging)
        run = FOLLOW/'runs'/manifest['identity'];pairs = []
        for name,row in value['files'].items():
            if name=='matrix.json':continue
            target = run/name[4:]
            if not target.resolve().is_relative_to(run.resolve()):raise ValueError('Restore target leaves its isolated run')
            if target.exists() and (not target.is_file() or digest(target)!=row['sha256'] or target.stat().st_size!=row['bytes']):
                raise ValueError('Preserve differing existing follow-up evidence: '+name)
            pairs.append((staging/name,target,row))
        for source,target,row in pairs:
            if target.exists():continue
            target.parent.mkdir(parents=True,exist_ok=True)
            # A link commits the already checked staging bytes without a partial
            # canonical file or an overwrite window; all paths share a filesystem.
            os.link(source,target)
        actual = completion(manifest_path,manifest)
        if actual!=value['outcome']:raise ValueError('Restored predictions/models differ from the returned outcome')
    result = {'manifest_identity':manifest['identity'],'matrix_sha256':digest(manifest_path),
              'return_manifest_identity':value['identity'],'archive_sha256':receipt['sha256'],
              'all_file_hashes_verified':True,'restored_file_count':len(pairs),'outcome':actual,
              'actual_new_model_CPU_acceptance_passed':False,'independent_research_success_claimed':False,
              'restored_at_utc':utc_now()}
    result['identity'] = identity(result)
    write_json(directory/'restored_receipt.json',result)
    return result


def fetch(manifest_path,manifest):
    """Called inside a guarded local worker; SSH/SCP share its process group."""
    from scripts.finish_fusion_100h import SSH_OPTIONS,HOST,REMOTE_ROOT
    directory = FOLLOW/'returns'/manifest['identity'];directory.mkdir(parents=True,exist_ok=True)
    relative = str((directory/'bundle.tar').relative_to(ROOT))
    remote_receipt = REMOTE_ROOT+'/'+str((directory/'receipt.json').relative_to(ROOT))
    command = [REMOTE_ROOT+'/.venv/bin/python','-c',
               'import pathlib,sys;print(pathlib.Path(sys.argv[1]).read_text())',remote_receipt]
    result = subprocess.run(['ssh','-p','19046',*SSH_OPTIONS,HOST,shlex.join(command)],
                            capture_output=True,text=True,timeout=45,check=True)
    receipt = json.loads(result.stdout)
    if (receipt['identity']!=identity({k:v for k,v in receipt.items() if k!='identity'})
            or receipt['manifest_identity']!=manifest['identity'] or receipt['matrix_sha256']!=digest(manifest_path)
            or receipt['archive']!=relative or receipt['actual_new_model_CPU_acceptance_passed'] is not False
            or receipt['independent_research_success_claimed'] is not False
            or type(receipt['bytes']) is not int or receipt['bytes']<=0):
        raise ValueError('Source return receipt or fixed archive path differs')
    receipt_path = directory/'source_receipt.json'
    if receipt_path.exists() and read_json(receipt_path)!=receipt:
        raise ValueError('Preserve the previous source return receipt; its binding differs')
    write_json(receipt_path,receipt)
    archive = directory/'downloaded_bundle.tar'
    if not archive.exists() or digest(archive)!=receipt['sha256'] or archive.stat().st_size!=receipt['bytes']:
        if shutil.disk_usage(directory).free<3*receipt['bytes']+2**30:
            raise RuntimeError('Insufficient disk for the downloaded follow-up return and restore')
        left = (datetime.fromisoformat(DEADLINE)-datetime.now(timezone.utc)).total_seconds()
        if left<=65:raise TimeoutError('Original cutoff leaves no download and stop grace')
        partial = directory/('download_'+str(time.time_ns())+'.partial')
        subprocess.run(['scp','-P','19046',*SSH_OPTIONS,HOST+':'+REMOTE_ROOT+'/'+relative,str(partial)],
                       capture_output=True,text=True,timeout=min(3600,left-60),check=True)
        if digest(partial)!=receipt['sha256'] or partial.stat().st_size!=receipt['bytes']:
            raise ValueError('Downloaded return differs; preserve the partial file')
        if archive.exists():archive.rename(directory/('retained_download_'+str(time.time_ns())+'.tar'))
        os.link(partial,archive);partial.unlink()
    return restore(archive,receipt_path,manifest_path,manifest)


def worker(args):
    request = bind_guarded_worker(args.attempt,ENTRY,'delivery_cpu')
    manifest = check_frozen(args.manifest)
    admission = identified(Path(args.attempt)/'launch_admission.json')
    if (request['limit']!=manifest['limit'] or request['stage_seconds']>CPU_SECONDS
            or admission['manifest_identity']!=manifest['identity'] or admission['entry_sha256']!=digest(ENTRY)
            or admission['mode']!=args.mode or admission['matrix_sha256']!=digest(args.manifest)
            or request['stage_seconds']!=admission['stage_seconds']):
        raise ValueError('Return worker differs from its exact guarded admission')
    if args.mode=='bundle':
        original_idle(manifest)
        return build(args.manifest,manifest)
    if os.environ.get('FUSION_EXECUTION_PROFILE','local')!='local':raise RuntimeError('Restore on the local research host')
    current = verify_delivery(BASE)
    if current['files']!=manifest['delivery']['files']:raise ValueError('Original local delivery changed before follow-up return')
    if args.mode=='fetch':return fetch(args.manifest,manifest)
    if (admission['archive_sha256']!=digest(args.archive) or admission['receipt_sha256']!=digest(args.receipt)):
        raise ValueError('Return worker archive or receipt changed before restore')
    return restore(args.archive,args.receipt,args.manifest,manifest)


def launch(args):
    manifest_path = Path(args.manifest).resolve();manifest = check_frozen(manifest_path)
    with execution_locks():
        if args.mode=='bundle':original_idle(manifest)
        elif os.environ.get('FUSION_EXECUTION_PROFILE','local')!='local':raise RuntimeError('Restore only on the local host')
        left = (datetime.fromisoformat(DEADLINE)-datetime.now(timezone.utc)).total_seconds()
        if left<=65:raise TimeoutError('Original deadline leaves no guarded CPU return time')
        cap = min(CPU_SECONDS,left)
        run = FOLLOW/'runs'/manifest['identity'];prefix = 'return_'+args.mode+'_'
        previous = sorted((run/'attempts').glob(prefix+'*'))
        for prior in previous:
            if (prior/'guard_request.json').exists():spent_seconds(prior/'guard_request.json',CPU_SECONDS)
        attempt = run/'attempts'/(prefix+str(len(previous)+1).zfill(2))
        if attempt.exists():raise RuntimeError('Preserve the previous partial CPU return launch')
        admission = {'manifest_identity':manifest['identity'],'matrix_sha256':digest(manifest_path),
                     'entry_sha256':digest(ENTRY),'mode':args.mode,'stage_seconds':cap,'issued_at_utc':utc_now()}
        command = [sys.executable,str(ENTRY),'--worker','--manifest',str(manifest_path),
                   '--mode',args.mode,'--attempt',str(attempt)]
        if args.mode=='restore':
            admission.update(archive_sha256=digest(args.archive),receipt_sha256=digest(args.receipt))
            command += ['--archive',str(Path(args.archive).resolve()),'--receipt',str(Path(args.receipt).resolve())]
        admission['identity'] = identity(admission);write_json(attempt/'launch_admission.json',admission)
        os.environ['PYTHONPATH'] = str(ROOT)
        process,dispatch = guard.start(attempt,command,manifest['limit'],cap,purpose='delivery_cpu')
        try:code = process.wait(timeout=max(1,dispatch['stop_by_unix']-time.time()+10))
        finally:stop_guard(process,attempt)
        state = read_json(attempt/'guard_state.json')
        if code!=0 or state['stage']!='completed':raise RuntimeError('CPU return failed/stopped; preserve and inspect its evidence')
        print({'mode':args.mode,'complete':True,'actual_new_model_CPU_acceptance_passed':False})


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--mode',choices=('bundle','restore','fetch'),required=True)
    parser.add_argument('--archive',type=Path);parser.add_argument('--receipt',type=Path)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--attempt',type=Path,help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.mode=='restore' and (not args.archive or not args.receipt):parser.error('Restore needs its exact archive and receipt')
    if args.mode!='restore' and (args.archive or args.receipt):parser.error('Bundle/fetch paths are fixed by the matrix')
    if args.worker:
        if not args.attempt:parser.error('Internal worker needs its exact guard attempt')
        worker(args)
    else:
        if args.attempt:parser.error('Attempt paths are fixed by the launcher')
        launch(args)
