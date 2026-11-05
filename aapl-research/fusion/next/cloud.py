"""Explicit allowlist for an AutoDL training bundle; no keys or sealed data."""
import argparse
import importlib.metadata as metadata
import os
from pathlib import Path
import tarfile
import time

from .common import ROOT,OUT,PLAN,digest,read_json,write_json,utc_now,source_files
from .foundation import manifest as weight_manifest


def write_profile():
    # Hardware and quoted price are from the user-provided screenshot, not a live instance.
    report={'recorded_at_utc':utc_now(),'source':'user specified AutoDL screenshot',
            'training_location':'AutoDL','gpu_model_requested':'RTX 5090','gpus':1,'vram_gib_quoted':32,
            'cpu_cores_quoted':25,'ram_gib_quoted':120,'data_disk_gib_quoted':50,
            'price_cny_per_hour_quoted':2.88,'24h_cost_cny_at_quoted_rate':69.12,
            'live_hardware_verified':False,'connection_status':'connection supplied; user advised to shut down while local preparation continues',
            'ssh':{'host':'connect.bjb1.seetacloud.com','port':23367,'user':'root',
                   'verify_connection_after_instance_restart':True},
            'remote_training_started':False,'remote_start_gate':'F1 passed and local training bundle validated; confirm instance is running',
            'paid_rental_performed_by_agent':False,'local_work':['F0 audit','F1 acquisition and all-day data audit','code and training bundle'],
            'remote_work':['F2 15-minute maximum-shape admission','F3-F5 training and mechanisms'],
            'local_final_evaluation':True,'Q3_data_upload_allowed':False,'credentials_upload_allowed':False,
            'plan_sha256':digest(PLAN)}
    write_json(OUT/'execution_profile.json',report)
    return report


def bundle(include_data=False):
    # Name patterns are never used as the sole credential exclusion mechanism.
    # Only individually enumerated code/model files and F1-admitted array shards enter.
    weight=weight_manifest();items=source_files()
    for folder in ('fusion/next','aapl/vendor'):
        for p in sorted((ROOT/folder).glob('*')):
            if p.suffix in ('.py','.txt','.json','.md'):items[str(p.relative_to(ROOT))]=p
    for name in ('fusion/__init__.py','aapl/__init__.py','aapl/common.py','aapl/data.py','aapl/models.py',
                 'FUSION_IMPLEMENTATION_PLAN.md'):
        p=ROOT/name;items[name]=p
    items['outputs/fusion_next/foundation_weights.json']=OUT/'foundation_weights.json'
    for name in ('previous_selected_hyperparameters.json','search_design_draft.json'):
        if (OUT/name).exists():items['outputs/fusion_next/'+name]=OUT/name
    old=Path('/home/ayaka/xPatch');row=read_json(old/'configs/aapl_v2_model_lock.json')['models']['chronos2']
    for name,info in weight['files'].items():
        p=old/row['directory']/name
        if digest(p)!=info['sha256']:raise ValueError('Weight package identity changed')
        items['weights/chronos2/'+name]=p
    if include_data:
        admission=read_json(OUT/'data_admission.json')
        audit=read_json(OUT/'data_audit.json')
        if not admission['passed'] or not audit['passed']:
            raise ValueError('Full F1 admission and independent array audit must pass before data packaging')
        for name in ('data_admission.json','data_audit.json','time_splits.json','data_policy.json','prepared_symbols.json'):
            items['outputs/fusion_next/'+name]=OUT/name
        for name in ('timestamps_ns.npy','dates.npy','session_start.npy','session_end.npy','common_aapl_origins.npy'):
            p=OUT/'data'/name
            if digest(p)!=audit['global_files_sha256'][name]:raise ValueError('Audited calendar/origin file changed')
            items[str(p.relative_to(ROOT))]=p
        names=list(dict.fromkeys(admission['admitted_training_stocks']+admission['background']+admission['market_peers']))
        fields={'features.npy','observed.npy','close.npy','origins.npy','target.npy'}
        for symbol in names:
            directory=OUT/'data'/symbol;receipt=directory/'manifest.json';row=read_json(receipt)
            if digest(receipt)!=audit['symbols'][symbol]['manifest_sha256'] or set(row['files'])!=fields:
                raise ValueError('Audited stock manifest changed or contains unexpected fields')
            items[str(receipt.relative_to(ROOT))]=receipt
            for name,sha in row['files'].items():
                p=directory/name
                if digest(p)!=sha:raise ValueError('Audited stock array changed')
                items[str(p.relative_to(ROOT))]=p
    export=OUT/'cloud';export.mkdir(parents=True,exist_ok=True)
    for name in ('bootstrap.sh','run.sh','requirements.txt'):
        items['cloud/'+name]=ROOT/'fusion/next/cloud_assets'/name
        items['fusion/next/cloud_assets/'+name]=ROOT/'fusion/next/cloud_assets'/name
    receipt={'created_at_utc':utc_now(),'plan_sha256':digest(PLAN),'contains_development_data':include_data,
             'credentials_included':False,'sealed_data_included':False,
             'files':{name:{'sha256':digest(path),'bytes':path.stat().st_size} for name,path in items.items()}}
    manifest_path=export/'bundle_manifest.json';write_json(manifest_path,receipt)
    path=export/('training_bundle.tar' if include_data else 'code_weights_bundle.tar')
    with tarfile.open(path,'w',dereference=True) as tar:
        for name,p in items.items():tar.add(p,arcname='aapl-fusion/'+name,recursive=False)
        tar.add(manifest_path,arcname='aapl-fusion/bundle_manifest.json',recursive=False)
    with tarfile.open(path) as tar:
        names=tar.getnames()
        if any('holdout_sealed' in n or '/.env' in n or 'credentials' in n for n in names):
            raise RuntimeError('Prohibited file in allowlisted cloud bundle')
    write_json(export/'bundle_receipt.json',{'archive':str(path.relative_to(ROOT)),'sha256':digest(path),
               'bytes':path.stat().st_size,'files':len(items)+1,'contains_development_data':include_data,
               'credential_and_holdout_exclusion_verified':True,'created_at_utc':utc_now()})
    return path


def verify_bundle(path):
    """Check the concrete archive in isolation, including portable imports but no model load."""
    import json
    import subprocess
    import sys
    import tempfile
    path=Path(path).resolve()
    with tempfile.TemporaryDirectory(prefix='bundle_verify_',dir=ROOT/'tmp') as temporary:
        directory=Path(temporary)
        with tarfile.open(path) as archive:
            for member in archive.getmembers():
                if not member.isfile():raise ValueError('Only regular files are allowed in the training archive')
                target=(directory/member.name).resolve()
                if not target.is_relative_to(directory) or not member.name.startswith('aapl-fusion/'):
                    raise ValueError('Archive path leaves its isolated root')
            archive.extractall(directory,filter='data')
        isolated=directory/'aapl-fusion';receipt=read_json(isolated/'bundle_manifest.json')
        for name,row in receipt['files'].items():
            if digest(isolated/name)!=row['sha256']:raise ValueError('Extracted file differs: '+name)
        program="""import importlib,json,pathlib
root=pathlib.Path.cwd()
names=['fusion.next.'+p.stem for p in (root/'fusion/next').glob('*.py') if p.stem!='__init__']
for name in names:
 module=importlib.import_module(name)
 assert pathlib.Path(module.__file__).is_relative_to(root)
from fusion.next.foundation import manifest
from fusion.next.common import source_identity
receipt=manifest()
assert receipt['revision']
print(json.dumps({'isolated_modules_imported':len(names),'foundation_manifest_portable':True,'source_identity':source_identity()}))
"""
        env=dict(os.environ,PYTHONPATH=str(isolated),PYTHONDONTWRITEBYTECODE='1')
        result=subprocess.run([sys.executable,'-c',program],cwd=isolated,env=env,capture_output=True,text=True,timeout=60)
        if result.returncode:raise RuntimeError('Isolated bundle import failed: '+result.stderr[-3000:])
        from .common import source_identity
        imported=json.loads(result.stdout)
        if imported['source_identity']!=source_identity():raise ValueError('Packaged source identity differs from local source')
        verification={'archive_sha256':digest(path),'files_hash_verified':len(receipt['files']),
                      'no_model_instantiated':True,'no_remote_connection':True,
                      'isolated_import':imported,'verified_at_utc':utc_now(),
                      'contains_development_data':receipt['contains_development_data']}
    write_json(OUT/'cloud/bundle_verification.json',verification)
    return verification


def return_bundle():
    """Retain cloud fits and raw development evidence needed for local F7/F8."""
    from .accounting import totals
    from .common import source_identity
    if os.environ.get('FUSION_EXECUTION_PROFILE')!='autodl':raise RuntimeError('Create the result return bundle on the training host')
    if totals()['active_jobs']:raise RuntimeError('Do not snapshot while an accounted training or CPU job is running')
    if not (OUT/'confirmation.json').exists():raise ValueError('The fixed F6 export/confirmation must finish before returning final artifacts')
    fixed=('search_manifest.json','budget_projection.json','performance_admission.json','performance_cpu.json',
           'hardware_cloud.json','environment_setup.json','F3_complete.json','F4_complete.json','F5_complete.json','system_freeze.json',
           'confirmation_started.json','confirmation.json','confirmation_predictions.npz','usage_summary.json')
    items={str((OUT/name).relative_to(ROOT)):OUT/name for name in fixed if (OUT/name).is_file()}
    # Include raw predictions as well as fitted weights: local final entry rechecks
    # control rankings and three-seed results rather than trusting cloud summaries.
    for folder in ('training','trials','selections','seed_verification','oof','frozen_fits',
                   'deployment','usage_events','performance','search_progress','environment_setup_events'):
        for path in (OUT/folder).rglob('*'):
            if path.is_file() and path.suffix in ('.json','.npz','.npy','.pt','.txt'):
                if not path.resolve().is_relative_to(OUT):raise ValueError('Result file leaves its dedicated tree')
                items[str(path.relative_to(ROOT))]=path
    receipt={'created_at_utc':utc_now(),'source_identity':source_identity(),'plan_sha256':digest(PLAN),
             'Q3_data_included':False,'raw_development_market_data_included':False,'credentials_included':False,
             'files':{name:{'sha256':digest(path),'bytes':path.stat().st_size} for name,path in items.items()}}
    directory=OUT/'cloud';directory.mkdir(exist_ok=True)
    manifest=directory/'return_manifest.json';write_json(manifest,receipt)
    archive=directory/'return_bundle.tar'
    with tarfile.open(archive,'w',dereference=True) as stream:
        for name,path in items.items():stream.add(path,arcname=name,recursive=False)
        stream.add(manifest,arcname='outputs/fusion_next/cloud/return_manifest.json',recursive=False)
    summary={'archive':str(archive.relative_to(ROOT)),'sha256':digest(archive),'bytes':archive.stat().st_size,
             'files':len(items),'source_identity':source_identity(),'created_at_utc':utc_now()}
    write_json(directory/'return_receipt.json',summary);return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--include-data',action='store_true');p.add_argument('--profile-only',action='store_true');p.add_argument('--verify',action='store_true')
    p.add_argument('--return-results',action='store_true');a=p.parse_args()
    if a.return_results:print(return_bundle())
    else:write_profile()
    if not a.profile_only and not a.return_results:
        path=bundle(a.include_data)
        print(verify_bundle(path) if a.verify else path)
