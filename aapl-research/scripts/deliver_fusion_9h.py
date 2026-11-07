"""Verified return and actual CPU acceptance for the shortened A-only protocol.

No GPU work, final-period evaluation or changes to the old receipt chain.
The CPU supervisor uses the original absolute deadline and a two-hour stage cap.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import time

import psutil

from fusion.next.common import ROOT, OUT as BASE, digest, identity, read_json, utc_now, write_json
from scripts import followup_guard as guard
from scripts import fusion_9h as focus
from scripts import deliver_fusion_followup as cpu
from scripts.followup_admission import identified, verify_delivery
from scripts.run_fusion_followup import bind_guarded_worker, execution_locks
from scripts.finish_fusion_100h import SSH_OPTIONS, HOST, REMOTE_ROOT

ENTRY=Path(__file__).resolve()


def safe_relative(name):
    path=PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or str(path)!=name or '\\' in name:
        raise ValueError('Unsafe archive path: '+name)
    return path


def permitted(name,manifest):
    safe_relative(name)
    run='outputs/fusion_followup/runs/'+manifest['identity']+'/'
    diagnostic=str(focus.DIAGNOSTIC.relative_to(ROOT))+'/'
    if name.startswith(run):
        part=name[len(run):]
        return (part.startswith(('gpu/trials/','gpu/training/','gpu/usage_events/',
                                'delivery/','delivery_fit/training/','delivery_fit/deployment/'))
                or part in ('review.json','gpu_completion.json','delivery_fit/fit_result.json',
                            'attempts/gpu_01/guard_request.json','attempts/gpu_01/guard_state.json'))
    return name==str(manifest_path(manifest).relative_to(ROOT)) or (name.startswith(diagnostic)
        and name[len(diagnostic):] in ('diagnostic_result.json','guard_request.json','guard_state.json'))


def manifest_path(manifest):
    return focus.FOLLOW/'preparation'/('nine_hour_'+manifest['identity'][:16]+'.json')


def inspect(archive,receipt,manifest):
    if (receipt['manifest_identity']!=manifest['identity'] or digest(archive)!=receipt['sha256']
            or archive.stat().st_size!=receipt['bytes'] or receipt['Q2_Q3_scored'] is not False):
        raise ValueError('Formal return identity/bytes differ')
    names=set()
    with tarfile.open(archive,'r') as tar:
        for member in tar:
            if (member.name in names or not member.isfile() or not permitted(member.name,manifest)
                    or member.name not in receipt['members']):
                raise ValueError('Unexpected/duplicate/non-regular return member: '+member.name)
            names.add(member.name);expected=receipt['members'][member.name]
            if member.size!=expected['bytes']:raise ValueError('Return member size differs')
            stream=tar.extractfile(member);sha=hashlib.sha256()
            for block in iter(lambda:stream.read(1024*1024),b''):sha.update(block)
            if sha.hexdigest()!=expected['sha256']:raise ValueError('Return member SHA differs')
            target=ROOT/member.name
            # Check every conflict before writing any restored model/research file.
            if target.is_symlink() or (target.exists() and (not target.is_file() or digest(target)!=expected['sha256'])):
                raise ValueError('Preserve different local return member: '+member.name)
            for ancestor in target.parents:
                if ancestor==ROOT:break
                if ancestor.is_symlink():raise ValueError('Return destination crosses a symlink')
    if names!=set(receipt['members']):raise ValueError('Return member inventory is incomplete')


def fetch(path,manifest):
    directory=focus.FOLLOW/'returns'/manifest['identity'];directory.mkdir(parents=True,exist_ok=True)
    receipt_path=directory/'receipt.json';archive=directory/'bundle.tar'
    remote_receipt='outputs/fusion_followup/runs/'+manifest['identity']+'/return/receipt.json'
    # The source supervisor and complete worker group must actually be gone.
    program="""import json,pathlib,psutil
from scripts import fusion_9h_guard as guard
root=pathlib.Path(%r);run=root/'outputs/fusion_followup/runs'/%r
state=json.loads((run/'attempts/gpu_01/guard_state.json').read_text())
dispatch=json.loads((run/'scheduler_dispatch.json').read_text())
assert state['stage']=='completed' and state['worker_returncode']==0
assert not guard.exact_alive(state['guard']) and not guard.group_members(state['worker'])
assert not guard.exact_alive(dispatch['scheduler']), 'Return scheduler still active'
print((root/%r).read_text())
"""%(REMOTE_ROOT,manifest['identity'],remote_receipt)
    result=subprocess.run(['ssh','-p','19046',*SSH_OPTIONS,HOST,REMOTE_ROOT+'/.venv/bin/python -'],
        input=program,text=True,capture_output=True,timeout=30,check=True)
    receipt=json.loads(result.stdout)
    if receipt['identity']!=identity({k:v for k,v in receipt.items() if k!='identity'}):raise ValueError('Remote receipt identity differs')
    expected_archive='outputs/fusion_followup/runs/'+manifest['identity']+'/return/bundle.tar'
    if receipt['archive']!=expected_archive:raise ValueError('Remote archive leaves its fixed path')
    focus.immutable(receipt_path,receipt)
    if not archive.exists():
        partial=directory/'bundle.partial.tar'
        if partial.exists():
            if partial.stat().st_size==receipt['bytes'] and digest(partial)==receipt['sha256']:partial.rename(archive)
            else:partial.rename(directory/('interrupted_'+str(time.time_ns())+'.tar'))
        if not archive.exists():
            subprocess.run(['scp','-P','19046',*SSH_OPTIONS,HOST+':'+REMOTE_ROOT+'/'+expected_archive,str(partial)],
                timeout=900,check=True)
            if partial.stat().st_size!=receipt['bytes'] or digest(partial)!=receipt['sha256']:
                raise ValueError('Downloaded archive differs; partial preserved')
            partial.rename(archive)
    inspect(archive,receipt,manifest)
    with tarfile.open(archive,'r') as tar:
        for member in tar:
            target=ROOT/member.name
            if target.exists():continue
            target.parent.mkdir(parents=True,exist_ok=True)
            partial=target.with_name(target.name+'.restore_partial')
            if partial.exists():partial.rename(partial.with_name(partial.name+'.'+str(time.time_ns())))
            with partial.open('xb') as output,tar.extractfile(member) as source:shutil.copyfileobj(source,output)
            if digest(partial)!=receipt['members'][member.name]['sha256']:raise ValueError('Restored file differs')
            partial.rename(target)
    restored={'manifest_identity':manifest['identity'],'receipt_identity':receipt['identity'],
        'archive_sha256':receipt['sha256'],'all_file_hashes_verified':True,'Q2_Q3_scored':False}
    restored['identity']=identity(restored);focus.immutable(directory/'restored_receipt.json',restored)
    return receipt,restored


def binding(path,manifest,parent):
    original=verify_delivery(BASE)
    if original['files']!=parent['delivery']['files']:raise ValueError('Original actual delivery changed')
    run=focus.FOLLOW/'runs'/manifest['identity']
    complete=identified(run/'gpu_completion.json');fit=identified(run/'delivery_fit/fit_result.json')
    from fusion.next.panel import Panel
    from scripts.fit_fusion_followup import verify_export,checkpoint_path,system_config
    frozen=identified(run/'delivery/engineering_freeze.json')
    systems=[{'config':system_config(frozen,s),'checkpoint':str(checkpoint_path(run/'delivery_fit',frozen,s))} for s in focus.SEEDS]
    exported=verify_export(run/'delivery_fit/deployment/pre_Q2',frozen,digest(run/'delivery/engineering_freeze.json'),systems)
    if (fit['model_identity']!=exported['identity'] or complete['model_identity']!=exported['identity']
            or complete['completed_validation_fits']!=9 or complete['completed_engineering_seeds']!=list(focus.SEEDS)
            or fit['Q2_Q3_scored'] is not False or fit['actual_CPU_acceptance_passed'] is not False):
        raise ValueError('Returned model chain is incomplete')
    review=focus.review(manifest,parent,Panel.load())
    if frozen['development_review_identity']!=review['identity']:raise ValueError('Local development recomputation differs')
    restored=identified(focus.FOLLOW/'returns'/manifest['identity']/'restored_receipt.json')
    body={'manifest_identity':manifest['identity'],'matrix_sha256':digest(path),
        'return_binding':restored,'entry_sha256':digest(ENTRY),
        'outcome':{'new_model_exported':True,'deployment_identity':exported['identity'],
            'development_qualification_passed':review['development_qualification_passed']},
        'source_identity':parent['source_identity'],'plan_sha256':parent['plan_sha256'],'limit':parent['limit'],
        'cold_timing':{'clock':'CLOCK_BOOTTIME minus kernel process birth ticks',
            'helper_sha256':digest(ROOT/'scripts/original_delivery_timing_runtime.py'),
            'full_process_to_first_durable_output':True,'ceiling_seconds':60},
        'independent_research_success_claimed':False,'Q2_Q3_scored':False}
    body['identity']=identity(body)
    return body


@contextmanager
def measured_runtime(directory,frozen,body):
    """Reuse all CPU checks with the verified wall-clock-independent birth clock."""
    from scripts.original_delivery_timing_runtime import process_birth,elapsed_from_birth
    with ORIGINAL_RUNTIME(directory,frozen,body) as runtime:
        from fusion.next import common
        birth=process_birth();first=[]
        stream_class=runtime.delivery.Stream;publish=runtime.delivery.write_json
        class Measured(stream_class):
            def predict(self,*args,**kwargs):
                result=super().predict(*args,**kwargs)
                if result['status']=='ok' and not first:first.append(elapsed_from_birth(birth))
                return result
        def write(path,value):
            # Existing publication installs FP32/binding checks. Correct the
            # persisted process interval after it, using only kernel birth time.
            result=publish(path,value)
            if Path(path).name=='progress.json' and 'processes' in value:
                for row in value['processes']:
                    if row['pid']==birth['pid']:
                        row['kernel_process_birth']=birth
                        if first:row.update(full_process_to_first_output_seconds=first[0][0],
                            first_durable_output_boottime_seconds=first[0][1])
                common.write_json(path,value)
            elif Path(path).name=='report.json' and 'cold_load_measurements_by_process' in value:
                rows=value['cold_load_measurements_by_process']
                for row in rows:
                    b=row['kernel_process_birth'];expected=row['first_durable_output_boottime_seconds']-b['start_ticks']/b['ticks_per_second']
                    if expected!=row['full_process_to_first_output_seconds']:raise ValueError('Full process timing differs')
                value['cold_scope']=body['cold_timing']['clock']+' through first durable prediction'
                value['cold_timing_complete']=True;value['cold_timing_binding']=body['cold_timing']
                common.write_json(path,value)
            return result
        runtime.delivery.Stream=Measured;runtime.delivery.write_json=write
        base_files=runtime.package.source_files
        def files():
            return {**base_files(),'evidence/FUSION_9H_AMENDMENT.md':focus.PLAN,
                'evidence/nine_hour_manifest.json':manifest_path({'identity':body['manifest_identity']}),
                'evidence/nine_hour_CPU_entry.py':ENTRY}
        runtime.package.source_files=files
        yield runtime


ORIGINAL_RUNTIME=cpu.local_runtime


def child(request_path):
    row=identified(request_path);attempt=Path(row['attempt'])
    request=identified(attempt/'guard_request.json');state=read_json(attempt/'guard_state.json')
    if (row['entry_sha256']!=digest(ENTRY) or state['stage']!='running'
            or state['worker']!=row['parent'] or state['request_identity']!=request['identity']
            or request['purpose']!='delivery_cpu' or request['guard_sha256']!=digest(ROOT/'scripts/followup_guard.py')
            or row['guard_identity']!=request['identity'] or os.getppid()!=row['parent']['pid']
            or os.getsid(0)!=row['parent']['pid'] or os.getpgid(0)!=row['parent']['pid']
            or not guard.exact_alive(row['parent']) or not guard.exact_alive(state['guard'])
            or not guard.exact_alive(request['owner']) or time.time()>=request['stop_by_unix']
            or row['command']!=[sys.executable,str(ENTRY),'--child',str(request_path)]
            or psutil.Process(row['parent']['pid']).cmdline()!=request['command']):
        raise ValueError('CPU child is not bound to its live exact supervisor')
    body=identified(request_path.parent.parent/'binding.json')
    if body['identity']!=row['binding_identity'] or body['entry_sha256']!=digest(ENTRY):raise ValueError('CPU binding changed')
    manifest=identified(Path(row['manifest']))
    if digest(Path(row['manifest']))!=body['matrix_sha256']:raise ValueError('CPU manifest changed')
    directory,frozen,deployment=cpu.prepare_local(manifest,body)
    if row['phase'] not in ('replay','package') or row['stop_after'] not in (None,128):raise ValueError('Unknown CPU phase')
    cpu.local_runtime=measured_runtime
    return cpu.run_phase(row['phase'],directory,frozen,deployment,body,row['resume'],row['stop_after'])


def invoke(path,attempt,directory,body,phase,resume=False,stop_after=None):
    request=identified(attempt/'guard_request.json')
    row_path=directory/'process_requests'/(str(time.time_ns())+'.json')
    command=[sys.executable,str(ENTRY),'--child',str(row_path)]
    row={'entry_sha256':digest(ENTRY),'attempt':str(attempt),'guard_identity':request['identity'],
        'manifest':str(path),'parent':guard.process_identity(),'binding_identity':body['identity'],
        'phase':phase,'resume':resume,'stop_after':stop_after,'command':command}
    row['identity']=identity(row);write_json(row_path,row)
    log=directory/'process_logs'/(row_path.stem+'.log');log.parent.mkdir(exist_ok=True)
    with log.open('xb') as stream:
        process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','FUSION_EXECUTION_PROFILE':'local'})
        dispatch={'process':guard.process_identity(process.pid),'request_identity':row['identity']}
        write_json(row_path.with_suffix('.dispatch.json'),dispatch)
        try:code=process.wait(timeout=max(1,request['stop_by_unix']-time.time()))
        except BaseException:process.kill();process.wait();raise
    write_json(row_path.with_suffix('.exit.json'),{**dispatch,'returncode':code,'exited_at_utc':utc_now()})
    if code:raise RuntimeError('CPU phase failed; retained log '+str(log))


def worker(path,attempt):
    request=bind_guarded_worker(attempt,ENTRY,'delivery_cpu')
    if os.environ.get('FUSION_EXECUTION_PROFILE','local')!='local':raise RuntimeError('CPU acceptance is local only')
    manifest,parent,_=focus.checked(path)
    if request['limit']!=parent['limit'] or request['stage_seconds']>7200:raise ValueError('CPU time limit differs')
    fetch(path,manifest);body=binding(path,manifest,parent);body['identity']=identity({k:v for k,v in body.items() if k!='identity'})
    directory,frozen,deployment=cpu.prepare_local(manifest,body)
    root=directory/'deployment_acceptance'/body['outcome']['deployment_identity']
    if not (root/'report.json').exists():
        if not (root/'state.json').exists():invoke(path,attempt,directory,body,'replay',stop_after=128)
        invoke(path,attempt,directory,body,'replay',resume=True)
    report=cpu.checked_replay(directory,body)
    invoke(path,attempt,directory,body,'package')
    report=cpu.checked_replay(directory,body)
    archive=directory/'local_packages'/(report['model_identity']+'.tar')
    verification=read_json(archive.with_suffix('.verification.json'))
    if (verification['archive_sha256']!=digest(archive) or not verification['isolated_example_and_fresh_process_resume_passed']
            or len(verification['actual_processes'])!=3 or not report.get('cold_timing_complete')):
        raise ValueError('Actual standalone package or full-process timing is incomplete')
    result={'stage':'nine_hour_CPU_delivery_complete','manifest_identity':manifest['identity'],
        'model_identity':report['model_identity'],'binding_identity':body['identity'],
        'actual_new_model_CPU_acceptance_passed':report['engineering_acceptance_passed'],
        'development_qualification_passed':body['outcome']['development_qualification_passed'],
        'package':str(archive.relative_to(ROOT)),'package_sha256':digest(archive),
        'CPU_report_sha256':digest(root/'report.json'),'verification_sha256':digest(archive.with_suffix('.verification.json')),
        'cold_seconds':report['cold_load_seconds'],'warm_p95_seconds':report['warm_single_origin_p95_seconds'],
        'standalone_package_actually_executed':True,'independent_success_claimed':False,
        'Q2_Q3_scored':False,'completed_at_utc':utc_now()}
    result['identity']=identity(result);focus.immutable(directory/'completion.json',result)
    return result


def launch(path):
    manifest,parent,_=focus.checked(path);run=focus.FOLLOW/'runs'/manifest['identity']
    attempt=run/'attempts/local_cpu_01'
    if attempt.exists():raise RuntimeError('CPU attempt exists; inspect real process and saved progress before recovery')
    with execution_locks():
        attempt.mkdir(parents=True)
        process,dispatch=guard.start(attempt,[sys.executable,str(ENTRY),'--mode','worker','--manifest',str(path),
            '--attempt',str(attempt)],parent['limit'],7200,purpose='delivery_cpu')
        write_json(attempt/'scheduler_dispatch.json',{'scheduler':guard.process_identity(),'guard':dispatch})
        code=process.wait()
        if code:raise RuntimeError('CPU delivery failed; preserve real evidence')
        print(read_json(run/'local_delivery/completion.json'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('deliver','worker'))
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--attempt',type=Path)
    parser.add_argument('--child',type=Path)
    args=parser.parse_args()
    if args.child:child(args.child.resolve())
    elif args.mode=='worker':worker(args.manifest.resolve(),args.attempt.resolve())
    else:launch(args.manifest.resolve())
