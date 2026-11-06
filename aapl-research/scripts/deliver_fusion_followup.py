"""Accept and package the exact returned follow-up model on local CPU.

The original replay, Stream, FP32 adapter and standalone package implementation
are reused in a separate output namespace. This entry never trains or scores
Q2/Q3, and cannot turn development qualification into independent evidence.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import psutil

from fusion.next.common import ROOT,OUT as BASE,PLAN,digest,identity,read_json,source_identity,utc_now,write_json
from scripts import followup_guard as guard
from scripts import return_fusion_followup as returned
from scripts.followup_admission import check_frozen,identified,verify_delivery,NUMERICAL,SPEED
from scripts.followup_runtime import FOLLOW,DEADLINE
from scripts.run_fusion_followup import execution_locks,spent_seconds,bind_guarded_worker,stop_guard


ENTRY = Path(__file__).resolve()
ORIGINS = 256
CPU_SECONDS = 2*3600


def checked_return(manifest_path,manifest=None):
    manifest = manifest or check_frozen(manifest_path)
    original = verify_delivery(BASE)
    if original['files']!=manifest['delivery']['files']:raise ValueError('Original delivery changed before follow-up acceptance')
    directory = FOLLOW/'returns'/manifest['identity']
    restored = identified(directory/'restored_receipt.json')
    receipt_path = directory/('source_receipt.json' if (directory/'source_receipt.json').exists() else 'receipt.json')
    receipt = identified(receipt_path)
    archive = directory/('downloaded_bundle.tar' if (directory/'downloaded_bundle.tar').exists() else 'bundle.tar')
    value = returned.inspect_archive(archive,receipt,manifest_path,manifest)
    outcome = returned.completion(manifest_path,manifest)
    if (restored['manifest_identity']!=manifest['identity'] or restored['matrix_sha256']!=digest(manifest_path)
            or restored['return_manifest_identity']!=value['identity'] or restored['archive_sha256']!=receipt['sha256']
            or restored['all_file_hashes_verified'] is not True or restored['outcome']!=outcome
            or outcome!=value['outcome'] or restored['actual_new_model_CPU_acceptance_passed'] is not False
            or restored['independent_research_success_claimed'] is not False):
        raise ValueError('Acceptance needs the exact verified formal return and recomputed outcome')
    return {'manifest_identity':manifest['identity'],'matrix_sha256':digest(manifest_path),
            'return_archive_sha256':receipt['sha256'],'return_manifest_identity':value['identity'],
            'restore_receipt_sha256':digest(directory/'restored_receipt.json'),'outcome':outcome,
            'source_identity':manifest['source_identity'],'plan_sha256':manifest['plan_sha256'],
            'limit':manifest['limit'],'entry_sha256':digest(ENTRY)}


def prepare_local(manifest,binding):
    run = FOLLOW/'runs'/manifest['identity'];directory = run/'local_delivery'
    directory.mkdir(parents=True,exist_ok=True)
    path = directory/'binding.json'
    if path.exists() and identified(path)!=binding:raise ValueError('Preserve the existing local delivery binding')
    if not path.exists():write_json(path,binding)
    if not binding['outcome']['new_model_exported']:return directory,None,None
    frozen_path = run/'delivery/engineering_freeze.json';frozen = identified(frozen_path)
    # The compatibility filename contains the exact engineering-freeze bytes.
    # No research freeze or Q2 confirmation is invented for the original engine.
    copies = {'system_freeze.json':frozen_path,
              'performance/portable_inference_policy.json':BASE/'performance/portable_inference_policy.json',
              'performance/future_speed_policy.json':BASE/'performance/future_speed_policy.json',
              'performance/portable_package_policy.json':BASE/'performance/portable_package_policy.json'}
    if frozen['config'].get('foundation',True):copies['foundation_weights.json']=BASE/'foundation_weights.json'
    for name,source in copies.items():
        target = directory/name
        if target.exists() and digest(target)!=digest(source):raise ValueError('Local delivery copy differs: '+name)
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    link = directory/'data'
    if link.is_symlink():
        if link.resolve()!=(BASE/'data').resolve():raise ValueError('Local delivery input data link differs')
    elif link.exists():raise ValueError('Do not replace existing local delivery inputs')
    else:link.symlink_to(BASE/'data',target_is_directory=True)
    summary = {'artifact_type':'followup_development_only_no_final_evaluation','manifest_identity':manifest['identity'],
        'engineering_freeze_sha256':digest(frozen_path),'selected_candidate':frozen['selected_candidate'],
        'development_qualification_passed':frozen['development_qualification_passed'],
        'independent_research_success_claimed':False,'Q2_Q3_scored':False,'original_F7_replaced':False}
    if (directory/'confirmation.json').exists() and read_json(directory/'confirmation.json')!=summary:
        raise ValueError('Preserve the existing development-only package summary')
    write_json(directory/'confirmation.json',summary)
    deployment = run/'delivery_fit/deployment/pre_Q2'
    return directory,frozen,deployment


def qualified(report,contracts=True):
    """Engineering constraints only; no predictive gate or research claim."""
    numbers = [report.get(k) for k in ('cold_load_seconds','warm_single_origin_p95_seconds',
                                      'batch_stream_max_prediction_difference','batch_stream_tolerance')]
    return bool(all(v is not None and np.isfinite(v) and v>=0 for v in numbers)
        and report['origins']==ORIGINS and report['timed_origins']==ORIGINS and report['timing_complete']
        and report['cold_load_seconds']<=60 and report['warm_single_origin_p95_seconds']<=1
        and report['resume_without_duplicate_prediction'] and report['distinct_exit_and_resume_processes']
        and report['batch_stream_max_prediction_difference']<=report['batch_stream_tolerance'] and contracts)


class PackageProcesses:
    """Retain identities and failed output for the original package verifier."""
    def __init__(self,directory):self.directory=directory;self.records=[]
    def run(self,command,**kwargs):
        if kwargs.pop('capture_output',False) is not True or kwargs.pop('text',False) is not True:
            raise ValueError('Package verification requires captured text output')
        timeout = kwargs.pop('timeout')
        started = time.time();process = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,**kwargs)
        row = {'process':guard.process_identity(process.pid),'command':command,'launched_at_unix':started}
        try:
            output,error = process.communicate(timeout=timeout)
        except BaseException:
            process.kill();output,error = process.communicate()
            row.update(returncode=process.returncode,stdout=output,stderr=error,interrupted=True)
            write_json(self.directory/(str(time.time_ns())+'.json'),row);raise
        row.update(returncode=process.returncode,stdout=output,stderr=error,completed_at_utc=utc_now())
        self.records.append(row);write_json(self.directory/(str(time.time_ns())+'.json'),row)
        return subprocess.CompletedProcess(command,process.returncode,output,error)


@contextmanager
def local_runtime(directory,frozen,binding):
    """Adapt output locations/config views; preserve all original model math."""
    from fusion.next import common,delivery,stream,hardware,local_package,trainer,inference,experiment,experts,panel
    from scripts import portable_inference_runtime as numerical,future_speed_runtime as speed,portable_package_runtime as packaging
    if numerical._installed or speed._installed or packaging._installed:raise RuntimeError('Use a clean independent local acceptance process')
    directory = Path(directory).resolve()
    if not directory.is_relative_to(FOLLOW/'runs') or directory.name!='local_delivery':
        raise ValueError('Use the isolated follow-up local delivery namespace')
    original_writer,original_reader = common.write_json,delivery.read_json
    saved = []
    def set_value(module,name,value):
        saved.append((module,name,getattr(module,name)));setattr(module,name,value)
    def fenced(path,value):
        if not Path(path).resolve().is_relative_to(directory):raise ValueError('Local acceptance write leaves its isolated namespace')
        return original_writer(path,value)
    process = guard.process_identity();first_output = []
    original_stream = delivery.Stream
    class MeasuredStream(original_stream):
        def predict(self,*args,**kwargs):
            result = super().predict(*args,**kwargs)
            if not first_output:first_output.append(max(0,time.time()-process['created_unix']))
            return result
    def publish(path,value):
        if Path(path).name=='progress.json' and 'processes' in value:
            for row in value['processes']:
                if row['pid']==process['pid']:
                    row['created_unix']=process['created_unix']
                    if first_output:row['full_process_to_first_output_seconds']=first_output[0]
            value['followup_binding_identity']=binding['identity']
        if Path(path).name=='report.json' and 'warm_single_origin_p95_seconds' in value:
            processes = value['cold_load_measurements_by_process']
            if any('created_unix' not in p or 'full_process_to_first_output_seconds' not in p for p in processes):
                raise ValueError('Replay lacks full-process cold-start measurements')
            value['engine_cold_load_seconds']=value['cold_load_seconds']
            value['cold_load_seconds']=max(p['full_process_to_first_output_seconds'] for p in processes)
            value['cold_scope']='actual fresh process creation through its first durable output, including admission and imports'
            value['distinct_exit_and_resume_processes']=len({(p['pid'],p['created_unix']) for p in processes})>=2
            value.update(followup_binding_identity=binding['identity'],evaluation_precision='float32',
                numerical_policy_identity=NUMERICAL,speed_policy_identity=SPEED,
                engineering_acceptance_passed=False,independent_research_success_claimed=False,
                actual_input_contract_checks_pending=True,Q2_Q3_scored=False)
            value['engineering_acceptance_passed']=qualified(value)
        return fenced(path,value)
    def read_view(path):
        value = original_reader(path)
        if Path(path).resolve()==(directory/'system_freeze.json').resolve():
            return {**value,'selected_system':{'config':frozen['config']}}
        return value
    original_temporary = local_package.tempfile
    class Temporary:
        def TemporaryDirectory(self,*args,**kwargs):
            kwargs['dir']=directory/'tmp';Path(kwargs['dir']).mkdir(exist_ok=True)
            return original_temporary.TemporaryDirectory(*args,**kwargs)
    proxy = PackageProcesses(directory/'package_processes')
    try:
        for module in (common,delivery,local_package,hardware):set_value(module,'OUT',directory)
        for module in (common,stream,local_package):set_value(module,'write_json',fenced)
        set_value(delivery,'write_json',publish);set_value(delivery,'read_json',read_view)
        set_value(delivery,'Stream',MeasuredStream)
        original_source=delivery.read_source
        def checked_source(path):
            row=read_json(BASE/'data/AAPL/manifest.json')['source']
            if Path(path).resolve()!=Path(row['source_csv_read_only']).resolve() or digest(path)!=row['sha256']:
                raise ValueError('Replay AAPL raw source differs from the frozen data manifest')
            return original_source(path)
        set_value(delivery,'read_source',checked_source)
        original_pages=delivery.pages_frame
        def checked_pages(symbol):
            source=read_json(BASE/'data'/symbol/'manifest.json')['source']
            if digest(BASE/'raw'/symbol/'receipt.json')!=source['receipt_sha256']:
                raise ValueError('Replay background raw receipt differs from the frozen data manifest')
            return original_pages(symbol)
        set_value(delivery,'pages_frame',checked_pages)
        set_value(local_package,'tempfile',Temporary());set_value(local_package,'subprocess',proxy)
        # These are the only mutations performed by the two unchanged adapters.
        for module,name in ((trainer.Trainer,'predict'),(trainer.Trainer,'run_until'),
                (inference.FrozenSystem,'predict_origin'),(experiment,'save_prediction'),(experiment,'read_prediction'),
                (experiment,'write_oof'),(experts,'write_oof'),(local_package,'source_files'),(local_package,'shutil'),
                (panel.Panel,'windows'),(numerical,'_installed'),(speed,'_installed'),(packaging,'_installed')):
            set_value(module,name,getattr(module,name))
        numerical.install();speed.install();packaging.install()
        base_files,base_documentation = local_package.source_files,local_package.documentation
        def package_files():
            return {**base_files(),'FUSION_FOLLOWUP_PLAN.md':ROOT/'FUSION_FOLLOWUP_PLAN.md',
                'evidence/followup_binding.json':directory/'binding.json',
                'evidence/followup_input_contracts.json':directory/'input_contracts.json',
                'evidence/followup_CPU_entry.py':ENTRY}
        def documentation(model,report,example):
            text = base_documentation(model,report,example)
            text = text.replace('预测研究是否达标见 `evidence/confirmation.json`。',
                '本包属于选择后的开发版本；`evidence/confirmation.json` 记录开发门槛，独立未来验证仍未完成。')
            return text
        set_value(local_package,'source_files',package_files);set_value(local_package,'documentation',documentation)
        yield SimpleNamespace(delivery=delivery,package=local_package,processes=proxy)
    finally:
        speed.clear()
        for module,name,value in reversed(saved):setattr(module,name,value)


def input_contracts(directory,panel,predictor,config):
    """Exercise real returned weights and retained inputs, including durable faults."""
    from fusion.next.stream import Stream
    model = predictor.model_identity
    state = read_json(directory/'deployment_acceptance'/model/'state.json')
    if state['integrity']!=identity({k:v for k,v in state.items() if k!='integrity'}):raise ValueError('Acceptance state changed')
    origin = max(state['outputs']);expected = state['outputs'][origin]
    if expected['status']!='ok':raise ValueError('Failed acceptance output cannot be a package example')
    root = directory/'contract_states';root.mkdir(exist_ok=True)
    def new(name,pred=predictor):
        import copy
        path = root/(name+'.json')
        if path.exists():raise ValueError('Preserve the previous input contract attempt')
        stream = Stream(panel.stamps,predictor.stocks,predictor.background,config,predictor.models[0][1],pred,model,path)
        stream.bars=copy.deepcopy(state['bars'])
        return stream
    name='AAPL';stamp=pd.Timestamp(origin);received=stamp+pd.Timedelta(minutes=16)
    valid = new('valid');row=valid.bars[name][origin]['values']
    baseline = valid.predict(stamp,received)
    if baseline['status']!='ok' or abs(baseline['predicted_log_return']-expected['predicted_log_return'])>1e-8:
        raise ValueError('Retained input context differs from actual accepted prediction')
    duplicate = valid.ingest(name,stamp,row,received)['status']=='duplicate_ignored'
    rejected = {}
    cases = {'conflicting_duplicate':{**row,'volume':row['volume']+1},'nonfinite':{**row,'close':float('nan')},
             'inconsistent_OHLCV':{**row,'high':row['low']-1}}
    for key,values in cases.items():
        try:valid.ingest(name,stamp,values,received)
        except ValueError:rejected[key]=True
        else:raise ValueError('Invalid input contract was accepted: '+key)
    try:valid.ingest(name,stamp,row,stamp)
    except ValueError:rejected['unfinished_bar']=True
    else:raise ValueError('Unfinished bar was accepted')
    missing = new('missing_current');missing.bars[predictor.background[0]].pop(origin)
    failed = missing.predict(stamp,received)
    if failed['status']!='input_or_model_failure':raise ValueError('Missing current bar produced an unmarked prediction')
    recovered = Stream(panel.stamps,predictor.stocks,predictor.background,config,predictor.models[0][1],
        predictor,model,missing.state_path);recovered.resume()
    durable = recovered.predict(stamp)['status']=='already_generated' and 'reason' in recovered.outputs[origin]
    allowed = new('missing_two_context_bars')
    keys = sorted(k for k in allowed.bars[name] if pd.Timestamp(k)<stamp)[-2:]
    for key in keys:allowed.bars[name].pop(key)
    allowed_output = allowed.predict(stamp,received)
    if allowed_output['status']!='ok':raise ValueError('Two context gaps failed the admitted missing-input policy')
    excessive = new('missing_three_context_bars')
    for key in sorted(k for k in excessive.bars[name] if pd.Timestamp(k)<stamp)[-3:]:excessive.bars[name].pop(key)
    excessive_output=excessive.predict(stamp,received)
    late = new('delayed_timing');late_output=late.predict(stamp,received,mode='timely_forecast')
    class Fault:
        def predict_origin(self,*args,**kwargs):raise RuntimeError('controlled acceptance model failure')
    fault = new('controlled_model_failure',Fault());fault_output=fault.predict(stamp,received)
    checks = {'exact_retained_context_reproduces_prediction':True,'duplicate_bar_ignored':duplicate,**rejected,
        'missing_current_is_failed_and_durable':durable,'two_context_gaps_follow_admitted_policy':True,
        'excessive_context_gaps_rejected':excessive_output['status']=='input_or_model_failure',
        'delayed_inputs_cannot_claim_timely_forecast':late_output['status']=='input_or_model_failure',
        'controlled_model_failure_is_journaled':fault_output['status']=='input_or_model_failure'
            and read_json(fault.state_path)['outputs'][origin]['reason']=='controlled acceptance model failure'}
    if not all(checks.values()):raise ValueError('A required real input/recovery contract failed')
    result = {'model_identity':model,'checks':checks,'all_passed':True,'Q3_accessed':False,
              'controlled_failures_are_not_research_scores':True,'independent_research_success_claimed':False,
              'state_sha256':{p.name:digest(p) for p in root.glob('*.json')},'checked_at_utc':utc_now()}
    result['identity']=identity(result);write_json(directory/'input_contracts.json',result)
    return result


def checked_replay(directory,binding):
    model = binding['outcome']['deployment_identity'];root=directory/'deployment_acceptance'/model
    report,progress = read_json(root/'report.json'),read_json(root/'progress.json')
    if (report['model_identity']!=model or report.get('followup_binding_identity')!=binding['identity']
            or report['evaluation_precision']!='float32' or report['numerical_policy_identity']!=NUMERICAL
            or report['speed_policy_identity']!=SPEED or report['Q3_accessed'] is not False
            or report['Q2_Q3_scored'] is not False or report['independent_research_success_claimed'] is not False
            or report['origins']!=ORIGINS or report['timed_origins']!=ORIGINS or report['timing_complete'] is not True
            or progress['followup_binding_identity']!=binding['identity']
            or progress['recipe']['system_freeze_sha256']!=digest(directory/'system_freeze.json')
            or progress['recipe']['source_identity']!=source_identity() or progress['recipe']['model_identity']!=model
            or len({(p['pid'],p['created_unix']) for p in progress['processes']})<2):
        raise ValueError('Actual full FP32 replay, fresh processes or its binding is incomplete')
    from fusion.next.delivery import score_journal
    if score_journal(root/'predictions.npz')!=report['recomputed_scores']:raise ValueError('Replay predictions or scores changed')
    from fusion.next.panel import Panel
    panel=Panel.load();rows=panel.origins['AAPL']
    rows=rows[(panel.dates[rows]>='2026-01-01')&(panel.dates[rows]<='2026-03-31')][:ORIGINS]
    if len(rows)!=ORIGINS:raise ValueError('Actual replay did not contain 256 fixed Q1 origins')
    state=read_json(root/'state.json')
    if (state['integrity']!=identity({k:v for k,v in state.items() if k!='integrity'})
            or state['recipe']['model_identity']!=model
            or set(state['outputs'])!={panel.stamps[row].isoformat() for row in rows}
            or any(row['status']!='ok' for row in state['outputs'].values())):
        raise ValueError('Replay journal contains missing, changed or failed origins')
    with np.load(root/'predictions.npz',allow_pickle=False) as saved:
        journal=np.asarray([state['outputs'][panel.stamps[row].isoformat()]['predicted_log_return'] for row in rows])
        if (not np.array_equal(saved['rows'],rows) or not np.array_equal(saved['dates'],panel.dates[rows])
                or not np.array_equal(saved['target'],panel.targets['AAPL'][rows]) or not np.array_equal(saved['prediction'],journal)):
            raise ValueError('Replay saved predictions differ from the fixed origins, targets or durable journal')
    timing=progress['latencies_seconds']
    if set(timing)!={str(int(row)) for row in rows} or any(not np.isfinite(t) or t<0 for t in timing.values()):
        raise ValueError('Replay did not time every fixed origin')
    p95=float(np.quantile([timing[str(int(row))] for row in rows],.95))
    cold=max(p['full_process_to_first_output_seconds'] for p in progress['processes'])
    if report['warm_single_origin_p95_seconds']!=p95 or report['cold_load_seconds']!=cold:
        raise ValueError('Replay latency or cold-start summary differs from actual measurements')
    if report['engineering_acceptance_passed']!=qualified(report):
        raise ValueError('Replay engineering acceptance differs from the preserved thresholds')
    for p in progress['processes']:
        if guard.exact_alive({'pid':p['pid'],'created_unix':p['created_unix']}):
            raise RuntimeError('Replay child is still alive; wait for its actual exit')
    return report


def run_phase(phase,directory,frozen,deployment,binding,resume=False,stop_after=None):
    """Low-level reuse for isolated CPU integration tests and admitted children."""
    with local_runtime(directory,frozen,binding) as runtime:
        if phase=='replay':
            return runtime.delivery.replay(deployment,ORIGINS,resume=resume,stop_after=stop_after)
        report = checked_replay(directory,binding)
        from fusion.next.panel import Panel
        from fusion.next.inference import FrozenSystem
        if (directory/'input_contracts.json').exists():contracts=identified(directory/'input_contracts.json')
        else:
            pending=directory/'contract_states'
            if pending.exists():pending.rename(directory/('incomplete_contract_states_'+str(time.time_ns())))
            contracts=input_contracts(directory,Panel.load(),FrozenSystem(deployment,'cpu'),frozen['config'])
        if (contracts['model_identity']!=report['model_identity'] or not contracts['all_passed'] or not all(contracts['checks'].values())
                or any(digest(directory/'contract_states'/name)!=sha for name,sha in contracts['state_sha256'].items())):
            raise ValueError('Input contract evidence belongs to another model or failed')
        report['actual_input_contract_checks_pending']=False
        report['engineering_acceptance_passed']=qualified(report,contracts['all_passed'])
        write_json(directory/'deployment_acceptance'/report['model_identity']/'report.json',report)
        archive=directory/'local_packages'/(report['model_identity']+'.tar')
        receipt=archive.with_suffix('.receipt.json')
        if not receipt.exists():
            for path in (archive.with_suffix(''),archive):
                if path.exists():path.rename(path.with_name(path.name+'.incomplete_'+str(time.time_ns())))
            archive=runtime.package.build(deployment)
        saved=read_json(receipt)
        if saved['archive_sha256']!=digest(archive) or saved['archive_bytes']!=archive.stat().st_size:
            raise ValueError('Preserve the existing package; its bytes changed')
        verification=runtime.package.verify(archive)
        actual=runtime.processes.records
        if (len(actual)!=3 or [r['command'][-1] for r in actual]!=['--verify-only','--example','--example']
                or any(r['returncode']!=0 for r in actual)
                or len({(r['process']['pid'],r['process']['created_unix']) for r in actual})!=3):
            raise ValueError('Standalone package did not execute three distinct successful processes')
        verification.update(followup_binding_identity=binding['identity'],actual_processes=actual,
            independent_research_success_claimed=False,Q2_Q3_scored=False)
        write_json(archive.with_suffix('.verification.json'),verification)
        return {'report':report,'package':str(archive.relative_to(ROOT)),'verification':verification}


def child_request(path):
    path=Path(path).resolve()
    if not path.is_relative_to(FOLLOW/'runs') or path.parent.name!='process_requests':
        raise ValueError('CPU child request leaves its exact local delivery namespace')
    row=identified(path);attempt=Path(row['guard_attempt'])
    request=identified(attempt/'guard_request.json');state=read_json(attempt/'guard_state.json')
    parent=row['parent']
    if (request['purpose']!='delivery_cpu' or request['limit']!=row['limit'] or request['limit']['deadline_utc']!=DEADLINE
            or state['stage']!='running' or state['request_identity']!=request['identity']
            or row['guard_request_identity']!=request['identity'] or state['worker']!=parent
            or not guard.exact_alive(parent) or not guard.exact_alive(state['guard']) or not guard.exact_alive(request['owner'])
            or os.getppid()!=parent['pid'] or os.getpgid(0)!=parent['pid'] or os.getsid(0)!=parent['pid']
            or row['entry_sha256']!=digest(ENTRY) or request['guard_sha256']!=digest(ROOT/'scripts/followup_guard.py')
            or time.time()>=request['stop_by_unix'] or row['command']!=[sys.executable,str(ENTRY),'--child-request',str(path)]):
        raise ValueError('CPU replay/package child is not bound to the live exact guarded parent')
    if psutil.Process().cmdline()!=row['command']:raise ValueError('CPU child arguments differ from its fixed request')
    if psutil.Process(parent['pid']).cmdline()!=request['command']:
        raise ValueError('CPU child parent command differs from its guarded entry')
    manifest_path=Path(row['manifest']);manifest=check_frozen(manifest_path)
    if digest(manifest_path)!=row['matrix_sha256']:raise ValueError('CPU child matrix differs')
    binding=identified(path.parent.parent/'binding.json')
    if binding['identity']!=row['binding_identity'] or binding['entry_sha256']!=digest(ENTRY):
        raise ValueError('CPU child return binding differs')
    # Recompute the exact prediction/checkpoint chain before loading any model.
    if returned.completion(manifest_path,manifest)!=binding['outcome']:raise ValueError('Returned outcome changed before CPU child')
    directory,frozen,deployment=prepare_local(manifest,binding)
    if row['phase'] not in ('replay','package') or row['stop_after'] not in (None,ORIGINS//2):
        raise ValueError('CPU child phase or controlled stop differs')
    return run_phase(row['phase'],directory,frozen,deployment,binding,row['resume'],row['stop_after'])


def invoke_child(manifest_path,attempt,directory,binding,phase,resume=False,stop_after=None):
    request=identified(Path(attempt)/'guard_request.json')
    path=directory/'process_requests'/(str(time.time_ns())+'.json')
    command=[sys.executable,str(ENTRY),'--child-request',str(path)]
    row={'manifest':str(Path(manifest_path).resolve()),'matrix_sha256':digest(manifest_path),
         'binding_identity':binding['identity'],'entry_sha256':digest(ENTRY),'phase':phase,'resume':resume,'stop_after':stop_after,
         'guard_attempt':str(Path(attempt).resolve()),'guard_request_identity':request['identity'],
         'parent':guard.process_identity(),'limit':request['limit'],'command':command}
    row['identity']=identity(row);write_json(path,row)
    log=directory/'process_logs'/(path.stem+'.log');log.parent.mkdir(exist_ok=True)
    with log.open('xb') as output:
        process=subprocess.Popen(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,
            env={**os.environ,'PYTHONPATH':str(ROOT),'CUDA_VISIBLE_DEVICES':'','FUSION_EXECUTION_PROFILE':'local'})
        dispatch={'process':guard.process_identity(process.pid),'request_identity':row['identity'],'command':command}
        write_json(path.with_suffix('.dispatch.json'),dispatch)
        try:code=process.wait(timeout=max(1,request['stop_by_unix']-time.time()))
        except BaseException:
            process.kill();process.wait();raise
    write_json(path.with_suffix('.exit.json'),{**dispatch,'returncode':code,'exited_at_utc':utc_now()})
    if code:raise RuntimeError('Local follow-up child failed; inspect preserved '+str(log.relative_to(ROOT)))


def worker(manifest_path,attempt):
    request=bind_guarded_worker(attempt,ENTRY,'delivery_cpu');manifest=check_frozen(manifest_path)
    if os.environ.get('FUSION_EXECUTION_PROFILE','local')!='local':raise RuntimeError('Follow-up acceptance is local CPU only')
    admission=identified(Path(attempt)/'launch_admission.json')
    body=checked_return(manifest_path,manifest);body['identity']=identity(body)
    if (request['limit']!=manifest['limit'] or request['stage_seconds']>CPU_SECONDS
            or admission['binding']!=body or admission['stage_seconds']!=request['stage_seconds']):
        raise ValueError('Local follow-up delivery admission changed')
    directory,frozen,deployment=prepare_local(manifest,body)
    if body['outcome']['new_model_exported']:
        root=directory/'deployment_acceptance'/body['outcome']['deployment_identity']
        if not (root/'report.json').exists():
            if not (root/'state.json').exists():invoke_child(manifest_path,attempt,directory,body,'replay',stop_after=ORIGINS//2)
            invoke_child(manifest_path,attempt,directory,body,'replay',resume=True)
        report=checked_replay(directory,body)
        invoke_child(manifest_path,attempt,directory,body,'package')
        report=checked_replay(directory,body)
        archive=directory/'local_packages'/(report['model_identity']+'.tar')
        verification=read_json(archive.with_suffix('.verification.json'))
        if (verification['followup_binding_identity']!=body['identity'] or verification['archive_sha256']!=digest(archive)
                or verification['isolated_example_and_fresh_process_resume_passed'] is not True):
            raise ValueError('Actual package verification is missing or belongs to another return')
        result={'stage':'followup_CPU_delivery_complete','model_identity':report['model_identity'],
            'actual_new_model_CPU_acceptance_passed':report['engineering_acceptance_passed'],
            'CPU_acceptance_report_sha256':digest(root/'report.json'),'input_contracts_sha256':digest(directory/'input_contracts.json'),
            'package':str(archive.relative_to(ROOT)),'package_sha256':digest(archive),
            'package_verification_sha256':digest(archive.with_suffix('.verification.json')),
            'standalone_package_actually_executed':True}
    else:
        result={'stage':'followup_negative_pilot_complete_original_package_retained',
            'model_identity':body['outcome']['retained_original_model_identity'],
            'package_sha256':body['outcome']['retained_original_package_sha256'],
            'actual_new_model_CPU_acceptance_passed':False,'new_model_exported':False,'original_actual_CPU_package_retained':True}
    result.update(manifest_identity=manifest['identity'],binding_identity=body['identity'],
        independent_research_success_claimed=False,Q2_Q3_scored=False,completed_at_utc=utc_now())
    result['identity']=identity(result);write_json(directory/'completion.json',result)
    return result


def launch(manifest_path):
    manifest_path=Path(manifest_path).resolve();manifest=check_frozen(manifest_path)
    if os.environ.get('FUSION_EXECUTION_PROFILE','local')!='local':raise RuntimeError('Accept only on local CPU')
    with execution_locks():
        binding=checked_return(manifest_path,manifest);binding['identity']=identity(binding)
        run=FOLLOW/'runs'/manifest['identity']
        if (run/'local_delivery/completion.json').exists():raise RuntimeError('Local follow-up delivery completed; verify existing evidence')
        previous=sorted((run/'attempts').glob('CPU_delivery_*'))
        for prior in previous:
            if (prior/'guard_request.json').exists():spent_seconds(prior/'guard_request.json',CPU_SECONDS)
        left=(datetime.fromisoformat(DEADLINE)-datetime.now(timezone.utc)).total_seconds()
        if left<=65:raise TimeoutError('Original cutoff leaves no CPU delivery and stop grace')
        cap=min(CPU_SECONDS,left);attempt=run/'attempts'/('CPU_delivery_'+str(len(previous)+1).zfill(2))
        if attempt.exists():raise RuntimeError('Preserve previous partial CPU delivery launch')
        admission={'binding':binding,'stage_seconds':cap,'issued_at_utc':utc_now()};admission['identity']=identity(admission)
        write_json(attempt/'launch_admission.json',admission)
        command=[sys.executable,str(ENTRY),'--manifest',str(manifest_path),'--worker','--attempt',str(attempt)]
        os.environ['PYTHONPATH']=str(ROOT)
        process,dispatch=guard.start(attempt,command,manifest['limit'],cap,purpose='delivery_cpu')
        try:code=process.wait(timeout=max(1,dispatch['stop_by_unix']-time.time()+10))
        finally:stop_guard(process,attempt)
        if code or read_json(attempt/'guard_state.json')['stage']!='completed':raise RuntimeError('Local follow-up delivery failed/stopped; inspect saved logs')
        print(identified(run/'local_delivery/completion.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path);parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--attempt',type=Path,help=argparse.SUPPRESS);parser.add_argument('--child-request',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.child_request:
        if args.manifest or args.attempt or args.worker:parser.error('Internal child uses its exact request only')
        child_request(args.child_request)
    elif args.worker:
        if not args.manifest or not args.attempt:parser.error('Internal worker requires matrix and exact guard attempt')
        worker(args.manifest,args.attempt)
    else:
        if not args.manifest or args.attempt:parser.error('Specify the frozen matrix; attempts are fixed by the launcher')
        launch(args.manifest)
