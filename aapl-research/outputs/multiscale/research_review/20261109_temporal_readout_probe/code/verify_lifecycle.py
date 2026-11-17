"""Actual synthetic process restoration and frozen maximum-shape cost checks."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parents[1]
WORKER=BASE/'code/temporal_worker.py'


def load(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert set(os.sched_getaffinity(0))=={24}
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    runs=BASE/'engineering';rows=[];times=[]

    def command(name,batches=8,epochs=2,*extra):
        return [sys.executable,str(WORKER),'--mode','synthetic','--run-dir',str(runs/name),
                '--synthetic-batches',str(batches),'--synthetic-epochs',str(epochs),*extra]

    def invoke(name,batches=8,epochs=2,*extra,expected=0):
        started=time.perf_counter()
        result=subprocess.run(command(name,batches,epochs,*extra),capture_output=True,text=True)
        times.append({'name':name,'batches':batches,'epochs':epochs,'wall_seconds':time.perf_counter()-started,
                      'exit_code':result.returncode})
        (runs/(name+f'_exit{result.returncode}.log')).write_text(result.stdout+result.stderr)
        if result.returncode!=expected:
            raise RuntimeError(f'{name}: wanted {expected},actual {result.returncode}: {result.stderr[-3000:]}')
        return result

    def payload(name):
        folder=runs/name;manifest=load(folder/'CHECKPOINT.json')
        path=folder/manifest['payload_file'];assert sha(path)==manifest['payload_sha256']
        value=torch.load(path,map_location='cpu',weights_only=True)
        assert value['identity']==manifest['identity']
        return value

    def equal_tree(left,right):
        if torch.is_tensor(left):return torch.equal(left,right)
        if isinstance(left,dict):return left.keys()==right.keys() and all(equal_tree(left[k],right[k]) for k in left)
        if isinstance(left,(list,tuple)):return len(left)==len(right) and all(equal_tree(a,b) for a,b in zip(left,right))
        return left==right

    def compare(left,right):
        a,b=payload(left),payload(right)
        for key in ['identity','binding','model','initial_model','optimizer','numpy_rng','torch_rng']:
            assert equal_tree(a[key],b[key]),key
        for key in ['epoch','cursor','steps','permutation','phase','epoch_losses','loss_sum','seen']:
            assert equal_tree(a['state'][key],b['state'][key]),key
        ar,br=load(runs/left/'RESULT.json'),load(runs/right/'RESULT.json')
        assert ar['status']==br['status']
        if ar['tail_scored']:
            assert ar['tail_metrics']==br['tail_metrics']
            assert ar['B_vs_controls']==br['B_vs_controls']
            with np.load(runs/left/'PAIRED_TAIL_LOSSES.npz') as x,np.load(runs/right/'PAIRED_TAIL_LOSSES.npz') as y:
                assert x.files==y.files and all(np.array_equal(x[k],y[k]) for k in x.files)
        assert a['binding']['stage_deadline_epoch']==load(BASE/'STAGE_SCOPE.json')['deadline_epoch']
        return {'compared':[left,right],'training_steps':a['state']['steps'],
                'model_optimizer_initial_RNG_state_exact':True,'losses_and_predictions_exact':True,
                'original_deadline_preserved':True}

    invoke('continuous8')
    invoke('pause8',8,2,'--pause-after-steps','3',expected=2)
    assert payload('pause8')['state']['steps']==3
    invoke('pause8');rows.append({'name':'actual_batch_pause_resume',**compare('continuous8','pause8')})
    invoke('timeout100',100,2,'--segment-seconds','0.05',expected=2)
    assert 0<payload('timeout100')['state']['steps']<160
    invoke('timeout100',100,2)
    invoke('continuous100',100,2)
    rows.append({'name':'actual_elapsed_timeout_resume',**compare('continuous100','timeout100')})

    child=subprocess.Popen(command('signal100',100,2),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    observation=None;until=time.monotonic()+30
    while time.monotonic()<until:
        path=runs/'signal100/STATUS.json'
        if path.exists():
            value=load(path)
            if value['phase']=='training' and value['steps']>0:
                observation=value['steps'];child.send_signal(signal.SIGUSR1);break
        if child.poll() is not None:break
        time.sleep(0.01)
    out,err=child.communicate(timeout=30)
    (runs/'signal100_interrupted.log').write_text(out+err)
    assert observation is not None and child.returncode==2
    paused=payload('signal100')['state']['steps'];assert 0<paused<160
    invoke('signal100',100,2)
    rows.append({'name':'actual_SIGUSR1_resume','observed_step':observation,'saved_step':paused,
                 **compare('continuous100','signal100')})

    # Alter actual saved tensors without altering identity: checksum rejects it.
    for name,field in [('damaged_model','model'),('damaged_optimizer','optimizer'),('damaged_initial','initial_model')]:
        directory=runs/name;directory.mkdir()
        manifest=load(runs/'continuous8/CHECKPOINT.json')
        source=runs/'continuous8'/manifest['payload_file']
        value=torch.load(source,map_location='cpu',weights_only=True)
        if field=='optimizer':
            first=next(iter(value[field]['state'].values()));first['exp_avg'].view(-1)[0]+=1
        else:next(iter(value[field].values())).view(-1)[0]+=1
        torch.save(value,directory/manifest['payload_file'])
        (directory/'CHECKPOINT.json').write_text(json.dumps(manifest))
        invoke(name,expected=1)
        assert not (directory/'RESULT.json').exists()
        rows.append({'name':name,'actual_payload_mutation_rejected':True})

    # A committed pointer survives an orphan left by a failed later commit.
    directory=runs/'orphan8';directory.mkdir()
    manifest=load(runs/'continuous8/CHECKPOINT.json')
    shutil.copy2(runs/'continuous8'/manifest['payload_file'],directory/manifest['payload_file'])
    (directory/'CHECKPOINT.json').write_text(json.dumps(manifest))
    orphan=directory/f"CHECKPOINT.{manifest['generation']+1:06d}.pt";orphan.write_bytes(b'uncommitted orphan')
    invoke('orphan8');assert orphan.read_bytes()==b'uncommitted orphan'
    rows.append({'name':'orphan_generation_preserved_and_skipped',**compare('continuous8','orphan8')})

    # Isolated hand-built queue fixtures exercise the real controller without
    # ever admitting a market worker. All four valid synthetic results exist.
    fixture=runs/'controller_fixture';(fixture/'code').mkdir(parents=True)
    shutil.copy2(BASE/'code/run_four_states.py',fixture/'code/run_four_states.py')
    deadline=load(BASE/'STAGE_SCOPE.json')['deadline_epoch']
    (fixture/'STAGE_SCOPE.json').write_text(json.dumps({'deadline_epoch':deadline}))
    (fixture/'CODE_FREEZE.json').write_text(json.dumps({'synthetic_fixture':True}))
    states=[{'id':f'synthetic{i}','cfg':{'synthetic':i},'fixed_epochs':2} for i in range(4)]
    (fixture/'STAGE_INPUT_IDENTITIES.json').write_text(json.dumps({'states':states}))
    admission={'status':'admitted_four_fixed_temporal_development_states',
        'controller_sha256':sha(fixture/'code/run_four_states.py'),
        'code_freeze_sha256':sha(fixture/'CODE_FREEZE.json'),
        'stage_input_identities_sha256':sha(fixture/'STAGE_INPUT_IDENTITIES.json'),
        'state_ids':[s['id'] for s in states]}
    (fixture/'MARKET_ADMISSION.json').write_text(json.dumps(admission))

    def fixture_result(state):
        directory=fixture/'runs'/state['id'];directory.mkdir(parents=True,exist_ok=True)
        binding={'stage_deadline_epoch':deadline,'state_id':state['id'],'cfg':state['cfg'],
            'fixed_epochs':2,'scope_sha256':sha(fixture/'STAGE_SCOPE.json'),
            'code_freeze_sha256':admission['code_freeze_sha256'],
            'stage_input_identities_sha256':admission['stage_input_identities_sha256']}
        result={'status':'completed_numerical_nonqualification','binding':binding,
            'identity':hashlib.sha256(json.dumps(binding,sort_keys=True,allow_nan=False).encode()).hexdigest(),
            'Q3_values_read':False,'new_candidate_admitted':False,'tail_scored':False}
        (directory/'RESULT.json').write_text(json.dumps(result))
        (directory/'RESULT_COMMIT.json').write_text(json.dumps({'identity':result['identity'],
            'result_sha256':sha(directory/'RESULT.json')}))
        return directory,result

    for state in states:fixture_result(state)
    valid=subprocess.run([sys.executable,str(fixture/'code/run_four_states.py')],capture_output=True,text=True)
    assert valid.returncode==0 and load(fixture/'QUEUE_STATE.json')['status']=='completed_all_four_fixed_states'
    for kind in ['payload_checksum','identity','state_binding']:
        directory,value=fixture_result(states[0])
        if kind=='payload_checksum':value['extra_accidental_mutation']=1
        elif kind=='identity':value['identity']='wrong'
        else:
            value['binding']['cfg']={'synthetic':'wrong'}
            value['identity']=hashlib.sha256(json.dumps(value['binding'],sort_keys=True,allow_nan=False).encode()).hexdigest()
        (directory/'RESULT.json').write_text(json.dumps(value))
        if kind!='payload_checksum':
            (directory/'RESULT_COMMIT.json').write_text(json.dumps({'identity':value['identity'],
                'result_sha256':sha(directory/'RESULT.json')}))
        rejected=subprocess.run([sys.executable,str(fixture/'code/run_four_states.py')],capture_output=True,text=True)
        (runs/f'controller_reject_{kind}.log').write_text(rejected.stdout+rejected.stderr)
        assert rejected.returncode==1
        assert not list((fixture/'runs').glob('*/WORKER_LOG.txt'))
        rows.append({'name':'controller_rejects_'+kind,'no_worker_dispatched':True})

    # Nominal128 singleton rule really visits129 and never drops the last row.
    spec=importlib.util.spec_from_file_location('temporal_worker_helpers',WORKER)
    helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
    assert helpers.next_batch_end(0,129)==129 and helpers.next_batch_end(128,257)==257
    rows.append({'name':'original_singleton_merge129','passed':True})

    # Prefix scaler cannot depend on purged later labels, even if those change.
    scope=load(BASE/'STAGE_SCOPE.json');parent=Path(scope['parent_directory'])
    reference=load(parent/'INPUT_IDENTITIES.json')['reference_root'];sys.path.insert(0,reference)
    from multiscale.bars import Bars
    from multiscale.features import CloseWindows,fit_scaler
    from multiscale.models import make_model
    n=2000;ns=np.arange(n,dtype=np.int64)*60_000_000_000
    close=100*np.exp(np.cumsum(np.random.default_rng(42).normal(0,0.001,n)))
    origins=np.arange(479,1503);cut=4*len(origins)//5;tail=origins[cut:]
    update=origins[:cut];update=update[ns[update+60]<ns[tail[0]]]
    a=Bars('minute',ns-60_000_000_000,ns,close,np.ones(n,dtype=bool),np.full(n,'2026-01-01'))
    alternate=close.copy();alternate[tail[0]:]*=np.exp(np.linspace(0,1,n-tail[0]))
    b=Bars('minute',a.start_ns,ns,alternate,a.observed,a.session_date)
    sa=fit_scaler(CloseWindows(a,480),update,60,'future_log_return')
    sb=fit_scaler(CloseWindows(b,480),update,60,'future_log_return')
    assert sa==sb
    rows.append({'name':'purged_tail_value_mutation_leaves_all_prefix_scalers_exact','passed':True})

    # Native maximum129 shape includes real backward/Adam and extra D forward.
    torch.manual_seed(2026);net=make_model('lstm',480,60)
    with torch.no_grad():net.head.weight.zero_();net.head.bias.zero_()
    opt=torch.optim.Adam(net.parameters(),lr=0.001)
    x=torch.randn(129,480,1);anchor=torch.randn(129,1);y=torch.randn(129,60)
    for _ in range(3):
        opt.zero_grad(set_to_none=True);loss=(net(x,anchor)-y).square().mean();loss.backward();opt.step()
    started=time.perf_counter()
    for _ in range(20):
        opt.zero_grad(set_to_none=True);loss=(net(x,anchor)-y).square().mean();loss.backward();opt.step()
    train_rate=(time.perf_counter()-started)/20
    net.eval();started=time.perf_counter()
    with torch.inference_mode():
        for _ in range(20):net(x,anchor)
    forward_rate=(time.perf_counter()-started)/20
    result={'status':'passed_actual_training_lifecycle_and_synthetic_qualification',
        'checks':rows,'subprocess_timings':times,'maximum_shape':{'batch':129,'lookback':480,'horizon':60},
        'max_shape_training_seconds_per_batch':train_rate,'max_shape_forward_seconds_per_batch':forward_rate,
        'source_sha256':{'worker':sha(WORKER),'lifecycle':sha(Path(__file__))},
        'code_freeze_sha256':sha(BASE/'CODE_FREEZE.json'),'stage_deadline_epoch':scope['deadline_epoch'],
        'market_arrays_or_old_checkpoint_weights_loaded':False,'GPU':False,'Q3_values_read':False}
    (BASE/'WORKER_RUNTIME_ACCEPTANCE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
