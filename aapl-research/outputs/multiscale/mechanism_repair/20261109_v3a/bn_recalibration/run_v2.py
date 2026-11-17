"""Versioned numerical acceptance wrapper; frozen intervention and gate unchanged."""
from pathlib import Path
import sys, os, signal, json, time
import numpy as np
import torch
import diagnostic as original
import diagnostic_v2 as revised

HERE=Path(__file__).resolve().parent

def checks(cfg,inner):
    out=HERE/'cells'/cfg['id'];path=out/'NUMERIC_ACCEPTANCE_V2.json'
    if path.exists():return
    folder,done,saved=revised.source(cfg)
    assert revised.sha(folder/'predictions.npz')==done['prediction_sha256']
    data=revised.Data(cfg)
    adapted=torch.load(out/'adapted.pt',map_location='cpu',weights_only=True)
    idx=data.splits['stop'][:19];records={}
    for name,record in (('original',saved),('adapted',adapted)):
        net=revised.load_net(cfg,record,data)
        ref=revised.forward(data,net,idx,19);rows=[]
        for batch in (1,7):
            value=revised.forward(data,net,idx,batch)
            assert np.allclose(value,ref,rtol=1e-6,atol=1e-8)
            rows.append({'batch':batch,'max_abs_error':float(np.max(abs(value-ref))),
                'revised_allclose':True,'original_tolerance_allclose':bool(np.allclose(value,ref,rtol=1e-6,atol=1e-9))})
        records[name]={'raw_inverse_output_dtype':str(ref.dtype),'network_dtype':str(next(net.parameters()).dtype),
            'output_max_abs':float(abs(ref).max()),'output_mean_abs':float(abs(ref).mean()),'checks':rows}
    # Canonical128 predictions are the same function across versioned acceptance.
    net=revised.load_net(cfg,saved,data);po=revised.forward(data,net,data.splits['stop'])
    co=revised.calibrate(data,po,data.splits['stop'])
    net=revised.load_net(cfg,adapted,data);pa=revised.forward(data,net,data.splits['stop'])
    ca=revised.calibrate(data,pa,data.splits['stop'])
    assert co==inner['original'] and ca==inner['adapted']
    revised.atomic(path,{'status':'passed_revised_tolerance','job':cfg,'records':records,
        'rtol':1e-6,'atol':1e-8,'canonical128_inner_predictions_reproduced_exact_metrics':True,
        'original_canonical_prediction_sha256':done['prediction_sha256'],
        'source_outer_prediction_file_hash_verified':True,
        'numerical_revision_sha256':revised.sha(HERE/'NUMERICAL_ACCEPTANCE_REVISION.json'),
        'original_failed_tolerance_preserved':True,'at_utc':revised.now()})

def main():
    revision=revised.read(HERE/'NUMERICAL_ACCEPTANCE_REVISION.json')
    assert revised.sha(HERE/'diagnostic.py')==revision['original_source_sha256']
    assert revised.sha(HERE/'diagnostic_v2.py')==revision['revised_source_sha256']
    identity={'run_v2.py':revised.sha(__file__),
              'revision_sha256':revised.sha(HERE/'NUMERICAL_ACCEPTANCE_REVISION.json'),
              'original_source_freeze_sha256':revised.sha(HERE/'SOURCE_FREEZE.json')}
    freeze=HERE/'RUNNER_V2_FREEZE.json'
    if freeze.exists():assert revised.read(freeze)['identity']==identity
    else:revised.atomic(freeze,{'identity':identity,'at_utc':revised.now(),'at_epoch':time.time(),
                              'adapter_outer_scores_seen':False})
    original.init()
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):signal.signal(sig,revised.interrupt)
    raw_cell=revised.cell
    def checked_cell(cfg):
        inner=raw_cell(cfg);checks(cfg,inner);return inner
    revised.cell=checked_cell
    try:
        gate=revised.frozen_gate()
        for task in gate['full_tasks']:
            profile,rest=task.split('_h',1);h=int(rest.split('_',1)[0])
            for fold in range(1,6):
                for seed in (2026,2027,2028):revised.cell(revised.cfg_for(profile,h,fold,seed))
        revised.atomic(HERE/'OUTER_START.json',{'gate_sha256':revised.sha(HERE/'GATE_AND_ADMISSION.json'),
            'original_source_freeze_sha256':revised.sha(HERE/'SOURCE_FREEZE.json'),
            'runner_v2_freeze_sha256':revised.sha(freeze),'at_epoch':time.time(),'at_utc':revised.now()})
        revised.summarize(gate)
    except revised.Pause:
        revised.atomic(HERE/'PAUSED.json',{'status':'paused','deadline_epoch':revised.DEADLINE,
            'signal':revised.STOP,'at_utc':revised.now(),'runner':'run_v2.py'})
        print(json.dumps({'status':'paused','checkpoints_preserved':True}),flush=True)

if __name__=='__main__':main()
