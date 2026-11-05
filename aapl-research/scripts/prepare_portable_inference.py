"""Preserve BF16 evidence, freeze an inference policy, then rescore final experts."""
import argparse
import copy
import io
import json
from pathlib import Path
import tarfile

import psutil

from fusion.next.common import ROOT, OUT, digest, identity, read_json, write_json, source_identity, utc_now

POLICY=OUT/'performance/portable_inference_policy.json'
ARCHIVE=OUT/'cloud/incremental_backups/before_portable_inference.tar.gz'
RECEIPT=ARCHIVE.with_suffix('.receipt.json')


def require_stopped():
    dispatch=read_json(OUT/'cloud/bounded_run_dispatch.json')
    try:
        process=psutil.Process(dispatch['pid'])
        if abs(process.create_time()-dispatch['created_unix'])<.01 and process.status()!=psutil.STATUS_ZOMBIE:
            raise RuntimeError('The original driver must finish its CPU search and exit first')
    except psutil.NoSuchProcess:pass
    if (OUT/'F3_complete.json').exists() or any((OUT/'oof').glob('*/*.json')):
        raise RuntimeError('This amendment is defined before main OOF starts')
    for family in ('ridge','lightgbm','xpatch','patchtst'):
        if not (OUT/'selections'/('single_'+family+'.json')).exists():
            raise RuntimeError('All original expert searches must finish first')
    from fusion.next.accounting import recover_interrupted, totals
    recover_interrupted()
    if totals()['active_jobs']:raise RuntimeError('An accounted CPU/GPU job remains active')


def prepare():
    require_stopped()
    if POLICY.exists():return read_json(POLICY)
    scope=read_json(OUT/'performance/scope_100h.json')
    if scope['source_identity']!=source_identity():raise ValueError('The admitted engine changed')
    items={};selections={}
    for path in sorted((OUT/'trials').rglob('result.json')):
        prediction=path.with_name('prediction.npz');result=read_json(path)
        if digest(prediction)!=result['prediction_sha256']:raise ValueError('Prior prediction changed')
        for item in (path,prediction):items[str(item.relative_to(ROOT))]=item.read_bytes()
    for family in ('ridge','lightgbm','xpatch','patchtst'):
        path=OUT/'selections'/('single_'+family+'.json');selections[family]=digest(path)
        items[str(path.relative_to(ROOT))]=path.read_bytes()
        if family not in ('xpatch','patchtst'):continue
        for row in read_json(path)['rungs'][-1]['rankings']:
            for quarter in ('2025Q1','2025Q3','2026Q1'):
                path=OUT/'training'/('single_'+family)/row['config']['candidate_id']/quarter/'seed_2026.pt'
                receipt=path.with_suffix('.pt.json');progress=path.with_suffix('.progress.json')
                meta=read_json(receipt)
                if digest(path)!=meta['sha256'] or meta['step']!=meta['recipe']['total_updates']:
                    raise ValueError('Prior full expert checkpoint changed')
                for item in (path,receipt,progress):items[str(item.relative_to(ROOT))]=item.read_bytes()
    import hashlib
    manifest={'source_identity':source_identity(),'scope_identity':scope['identity'],'created_at_utc':utc_now(),
        'files':{name:{'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)} for name,data in items.items()},
        'prior_selections_sha256':selections,'Q3_values_included':False,'canonical_results_changed':False}
    items['before_portable_manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    ARCHIVE.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(ARCHIVE,'w:gz',compresslevel=3) as stream:
        for name,data in items.items():
            member=tarfile.TarInfo(name);member.size=len(data);stream.addfile(member,io.BytesIO(data))
    receipt={'archive':str(ARCHIVE.relative_to(ROOT)),'sha256':digest(ARCHIVE),'bytes':ARCHIVE.stat().st_size,
        'source_identity':source_identity(),'scope_identity':scope['identity'],'created_at_utc':utc_now()}
    write_json(RECEIPT,receipt)
    policy={'version':'portable-inference-fp32-v1','created_at_utc':utc_now(),
        'reason':'Actual BF16 CPU/CUDA expert parity failed; same-weight FP32 parity passed on six known-quarter checkpoints',
        'base_engine_source_identity':source_identity(),'scope_identity':scope['identity'],
        'inference_precision':'float32','training_precision_changed':False,
        'epochs_or_search_scope_changed':False,'hard_deadline_changed':False,
        'BF16_screening_rungs_retained':[.1,.3],'FP32_final_expert_rungs_rescored':[1.],
        'prior_evidence_archive_sha256':receipt['sha256'],'prior_selections_sha256':selections,
        'runtime_sha256':digest(ROOT/'scripts/portable_inference_runtime.py'),
        'CPU_entrypoint_sha256':digest(ROOT/'scripts/portable_cpu_run.py'),
        'launcher_sha256':digest(ROOT/'scripts/start_portable_fusion.py'),
        'rescore_script_sha256':digest(Path(__file__)),
        'same_weight_GPU_FP32_reference_sha256':digest(OUT/'cloud/native_expert_FP32_probe.npz'),
        'hard_limit_sha256':digest(OUT/'cloud/hard_time_limit.json'),'Q3_accessed':False}
    policy['identity']=identity(policy);write_json(POLICY,policy)
    for script in ('portable_inference_runtime.py','portable_cpu_run.py','start_portable_fusion.py','prepare_portable_inference.py'):
        (OUT/'performance'/('portable_'+script+'.txt')).write_bytes((ROOT/'scripts'/script).read_bytes())
    return policy


def rescore():
    require_stopped();policy=read_json(POLICY)
    verified=read_json(OUT/'cloud/portable_prior_backup_verified.json')
    if not verified['all_member_hashes_verified'] or verified['archive_sha256']!=policy['prior_evidence_archive_sha256']:
        raise ValueError('The original evidence must be independently backed up locally first')
    if digest(Path(__file__))!=policy['rescore_script_sha256']:raise ValueError('Rescore script changed')
    from scripts.portable_inference_runtime import install
    install()
    import gc
    import numpy as np
    import torch
    from fusion.next.hardware import runtime, ResourceGuard
    from fusion.next.panel import Panel
    from fusion.next.inference import checkpoint_model
    from fusion.next.experiment import score_rows,save_prediction,read_prediction
    from fusion.next.recipe import for_fit
    from fusion.next.accounting import Usage
    device=runtime(2026)
    if device!='cuda':raise RuntimeError('Rescore exact cloud weights on their training GPU')
    panel=Panel.load();outputs=[];weights={}
    with tarfile.open(ARCHIVE) as archive:
        originals={family:json.load(archive.extractfile('outputs/fusion_next/selections/single_'+family+'.json')) for family in ('xpatch','patchtst')}
    for family,original in originals.items():
        path=OUT/'selections'/('single_'+family+'.json');current=read_json(path)
        if digest(path)!=policy['prior_selections_sha256'][family] and current.get('numerical_policy_identity')!=policy['identity']:
            raise ValueError('Expert selection changed outside the amendment')
        rankings=[]
        for candidate in original['rungs'][-1]['rankings']:
            config=candidate['config'];fold_scores={}
            for quarter in ('2025Q1','2025Q3','2026Q1'):
                fold,rows=score_rows(panel,quarter);effective=for_fit(config,fold['fit_end']);updates=effective['total_updates']
                directory=OUT/'trials'/('single_'+family)/config['candidate_id']/quarter/'seed_2026'/f'updates_{updates}'
                checkpoint=OUT/'training'/('single_'+family)/config['candidate_id']/quarter/'seed_2026.pt'
                weights[str(checkpoint.relative_to(ROOT))]=digest(checkpoint)
                details={'config':effective,'fold':fold,'seed':2026,'updates':updates,'source':source_identity()}
                saved=read_json(directory/'result.json')
                if saved['details'].get('numerical_policy_identity')==policy['identity']:
                    report,_=read_prediction(directory,rows,details)
                else:
                    runtime(2026);recipe=read_json(checkpoint.with_suffix('.pt.json'))['recipe']
                    with Usage('performance','FP32 inference-only rescore/'+family+'/'+config['candidate_id']+'/'+quarter) as usage:
                        model,scalers,_=checkpoint_model(panel,checkpoint,recipe['config'],'cuda');parts=[]
                        with torch.inference_mode():
                            guard=ResourceGuard('cuda')
                            for first in range(0,len(rows),effective['batch_size']):
                                selected=rows[first:first+effective['batch_size']]
                                batch=panel.windows(np.zeros(len(selected),dtype=np.int64),selected,effective,scalers,device='cuda',include_labels=False)
                                with torch.autocast('cuda',enabled=False):parts.append(model(batch)['prediction'].float().cpu().numpy())
                                if first%(20*effective['batch_size'])==0:usage.record();guard.check()
                        prediction=np.concatenate(parts)
                        if not np.isfinite(prediction).all():raise FloatingPointError('FP32 rescore contains nonfinite predictions')
                        report=save_prediction(directory,rows,prediction,panel,details)
                        report['parameters']=saved['parameters'];write_json(directory/'result.json',report)
                        del model,parts,batch;gc.collect();torch.cuda.empty_cache()
                fold_scores[quarter]=report['scores'];outputs.append({'result':str((directory/'result.json').relative_to(ROOT)),'sha256':digest(directory/'result.json')})
            mse=sum(s['n']*s['mse'] for s in fold_scores.values())/sum(s['n'] for s in fold_scores.values())
            rankings.append({'config':config,'mse':mse,'fold_scores':fold_scores})
        rankings.sort(key=lambda row:(row['mse'],row['config']['candidate_id']))
        updated=copy.deepcopy(original)
        for rung in updated['rungs']:rung['evaluation_precision']='bf16' if rung['prefix']<1 else 'float32'
        updated['rungs'][-1]['rankings']=rankings;updated['selected']=rankings[0]['config'];updated['top_two']=[r['config'] for r in rankings[:2]]
        updated.update(numerical_policy_identity=policy['identity'],prior_BF16_selection_sha256=policy['prior_selections_sha256'][family],completed_at_utc=utc_now())
        write_json(path,updated)
    if any(digest(ROOT/name)!=sha for name,sha in weights.items()):raise ValueError('Inference rescore changed a trained weight')
    result={'numerical_policy_identity':policy['identity'],'source_identity':source_identity(),'all_trained_checkpoint_hashes_unchanged':True,
        'rescored_full_quarter_results':outputs,'prior_evidence_archive_sha256':policy['prior_evidence_archive_sha256'],'completed_at_utc':utc_now(),'Q3_accessed':False}
    write_json(OUT/'performance/portable_inference_rescore_complete.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--rescore',action='store_true')
    parser.add_argument('--managed-entry',choices=['run_fusion_100h.py'],help='Register this preparation with the existing project time limiter')
    args=parser.parse_args()
    print(json.dumps(rescore() if args.rescore else prepare()))
