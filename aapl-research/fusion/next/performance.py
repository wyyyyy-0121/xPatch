"""F2 cloud-only maximum-shape admission, using 2021-2022 training observations."""
import argparse
import copy
import gc
import time

import numpy as np
import torch

from .common import OUT, PLAN, digest, read_json, write_json, utc_now,source_identity
from .hardware import runtime, inspect_hardware, ResourceGuard
from .panel import Panel, BalancedSampler
from .models import NeuralFusion,MarketBranch
from .expert_models import NativeExpert
from .trainer import optimizer_for, loss_for
from .checkpoint import changing_state, capture_rng, restore_rng, save, load
from .accounting import Usage
from .cache import RepresentationCache
from .experts import fit_cpu,predict_cpu


BASE={'family':'fusion','lookback':240,'projection':128,'hidden':64,'market_encoder':'transformer',
      'residual_cap':.5,'dropout':0.,'lr':3e-4,'weight_decay':.01,'loss':'raw','information':'ohlcv',
      'foundation_mode':'frozen','rank':16,'precision':'fp32','fused_optimizer':False}


def train_batch(model,opt,batch,config):
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config.get('precision')=='bf16'):
        output=model(batch);loss=loss_for(output,batch,config)
    if not torch.isfinite(loss):raise FloatingPointError('F2 nonfinite loss')
    loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
    if not torch.isfinite(norm):raise FloatingPointError('F2 nonfinite gradients')
    opt.step();return float(loss.detach())


def benchmark(model,batches,config,updates=12,cached=False):
    model.train();opt=optimizer_for(model,config,'cuda');torch.cuda.synchronize();start=time.perf_counter()
    losses=[]
    for step in range(updates):
        batch=batches[step%len(batches)]
        if not cached:batch={k:v for k,v in batch.items() if k!='foundation_cached'}
        losses.append(train_batch(model,opt,batch,config))
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    model.eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=config.get('precision')=='bf16'):
        predictions=model(batches[0] if cached else {k:v for k,v in batches[0].items() if k!='foundation_cached'})['prediction'].float().cpu().numpy()
    return {'updates':updates,'seconds':elapsed,'seconds_per_update':elapsed/updates,
            'losses':losses,'predictions':predictions.tolist()}


def primitive_experts(batch):
    r=batch['foundation_context'][:,:,0]
    return torch.stack([r[:,-12:].sum(1),r.mean(1)*12,r[:,-48:].mean(1)*12,torch.zeros_like(r[:,0])],1)


def end_to_end(model,panel,ids,scalers,config,representation_cache=None):
    model.train();opt=optimizer_for(model,config,'cuda');torch.cuda.synchronize();start=time.perf_counter()
    for stocks,rows in ids:
        batch=panel.windows(stocks,rows,config,scalers,device='cuda')
        batch['experts']=primitive_experts(batch)
        if representation_cache is not None:batch['foundation_cached']=representation_cache.get(batch)
        train_batch(model,opt,batch,config)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    return {'seconds':elapsed,'updates':len(ids),'seconds_per_update':elapsed/len(ids),
            'includes_sampling_windows_transfer_cache_IO_optimizer':True}


@torch.inference_mode()
def inference_timing(model,panel,ids,scalers,config):
    model.eval();torch.cuda.synchronize();start=time.perf_counter();count=0
    for stocks,rows in ids:
        batch=panel.windows(stocks,rows,config,scalers,device='cuda',include_labels=False)
        batch['experts']=primitive_experts(batch)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config.get('precision')=='bf16'):
            prediction=model(batch)['prediction']
        if not torch.isfinite(prediction).all():raise FloatingPointError('F2 nonfinite inference')
        count+=len(rows)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    return {'seconds':elapsed,'origins':count,'seconds_per_origin':elapsed/count,'includes_windows_transfer':True}


def cpu_admission():
    if (OUT/'search_manifest.json').exists():raise ValueError('CPU admission must precede score/search freeze')
    if not read_json(OUT/'data_audit.json')['passed']:raise ValueError('Complete F1 audit required')
    panel=Panel.load();config={'lookback':240,'information':'ohlcv','training_scope':'multistock','rows_per_day':16,
        'family':'lightgbm','trees':1000,'num_leaves':31,'max_depth':-1,'min_child_samples':100,
        'learning_rate':.03,'reg_lambda':10.,'colsample_bytree':1.,'cpu_threads':min(8,len(__import__('os').sched_getaffinity(0)))}
    report={'started_at_utc':utc_now(),'source_identity':source_identity(),'plan_sha256':digest(PLAN),
            'training_data_only':True,'official_validation_scores_read':False,'hardware':inspect_hardware(False),'measurements':{}}
    with Usage('performance','F2 CPU tree/ridge end-to-end',gpu=False):
        for family in ('lightgbm','ridge'):
            cfg={**config,'family':family,'alpha':1000.}
            predict,scalers,measurement=fit_cpu(panel,cfg,'2022-12-30',OUT/'performance/cpu'/family)
            rows=panel.origins['AAPL'];rows=rows[(panel.dates[rows]>='2022-10-01')&(panel.dates[rows]<='2022-12-30')][:4096]
            start=time.perf_counter();predict_cpu(predict,panel,np.zeros(len(rows),dtype=np.int64),rows,cfg,scalers)
            measurement['prediction_seconds_per_origin']=(time.perf_counter()-start)/len(rows)
            measurement['training_calendar_days']=int(len(np.unique(panel.dates[panel.dates<='2022-12-30'])))
            report['measurements'][family]=measurement
    report['completed_at_utc']=utc_now();write_json(OUT/'performance_cpu.json',report);return report


def run(batch_size=64,stress_seconds=900):
    if stress_seconds<900:raise ValueError('F2 continuous stress must last at least 15 minutes')
    device=runtime(2026)
    if device!='cuda':raise RuntimeError('Large-model F2 is assigned to the AutoDL GPU; local execution refused')
    admission=read_json(OUT/'data_audit.json')
    if not admission['passed']:raise ValueError('Complete F1 audit required before GPU admission')
    panel=Panel.load();start=time.perf_counter();scalers=panel.scaler('2022-12-30');scaler_seconds=time.perf_counter()-start
    sampler=BalancedSampler(panel,'2022-12-30',2026)
    config={**BASE,'batch_size':batch_size,'raw_loss_unit':scalers['AAPL']['target_std']}
    report={'started_at_utc':utc_now(),'plan_sha256':digest(PLAN),'hardware':inspect_hardware(True),
            'source_identity':source_identity(),
            'data_audit_sha256':digest(OUT/'data_audit.json'),'config':config,'training_dates_only':True,
            'official_validation_scores_read':False,'passed':False,
            'measurements':{'scaler_fit_seconds':scaler_seconds,
                            'scaler_training_calendar_days':len(np.unique(panel.dates[panel.dates<='2022-12-30']))}}
    path=OUT/'performance_admission.json';write_json(path,report)
    with Usage('performance','F2 maximum-shape and numerical/resume admission') as usage:
        guard=ResourceGuard('cuda');initialization=time.perf_counter()
        model=NeuralFusion(config,12,12*len(panel.background),len(panel.stocks),'cuda').to('cuda')
        report['measurements']['fusion_initialization_seconds']=time.perf_counter()-initialization
        batches=[]
        for _ in range(8):
            stocks,rows=sampler.sample(batch_size)
            if np.any(panel.dates[rows]>'2022-12-30'):raise ValueError('F2 date boundary violated')
            batch=panel.windows(stocks,rows,config,scalers,device='cuda')
            # Causal primitive forecasts provide nonzero gate inputs during the speed test.
            batch['experts']=primitive_experts(batch)
            with torch.no_grad():batch['foundation_cached']=model.foundation.latent(batch['foundation_context']).detach().float()
            batches.append(batch)
        initial=changing_state(model);rng=capture_rng()
        raw=benchmark(model,batches,config);report['measurements']['frozen_raw_fp32']=raw
        model.load_state_dict(initial,strict=False);restore_rng(rng)
        optimized={**config,'fused_optimizer':True}
        cached=benchmark(model,batches,optimized,cached=True);report['measurements']['frozen_cached_fp32']=cached
        a=np.asarray(raw['predictions']);b=np.asarray(cached['predictions']);unit=config['raw_loss_unit']
        checks={'cache_and_fused_prediction_rmse':float(np.sqrt(np.mean((a-b)**2))/unit)<1e-4,
                'cache_and_fused_loss_difference':bool(np.allclose(raw['losses'],cached['losses'],rtol=1e-4,atol=1e-6))}
        if torch.cuda.is_bf16_supported():
            model.load_state_dict(initial,strict=False);restore_rng(rng)
            bf16=benchmark(model,batches,{**optimized,'precision':'bf16'},cached=True)
            report['measurements']['frozen_cached_bf16']=bf16
            c=np.asarray(bf16['predictions'])
            checks['bf16_prediction_rmse']=float(np.sqrt(np.mean((b-c)**2))/unit)<.05
            checks['bf16_loss_difference']=bool(np.allclose(cached['losses'],bf16['losses'],rtol=.05,atol=.001))
            model.load_state_dict(initial,strict=False);restore_rng(rng)
            raw_bf16=benchmark(model,batches,{**optimized,'precision':'bf16'},cached=False)
            report['measurements']['frozen_raw_bf16']=raw_bf16
            c=np.asarray(raw_bf16['predictions'])
            checks['bf16_encoder_prediction_rmse']=float(np.sqrt(np.mean((a-c)**2))/unit)<.05
            checks['bf16_encoder_loss_difference']=bool(np.allclose(raw['losses'],raw_bf16['losses'],rtol=.05,atol=.001))
        else:checks['bf16_supported']=False
        # Float16 is a storage optimization; test it separately from encoder compute precision.
        model.load_state_dict(initial,strict=False);model.eval()
        with torch.no_grad():
            full=model(batches[0])['prediction'];half=dict(batches[0]);half['foundation_cached']=half['foundation_cached'].half().float()
            rounded=model(half)['prediction']
        checks['fp16_cache_prediction_rmse']=float(((full-rounded).square().mean().sqrt()/unit).cpu())<.001
        ids=[sampler.sample(batch_size) for _ in range(24)]
        report['measurements']['frozen_inference']=inference_timing(model,panel,ids[:8],scalers,config)
        model.load_state_dict(initial,strict=False);restore_rng(rng)
        report['measurements']['end_to_end_raw']=end_to_end(model,panel,ids,scalers,config)
        representation_cache=RepresentationCache(panel,model.foundation,config,allowed_end='2022-12-30')
        model.load_state_dict(initial,strict=False);restore_rng(rng)
        report['measurements']['end_to_end_cache_population']=end_to_end(model,panel,ids,scalers,optimized,representation_cache)
        model.load_state_dict(initial,strict=False);restore_rng(rng)
        report['measurements']['end_to_end_cache_reuse']=end_to_end(model,panel,ids,scalers,optimized,representation_cache)
        del representation_cache
        # Exercise a real optimizer interruption with the maximum-shape fusion network.
        model.load_state_dict(initial,strict=False);restore_rng(rng);model.train()
        opt=optimizer_for(model,config,'cuda')
        for step in range(3):train_batch(model,opt,batches[step],config)
        checkpoint_path=OUT/'performance/f2_resume.pt';recipe={'config':config,'purpose':'F2 training-only recovery'}
        save(checkpoint_path,model,opt,{'main':sampler},3,recipe)
        expected_losses=[train_batch(model,opt,batches[step],config) for step in range(3,6)]
        expected=changing_state(model)
        load(checkpoint_path,model,opt,{'main':sampler},recipe)
        observed_losses=[train_batch(model,opt,batches[step],config) for step in range(3,6)]
        actual=changing_state(model)
        checks['resume_losses_equal']=expected_losses==observed_losses
        checks['resume_changing_state_equal']=all(torch.equal(v,actual[name]) for name,v in expected.items())
        # Before destroying the frozen model, measure origin isolation with real Chronos groups.
        model.eval();whole=batches[0]
        with torch.no_grad():
            joined=model({k:v for k,v in whole.items() if k!='foundation_cached'})['prediction'][:2]
            separate=torch.cat([model({k:v[i:i+1] for k,v in whole.items() if k!='foundation_cached'})['prediction'] for i in range(2)])
        checks['foundation_origin_isolation']=bool(torch.allclose(joined,separate,rtol=1e-4,atol=unit*1e-4))
        # Both prescribed market encoders need actual timing. The small TCN is
        # not assumed cheaper merely because its parameter count is smaller.
        model.market=MarketBranch(12*len(panel.background),32,'tcn').to('cuda')
        report['measurements']['frozen_tcn_end_to_end']=end_to_end(model,panel,ids[:8],scalers,config)
        report['measurements']['frozen_tcn_inference']=inference_timing(model,panel,ids[:8],scalers,config)
        del model,opt,initial,expected,actual;gc.collect();torch.cuda.empty_cache()
        for family in ('xpatch','patchtst'):
            native_config={**config,'family':family,'patch':8,'stride':4,'d_model':64}
            native=NativeExpert(native_config,12*(1+len(panel.background)),len(panel.stocks)).to('cuda')
            report['measurements'][family]=benchmark(native,batches,native_config)
            report['measurements'][family+'_end_to_end']=end_to_end(native,panel,ids,scalers,native_config)
            report['measurements'][family+'_inference']=inference_timing(native,panel,ids[:8],scalers,native_config)
            del native;gc.collect();torch.cuda.empty_cache();guard.check();usage.record()
        # Maximum LoRA rank, largest context, all market variables, multistock batch.
        lora_config={**config,'foundation_mode':'lora','precision':'fp32'}
        model=NeuralFusion(lora_config,12,len(panel.background)*12,len(panel.stocks),'cuda').to('cuda')
        initial=changing_state(model);rng=capture_rng()
        lora_fp32=benchmark(model,batches,lora_config,cached=False)
        report['measurements']['lora_fp32']=lora_fp32
        model.load_state_dict(initial,strict=False);restore_rng(rng)
        if checks.get('bf16_encoder_prediction_rmse') and checks.get('bf16_encoder_loss_difference'):
            lora_bf16=benchmark(model,batches,{**lora_config,'precision':'bf16'},cached=False)
            report['measurements']['lora_bf16']=lora_bf16
            a=np.asarray(lora_fp32['predictions']);b=np.asarray(lora_bf16['predictions'])
            checks['bf16_lora_prediction_rmse']=float(np.sqrt(np.mean((a-b)**2))/unit)<.05
            checks['bf16_lora_loss_difference']=bool(np.allclose(lora_fp32['losses'],lora_bf16['losses'],rtol=.05,atol=.001))
            # Formal recipes currently admit FP32. Keep stress and cost timing on
            # FP32 even when the optional BF16 comparison passes.
        model.load_state_dict(initial,strict=False);restore_rng(rng)
        del initial
        model.train();opt=optimizer_for(model,lora_config,'cuda');torch.cuda.synchronize();start=time.perf_counter();steps=0
        while time.perf_counter()-start<stress_seconds:
            batch={k:v for k,v in batches[steps%len(batches)].items() if k!='foundation_cached'}
            train_batch(model,opt,batch,lora_config);steps+=1
            if steps%10==0:
                resources=guard.check();usage.record()
                write_json(OUT/'performance_stress_progress.json',{'seconds':time.perf_counter()-start,'updates':steps,'resources':resources,'Q3_accessed':False})
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        adapters=[name for name,p in model.named_parameters() if p.requires_grad and 'lora_' in name]
        checks['real_LoRA_parameters_attached']=bool(adapters)
        checks['maximum_shape_continuous_15min']=elapsed>=900 and steps>0
        report['measurements']['lora_stress']={'seconds':elapsed,'updates':steps,'seconds_per_update':elapsed/steps,
             'resources':guard.check(),'adapter_parameter_names':adapters,
             'new_trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
        report['measurements']['lora_inference']=inference_timing(model,panel,ids[:8],scalers,lora_config)
        report['measurements']['lora_end_to_end']=end_to_end(model,panel,ids[:8],scalers,lora_config)
        model.market=MarketBranch(12*len(panel.background),32,'tcn').to('cuda')
        report['measurements']['lora_tcn_end_to_end']=end_to_end(model,panel,ids[:8],scalers,lora_config)
        report['measurements']['lora_tcn_inference']=inference_timing(model,panel,ids[:8],scalers,lora_config)
        # FP32 can remain the admitted policy when optional lower precision fails.
        optional={k for k in checks if k.startswith('bf16_')}|{'fp16_cache_prediction_rmse'}
        report.update(checks=checks,numerical_and_resource_checks_passed=all(v for k,v in checks.items() if k not in optional),
                      completed_at_utc=utc_now(),passed=False,
                      remaining_gate='CPU/tree timing, end-to-end all-fold budget projection with 50% margin, then search freeze')
        write_json(path,report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batch-size',type=int,choices=(32,64),default=64)
    parser.add_argument('--cpu-only',action='store_true')
    parser.add_argument('--stress-seconds',type=int,default=900);args=parser.parse_args()
    cpu_admission() if args.cpu_only else run(args.batch_size,args.stress_seconds)
