"""Pre-score batch/precision throughput sweep on the fixed 2021-2022 training data."""
import argparse
import gc
import time

import torch

from .common import OUT,PLAN,digest,source_identity,write_json,utc_now
from .hardware import runtime,ResourceGuard,gpu_telemetry
from .panel import Panel,BalancedSampler
from .models import NeuralFusion
from .performance import BASE,primitive_experts,train_batch
from .trainer import optimizer_for
from .accounting import Usage


def run(examples=8192):
    if runtime(2026)!='cuda':raise RuntimeError('Batch scaling is assigned to the cloud GPU')
    if (OUT/'search_manifest.json').exists():raise ValueError('Batch choice must precede official scores')
    panel=Panel.load();scalers=panel.scaler('2022-12-30')
    report={'started_at_utc':utc_now(),'source_identity':source_identity(),'plan_sha256':digest(PLAN),
            'training_only':True,'official_validation_scores_read':False,'examples_per_configuration':examples,'measurements':[]}
    output=OUT/'batch_scaling.json'
    with Usage('performance','pre-score batch and precision throughput scaling') as usage:
        for mode in ('frozen','lora'):
            for precision in ('fp32','bf16'):
                if precision=='bf16' and not torch.cuda.is_bf16_supported():continue
                config={**BASE,'foundation_mode':mode,'precision':precision,'fused_optimizer':True,
                        'raw_loss_unit':scalers['AAPL']['target_std']}
                model=NeuralFusion(config,12,12*len(panel.background),len(panel.stocks),'cuda').to('cuda')
                batches=[64,128,256,512]
                # The largest shape is conditional on actual resource headroom.
                for size in batches:
                    row={'mode':mode,'precision':precision,'batch_size':size,'status':'running'}
                    guard=ResourceGuard('cuda');warm=BalancedSampler(panel,'2022-12-30',6026)
                    sampler=BalancedSampler(panel,'2022-12-30',2026);opt=optimizer_for(model,config,'cuda')
                    try:
                        model.train();torch.cuda.reset_peak_memory_stats()
                        for _ in range(3):
                            stocks,origins=warm.sample(size);batch=panel.windows(stocks,origins,config,scalers,device='cuda')
                            batch['experts']=primitive_experts(batch);train_batch(model,opt,batch,config)
                        guard.check();torch.cuda.synchronize();start=time.perf_counter();cpu_start=time.process_time();count=0
                        losses=[];updates=max(1,examples//size)
                        for step in range(updates):
                            stocks,origins=sampler.sample(size)
                            batch=panel.windows(stocks,origins,config,scalers,device='cuda');batch['experts']=primitive_experts(batch)
                            losses.append(train_batch(model,opt,batch,config));count+=size
                            if step%8==0:guard.check();usage.record()
                        torch.cuda.synchronize();elapsed=time.perf_counter()-start
                        row.update(status='passed',examples=count,updates=updates,seconds=elapsed,
                            examples_per_second=count/elapsed,seconds_per_update=elapsed/updates,
                            process_CPU_seconds=time.process_time()-cpu_start,resources=guard.check(),
                            first_training_loss=losses[0],last_training_loss=losses[-1],gpu=gpu_telemetry(),
                            includes_sampling_windows_transfer_backward_optimizer=True)
                        if size==512 and row['resources']['peak_device_used_vram_bytes']<.7*torch.cuda.get_device_properties(0).total_memory:
                            batches.append(1024)
                    except (torch.OutOfMemoryError,MemoryError) as exc:
                        row.update(status='resource_rejected',error_type=type(exc).__name__,error=str(exc),
                            resources={'peak_rss_bytes':guard.peak_rss,
                                       'minimum_available_ram_bytes':guard.minimum_available,
                                       'peak_device_used_vram_bytes':guard.peak_vram})
                    except BaseException as exc:
                        row.update(status='failed',error_type=type(exc).__name__,error=str(exc))
                        raise
                    finally:
                        report['measurements'].append(row);write_json(output,report)
                        if 'batch' in locals():del batch
                        del opt;gc.collect();torch.cuda.empty_cache()
                    if row['status']!='passed':break
                del model;gc.collect();torch.cuda.empty_cache()
    recommendations={}
    for mode in ('frozen','lora'):
        candidates=[r for r in report['measurements'] if r['mode']==mode and r['status']=='passed']
        if not candidates:raise RuntimeError('No resource-admitted throughput measurement for '+mode)
        best=max(r['examples_per_second'] for r in candidates)
        candidates=[r for r in candidates if r['examples_per_second']>=.95*best]
        chosen=min(candidates,key=lambda r:(r['batch_size'],r['seconds']))
        recommendations[mode]={key:chosen[key] for key in ('batch_size','precision','examples_per_second','seconds_per_update')}
    report.update(completed_at_utc=utc_now(),recommendations=recommendations,
                  choice_policy='smallest admitted batch within 5% of best end-to-end throughput; final maximum-shape admission still required')
    write_json(output,report);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--examples',type=int,default=8192);args=parser.parse_args()
    if args.examples<8192 or args.examples%1024:parser.error('Use at least 8192 examples, divisible by 1024')
    print(run(args.examples)['recommendations'])
