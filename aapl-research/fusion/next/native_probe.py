"""Training-only native-model batch and CPU-prefetch throughput comparison."""
import gc
import torch

from .accounting import Usage
from .common import OUT,source_identity,utc_now,write_json
from .hardware import runtime,ResourceGuard,gpu_telemetry
from .panel import Panel,CoverageSampler
from .expert_models import NativeExpert
from .performance import BASE,end_to_end
from .checkpoint import changing_state,capture_rng,restore_rng


def run():
    if runtime(2026)!='cuda':raise RuntimeError('Native throughput measurement requires the cloud GPU')
    if (OUT/'search_manifest.json').exists():raise ValueError('Batch choices must precede formal scores')
    panel=Panel.load();scalers=panel.scaler('2022-12-30');rows=[]
    report={'source_identity':source_identity(),'started_at_utc':utc_now(),'training_only':True,
            'official_scores_read':False,'measurements':rows}
    with Usage('performance','native batch scaling and bounded CPU prefetch') as usage:
        for family in ('xpatch','patchtst'):
            for size in (256,512,1024):
                config={**BASE,'family':family,'patch':8,'stride':4,'d_model':64,'batch_size':size,
                    'precision':'bf16','fused_optimizer':True,'pin_memory':True,
                    'raw_loss_unit':scalers['AAPL']['target_std']}
                model=NativeExpert(config,12*(1+len(panel.background)),len(panel.stocks)).to('cuda')
                sampler=CoverageSampler(panel,'2022-12-30',2026)
                ids=[sampler.sample(size) for _ in range(65536//size)]
                initial=changing_state(model);rng=capture_rng();row={'family':family,'batch_size':size}
                guard=ResourceGuard('cuda');torch.cuda.reset_peak_memory_stats()
                try:
                    end_to_end(model,panel,ids[:3],scalers,{**config,'prefetch_batches':2})
                    for depth in (0,2):
                        model.load_state_dict(initial,strict=False);restore_rng(rng)
                        result=end_to_end(model,panel,ids,scalers,{**config,'prefetch_batches':depth})
                        row['prefetch_'+str(depth)]={**result,'examples_per_second':65536/result['seconds']}
                        guard.check();usage.record()
                    row.update(status='passed',prefetch_losses_equal=row['prefetch_0']['losses']==row['prefetch_2']['losses'],
                               resources=guard.check(),gpu=gpu_telemetry())
                    if not row['prefetch_losses_equal']:raise ValueError('Native prefetch changed the training trajectory')
                except (MemoryError,torch.OutOfMemoryError) as error:
                    row.update(status='resource_rejected',error=str(error))
                finally:
                    rows.append(row);write_json(OUT/'native_scaling.json',report)
                    del model,initial;gc.collect();torch.cuda.empty_cache()
                if row['status']!='passed':break
    choices={}
    for family in ('xpatch','patchtst'):
        admitted=[r for r in rows if r['family']==family and r['status']=='passed']
        best=max(r['prefetch_2']['examples_per_second'] for r in admitted)
        pick=min((r for r in admitted if r['prefetch_2']['examples_per_second']>=.95*best),key=lambda r:r['batch_size'])
        choices[family]={'batch_size':pick['batch_size'],'examples_per_second':pick['prefetch_2']['examples_per_second'],
            'prefetch_gain':pick['prefetch_2']['examples_per_second']/pick['prefetch_0']['examples_per_second']-1}
    report.update(completed_at_utc=utc_now(),choices=choices)
    write_json(OUT/'native_scaling.json',report);print(choices)


if __name__=='__main__':run()
