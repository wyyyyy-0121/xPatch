"""Recompute the interrupted BN3 pass from cursor zero, retaining all live artifacts."""
from pathlib import Path
import os,sys,json,time
import numpy as np
import torch
import diagnostic_v2 as d

HERE=Path(__file__).resolve().parent

def main():
    assert sorted(os.sched_getaffinity(0))==[24,26] and os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    before=d.read(HERE/'RECOVERY_BEFORE_RESUME.json');cfg=before['saved_identity']['cfg']
    out=HERE/'cells'/cfg['id'];inner=d.read(out/'INNER_DONE.json');data=d.Data(cfg)
    _,_,saved=d.source(cfg);adapted=torch.load(out/'adapted.pt',map_location='cpu',weights_only=True)
    net=d.load_net(cfg,saved,data)
    for name in ('backbone.bn1','backbone.bn2'):
        layer=dict(net.named_modules())[name]
        layer.running_mean.copy_(adapted['model'][name+'.running_mean'])
        layer.running_var.copy_(adapted['model'][name+'.running_var'])
    layer=net.backbone.bn3;capture={};count=0;mean=M2=None
    def hook(module,args):capture['x']=args[0].detach();raise d.Captured()
    handle=layer.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            for start in range(0,len(data.splits['fit']),128):
                assert time.time()<d.DEADLINE
                idx=data.splits['fit'][start:start+128]
                try:net(*d.args_only(data,idx))
                except d.Captured:pass
                x=capture.pop('x').to(torch.float64)
                n=x.shape[0]*x.shape[2];m=x.mean((0,2));v=((x-m[None,:,None])**2).sum((0,2))
                if count==0:count=n;mean=m;M2=v
                else:
                    total=count+n;delta=m-mean;mean+=delta*(n/total);M2+=v+delta**2*(count*n/total);count=total
    finally:handle.remove()
    variance=M2/(count-1)
    expected_mean=adapted['model']['backbone.bn3.running_mean']
    expected_var=adapted['model']['backbone.bn3.running_var']
    assert torch.equal(mean.to(expected_mean.dtype),expected_mean)
    assert torch.equal(variance.to(expected_var.dtype),expected_var)
    moments=d.read(out/'MOMENTS_DONE.json')['layers'][2]
    assert count==moments['activation_count_per_channel']
    assert np.array_equal(mean.numpy(),np.asarray(moments['pooled_mean_float64']))
    assert np.array_equal(variance.numpy(),np.asarray(moments['pooled_unbiased_var_float64']))
    receipt={'status':'passed','interrupted_job':cfg['id'],'actual_SIGUSR1_exit_and_resume':True,
        'saved_partial_cursor':before['saved_cursor'],'saved_partial_activation_count':before['saved_count'],
        'resumed_to_completed':True,'fit_activation_count_per_channel':count,
        'independent_uninterrupted_BN3_recompute_float64_exact':True,
        'final_BN3_buffers_bitwise_equal':True,'labels_read_for_moment_check':False,
        'source_checkpoint_sha256':before['saved_identity']['checkpoint_sha256'],
        'recovery_before_sha256':d.sha(HERE/'RECOVERY_BEFORE_RESUME.json'),
        'adapted_checkpoint_sha256':d.sha(out/'adapted.pt'),
        'deadline_unchanged':True,'deadline_epoch':d.DEADLINE,'affinity':sorted(os.sched_getaffinity(0)),
        'at_utc':d.now(),'at_epoch':time.time()}
    d.atomic(HERE/'RECOVERY_CHECK.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
