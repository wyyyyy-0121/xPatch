"""Apply exactly the original inner-stop scalar to every paired seed."""
from common import *
import os,time
import numpy as np
import torch
from engine import Data
from networks import create
from multiscale.bars import targets


def main():
    assert len(os.sched_getaffinity(0))<=2 and os.environ['CUDA_VISIBLE_DEVICES']=='-1'
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    manifest=read(BASE/'protocol/JOBS.json')
    assert read(BASE/'audit/COMPLETE.json')['cells']==180
    rules={'code_sha256':sha(__file__),'jobs_sha256':sha(BASE/'protocol/JOBS.json'),
           'formula':'alpha=clip(sum(q*r)/sum(q*q),0,1), exact original stop endpoints',
           'baseline':'zero returns or observed current close','outer_fitting':False,'Q3_values_read':False}
    protocol=BASE/'protocol/CALIBRATION.json'
    if protocol.exists():assert read(protocol)['rules']==rules
    else:atomic(protocol,{'rules':rules,'new_aggregate_results_seen':False,'at_utc':now()})
    for cfg in manifest['jobs']:
        out=BASE/'calibration'/cfg['id']
        if (out/'DONE.json').exists():continue
        if time.time()>=manifest['deadline_epoch']:
            atomic(BASE/'audit/CALIBRATION_PAUSED.json',{'status':'paused','at_utc':now()});return
        src=folder(cfg,manifest);done=read(src/'DONE.json')
        assert done['job']==cfg and sha(src/'best_trained.pt')==done['checkpoint_sha256']
        saved=torch.load(src/'best_trained.pt',map_location='cpu',weights_only=True)
        data=Data(cfg);net=create(cfg,data.scaler);net.load_state_dict(saved['model']);net.eval()
        num=den=0.;si=data.splits['stop']
        with torch.inference_mode():
            for start in range(0,len(si),128):
                pos=np.arange(start,min(start+128,len(si)));args,_=data.batch('stop',pos);idx=si[pos]
                pred=data.invert(net(*args).numpy(),idx)[:,-1];base=data.baseline(idx)[:,-1]
                y=targets(data.bars,idx,cfg['horizon'],cfg['target'])[:,-1]
                q,r=pred-base,y-base
                num+=float(np.dot(q,r));den+=float(np.dot(q,q))
        alpha=float(np.clip(num/den,0,1)) if den else 0.
        assert sha(src/'predictions.npz')==done['prediction_sha256']
        with np.load(src/'predictions.npz',allow_pickle=False) as a:
            origins=a['origins'];trained=a['trained'];coupled=a['coupled']
        assert np.array_equal(origins,data.splits['validation'])
        base=data.baseline(origins);prediction=base+alpha*(trained-base)
        npz(out/'predictions.npz',origins=origins,prediction=prediction)
        prior_cal=V3/'calibration_candidate'/cfg['id']
        exact_reuse=None
        if (prior_cal/'DONE.json').exists():
            with np.load(prior_cal/'predictions.npz') as a:old=a['prediction']
            exact_reuse=alpha==read(prior_cal/'DONE.json')['alpha'] and np.array_equal(old,prediction)
            assert exact_reuse
        atomic(out/'DONE.json',{'status':'completed','job':cfg['id'],'alpha':alpha,
            'source_checkpoint_sha256':done['checkpoint_sha256'],'calibration_code_sha256':sha(__file__),
            'prediction_sha256':sha(out/'predictions.npz'),'v3_reuse_exact':exact_reuse,
            'inner_stop_only':True,'Q3_values_read':False,'at_utc':now()})
    atomic(BASE/'audit/CALIBRATION_COMPLETE.json',{'status':'completed','cells':180,'Q3_values_read':False,'at_utc':now()})
    print('All180 exact inner calibrations complete.',flush=True)


if __name__=='__main__':main()
