"""Replay every newly trained original-style predictor from saved weights and past inputs."""
from common import *
import torch,time
from models import create
from multiscale.features import CloseWindows,Scaler


def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    bars,_,folds,admission=bars_and_folds();windows=CloseWindows(bars,480,admission['split_actions'])
    jobs=[j for j in read(BASE/'protocol/JOBS.json')['jobs'] if j['phase']=='probe' or j['kind']=='lstm']
    records=[]
    for job in jobs:
        folder=BASE/'runs'/job['id'];done=read(folder/'DONE.json')
        saved=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
        assert sha(folder/'best.pt')==done['checkpoint_sha256']
        model=create(job['kind']);model.load_state_dict(saved['model']);model.eval();scaler=Scaler(**saved['scaler'])
        origins=folds[job['fold']-1]['validation']
        with np.load(folder/'prediction.npz',allow_pickle=False) as a:
            assert np.array_equal(a['origins'],origins);expected=a['prediction']
        exact=True;maximum=0.;start=time.time()
        with torch.inference_mode():
            for i in range(0,len(origins),512):
                x,anchor=windows.arrays(origins[i:i+512]);x,anchor=scaler.transform_x(x,anchor)
                p=scaler.invert_y(model(torch.from_numpy(x[:,:,None]),torch.from_numpy(anchor)).numpy())
                e=expected[i:i+len(p)];maximum=max(maximum,float(np.abs(p-e).max()));exact &= np.array_equal(p,e)
                assert np.allclose(p,e,rtol=1e-6,atol=1e-10),(job['id'],maximum)
        records.append({'job':job['id'],'origins':len(origins),'all_steps':60,
            'exact_array_match':bool(exact),'max_absolute_difference':maximum,'target_files_read':False,
            'checkpoint_sha256':done['checkpoint_sha256'],'prediction_sha256':done['prediction_sha256'],
            'wall_seconds':time.time()-start})
        atomic(BASE/'audit/REFERENCE_REPLAY_PROGRESS.json',{'completed':len(records),'planned':len(jobs),'last':records[-1]})
        print(json.dumps({'replayed':job['id'],'exact':bool(exact)}),flush=True)
    atomic(BASE/'audit/REFERENCE_REPLAY.json',{'status':'passed','records':records,
        'predictors':len(records),'total_origin_predictor_pairs':sum(x['origins'] for x in records),
        'all_exact':all(x['exact_array_match'] for x in records),'no_target_files_read':True,'Q3_values_read':False})


if __name__=='__main__':main()
