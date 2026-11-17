"""Describe saved scaling functions; never refit or interpret them as probabilities."""
from common import *
import torch
from models import create


def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    decision=read(BASE/'protocol/STAGE_C_DECISION.json');cache=cache_arrays();rows=[]
    for fold in range(1,6):
        parts=np.load(BASE/'cache'/f'fold_{fold}.npz');rr=rows_for(parts['validation'])
        high=cache['volatility'][rr]>np.median(cache['volatility'][rows_for(parts['fit'])])
        y=cache['actual'][rr,-1]
        for kind in decision['formal_kinds']:
            if kind=='lstm':continue
            for seed in (2026,2027,2028):
                folder=BASE/'runs'/f'formal_{kind}_f{fold}_s{seed}'
                saved=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
                model=create(kind,-np.asarray(saved['scaler']['y_mean'])/np.asarray(saved['scaler']['y_std']))
                model.load_state_dict(saved['model']);model.eval();scale=[]
                if kind=='short_state':
                    st=saved['feature_stats']['gate']
                    for i in range(0,len(rr),2048):
                        gate=((cache['gate'][rr[i:i+2048]]-np.asarray(st['mean']))/np.asarray(st['std'])).astype('float32')
                        with torch.inference_mode():scale.append(torch.sigmoid(model.gating(torch.from_numpy(gate)))[:,1].numpy())
                    lam=np.concatenate(scale)
                elif kind=='short_const':lam=np.full(len(rr),float(torch.sigmoid(model.constants.detach()[1])))
                else:lam=np.ones(len(rr))
                with np.load(folder/'prediction.npz') as a:p=a['prediction'][:,-1]
                zero=float(np.mean(y*y))
                rows.append({'fold':fold,'kind':kind,'seed':seed,'best_epoch':saved['epoch'],
                    'lambda_mean':float(lam.mean()),'lambda_quantiles':np.quantile(lam,[.05,.5,.95]).tolist(),
                    'low_vol_lambda_mean':float(lam[~high].mean()),'high_vol_lambda_mean':float(lam[high].mean()),
                    'prediction_energy_ratio':float(np.mean(p*p)/zero),
                    'alignment_ratio':float(2*np.mean(p*y)/zero),
                    'prediction_mean':float(p.mean()),'realized_mean':float(y.mean())})
    atomic(BASE/'analysis/SCALE_DESCRIPTIONS.json',{'status':'completed','rows':rows,
        'scope':'saved-function descriptions, no causal or reliability calibration claim',
        'unused_parameters':'single-stream state model retains the unused first gate output and its 9 parameters; counts report requires_grad parameters',
        'Q3_values_read':False})


if __name__=='__main__':main()
