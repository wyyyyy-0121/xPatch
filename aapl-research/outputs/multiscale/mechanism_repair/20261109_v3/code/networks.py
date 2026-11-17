"""Keep the original architectures and change only the registered head initialization."""
from study import *
import torch
from multiscale.models import make_model

def create(cfg, scaler):
    if cfg['model']=='tiny':
        return old_models().TinyPath(cfg['horizon'],'long',
                -np.asarray(scaler['y_mean'])/np.asarray(scaler['y_std']))
    net=make_model(cfg['model'],cfg['lookback'],cfg['horizon'])
    if cfg['initialization']=='random':return net
    ym=np.asarray(scaler['y_mean']);ys=np.asarray(scaler['y_std'])
    bias=-ym/ys;weight=np.zeros_like(ym)
    if cfg['output']=='price':
        assert cfg['anchor']=='observed'
        bias=(scaler['anchor_mean']-ym)/ys;weight=scaler['anchor_std']/ys
    with torch.no_grad():
        if cfg['model']=='lstm':
            net.head.weight.zero_();net.head.bias.copy_(torch.as_tensor(bias,dtype=torch.float32))
            net.head.weight[:,-1].copy_(torch.as_tensor(weight,dtype=torch.float32))
        else:
            net.backbone.fc8.weight.zero_();net.backbone.fc8.bias.zero_()
            for j in range(net.native_horizon):
                net.anchor_head.weight[j,0]=float(weight[min(j,len(weight)-1)])
                net.anchor_head.bias[j]=float(bias[min(j,len(bias)-1)])
    return net

def objective(p,y,kind):
    v=(p-y).square()
    return v.mean() if kind=='path' else .5*v.mean()+.5*v[:,-1].mean()
