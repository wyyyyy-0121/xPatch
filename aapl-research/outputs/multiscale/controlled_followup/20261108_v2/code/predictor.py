"""Input-only portable predictors. This module does not retrieve data or load labels."""
from pathlib import Path
import sys,numpy as np,torch

CODE=Path(__file__).resolve().parent
sys.path.insert(0,str(CODE/'reference_source'))
sys.path.insert(0,str(CODE))
from models import create

class Predictor:
    def __init__(self,checkpoint):
        self.path=Path(checkpoint)
        self.saved=torch.load(self.path,map_location='cpu',weights_only=True)
        self.cfg=self.saved['cfg'];self.scaler=self.saved['scaler']
        self.ym=np.asarray(self.scaler['y_mean']);self.ys=np.asarray(self.scaler['y_std'])
        self.net=create(self.cfg,-self.ym/self.ys);self.net.load_state_dict(self.saved['model']);self.net.eval()
    def predict_prepared(self,x,anchor):
        """x is demeaned float32 log history for tiny, log-relative float64 for original."""
        x=np.asarray(x);anchor=np.asarray(anchor,dtype='float64').reshape(-1,1)
        assert x.ndim==2 and x.shape==(len(anchor),self.cfg['lookback'])
        if not np.isfinite(x).all() or not np.isfinite(anchor).all() or not (anchor>0).all():raise ValueError('Invalid inputs')
        x=((x-self.scaler['x_mean'])/self.scaler['x_std']).astype('float32')
        if self.cfg['family']=='tiny':args=(torch.from_numpy(x),)
        else:
            a=((anchor-self.scaler['anchor_mean'])/self.scaler['anchor_std']).astype('float32')
            args=(torch.from_numpy(x[:,:,None]),torch.from_numpy(a))
        with torch.inference_mode():p=self.net(*args).numpy()*self.ys+self.ym
        if self.cfg['family']=='tiny' and self.cfg['target']=='future_close':p+=anchor
        return p
    def predict(self,closes):
        """Completed chronological closes, all on share basis known at each origin."""
        c=np.asarray(closes,dtype='float64')
        if c.ndim!=2 or c.shape[1]!=self.cfg['lookback'] or not np.isfinite(c).all() or not (c>0).all():
            raise ValueError('Supply positive finite batch x lookback closes')
        logs=np.log(c);x=logs-logs[:,-1,None]
        if self.cfg['family']=='tiny':x=(x-x.mean(1,keepdims=True)).astype('float32')
        return self.predict_prepared(x,c[:,-1])

