"""Twelve prescribed simple controls, fit on past OOF experts only."""
import numpy as np
from scipy.optimize import lsq_linear, minimize


def fit(experts,target,config,weights=None):
    x=np.asarray(experts,dtype=np.float64);y=np.asarray(target,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=4 or len(x)!=len(y):raise ValueError('Expected four complete OOF experts')
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Incomplete OOF training rows')
    w=np.ones(len(y))/len(y) if weights is None else np.asarray(weights)/np.sum(weights)
    unit=max(float(np.sqrt(np.sum(w*y*y))),1e-6);x=x/unit;y=y/unit
    alpha=config.get('alpha',0.)
    if config['family']=='nonnegative_ridge':
        a=np.vstack([x*np.sqrt(w[:,None]),np.eye(4)*np.sqrt(alpha)])
        b=np.r_[y*np.sqrt(w),np.zeros(4)]
        result=lsq_linear(a,b,bounds=(0.,np.inf),tol=1e-12)
        if not result.success:raise RuntimeError('Nonnegative fit failed')
        coefficient=result.x
    elif config['family']=='convex':
        def objective(c):return np.sum(w*(x@c-y)**2)+alpha*np.sum(c*c)
        def gradient(c):return 2*x.T@(w*(x@c-y))+2*alpha*c
        result=minimize(objective,np.zeros(4),jac=gradient,bounds=[(0.,1.)]*4,
                        constraints=[{'type':'ineq','fun':lambda c:1.-c.sum(),'jac':lambda c:-np.ones(4)}],
                        method='SLSQP',options={'maxiter':1000,'ftol':1e-12})
        if not result.success:raise RuntimeError('Convex fit failed: '+result.message)
        coefficient=np.clip(result.x,0,1)
        if coefficient.sum()>1:coefficient/=coefficient.sum()
    elif config['family']=='shrink_best':
        best=int(np.argmin(np.sum(w[:,None]*(x-y[:,None])**2,axis=0)))
        coefficient=np.zeros(4);coefficient[best]=config['factor']
    else:raise ValueError(config['family'])
    return {'config':config,'coefficient':coefficient.tolist(),
            'zero_expert_weight':max(0.,1.-float(coefficient.sum())) if config['family']=='convex' else None}


def predict(model,experts):return np.asarray(experts,dtype=np.float64)@np.asarray(model['coefficient'])
