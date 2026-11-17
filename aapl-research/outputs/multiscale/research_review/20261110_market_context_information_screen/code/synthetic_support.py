"""Artificial feature/label source for lifecycle acceptance; no external sources."""
import numpy as np

class ArtificialSource:
    def __init__(self,rules,guard):
        self.guard=guard;self.phase='no_labels';self.states=[];self.label_calls=[]
        for task in rules['tasks']:
            for fold in task['folds']:self.states.append(ArtificialState(self,task,fold))
    def allow_prefix_labels(self):self.phase='prefix'
    def allow_tail_labels(self):self.phase='tail'

class ArtificialState:
    def __init__(self,source,task,fold):
        self.source=source;self.task=task['name'];self.frequency=task['frequency'];self.L=task['L'];self.H=task['H'];self.fold=fold
        self.key=self.task+'_fold_'+str(fold)
        n=2200 if self.frequency=='minute' else 300
        self.P=np.arange(self.L+1,self.L+1+n,dtype=np.int64)
        self.T=np.arange(self.P[-1]+self.H+2,self.P[-1]+self.H+66,dtype=np.int64)
        self.mVP=self.P%11!=0;self.mVT=self.T%11!=0
        self.mSP=self.P%13!=0;self.mST=self.T%13!=0
        self.PV=self.P[self.mVP];self.complete_PV=self.mSP[self.mVP]
        rng=np.random.default_rng(20261110+fold+(100 if self.frequency=='hour' else 0))
        count=int(self.T[-1])+1
        self.a=rng.normal(size=(count,self.L+18));self.a[:,self.L-1]=0.;self.a[:,2]=self.a[:,1]
        self.u=rng.normal(size=count);self.spy=rng.normal(0,.001,size=(count,self.L-1))
    def A(self,origins):self.source.guard();return self.a[origins].copy()
    def B(self,origins):
        if not (origins%11!=0).all():raise ValueError('artificial_B_missing_volume')
        a=self.A(origins);u=self.u[origins];return np.column_stack((a,u,a[:,self.L+1]*u))
    def z(self,origins):
        if not (origins%13!=0).all():raise ValueError('artificial_SPY_missing_window')
        return self.spy[origins].copy()
    def plus(self,origins,complete,mean):
        z=np.broadcast_to(mean,(len(origins),self.L-1)).copy()
        if complete.any():z[complete]=self.z(origins[complete])
        return np.column_stack((self.B(origins),z))
    def labels(self,origins,phase):
        if self.source.phase!=phase:raise ValueError('artificial_label_phase_guard')
        domain=self.P if phase=='prefix' else self.T;positions=np.searchsorted(domain,origins)
        if (positions>=len(domain)).any() or not np.array_equal(domain[positions],origins):raise ValueError('artificial_label_domain_guard')
        self.source.label_calls.append((self.key,phase,len(origins)))
        return np.asarray(np.sin(origins[:,None]*.017+np.arange(self.H)[None,:]*.03)*.001+
            self.a[origins,0,None]*.0001+self.spy[origins,0,None]*.1,dtype=np.float64)
