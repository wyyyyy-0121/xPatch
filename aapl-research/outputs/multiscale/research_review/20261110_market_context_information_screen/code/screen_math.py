"""One fixed FP64 Ridge dictionary and full-domain routes; no file or label I/O."""
import numpy as np
PENALTY=.1
STD_FLOOR=1e-12
METHODS=('A_full','base','plus','zero','prefix_mean')
EXPERTS=('A_full','B_valid','Plus_valid')

def empty_stats(columns,horizon):
    return {'n':0,'sx':np.zeros(columns),'sy':np.zeros(horizon),
            'xx':np.zeros((columns,columns)),'xy':np.zeros((columns,horizon))}

def update(stats,x,y):
    if (x.ndim!=2 or y.ndim!=2 or not len(x) or len(x)!=len(y)
            or x.dtype!=np.float64 or y.dtype!=np.float64
            or x.shape[1]!=len(stats['sx']) or y.shape[1]!=len(stats['sy'])
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError('fixed_Ridge_invalid_batch')
    stats['n']+=len(x); stats['sx']+=x.sum(axis=0);stats['sy']+=y.sum(axis=0)
    stats['xx']+=x.T@x;stats['xy']+=x.T@y

def solve(stats):
    n=int(stats['n'])
    if n<2:raise ValueError('fixed_Ridge_insufficient_pool')
    mx=stats['sx']/n;my=stats['sy']/n
    cov=stats['xx']/n-np.outer(mx,mx)
    margin=np.maximum(np.maximum(np.abs(np.diag(stats['xx'])/n),mx*mx),1.)
    if (np.diag(cov)<-1e-10*margin).any():raise ValueError('fixed_Ridge_negative_variance')
    sd=np.sqrt(np.maximum(np.diag(cov),0));sd=np.where(sd>STD_FLOOR,sd,1.)
    gram=cov/sd[:,None]/sd[None,:];gram=(gram+gram.T)/2
    right=(stats['xy']/n-np.outer(mx,my))/sd[:,None]
    if not all(np.isfinite(a).all() for a in (mx,my,sd,gram,right)):
        raise ValueError('fixed_Ridge_nonfinite_moments')
    matrix=gram+PENALTY*np.eye(len(mx));coef=np.linalg.solve(matrix,right)
    denominator=np.linalg.norm(matrix,np.inf)*np.linalg.norm(coef,np.inf)+np.linalg.norm(right,np.inf)
    residual=float(np.linalg.norm(matrix@coef-right,np.inf)/max(float(denominator),np.finfo(float).tiny))
    if not np.isfinite(coef).all() or not np.isfinite(residual) or residual>1e-10:
        raise ValueError('fixed_Ridge_stationarity_failure')
    return {'coef':coef,'x_mean':mx,'x_std':sd,'y_mean':my,
            'stationarity':np.asarray(residual),'n_train':np.asarray(n,dtype=np.int64)}

def predict(model,x):
    if x.ndim!=2 or x.shape[1]!=len(model['x_mean']) or not np.isfinite(x).all():
        raise ValueError('fixed_Ridge_invalid_prediction_features')
    result=model['y_mean']+((x-model['x_mean'])/model['x_std'])@model['coef']
    if not np.isfinite(result).all():raise ValueError('fixed_Ridge_nonfinite_prediction')
    return result

def route(a,b_selected,plus_selected,mV,mS):
    if (mV.dtype!=np.bool_ or mS.dtype!=np.bool_ or mV.shape!=(len(a),)
            or mS.shape!=mV.shape or b_selected.shape!=(int(mV.sum()),a.shape[1])
            or plus_selected.shape!=(int((mV&mS).sum()),a.shape[1])):
        raise ValueError('route_domain_mismatch')
    base=a.copy();base[mV]=b_selected
    plus=base.copy();plus[mV&mS]=plus_selected
    # Assignment copies exact FP64 payloads, including negative zeros.
    if not np.array_equal(plus[~(mV&mS)].view(np.uint64),base[~(mV&mS)].view(np.uint64)):
        raise ValueError('fallback_not_bitwise_identical')
    return base,plus

def score(y,predictions,mV,mS):
    if set(predictions)!=set(METHODS) or y.ndim!=2 or not len(y) or not np.isfinite(y).all():
        raise ValueError('score_complete_full_domain_required')
    metrics={};errors={}
    for name in METHODS:
        pred=predictions[name]
        if pred.shape!=y.shape or not np.isfinite(pred).all():raise ValueError('score_nonfinite_or_misaligned')
        error=np.square(pred-y);errors[name]=error
        metrics[name]={'endpoint_mse':float(error[:,-1].mean()),'path_mse':float(error.mean())}
    if not all(np.isfinite(v) and v>=0 for row in metrics.values() for v in row.values()):
        raise ValueError('score_unqualified_metrics')
    plus=metrics['plus']
    gates={'endpoint_below_'+m:bool(plus['endpoint_mse']<metrics[m]['endpoint_mse'])
           for m in ('A_full','base','zero','prefix_mean')}
    gates.update({'path_not_above_'+m:bool(plus['path_mse']<=metrics[m]['path_mse']) for m in ('A_full','base')})
    groups={}
    for v in (False,True):
        for s in (False,True):
            mask=(mV==v)&(mS==s)
            groups['mV'+str(int(v))+'_mS'+str(int(s))]={'count':int(mask.sum()),
                'error_sums':{m:{'endpoint':float(errors[m][mask,-1].sum()),
                                 'path':float(errors[m][mask].sum())} for m in METHODS}}
    return metrics,gates,groups

def aggregate(rows):
    primaries={}
    if len(rows)!=4 or len({r['state'] for r in rows})!=4:raise ValueError('exact_four_state_results_required')
    for task in sorted({r['task'] for r in rows}):
        selected=[r for r in rows if r['task']==task]
        if sorted(r['fold'] for r in selected)!=[1,2]:raise ValueError('primary_requires_two_fixed_folds')
        plus=sum(r['metrics']['plus']['endpoint_mse'] for r in selected)
        zero=sum(r['metrics']['zero']['endpoint_mse'] for r in selected)
        if not np.isfinite(plus) or not np.isfinite(zero) or zero<=0:raise ValueError('primary_zero_denominator_unqualified')
        gain=1-plus/zero
        if not np.isfinite(gain):raise ValueError('primary_nonfinite_gain')
        primaries[task]={'gain_overzero':float(gain),'passed_1percent':bool(gain>=.01)}
    if len(primaries)!=2:raise ValueError('exact_two_primary_results_required')
    return primaries,bool(all(r['state_pass'] for r in rows) and all(p['passed_1percent'] for p in primaries.values()))
