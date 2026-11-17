"""One fixed level deletion. Permitted saved predictions only; no fitting."""
import io
import os
import signal
import sys
import time
import zipfile
from runtime import BASE, MEMBERS, require, sha, read, write, setup, load_protocol, guard, decision

PAUSE=False
def request_pause(signum, frame):
    global PAUSE
    PAUSE=True

def calculate(arrays, spec):
    import numpy as np
    n,h=spec['count'],spec['horizon']
    origins=arrays['origins']
    require(origins.dtype==np.dtype('int64') and origins.shape==(n,) and np.all(np.diff(origins)>0), 'Origin dtype/order')
    require(sha_bytes(np.asarray(origins,dtype='<i8').tobytes())==spec['tail_sha256'], 'Tail origin identity')
    y=arrays['labels']; p=arrays['same_information_Ridge']; bfull=arrays['prefix_update_mean']
    for name in MEMBERS[1:]:
        a=arrays[name]
        require(a.dtype==np.dtype('float64') and a.shape==(n,h) and np.isfinite(a).all(), 'Array identity '+name)
    require(np.count_nonzero(arrays['zero'])==0, 'Saved zero not zero')
    require(np.array_equal(bfull,np.broadcast_to(bfull[0],(n,h))), 'Prefix mean not constant')
    b=bfull[0].copy(); g=p-b
    forecasts={key:arrays[key] for key in MEMBERS[2:]}
    forecasts.update(g=g,half_Ridge=.5*p,half_g=.5*g)
    per_h={key:np.mean(np.square(y-value),axis=0,dtype=np.float64) for key,value in forecasts.items()}
    metrics={key:{'endpoint_MSE':float(value[-1]),'path_MSE':float(value.mean())} for key,value in per_h.items()}
    b2=b*b; cross=2*b*np.mean(y-g,axis=0,dtype=np.float64)
    observed=per_h['same_information_Ridge']-per_h['g']
    scale=np.maximum.reduce([np.mean(y*y,axis=0),np.mean(p*p,axis=0),np.mean(g*g,axis=0),b2,
                             np.full(h,np.finfo(np.float64).tiny)])
    tol=256*np.finfo(np.float64).eps*scale
    residual=observed-(b2-cross)
    require(np.all(np.abs(residual)<=tol), 'New raw-unit paired identity failed')
    # Roundoff-qualified equality of the same centered prediction; no tail recentering.
    centered_error=np.abs((g+b)-p)
    require(np.all(centered_error<=8*np.finfo(np.float64).eps*np.maximum(np.abs(p)+np.abs(b),np.finfo(float).tiny)), 'p/g mapping')
    rows=[{'horizon':i+1,'b':float(b[i]),'b_square':float(b2[i]),'twice_b_residual_mean':float(cross[i]),
           'L_p_minus_L_g':float(observed[i]),'identity_residual':float(residual[i]),'identity_tolerance':float(tol[i]),
           'MSE':{key:float(value[i]) for key,value in per_h.items()}} for i in range(h)]
    return {'state':spec['state'],'count':n,'horizon':h,'metrics':metrics,'all_horizons':rows,
            'tail_sha256':spec['tail_sha256'],'maximum_identity_residual':float(np.max(np.abs(residual))),
            'identity_qualified':True,'same_slopes_by_construction':True}

def sha_bytes(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()

def main():
    setup(24)
    signal.signal(signal.SIGUSR1,request_pause)
    signal.signal(signal.SIGTERM,request_pause)
    import numpy as np
    p=load_protocol(); spec=read(BASE/'DESIGN_SPEC.json'); output=BASE/'run'
    identity=sha(BASE/'EXECUTION_PROTOCOL.json')
    checkpoint=output/'CHECKPOINT.json'
    saved=read(checkpoint) if checkpoint.exists() else {'protocol_sha256':identity,'completed':[],'worker_pids':[]}
    require(saved['protocol_sha256']==identity, 'Checkpoint protocol changed')
    require([item['state'] for item in saved['completed']]==[item['state'] for item in spec['states'][:len(saved['completed'])]], 'Checkpoint state prefix changed')
    for item in saved['completed']:
        require(sha(output/(item['state']+'.json'))==item['sha256'],'Resumed state artifact changed')
    saved['worker_pids'].append(os.getpid())
    write(checkpoint,saved)
    for entry in spec['states'][len(saved['completed']):]:
        guard(p,30)
        require(sha(entry['path'])==entry['sha256'], 'Saved archive identity changed')
        with zipfile.ZipFile(entry['path']) as z:
            require(sorted(z.namelist())==sorted(key+'.npy' for key in MEMBERS), 'Unexpected/duplicate saved member')
            arrays={key:np.load(io.BytesIO(z.read(key+'.npy')),allow_pickle=False) for key in MEMBERS}
        result=calculate(arrays,entry)
        del arrays
        write(output/(entry['state']+'.json'),result)
        saved['completed'].append({'state':entry['state'],'sha256':sha(output/(entry['state']+'.json'))})
        write(checkpoint,saved)
        guard(p,20)
        if len(saved['completed'])==1:
            write(output/'FIRST_COMMIT.json',{'pid':os.getpid(),'checkpoint_sha256':sha(checkpoint)})
            # Explicit controller rendezvous permits a real external signal before next state.
            while not PAUSE and not (output/'CONTINUE').exists():
                guard(p,30); time.sleep(.01)
        if PAUSE:
            write(output/'PAUSED.json',{'pid':os.getpid(),'protocol_sha256':identity,
                                      'checkpoint_sha256':sha(checkpoint),'completed_states':len(saved['completed'])})
            return 75
    guard(p,20)
    rows=[]
    for item in saved['completed']:
        file=output/(item['state']+'.json')
        require(sha(file)==item['sha256'],'Committed state changed')
        rows.append(read(file))
    write(output/'RESULTS.json',{'protocol_sha256':identity,'records':rows,'training_updates':0,
                               'model_inference':0,'Q3_read':False,'GPU':False,'CPU':[24]})
    write(output/'DECISION.json',decision(rows))
    write(output/'WORKER_DONE.json',{'pid':os.getpid(),'completed_states':4,'protocol_sha256':identity})
    return 0

if __name__=='__main__':
    try: sys.exit(main())
    except Exception as exc:
        write(BASE/'run'/'WORKER_FAILURE.json',{'pid':os.getpid(),'error':repr(exc)})
        raise
