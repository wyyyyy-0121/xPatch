from pathlib import Path
import hashlib,json,os,sys,tempfile
import numpy as np

BASE=Path(__file__).resolve().parents[1]
ROOT=Path('/home/ayaka/wslcode/aapl-research')
SOURCE=ROOT/'outputs/multiscale/admission/alpaca_20261107T052228573242Z'
sys.path.insert(0,str(BASE/'code/reference_source'))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text())

def atomic(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',dir=p.parent,delete=False,encoding='utf8') as f:
        json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False,
                  default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())
        f.flush();os.fsync(f.fileno());tmp=f.name
    os.replace(tmp,p)

def bars_and_folds():
    from multiscale.matrix_executor import read_bars
    from multiscale.splits import rolling_folds
    from multiscale.development import validate_split,bars_identity
    from multiscale.training import digest_indices
    admission=read(SOURCE/'admission.json')
    bars=read_bars(SOURCE,admission,'minute')
    op=SOURCE/'origins_minute_h60.npy'
    assert sha(op)==admission['files'][op.name]
    origins=np.load(op,allow_pickle=False)
    assert str(bars.session_date[-1])<='2026-06-30'
    folds=rolling_folds(bars,origins,60)
    assert len(folds)==5
    manifest=BASE/'protocol/DATA.json'
    if manifest.exists():
        m=read(manifest);assert bars_identity(bars)==m['bars_sha256']
        for f in folds:
            for key,idx in zip(('train','fit','stop','validation'),validate_split(bars,f,60)):
                assert digest_indices(idx)==m['folds'][str(f['fold'])]['split_sha256'][key]
    return bars,origins,folds,admission

def small_features(log_history):
    """Only the supplied 480 completed historical closes; no target access."""
    from scipy.signal import lfilter
    x=np.asarray(log_history,dtype=np.float64)
    assert x.ndim==2 and x.shape[1]==480 and np.isfinite(x).all()
    def ema(span):
        a=2/(span+1)
        v,_=lfilter([a],[1,-(1-a)],x,axis=1,zi=((1-a)*x[:,0])[:,None])
        return v
    e8,e32=ema(8),ema(32)
    r=np.diff(x,axis=1)
    s60=np.maximum(r[:,-60:].std(axis=1),1e-6)
    trend=np.column_stack((x[:,-1]-e8[:,-1],e8[:,-1]-e32[:,-1],
                            e8[:,-1]-e8[:,-9],e32[:,-1]-e32[:,-33]))/s60[:,None]
    gate=np.column_stack((r[:,-8:].std(axis=1)/s60,
        r[:,-32:].std(axis=1)/np.maximum(r[:,-120:].std(axis=1),1e-6),
        np.abs(e8[:,-1]-e32[:,-1])/s60,np.abs(x[:,-1]-e8[:,-1])/s60))
    residual=(x-e32)/s60[:,None]
    return trend.astype('float32'),residual.astype('float32'),gate.astype('float32'),s60

def cache_arrays():
    return {k:np.load(BASE/'cache'/f'{k}.npy',mmap_mode='r',allow_pickle=False)
            for k in ('origins','trend','residual','gate','volatility','actual')}

def rows_for(origins):
    allidx=np.load(BASE/'cache/origins.npy',mmap_mode='r',allow_pickle=False)
    rows=np.searchsorted(allidx,origins);assert np.array_equal(allidx[rows],origins)
    return rows

def code_hashes():
    return {p.name:sha(p) for p in (BASE/'code').glob('*.py')
            if p.name in ('common.py','models.py','engine.py','worker.py')}
