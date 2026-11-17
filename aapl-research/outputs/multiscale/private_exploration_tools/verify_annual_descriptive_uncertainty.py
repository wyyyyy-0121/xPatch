"""Independently repeat annual descriptive bootstrap intervals without p-values."""
from datetime import datetime,timezone
import hashlib,json,math,sys
from pathlib import Path
import numpy as np

ROOT=Path('/home/ayaka/wslcode/aapl-research')
OUT=ROOT/'outputs/multiscale'
ANNUAL=OUT/'private_annual_short_development/20261107T143734746737Z'
ADMISSION=OUT/'admission/alpaca_20261107T052228573242Z/admission.json'
MODELS=('last_close','drift','seasonal','ridge','mlp','gru','lstm','xpatch')
d=Path(sys.argv[1]).resolve()

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def require(ok,msg):
 if not ok:raise ValueError(msg)
def near(a,b):return math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-15)
def quantile(sorted_values,q):
 pos=(len(sorted_values)-1)*q;i=int(math.floor(pos));f=pos-i
 return sorted_values[i]*(1-f)+sorted_values[min(i+1,len(sorted_values)-1)]*f

plan=load(d/'PRIVATE_PROTOCOL.json');index=load(d/'PRIVATE_INDEX.json')
state=load(ANNUAL/'PRIVATE_QUEUE_STATE.json');ad=load(ADMISSION)
require(index['status']=='private_annual_descriptive_intervals_complete' and len(index['tasks'])==2 and
        plan['resamples']==5000 and plan['seed']==2026 and plan['block_lengths_monthly_sessions']==[12,24],
        'Annual interval plan differs')
require(sha(ANNUAL/'PRIVATE_PROTOCOL.json')==plan['source_annual_protocol_sha256'] and
        sha(ANNUAL/'PRIVATE_QUEUE_STATE.json')==plan['source_annual_queue_sha256'] and
        sha(ADMISSION)==plan['source_admission_sha256'],'Source identity differs')
bars=ADMISSION.parent/'bars_month.npz'
require(sha(bars)==plan['source_monthly_bars_sha256'],'Monthly metadata identity differs')
with np.load(bars,allow_pickle=False) as z:labels=z['session_date'].copy()
checked=0;degenerate=0
for entry in index['tasks']:
 path=Path(entry['path']);require(sha(path)==entry['sha256'],'Interval result changed')
 data=load(path);target=entry['target'];groups=[];diffs={m:[] for m in MODELS}
 for fold in range(1,6):
  key=f'annual_h12_{target}_f{fold}';job=state['jobs'][key];folder=Path(job['fold_directory']);rp=folder/'fold_receipt.json';rec=load(rp)
  require(sha(rp)==job['receipt_sha256'] and rec['Q3_values_read'] is False,'Fold receipt differs')
  tp=folder/'targets.npz';require(sha(tp)==rec['targets_sha256'],'Target file differs')
  with np.load(tp,allow_pickle=False) as z:
   origins=z['origins'].copy();dec=z['decision_ns'].copy();actual=np.asarray(z['actual'][:,-1],dtype=float)
  groups.extend(str(x) for x in labels[origins])
  bp=folder/'prediction_last_close.npz';require(sha(bp)==rec['models']['last_close']['prediction_sha256'],'Baseline differs')
  with np.load(bp,allow_pickle=False) as z:base=np.asarray(z['prediction'][:,-1],dtype=float)
  base_loss=(base-actual)**2
  for model in MODELS:
   pp=folder/f'prediction_{model}.npz';require(sha(pp)==rec['models'][model]['prediction_sha256'],'Prediction differs')
   with np.load(pp,allow_pickle=False) as z:
    require(np.array_equal(z['origins'],origins) and np.array_equal(z['decision_ns'],dec),'Origins differ')
    pred=np.asarray(z['prediction'][:,-1],dtype=float)
   diffs[model].extend(((pred-actual)**2-base_loss).tolist())
 grid=sorted(set(str(x) for x in labels if groups[0]<=str(x)<=groups[-1]))
 require(len(groups)==47 and len(grid)>=25,'Annual session grid differs')
 positions={g:i for i,g in enumerate(grid)}
 for model in MODELS:
  sums=[0.0]*len(grid);counts=[0]*len(grid)
  for group,value in zip(groups,diffs[model],strict=True):
   i=positions[group];sums[i]+=value;counts[i]+=1
  for length,stored in zip((12,24),data['models'][model],strict=True):
   require(stored['block_length']==length and stored['resamples']==5000 and stored['seed']==2026,
           'Stored interval settings differ')
   rng=np.random.default_rng(2026);block_count=math.ceil(len(grid)/length);draws=[];attempts=0
   while len(draws)<5000:
    attempts+=1
    starts=rng.integers(0,len(grid)-length+1,size=block_count)
    sampled=[int(start)+off for start in starts for off in range(length)][:len(grid)]
    total=sum(counts[i] for i in sampled)
    if not total:continue
    draws.append(math.fsum(sums[i] for i in sampled)/total)
   require(near(math.fsum(diffs[model])/len(groups),stored['mean_paired_loss_difference']) and
           attempts-5000==stored['empty_resamples_skipped'],'Observed difference or empty draws differ')
   if all(x==draws[0] for x in draws):
    require(stored['confidence_interval'] is None and stored['status']=='degenerate_bootstrap','Degenerate interval differs')
    degenerate+=1
   else:
    ordered=sorted(draws);lo=quantile(ordered,0.025);hi=quantile(ordered,0.975)
    require(stored['status']=='descriptive_development_only' and near(lo,stored['confidence_interval'][0]) and
            near(hi,stored['confidence_interval'][1]),'Percentile interval differs')
   require('centered_one_sided_p_improvement' not in stored,'Development p-value was reported')
   checked+=1
out=d/'PRIVATE_ANNUAL_INTERVAL_VERIFICATION.json'
require(not out.exists(),'Verification already exists')
result={'status':'private_annual_descriptive_intervals_independently_recomputed',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'plan_sha256':sha(d/'PRIVATE_PROTOCOL.json'),'index_sha256':sha(d/'PRIVATE_INDEX.json'),
        'script_sha256':sha(Path(__file__).resolve()),
        'checks':{'tasks':2,'model_block_intervals':checked,'degenerate_intervals':degenerate,
                  'independent_scalar_resampling_and_quantiles_match':True,'p_values_reported':False},
        'independent_test':False,'publication_rights':'not confirmed',
        'limitations':['Reproducing the bootstrap does not establish stationarity, independence, or significance.',
                       'Annual samples remain small and previously exposed.']}
out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
print(json.dumps({'status':result['status'],'path':str(out),'checks':result['checks']},ensure_ascii=False))
