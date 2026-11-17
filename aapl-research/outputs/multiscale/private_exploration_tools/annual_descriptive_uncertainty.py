"""Describe block-resampled annual loss differences on already-exposed folds."""
from datetime import datetime,timezone
import hashlib,json,sys,time
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
from multiscale.evaluation import paired_block_bootstrap

ROOT=Path('/home/ayaka/wslcode/aapl-research')
OUT=ROOT/'outputs/multiscale'
ANNUAL=OUT/'private_annual_short_development/20261107T143734746737Z'
ADMISSION=OUT/'admission/alpaca_20261107T052228573242Z/admission.json'
RECIPE=ROOT/'multiscale/RECIPE_DRAFT.json'
EVALUATION=ROOT/'multiscale/evaluation.py'
MODELS=('last_close','drift','seasonal','ridge','mlp','gru','lstm','xpatch')
CUTOFF_NS=int(datetime(2026,7,1,tzinfo=ZoneInfo('America/New_York')).timestamp())*1_000_000_000

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def require(ok,msg):
 if not ok:raise ValueError(msg)
def write(p,v):
 with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')

ad=load(ADMISSION);protocol=load(ANNUAL/'PRIVATE_PROTOCOL.json');state=load(ANNUAL/'PRIVATE_QUEUE_STATE.json')
recipe=load(RECIPE)['analysis']
require(protocol['admission_sha256']==sha(ADMISSION) and state['completed']==10 and
        ad['Q3_values_read'] is False,'Annual candidate or completed state differs')
require(recipe['block_lengths_by_profile']['annual']==[12,24] and recipe['resamples']==5000 and
        recipe['seed']==2026 and recipe['confidence']==0.95 and recipe['development_p_values'] is False,
        'Draft analysis settings differ')
bar_path=ADMISSION.parent/'bars_month.npz'
require(sha(bar_path)==ad['files']['bars_month.npz'],'Monthly session metadata differs')
with np.load(bar_path,allow_pickle=False) as z:labels=z['session_date'].copy()
require(labels.dtype.kind=='U' and (labels<='2026-06-30').all() and (labels[1:]>=labels[:-1]).all(),
        'Invalid or post-cutoff session grid')
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
d=OUT/'private_annual_uncertainty'/stamp;d.mkdir(parents=True,exist_ok=False)
plan={'status':'private_annual_descriptive_uncertainty_plan','created_at_utc':datetime.now(timezone.utc).isoformat(),
      'tasks':['annual_h12_future_close','annual_h12_future_log_return'],
      'block_lengths_monthly_sessions':[12,24],'resamples':5000,'seed':2026,'confidence':0.95,
      'method':'non-circular moving blocks on the full original monthly-session grid; paired endpoint squared-loss difference weighted by origin count',
      'comparison':'model loss minus persistence / zero-return loss; negative favors the model',
      'source_annual_protocol_sha256':sha(ANNUAL/'PRIVATE_PROTOCOL.json'),'source_annual_queue_sha256':sha(ANNUAL/'PRIVATE_QUEUE_STATE.json'),
      'source_admission_sha256':sha(ADMISSION),'source_monthly_bars_sha256':sha(bar_path),
      'source_recipe_sha256':sha(RECIPE),'source_evaluation_sha256':sha(EVALUATION),
      'script_sha256':sha(Path(__file__).resolve()),'independent_test':False,'p_values_reported':False,'Q3_values_read':False,
      'limitations':['Results and target periods were already exposed; intervals are descriptive and post-exposure.',
                     'Overlapping paths, few annual periods, nonstationarity and multiple comparisons limit interpretation.',
                     'Data publication rights and point-in-time quality remain unverified.']}
write(d/'PRIVATE_PROTOCOL.json',plan)
index=[];start=time.monotonic()
for target in ('future_close','future_log_return'):
 groups=[];diffs={m:[] for m in MODELS};prior=None
 for fold in range(1,6):
  key=f'annual_h12_{target}_f{fold}';job=state['jobs'][key];folder=Path(job['fold_directory']);rp=folder/'fold_receipt.json';rec=load(rp)
  require(sha(rp)==job['receipt_sha256'] and rec['Q3_values_read'] is False and
          rec['config']['target']==target and rec['config']['horizon']==12,'Annual fold identity differs')
  tp=folder/'targets.npz';require(sha(tp)==rec['targets_sha256'],'Target identity differs')
  with np.load(tp,allow_pickle=False) as z:
   origins=np.asarray(z['origins']);decisions=np.asarray(z['decision_ns']);completions=np.asarray(z['target_completion_ns'])
   require((completions<CUTOFF_NS).all(),'Target crosses protected cutoff')
   actual=np.asarray(z['actual'][:,-1],dtype=np.float64)
  require(prior is None or int(decisions[0])>prior,'Outer decisions overlap')
  prior=int(decisions[-1]);groups.append(labels[origins])
  pp=folder/'prediction_last_close.npz';require(sha(pp)==rec['models']['last_close']['prediction_sha256'],'Baseline identity differs')
  with np.load(pp,allow_pickle=False) as z:
   require(np.array_equal(z['origins'],origins) and np.array_equal(z['decision_ns'],decisions),'Baseline origins differ')
   base=np.asarray(z['prediction'][:,-1],dtype=np.float64)
  base_loss=np.square(base-actual)
  for m in MODELS:
   path=folder/f'prediction_{m}.npz';require(sha(path)==rec['models'][m]['prediction_sha256'],'Prediction identity differs')
   with np.load(path,allow_pickle=False) as z:
    require(np.array_equal(z['origins'],origins) and np.array_equal(z['decision_ns'],decisions),'Model origins differ')
    prediction=np.asarray(z['prediction'][:,-1],dtype=np.float64)
   diffs[m].append(np.square(prediction-actual)-base_loss)
 groups=np.concatenate(groups)
 grid=np.unique(labels);grid=grid[(grid>=groups[0])&(grid<=groups[-1])]
 require(len(groups)==47 and len(np.unique(groups))==47 and len(grid)>=25,'Annual group count differs')
 intervals={}
 for m in MODELS:
  diff=np.concatenate(diffs[m]);intervals[m]=[]
  for length in (12,24):
   value=paired_block_bootstrap(diff,groups,block_length=length,resamples=5000,seed=2026,
                                confidence=0.95,group_order=grid)
   value.pop('centered_one_sided_p_improvement',None)
   value['status']='descriptive_development_only' if value['bootstrap_status']=='ok' else 'degenerate_bootstrap'
   value['interpretation']=('Descriptive paired loss difference only; negative favors the model. Not independent confirmation.'
                            if value['bootstrap_status']=='ok' else
                            'All resampled means are identical; no interval is reported.')
   intervals[m].append(value)
  print(json.dumps({'stage':'annual_model_intervals','target':target,'model':m,'elapsed_seconds':round(time.monotonic()-start,1)}),flush=True)
 payload={'status':'private_annual_descriptive_not_independent','profile':'annual','horizon':12,'target':target,
          'folds':5,'origins':len(groups),'models':intervals,'independent_test':False,'p_values_reported':False,
          'limits':plan['limitations']}
 path=d/f'PRIVATE_annual_h12_{target}.json';write(path,payload)
 index.append({'profile':'annual','horizon':12,'target':target,'path':str(path),'sha256':sha(path)})
result={'status':'private_annual_descriptive_intervals_complete','tasks':index,'independent_test':False,
        'p_values_reported':False,'elapsed_seconds':time.monotonic()-start}
write(d/'PRIVATE_INDEX.json',result)
print(json.dumps({'status':result['status'],'directory':str(d),'tasks':len(index),
                  'elapsed_seconds':round(result['elapsed_seconds'],1)}),flush=True)
