"""Score the already-fixed annual nonoverlap sensitivity from saved artifacts."""
from datetime import datetime,timezone
import csv,hashlib,io,json,math,re,sys
from pathlib import Path
import numpy as np

ROOT=Path('/home/ayaka/wslcode/aapl-research')
OUT=ROOT/'outputs/multiscale'
ANNUAL=OUT/'private_annual_short_development/20261107T143734746737Z'
MODELS=('last_close','drift','seasonal','ridge','mlp','gru','lstm','xpatch')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text(encoding='utf8'))
def require(ok,msg):
 if not ok:raise ValueError(msg)
def csv_text(rows):
 s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);return s.getvalue()

selection_path=Path(sys.argv[1]).resolve()
selection=load(selection_path)
require(selection['status']=='annual_nonoverlap_selection_frozen_before_this_recalculation' and
        selection['selected_origins']==4 and selection['prediction_scores_read_during_selection'] is False,
        'Selection is not fixed')
require(sha(ANNUAL/'PRIVATE_PROTOCOL.json')==selection['annual_protocol_sha256'] and
        sha(ANNUAL/'PRIVATE_QUEUE_STATE.json')==selection['annual_queue_state_sha256'],
        'Annual protocol or state changed')
wanted={x['origin_index']:x for x in selection['selected']}
require(len(wanted)==4 and all(b-a>=12 for a,b in zip(sorted(wanted),sorted(wanted)[1:])),
        'Selection does not satisfy nonoverlap')
state=load(ANNUAL/'PRIVATE_QUEUE_STATE.json')
window_rows=[]
metrics=[]
for target in ('future_close','future_log_return'):
 found=set()
 accum={m:[] for m in MODELS}
 for key,job in sorted(state['jobs'].items()):
  if f'_{target}_' not in key:continue
  folder=Path(job['fold_directory']);rp=folder/'fold_receipt.json';rec=load(rp)
  require(sha(rp)==job['receipt_sha256'] and rec['Q3_values_read'] is False and
          rec['config']['target']==target and rec['config']['horizon']==12,'Fold receipt differs')
  tp=folder/'targets.npz';require(sha(tp)==rec['targets_sha256'],'Target identity differs')
  with np.load(tp,allow_pickle=False) as z:
   origins=np.asarray(z['origins']);decisions=np.asarray(z['decision_ns']);completions=np.asarray(z['target_completion_ns'])
   indices=np.asarray([i for i,v in enumerate(origins) if int(v) in wanted],dtype=np.int64)
   if not len(indices):continue
   actual=np.asarray(z['actual'],dtype=np.float64)[indices]
   anchors=np.asarray(z['anchor'],dtype=np.float64)[indices]
  for idx in indices:
   origin=int(origins[idx]);s=wanted[origin]
   require(origin not in found and int(decisions[idx])==s['decision_ns'] and
           int(completions[idx])==s['target_completion_ns'],'Selected target metadata differs')
   found.add(origin)
  for model in MODELS:
   pp=folder/f'prediction_{model}.npz';require(sha(pp)==rec['models'][model]['prediction_sha256'],
                                               'Prediction hash differs')
   with np.load(pp,allow_pickle=False) as z:
    require(np.array_equal(z['origins'],origins) and np.array_equal(z['decision_ns'],decisions),
            'Prediction origin differs')
    predicted=np.asarray(z['prediction'],dtype=np.float64)[indices]
   require(predicted.shape==actual.shape and np.isfinite(predicted).all(),'Prediction dimensions differ')
   for j,idx in enumerate(indices):
    origin=int(origins[idx]);error=predicted[j]-actual[j];meta=wanted[origin]
    accum[model].append({'origin_index':origin,'error':error,'anchor':float(anchors[j]),
                         'predicted_end':float(predicted[j,-1]),'actual_end':float(actual[j,-1])})
    window_rows.append({'target':target,'model':model,'origin_index':origin,
                        'origin_session':meta['origin_session'],'first_target_session':meta['first_target_session'],
                        'last_target_session':meta['last_target_session'],
                        'endpoint_squared_error':float(error[-1]**2),'endpoint_absolute_error':float(abs(error[-1])),
                        'path_mse':float(np.mean(np.square(error)))})
 require(found==set(wanted),'A selected origin was not scored')
 baseline=None
 per_model={}
 for model in MODELS:
  rows=sorted(accum[model],key=lambda x:x['origin_index'])
  require([r['origin_index'] for r in rows]==sorted(wanted),'Model selection differs')
  errors=np.stack([r['error'] for r in rows])
  mse=float(np.mean(np.square(errors[:,-1])));mae=float(np.mean(np.abs(errors[:,-1])))
  data={'target':target,'model':model,'selected_origins':len(rows),'endpoint_mse':mse,
        'endpoint_rmse':math.sqrt(mse),'endpoint_mae':mae,'path_mse':float(np.mean(np.square(errors))),
        'path_mae':float(np.mean(np.abs(errors)))}
  per_model[model]=data
  if model=='last_close':baseline=mse
 require(baseline is not None and baseline>0,'Baseline MSE is zero or missing')
 for model in MODELS:
  per_model[model]['relative_improvement']=1-per_model[model]['endpoint_mse']/baseline
  metrics.append(per_model[model])
require(len(metrics)==16 and len(window_rows)==64,'Unexpected result coverage')
directory=selection_path.parent
paths={'results_csv':directory/'PRIVATE_ANNUAL_NONOVERLAP_RESULTS.csv',
       'window_losses_csv':directory/'PRIVATE_ANNUAL_NONOVERLAP_WINDOW_LOSSES.csv',
       'results_json':directory/'PRIVATE_ANNUAL_NONOVERLAP_RESULTS.json',
       'report_en':directory/'PRIVATE_ANNUAL_NONOVERLAP_REPORT_EN.md'}
require(all(not p.exists() for p in paths.values()),'Sensitivity outputs already exist')
paths['results_csv'].write_text(csv_text(metrics),encoding='utf8',newline='')
paths['window_losses_csv'].write_text(csv_text(window_rows),encoding='utf8',newline='')
by={(x['target'],x['model']):x for x in metrics}
lines=['# Annual H12 nonoverlapping-window sensitivity — private development only','',
       'The previously planned rule selects the first eligible validation origin and then the next origin '
       'at least 12 monthly bars later. It yields four nonoverlapping forecast paths out of 47 already-exposed '
       'validation origins; nonoverlap does not make the paths statistically independent.','',
       f"First/last selected origin sessions: {selection['first_origin_session']} to {selection['last_origin_session']}. "
       f"First/last target sessions: {selection['first_target_session']} to {selection['last_target_session']}.",'',
       'Endpoint relative improvement is 100 × (1 − model MSE / persistence or zero-return MSE). '
       'Price MSE is (USD per share)²; log-return MSE is dimensionless. No confidence interval or significance '
       'claim is made from four selected windows.','',
       '| Model | Price MSE | Price improvement | Return MSE | Return improvement |',
       '| --- | ---: | ---: | ---: | ---: |']
for m in MODELS:
 p=by[('future_close',m)];r=by[('future_log_return',m)]
 lines.append(f"| {m} | {p['endpoint_mse']:.6g} | {100*p['relative_improvement']:.2f}% | "
              f"{r['endpoint_mse']:.6g} | {100*r['relative_improvement']:.2f}% |")
lines+=['','This is a descriptive sensitivity on previously viewed market periods. '
        'It is not an independent test, a convergence check, a trading evaluation, or evidence of data publication rights.','']
paths['report_en'].write_text('\n'.join(lines),encoding='utf8')
result={'status':'private_annual_nonoverlap_descriptive_results',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'selection':str(selection_path),'selection_sha256':sha(selection_path),
        'script_sha256':sha(Path(__file__).resolve()),'selected_origins':len(wanted),
        'metrics':metrics,'files':{name:{'path':str(path),'sha256':sha(path)}
                                  for name,path in paths.items() if name!='results_json'},
        'independent_test':False,'Q3_values_read':False,'publication_rights':'not confirmed',
        'limitations':['Only four nonoverlapping but potentially dependent windows.',
                       'The sensitivity uses previously exposed development predictions.',
                       'No inferential confidence interval or significance claim is supported here.']}
paths['results_json'].write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
print(json.dumps({'status':result['status'],'directory':str(directory),
                  'results_json':str(paths['results_json']),'report':str(paths['report_en']),
                  'return_gru_improvement':by[('future_log_return','gru')]['relative_improvement'],
                  'price_gru_improvement':by[('future_close','gru')]['relative_improvement']},ensure_ascii=False))
