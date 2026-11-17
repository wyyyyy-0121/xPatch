"""Independently verify the selected annual sensitivity using scalar arithmetic."""
from datetime import datetime,timezone
import csv,hashlib,json,math,sys
from pathlib import Path
import numpy as np

ROOT=Path('/home/ayaka/wslcode/aapl-research')
ANNUAL=ROOT/'outputs/multiscale/private_annual_short_development/20261107T143734746737Z'
MODELS=('last_close','drift','seasonal','ridge','mlp','gru','lstm','xpatch')
d=Path(sys.argv[1]).resolve()

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def read(p):
 with Path(p).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
def require(ok,msg):
 if not ok:raise ValueError(msg)
def near(a,b):return math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-16)

selection_path=d/'PRIVATE_ANNUAL_NONOVERLAP_SELECTION.json'
result_path=d/'PRIVATE_ANNUAL_NONOVERLAP_RESULTS.json'
selection=load(selection_path);result=load(result_path);state=load(ANNUAL/'PRIVATE_QUEUE_STATE.json')
require(sha(selection_path)==result['selection_sha256'] and len(selection['selected'])==4,'Selection identity differs')
for item in result['files'].values():require(sha(item['path'])==item['sha256'],'Sensitivity artifact hash differs')
rows=read(d/'PRIVATE_ANNUAL_NONOVERLAP_WINDOW_LOSSES.csv')
pooled=read(d/'PRIVATE_ANNUAL_NONOVERLAP_RESULTS.csv')
lookup={(r['target'],r['model'],int(r['origin_index'])):r for r in rows}
pool={(r['target'],r['model']):r for r in pooled}
require(len(lookup)==64 and len(pool)==16,'Sensitivity row count differs')
checked=[]
for target in ('future_close','future_log_return'):
 for model in MODELS:
  squares=[];absolutes=[];paths=[]
  for selected in selection['selected']:
   origin=selected['origin_index'];key=f'annual_h12_{target}_f{selected["fold"]}'
   job=state['jobs'][key];folder=Path(job['fold_directory']);rp=folder/'fold_receipt.json';rec=load(rp)
   require(sha(rp)==job['receipt_sha256'] and rec['Q3_values_read'] is False,'Receipt differs')
   tp=folder/'targets.npz';pp=folder/f'prediction_{model}.npz'
   require(sha(tp)==rec['targets_sha256'] and sha(pp)==rec['models'][model]['prediction_sha256'],
           'Saved target or prediction differs')
   with np.load(tp,allow_pickle=False) as z:
    origins=np.asarray(z['origins']);matches=np.flatnonzero(origins==origin)
    require(len(matches)==1 and int(z['decision_ns'][matches[0]])==selected['decision_ns'] and
            int(z['target_completion_ns'][matches[0]])==selected['target_completion_ns'],'Selected origin differs')
    actual=np.asarray(z['actual'],dtype=np.float64)[matches[0]]
   with np.load(pp,allow_pickle=False) as z:
    require(np.array_equal(z['origins'],origins),'Prediction origins differ')
    prediction=np.asarray(z['prediction'],dtype=np.float64)[matches[0]]
   errors=[float(p)-float(a) for p,a in zip(prediction,actual,strict=True)]
   sq=errors[-1]*errors[-1];ab=abs(errors[-1]);path=math.fsum(x*x for x in errors)/len(errors)
   row=lookup[(target,model,origin)]
   require(near(sq,row['endpoint_squared_error']) and near(ab,row['endpoint_absolute_error']) and
           near(path,row['path_mse']) and row['origin_session']==selected['origin_session'] and
           row['first_target_session']==selected['first_target_session'] and
           row['last_target_session']==selected['last_target_session'],'Window metric differs')
   squares.append(sq);absolutes.append(ab);paths.append(path)
  m=math.fsum(squares)/4;a=math.fsum(absolutes)/4;p=math.fsum(paths)/4
  row=pool[(target,model)]
  require(near(m,row['endpoint_mse']) and near(math.sqrt(m),row['endpoint_rmse']) and
          near(a,row['endpoint_mae']) and near(p,row['path_mse']),'Pooled metric differs')
  checked.append((target,model))
 for model in MODELS:
  row=pool[(target,model)];base=pool[(target,'last_close')]
  require(near(1-float(row['endpoint_mse'])/float(base['endpoint_mse']),row['relative_improvement']),
          'Relative baseline comparison differs')
out=d/'PRIVATE_ANNUAL_NONOVERLAP_INDEPENDENT_CHECK.json'
require(not out.exists(),'Verification already exists')
record={'status':'private_annual_nonoverlap_independently_recomputed',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'selection_sha256':sha(selection_path),'results_sha256':sha(result_path),
        'script_sha256':sha(Path(__file__).resolve()),
        'checks':{'selected_windows':4,'window_model_rows':len(lookup),'pooled_model_rows':len(checked),
                  'scalar_loss_recalculation_matches':True,'baseline_relative_comparisons_match':True,
                  'same_origin_sessions_for_both_targets':True},
        'independent_test':False,'publication_rights':'not confirmed',
        'limitations':['Arithmetic verification on four previously exposed windows is not independent confirmation.',
                       'Nonoverlap does not establish independence or statistical significance.']}
out.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
print(json.dumps({'status':record['status'],'path':str(out),'checks':record['checks']},ensure_ascii=False))
