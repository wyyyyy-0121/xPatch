"""Independently check all rows of the private interval supplement."""
from collections import Counter
from datetime import datetime,timezone
from hashlib import sha256
import csv,json,math,sys
from pathlib import Path

d=Path(sys.argv[1]).resolve()
audit_path=d/'PRIVATE_INTERVAL_SUPPLEMENT_AUDIT.json'
csv_path=d/'PRIVATE_INTERVALS_ALL_TASKS.csv'
readme=d/'PRIVATE_INTERVALS_README_EN.md'
audit=json.loads(audit_path.read_text(encoding='utf8'))
def digest(p):return sha256(Path(p).read_bytes()).hexdigest()
def require(ok,msg):
 if not ok:raise ValueError(msg)
def near(a,b):return math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-15)
require(digest(csv_path)==audit['csv_sha256'] and digest(readme)==audit['readme_sha256'],'Supplement identity differs')
with csv_path.open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
require(len(rows)==192 and len({(r['profile'],r['horizon'],r['target'],r['model'],r['block_length']) for r in rows})==192,
        'Interval row count or uniqueness differs')
sources={}
for row in rows:
 path=Path(row['source_path'])
 require(digest(path)==row['source_sha256'],'Interval source changed')
 if str(path) not in sources:sources[str(path)]=json.loads(path.read_text(encoding='utf8'))
 source=sources[str(path)]
 require(source['independent_test'] is False and source['p_values_reported'] is False and
         source['profile']==row['profile'] and source['target']==row['target'] and
         int(source['horizon'])==int(row['horizon']) and source['origins']==int(row['origins']),
         'Interval task metadata differs')
 found=[x for x in source['models'][row['model']] if x['block_length']==int(row['block_length'])]
 require(len(found)==1,'Model block interval is missing')
 item=found[0];interval=item.get('confidence_interval')
 require(item['status']==row['status'] and near(item['mean_paired_loss_difference'],row['mean_model_minus_baseline_loss']) and
         item['groups']==int(row['groups']) and item['observed_groups']==int(row['observed_groups']) and
         'centered_one_sided_p_improvement' not in item,'Interval metadata differs')
 if interval is None:
  require(row['lower']=='' and row['upper']=='','Unavailable interval was given numeric bounds')
 else:
  require(near(interval[0],row['lower']) and near(interval[1],row['upper']) and
          int(row['resamples'])==5000 and int(row['seed'])==2026,'Interval bounds differ')
counts=Counter(x['status'] for x in rows)
require(dict(counts)==audit['checks']['status_counts'] and len(sources)==12,'Interval status totals differ')
target=[r for r in rows if r['profile']=='annual' and r['target']=='future_log_return' and r['model']=='gru']
target.sort(key=lambda r:int(r['block_length']))
plain=readme.read_text(encoding='utf8')
for row in target:
 require(f"[{float(row['lower']):.6g}, {float(row['upper']):.6g}]" in plain,'README example differs')
out=d/'PRIVATE_INTERVAL_SUPPLEMENT_VERIFICATION.json'
require(not out.exists(),'Supplement verification already exists')
record={'status':'private_interval_supplement_independently_checked',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'audit_sha256':digest(audit_path),'csv_sha256':digest(csv_path),'readme_sha256':digest(readme),
        'script_sha256':digest(Path(__file__).resolve()),
        'checks':{'task_sources':len(sources),'model_block_rows':len(rows),
                  'every_row_matches_hashed_source':True,'unavailable_intervals_blank':True,
                  'README_example_matches':True,'p_values_reported':False},
        'limitations':['These are exposed development descriptions, not independent statistical confirmation.',
                       'Resampling checks cannot establish stationarity, data rights or trading profitability.']}
out.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
print(json.dumps({'status':record['status'],'path':str(out),'checks':record['checks']},ensure_ascii=False))
