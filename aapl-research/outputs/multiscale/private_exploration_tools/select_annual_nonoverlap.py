"""Freeze the planned annual nonoverlap selection using timestamps, not scores."""
from datetime import datetime,timezone
import hashlib,json,re
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np

ROOT=Path('/home/ayaka/wslcode/aapl-research')
OUT=ROOT/'outputs/multiscale'
ANNUAL=OUT/'private_annual_short_development/20261107T143734746737Z'
ADMISSION=OUT/'admission/alpaca_20261107T052228573242Z/admission.json'
PLAN=ROOT/'MULTISCALE_RESEARCH_PLAN.md'
H=12
CUTOFF_NS=int(datetime(2026,7,1,tzinfo=ZoneInfo('America/New_York')).timestamp())*1_000_000_000

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text(encoding='utf8'))
def require(ok,msg):
 if not ok:raise ValueError(msg)
ad=load(ADMISSION);protocol=load(ANNUAL/'PRIVATE_PROTOCOL.json');state=load(ANNUAL/'PRIVATE_QUEUE_STATE.json')
require(protocol['admission_sha256']==sha(ADMISSION) and ad['Q3_values_read'] is False,'Candidate or cutoff changed')
require(state['completed']==10 and '不重叠年度起点' in PLAN.read_text(encoding='utf8'),'Plan or fold state differs')
bar_path=ADMISSION.parent/'bars_month.npz'
require(sha(bar_path)==ad['files']['bars_month.npz'],'Monthly calendar identity changed')
with np.load(bar_path,allow_pickle=False) as z:
 dates=z['session_date'];observed=z['observed'];ends=z['end_ns']
# Do not load prices or prediction scores while fixing this selection.
by_target={}
for target in ('future_close','future_log_return'):
 rows=[]
 for key,job in sorted(state['jobs'].items()):
  if f'_{target}_' not in key:continue
  folder=Path(job['fold_directory']);rp=folder/'fold_receipt.json';rec=load(rp)
  require(sha(rp)==job['receipt_sha256'] and rec['status']=='completed' and
          rec['Q3_values_read'] is False and rec['config']['target']==target and
          rec['config']['horizon']==H,'Fold identity differs')
  tp=folder/'targets.npz'
  require(sha(tp)==rec['targets_sha256'],'Target metadata identity differs')
  with np.load(tp,allow_pickle=False) as z:
   origins=np.asarray(z['origins']);decisions=np.asarray(z['decision_ns']);completions=np.asarray(z['target_completion_ns'])
  require((completions<CUTOFF_NS).all(),'Target metadata crosses cutoff')
  fold=int(re.search(r'_f(\d+)$',key).group(1))
  for index,decision,completion in zip(origins,decisions,completions,strict=True):
   index=int(index)
   require(index+H<len(dates) and observed[index:index+H+1].all() and
           int(ends[index])==int(decision) and int(ends[index+H])==int(completion),
           'Monthly calendar and target metadata differ')
   rows.append({'origin_index':index,'fold':fold,'decision_ns':int(decision),'target_completion_ns':int(completion),
                'origin_session':str(dates[index]),'first_target_session':str(dates[index+1]),
                'last_target_session':str(dates[index+H])})
 rows.sort(key=lambda x:x['origin_index'])
 require(len(rows)==47 and len({r['origin_index'] for r in rows})==47,'Annual origin count differs')
 chosen=[]
 for row in rows:
  if not chosen or row['origin_index']-chosen[-1]['origin_index']>=H:chosen.append(row)
 by_target[target]={'all':rows,'chosen':chosen}
require(by_target['future_close']==by_target['future_log_return'],'Price and return origin metadata differ')
selected=by_target['future_close']['chosen']
require(len(selected)==4 and all(selected[i+1]['origin_index']-selected[i]['origin_index']>=H
                                 for i in range(len(selected)-1)),'Annual nonoverlap selection differs')
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
directory=OUT/'private_annual_nonoverlap'/stamp
directory.mkdir(parents=True,exist_ok=False)
manifest={
 'status':'annual_nonoverlap_selection_frozen_before_this_recalculation',
 'created_at_utc':datetime.now(timezone.utc).isoformat(),
 'scope':'previously exposed private development sensitivity; no independent confirmation',
 'selection_rule':'sort all saved validation origins; select earliest, then each first origin at least 12 monthly-bar indices after the prior selected origin',
 'rule_source':str(PLAN),'rule_source_sha256':sha(PLAN),
 'admission_sha256':sha(ADMISSION),'annual_protocol_sha256':sha(ANNUAL/'PRIVATE_PROTOCOL.json'),
 'annual_queue_state_sha256':sha(ANNUAL/'PRIVATE_QUEUE_STATE.json'),
 'monthly_bar_calendar_sha256':sha(bar_path),'candidate_origins':47,'selected_origins':len(selected),
 'selected':selected,'price_and_return_origins_match':True,
 'first_origin_session':selected[0]['origin_session'],'last_origin_session':selected[-1]['origin_session'],
 'first_target_session':selected[0]['first_target_session'],'last_target_session':selected[-1]['last_target_session'],
 'prediction_scores_read_during_selection':False,'independent_test':False,'Q3_values_read':False,
 'publication_rights':'not confirmed',
 'limitations':['Nonoverlap does not imply statistical independence.',
                'Selection rule was in the plan, but the development results had already been examined.',
                'Only four windows are available; this cannot establish statistical significance or stable annual performance.']}
path=directory/'PRIVATE_ANNUAL_NONOVERLAP_SELECTION.json'
path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
print(json.dumps({'status':manifest['status'],'directory':str(directory),'path':str(path),
                  'sha256':sha(path),'selected':selected},ensure_ascii=False))
