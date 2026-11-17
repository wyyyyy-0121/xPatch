"""Combine verified descriptive intervals for twelve complete private tasks."""
from collections import Counter
from datetime import datetime,timezone
import csv,hashlib,io,json,math
from pathlib import Path

ROOT=Path('/home/ayaka/wslcode/aapl-research')
OUT=ROOT/'outputs/multiscale'
MAIN=OUT/'private_uncertainty_analysis/20261107T142331332993Z'
YEAR=OUT/'private_annual_uncertainty/20261107T172219419562Z'
PAPER=OUT/'private_working_manuscripts/20261107T170837160967Z'
MODELS=('last_close','drift','seasonal','ridge','mlp','gru','lstm','xpatch')
BLOCKS={'annual':[12,24],'quarterly':[3,6],'monthly':[21,42],'hourly':[2,5]}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def require(ok,msg):
 if not ok:raise ValueError(msg)
def near(a,b):return math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-15)
def csv_text(rows):
 stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 return stream.getvalue()

main_index=load(MAIN/'PRIVATE_INDEX.json');main_check=load(MAIN/'PRIVATE_INTERVAL_VERIFICATION.json')
year_index=load(YEAR/'PRIVATE_INDEX.json');year_check=load(YEAR/'PRIVATE_ANNUAL_INTERVAL_VERIFICATION.json')
paper_audit=load(PAPER/'PRIVATE_ANNUAL_MANUSCRIPT_AUDIT.json')
require(sha(PAPER/'PRIVATE_WORKING_MANUSCRIPT_EN.md')==paper_audit['manuscript_sha256'],'Manuscript identity differs')
require(sha(MAIN/'PRIVATE_INDEX.json')==main_check['index_sha256'] and main_check['tasks']==11 and
        main_check['p_values_reported'] is False,'Prior interval audit differs')
require(sha(YEAR/'PRIVATE_INDEX.json')==year_check['index_sha256'] and
        year_check['checks']['model_block_intervals']==32 and year_check['checks']['p_values_reported'] is False,
        'Annual interval audit differs')
with (PAPER/'PRIVATE_METRICS_POOLED.csv').open(encoding='utf8',newline='') as f:pooled=list(csv.DictReader(f))
pmap={(r['profile'],int(r['horizon']),r['target'],r['model']):r for r in pooled}
require(len(pmap)==96 and sha(PAPER/'PRIVATE_METRICS_POOLED.csv')==paper_audit['csv_sha256']['PRIVATE_METRICS_POOLED.csv'],
        'Pooled metrics identity differs')
entries=[x for x in main_index['tasks'] if x['profile']!='minute']+year_index['tasks']
require(len(entries)==12 and len({(x['profile'],x['horizon'],x['target']) for x in entries})==12,
        'Expected twelve complete tasks')
rows=[];task_info=[]
for entry in entries:
 path=Path(entry['path']);require(sha(path)==entry['sha256'],'Interval task identity differs')
 data=load(path);profile=entry['profile'];h=int(entry['horizon']);target=entry['target']
 require(data['independent_test'] is False and data['p_values_reported'] is False and data['folds']==5 and
         set(data['models'])==set(MODELS),'Interval task scope differs')
 task_info.append({'profile':profile,'horizon':h,'target':target,'origins':data['origins'],'block_lengths':BLOCKS[profile]})
 for model in MODELS:
  results=data['models'][model]
  require(len(results)==2 and [x['block_length'] for x in results]==BLOCKS[profile],'Block lengths differ')
  model_mse=float(pmap[(profile,h,target,model)]['endpoint_mse'])
  baseline_mse=float(pmap[(profile,h,target,'last_close')]['endpoint_mse'])
  for item in results:
   require('centered_one_sided_p_improvement' not in item and near(item['mean_paired_loss_difference'],model_mse-baseline_mse),
           'Paired loss difference or no-p-value rule differs')
   interval=item.get('confidence_interval')
   status=item['status']
   if status=='descriptive_development_only':
    require(isinstance(interval,list) and len(interval)==2 and interval[0]<interval[1] and
            item['resamples']==5000 and item['seed']==2026,'Invalid descriptive interval')
   else:
    require(status in ('degenerate_bootstrap','insufficient_groups') and interval is None,'Invalid unavailable interval')
   rows.append({'profile':profile,'horizon':h,'target':target,'model':model,
                'origins':data['origins'],'block_length':item['block_length'],
                'groups':item['groups'],'observed_groups':item['observed_groups'],
                'mean_model_minus_baseline_loss':item['mean_paired_loss_difference'],
                'lower':None if interval is None else interval[0],
                'upper':None if interval is None else interval[1],
                'status':status,'resamples':item.get('resamples',0),'seed':item.get('seed',''),
                'source_path':str(path),'source_sha256':entry['sha256']})
require(len(rows)==192,'Unexpected interval row count')
rows.sort(key=lambda x:(x['profile'],x['horizon'],x['target'],MODELS.index(x['model']),x['block_length']))
status_counts=Counter(x['status'] for x in rows)
annual_gru=[x for x in rows if x['profile']=='annual' and x['target']=='future_log_return' and x['model']=='gru']
require(len(annual_gru)==2 and annual_gru[0]['lower']<0<annual_gru[0]['upper'] and annual_gru[1]['upper']<0,
        'Illustrative annual GRU interval pattern differs')
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
d=OUT/'private_descriptive_interval_supplement'/stamp;d.mkdir(parents=True,exist_ok=False)
csv_path=d/'PRIVATE_INTERVALS_ALL_TASKS.csv';csv_path.write_text(csv_text(rows),encoding='utf8',newline='')
lines=['# Descriptive block intervals — private development supplement','',
       'These are post-exposure descriptions of 12 completed five-fold tasks, not independent confirmation. '
       'They use paired endpoint squared-loss differences (model minus persistence/zero-return), non-circular '
       'moving blocks on the original ordered session grid, 5,000 resamples, seed 2026 and nominal 95% percentile '
       'intervals. Negative differences favor the model. No development p-values are reported.','',
       'The CSV includes all eight models and both planned block lengths per task. Price differences are in '
       '(USD per share)²; return differences are in squared log-return units. Missing or degenerate intervals '
       'are left blank rather than represented as zero-width confidence intervals.','',
       '| Scale | Horizon | Target | Origins | Block lengths (sessions) |',
       '| --- | ---: | --- | ---: | --- |']
for x in sorted(task_info,key=lambda r:(r['profile'],r['horizon'],r['target'])):
 lines.append(f"| {x['profile']} | {x['horizon']} | {x['target']} | {x['origins']:,} | "
              f"{x['block_lengths'][0]}, {x['block_lengths'][1]} |")
lines+=['','For illustration, annual H12 return GRU has a 12-month block interval '
        f"[{annual_gru[0]['lower']:.6g}, {annual_gru[0]['upper']:.6g}] and a 24-month block interval "
        f"[{annual_gru[1]['lower']:.6g}, {annual_gru[1]['upper']:.6g}] for model-minus-baseline squared loss. "
        'The change in whether the interval includes zero illustrates sensitivity to block length; it is not a '
        'significance claim. The four nonoverlapping annual windows are a separate sensitivity and are not these '
        '47-origin intervals.','',
        'These intervals do not prove stationarity, independence, convergence, publication rights or trading profitability. '
        'The incomplete minute matrix is not included in this supplement.','']
readme=d/'PRIVATE_INTERVALS_README_EN.md';readme.write_text('\n'.join(lines),encoding='utf8')
audit={'status':'private_twelve_task_descriptive_interval_supplement',
       'created_at_utc':datetime.now(timezone.utc).isoformat(),
       'paper':str(PAPER/'PRIVATE_WORKING_MANUSCRIPT_EN.md'),'paper_sha256':sha(PAPER/'PRIVATE_WORKING_MANUSCRIPT_EN.md'),
       'prior_index_sha256':sha(MAIN/'PRIVATE_INDEX.json'),'prior_verification_sha256':sha(MAIN/'PRIVATE_INTERVAL_VERIFICATION.json'),
       'annual_index_sha256':sha(YEAR/'PRIVATE_INDEX.json'),'annual_verification_sha256':sha(YEAR/'PRIVATE_ANNUAL_INTERVAL_VERIFICATION.json'),
       'script_sha256':sha(Path(__file__).resolve()),'csv_sha256':sha(csv_path),'readme_sha256':sha(readme),
       'checks':{'tasks':len(entries),'model_block_rows':len(rows),'status_counts':dict(status_counts),
                 'all_mean_differences_match_pooled_mse':True,'p_values_reported':False},
       'independent_test':False,'publication_rights':'not confirmed',
       'limitations':['Descriptive post-exposure intervals are not independent significance evidence.',
                      'The incomplete minute matrix and vendor publication rights remain unresolved.']}
ap=d/'PRIVATE_INTERVAL_SUPPLEMENT_AUDIT.json';ap.write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
print(json.dumps({'status':audit['status'],'directory':str(d),'csv':str(csv_path),'readme':str(readme),
                  'audit':str(ap),'checks':audit['checks']},ensure_ascii=False))
