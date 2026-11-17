"""Integrate descriptive block-interval evidence into the private manuscript."""
from datetime import datetime,timezone
import csv,hashlib,json
from pathlib import Path
import shutil

ROOT=Path('/home/ayaka/wslcode/aapl-research')
OUT=ROOT/'outputs/multiscale'
PARENT=OUT/'private_working_manuscripts/20261107T170837160967Z'
SUPP=OUT/'private_descriptive_interval_supplement/20261107T172544855837Z'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def require(ok,msg):
 if not ok:raise ValueError(msg)
pmd=PARENT/'PRIVATE_WORKING_MANUSCRIPT_EN.md'
pa=load(PARENT/'PRIVATE_ANNUAL_MANUSCRIPT_AUDIT.json');pw=load(PARENT/'PRIVATE_WORD_VERIFICATION.json')
sa=load(SUPP/'PRIVATE_INTERVAL_SUPPLEMENT_AUDIT.json');sv=load(SUPP/'PRIVATE_INTERVAL_SUPPLEMENT_VERIFICATION.json')
require(sha(pmd)==pa['manuscript_sha256']==pw['markdown_sha256'] and
        sha(SUPP/'PRIVATE_INTERVALS_ALL_TASKS.csv')==sa['csv_sha256']==sv['csv_sha256'] and
        sv['checks']['model_block_rows']==192 and sv['checks']['p_values_reported'] is False,
        'Parent manuscript or interval supplement changed')
with (SUPP/'PRIVATE_INTERVALS_ALL_TASKS.csv').open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
annual=[r for r in rows if r['profile']=='annual' and r['target']=='future_log_return' and r['model']=='gru']
annual.sort(key=lambda r:int(r['block_length']))
require(len(rows)==192 and len(annual)==2 and [int(r['block_length']) for r in annual]==[12,24] and
        float(annual[0]['lower'])<0<float(annual[0]['upper']) and float(annual[1]['upper'])<0,
        'Annual illustrative intervals differ')
text=pmd.read_text(encoding='utf8')
old=('The selected GRU return result is 64.53% relative to zero return, but the small set cannot '
     'establish a stable advantage.')
new=('For the selected windows, the GRU endpoint MSE is 64.53% lower than the zero-return baseline, '
     'but the small set cannot establish a stable advantage.')
require(text.count(old)==1,'Annual interpretation paragraph changed')
text=text.replace(old,new,1)
marker='### Split sensitivity and label definition'
require(text.count(marker)==1,'Interval insertion point changed')
interval_text=(
    '### Descriptive uncertainty across completed tasks\n\n'
    'For all 12 completed five-fold tasks, paired endpoint squared-loss differences were summarized with '
    '5,000 non-circular moving-block resamples on the original ordered session grid, using the two planned '
    'block lengths for each scale. The complete private supplement contains 192 task–model–block rows; '
    'degenerate intervals are left undefined and no development p-values are reported. Negative differences '
    'favor the model over persistence or zero return. These post-exposure intervals are descriptive and rely '
    'on assumptions that this study cannot establish.\n\n'
    'For example, on the annual H12 return task using all 47 overlapping origins, the nominal 95% percentile '
    f"interval for GRU-minus-zero-return squared loss is [{float(annual[0]['lower']):.6g}, "
    f"{float(annual[0]['upper']):.6g}] with 12-month blocks and "
    f"[{float(annual[1]['lower']):.6g}, {float(annual[1]['upper']):.6g}] with 24-month blocks. "
    'The difference in whether these ranges include zero illustrates sensitivity to the block length; it is '
    'not an independent significance claim. This 47-origin analysis is distinct from the four-window '
    'nonoverlap sensitivity.\n\n'
)
text=text.replace(marker,interval_text+marker,1)
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
directory=OUT/'private_working_manuscripts'/stamp;directory.mkdir(parents=True,exist_ok=False)
md=directory/'PRIVATE_WORKING_MANUSCRIPT_EN.md';md.write_text(text,encoding='utf8')
for name in ('PRIVATE_METRICS_POOLED.csv','PRIVATE_METRICS_BY_FOLD.csv','TRAINING_HISTORY_AUDIT.csv',
             'PRIVATE_ANNUAL_NONOVERLAP_RESULTS.csv'):
 shutil.copy2(PARENT/name,directory/name)
for name in ('PRIVATE_INTERVALS_ALL_TASKS.csv','PRIVATE_INTERVALS_README_EN.md'):
 shutil.copy2(SUPP/name,directory/name)
audit={
 'status':'private_descriptive_uncertainty_integrated_not_for_submission',
 'created_at_utc':datetime.now(timezone.utc).isoformat(),
 'parent_directory':str(PARENT),'parent_manuscript_sha256':sha(pmd),
 'parent_annual_audit_sha256':sha(PARENT/'PRIVATE_ANNUAL_MANUSCRIPT_AUDIT.json'),
 'parent_word_verification_sha256':sha(PARENT/'PRIVATE_WORD_VERIFICATION.json'),
 'pool_audit_path':pa['pool_audit_path'],'pool_audit_sha256':pa['pool_audit_sha256'],
 'metrics_parent_manuscript_sha256':pa['metrics_parent_manuscript_sha256'],
 'annual_selection_path':pa['annual_selection_path'],'annual_selection_sha256':pa['annual_selection_sha256'],
 'annual_results_path':pa['annual_results_path'],'annual_results_sha256':pa['annual_results_sha256'],
 'annual_independent_check_path':pa['annual_independent_check_path'],'annual_independent_check_sha256':pa['annual_independent_check_sha256'],
 'interval_supplement_audit':str(SUPP/'PRIVATE_INTERVAL_SUPPLEMENT_AUDIT.json'),
 'interval_supplement_audit_sha256':sha(SUPP/'PRIVATE_INTERVAL_SUPPLEMENT_AUDIT.json'),
 'interval_supplement_verification':str(SUPP/'PRIVATE_INTERVAL_SUPPLEMENT_VERIFICATION.json'),
 'interval_supplement_verification_sha256':sha(SUPP/'PRIVATE_INTERVAL_SUPPLEMENT_VERIFICATION.json'),
 'script_sha256':sha(Path(__file__).resolve()),'manuscript_sha256':sha(md),
 'csv_sha256':{name:sha(directory/name) for name in
   ('PRIVATE_METRICS_POOLED.csv','PRIVATE_METRICS_BY_FOLD.csv','TRAINING_HISTORY_AUDIT.csv',
    'PRIVATE_ANNUAL_NONOVERLAP_RESULTS.csv','PRIVATE_INTERVALS_ALL_TASKS.csv')},
 'checks':{'completed_tasks_with_descriptive_intervals':12,'model_block_rows':192,
           'degenerate_intervals':28,'p_values_reported':False,
           'annual_gru_12_month_interval':[float(annual[0]['lower']),float(annual[0]['upper'])],
           'annual_gru_24_month_interval':[float(annual[1]['lower']),float(annual[1]['upper'])],
           'annual_nonoverlap_interpretation_clarified':True},
 'independent_test':False,'Q3_values_read':False,'training_started':False,'publication_rights':'not confirmed',
 'limitations':['Post-exposure intervals do not establish independent significance or stationarity.',
                'The four nonoverlapping annual windows remain potentially dependent.',
                'Minute folds, publication rights, final protocol and conclusion remain unresolved.']
}
ap=directory/'PRIVATE_ANNUAL_MANUSCRIPT_AUDIT.json'
ap.write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
print(json.dumps({'status':audit['status'],'directory':str(directory),'manuscript':str(md),'audit':str(ap)},ensure_ascii=False))
