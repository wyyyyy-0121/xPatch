"""Add evidence-backed provenance and a quantitative abstract to the private draft."""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path('/home/ayaka/wslcode/aapl-research')
OUT=ROOT/'outputs/multiscale'
PARENT=OUT/'private_working_manuscripts/20261107T163546933946Z'
ADMISSION=OUT/'admission/alpaca_20261107T052228573242Z/admission.json'
H60=OUT/'private_minute_h60_aggregate/20261107T144910918784Z/PRIVATE_PARTIAL_AGGREGATE.json'
SEEDS=OUT/'private_split_neutral_seed_replication/20261107T150345851174Z/PRIVATE_SEED_COMPARISON.json'
PROTOCOLS=[
    OUT/'private_three_scale_development/20261107T134128870773Z/PRIVATE_PROTOCOL.json',
    OUT/'private_annual_short_development/20261107T143734746737Z/PRIVATE_PROTOCOL.json',
    OUT/'private_minute_parallel_folds/20261107T143936100555Z/PRIVATE_PROTOCOL.json',
    OUT/'private_minute_breadth_development/20261107T144302924248Z/PRIVATE_PROTOCOL.json',
]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path):
    return json.loads(Path(path).read_text(encoding='utf8'))

def require(ok,message):
    if not ok:raise ValueError(message)

detail=load(PARENT/'PRIVATE_DETAILED_RESULTS_AUDIT.json')
pool_check=load(PARENT/'PRIVATE_POOL_VERIFICATION.json')
word_check=load(PARENT/'PRIVATE_WORD_VERIFICATION.json')
parent_md=PARENT/'PRIVATE_WORKING_MANUSCRIPT_EN.md'
require(sha(parent_md)==detail['manuscript_sha256']==pool_check['manuscript_sha256'], 'Parent draft changed')
require(sha(PARENT/'PRIVATE_WORKING_MANUSCRIPT_EN.docx')==word_check['formatted_docx_sha256'],
        'Parent formatted Word changed')
for name,key in [('PRIVATE_METRICS_POOLED.csv','pooled_csv_sha256'),
                 ('PRIVATE_METRICS_BY_FOLD.csv','fold_csv_sha256'),
                 ('TRAINING_HISTORY_AUDIT.csv','training_history_csv_sha256')]:
    require(sha(PARENT/name)==detail[key], 'Parent evidence CSV changed: '+name)
with (PARENT/'PRIVATE_METRICS_POOLED.csv').open(encoding='utf8',newline='') as f:
    rows=list(csv.DictReader(f))
neural={'mlp','gru','lstm','xpatch'}
price=[r for r in rows if r['target']=='future_close' and r['model'] in neural]
require(len(price)==24 and all(float(r['relative_improvement'])<0 for r in price), 'Price comparison changed')
minute=load(H60)
seed=load(SEEDS)
require(minute['completed_folds']==[1,2,3] and minute['independent_test'] is False and
        seed['same_targets_and_splits'] is True and seed['independent_test'] is False,
        'Exploratory result scope changed')
minute_relative=100*minute['models']['gru']['relative_improvement']
seed_relative=[100*seed['relative_mse_improvement_vs_zero_return']['gru'][str(s)] for s in (2026,2027,2028)]
require(minute_relative<0 and seed_relative[0]>0 and seed_relative[1]<0 and seed_relative[2]>0,
        'Exploratory result direction changed')

ad=load(ADMISSION)
ad_sha=sha(ADMISSION)
require(all(load(p).get('admission_sha256')==ad_sha for p in PROTOCOLS), 'Candidate identity differs across protocols')
require(ad['Q3_values_read'] is False and ad['long_source_academic_license_verified'] is False,
        'Candidate cutoff or rights status changed')
daily=ad['source']
calendar=ad['daily_calendar']
minutes=ad['minute_source']
intraday=ad['intraday_aggregation']
require(daily['rows']==calendar['source_observed_sessions']==calendar['expected_sessions']==2637 and
        calendar['usable_before_cutoff']==2636 and calendar['missing_sessions']==[] and
        calendar['cutoff_excluded_sessions']==['2026-06-30'], 'Daily candidate inventory differs')
counts=intraday['hours_per_session_counts']
require(sum(int(v) for v in counts.values())==intraday['sessions'] and
        sum(int(k)*int(v) for k,v in counts.items())==intraday['full_hour_bars'],
        'Hourly aggregation inventory differs')
require(ad['split_actions']==[{'date':'2020-08-31','ratio':4.0}], 'Candidate split list differs')

text=parent_md.read_text(encoding='utf8')
abstract_start='## Abstract\n\n'
intro_start='\n## 1. Introduction'
require(text.count(abstract_start)==1 and text.count(intro_start)==1, 'Abstract boundaries changed')
prefix,rest=text.split(abstract_start,1)
old_abstract,tail=rest.split(intro_start,1)
abstract=(
    'Forecast accuracy for one stock can vary across horizons and target representations. '
    'This private historical-development study compares persistence, drift, seasonal repetition, ridge regression, '
    'an MLP, GRU, LSTM and an adapted xPatch model on Apple Inc. (AAPL) prices and cumulative log returns across '
    'five temporal scales. At this snapshot, 12 of 20 tasks have five completed folds, while the minute matrix remains '
    'incomplete. Across the six completed nominal-price tasks, none of 24 neural task–model comparisons achieved lower '
    'pooled endpoint MSE than persistence. For minute H60 returns, the GRU had '
    f'{abs(minute_relative):.2f}% higher pooled endpoint MSE than zero return across three folds. '
    'In a post-hoc split-neutral monthly comparison, the GRU’s relative MSE improvements against zero return were '
    f'{seed_relative[0]:.2f}%, {seed_relative[1]:.2f}% and {seed_relative[2]:.2f}% across three seeds. '
    'The short-history annual task has only 47 overlapping validation origins. These previously exposed development '
    'results do not establish independent confirmation or trading profitability; source publication rights and the '
    'remaining experiments are unresolved.'
)
text=prefix+abstract_start+abstract+'\n'+intro_start+tail

data_marker=('Longer public candidates have been investigated, but their upstream provenance, adjustment rules, '
             'and applicable publication rights are not yet fully established. The source review is documented separately. '
             '**[Insert the admitted provider, version, licence, exact periods, coverage, missing observations, source hashes, '
             'and retrieval dates. Do not name a candidate as the study dataset before admission.]**')
require(text.count(data_marker)==1, 'Candidate data paragraph changed')
data_note=(
    '\n\nThe following inventory describes the Alpaca SIP/raw candidate used in the private development snapshot. '
    'It is a date-coverage and processing record, not a publication-admission decision, a guarantee of historical '
    'point-in-time availability, or a substitute for checking the vendor’s close convention.\n\n'
    '| Candidate inventory item | Recorded development evidence |\n'
    '| --- | --- |\n'
    f"| Daily source records | {daily['first']} to {daily['last']}; {daily['rows']:,} recorded daily rows. |\n"
    f"| Daily calendar coverage | {calendar['source_observed_sessions']:,} of {calendar['expected_sessions']:,} expected "
    'session dates present under the XNYS calendar proxy. |\n'
    f"| Daily cutoff | {calendar['usable_before_cutoff']:,} dates usable before cutoff; 2026-06-30 was excluded under "
    'the conservative next-midnight completion rule. |\n'
    f"| Minute candidate | {minutes['source_rows']:,} source rows recorded in the candidate inventory. |\n"
    f"| Complete hourly bars | {intraday['full_hour_bars']:,} bars from {intraday['sessions']:,} regular sessions; "
    f"{intraday['residual_minutes_total']:,} shorter residual minutes excluded. |\n"
    f"| Session lengths | {counts['6']:,} sessions contain six complete hourly bars and {counts['3']:,} contain three. |\n"
    '| Corporate actions | A 4-for-1 split effective 2020-08-31 is recorded; completeness of other actions remains under review. |\n'
    '| Rights and close definition | Account-specific publication and redistribution rights remain unverified. '
    'A minute-derived session close is not necessarily the official closing-auction price. |'
)
text=text.replace(data_marker,data_marker+data_note,1)

old_discussion=('In this exposed development snapshot, simple baselines are competitive and some model rankings vary '
                'substantially between time folds and random seeds. A split-neutral target reduces the mechanical effect of '
                'the 2020 split on the monthly baseline but does not establish a stable model advantage. These observations '
                'cannot be generalized as independent evidence. **[Complete the conclusion after the remaining comparisons '
                'and any independent evaluation.]**')
new_discussion=(
    'In this exposed development snapshot, all 24 neural comparisons on the six completed nominal-price tasks have '
    'higher pooled endpoint MSE than persistence under the recorded training budget. This is a comparison of the stated '
    'recipes, not a claim about an architecture’s attainable limit. Large negative percentage improvements can arise when '
    'the persistence error is small; the absolute errors in Appendix A give the denominators their proper context.\n\n'
    'The 2020 split concentrates much of the nominal monthly baseline error in overlapping paths around a single event. '
    'A post-hoc split-neutral target changes that mechanical contribution, while the corresponding GRU result changes '
    'sign across three seeds. The annual and quarterly windows are few and overlapping, and the minute matrix remains '
    'unfinished. These observations are descriptive and do not establish independent or tradable performance. '
    '**[Complete the conclusion after the remaining comparisons and any independent evaluation.]**'
)
require(text.count(old_discussion)==1, 'Prior discussion paragraph changed')
text=text.replace(old_discussion,new_discussion,1)

stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
directory=OUT/'private_working_manuscripts'/stamp
directory.mkdir(parents=True,exist_ok=False)
draft=directory/'PRIVATE_WORKING_MANUSCRIPT_EN.md'
draft.write_text(text,encoding='utf8')
for name in ('PRIVATE_METRICS_POOLED.csv','PRIVATE_METRICS_BY_FOLD.csv','TRAINING_HISTORY_AUDIT.csv'):
    shutil.copy2(PARENT/name,directory/name)
audit={
  'status':'private_provenance_and_quantitative_abstract_added_not_for_submission',
  'created_at_utc':datetime.now(timezone.utc).isoformat(),
  'parent_directory':str(PARENT),'parent_manuscript_sha256':sha(parent_md),
  'parent_detail_audit_sha256':sha(PARENT/'PRIVATE_DETAILED_RESULTS_AUDIT.json'),
  'parent_pool_check_sha256':sha(PARENT/'PRIVATE_POOL_VERIFICATION.json'),
  'parent_word_check_sha256':sha(PARENT/'PRIVATE_WORD_VERIFICATION.json'),
  'candidate_inventory':str(ADMISSION),'candidate_inventory_sha256':ad_sha,
  'protocol_sha256':{str(p):sha(p) for p in PROTOCOLS},
  'minute_h60_snapshot_sha256':sha(H60),'seed_comparison_sha256':sha(SEEDS),
  'script_sha256':sha(Path(__file__).resolve()),'manuscript_sha256':sha(draft),
  'csv_sha256':{name:sha(directory/name) for name in
                ('PRIVATE_METRICS_POOLED.csv','PRIVATE_METRICS_BY_FOLD.csv','TRAINING_HISTORY_AUDIT.csv')},
  'checks':{
      'price_neural_task_model_comparisons':len(price),'price_comparisons_better_than_persistence':0,
      'minute_h60_gru_relative_percent_3_folds':minute_relative,
      'monthly_split_neutral_gru_relative_percent_by_seed':dict(zip(('2026','2027','2028'),seed_relative)),
      'all_four_protocols_use_same_candidate_inventory':True,'daily_calendar_counts_reconciled':True,
      'hourly_aggregation_counts_reconciled':True,'prior_96_pooled_and_480_fold_rows_preserved':True},
  'Q3_values_read':False,'training_started':False,'independent_test':False,'publication_rights':'not confirmed',
  'limitations':['Private development results remain exposed and incomplete.',
                 'Candidate date coverage does not prove data quality, vendor finality, corporate-action completeness or publication rights.',
                 'The final abstract and conclusion must be revised after the remaining work.']
}
ap=directory/'PRIVATE_PROVENANCE_AUDIT.json'
ap.write_text(json.dumps(audit,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
print(json.dumps({'status':audit['status'],'directory':str(directory),'manuscript':str(draft),'audit':str(ap)},ensure_ascii=False))
