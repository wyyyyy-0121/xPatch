"""Integrate a pre-specified, already-exposed annual sensitivity into the private manuscript."""
from datetime import datetime,timezone
import csv,hashlib,json,math
from pathlib import Path
import shutil

ROOT=Path('/home/ayaka/wslcode/aapl-research')
OUT=ROOT/'outputs/multiscale'
PARENT=OUT/'private_working_manuscripts/20261107T164541036582Z'
SENS=OUT/'private_annual_nonoverlap/20261107T170513038674Z'
OLDMETRICS=OUT/'private_working_manuscripts/20261107T163546933946Z'
MODELS=('last_close','drift','seasonal','ridge','mlp','gru','lstm','xpatch')
LABELS={'last_close':'Persistence / zero','drift':'Drift','seasonal':'Seasonal','ridge':'Ridge',
        'mlp':'MLP','gru':'GRU','lstm':'LSTM','xpatch':'xPatch'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def require(ok,msg):
 if not ok:raise ValueError(msg)
def near(a,b):return math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-16)

parent_md=PARENT/'PRIVATE_WORKING_MANUSCRIPT_EN.md'
parent_audit=load(PARENT/'PRIVATE_PROVENANCE_AUDIT.json')
parent_word=load(PARENT/'PRIVATE_WORD_VERIFICATION.json')
require(sha(parent_md)==parent_audit['manuscript_sha256']==parent_word['markdown_sha256'],'Parent draft changed')
selection_path=SENS/'PRIVATE_ANNUAL_NONOVERLAP_SELECTION.json'
result_path=SENS/'PRIVATE_ANNUAL_NONOVERLAP_RESULTS.json'
independent_path=SENS/'PRIVATE_ANNUAL_NONOVERLAP_INDEPENDENT_CHECK.json'
selection=load(selection_path);result=load(result_path);independent=load(independent_path)
require(sha(selection_path)==result['selection_sha256']==independent['selection_sha256'] and
        sha(result_path)==independent['results_sha256'] and selection['selected_origins']==4 and
        independent['checks']['scalar_loss_recalculation_matches'] and
        result['independent_test'] is False,'Annual sensitivity evidence changed')
for item in result['files'].values():require(sha(item['path'])==item['sha256'],'Annual sensitivity artifact changed')
with (SENS/'PRIVATE_ANNUAL_NONOVERLAP_RESULTS.csv').open(encoding='utf8',newline='') as f:
 rows=list(csv.DictReader(f))
by={(x['target'],x['model']):x for x in rows}
require(len(by)==16 and all(int(x['selected_origins'])==4 for x in rows),'Sensitivity coverage differs')
for target in ('future_close','future_log_return'):
 base=float(by[(target,'last_close')]['endpoint_mse'])
 require(base>0,'Sensitivity baseline is zero')
 for model in MODELS:
  x=by[(target,model)]
  require(near(1-float(x['endpoint_mse'])/base,x['relative_improvement']),'Sensitivity improvement differs')
parent_ref=PARENT/'reference_evidence_20261108/REFERENCE_EVIDENCE_VERIFICATION.json'
refcheck=load(parent_ref)
require(refcheck['checks']['entries']==17 and refcheck['checks']['holm_issue_in_source_is_missing'],
        'Reference evidence status differs')
text=parent_md.read_text(encoding='utf8')
marker='### Split sensitivity and label definition'
require(text.count(marker)==1,'Sensitivity insertion point changed')
lines=[
 '### Annual sensitivity: nonoverlapping forecast windows','',
 'The previously planned selection takes the earliest eligible annual validation origin and then each next origin '
 'at least twelve monthly bars later. It selects four nonoverlapping forecast paths from the 47 already-exposed origins. '
 f"The first and last selected origin sessions are {selection['first_origin_session']} and {selection['last_origin_session']}; "
 f"the first and last target sessions are {selection['first_target_session']} and {selection['last_target_session']}. "
 'The selection and the arithmetic were checked against the saved fold artifacts.','',
 '| Model | Price endpoint MSE | Price improvement | Return endpoint MSE | Return improvement |',
 '| --- | ---: | ---: | ---: | ---: |',
]
for model in MODELS:
 p=by[('future_close',model)];r=by[('future_log_return',model)]
 lines.append(f"| {LABELS[model]} | {float(p['endpoint_mse']):.6g} | {100*float(p['relative_improvement']):.2f}% | "
              f"{float(r['endpoint_mse']):.6g} | {100*float(r['relative_improvement']):.2f}% |")
lines+=['',
 'This is a different four-window subset from the 47-origin pooled comparison. The selected GRU return result is '
 f"{100*float(by[('future_log_return','gru')]['relative_improvement']):.2f}% relative to zero return, "
 'but the small set cannot establish a stable advantage. Nonoverlapping label windows can remain dependent; '
 'no confidence interval or significance claim is made. The data and predictions were already exposed during development.','']
text=text.replace(marker,'\n'.join(lines)+'\n'+marker,1)
old_ref=('17. Holm, S. “A Simple Sequentially Rejective Multiple Test Procedure.” *Scandinavian Journal of Statistics*, '
         '6(2), 65–70, 1979. https://www.jstor.org/stable/4615733')
new_ref=('17. Holm, S. “A Simple Sequentially Rejective Multiple Test Procedure.” *Scandinavian Journal of Statistics*, '
         '6, 65–70, 1979. https://www.jstor.org/stable/4615733')
require(text.count(old_ref)==1,'Holm reference changed')
text=text.replace(old_ref,new_ref,1)
old_note=('Reference metadata, versions, and the scope of prior reading must be checked again against the target venue '
          'before submission. References 13 and 17 do not imply that their associated procedures have already been run.')
new_note=('Reference metadata, versions, and the scope of prior reading must be checked again against the target venue '
          'before submission. The issue number for reference 17 is omitted pending verification. References 13 and 17 '
          'do not imply that their associated procedures have already been run.')
require(text.count(old_note)==1,'Reference note changed')
text=text.replace(old_note,new_note,1)

stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
directory=OUT/'private_working_manuscripts'/stamp
directory.mkdir(parents=True,exist_ok=False)
md=directory/'PRIVATE_WORKING_MANUSCRIPT_EN.md';md.write_text(text,encoding='utf8')
for name in ('PRIVATE_METRICS_POOLED.csv','PRIVATE_METRICS_BY_FOLD.csv','TRAINING_HISTORY_AUDIT.csv'):
 shutil.copy2(PARENT/name,directory/name)
shutil.copy2(SENS/'PRIVATE_ANNUAL_NONOVERLAP_RESULTS.csv',directory/'PRIVATE_ANNUAL_NONOVERLAP_RESULTS.csv')
audit={
 'status':'private_annual_sensitivity_integrated_not_for_submission',
 'created_at_utc':datetime.now(timezone.utc).isoformat(),
 'parent_directory':str(PARENT),'parent_manuscript_sha256':sha(parent_md),
 'parent_provenance_audit_sha256':sha(PARENT/'PRIVATE_PROVENANCE_AUDIT.json'),
 'parent_word_verification_sha256':sha(PARENT/'PRIVATE_WORD_VERIFICATION.json'),
 'parent_reference_verification_sha256':sha(parent_ref),
 'pool_audit_path':str(OLDMETRICS/'PRIVATE_POOL_VERIFICATION.json'),
 'pool_audit_sha256':sha(OLDMETRICS/'PRIVATE_POOL_VERIFICATION.json'),
 'metrics_parent_manuscript_sha256':load(OLDMETRICS/'PRIVATE_POOL_VERIFICATION.json')['manuscript_sha256'],
 'annual_selection_path':str(selection_path),'annual_selection_sha256':sha(selection_path),
 'annual_results_path':str(result_path),'annual_results_sha256':sha(result_path),
 'annual_independent_check_path':str(independent_path),'annual_independent_check_sha256':sha(independent_path),
 'script_sha256':sha(Path(__file__).resolve()),'manuscript_sha256':sha(md),
 'csv_sha256':{name:sha(directory/name) for name in
               ('PRIVATE_METRICS_POOLED.csv','PRIVATE_METRICS_BY_FOLD.csv',
                'TRAINING_HISTORY_AUDIT.csv','PRIVATE_ANNUAL_NONOVERLAP_RESULTS.csv')},
 'checks':{'annual_nonoverlap_windows':4,'annual_nonoverlap_model_rows':16,
           'selection_rule_matches_plan':True,'independent_scalar_recalculation_matches':True,
           'holm_unverified_issue_omitted':True,'previous_detailed_metrics_preserved':True},
 'independent_test':False,'Q3_values_read':False,'training_started':False,'publication_rights':'not confirmed',
 'limitations':['Four nonoverlapping windows remain potentially dependent and are not an independent confirmation.',
                'Minute folds, publication rights, final protocol and conclusion remain unresolved.']
}
ap=directory/'PRIVATE_ANNUAL_MANUSCRIPT_AUDIT.json'
ap.write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
print(json.dumps({'status':audit['status'],'directory':str(directory),'manuscript':str(md),'audit':str(ap)},ensure_ascii=False))
