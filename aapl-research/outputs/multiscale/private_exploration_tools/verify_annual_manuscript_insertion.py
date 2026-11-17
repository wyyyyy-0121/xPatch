"""Verify annual-sensitivity manuscript insertion and source preservation."""
from datetime import datetime,timezone
from hashlib import sha256
import csv,json
from pathlib import Path
import sys

root=Path(sys.argv[1]).resolve()
audit_path=root/'PRIVATE_ANNUAL_MANUSCRIPT_AUDIT.json'
audit=json.loads(audit_path.read_text(encoding='utf8'))
md=root/'PRIVATE_WORKING_MANUSCRIPT_EN.md'
parent=Path(audit['parent_directory'])/'PRIVATE_WORKING_MANUSCRIPT_EN.md'
def digest(p):return sha256(Path(p).read_bytes()).hexdigest()
def require(ok,msg):
 if not ok:raise ValueError(msg)
require(digest(md)==audit['manuscript_sha256'] and digest(parent)==audit['parent_manuscript_sha256'],
        'Manuscript identity differs')
selection_path=Path(audit['annual_selection_path']);results_path=Path(audit['annual_results_path'])
verification_path=Path(audit['annual_independent_check_path'])
require(digest(selection_path)==audit['annual_selection_sha256'] and digest(results_path)==audit['annual_results_sha256'] and
        digest(verification_path)==audit['annual_independent_check_sha256'],'Sensitivity evidence changed')
selection=json.loads(selection_path.read_text());results=json.loads(results_path.read_text())
with (root/'PRIVATE_ANNUAL_NONOVERLAP_RESULTS.csv').open(encoding='utf8',newline='') as f:
 rows=list(csv.DictReader(f))
require(digest(root/'PRIVATE_ANNUAL_NONOVERLAP_RESULTS.csv')==audit['csv_sha256']['PRIVATE_ANNUAL_NONOVERLAP_RESULTS.csv'],
        'Sensitivity CSV changed')
by={(x['target'],x['model']):x for x in rows}
models=('last_close','drift','seasonal','ridge','mlp','gru','lstm','xpatch')
labels=('Persistence / zero','Drift','Seasonal','Ridge','MLP','GRU','LSTM','xPatch')
text=md.read_text(encoding='utf8')
heading='### Annual sensitivity: nonoverlapping forecast windows'
marker='### Split sensitivity and label definition'
require(text.count(heading)==1 and text.count(marker)==1,'Sensitivity section boundaries differ')
prefix,rest=text.split(heading,1);section,suffix=rest.split(marker,1)
for key in ('first_origin_session','last_origin_session','first_target_session','last_target_session'):
 require(selection[key] in section,'Sensitivity date missing')
header='| Model | Price endpoint MSE | Price improvement | Return endpoint MSE | Return improvement |'
lines=section.splitlines();require(lines.count(header)==1,'Sensitivity table missing')
i=lines.index(header);actual=[]
for line in lines[i+2:]:
 if not line.startswith('|'):break
 actual.append([x.strip() for x in line.strip('|').split('|')])
require(len(actual)==8,'Sensitivity row count differs')
for model,label,row in zip(models,labels,actual,strict=True):
 p=by[('future_close',model)];r=by[('future_log_return',model)]
 expected=[label,f"{float(p['endpoint_mse']):.6g}",f"{100*float(p['relative_improvement']):.2f}%",
           f"{float(r['endpoint_mse']):.6g}",f"{100*float(r['relative_improvement']):.2f}%"]
 require(row==expected,'Sensitivity table differs: '+model)
reconstructed=prefix+marker+suffix
new_ref=('17. Holm, S. “A Simple Sequentially Rejective Multiple Test Procedure.” *Scandinavian Journal of Statistics*, '
         '6, 65–70, 1979. https://www.jstor.org/stable/4615733')
old_ref=('17. Holm, S. “A Simple Sequentially Rejective Multiple Test Procedure.” *Scandinavian Journal of Statistics*, '
         '6(2), 65–70, 1979. https://www.jstor.org/stable/4615733')
new_note=('Reference metadata, versions, and the scope of prior reading must be checked again against the target venue '
          'before submission. The issue number for reference 17 is omitted pending verification. References 13 and 17 '
          'do not imply that their associated procedures have already been run.')
old_note=('Reference metadata, versions, and the scope of prior reading must be checked again against the target venue '
          'before submission. References 13 and 17 do not imply that their associated procedures have already been run.')
require(reconstructed.count(new_ref)==1 and reconstructed.count(new_note)==1,'Reference edit differs')
reconstructed=reconstructed.replace(new_ref,old_ref,1).replace(new_note,old_note,1)
require(reconstructed==parent.read_text(encoding='utf8'),'Unrelated parent manuscript text changed')
out=root/'PRIVATE_ANNUAL_INSERTION_VERIFICATION.json'
require(not out.exists(),'Insertion verification already exists')
record={'status':'private_annual_sensitivity_manuscript_insertion_verified',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'manuscript_sha256':digest(md),'source_audit_sha256':digest(audit_path),
        'selection_sha256':digest(selection_path),'results_sha256':digest(results_path),
        'script_sha256':digest(Path(__file__).resolve()),
        'checks':{'model_rows':8,'numeric_cells_match':32,'selected_dates_match':True,
                  'other_parent_text_preserved':True,'unverified_holm_issue_omitted':True},
        'limitations':['The four windows are exposed development evidence and potentially dependent.',
                       'This check does not establish data rights or independent confirmation.']}
out.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
print(json.dumps({'status':record['status'],'path':str(out),'checks':record['checks']},ensure_ascii=False))
