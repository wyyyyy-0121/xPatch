"""Check source evidence, every Word table cell and the rendered interim draft."""
from collections import Counter
from datetime import datetime,timezone
import csv,hashlib,json,re,subprocess,sys
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile


root=Path(sys.argv[1]).resolve()
md=root/'PRIVATE_WORKING_MANUSCRIPT_EN.md';raw=root/'PRIVATE_WORKING_MANUSCRIPT_RAW.docx'
word=root/'PRIVATE_WORKING_MANUSCRIPT_EN.docx';pdf=root/'PRIVATE_WORKING_MANUSCRIPT_EN.pdf'
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf8'))
def require(ok,msg):
 if not ok:raise ValueError(msg)
def command(args):return subprocess.run(args,capture_output=True,text=True,check=True).stdout
def document(path):
 with ZipFile(path) as z:
  d=ET.fromstring(z.read('word/document.xml'));rel=ET.fromstring(z.read('word/_rels/document.xml.rels'))
  tables=[]
  for t in d.findall('.//w:tbl',ns):
   rows=[]
   for row in t.findall('./w:tr',ns):
    rows.append([''.join(x.text or '' for x in cell.iter() if x.tag in (f"{{{ns['w']}}}t",f"{{{ns['m']}}}t")).strip() for cell in row.findall('./w:tc',ns)])
   tables.append(rows)
  return {'text':[x.text or '' for x in d.findall('.//w:t',ns)],
          'math':[''.join(x.itertext()) for x in d.findall('.//m:oMath',ns)],
          'links':sorted(x.attrib.get('Target','') for x in rel if x.attrib.get('Type','').endswith('/hyperlink')),
          'media':sorted(hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith('word/media/')),
          'tables':tables,'drawings':len(d.findall('.//w:drawing',ns))}
def markdown_tables(body):
 result=[];current=[]
 for line in body.splitlines():
  if line.startswith('|'):
   if re.match(r'^\|\s*:?-+',line):continue
   current.append([s.strip().replace('\\(','').replace('\\)','') for s in line.strip('|').split('|')])
  elif current:result.append(current);current=[]
 if current:result.append(current)
 return result

audit=load(root/'MANUSCRIPT_UPDATE_AUDIT.json');require(sha(md)==audit['manuscript_sha256'],'Markdown changed')
result=load(root/'all_task_results/RESULTS.json');require(sha(root/'all_task_results/RESULTS.json')==audit['results_sha256'],'Result evidence changed')
for name,h in result['csv_sha256'].items():require(sha(root/'all_task_results'/name)==h,'Result CSV changed')
before,after=document(raw),document(word)
for key in ('text','math','links','media','tables'):require(before[key]==after[key],'Formatting changed '+key)
require((len(after['math']),len(after['links']),len(after['tables']),after['drawings'])==(30,18,33,2),'Unexpected document structure')
body=md.read_text(encoding='utf8');require(markdown_tables(body)==after['tables'],'Markdown-to-Word table cells differ')
appendix=body.split('## Appendix A. Detailed historical development results',1)[1]
require(len(re.findall(r'^### ',appendix,re.M))==20,'Appendix lacks twenty tasks')
task_tables=markdown_tables(appendix);require(len(task_tables)==20,'Appendix table count differs')
models=('last_close','drift','seasonal','ridge','mlp','gru','lstm','xpatch')
ordered=sorted(result['tasks'],key=lambda t:(('annual','quarterly','monthly','hourly','minute').index(t['profile']),t['horizon'],t['target']))
checked=0
for task,t in zip(ordered,task_tables):
 require(len(t)==9,'A task table lacks eight models')
 for model,row in zip(models,t[1:]):
  values=task['pooled']['models'][model]
  expected=[f'{values[k]:.6g}' for k in ('endpoint_mse','endpoint_rmse','endpoint_mae','path_mse')]
  expected += ['—' if values[k] is None else f'{100*values[k]:.2f}%' for k in ('relative_improvement','direction_coverage','direction_accuracy_when_active')]
  require(row[1:]==expected,'Appendix numeric values differ from recomputed results')
  checked+=len(expected)
a=command(['pandoc','--from=markdown+tex_math_single_backslash+autolink_bare_uris','--to=plain','--wrap=none',str(md)])
b=command(['pandoc','--from=docx','--to=plain','--wrap=none',str(word)])
pattern=r"[A-Za-z]{3,}(?:[’'-][A-Za-z]+)?"
ca,cb=Counter(re.findall(pattern,a)),Counter(re.findall(pattern,b))
for key in ('mathrm','qquad','quad','mspace','leq'):ca.pop(key,None);cb.pop(key,None)
require(ca==cb,'Prose word counts changed')
info=command(['pdfinfo',str(pdf)]);match=re.search(r'^Pages:\s+(\d+)\s*$',info,re.M);require(match is not None,'PDF page count missing')
text=command(['pdftotext','-layout','-enc','UTF-8',str(pdf),'-']);pages=[p for p in text.split('\f') if p.strip()]
require(len(pages)==int(match[1])==19 and all('PRIVATE WORKING DRAFT' in p for p in pages),'PDF pages or footers differ')
for phrase in (f"{result['completed_folds']} of 100",'320 task','eight updates','03:30:01','179,288','References'):
 require(phrase in text,'PDF lacks expected text: '+phrase)
with (root/'RENDERED_TEXT.txt').open('x',encoding='utf8') as f:f.write(text)
receipt={'status':'all_twenty_task_working_manuscript_content_verified','created_at_utc':datetime.now(timezone.utc).isoformat(),
 'markdown_sha256':sha(md),'raw_docx_sha256':sha(raw),'formatted_docx_sha256':sha(word),'rendered_pdf_sha256':sha(pdf),
 'manuscript_audit_sha256':sha(root/'MANUSCRIPT_UPDATE_AUDIT.json'),'verification_script_sha256':sha(Path(__file__)),
 'checks':{'editable_equations':30,'hyperlinks':18,'tables':33,'body_drawings':2,'appendix_tasks':20,
           'appendix_numeric_fields_matched_to_recomputed_results':checked,'all_markdown_table_cells_match_word':True,
           'body_text_equations_links_media_tables_preserved_by_formatting':True,'prose_word_counts_match':True,
           'rendered_pages':19,'all_pages_have_working_draft_footer':True},
 'visual_review_pending':True,'completed_folds':result['completed_folds'],'complete_tasks':result['complete_tasks'],
 'independent_test':False,'Q3_values_read':False}
with (root/'WORD_CONTENT_VERIFICATION.json').open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps({'status':receipt['status'],'checks':receipt['checks']}))
