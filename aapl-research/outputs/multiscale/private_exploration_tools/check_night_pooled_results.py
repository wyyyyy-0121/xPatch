"""Independently audit weighted pooling and appendix cells for a private snapshot."""
from collections import defaultdict
import csv
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import re
import sys

root = Path(sys.argv[1]).resolve()
audit_path = root / 'PRIVATE_DETAILED_RESULTS_AUDIT.json'
md_path = root / 'PRIVATE_WORKING_MANUSCRIPT_EN.md'
pooled_path = root / 'PRIVATE_METRICS_POOLED.csv'
fold_path = root / 'PRIVATE_METRICS_BY_FOLD.csv'
out_path = root / 'PRIVATE_POOL_VERIFICATION.json'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def require(ok, message):
    if not ok:
        raise ValueError(message)

def read(path):
    with path.open(newline='', encoding='utf8') as f:
        return list(csv.DictReader(f))

def near(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=1e-16)

audit = json.loads(audit_path.read_text(encoding='utf8'))
require(digest(md_path) == audit['manuscript_sha256'] and digest(pooled_path) == audit['pooled_csv_sha256'] and
        digest(fold_path) == audit['fold_csv_sha256'], 'Detailed evidence output hash changed')
folds, pools = read(fold_path), read(pooled_path)
require(len(folds) == 480 and len(pools) == 96, 'Unexpected metrics inventory')
key = lambda row: (row['profile'], int(row['horizon']), row['target'], row['model'])
grouped = defaultdict(list)
for row in folds:
    grouped[key(row)].append(row)
pooled = {key(row): row for row in pools}
require(len(grouped) == 96 and len(pooled) == 96 and set(grouped) == set(pooled), 'Task-model keys differ')
numeric_fields = ('endpoint_mse', 'endpoint_mae', 'path_mse', 'path_mae')
with localcontext() as ctx:
    ctx.prec = 40
    for k, rows in grouped.items():
        rows.sort(key=lambda x: int(x['fold']))
        require([int(x['fold']) for x in rows] == [1, 2, 3, 4, 5], 'Fold set differs')
        n = sum(int(x['origins']) for x in rows)
        item = pooled[k]
        require(n == int(item['origins']) and int(item['folds']) == 5, 'Pooled origins differ')
        for field in numeric_fields:
            independent = sum(Decimal(x[field]) * int(x['origins']) for x in rows) / Decimal(n)
            require(near(independent, item[field]), 'Pooled weighted metric differs: ' + field)
        require(near(math.sqrt(float(item['endpoint_mse'])), item['endpoint_rmse']), 'Pooled RMSE differs')
        active = sum(int(x['active_count']) for x in rows)
        correct = sum(int(x['correct_active_count']) for x in rows)
        require(0 <= correct <= active <= n and active == int(item['active_count']) and
                correct == int(item['correct_active_count']), 'Directional counts differ')
        require(near(Decimal(active) / Decimal(n), item['direction_coverage']), 'Directional coverage differs')
        if active:
            require(item['direction_accuracy_when_active'] and
                    near(Decimal(correct) / Decimal(active), item['direction_accuracy_when_active']),
                    'Active directional accuracy differs')
        else:
            require(item['direction_accuracy_when_active'] == '', 'Abstention accuracy must be undefined')
        base = pooled[(k[0], k[1], k[2], 'last_close')]
        require(near(Decimal(1) - Decimal(item['endpoint_mse']) / Decimal(base['endpoint_mse']),
                     item['relative_improvement']), 'Pooled baseline-relative result differs')

labels = {'last_close':'Persistence / zero','drift':'Drift','seasonal':'Seasonal','ridge':'Ridge',
          'mlp':'MLP','gru':'GRU','lstm':'LSTM','xpatch':'xPatch'}
models = ('last_close','drift','seasonal','ridge','mlp','gru','lstm','xpatch')
tasks = [('annual',12,'Annual H12 (short history)'),('quarterly',3,'Quarterly H3'),
         ('monthly',21,'Monthly H21'),('hourly',1,'Hourly H1'),('hourly',4,'Hourly H4'),
         ('hourly',8,'Hourly H8')]
body = md_path.read_text(encoding='utf8').split('## Appendix A. Detailed historical development results',1)[1]
checked = 0
for profile,horizon,label in tasks:
    for target,title in [('future_close','nominal close'),('future_log_return','cumulative log return')]:
        marker = f'### {label}: {title}'
        require(body.count(marker) == 1, 'Appendix task heading differs')
        segment = body.split(marker,1)[1].split('\n### ',1)[0]
        lines = segment.splitlines()
        header = '| Model | End MSE | End RMSE | End MAE | Path MSE | Improvement | Direction coverage | Active accuracy |'
        require(lines.count(header) == 1, 'Appendix table header differs')
        i = lines.index(header)
        require(lines[i+1].startswith('| ---'), 'Appendix separator missing')
        rows=[]
        for line in lines[i+2:]:
            if not line.startswith('|'):
                break
            rows.append([x.strip() for x in line.strip('|').split('|')])
        require(len(rows) == 8, 'Appendix model rows differ')
        ref = pooled[(profile,horizon,target,'last_close')]
        require(f"{int(ref['origins']):,} origins" in segment and ref['decision_first_ny'] in segment and
                ref['decision_last_ny'] in segment and ref['target_last_completion_ny'] in segment,
                'Appendix task date or count differs')
        for model,row in zip(models,rows):
            x=pooled[(profile,horizon,target,model)]
            acc = '—' if not x['direction_accuracy_when_active'] else f"{100*float(x['direction_accuracy_when_active']):.2f}%"
            expected=[labels[model],f"{float(x['endpoint_mse']):.6g}",f"{float(x['endpoint_rmse']):.6g}",
                      f"{float(x['endpoint_mae']):.6g}",f"{float(x['path_mse']):.6g}",
                      f"{100*float(x['relative_improvement']):.2f}%",f"{100*float(x['direction_coverage']):.2f}%",acc]
            require(row == expected, 'Appendix row differs: ' + label + ' ' + model)
            checked += 7

receipt={'status':'private_pooled_metrics_and_appendix_verified',
         'created_at_utc':datetime.now(timezone.utc).isoformat(),
         'manuscript_sha256':digest(md_path),'pooled_csv_sha256':digest(pooled_path),
         'fold_csv_sha256':digest(fold_path),'source_audit_sha256':digest(audit_path),
         'script_sha256':digest(Path(__file__).resolve()),
         'checks':{'fold_rows':len(folds),'pooled_rows':len(pools),'five_folds_per_row':True,
                   'decimal_weighted_mse_mae_and_path_errors_match':True,'pooled_rmse_matches':True,
                   'directional_counts_and_abstentions_match':True,'baseline_relative_results_match':True,
                   'appendix_numeric_cells_match_csv':checked},
         'independent_test':False,'publication_rights':'not confirmed'}
require(not out_path.exists(), 'Pool verification receipt already exists')
out_path.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
print(json.dumps({'status':receipt['status'],'path':str(out_path),'checks':receipt['checks']},ensure_ascii=False))
