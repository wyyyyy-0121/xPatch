"""Illustrative CPU sizing from one actual private minute development fold."""
from datetime import datetime,timezone
import json
from pathlib import Path

from multiscale.bars import file_digest
from multiscale.spec import OUT


def main():
    minute=OUT/'private_minute_full_neural/20261107T140317864464Z'
    state=json.loads((minute/'PRIVATE_STATE.json').read_text());check=json.loads((minute/'PRIVATE_INDEPENDENT_CHECK.json').read_text())
    if state.get('status')!='completed' or check.get('status')!='private_full_minute_neural_independently_recomputed':
        raise ValueError('A fully verified first fold is required')
    ap=OUT/'admission/alpaca_20261107T052228573242Z/admission.json';ad=json.loads(ap.read_text())
    rows=[r for r in ad['tasks'] if r['profile']=='minute' and r['target']=='future_log_return']
    if sorted(r['horizon'] for r in rows)!=[5,15,30,60]:raise ValueError('Expected four minute horizons')
    base=next(r for r in rows if r['horizon']==60)['folds'][0]['inner_fit_origins'];wall=state['elapsed_seconds']
    ratios={str(r['horizon']):sum(f['inner_fit_origins']/base for f in r['folds']) for r in rows}
    all_seconds=wall*2*sum(ratios.values());remaining_seconds=all_seconds-wall
    out={'status':'illustrative_private_cpu_scenario_not_a_budget_approval',
         'created_at_utc':datetime.now(timezone.utc).isoformat(),
         'method':'multiply measured first-fold wall time by inner-fit-origin ratios, twice for price and return targets',
         'measured_first_fold_seconds':wall,'measured_fit_origins':base,'threads':2,'device':'cpu',
         'measured_epochs':{n:v['epochs'] for n,v in state['models'].items()},
         'measured_model_seconds':{n:v['model_elapsed_seconds'] for n,v in state['models'].items()},
         'fold_fit_origin_ratios_by_horizon':ratios,'planned_minute_tasks':8,'planned_folds':40,
         'all_neural_folds_scenario_hours':all_seconds/3600,'remaining_neural_folds_scenario_hours':remaining_seconds/3600,
         'remaining_plus_30pct_scenario_hours':remaining_seconds*1.3/3600,
         'minute_state_sha256':file_digest(minute/'PRIVATE_STATE.json'),
         'minute_verification_sha256':file_digest(minute/'PRIVATE_INDEPENDENT_CHECK.json'),'admission_sha256':file_digest(ap),
         'limits':['One first-fold measurement only; the largest fold has not been benchmarked on this CPU.',
                   'Assumes fit-size scaling and the same actual epoch counts; stopping, prediction, target and horizon costs can differ.',
                   'Not an upper bound, confidence interval, GPU estimate, approved compute budget or cost quote.',
                   'Controls, additional seeds, retries and formal publication rights are not included.']}
    directory=OUT/'private_compute_projections';directory.mkdir(parents=True,exist_ok=True)
    p=directory/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    with p.open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps({'status':out['status'],'path':str(p),'first_fold_seconds':round(wall,1),
                      'all_eight_tasks_hours':round(out['all_neural_folds_scenario_hours'],1),
                      'remaining_plus_30pct_hours':round(out['remaining_plus_30pct_scenario_hours'],1)}))


if __name__=='__main__':main()
