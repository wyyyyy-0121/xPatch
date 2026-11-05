"""F6 pre-confirmation system freeze; does not open Q2 labels or any Q3 values."""
import numpy as np

from .common import OUT,PLAN,digest,identity,read_json,write_json,utc_now,source_identity
from .evaluation import point_scores,minimum_detectable_improvement
from .experiment import admitted,score_rows
from .panel import Panel
from .splits import VALIDATION
from .accounting import totals


def prepare():
    specification=admitted();results=read_json(OUT/'F5_complete.json');panel=Panel.load()
    if (OUT/'system_freeze.json').exists():raise ValueError('A system is already frozen; Q2 cannot select a replacement')
    selected=results['selected'];config=selected['config'];family='lora' if config.get('foundation_mode')=='lora' else 'fusion'
    namespace='lora' if family=='lora' else ('mechanism_plain_mlp' if config.get('fusion_type')=='plain_mlp' else
               ('transfer_multi' if config['candidate_id'].startswith('transfer_') else 'fusion'))
    seed_path=OUT/'seed_verification'/namespace/config['candidate_id']
    if digest(seed_path/'predictions.npz')!=selected['prediction_sha256']:raise ValueError('Selected seed predictions changed')
    with np.load(seed_path/'predictions.npz') as saved:
        y=saved['target'].copy();prediction=saved['ensemble'].copy();quarters=saved['quarters'].copy()
        seeds=saved['predictions'].copy()
    single=[]
    for name in ('ridge','lightgbm','xpatch','patchtst'):
        selection=read_json(OUT/'selections'/('single_'+name+'.json'))
        best=selection['rungs'][-1]['rankings'][0]
        single.append({'family':name,'config':selection['selected'],'mse':best['mse']})
    single.sort(key=lambda row:(row['mse'],row['family']));best_single=single[0]
    best_simple=read_json(OUT/'selections/simple.json')['selected']
    gate={'pooled_improvement_at_least_1pct':selected['system']['mse_improvement_vs_zero']>=.01,
          'beats_fixed_strongest_single':selected['system']['mse']<best_single['mse'],
          'beats_fixed_strongest_simple':selected['system']['mse']<best_simple['mse'],
          'at_least_two_validation_quarters_improve':sum(r['mse_improvement_vs_zero']>0 for r in selected['quarter_scores'].values())>=2,
          'all_three_seeds_improve':all(r['mse_improvement_vs_zero']>0 for r in selected['individual_seeds'].values())}
    times=[]
    for quarter in VALIDATION:
        _,rows=score_rows(panel,quarter);times.extend(panel.dates[rows])
    if len(times)!=len(y):raise ValueError('Selected seed result does not cover exact validation origins')
    zero_delta=(prediction-y)**2-y*y
    power=minimum_detectable_improvement(zero_delta,y,np.asarray(times))
    value={'frozen_at_utc':utc_now(),'source_identity':source_identity(),'plan_sha256':digest(PLAN),
           'search_manifest_sha256':digest(OUT/'search_manifest.json'),'F5_sha256':digest(OUT/'F5_complete.json'),
           'selected_system':selected,'selected_namespace':namespace,'fixed_strongest_single':best_single,
           'fixed_strongest_simple':best_simple,'all_single_controls':single,
           'development_gate':gate,'development_gate_passed':all(gate.values()),
           'Q2_status':'known process confirmation; prior CSV access/calibration; one fixed system only',
           'Q3_access_authorized_by_this_freeze':False,'final_horizon_minutes':12,'primary_metric':'raw MSE',
           'statistics':{'block_days':5,'repeats':5000,'null_centered':True,'holm_controls':['zero','single','simple'],
                        'sensitivities':[1,10],'relative_improvement_threshold':.01,'alpha':.05,'minimum_months_improved':2},
           'minimum_detectable_improvement':power,'usage_at_freeze':totals(),
           'stop_rule':'If development or one Q2 confirmation fails, keep Q3 sealed and deliver the runnable frozen system and negative evidence'}
    value['identity']=identity(value);write_json(OUT/'system_freeze.json',value)
    return value


if __name__=='__main__':
    result=prepare();print({'development_gate_passed':result['development_gate_passed'],'checks':result['development_gate']})
