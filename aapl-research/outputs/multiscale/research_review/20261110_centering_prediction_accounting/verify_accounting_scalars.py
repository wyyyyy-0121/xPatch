"""Frozen scalar-only check; no array/checkpoint reads or numerical imports."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import os

BASE = Path(__file__).resolve().parent
OLD = BASE.parent / '20261110_function_centering_matched_probe/market_runs/market/RESULTS.json'
STATES = [('minute_f1',12853,60),('minute_f2',25753,60),('hourly_f1',228,8),('hourly_f2',459,8)]
PREDICTORS = ['D','Z','C','zero','prefix_update_mean','same_information_Ridge']
ATOL, RTOL = 1e-18, 1e-12

def main():
    assert sorted(os.sched_getaffinity(0)) == [26]
    paths = [BASE/'RESULTS.json', OLD]
    raw = [p.read_bytes() for p in paths]
    new, old = [json.loads(v) for v in raw]
    checks = []
    def close(name, actual, expected):
        assert isinstance(actual,(float,int)) and isinstance(expected,(float,int))
        assert math.isfinite(actual) and math.isfinite(expected)
        difference = abs(actual-expected)
        assert difference <= ATOL+RTOL*max(abs(actual),abs(expected)), name
        checks.append({'name':name,'max_abs':difference,'passed':True})
    assert new['status'] == 'completed_descriptive_per_horizon_accounting'
    assert old['status'] == 'completed_all_fixed_states'
    assert not old['decision']['continue_to_formal_development_review']
    assert [(r['state'],r['count'],r['horizon']) for r in new['states']] == STATES
    assert [(r['state'],r['count'],r['horizon']) for r in old['results']] == STATES
    for key in ['new_training_updates','new_model_inference']: assert new[key] == 0
    for key in ['new_candidate_admitted','independent_market_confirmation','Q3_read','full_goal_achieved','deadline_reset']: assert new[key] is False
    assert new['C_recipe_remains_closed'] is True
    for row, previous in zip(new['states'],old['results']):
        sid, horizon = row['state'], row['horizon']
        assert list(row['predictors']) == PREDICTORS
        common = row['predictors']['zero']['per_horizon']
        for name, pred in row['predictors'].items():
            ph, summary = pred['per_horizon'], pred['summary']
            assert set(ph) == {'mean_y','mean_prediction','prediction_variance','covariance_y_prediction','zero_MSE','MSE','MSE_reduction','level_contribution','dynamic_contribution'}
            assert all(len(v)==horizon and all(math.isfinite(x) for x in v) for v in ph.values())
            for k in range(horizon):
                my,mp,vp,cov,z,mse,delta,level,dynamic = [ph[x][k] for x in ['mean_y','mean_prediction','prediction_variance','covariance_y_prediction','zero_MSE','MSE','MSE_reduction','level_contribution','dynamic_contribution']]
                assert vp >= 0 and z > 0 and mse >= 0
                for label,a,e in [('level',level,2*my*mp-mp*mp),('dynamic',dynamic,2*cov-vp),('delta',delta,z-mse),('sum',delta,level+dynamic),('same_ymean',my,common['mean_y'][k]),('same_zeroMSE',z,common['zero_MSE'][k])]: close(f'{sid}:{name}:h{k+1}:{label}',a,e)
                if name in ['zero','prefix_update_mean']:
                    for label,value in [('variance',vp),('covariance',cov),('dynamic',dynamic)]: close(f'{sid}:{name}:h{k+1}:{label}',value,0.)
                if name == 'zero':
                    for label,value in [('mean',mp),('level',level),('delta',delta)]: close(f'{sid}:zero:h{k+1}:{label}',value,0.)
            assert set(summary) == {'endpoint','path'}
            for group, take in [('endpoint',lambda x:x[-1]),('path',lambda x:math.fsum(x)/horizon)]:
                values = summary[group]; z = take(ph['zero_MSE'])
                assert set(values) == {'zero_MSE','MSE','MSE_reduction','level_contribution','dynamic_contribution','total_gain_percent','level_gain_percentage_points','dynamic_gain_percentage_points'}
                for key in ['zero_MSE','MSE','MSE_reduction','level_contribution','dynamic_contribution']: close(f'{sid}:{name}:{group}:{key}',values[key],take(ph[key]))
                close(f'{sid}:{name}:{group}:old_MSE',values['MSE'],previous['predictors'][name][group+'_MSE'])
                close(f'{sid}:{name}:{group}:old_zeroMSE',z,previous['predictors']['zero'][group+'_MSE'])
                for shown,source in [('total_gain_percent','MSE_reduction'),('level_gain_percentage_points','level_contribution'),('dynamic_gain_percentage_points','dynamic_contribution')]: close(f'{sid}:{name}:{group}:{shown}',values[shown],100*values[source]/z)
                close(f'{sid}:{name}:{group}:percentage_sum',values['total_gain_percent'],values['level_gain_percentage_points']+values['dynamic_gain_percentage_points'])
    output = BASE/'INDEPENDENT_SCALAR_VERIFICATION.json'
    assert not output.exists()
    result = {'status':'passed_independent_scalar_accounting_verification','at_UTC':datetime.now(timezone.utc).isoformat(),'CPU':[26],'numeric_atol':ATOL,'numeric_rtol':RTOL,'checks_passed':len(checks),'checks':checks,'inputs':[{'path':str(p),'sha256':hashlib.sha256(v).hexdigest()} for p,v in zip(paths,raw)],'arrays_or_checkpoints_read':False,'numerical_libraries_imported':False,'moments_recomputed_from_arrays':False,'scope':'Algebra and aggregation consistency of saved scalar moments only; not fresh moment reconstruction, causal evidence or a continuation gate.','new_training_updates':0,'new_candidate_admitted':False,'C_recipe_remains_closed':True,'independent_market_confirmation':False,'Q3_read':False,'full_goal_achieved':False}
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    temporary.replace(output)
    print(json.dumps({'status':result['status'],'checks_passed':len(checks)}))

if __name__ == '__main__': main()
