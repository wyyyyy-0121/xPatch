"""Generate identity/design metadata. Does not decode array members."""
from pathlib import Path
from runtime import BASE, read, write, sha, require, MEMBERS

def main():
    repo=BASE.parents[3]
    # BASE is outputs/multiscale/research_review/<stage>.
    require((repo/'multiscale').is_dir(),'Repository root')
    old=BASE.parent/'20261110_function_centering_matched_probe'
    previous=read(BASE.parent/'20261110_centering_prediction_accounting/PROTOCOL.json')
    receipt=read(old/'market_runs/market/INPUT_RECEIPT.json')
    source=[old/'code/native_core.py',old/'code/matched_worker.py',
            repo/'outputs/multiscale/mechanism_repair/20261109_v3/code/reference_source/multiscale/features.py']
    states=[]
    for entry,part in zip(previous['inputs'],receipt['state_partitions']):
        require(entry['state']==part['id'],'State order')
        states.append(entry|{'count':part['tail_count'],'horizon':60 if entry['state'].startswith('minute') else 8,
                             'tail_sha256':part['tail_sha256'],'partition':part})
    spec={'schema':'level_constraint_design_v1','created_before_new_array_decode':True,
          'states':states,'permitted_members':[k+'.npy' for k in MEMBERS],
          'source_files':{str(p):sha(p) for p in source},
          'source_metadata':{str(old/'market_runs/market'/name):sha(old/'market_runs/market'/name)
                             for name in ['INPUT_RECEIPT.json','GLOBAL_FREEZE.json','RESULTS.json']},
          'controls_for_gate':['same_information_Ridge','zero','prefix_update_mean','half_Ridge','half_g'],
          'state_minimum_fractional_improvement':.001,'equal_fold_primary_minimum_fractional_improvement':.01,
          'total_gate_checks':42,'numeric_identity_eps_factor':256,'independent_loss_rtol':1e-11,
          'independent_percentage_point_atol':1e-8,'Q3_allowed':False,
          'new_market_fitting_allowed':False,'market_neural_inference_allowed':False,
          'runtime_seconds':180,'maximum_analysis_CPUs':[24,26],'GPU':False,
          'old_accounting_remains_closed':True,'not_independent_market_confirmation':True}
    require(not (BASE/'DESIGN_SPEC.json').exists(),'Do not replace first design')
    write(BASE/'DESIGN_SPEC.json',spec)

if __name__=='__main__': main()
