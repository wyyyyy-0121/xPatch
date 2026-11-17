"""Synthetic-only evaluator mapping/statistics acceptance before market scoring."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util,hashlib,json,os,sys
import numpy as np

R=Path(__file__).resolve().parent;B=R.parent;V=B.with_name('20261109_v3')
sys.path.insert(0,str(R))
from verify_context import independently_bootstrap

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert sorted(os.sched_getaffinity(0))==[24,26] and os.environ['CUDA_VISIBLE_DEVICES']=='-1'
    file=B/'code/evaluate_context.py';before=sha(file)
    spec=importlib.util.spec_from_file_location('synthetic_context_evaluator',file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rng=np.random.default_rng(144);idx=np.arange(11);y=rng.normal(0,.02,(11,3));truth=y[:,-1]
    task='synthetic_h3_future_log_return';fold=1
    arrays={};receipts={};expected={'zero':truth**2}
    candidate={'loss_zero':truth**2,'loss_original':np.arange(11)*.001+.01,'loss_original_shrink':np.arange(11)*.001+.02}
    expected['archived_default_raw']=candidate['loss_original'];expected['archived_default_shrink']=candidate['loss_original_shrink']
    factorial={'loss_zero':truth**2}
    for i,name in enumerate(['history','history_anchor','history_selected','history_anchor_selected']):
        value=np.arange(11)*.002+.03+i
        candidate['loss_ridge_'+name]=value;factorial['loss_ridge_'+name]=value;expected['ridge_'+name]=value
    mean={'loss_baseline':truth**2,'loss_raw':np.arange(11)*.003+.04,'loss_shrink':np.arange(11)*.003+.05}
    expected['fit_mean_raw']=mean['loss_raw'];expected['fit_mean_shrink']=mean['loss_shrink']
    arrays[str(module.V3/'analysis/candidate_paired_losses'/task/'fold_1.npz')]=candidate
    arrays[str(module.V3/'analysis/paired_losses'/task/'xpatch'/'fold_1.npz')]=factorial
    arrays[str(module.V3A/'mean_controls/predictions'/task/'fold_1.npz')]=mean
    for label,variant in [('default','anchor_random'),('repair','anchor_baseline')]:
        per={'raw':[],'shrink':[]}
        for s,alpha in [(2026,.1),(2027,.4),(2028,.8)]:
            job=f'{task}_xpatch_{variant}_f1_s{s}';raw=rng.normal(0,.03,(11,3));shrunk=raw*alpha
            folder=module.V3/'runs'/job;cal=module.V3/'calibration'/job
            arrays[str(folder/'predictions.npz')]={'trained':raw}
            arrays[str(cal/'predictions.npz')]={'prediction':shrunk}
            receipts[str(folder/'DONE.json')]={'prediction_sha256':'synthetic','checkpoint_sha256':'ck'+job}
            receipts[str(cal/'DONE.json')]={'prediction_sha256':'synthetic','source_checkpoint_sha256':'ck'+job,'alpha':alpha}
            for mode,p in [('raw',raw),('shrink',shrunk)]:
                error=(p[:,-1]-truth)**2;expected[f'{label}_seed_{s}_{mode}']=error;per[mode].append(error)
        for mode,key in [('raw','trained'),('shrink','shrink')]:
            pooled=np.mean(per[mode],axis=0);expected[f'{label}_three_seed_{mode}']=pooled;factorial[f'loss_{variant}__{key}']=pooled
    module.read=lambda p:receipts[str(p)]
    module.load_arrays=lambda p,origins,expected_sha=None:{k:v.copy() for k,v in arrays[str(p)].items()}
    loaded=module.load_controls(task,fold,idx,y)
    assert len(loaded)==25 and set(loaded)==set(expected)
    assert all(np.array_equal(loaded[k],expected[k]) for k in loaded)
    for cell in module.CELLS:
        for mode in ('raw','shrink'):loaded[cell+'_'+mode]=rng.uniform(.01,.1,len(idx))
    controls=[k for k in loaded if k not in ('D_joint_raw','D_joint_shrink')]
    assert len(loaded)==33 and len(controls)==31 and set(module.DIAGNOSTIC_PAIRS)=={('D_joint_raw','C_history_raw'),('D_joint_raw','B_context_raw'),('B_context_raw','A_mean_raw')}
    names=list(loaded);groups=[];independent=[]
    for f in range(5):
        dates=np.asarray([f'2024-{f+1:02d}-{d+1:02d}' for d in range(10)])
        bars=SimpleNamespace(session_date=np.repeat(dates,4))
        count=np.asarray([1,3,0,4,2,1,3,0,2,4]);origins=np.concatenate([np.arange(d*4,d*4+c) for d,c in enumerate(count)])
        losses={name:rng.uniform(.01,.1,len(origins)) for name in names}
        group=module.day_groups(bars,origins,losses);groups.append(group)
        assert len(group['counts'])==10 and np.array_equal(group['counts'],count)
        positions=origins//4
        sums=np.column_stack([np.bincount(positions,weights=losses[name],minlength=10) for name in names])
        independent.append((count,sums))
    pairs=[('D_joint_shrink',c) for c in controls]+module.DIAGNOSTIC_PAIRS
    n=0
    for block in (1,2,5):
        expected_intervals=independently_bootstrap(independent,names,pairs,block)
        result=module.paired_intervals(groups,pairs,block,task)
        for row in result:
            wanted=expected_intervals[(row['primary'],row['reference'],block)]
            for key,value in wanted.items():
                if isinstance(value,(str,bool)) or value is None:assert row[key]==value,(key,row[key],value)
                else:assert np.allclose(row[key],value,rtol=5e-12,atol=2e-13),(key,row[key],value)
            n+=1
    # Static boundary assertions are paired with the full market verifier later.
    source=file.read_text()
    static={
        'candidate_fixed':module.CANDIDATE=='D_joint_shrink',
        'all31controls_no_Dself':"assert len(controls) == 31" in source,
        'evaluation_freeze_hashes_inputs':"'input_hashes': inputs" in source and "'evaluator_sha256': sha(__file__)" in source,
        'old_control_saved_alpha_only':"shrunk, raw * cal['alpha']" in source,
        'deadline_and_signal_guard':"if STOP or time.time() >= DEADLINE" in source,
        'resume_verifies_fold_and_freeze':"receipt['evaluation_freeze_sha256'] == sha(BASE / 'protocol/EVALUATION_FREEZE.json')" in source,
        'inner_gate_kept_for_confirmation':"'independent_confirmation_eligible': perf and inner" in source,
        'path_losses_separate':"analysis/path_losses" in source or "'analysis/path_losses'" in source or "'analysis' / 'path_losses'" in source,
        'source_unchanged_during_review':sha(file)==before}
    assert all(static.values()),static
    receipt={'status':'passed_synthetic_and_static_pre_outer_review','evaluator_sha256':before,'legacy_mapping_columns_verified':25,'endpoint_loss_columns':33,'candidate_controls':31,'synthetic_interval_rows_verified':n,'synthetic_zero_origin_days_per_fold':2,'synthetic_block_lengths':[1,2,5],'static_checks':static,'blockers':[],'market_stop_scores_read':False,'market_outer_scores_read':False,'Q3_values_read':False,'source':str(file),'reviewer_sha256':sha(Path(__file__))}
    (R/'EVALUATOR_PRE_REVIEW.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
