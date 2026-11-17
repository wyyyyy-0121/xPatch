"""Small input-only checkpoint replay and loss-statistics capsule."""
from pathlib import Path
import os,sys,json,shutil,time
import numpy as np
import torch
import diagnostic_v2 as d
from portable_stats import day_groups

HERE=Path(__file__).resolve().parent

def main():
    assert sorted(os.sched_getaffinity(0))==[24,26] and os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    result=d.read(HERE/'RESULTS.json');controls=d.read(HERE/'CONTROL_AUDIT.json');root=HERE/'replay_capsule'
    root.mkdir(exist_ok=True);code=root/'code';code.mkdir(exist_ok=True)
    shutil.copytree(d.V3/'code/reference_source',code/'reference_source',dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('portable_predictor.py','portable_stats.py','tiny_architecture.py'):shutil.copy2(d.V3/'code'/name,code/name)
    for path in (HERE/'capsule_code').glob('*.py'):shutil.copy2(path,code/path.name)
    shutil.copy2(HERE/'CONTROL_AUDIT.json',root/'CONTROL_AUDIT.json')
    cases=[]
    for task,summary in result['tasks'].items():
        for row in summary['cell_receipts']:
            cfg=row['job'];data=d.Data(cfg);folder=HERE/'cells'/cfg['id'];inner=d.read(folder/'INNER_DONE.json')
            idx=data.splits['validation'][:128];history,anchor=data.windows.arrays(idx)
            with np.load(folder/'adapted_predictions.npz',allow_pickle=False) as a:
                assert np.array_equal(a['origins'][:len(idx)],idx);expected=a['prediction'][:len(idx)]
            cp=root/'checkpoints'/(cfg['id']+'.pt');cp.parent.mkdir(exist_ok=True);shutil.copy2(folder/'adapted.pt',cp)
            case=root/'cases'/(cfg['id']+'.npz')
            d.npz(case,origins=idx,log_relative_history=history,current_close=anchor,
                expected_raw=expected,expected_shrink=inner['adapted']['alpha']*expected,
                decision_ns=data.bars.end_ns[idx])
            cases.append({'job':cfg['id'],'task':task,'fold':cfg['fold'],'seed':cfg['seed'],
                'scope':summary['scope'],'origins':len(idx),'file':str(case.relative_to(root)),
                'file_sha256':d.sha(case),'checkpoint':str(cp.relative_to(root)),
                'checkpoint_sha256':d.sha(cp),'alpha':inner['adapted']['alpha']})
    d.atomic(root/'CASE_MANIFEST.json',{'cases':cases,'case_count':len(cases),'selection':'fixed first canonical128 validation origins per retained cell, no score selection',
        'inputs':'past-only split-aware log-relative close windows and known current close','future_targets_in_input_cases':False})
    for task,summary in controls['tasks'].items():
        profile,rest=task.split('_h',1);h=int(rest.split('_',1)[0])
        for fold in range(1,6):
            with np.load(HERE/'all_control_losses'/task/f'fold_{fold}.npz',allow_pickle=False) as a:
                origins=a['origins'];losses={name:a[name] for name in a.files if name!='origins'}
            data=d.Data(d.cfg_for(profile,h,fold,2026));group=day_groups(data.bars,origins,losses)
            values={'origins':origins,'session_date':data.bars.session_date[origins],
                    'names':np.asarray(group['names']),'counts':group['counts'],'sums':group['sums'],
                    **{'loss_'+name:value for name,value in losses.items()}}
            for seed in (2026,2027,2028):
                cfg=d.cfg_for(profile,h,fold,seed)
                with np.load(HERE/'cells'/cfg['id']/'outer_losses.npz',allow_pickle=False) as a:
                    assert np.array_equal(origins,a['origins'])
                    for mode in ('adapted','adapted_shrink'):values[f'seed_{seed}_{mode}']=a[mode]
            d.npz(root/'statistics'/task/f'fold_{fold}.npz',**values)
    (root/'README.md').write_text('''# Fixed-weight BatchNorm replay capsule

This capsule retains17fixed-weight adapters: minuteH60fivefolds×threeseeds and hourlyH8firsttwofolds×oneseed only. It does not expand the failed hourly screening branch or constitute independent validation. Every checkpoint includes the original cfg/scaler and changes only sixBatchNorm running-moment buffers.

Run `python code/replay.py` to evaluate the firstcanonical128past-input cases per retained cell. The input arrays contain known-at-origin split-aware log-relative histories and currentcloses; callers must separately enforce calendars and past-only corporate-action treatment. No future targets are used by replay. Float32 shape-dependent rounding is accepted under the documented rtol1e-6/atol1e-8engineering threshold; it is not a bitwise portability claim.

Run `python code/reproduce_statistics.py` to recompute pooled losses, all three seed directions, fivefold directions and bothnominal dayblockintervals for every complete BNtask against every registered strong control. Included per-origin losses, seedlosses and complete daygroupcounts/sums support the arithmetic without originalworkspace files or network. All periods remain exposeddevelopment. Successful replay/statistics does not establish predictive success.

Runtime uses the existing recorded CPU NumPy/PyTorch environment. Code/reference_source is an unchanged frozen dependency snapshot; only portable_predictor.py, portable_stats.py and their imported model dependencies are used by these entrypoints. The enclosing study preserves original source/rules and the failed1e-9batchtolerance, followed by a single-line engineering revision before outer scoring.
''',encoding='utf-8')
    hashes={str(p.relative_to(root)):d.sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='MANIFEST.json'}
    d.atomic(root/'MANIFEST.json',{'files':hashes,'manifest_excludes_itself':True,'case_count':len(cases),
        'Q3_values_read':False,'independent_confirmation':False})
    d.atomic(HERE/'CAPSULE_BUILD.json',{'status':'built_requires_independent_replay','root':str(root),
        'case_count':len(cases),'total_example_origins':sum(c['origins'] for c in cases),
        'statistics_full_tasks':list(controls['tasks']),'manifest_files':len(hashes),'at_utc':d.now()})
    print(json.dumps({'status':'built','case_count':len(cases),'manifest_files':len(hashes),'root':str(root)}),flush=True)

if __name__=='__main__':main()
