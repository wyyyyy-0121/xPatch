"""Summarize the four fixed probes; no model/data labels or recipe selection."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,os,time
import numpy as np
BASE=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return json.loads(p.read_text())

def groups_for(bars, origins, losses):
    dates = bars.session_date[origins]
    grid = np.unique(bars.session_date[(bars.session_date >= dates[0]) & (bars.session_date <= dates[-1])])
    ix = np.searchsorted(grid, dates)
    names = list(losses)
    sums = np.column_stack([np.bincount(ix, weights=losses[k], minlength=len(grid)) for k in names])
    return {'sums': sums, 'counts': np.bincount(ix, minlength=len(grid)), 'names': names}

def bootstrap(groups, pairs, block, task):
    names = groups[0]['names']
    rng = np.random.default_rng(2026)
    R = 5000
    if any((len(g['counts']) <= block for g in groups)):
        return [{'task': task, 'primary': a, 'reference': b, 'block': block, 'status': 'insufficient_groups', 'supports_improvement': False} for a, b in pairs]
    samples = np.zeros((R, len(names)))
    ns = np.zeros(R)
    point = np.zeros(len(names))
    N = 0
    for g in groups:
        n = len(g['counts'])
        point += g['sums'].sum(0)
        N += g['counts'].sum()
        for start in range(0, R, 100):
            r = min(100, R - start)
            starts = rng.integers(0, n - block + 1, size=(r, (n + block - 1) // block))
            idx = (starts[:, :, None] + np.arange(block)).reshape(r, -1)[:, :n]
            samples[start:start + r] += g['sums'][idx].sum(1)
            ns[start:start + r] += g['counts'][idx].sum(1)
    assert (ns > 0).all()
    samples /= ns[:, None]
    point /= N
    out = []
    for primary, reference in pairs:
        i, j = (names.index(primary), names.index(reference))
        delta = samples[:, j] - samples[:, i]
        varying = bool(np.any(delta != delta[0]))
        ci = np.quantile(delta, [0.025, 0.975]).tolist() if varying else None
        rel = 100 * (1 - samples[:, i] / samples[:, j])
        out.append({'task': task, 'primary': primary, 'reference': reference, 'block': block, 'repetitions': R, 'status': 'descriptive' if varying else 'degenerate', 'mse_reference_minus_primary': float(point[j] - point[i]), 'difference_ci95': ci, 'relative_improvement_pct': float(100 * (1 - point[i] / point[j])), 'relative_ci95_pct': np.quantile(rel, [0.025, 0.975]).tolist() if varying else None, 'supports_improvement': bool(ci and ci[0] > 0), 'scope': 'exposed historical development; not multiplicity-adjusted'})
    return out

def main():
    assert set(os.sched_getaffinity(0))=={24}
    scope=load(BASE/'STAGE_SCOPE.json');assert time.time()<scope['deadline_epoch']
    inputs=load(BASE/'STAGE_INPUT_IDENTITIES.json')
    independent=load(BASE/'INDEPENDENT_TEMPORAL_REPLAY.json')
    assert independent['status']=='completed_all_four_reconstructions_passed'
    rows=[];intervals=[];bindings={}
    for state in inputs['states']:
        folder=BASE/'runs'/state['id'];primary=load(folder/'RESULT.json');commit=load(folder/'RESULT_COMMIT.json')
        assert sha(folder/'RESULT.json')==commit['result_sha256']
        bindings[state['id']]={'RESULT':sha(folder/'RESULT.json'),'independent_verified':True}
        update=load(folder/'UPDATE_METRICS.json')['path_standardized_MSE']
        row={'state_id':state['id'],'task':'minute_h60' if state['cfg']['frequency']=='minute' else 'hourly_h8',
             'fold':state['cfg']['fold'],'fixed_epochs':state['fixed_epochs'],
             'training_steps':load(folder/'CHECKPOINT.json')['steps'],
             'update_count':primary['binding']['update_count'],'tail_count':primary['binding']['tail_count'],
             'status':primary['status'],'tail_scored':primary['tail_scored'],
             'update_path_MSE':update,'update_fit_reduction_percent_over_A':100*(1-update['B']/update['A']),
             'all_three_readouts_numerically_qualified':all(s['solver_qualified'] for s in primary['readouts'].values()),
             'readout_exclusions':{a:s['numerical_gates'] for a,s in primary['readouts'].items() if not s['solver_qualified']}}
        if primary['tail_scored']:
            row['tail_metrics']=primary['tail_metrics']
            row['B_improvement_percent_over_A']={m:100*(1-primary['tail_metrics']['B'][m]/primary['tail_metrics']['A'][m]) for m in ['path_standardized_MSE','endpoint_raw_MSE']}
            row['all_required_comparisons_numerically_separated_under_estimate']=all(
                primary['B_vs_controls'][ref][metric]['numerically_separated_under_frozen_estimate']
                for ref in ['A','C','D','zero'] for metric in ['path','endpoint'])
            assert sha(folder/'PAIRED_TAIL_LOSSES.npz')==primary['paired_loss_sha256']
            with np.load(folder/'PAIRED_TAIL_LOSSES.npz',allow_pickle=False) as archive:
                origins=archive['origins'];losses={k:archive[k] for k in archive.files if k!='origins'}
            assert sha(Path(state['source_bars']))==state['source_bars_sha256']
            with np.load(state['source_bars'],allow_pickle=False) as archive: dates=archive['session_date']
            assert np.all(dates<='2026-06-30')
            bars=SimpleNamespace(session_date=dates)
            for metric in ['path','endpoint']:
                names=['B','A','C','D','zero','mean'];g=groups_for(bars,origins,{name:losses[name+'_'+metric] for name in names})
                for block in ([1,5] if row['task']=='minute_h60' else [2,5]):
                    output=bootstrap([g],[('B',ref) for ref in names[1:]],block,state['id']+'__'+metric)
                    for item in output:item.update(state_id=state['id'],metric=metric,session_grid_count=len(g['counts']),
                        numerical_margin_not_included_in_statistical_interval=True,formal_success_claim_allowed=False)
                    intervals.extend(output)
        else:
            row['all_required_comparisons_numerically_separated_under_estimate']=False
        rows.append(row)
    all_numeric=all(row['tail_scored'] and row['all_three_readouts_numerically_qualified'] for row in rows)
    task_checks={task:all(row['all_required_comparisons_numerically_separated_under_estimate'] for row in rows if row['task']==task)
                 and len([row for row in rows if row['task']==task])==2 for task in ['minute_h60','hourly_h8']}
    gate=all_numeric and any(task_checks.values())
    result={'status':'fixed_temporal_probe_complete_candidate_continuation_passed' if gate else 'fixed_temporal_probe_complete_no_continuation',
        'states':rows,'states_trained':len(rows),'tail_scored_states':sum(row['tail_scored'] for row in rows),
        'total_training_steps':sum(row['training_steps'] for row in rows),
        'total_update_row_instances':sum(row['update_count'] for row in rows),
        'total_planned_tail_row_instances':sum(row['tail_count'] for row in rows),
        'total_scored_tail_row_instances':sum(row['tail_count'] for row in rows if row['tail_scored']),
        'all_four_numerically_complete':all_numeric,'task_directional_checks':task_checks,
        'scientific_continuation_gate_passed':gate,'recipe_stopped':not gate,
        'new_candidate_admitted':False,'formal_method_success':False,'independent_market_confirmation':False,
        'Q3_values_read':False,'original_stop_outer_scored':False,'GPU':False,'research_goal_achieved':False,
        'same_data_numerical_reconstruction_passed':True,
        'source_binding':bindings,'independent_replay_sha256':sha(BASE/'INDEPENDENT_TEMPORAL_REPLAY.json'),
        'bootstrap_scope':'Original prespecified5000draws seed2026, minute1/5sessions hour2/5; perstate descriptive only. Numerical uncertainty not incorporated; no posthoc candidate selection.',
        'intervals':intervals,'analysis_source_sha256':sha(Path(__file__))}
    assert time.time()<scope['deadline_epoch']
    (BASE/'RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['status','states_trained','tail_scored_states','total_training_steps','scientific_continuation_gate_passed']}))
if __name__=='__main__':main()
