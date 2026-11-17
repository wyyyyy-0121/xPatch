"""Fit-only unconditional change paths: a post-observation, frozen strong control.

This script owns only its mean_controls directory. It neither trains a network
nor changes the preceding study, and never fits with outer labels.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, csv, hashlib, json, os, signal, sys, time
import numpy as np

BASE = Path(__file__).resolve().parents[1]
V3 = BASE.parent.with_name('20261109_v3')
sys.path.insert(0, str(V3/'code'))
from study import PRIOR, SOURCE, read, sha, atomic, npz, now
from mechanism_stats import day_groups, paired_intervals
from multiscale.matrix_executor import read_bars
from multiscale.bars import targets
from multiscale.corporate_actions import split_event_dates, split_event_flags

STOP = False
BATCH = 4096
DEADLINE = 1794205909.6228158
PRIMARY = ['minute_h60_future_log_return', 'hourly_h8_future_log_return']


def stopping(*_):
    global STOP
    STOP = True


class Paused(Exception):
    pass


def check_time():
    if STOP or time.time() >= DEADLINE:
        raise Paused()


def index_hash(values):
    return hashlib.sha256(np.asarray(values, dtype='<i8').tobytes()).hexdigest()


def freeze():
    data = read(PRIOR/'protocol/DATA.json')
    rules = {
        'name': 'fit_only_unconditional_change_path_control',
        'context': 'post-observation follow-up after v3 candidate scores; not external preregistration',
        'hypothesis': 'Some apparent long-scale sequence-model gains may be explained by a common fit-history drift path rather than sequence-dependent forecasts.',
        'tasks': sorted(data['task_files']), 'folds': [1, 2, 3, 4, 5],
        'fit_only_mean': {'return': 'mean of original cumulative log-return targets over the purged inner fit, one mean per path step',
                         'price': 'mean of original future nominal dollar prices minus the corresponding origin close over the purged inner fit, then added to each forecast origin close'},
        'outputs': ['baseline', 'raw', 'shrink'],
        'baseline': 'exact zero cumulative return or current-close persistence',
        'calibration': 'alpha=clip(sum(q*r)/sum(q*q),0,1), q=raw endpoint minus baseline, r=actual endpoint minus baseline; inner stopping tail only; same alpha for every step',
        'denominator_zero_rule': 'alpha=0', 'no_output_clipping': True,
        'fit_and_calibration_batch_size': BATCH, 'arithmetic': 'float64; ordered batches and summation',
        'comparison': 'reuse frozen v3 per-origin loss averages and their existing seed counts; never score an averaged neural prediction as if it were mean seed loss',
        'primary_return_tasks_unchanged': PRIMARY,
        'statistics': {'endpoint_mse': True, 'endpoint_mae': True, 'path_mse': True,
                       'paired_intervals': 'same exposed-development day/month block arithmetic as v3, 5000 draws seed2026, nominal95%; no independent or multiplicity-adjusted claim',
                       'blocks': {'annual': [12,24], 'quarterly': [3,6], 'monthly': [21,42], 'hourly': [2,5], 'minute': [1,5]}},
        'long_scale_sensitivity': {
            'original_nonoverlap': 'earliest-origin greedy across folds, next origin >= previous origin+H',
            'phase_sensitivity': 'repeat greedy starting at each of first H chronological origins; dependent sensitivity chains, no tests treating phases as independent',
            'split': 'origin_session < effective_date <= final_target_session; descriptive groups only; empty crossing groups are not estimable'},
        'deadline_epoch': DEADLINE, 'cpu_affinity': [24,26], 'gpu': False,
        'Q3_values_read': False, 'independent_confirmation': False,
        'method_success_claimed': False,
    }
    assert len(rules['tasks']) == 20
    inputs = {str(PRIOR/'protocol/DATA.json'): sha(PRIOR/'protocol/DATA.json'),
              str(V3/'analysis/CANDIDATE_RESULTS.json'): sha(V3/'analysis/CANDIDATE_RESULTS.json'),
              str(V3/'protocol/CANDIDATE_JOBS.json'): sha(V3/'protocol/CANDIDATE_JOBS.json'),
              str(V3/'protocol/BUDGET.json'): sha(V3/'protocol/BUDGET.json'),
              str(V3/'code/mechanism_stats.py'): sha(V3/'code/mechanism_stats.py')}
    source = read(V3/'data/SOURCE_HASHES.json')
    for name, digest in source['files'].items():
        assert sha(SOURCE/name) == digest
        inputs[str(SOURCE/name)] = digest
    for task, digest in data['task_files'].items():
        path = PRIOR/'cache'/task/'TASK.json'; assert sha(path) == digest
        inputs[str(path)] = digest
        for rec in read(path)['folds'].values():
            inputs[rec['split_file']] = sha(rec['split_file'])
            inputs[rec['targets']] = rec['targets_sha256']
    for module in ('multiscale.bars', 'multiscale.matrix_executor', 'multiscale.corporate_actions'):
        path = Path(sys.modules[module].__file__).resolve(); inputs[str(path)] = sha(path)
    freeze_record = {'rules': rules, 'script_sha256': sha(__file__), 'input_hashes': inputs,
                     'new_control_outer_scores_seen': False, 'at_utc': now(), 'frozen_epoch': time.time()}
    path = BASE/'RULES.json'
    if path.exists():
        saved = read(path)
        for name in ('rules','script_sha256','input_hashes'):
            assert saved[name] == freeze_record[name], name
        return saved
    assert not list((BASE/'predictions').glob('*/fold_*.npz'))
    atomic(path, freeze_record)
    return freeze_record


def baseline(bars, indices, horizon, target):
    if target == 'future_log_return':
        return np.zeros((len(indices), horizon), dtype=np.float64)
    return np.broadcast_to(bars.close[indices,None], (len(indices),horizon))


def fit_mean(task, fold, bars, meta, fit, identity):
    folder = BASE/'fits'/task/f'fold_{fold}'; folder.mkdir(parents=True, exist_ok=True)
    path = folder/'PROGRESS.npz'; count = cursor = 0; total = np.zeros(meta['horizon'])
    if path.exists():
        with np.load(path, allow_pickle=False) as a:
            assert str(a['identity']) == identity
            count = int(a['count']); cursor = int(a['cursor']); total = a['total'].copy()
    for start in range(cursor, len(fit), BATCH):
        check_time(); idx = fit[start:start+BATCH]
        y = targets(bars, idx, meta['horizon'], meta['target'])
        if meta['target'] == 'future_close':
            y = y-bars.close[idx,None]
        total += y.sum(axis=0, dtype=np.float64); count += len(idx); cursor = start+len(idx)
        npz(path, identity=np.asarray(identity), total=total, count=np.asarray(count), cursor=np.asarray(cursor))
    assert count == len(fit)
    return total/count


def metrics(prediction, actual):
    error = prediction-actual
    return {'endpoint_mse': float(np.mean(error[:,-1]**2)),
            'endpoint_mae': float(np.mean(abs(error[:,-1]))),
            'path_mse': float(np.mean(error**2))}


def gain(value, reference):
    return float(100*(1-value/reference)) if reference > 0 else None


def fold_run(task, fold, bars, meta, frozen, previous):
    out = BASE/'fits'/task/f'fold_{fold}'; out.mkdir(parents=True, exist_ok=True)
    prediction_file = BASE/'predictions'/task/f'fold_{fold}.npz'
    identity = hashlib.sha256(json.dumps({'rules': sha(BASE/'RULES.json'), 'task': task, 'fold': fold}, sort_keys=True).encode()).hexdigest()
    if (out/'DONE.json').exists():
        done = read(out/'DONE.json'); assert done['identity'] == identity
        assert sha(prediction_file) == done['prediction_sha256']
        return done
    rec = meta['folds'][str(fold)]
    assert sha(rec['split_file']) == frozen['input_hashes'][rec['split_file']]
    with np.load(rec['split_file'], allow_pickle=False) as a:
        splits = {k:a[k].copy() for k in a.files}
    for part, values in splits.items():
        assert index_hash(values) == rec['split_sha256'][part]
    fit, tail, validation = (splits[k] for k in ('fit','stop','validation')); h=meta['horizon']
    assert bars.end_ns[fit[-1]+h] < bars.end_ns[tail[0]]
    assert bars.end_ns[tail[-1]+h] < bars.end_ns[validation[0]]
    mean = fit_mean(task, fold, bars, meta, fit, identity)
    num=den=truth_total=0.
    for start in range(0,len(tail),BATCH):
        check_time(); idx=tail[start:start+BATCH]
        b=baseline(bars,idx,h,meta['target']); raw=b+mean[None,:]
        y=targets(bars,idx,h,meta['target']); q=raw[:,-1]-b[:,-1]; r=y[:,-1]-b[:,-1]
        num+=float(np.dot(q,r));den+=float(np.dot(q,q));truth_total+=float(np.dot(r,r))
    alpha=float(np.clip(num/den,0,1)) if den else 0.
    inner=(truth_total-2*alpha*num+alpha*alpha*den)/len(tail)
    endpoints=min(truth_total,truth_total-2*num+den)/len(tail)
    assert inner <= endpoints+1e-10*max(1.,abs(endpoints))
    check_time(); started=time.time()
    assert sha(rec['targets']) == rec['targets_sha256']
    with np.load(rec['targets'],allow_pickle=False) as a:
        origins=a['origins'].copy();y=a['actual'].copy();anchors=a['anchor'].copy()
    assert np.array_equal(origins,validation) and np.array_equal(anchors,bars.close[origins])
    assert np.array_equal(y,targets(bars,origins,h,meta['target']))
    b=baseline(bars,origins,h,meta['target']);raw=b+mean[None,:];shrunk=b+alpha*(raw-b)
    arrays={'origins':origins,'anchor':anchors,'raw':raw,'shrink':shrunk,
            'session_date':bars.session_date[origins], 'target_session_date':bars.session_date[origins+h]}
    local={}
    for name,pred in [('baseline',b),('raw',raw),('shrink',shrunk)]:
        assert np.isfinite(pred).all()
        local[name]=metrics(pred,y)
        arrays['loss_'+name]=(pred[:,-1]-y[:,-1])**2
        arrays['absolute_'+name]=abs(pred[:,-1]-y[:,-1])
        arrays['path_loss_'+name]=np.mean((pred-y)**2,axis=1)
    compare=V3/'analysis/candidate_paired_losses'/task/f'fold_{fold}.npz'
    comparison_hash=sha(compare)
    with np.load(compare,allow_pickle=False) as a:
        assert np.array_equal(origins,a['origins'])
        for key in a.files:
            if key.startswith('loss_'):
                name='v3_'+key.removeprefix('loss_');arrays['loss_'+name]=a[key].copy()
                local[name]={'endpoint_mse':float(a[key].mean()),'endpoint_mae':None,'path_mse':None}
    v3rows=[r for r in previous['rows'] if r['task']==task and r['mode']=='candidate' and r['fold']==fold]
    assert v3rows
    local['v3_candidate']['endpoint_mae']=float(np.mean([r['endpoint_mae'] for r in v3rows]))
    local['v3_candidate']['path_mse']=float(np.mean([r['path_mse'] for r in v3rows]))
    assert np.isclose(local['v3_candidate']['endpoint_mse'],np.mean([r['endpoint_mse'] for r in v3rows]),rtol=1e-12)
    for row in local.values():
        row['improvement_pct']=gain(row['endpoint_mse'],local['baseline']['endpoint_mse'])
    npz(prediction_file,**arrays)
    done={'status':'completed','task':task,'fold':fold,'identity':identity,'origins':len(origins),
          'mean_path':mean.tolist(),'alpha':alpha,'fit_count':len(fit),'stop_count':len(tail),
          'fit_labels_end_ns':int(bars.end_ns[fit[-1]+h]),'stop_first_decision_ns':int(bars.end_ns[tail[0]]),
          'stop_labels_end_ns':int(bars.end_ns[tail[-1]+h]),'validation_first_decision_ns':int(bars.end_ns[validation[0]]),
          'inner_stop_mse':inner,'inner_baseline_mse':truth_total/len(tail),
          'metrics':local,'existing_candidate_seed_count':len(v3rows),
          'v3_comparison_losses_sha256':comparison_hash,'prediction_sha256':sha(prediction_file),
          'rules_sha256':sha(BASE/'RULES.json'),'control_outer_scoring_started_epoch':started,
          'outer_labels_used_for_fitting_or_calibration':False,'Q3_values_read':False,'at_utc':now()}
    assert started > frozen['frozen_epoch']
    atomic(out/'DONE.json',done)
    print(json.dumps({'completed':task,'fold':fold,'raw_gain':local['raw']['improvement_pct'],'shrink_gain':local['shrink']['improvement_pct']}),flush=True)
    return done


def sensitivity(task, meta, bars, folds):
    arrays=[]
    for fold in range(1,6):
        with np.load(BASE/'predictions'/task/f'fold_{fold}.npz',allow_pickle=False) as a:
            arrays.append({k:a[k].copy() for k in a.files if k=='origins' or k.startswith(('loss_','path_loss_','absolute_'))})
    origins=np.concatenate([a['origins'] for a in arrays]);assert (np.diff(origins)>0).all()
    losses={k:np.concatenate([a[k] for a in arrays]) for k in arrays[0] if k.startswith('loss_')}
    def describe(mask):
        n=int(mask.sum())
        if not n:return {'origins':0,'status':'not_estimable'}
        baseline_mse=float(losses['loss_baseline'][mask].mean())
        return {'origins':n,'status':'descriptive', 'first_origin':str(bars.session_date[origins[mask][0]]),
                'last_origin':str(bars.session_date[origins[mask][-1]]),
                'methods':{k.removeprefix('loss_'):{'endpoint_mse':float(v[mask].mean()),'improvement_pct':gain(float(v[mask].mean()),baseline_mse)} for k,v in losses.items()}}
    h=meta['horizon']; chains=[]
    for start in range(min(h,len(origins))):
        mask=np.zeros(len(origins),dtype=bool);last=None
        for i in range(start,len(origins)):
            if last is None or origins[i]>=last+h:mask[i]=True;last=int(origins[i])
        chains.append({'starting_phase':start,**describe(mask)})
    events=split_event_dates(read(SOURCE/'admission.json')['split_actions'])
    cross=split_event_flags(bars.session_date[origins],bars.session_date[origins+h],events)
    return {'task':task,'profile':meta['profile'],'original_earliest_greedy_nonoverlap':chains[0],
            'dependent_phase_chains':chains,'phase_inference':'dependent sensitivity only; no phase-level independence or significance claim',
            'split_contrast_status':'descriptive_single_event' if cross.any() and (~cross).any() else 'not_estimable_no_crossing_origins' if not cross.any() else 'not_estimable_no_non_crossing_origins',
            'crosses_split':describe(cross),'no_split':describe(~cross),'distinct_reviewed_events':len(events),
            'independent_confirmation':False}


def aggregate(task, meta, bars, folds, previous):
    names=list(folds[0]['metrics']);n=sum(r['origins'] for r in folds);pooled=[];groups=[]
    for name in names:
        result={'task':task,'method':name,'origins':n,
                'positive_folds':sum(r['metrics'][name]['endpoint_mse']<r['metrics']['baseline']['endpoint_mse'] for r in folds),
                'seed_count':next(r['seed_count'] for r in previous['pooled'] if r['task']==task and r['method']=='candidate') if name in ('v3_candidate','v3_trained','v3_selected','v3_coupled') else 1 if name in ('v3_original','v3_original_shrink') else None,
                'deterministic_control':not name.startswith('v3_')}
        for metric in ('endpoint_mse','endpoint_mae','path_mse'):
            result[metric]=sum(r['metrics'][name][metric]*r['origins'] for r in folds)/n if all(r['metrics'][name][metric] is not None for r in folds) else None
        baseline_mse=sum(r['metrics']['baseline']['endpoint_mse']*r['origins'] for r in folds)/n
        result['improvement_pct']=gain(result['endpoint_mse'],baseline_mse);pooled.append(result)
    for fold in range(1,6):
        with np.load(BASE/'predictions'/task/f'fold_{fold}.npz',allow_pickle=False) as a:
            origins=a['origins'];losses={name:a['loss_'+name] for name in names}
            groups.append(day_groups(bars,origins,losses))
    if meta['profile'] in ('annual','quarterly','monthly'):
        groups=[{'names':groups[0]['names'],'counts':np.concatenate([g['counts'] for g in groups]),'sums':np.concatenate([g['sums'] for g in groups])}]
    refs=['baseline','raw','v3_candidate','v3_original_shrink','v3_ridge_history_selected','v3_ridge_history_anchor_selected']
    pairs=[('shrink',r) for r in refs]+[('raw','baseline')]
    intervals=[]
    for block in read(BASE/'RULES.json')['rules']['statistics']['blocks'][meta['profile']]:
        check_time();intervals.extend(paired_intervals(groups,pairs,block,task))
    extra=sensitivity(task,meta,bars,folds) if meta['profile'] in ('annual','quarterly','monthly') else None
    return pooled,intervals,extra


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze-only',action='store_true');args=parser.parse_args()
    assert set(os.sched_getaffinity(0)).issubset({24,26}) and os.environ['CUDA_VISIBLE_DEVICES']=='-1'
    assert read(V3/'protocol/BUDGET.json')['deadline_epoch']==DEADLINE
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):signal.signal(sig,stopping)
    frozen=freeze()
    if args.freeze_only:
        print(json.dumps({'frozen':str(BASE/'RULES.json'),'tasks':20,'folds':100,'new_control_scores_seen':False}));return
    check_time();previous=read(V3/'analysis/CANDIDATE_RESULTS.json')
    admission=read(SOURCE/'admission.json');cache={};all_folds=[];pooled=[];intervals=[];sensitivities=[]
    try:
        for task in frozen['rules']['tasks']:
            meta=read(PRIOR/'cache'/task/'TASK.json');frequency=meta['frequency']
            if frequency not in cache:
                cache[frequency]=read_bars(SOURCE,admission,frequency)
                assert cache[frequency].session_date[-1]<='2026-06-30'
            bars=cache[frequency];folds=[]
            for fold in range(1,6):
                check_time();folds.append(fold_run(task,fold,bars,meta,frozen,previous))
            local,local_intervals,extra=aggregate(task,meta,bars,folds,previous)
            pooled.extend(local);intervals.extend(local_intervals);all_folds.extend(folds)
            if extra:sensitivities.append(extra)
        result={'status':'completed','tasks':20,'folds':100,'pooled':pooled,'fold_results':all_folds,'intervals':intervals,
                'long_scale_sensitivity':sensitivities,'rules_sha256':sha(BASE/'RULES.json'),
                'primary_return_tasks_retained':PRIMARY,'previous_candidate_primary_gates_passed':previous['primary_tasks_both_pass'],
                'new_mean_controls_are_post_observation':True,'independent_confirmation':False,'method_success_claimed':False,
                'Q3_values_read':False,'at_utc':now(),'deadline_epoch':DEADLINE}
        atomic(BASE/'audit/RESULTS.json',result)
        flat=[{'task':r['task'],'fold':r['fold'],'method':name,'origins':r['origins'],'alpha':r['alpha'],**values} for r in all_folds for name,values in r['metrics'].items()]
        for name,rows in [('POOLED.csv',pooled),('FOLDS.csv',flat),('INTERVALS.csv',intervals)]:
            keys=list(dict.fromkeys(k for r in rows for k in r))
            with (BASE/name).open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
        invariant={'status':'passed','folds':len(all_folds),'task_count':len(set(r['task'] for r in all_folds)),
                   'frozen_before_each_new_outer_score':all(r['control_outer_scoring_started_epoch']>frozen['frozen_epoch'] for r in all_folds),
                   'fit_labels_before_stop_and_stop_labels_before_outer':all(r['fit_labels_end_ns']<r['stop_first_decision_ns'] and r['stop_labels_end_ns']<r['validation_first_decision_ns'] for r in all_folds),
                   'no_outer_fitting':True,'no_neural_training':True,'Q3_values_read':False,'affinity':sorted(os.sched_getaffinity(0)),
                   'at_utc':now(),'rules_sha256':sha(BASE/'RULES.json'),'results_sha256':sha(BASE/'audit/RESULTS.json')}
        assert invariant['frozen_before_each_new_outer_score'] and invariant['fit_labels_before_stop_and_stop_labels_before_outer']
        atomic(BASE/'audit/INVARIANTS.json',invariant)
        print(json.dumps({'status':'completed','tasks':20,'folds':100,'results':str(BASE/'audit/RESULTS.json')}),flush=True)
    except Paused:
        atomic(BASE/'audit/PAUSED.json',{'status':'paused','completed_folds':len(list((BASE/'fits').glob('*/fold_*/DONE.json'))),'deadline_epoch':DEADLINE,'at_utc':now()})
        print('Paused with per-fold means and completed predictions retained.',flush=True)


if __name__=='__main__':
    main()
