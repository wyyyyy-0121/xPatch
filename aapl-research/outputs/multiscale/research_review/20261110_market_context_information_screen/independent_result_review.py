"""Independent source/features/centered-Ridge reconstruction; no worker imports."""
from __future__ import annotations
import argparse
import calendar
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import time
import zipfile

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
BATCH = 2048
BAR_FIELDS = ('start_ns', 'end_ns', 'close', 'observed', 'session_date')
PARTS = ('full_fit', 'purged_prefix', 'unchanged_tail')


def sha(path, guard=lambda: None):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            guard(); data = stream.read(65536)
            if not data:
                return h.hexdigest()
            h.update(data)


def raw_sha(data):
    return hashlib.sha256(data).hexdigest()


def atomic(path, value):
    temp = path.with_suffix('.partial')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)


def selected(path, names, audit):
    out = {}
    with zipfile.ZipFile(path) as archive:
        listing = archive.namelist()
        for name in names:
            member = name+'.npy'
            if listing.count(member) != 1:
                raise ValueError('Selected member absent or duplicated')
            with archive.open(member) as stream:
                out[name] = np.lib.format.read_array(stream, allow_pickle=False)
            audit.append({'path': str(Path(path).resolve()), 'member': member})
    return out


def delayed_available(observed, length):
    n = len(observed)
    if n <= length:
        return np.zeros(n, dtype=bool)
    windows = np.lib.stride_tricks.sliding_window_view(observed, length)
    return np.r_[np.zeros(length, dtype=bool), windows.all(axis=1)[:-1]]


def read_sources(refs, guard, audit):
    """Explicit original-bar/volume members; SPY uses its own timestamp dictionary."""
    grids = {f: selected(path, BAR_FIELDS, audit) for f, path in refs['bars'].items()}
    abnormal = {}
    for frequency, path in refs['volume'].items():
        volume = selected(path, (*BAR_FIELDS, 'abnormal'), audit)
        for key in BAR_FIELDS:
            if not np.array_equal(grids[frequency][key], volume[key]):
                raise ValueError('Qualified volume cache differs from original grid')
        abnormal[frequency] = volume['abnormal']
    index = json.loads(Path(refs['spy_index_path']).read_bytes())
    if index['all_67_ranges_complete'] is not True or index['Q3_allowed'] is not False:
        raise ValueError('SPY source qualification missing')
    source = {}
    for page in index['pages']:
        guard(); ref = page['canonical_ref']; body = Path(ref['path']).read_bytes()
        if len(body) != ref['bytes'] or raw_sha(body) != ref['sha256']:
            raise ValueError('SPY canonical bytes changed')
        payload = json.loads(body)
        for ordinal, bar in enumerate(payload['bars']['SPY']):
            if ordinal % 256 == 0:
                guard()
            if not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:00(?:\.0{1,9})?(?:Z|\+00:00)', bar['t']):
                raise ValueError('Source timestamp is not exact UTC minute start')
            stamp = datetime.fromisoformat(bar['t'].replace('Z', '+00:00'))
            if stamp.second or stamp.microsecond:
                raise ValueError('Non-minute qualified timestamp')
            ns = calendar.timegm(stamp.utctimetuple())*1_000_000_000
            if ns in source or ns >= calendar.timegm((2026, 7, 1, 4, 0, 0))*1_000_000_000:
                raise ValueError('SPY duplicate or holdout boundary')
            close = float(bar['c'])
            if not np.isfinite(close) or close <= 0:
                raise ValueError('Invalid qualified SPY close')
            source[ns] = close
    minute = grids['minute']
    spy_minute = np.fromiter((source.get(int(t), np.nan) for t in minute['start_ns']),
                            dtype=np.float64, count=len(minute['start_ns']))
    del source
    hour_start, hour_end, hour_session, spy_hour = [], [], [], []
    first = 0
    while first < len(spy_minute):
        guard(); end = first+1
        while end < len(spy_minute) and minute['session_date'][end] == minute['session_date'][first]:
            end += 1
        if np.any(np.diff(minute['start_ns'][first:end]) != 60_000_000_000):
            raise ValueError('Original minute grid gap')
        for block in range((end-first)//60):
            a, b = first+block*60, first+(block+1)*60
            hour_start.append(minute['start_ns'][a]); hour_end.append(minute['end_ns'][b-1])
            hour_session.append(minute['session_date'][a])
            spy_hour.append(spy_minute[b-1] if np.isfinite(spy_minute[a:b]).all() else np.nan)
        first = end
    hour = grids['hour']
    for key, values in [('start_ns', hour_start), ('end_ns', hour_end), ('session_date', hour_session)]:
        if not np.array_equal(hour[key], np.asarray(values, dtype=hour[key].dtype)):
            raise ValueError('Independent original-hour grouping mismatch')
    return grids, abnormal, {'minute': spy_minute, 'hour': np.asarray(spy_hour)}


class FeatureSource:
    """Own formulas; no original CloseWindows, predictor, targets or screen imports."""
    def __init__(self, grid, abnormal, spy, length, horizon, actions):
        self.grid, self.abnormal, self.spy = grid, abnormal, spy
        self.length, self.horizon = length, horizon
        self.logclose = np.full(len(grid['close']), np.nan)
        self.logclose[grid['observed']] = np.log(grid['close'][grid['observed']])
        self.logsplit = np.zeros(len(spy))
        for action in actions:
            self.logsplit[grid['session_date'] >= str(action['date'])] += np.log(float(action['ratio']))
        valid_volume = np.isfinite(abnormal)
        self.mV = np.r_[np.zeros(horizon-1, dtype=bool),
                        np.lib.stride_tricks.sliding_window_view(valid_volume, horizon).all(axis=1)]
        self.mS = delayed_available(np.isfinite(spy), length)

    def x(self, origins, expert, spy_mean=None):
        o = np.asarray(origins, dtype=np.int64)
        indices = o[:, None] + np.arange(1-self.length, 1)
        if not self.grid['observed'][indices].all():
            raise ValueError('Unobserved original history')
        history = self.logclose[indices]+self.logsplit[indices]-self.logsplit[o, None]-self.logclose[o, None]
        lags = sorted({1, max(2, self.horizon//4), max(3, self.horizon//2),
                       self.horizon, 2*self.horizon, 4*self.horizon, self.length-1})
        columns = [-history[:, -1-k] for k in lags]
        for k in (self.horizon, 4*self.horizon):
            change = np.diff(history[:, -(k+1):], axis=1)
            columns.append(np.sqrt(np.mean(change*change, axis=1)))
        ny = pd.to_datetime(self.grid['end_ns'][o], utc=True).tz_convert('America/New_York')
        day_fraction = (ny.hour.to_numpy()*60+ny.minute.to_numpy()+ny.second.to_numpy()/60-570)/390
        columns.extend([np.sin(2*np.pi*day_fraction), np.cos(2*np.pi*day_fraction),
                        np.sin(4*np.pi*day_fraction), np.cos(4*np.pi*day_fraction)])
        columns.extend([(ny.dayofweek.to_numpy() == day).astype(float) for day in range(4)])
        base = np.column_stack((history, self.grid['close'][o, None], *columns))
        if expert == 'A':
            return base
        if not self.mV[o].all():
            raise ValueError('Volume-invalid origin supplied to expert')
        prior = self.abnormal[o[:, None]+np.arange(1-self.horizon, 1)]
        u = np.mean(prior, axis=1)
        base = np.column_stack((base, u, columns[lags.index(self.horizon)]*u))
        if expert == 'B':
            return base
        if expert != 'Plus' or spy_mean is None:
            raise ValueError('Unknown expert or missing fixed SPY mean')
        z = np.tile(spy_mean, (len(o), 1))
        complete = self.mS[o]
        if complete.any():
            close = self.spy[o[complete, None]+np.arange(-self.length, 0)]
            z[complete] = np.diff(np.log(close), axis=1)
        return np.column_stack((base, z))

    def raw_spy(self, origins):
        o = np.asarray(origins, dtype=np.int64)
        if not self.mS[o].all():
            raise ValueError('Incomplete SPY context cannot supply raw source')
        return np.diff(np.log(self.spy[o[:, None]+np.arange(-self.length, 0)]), axis=1)

    def y(self, origins):
        o = np.asarray(origins, dtype=np.int64)
        future = o[:, None]+np.arange(1, self.horizon+1)
        if not self.grid['observed'][o].all() or not self.grid['observed'][future].all():
            raise ValueError('Unobserved original target')
        if self.horizon == 60 and not np.all(self.grid['session_date'][o] == self.grid['session_date'][o+self.horizon]):
            raise ValueError('Minute target crosses session')
        return np.log(self.grid['close'][future]/self.grid['close'][o, None])


def direct_centered_fit(source, origins, expert, spy_mean, guard):
    """Two passes center raw observations before forming Gram; independent arithmetic."""
    sx = sy = None
    for first in range(0, len(origins), BATCH):
        guard(); batch = origins[first:first+BATCH]
        x, y = source.x(batch, expert, spy_mean), source.y(batch)
        if sx is None:
            sx, sy = np.zeros(x.shape[1]), np.zeros(y.shape[1])
        sx += np.sum(x, axis=0); sy += np.sum(y, axis=0)
    mx, my = sx/len(origins), sy/len(origins)
    gram = np.zeros((len(mx), len(mx))); cross = np.zeros((len(mx), len(my)))
    for first in range(0, len(origins), BATCH):
        guard(); batch = origins[first:first+BATCH]
        x, y = source.x(batch, expert, spy_mean)-mx, source.y(batch)-my
        gram += x.T@x; cross += x.T@y
    sd = np.sqrt(np.maximum(np.diag(gram)/len(origins), 0))
    sd = np.where(sd > 1e-12, sd, 1.)
    normalized = gram/len(origins)/sd[:, None]/sd[None, :]
    coef = np.linalg.solve(normalized+0.1*np.eye(len(mx)), cross/len(origins)/sd[:, None])
    return {'coef': coef, 'x_mean': mx, 'x_std': sd, 'y_mean': my}


def predict(model, x):
    return model['y_mean']+(x-model['x_mean'])/model['x_std']@model['coef']


def score_sums(predictions, truth):
    result = {}
    for key, value in predictions.items():
        error = np.square(value-truth)
        result[key] = {'endpoint_mse': float(np.mean(error[:, -1])), 'path_mse': float(np.mean(error))}
    return result


def science_decisions(rows):
    states, tasks = [], {}
    for row in rows:
        metrics = row['metrics']; plus = metrics['plus']
        if any(not np.isfinite(v) or v < 0 for m in metrics.values() for v in m.values()):
            raise ValueError('Invalid scientific metric')
        bits = {f'endpoint_below_{c}': plus['endpoint_mse'] < metrics[c]['endpoint_mse']
                for c in ('A_full', 'base', 'zero', 'prefix_mean')}
        bits.update({f'path_not_above_{c}': plus['path_mse'] <= metrics[c]['path_mse'] for c in ('A_full', 'base')})
        states.append({'task': row['task'], 'fold': row['fold'], 'checks': bits, 'passed': all(bits.values())})
        tasks.setdefault(row['task'], []).append(metrics)
    primary = {}
    for task, metrics in tasks.items():
        if len(metrics) != 2:
            raise ValueError('Need original two folds')
        zero = sum(x['zero']['endpoint_mse'] for x in metrics)/2
        plus = sum(x['plus']['endpoint_mse'] for x in metrics)/2
        if not zero > 0:
            raise ValueError('Nonpositive primary zero denominator')
        primary[task] = {'gain_overzero': 1-plus/zero, 'passed_1percent': bool(1-plus/zero >= 0.01)}
    return {'states': states, 'primary': primary,
            'overall_pass': all(x['passed'] for x in states) and all(x['passed_1percent'] for x in primary.values())}


def run(protocol_path):
    os.sched_setaffinity(0, {26})
    resource.setrlimit(resource.RLIMIT_AS, (3*1024**3, 3*1024**3))
    protocol_bytes = protocol_path.read_bytes(); protocol = json.loads(protocol_bytes)
    identity = raw_sha(protocol_bytes); began = time.time()
    if (protocol.get('market_arrays_admitted') is not True or protocol.get('GPU') is not False
            or protocol.get('Q3_allowed') is not False
            or protocol['deadline_epoch']-protocol['started_epoch'] != 900
            or not protocol['started_epoch'] <= began < protocol['deadline_epoch']-1
            or protocol['independent_reviewer_sha256'] != sha(Path(__file__))):
        raise ValueError('Independent review protocol identity invalid')
    def guard():
        if time.time() >= protocol['deadline_epoch']-1:
            raise TimeoutError('Original900s deadline reached')
    def expired(*_):
        raise TimeoutError('Original900s deadline reached')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, protocol['deadline_epoch']-1-time.time())
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        if os.environ.get(name) != '1':
            raise ValueError('Independent numerical library must use one thread')
    root = BASE/'run'; output = root/'INDEPENDENT_RESULT_REVIEW.json'
    if output.exists():
        raise ValueError('Do not overwrite independent result evidence')
    if sha(protocol['data_references_path'], guard) != protocol['data_references_sha256']:
        raise ValueError('Frozen data references changed')
    refs = json.loads(Path(protocol['data_references_path']).read_bytes())
    for path, expected in refs['dependencies'].items():
        if sha(path, guard) != expected:
            raise ValueError('Frozen input/source dependency changed')
    for path, expected in protocol['code_hashes'].items():
        if sha(path, guard) != expected:
            raise ValueError('Frozen worker code changed')
    if sha(protocol['science_rules_path'], guard) != protocol['science_rules_sha256']:
        raise ValueError('Scientific rule identity changed')
    primary_bytes = (root/'RESULTS.json').read_bytes(); primary = json.loads(primary_bytes)
    manifest_bytes = (root/'FROZEN_MODELS.json').read_bytes(); manifest = json.loads(manifest_bytes)
    if (manifest['protocol_sha256'] != identity or primary['protocol_sha256'] != identity
            or len(manifest['models']) != 12 or len(primary['rows']) != 4):
        raise ValueError('Global prefix-model freeze barrier missing')
    models_by_key = {(x['task'], x['fold'], x['expert']): x for x in manifest['models']}
    transforms_by_state = {x['state']: x for x in manifest['transforms']}
    rows_by_key = {(x['task'], x['fold']): x for x in primary['rows']}
    if len(models_by_key) != 12 or len(rows_by_key) != 4 or len(transforms_by_state) != 4:
        raise ValueError('Duplicate/missing state or model identity')
    audit = []; mask_audit = []; model_audit = []; comparisons = []
    def compare(name, expected, actual):
        a, b = np.asarray(expected), np.asarray(actual)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Invalid comparison shapes/values: '+name)
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-11, err_msg=name)
        comparisons.append({'name': name, 'values': int(a.size),
                            'maxabs': float(np.max(np.abs(a-b))) if a.size else 0.})
    grids, abnormal, spy = read_sources(refs, guard, audit)
    actions = json.loads(Path(refs['admission_path']).read_bytes())['split_actions']
    rows, qualifications = [], []
    for state in refs['states']:
        guard(); name, fold = state['task'], state['fold']; frequency = state['frequency']
        source = FeatureSource(grids[frequency], abnormal[frequency], spy[frequency],
                               state['lookback'], state['horizon'], actions)
        names = tuple(p+suffix for p in PARTS for suffix in ('_origins', '_complete_history'))
        parts = selected(state['coverage_masks_path'], names, mask_audit)
        fit = selected(state['split_path'], ('fit',), audit)['fit']
        p, tail = parts['purged_prefix_origins'], parts['unchanged_tail_origins']
        cut = len(fit)*4//5; ends = grids[frequency]['end_ns']
        expected_p = np.asarray([int(i) for i in fit[:cut]
                                 if int(ends[int(i)+state['horizon']]) < int(ends[int(fit[cut])])], dtype=np.int64)
        if (not np.array_equal(parts['full_fit_origins'], fit)
                or not np.array_equal(p, expected_p) or not np.array_equal(tail, fit[cut:])):
            raise ValueError('Original fit/prefix/tail identity changed')
        for part in PARTS:
            origins = parts[part+'_origins']
            if not np.array_equal(source.mS[origins], parts[part+'_complete_history']):
                raise ValueError('Delayed SPY history coverage changed')
        q = p[source.mV[p]]; complete = q[source.mS[q]]
        if len(p) < max(100, source.length+19) or len(q) < max(100, 2*source.length+20) or len(complete) < 100:
            raise ValueError('Frozen no-label domain qualification failed')
        z_sum = np.zeros(source.length-1); z_square = np.zeros(source.length-1)
        for first in range(0, len(complete), BATCH):
            guard(); z = source.raw_spy(complete[first:first+BATCH])
            z_sum += z.sum(axis=0); z_square += np.square(z).sum(axis=0)
        spy_mean = z_sum/len(complete)
        if not np.any(np.sqrt(np.maximum(z_square/len(complete)-spy_mean*spy_mean, 0)) > 1e-12):
            raise ValueError('Frozen SPY variation qualification failed')
        row = rows_by_key[(name, fold)]; state_name = row['state']
        transform = transforms_by_state[state_name]
        if sha(transform['path'], guard) != transform['sha256']:
            raise ValueError('Prefix transform identity changed')
        saved_transforms = selected(transform['path'], ('spy_raw_mean', 'prefix_mean'), model_audit)
        compare(state_name+'/spy_raw_mean', spy_mean, saved_transforms['spy_raw_mean'])
        own_models = {}
        for expert, pool, own_name in [('A_full', p, 'A'), ('B_valid', q, 'B'), ('Plus_valid', q, 'Plus')]:
            guard(); record = models_by_key[(name, fold, expert)]
            domain = raw_sha(pool.astype('<i8').tobytes())
            if (record['state'] != state_name or record['n_train'] != len(pool)
                    or record['domain_sha256'] != domain or sha(record['path'], guard) != record['sha256']):
                raise ValueError('Expert domain/model manifest identity invalid')
            saved = selected(record['path'], ('coef', 'x_mean', 'x_std', 'y_mean'), model_audit)
            own = direct_centered_fit(source, pool, own_name, spy_mean, guard)
            for key in saved:
                compare(state_name+'/'+expert+'/'+key, own[key], saved[key])
            if record['columns'] != len(own['x_mean']):
                raise ValueError('Expert column count changed')
            own_models[expert] = own
        prefix_mean = own_models['A_full']['y_mean']
        compare(state_name+'/prefix_mean', prefix_mean, saved_transforms['prefix_mean'])
        prediction_file = root/(state_name+'_TAIL.npz')
        names = ('origins', 'targets', 'A_full', 'base', 'plus', 'zero', 'prefix_mean', 'mV', 'mS')
        saved_tail = selected(prediction_file, names, model_audit)
        if (not np.array_equal(saved_tail['origins'], tail)
                or not np.array_equal(saved_tail['mV'], source.mV[tail])
                or not np.array_equal(saved_tail['mS'], source.mS[tail])):
            raise ValueError('Complete original tail or routing masks changed')
        predictions = {k: [] for k in ('A_full', 'base', 'plus', 'zero', 'prefix_mean')}; truths = []
        for first in range(0, len(tail), BATCH):
            guard(); o = tail[first:first+BATCH]
            a = predict(own_models['A_full'], source.x(o, 'A'))
            base = a.copy(); valid_v = source.mV[o]; valid_s = source.mS[o]
            if valid_v.any():
                base[valid_v] = predict(own_models['B_valid'], source.x(o[valid_v], 'B'))
            plus = base.copy(); both = valid_v & valid_s
            if both.any():
                plus[both] = predict(own_models['Plus_valid'], source.x(o[both], 'Plus', spy_mean))
            truths.append(source.y(o))
            for key, value in [('A_full', a), ('base', base), ('plus', plus),
                               ('zero', np.zeros_like(a)), ('prefix_mean', np.tile(prefix_mean, (len(o), 1)))]:
                predictions[key].append(value)
        truth = np.concatenate(truths)
        predictions = {k: np.concatenate(v) for k, v in predictions.items()}
        compare(state_name+'/targets', truth, saved_tail['targets'])
        for key, value in predictions.items():
            compare(state_name+'/'+key, value, saved_tail[key])
        unavailable = ~(source.mV[tail]&source.mS[tail])
        if not np.array_equal(saved_tail['plus'][unavailable].view(np.uint64),
                              saved_tail['base'][unavailable].view(np.uint64)):
            raise ValueError('Primary missing-history fallback not exact')
        if not np.array_equal(saved_tail['base'][~source.mV[tail]].view(np.uint64),
                              saved_tail['A_full'][~source.mV[tail]].view(np.uint64)):
            raise ValueError('Primary missing-volume fallback not exact')
        metrics = score_sums(predictions, truth)
        for method, values in metrics.items():
            for metric, value in values.items():
                compare(state_name+'/'+method+'/'+metric, value, row['metrics'][method][metric])
        for valid_volume in (False, True):
            for valid_spy in (False, True):
                key = 'mV'+str(int(valid_volume))+'_mS'+str(int(valid_spy))
                mask = (source.mV[tail] == valid_volume)&(source.mS[tail] == valid_spy)
                group = row['diagnostic_groups'][key]
                if group['count'] != int(mask.sum()):
                    raise ValueError('Predeclared routing diagnostic group count differs')
                for method, prediction in predictions.items():
                    error = np.square(prediction[mask]-truth[mask])
                    compare(state_name+'/'+key+'/'+method+'/endpoint_sum', float(error[:, -1].sum()),
                            group['error_sums'][method]['endpoint'])
                    compare(state_name+'/'+key+'/'+method+'/path_sum', float(error.sum()),
                            group['error_sums'][method]['path'])
        rows.append({'task': name, 'fold': fold, 'metrics': metrics})
        qualifications.append({'state': state_name, 'n_prefix': len(p), 'n_volume_prefix': len(q),
                               'n_complete_SPY_volume_prefix': len(complete), 'n_tail': len(tail),
                               'n_tail_volume': int(source.mV[tail].sum()),
                               'n_tail_joint_available': int((source.mV[tail]&source.mS[tail]).sum())})
    decisions = science_decisions(rows)
    for result in decisions['states']:
        primary_row = rows_by_key[(result['task'], result['fold'])]
        if primary_row['gates'] != result['checks'] or primary_row['state_pass'] != result['passed']:
            raise ValueError('Scientific per-state boolean decision disagrees')
    for task, result in decisions['primary'].items():
        compare(task+'/equal_fold_gain_overzero', result['gain_overzero'], primary['primaries'][task]['gain_overzero'])
        if result['passed_1percent'] != primary['primaries'][task]['passed_1percent']:
            raise ValueError('Primary1percent scientific decision disagrees')
    if primary['numerically_qualified'] is not True or primary['overall_information_gate'] != decisions['overall_pass']:
        raise ValueError('Overall information decision disagrees')
    if raw_sha((root/'FROZEN_MODELS.json').read_bytes()) != raw_sha(manifest_bytes):
        raise ValueError('Frozen model manifest changed during scoring/review')
    if raw_sha((root/'RESULTS.json').read_bytes()) != raw_sha(primary_bytes):
        raise ValueError('Primary result changed during independent review')
    for record in manifest['models']+manifest['transforms']:
        if sha(record['path'], guard) != record['sha256']:
            raise ValueError('Frozen prefix model/transform changed during review')
    guard()
    result = {'status': 'independent_all12_models_and_full_tail_review_passed',
              'protocol_sha256': identity, 'reviewer_sha256': sha(Path(__file__), guard),
              'primary_results_sha256': raw_sha(primary_bytes), 'frozen_models_sha256': raw_sha(manifest_bytes),
              'started_epoch': began, 'ended_epoch': time.time(), 'deadline_epoch': protocol['deadline_epoch'],
              'seconds': time.time()-began, 'CPU_affinity': sorted(os.sched_getaffinity(0)),
              'all12_expert_models_reconstructed': True, 'all_original_tail_matrices_compared': True,
              'independent_source_feature_and_centered_arithmetic': True, 'uses_primary_source_math_code': False,
              'same_numpy_solve_backend_disclosed': True, 'comparisons': comparisons,
              'scalar_comparison_count': sum(x['values'] for x in comparisons),
              'maxabs_numeric_difference': max(x['maxabs'] for x in comparisons),
              'qualifications': qualifications, 'decisions': decisions,
              'original_input_member_audit': audit, 'accepted_clock_mask_member_audit': mask_audit,
              'primary_model_and_tail_member_audit': model_audit,
              'Q3_read': False, 'HTTP_calls': 0, 'GPU': False,
              'independent_market_confirmation': False, 'research_goal_achieved': False}
    atomic(output, result); signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps({k: result[k] for k in ('status', 'seconds', 'scalar_comparison_count',
                                           'maxabs_numeric_difference')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    run(parser.parse_args().protocol)
