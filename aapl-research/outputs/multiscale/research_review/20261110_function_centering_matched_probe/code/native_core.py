"""Hash-pinned native AST pipeline, state permissions, and matched Ridge.

No source engine, loaders, original weights, targets or predictions are imported.
Numerical imports occur only after the separately frozen guarded entry point.
"""
import ast
from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path
import sys
import types
import zipfile

import numpy as np
import torch
from torch import nn


def require(ok, message):
    if not bool(ok):
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def index_digest(a):
    return hashlib.sha256(np.asarray(a, dtype='<i8').tobytes()).hexdigest()


def original_pipeline(protocol):
    """Compile complete, unmodified original AST nodes in isolated globals."""
    module = types.ModuleType('_aapl_matched_native')
    sys.modules[module.__name__] = module
    scope = module.__dict__
    scope.update(np=np, torch=torch, nn=nn, dataclass=dataclass)
    wanted = {'bars.py': ['Bars', 'targets'],
              'features.py': ['CloseWindows', 'Scaler', 'fit_scaler'],
              'models.py': ['Recurrent']}
    records = []
    for filename, names in wanted.items():
        paths = [Path(p) for p in protocol['source_files'] if Path(p).name == filename]
        require(len(paths) == 1, 'Ambiguous exact native source ' + filename)
        source = paths[0]
        require(digest(source) == protocol['source_files'][str(source)], 'Native source changed')
        tree = ast.parse(source.read_text(), filename=str(source))
        for name in names:
            nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name]
            require(len(nodes) == 1, 'Native AST node missing/ambiguous ' + name)
            node = nodes[0]
            compiled = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
            exec(compile(compiled, str(source) + ':exactAST:' + name, 'exec'), scope)
            records.append({'source': str(source), 'name': name,
                            'ast_sha256': hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()})
    scope['raw_targets'] = scope['targets']
    registry = {}
    def gated_targets(bars, origins, horizon, mode):
        state = registry.get(id(bars))
        require(state is not None, 'Unregistered bars target request')
        require(horizon == state.horizon and mode == 'future_log_return', 'Unregistered target recipe')
        return state.labels(origins)
    # Exact fit_scaler resolves this symbol; native target math is retained below the gate.
    scope['targets'] = gated_targets
    scope['registry'] = registry
    scope['source_ast_receipt'] = records
    return scope


class StateAdapter:
    def __init__(self, native, state_id, bars, fit, lookback, horizon, actions=(), expected=None):
        self.native = native
        self.id = state_id
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        # Independent Bars identity per state, shared read-only raw fields are explicit.
        self.bars = native['Bars'](bars.frequency, bars.start_ns, bars.end_ns, bars.close,
                                  bars.observed, bars.session_date)
        self._windows = native['CloseWindows'](self.bars, self.lookback, actions)
        fit_value = np.asarray(fit)
        require(fit_value.dtype.kind in 'iu', 'Fit origins must be integers')
        self.fit = np.asarray(fit_value, dtype=np.int64).copy()
        require(len(self.fit) > 2 and np.all(np.diff(self.fit) > 0), 'Invalid original fit origins')
        require(self.fit[0] >= lookback-1 and self.fit[-1]+horizon < len(bars.close), 'Fit outside bars')
        boundary = 4 * len(self.fit) // 5
        self.tail = self.fit[boundary:].copy()
        before = self.fit[:boundary]
        self.prefix = before[bars.end_ns[before+horizon] < bars.end_ns[self.tail[0]]].copy()
        require(len(self.prefix) > 1 and len(self.tail) > 0, 'Empty purged partition')
        self.phase = 'prefix'
        self.freeze = None
        self.target_calls = {'prefix_rows': 0, 'tail_rows': 0}
        native['registry'][id(self.bars)] = self
        self.partition = {'id': self.id, 'original_fit_count': len(self.fit), 'boundary': boundary,
                          'prefix_count': len(self.prefix), 'tail_count': len(self.tail),
                          'first_tail_origin': int(self.tail[0]),
                          'prefix_sha256': index_digest(self.prefix), 'tail_sha256': index_digest(self.tail)}
        if expected is not None:
            for key in ['original_fit_count', 'boundary', 'prefix_count', 'tail_count',
                        'first_tail_origin', 'prefix_sha256', 'tail_sha256']:
                require(self.partition[key] == expected[key], 'Partition mismatch ' + self.id + ':' + key)
        for value in (self.fit, self.prefix, self.tail):
            value.flags.writeable = False

    def _indices(self, origins):
        raw = np.asarray(origins)
        require(raw.dtype.kind in 'iu', 'Origin dtype must be integer, no float truncation')
        value = np.asarray(raw, dtype=np.int64)
        allowed = self.prefix if self.phase == 'prefix' else self.tail
        require(value.ndim == 1 and len(value) > 0, 'Empty/invalid state access')
        where = np.searchsorted(allowed, value)
        require(np.all(where < len(allowed)), 'Origin outside state phase')
        require(np.array_equal(allowed[where], value), 'Origin not in state phase')
        if self.phase == 'tail':
            require(self.freeze is not None, 'Tail target/feature request before global freeze')
        return value

    def arrays(self, origins):
        value = self._indices(origins)
        return self._windows.arrays(value)

    def labels(self, origins):
        value = self._indices(origins)
        result = self.native['raw_targets'](self.bars, value, self.horizon, 'future_log_return')
        self.target_calls[self.phase + '_rows'] += len(value)
        return result

    def scaler(self):
        require(self.phase == 'prefix', 'Scaler may only fit own purged prefix')
        return self.native['fit_scaler'](self, self.prefix, self.horizon, 'future_log_return', batch_size=4096)

    def permit_tail(self, receipt):
        require(receipt.get('all_models_controls_frozen') is True, 'Missing global model freeze')
        require(self.id in receipt['states'], 'State absent from global freeze')
        self.freeze = receipt
        self.phase = 'tail'

    def batch(self, scaler, indices):
        x, anchor = self.arrays(indices)
        x, anchor = scaler.transform_x(x, anchor)
        y = scaler.transform_y(self.labels(indices))
        return (torch.from_numpy(x[..., None]), torch.from_numpy(anchor), torch.from_numpy(y))


def fixed_ridge(X, Y):
    """Fixed normalized lambda .1, raw Y, centered unpenalized intercept."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    require(X.ndim == 2 and Y.ndim == 2 and len(X) == len(Y) > 1, 'Invalid Ridge fixture')
    require(np.isfinite(X).all() and np.isfinite(Y).all(), 'Nonfinite Ridge inputs')
    n, width = X.shape
    sx = np.zeros(width); sxx = np.zeros(width); sy = np.zeros(Y.shape[1])
    for first in range(0, n, 4096):
        x, y = X[first:first+4096], Y[first:first+4096]
        sx += x.sum(axis=0, dtype=np.float64)
        sxx += np.square(x).sum(axis=0, dtype=np.float64)
        sy += y.sum(axis=0, dtype=np.float64)
    mean, second, ymean = sx/n, sxx/n, sy/n
    variance = second - mean**2
    tolerance = 1e-10*np.maximum(1., np.maximum(np.abs(second), mean**2))
    require(np.all(variance >= -tolerance), 'Ridge negative variance beyond frozen allowance')
    scale = np.sqrt(np.maximum(variance, 0.))
    scale = np.where(scale > 1e-12, scale, 1.)
    gram = np.zeros((width, width)); cross = np.zeros((width, Y.shape[1]))
    for first in range(0, n, 4096):
        z = (X[first:first+4096]-mean)/scale
        centered = Y[first:first+4096]-ymean
        gram += z.T@z
        cross += z.T@centered
    A, R = gram/n + .1*np.eye(width), cross/n
    beta = np.linalg.solve(A, R)
    residual = float(np.linalg.norm(A@beta-R, ord=np.inf))
    denominator = float(np.linalg.norm(A, ord=np.inf)*np.linalg.norm(beta, ord=np.inf)+np.linalg.norm(R, ord=np.inf))
    backward = residual/denominator if denominator else (0. if residual == 0 else float('inf'))
    require(np.isfinite(beta).all() and np.isfinite(backward) and backward <= 1e-10,
            'Ridge normal equation gate failed')
    return {'coefficients': beta, 'x_mean': mean, 'x_std': scale, 'y_mean': ymean,
            'backward_error': backward, 'normalized_lambda': .1, 'observations': n}


def ridge_predict(model, X):
    value = (np.asarray(X, dtype=np.float64)-model['x_mean'])/model['x_std']@model['coefficients']+model['y_mean']
    require(np.isfinite(value).all(), 'Nonfinite Ridge predictions')
    return value


def control_fit(adapter, scaler):
    """Bounded input materialization (~600MB largest state); checkpoint before call."""
    rows, targets = [], []
    for first in range(0, len(adapter.prefix), 4096):
        indices = adapter.prefix[first:first+4096]
        x, anchor = adapter.arrays(indices)
        x, anchor = scaler.transform_x(x, anchor)
        rows.append(np.column_stack((x, anchor)))
        targets.append(adapter.labels(indices))
    X, Y = np.concatenate(rows), np.concatenate(targets)
    return fixed_ridge(X, Y)


def load_member(path, member):
    with zipfile.ZipFile(path, 'r') as archive:
        require(archive.namelist().count(member) == 1, 'Missing/duplicate admitted npy member')
        with archive.open(member, 'r') as stream:
            value = np.load(io.BytesIO(stream.read()), allow_pickle=False)
    return value


def load_market(protocol, native, design, output):
    require(protocol['phase'] == 'market' and protocol.get('market_admission') is True,
            'Market is not admitted by preparation')
    metadata = json.loads(Path(protocol['input_metadata_path']).read_text())
    admitted = protocol['market_files']
    loaded = {}
    decoded = []
    for profile in ['minute', 'hourly']:
        record = metadata['bars'][profile]
        path = Path(record['path'])
        require(str(path) in admitted and digest(path) == admitted[str(path)] == record['sha256'], 'Bar admission changed')
        fields = {name: load_member(path, name+'.npy') for name in ['start_ns','end_ns','close','observed','session_date']}
        require(fields['session_date'].dtype.kind == 'U' and np.all(fields['session_date'] <= '2026-06-30'), 'Q3 bar boundary violated')
        require(fields['start_ns'].dtype == np.int64 and fields['end_ns'].dtype == np.int64
                and fields['close'].dtype == np.float64 and fields['observed'].dtype == np.bool_, 'Bar dtype changed')
        for array in fields.values(): array.flags.writeable = False
        loaded[profile] = native['Bars']('minute' if profile == 'minute' else 'hour', **fields)
        decoded.append({'profile':profile,'path':str(path),'whole_preQ3_raw_fields_decoded':True,'fields':list(fields),'rows':len(fields['close'])})
    states = []
    for spec in design['states']:
        record = metadata['fits'][spec['profile']][str(spec['fold'])]
        path = Path(record['path'])
        require(str(path) in admitted and digest(path) == admitted[str(path)] == record['file_sha256'], 'Fit archive admission changed')
        fit = load_member(path, 'fit.npy')
        require(fit.dtype == np.int64 and len(fit) == record['count'] and index_digest(fit) == record['fit_sha256'], 'fit.npy identity changed')
        states.append(StateAdapter(native,spec['id'],loaded[spec['profile']],fit,spec['lookback'],spec['horizon'],metadata['split_actions'],expected=spec))
    ledger = []
    for a in states:
        for b in states:
            if a.id != b.id and a.bars.frequency == b.bars.frequency:
                ledger.append({'prefix_state':a.id,'tail_state':b.id,'origin_intersection_count':int(np.intersect1d(a.prefix,b.tail).size)})
    receipt = {'whole_preQ3_raw_fields_decoded':True,'raw_containers':decoded,'decoded_fit_members_only':['fit.npy'],'state_partitions':[s.partition for s in states],'crossstate_origin_membership':ledger,'dedicated_tail_targets_before_globalfreeze':False,'not_globally_blind':True,'Q3_read':False}
    (output/'INPUT_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return states


def synthetic_state(native, spec):
    n = spec['bars_count']
    rng = np.random.default_rng(spec['data_seed'])
    increments = .00001+.0003*rng.standard_normal(n)
    close = 100*np.exp(np.cumsum(increments,dtype=np.float64))
    start = spec['first_start_ns']+np.arange(n,dtype=np.int64)*spec['native_ns']
    bars = native['Bars']('minute',start,start+spec['native_ns'],close,np.ones(n,dtype=bool),np.full(n,spec['session_date'],dtype='<U10'))
    fit = np.arange(spec['original_fit_first'],spec['original_fit_first']+spec['original_fit_count'],dtype=np.int64)
    state = StateAdapter(native,'synthetic_native',bars,fit,spec['lookback'],spec['horizon'],spec['split_actions'])
    require(len(state.prefix) == spec['purged_prefix_count'] and len(state.tail) == spec['tail_count'], 'Synthetic partition changed')
    return state
