"""Pure in-memory, state-bound adapter; no source/manifest/data file I/O.

Manifest validation here is structural. The caller must verify actual files.
Whole future close columns in memory do not establish global time blindness.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo
from types import MappingProxyType
import hashlib
import json
import math
import re
import numpy as np

CONTRACT_SHA256 = "ad88dc3ebbd7ec974084a39397d81155971380ffa2186961b563ba38a2c1a942"
TASKS = {"minute_h60_future_log_return": (480, 60), "hourly_h8_future_log_return": (96, 8)}
ROLES = ("Q", "S", "C_cal", "T")
MODEL_TAGS = tuple(f"{arm}:s{seed}" for arm in ("C", "R") for seed in (2026, 2027, 2028)) + ("E_Ridge", "mean")
SHA = re.compile(r"[0-9a-f]{64}\Z")
NY = ZoneInfo("America/New_York")


class AdapterError(ValueError):
    pass


def require(value, message):
    if not bool(value):
        raise AdapterError(message)


def frozen(array):
    """Immutable bytes backing prevents callers re-enabling writeability."""
    array = np.ascontiguousarray(array)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def array_id(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def object_id(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class StateKey:
    task: str
    fold: int
    state_uid: str

    def __post_init__(self):
        require(self.task in TASKS and type(self.fold) is int and 1 <= self.fold <= 5
                and isinstance(self.state_uid, str) and bool(self.state_uid), "Invalid StateKey")

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class AccessKey:
    task: str
    fold: int
    state_uid: str
    role: str
    operation: str

    @property
    def state(self):
        return StateKey(self.task, self.fold, self.state_uid)


def access(state, role, operation):
    return AccessKey(state.task, state.fold, state.state_uid, role, operation)


class GlobalPhaseGate:
    def __init__(self, expected_states):
        states = tuple(expected_states)
        require(states and all(isinstance(s, StateKey) for s in states), "Expected StateKeys required")
        require(len(set(states)) == len(states) and len({s.state_uid for s in states}) == len(states)
                and len({(s.task, s.fold) for s in states}) == len(states), "Duplicate state identity")
        self.expected_states = states
        self._registered = {}
        self._models = None
        self._alphas = None
        self._phase = "TRAIN"

    @property
    def phase(self):
        return self._phase

    def model_ids(self, state):
        require(state in self.expected_states, "Unknown state")
        return tuple(state.state_uid + "/" + tag for tag in MODEL_TAGS)

    def register_state(self, state, roles_id):
        require(state in self.expected_states and state not in self._registered, "Unknown/reused state adapter")
        require(SHA.fullmatch(roles_id), "Invalid role identity")
        self._registered[state] = {"roles_id": roles_id, "scaler_id": None}

    def bind_scaler(self, state, roles_id, scaler_id):
        require(self._phase == "TRAIN" and state in self._registered, "Scaler outside TRAIN")
        entry = self._registered[state]
        require(entry["roles_id"] == roles_id and entry["scaler_id"] is None and SHA.fullmatch(scaler_id),
                "Scaler state mismatch or second fit")
        entry["scaler_id"] = scaler_id

    def check(self, key, expected_operation):
        require(isinstance(key, AccessKey) and key.state in self._registered, "Unknown access state")
        require(key.operation == expected_operation, "Operation key mismatch")
        if expected_operation == "fit_scaler":
            require(self._phase == "TRAIN" and key.role == "Q", "Only Q may fit scaler")
            return
        require(expected_operation in ("inputs", "labels", "alpha"), "Unknown operation")
        allowed = {"TRAIN": ("Q", "S"), "CALIBRATE": ("C_cal",), "TEST": ("T",)}[self._phase]
        require(key.role in allowed, "Role not admitted in current phase")
        require(expected_operation != "alpha" or (self._phase == "CALIBRATE" and key.role == "C_cal"),
                "Alpha requires calibration phase")

    def freeze_models(self, manifest):
        require(self._phase == "TRAIN" and self._models is None, "Models already frozen")
        require(set(self._registered) == set(self.expected_states), "Global states incomplete")
        ids = {mid for state in self.expected_states for mid in self.model_ids(state)}
        require(isinstance(manifest, dict) and set(manifest) == ids, "Exact eight model IDs per state required")
        for state in self.expected_states:
            bound = self._registered[state]
            require(bound["scaler_id"] is not None, "Missing state Q scaler")
            for mid in self.model_ids(state):
                row = manifest[mid]
                require(isinstance(row, dict) and set(row) == {"state", "model_sha256", "scaler_id", "roles_id"},
                        "Model manifest schema")
                require(row["state"] == state.as_dict() and SHA.fullmatch(str(row["model_sha256"]))
                        and row["scaler_id"] == bound["scaler_id"] and row["roles_id"] == bound["roles_id"],
                        "Model/scaler/roles state binding mismatch")
        self._models = json.loads(json.dumps(manifest))
        self._phase = "CALIBRATE"
        return {"phase": self._phase, "manifest_id": object_id(self._models), "actual_files_verified": False}

    def model_entry(self, state, model_id):
        require(self._models is not None and model_id in self.model_ids(state), "Unknown frozen state model")
        return json.loads(json.dumps(self._models[model_id]))

    def freeze_alphas(self, manifest):
        require(self._phase == "CALIBRATE" and self._alphas is None, "Alpha phase/order mismatch")
        require(isinstance(manifest, dict) and set(manifest) == set(self._models), "Exact global alpha IDs required")
        for state in self.expected_states:
            for mid in self.model_ids(state):
                row = manifest[mid]
                require(isinstance(row, dict) and set(row) == {"state", "model_id", "model_sha256", "scaler_id", "roles_id", "alpha", "alpha_id"},
                        "Alpha manifest schema")
                base = {k: v for k, v in row.items() if k != "alpha_id"}
                model = self._models[mid]
                require(row["state"] == state.as_dict() and row["model_id"] == mid
                        and row["model_sha256"] == model["model_sha256"]
                        and row["scaler_id"] == model["scaler_id"] and row["roles_id"] == model["roles_id"]
                        and type(row["alpha"]) in (int, float) and math.isfinite(row["alpha"])
                        and 0 <= row["alpha"] <= 1 and row["alpha_id"] == object_id(base),
                        "Alpha/model/scaler state binding mismatch")
        self._alphas = json.loads(json.dumps(manifest))
        self._phase = "TEST"
        return {"phase": self._phase, "manifest_id": object_id(self._alphas), "actual_files_verified": False}


def feature_names(horizon, lookback):
    lags = sorted({1, max(2, horizon//4), max(3, horizon//2), horizon, 2*horizon, 4*horizon, lookback-1})
    require(len(lags) == 7 and max(4*horizon, max(lags)) <= lookback-1, "Original seven-lag contract violated")
    return tuple([f"return_lag_{k}" for k in lags] + [f"rms_{horizon}", f"rms_{4*horizon}"]
                 + ["clock_sin1", "clock_cos1", "clock_sin2", "clock_cos2", "Monday", "Tuesday", "Wednesday", "Thursday"])


def context17(history, completion_ns, horizon):
    require(isinstance(history, np.ndarray) and history.dtype == np.float64 and history.ndim == 2
            and np.isfinite(history).all() and history.shape[0] > 0, "Invalid raw full history")
    require(isinstance(completion_ns, np.ndarray) and completion_ns.dtype == np.int64
            and completion_ns.shape == (len(history),), "Invalid completion clock")
    require(np.allclose(history[:, -1], 0., rtol=0., atol=1e-12), "Origin-relative history must end at zero")
    names = feature_names(horizon, history.shape[1])
    lags = [int(n.rsplit("_", 1)[1]) for n in names[:7]]
    values = [-history[:, -1-k] for k in lags]
    for k in (horizon, 4*horizon):
        d = np.diff(history[:, -(k+1):], axis=1)
        values.append(np.sqrt(np.mean(d*d, axis=1)))
    ny = [datetime.fromtimestamp(int(t)//1_000_000_000, timezone.utc).astimezone(NY) for t in completion_ns]
    u = np.asarray([(t.hour*60+t.minute+t.second/60-570.)/390. for t in ny], dtype=np.float64)
    weekday = np.asarray([t.weekday() for t in ny])
    require(np.all((weekday <= 4) & (u > 0) & (u <= 1)), "Origin outside original NY completed clock")
    values.extend((np.sin(2*np.pi*u), np.cos(2*np.pi*u), np.sin(4*np.pi*u), np.cos(4*np.pi*u)))
    values.extend((weekday == d).astype(np.float64) for d in range(4))
    result = np.column_stack(values)
    require(result.shape == (len(history), 17) and np.isfinite(result).all(), "Context17 invalid")
    return result


@dataclass(frozen=True)
class FrozenScaler:
    state: StateKey
    roles_id: str
    q_id: str
    n_q: int
    x_mean: float
    x_std: float
    anchor_mean: float
    anchor_std: float
    context_mean: tuple
    context_std: tuple
    y_mean: tuple
    y_std: tuple

    @property
    def scaler_id(self):
        return object_id(asdict(self))

    @property
    def zero_scaled(self):
        return frozen((-np.asarray(self.y_mean, dtype=np.float64)/np.asarray(self.y_std, dtype=np.float64)).astype(np.float32))

    def inverse(self, state, prediction):
        require(state == self.state, "Foreign state scaler inverse")
        p = np.asarray(prediction)
        require(p.ndim == 2 and p.shape[1] == len(self.y_mean) and p.dtype in (np.float32, np.float64)
                and np.isfinite(p).all(), "Invalid scaled prediction")
        result = p.astype(np.float64)*np.asarray(self.y_std) + np.asarray(self.y_mean)
        require(np.isfinite(result).all(), "Nonfinite inverse")
        return result

    def zero_inverse_bound(self):
        return 8*np.finfo(np.float32).eps*np.maximum(1., np.maximum(np.abs(self.y_mean), np.abs(self.y_std)))


def endpoint_alpha(prediction, target):
    p, y = np.asarray(prediction), np.asarray(target)
    require(p.ndim == 2 and p.shape == y.shape and p.shape[0] > 0 and p.shape[1] > 0
            and p.dtype in (np.float32, np.float64) and y.dtype == np.float64
            and np.isfinite(p).all() and np.isfinite(y).all(), "Invalid raw calibration paths")
    p = p.astype(np.float64)
    numerator = float(np.sum(p[:, -1]*y[:, -1], dtype=np.float64))
    denominator = float(np.sum(p[:, -1]*p[:, -1], dtype=np.float64))
    require(math.isfinite(numerator) and math.isfinite(denominator) and denominator >= 0, "Nonfinite alpha statistics")
    alpha = float(np.clip(numerator/denominator, 0., 1.)) if denominator != 0 else 0.
    return alpha


def apply_alpha(record, state, model_id, raw_path):
    require(record["state"] == state.as_dict() and record["model_id"] == model_id, "Foreign alpha state/model")
    base = {k: v for k, v in record.items() if k != "alpha_id"}
    require(record.get("alpha_id") == object_id(base), "Alpha identity changed")
    alpha = record["alpha"]
    require(type(alpha) in (int, float) and math.isfinite(alpha) and 0 <= alpha <= 1, "Invalid alpha")
    p = np.asarray(raw_path)
    require(p.ndim == 2 and p.dtype in (np.float32, np.float64) and np.isfinite(p).all(), "Invalid raw output path")
    return p.astype(np.float64)*alpha


class CommonAdapter:
    def __init__(self, state, gate, *, close, end_ns, session_date, roles, split_actions=()):
        require(isinstance(state, StateKey) and isinstance(gate, GlobalPhaseGate), "State/gate required")
        self.state, self.gate = state, gate
        self.L, self.H = TASKS[state.task]
        require(isinstance(close, np.ndarray) and close.dtype == np.float64 and close.ndim == 1
                and np.isfinite(close).all() and np.all(close > 0), "Invalid nominal close vector")
        n = len(close)
        require(isinstance(end_ns, np.ndarray) and end_ns.dtype == np.int64 and end_ns.shape == (n,)
                and np.all(end_ns[1:] > end_ns[:-1]), "Invalid bar clock")
        require(isinstance(session_date, np.ndarray) and session_date.dtype.kind == "U" and session_date.shape == (n,)
                and np.all(session_date[1:] >= session_date[:-1]), "Invalid historical sessions")
        for value in np.unique(session_date):
            require(date.fromisoformat(str(value)).isoformat() == str(value), "Noncanonical session date")
        require(isinstance(roles, dict) and set(roles) == set(ROLES), "Exact Q/S/C_cal/T roles required")
        self.roles = {}
        for name in ROLES:
            o = roles[name]
            require(isinstance(o, np.ndarray) and o.dtype == np.int64 and o.ndim == 1 and len(o) > 0
                    and np.all(o[1:] > o[:-1]) and np.all(o >= self.L-1) and np.all(o <= n-1-self.H),
                    "Invalid role origins: "+name)
            self.roles[name] = frozen(o)
        self.roles = MappingProxyType(self.roles)
        require(len(self.roles["Q"]) >= 2, "At least two Q origins required")
        joined = np.concatenate(tuple(self.roles.values()))
        require(len(np.unique(joined)) == len(joined), "Role origins overlap")
        for left, right in zip(ROLES[:-1], ROLES[1:]):
            require(end_ns[self.roles[left][-1]+self.H] < end_ns[self.roles[right][0]], "Strict role label purge failed")
        if state.task.startswith("minute"):
            require(all(np.all(session_date[o] == session_date[o+self.H]) for o in self.roles.values()),
                    "Original minute labels must stay in session")
        self._close, self._end, self._session = frozen(close), frozen(end_ns), frozen(session_date)
        self._log_close = frozen(np.log(self._close))
        log_split = np.zeros(n, dtype=np.float64); previous = None
        for action in split_actions:
            require(isinstance(action, dict) and set(action) == {"date", "ratio"}, "Split action schema")
            when, ratio = action["date"], action["ratio"]
            require(isinstance(when, str) and date.fromisoformat(when).isoformat() == when
                    and (previous is None or when > previous) and type(ratio) in (int, float)
                    and math.isfinite(ratio) and ratio > 0, "Invalid ordered split action")
            log_split[self._session >= when] += np.log(float(ratio)); previous = when
        self._log_split = frozen(log_split)
        self.roles_id = object_id({"state":state.as_dict(), "roles":{k:array_id(v) for k,v in self.roles.items()}})
        self._scaler = None
        gate.register_state(state, self.roles_id)

    def _origins(self, key, origins, operation):
        require(isinstance(key, AccessKey) and key.state == self.state, "Foreign adapter state key")
        self.gate.check(key, operation)
        o = origins
        require(isinstance(o, np.ndarray) and o.dtype == np.int64 and o.ndim == 1 and len(o) > 0
                and len(np.unique(o)) == len(o), "Batch needs unique int64 origins")
        allowed = self.roles[key.role]; positions = np.searchsorted(allowed, o)
        require(np.all(positions < len(allowed)) and np.array_equal(allowed[np.minimum(positions, len(allowed)-1)], o),
                "Batch origin outside keyed role")
        return o

    def _raw(self, origins):
        idx = origins[:, None] + np.arange(1-self.L, 1)
        x = self._log_close[idx] + self._log_split[idx] - self._log_split[origins, None] - self._log_close[origins, None]
        g = np.zeros(x.shape, dtype=np.bool_); g[:, 1:] = self._session[idx[:, 1:]] != self._session[idx[:, :-1]]
        last = np.max(np.where(g, np.arange(self.L), 0), axis=1)
        m = np.arange(self.L)[None, :] >= last[:, None]
        return {"history":x, "boundary":g, "suffix_mask":m, "anchor":self._close[origins,None],
                "context":context17(x, self._end[origins], self.H)}

    def _labels(self, origins):
        future = origins[:,None] + np.arange(1, self.H+1)
        return np.log(self._close[future]/self._close[origins,None])

    def raw_inputs(self, key, origins):
        o = self._origins(key, origins, "inputs")
        return {k:frozen(v) for k,v in self._raw(o).items()}

    @property
    def scaler(self):
        require(self._scaler is not None, "Q scaler not fitted")
        return self._scaler

    def fit_scaler(self, key):
        self._origins(key, self.roles["Q"], "fit_scaler")
        require(self._scaler is None, "Q scaler immutable: second fit denied")
        sx = sx2 = sa = sa2 = 0.
        sc = np.zeros(17); sc2 = np.zeros(17); sy = np.zeros(self.H); sy2 = np.zeros(self.H)
        q = self.roles["Q"]
        for start in range(0, len(q), 128):
            o=q[start:start+128]; raw=self._raw(o); x,a,c=raw["history"],raw["anchor"],raw["context"]; y=self._labels(o)
            sx += float(x.sum(dtype=np.float64)); sx2 += float(np.square(x).sum(dtype=np.float64))
            sa += float(a.sum(dtype=np.float64)); sa2 += float(np.square(a).sum(dtype=np.float64))
            sc += c.sum(axis=0,dtype=np.float64); sc2 += np.square(c).sum(axis=0,dtype=np.float64)
            sy += y.sum(axis=0,dtype=np.float64); sy2 += np.square(y).sum(axis=0,dtype=np.float64)
        n=len(q)
        def stats(total, squares, count):
            mean=total/count; std=np.sqrt(np.maximum(squares/count-mean*mean,0.))
            std=np.where(std>1e-12,std,1.)
            require(np.isfinite(mean).all() and np.isfinite(std).all(), "Nonfinite Q statistics")
            return mean,std
        xm,xs=stats(sx,sx2,n*self.L); am,astd=stats(sa,sa2,n)
        cm,cs=stats(sc,sc2,n); ym,ys=stats(sy,sy2,n)
        value=FrozenScaler(self.state,self.roles_id,array_id(q),n,float(xm),float(xs),float(am),float(astd),
                           tuple(map(float,cm)),tuple(map(float,cs)),tuple(map(float,ym)),tuple(map(float,ys)))
        self.gate.bind_scaler(self.state,self.roles_id,value.scaler_id)
        self._scaler=value
        return value

    def _checked_scaler(self, scaler):
        s=self.scaler if scaler is None else scaler
        require(isinstance(s,FrozenScaler) and s.state==self.state and s.roles_id==self.roles_id
                and s.scaler_id==self.scaler.scaler_id, "Foreign/mutated state scaler")
        return s

    def network_inputs(self,key,origins,scaler=None):
        o=self._origins(key,origins,"inputs"); s=self._checked_scaler(scaler); raw=self._raw(o)
        values={"x":((raw["history"]-s.x_mean)/s.x_std).astype(np.float32)[...,None],
                "anchor":((raw["anchor"]-s.anchor_mean)/s.anchor_std).astype(np.float32),
                "context":((raw["context"]-np.asarray(s.context_mean))/np.asarray(s.context_std)).astype(np.float32),
                "boundary":raw["boundary"],"suffix_mask":raw["suffix_mask"],"zero_scaled":s.zero_scaled,
                "origins":o}
        require(all(np.isfinite(values[k]).all() for k in ("x","anchor","context","zero_scaled")), "Nonfinite FP32 inputs")
        return {k:frozen(v) for k,v in values.items()}

    def labels(self,key,origins,*,scaled=False,scaler=None):
        o=self._origins(key,origins,"labels"); y=self._labels(o)
        if scaled:
            s=self._checked_scaler(scaler); y=((y-np.asarray(s.y_mean))/np.asarray(s.y_std)).astype(np.float32)
        require(np.isfinite(y).all(), "Nonfinite labels")
        return frozen(y)

    def ridge_features(self,key,origins,scaler=None):
        v=self.network_inputs(key,origins,scaler)
        x=v["x"][:,:,0]; g=v["boundary"]; m=v["suffix_mask"]
        product=(x*m.astype(np.float32)).astype(np.float32)
        result=np.concatenate((x.astype(np.float64),g.astype(np.float64),m.astype(np.float64),
                               product.astype(np.float64),v["anchor"].astype(np.float64),v["context"].astype(np.float64)),axis=1)
        require(result.shape==(len(origins),4*self.L+18) and np.isfinite(result).all(), "E_Ridge dictionary mismatch")
        return frozen(result)

    def calibrate(self,key,model_id,raw_prediction):
        o=self._origins(key,self.roles["C_cal"],"alpha")
        model=self.gate.model_entry(self.state,model_id)
        p=np.asarray(raw_prediction)
        require(p.shape==(len(o),self.H), "Calibration prediction role/path shape mismatch")
        value={"state":self.state.as_dict(),"model_id":model_id,"model_sha256":model["model_sha256"],
               "scaler_id":self.scaler.scaler_id,"roles_id":self.roles_id,
               "alpha":endpoint_alpha(p,self._labels(o))}
        value["alpha_id"]=object_id(value)
        return value
