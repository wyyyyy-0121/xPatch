"""Q-only enhanced Ridge and mean; one fixed dictionary and penalty.

The accepted FP64 sufficient-statistics/solve implementation is reused by hash.
No file/data loading, fitting or calibration occurs at module import.
"""
from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path
import numpy as np
from common_adapter import access

CORE_PATH = Path(__file__).resolve().parent.parent/'20261110_market_context_information_screen/code/screen_math.py'
CORE_SHA256 = '69a856f201f25f4e977233ac0e4a4b922679446ccf337fa2e4c1f51acc3a8164'


def load_core():
    if hashlib.sha256(CORE_PATH.read_bytes()).hexdigest() != CORE_SHA256:
        raise ValueError('Frozen Ridge numerical implementation changed')
    spec = importlib.util.spec_from_file_location('accepted_fixed_ridge_numerics', CORE_PATH)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def fit_controls(adapter, guard=lambda: None, observer=None):
    """Return eight-style fit records for E_Ridge/Qmean (neural handled elsewhere)."""
    if adapter.gate.phase != 'TRAIN':
        raise ValueError('Controls may fit only in Q/S training phase')
    core = load_core(); stats = core.empty_stats(4*adapter.L+18, adapter.H)
    q = adapter.roles['Q']
    for begin in range(0, len(q), 128):
        guard(); origins = q[begin:begin+128]
        x = adapter.ridge_features(access(adapter.state, 'Q', 'inputs'), origins)
        y = adapter.labels(access(adapter.state, 'Q', 'labels'), origins)
        core.update(stats, x, y)
        if observer is not None:
            observer({'origins_done': begin+len(origins), 'Q_count': len(q)})
    guard(); model = core.solve(stats); guard()
    mean = np.asarray(adapter.scaler.y_mean, dtype=np.float64)
    if stats['n'] != len(q) or not np.array_equal(model['y_mean'], mean):
        raise ValueError('Complete Q-domain model/scaler target mean mismatch')
    return {'E_Ridge': model, 'Q_mean': {'path': mean.copy(), 'n_train': np.asarray(len(q), dtype=np.int64)}}


def predict_control(model_name, model, adapter, role, origins):
    if model_name == 'Q_mean':
        # Check the role/operation and origin membership even for a constant model.
        adapter._origins(access(adapter.state, role, 'inputs'), origins, 'inputs')
        return np.broadcast_to(model['path'], (len(origins), adapter.H)).copy()
    if model_name == 'E_Ridge':
        x = adapter.ridge_features(access(adapter.state, role, 'inputs'), origins)
        return load_core().predict(model, x)
    raise ValueError('Only the two frozen Q controls')
