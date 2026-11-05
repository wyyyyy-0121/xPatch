"""Exact background-input reuse for future processes; the admitted engine stays intact.

Only immutable, read-only development panels are eligible. Live feed panels use
the original path. No model, sampler, RNG, precision, or optimizer is replaced.
"""
from collections import OrderedDict
import functools
import hashlib
from pathlib import Path
import threading
import weakref

import numpy as np
import torch


MAX_BYTES = 256 * 2**20
_entries = OrderedDict()
_bytes = 0
_lock = threading.RLock()
_installed = False


def clear():
    global _bytes
    with _lock:
        _entries.clear()
        _bytes = 0


def immutable(panel):
    return all(isinstance(panel.features[name], np.memmap)
               and panel.features[name].mode == 'r'
               and not panel.features[name].flags.writeable
               for name in panel.background)


def normalized(panel, name, scalers, information):
    """Reuse pointwise transforms; retain the original float32 operation order."""
    from fusion.next.panel import close_signals
    global _bytes
    s = scalers[name]
    stamp = hashlib.sha256()
    for key in ('volume_seasonal', 'center', 'std', 'close_center', 'close_std'):
        value = np.asarray(s[key])
        stamp.update(str((value.shape, value.dtype.str)).encode())
        stamp.update(value.tobytes())
    key = (id(panel), id(panel.features[name]), id(panel.clock_minute), name,
           information, stamp.digest())
    with _lock:
        saved = _entries.get(key)
        if saved is not None and saved[0]() is panel:
            _entries.move_to_end(key)
            return saved[1]
        raw = np.array(panel.features[name], dtype=np.float32, copy=True)
        raw[:, 4] -= s['volume_seasonal'][panel.clock_minute]
        values = np.clip((close_signals(raw) - s['close_center']) / s['close_std']
                         if information == 'close' else (raw - s['center']) / s['std'], -10, 10)
        if values.nbytes > MAX_BYTES:
            raise MemoryError('Background reuse exceeds its explicit memory budget')
        while _entries and _bytes + values.nbytes > MAX_BYTES:
            _, (_, evicted) = _entries.popitem(last=False)
            _bytes -= evicted.nbytes
        _entries[key] = (weakref.ref(panel), values)
        _bytes += values.nbytes
        return values


def windows(original, panel, stock_ids, rows, config, scalers, experts=None,
            device='cpu', cached=None, include_labels=True):
    width = 4 if config.get('information') == 'close' else 12
    if (not config.get('market_information', True) or not panel.background or not immutable(panel)
            or len(panel.dates)*width*4 > MAX_BYTES):
        return original(panel, stock_ids, rows, config, scalers, experts, device, cached, include_labels)
    # Let the original implementation validate all origins, labels and local
    # inputs before doing any extra indexing. Configurations remain immutable.
    batch = original(panel, stock_ids, rows, {**config, 'market_information': False},
                     scalers, experts, device, cached, include_labels)
    rows = np.asarray(rows, dtype=np.int64)
    idx = rows[:, None] + np.arange(1-config['lookback'], 1)
    information = config.get('information', 'ohlcv')
    market = np.concatenate([normalized(panel, name, scalers, information)[idx]
                             for name in panel.background], axis=2)
    if not np.isfinite(market).all():
        raise ValueError('Nonfinite model inputs')
    batch['market'] = torch.as_tensor(market, device=device)
    return batch


def checked_policy():
    from fusion.next.common import OUT, ROOT, read_json, identity, digest, source_identity
    policy = read_json(OUT/'performance/future_speed_policy.json')
    if policy['identity'] != identity({k:v for k,v in policy.items() if k != 'identity'}):
        raise ValueError('Future speed policy changed')
    if policy['base_engine_source_identity'] != source_identity():
        raise ValueError('Future speed policy belongs to another engine')
    if policy['runtime_sha256'] != digest(Path(__file__)):
        raise ValueError('Future speed implementation changed')
    if policy['launcher_sha256'] != digest(ROOT/'scripts/start_fast_fusion.py'):
        raise ValueError('Future speed launcher changed')
    from scripts.portable_inference_runtime import checked_policy as numerical_policy
    if policy['numerical_policy_identity'] != numerical_policy()['identity']:
        raise ValueError('Future speed numerical policy differs')
    if not policy['exact_input_parity_passed'] or policy['warm_speedup'] <= 1:
        raise ValueError('Unverified or slower input optimization')
    return policy


def install():
    global _installed
    if _installed:
        return
    policy = checked_policy()
    from fusion.next.panel import Panel
    from fusion.next import trainer
    from fusion.next.common import OUT, utc_now, write_json
    original = Panel.windows
    @functools.wraps(original)
    def fast(self, *args, **kwargs):
        return windows(original, self, *args, **kwargs)
    Panel.windows = fast
    run_until = trainer.Trainer.run_until
    @functools.wraps(run_until)
    def recorded(self, updates, usage=None):
        relative = self.path.relative_to(OUT/'training').with_suffix('')
        write_json(OUT/'performance/speed_activations'/relative/(str(self.step)+'_'+str(updates)+'.json'),
                   {'speed_policy_identity': policy['identity'], 'at_utc': utc_now(),
                    'starting_step': self.step, 'ending_step': updates,
                    'training_recipe_unchanged': True, 'exact_pointwise_inputs': True})
        return run_until(self, updates, usage)
    trainer.Trainer.run_until = recorded
    _installed = True
