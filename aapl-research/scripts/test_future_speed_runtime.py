import copy

import numpy as np
import pytest
import torch

from fusion.next.panel import Panel
from fusion.next.tests_data import two_sessions
from fusion.next.tests_training import make_panel
from scripts import future_speed_runtime as fast


@pytest.fixture
def data(tmp_path):
    frame, starts, ends = two_sessions()
    panel = make_panel(frame, starts, ends)
    for name, values in list(panel.features.items()):
        path = tmp_path/(name+'.npy')
        np.save(path, values)
        panel.features[name] = np.load(path, mmap_mode='r')
    fast.clear()
    yield panel, panel.scaler('2021-01-04')
    fast.clear()


@pytest.mark.parametrize('information', ['ohlcv', 'close'])
@pytest.mark.parametrize('lookback', [48, 96, 240])
def test_real_causal_fields_match_original_exactly(data, information, lookback):
    panel, scalers = data
    config = {'information': information, 'lookback': lookback}
    stocks = np.zeros(4, dtype=int)
    rows = np.array([390, 430, 430, 700])
    rng = torch.random.get_rng_state().clone()
    for labels in (True, False):
        a = Panel.windows(panel, stocks, rows, config, scalers, include_labels=labels)
        b = fast.windows(Panel.windows, panel, stocks, rows, config, scalers, include_labels=labels)
        assert a.keys() == b.keys()
        for name in a:
            torch.testing.assert_close(a[name], b[name], rtol=0, atol=0, equal_nan=True)
    assert torch.equal(rng, torch.random.get_rng_state())
    # Preparation state must not make coverage sampler deepcopy fail.
    copy.deepcopy(panel)


def test_changed_scaler_cannot_reuse_old_transforms(data):
    panel, scalers = data
    args = (panel, np.zeros(2, dtype=int), np.array([430, 700]), {'lookback': 96}, scalers)
    first = fast.windows(Panel.windows, *args)['market'].clone()
    scalers['SPY']['center'][3] += .001
    actual = fast.windows(Panel.windows, *args)['market']
    torch.testing.assert_close(actual, Panel.windows(*args)['market'], rtol=0, atol=0)
    assert not torch.equal(first, actual)


def test_live_mutable_panels_use_original_path(data):
    panel, scalers = data
    panel.features['SPY'] = np.array(panel.features['SPY'])
    def original(*args, **kwargs):
        return {'original': True}
    assert fast.windows(original, panel, None, None, {'lookback': 96}, scalers) == {'original': True}


def test_future_label_guard_and_inference_do_not_read_targets(data):
    panel, scalers = data
    last = len(panel.dates)-1
    with pytest.raises(ValueError, match='label'):
        fast.windows(Panel.windows, panel, [0], [last], {'lookback': 96}, scalers)
    class Forbidden:
        def __getitem__(self, key):
            raise AssertionError('Inference read targets')
    panel.targets = {'AAPL': Forbidden(), 'SPY': Forbidden()}
    result = fast.windows(Panel.windows, panel, [0], [last], {'lookback': 96}, scalers, include_labels=False)
    assert 'target' not in result and 'step_targets' not in result


def test_memory_limit_evicts_without_changing_inputs(data, monkeypatch):
    panel, scalers = data
    monkeypatch.setattr(fast, 'MAX_BYTES', len(panel.dates)*12*4)
    for information in ['ohlcv', 'close', 'ohlcv']:
        args = (panel, [0], [430], {'lookback': 48, 'information': information}, scalers)
        actual = fast.windows(Panel.windows, *args)
        torch.testing.assert_close(actual['market'], Panel.windows(*args)['market'], rtol=0, atol=0)
        assert fast._bytes <= fast.MAX_BYTES


@pytest.mark.parametrize('family', ['xpatch', 'patchtst'])
def test_actual_training_and_checkpoint_resume_stay_exact(data, monkeypatch, family):
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from fusion.next.common import OUT
    from fusion.next.expert_models import NativeExpert
    from fusion.next.panel import CoverageSampler
    from fusion.next.trainer import Trainer
    panel, _ = data
    config = {'family': family, 'lookback': 48, 'lr': .001, 'batch_size': 4,
              'total_updates': 4, 'loss': 'raw', 'dropout': .1,
              'sampler_policy': CoverageSampler.policy, 'prefetch_batches': 2}
    original = Panel.windows
    with TemporaryDirectory(prefix='unit_speed_resume_', dir=OUT/'training') as temporary:
        namespace = Path(temporary).name
        torch.manual_seed(81)
        reference = Trainer(panel, NativeExpert(config, 24, 1), config,
                            '2021-01-05', namespace+'/original', device='cpu')
        reference.run_until(4)
        expected_rng = torch.random.get_rng_state().clone()
        monkeypatch.setattr(Panel, 'windows', lambda self, *a, **kw: fast.windows(original, self, *a, **kw))
        torch.manual_seed(81)
        first = Trainer(panel, NativeExpert(config, 24, 1), config,
                        '2021-01-05', namespace+'/optimized', device='cpu')
        first.run_until(2)
        restored = Trainer(panel, NativeExpert(config, 24, 1), config,
                           '2021-01-05', namespace+'/optimized', device='cpu')
        restored.resume()
        restored.run_until(4)
        assert restored.last_loss == reference.last_loss
        assert restored.recipe == reference.recipe
        assert torch.equal(expected_rng, torch.random.get_rng_state())
        for name, expected in reference.model.state_dict().items():
            torch.testing.assert_close(expected, restored.model.state_dict()[name], rtol=0, atol=0)
        assert restored.samplers['main'].state_dict() == reference.samplers['main'].state_dict()
