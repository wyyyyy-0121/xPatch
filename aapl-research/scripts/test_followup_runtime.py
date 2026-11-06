import copy
import gc
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fusion.next import common,accounting,cache,checkpoint,experiment,trainer
from fusion.next.common import ROOT,digest,read_json,source_identity,write_json
from fusion.next.models import NeuralFusion
from fusion.next.hardware import runtime
from fusion.next.panel import CoverageSampler
from fusion.next.tests_data import two_sessions
from fusion.next.tests_training import make_panel
from scripts import followup_runtime as follow
from scripts.run_fusion_followup import clone_prefix,state_hash


@pytest.fixture
def paths():
    (follow.FOLLOW/'tests').mkdir(exist_ok=True)
    (follow.FOLLOW/'runs').mkdir(exist_ok=True)
    with ExitStack() as stack:
        read = Path(stack.enter_context(TemporaryDirectory(dir=follow.FOLLOW/'tests',prefix='inputs_')))
        write = Path(stack.enter_context(TemporaryDirectory(dir=follow.FOLLOW/'runs',prefix='unit_')))
        (read/'data').mkdir()
        yield read,write


def test_matrix_changes_only_the_prespecified_factors_and_does_not_share_mutable_tables():
    reference = {'family':'fusion','candidate_id':'base','foundation':True,'foundation_mode':'frozen',
                 'loss':'scaled','representation_cache':True,'lookback':240,
                 'sampler_policy':'balanced_complete_coverage_v1',
                 'updates_by_fit_end':{'2024-12-30':17884}}
    before = copy.deepcopy(reference)
    rows = follow.candidates(reference)
    assert reference == before
    assert [(r['config']['foundation'],r['config']['loss']) for r in rows] == [
        (True,'raw'),(False,'scaled'),(False,'raw')]
    assert all(not r['old_optimizer_or_checkpoint_reused'] for r in rows)
    assert len({r['config']['candidate_id'] for r in rows}) == 3
    for row in rows:
        for key in ('lookback','sampler_policy','updates_by_fit_end','foundation_mode'):
            assert row['config'][key] == reference[key]
    rows[0]['config']['updates_by_fit_end']['2024-12-30'] = 1
    assert rows[1]['config']['updates_by_fit_end'] == reference['updates_by_fit_end']


def test_large_run_date_and_original_deadline_are_not_reset():
    limit = {'total_wall_hours':100,'deadline_utc':follow.DEADLINE}
    with pytest.raises(RuntimeError,match='November 8'):
        follow.launch_window(limit,4,datetime.fromisoformat('2026-11-08T08:59:59+08:00'))
    valid = follow.launch_window(limit,4,datetime.fromisoformat(follow.EARLIEST))
    assert 32.5 < valid['remaining_hours'] < 32.6
    assert valid['training_admitted'] is False
    with pytest.raises(RuntimeError,match='reserve'):
        follow.launch_window(limit,20,datetime.fromisoformat(follow.EARLIEST))
    with pytest.raises(TimeoutError):
        follow.launch_window(limit,1,datetime.fromisoformat(follow.DEADLINE))
    with pytest.raises(ValueError,match='must not change'):
        follow.launch_window({**limit,'deadline_utc':'2026-11-10T00:00:00+00:00'},1)


def test_writes_and_locks_are_isolated_even_when_an_exception_occurs(paths):
    read,write = paths
    before = [(m,m.OUT,m.write_json) for m in (common,accounting,cache,experiment,trainer)]
    with pytest.raises(RuntimeError,match='controlled failure'):
        with follow.isolated_outputs(write,read):
            common.write_json(write/'allowed.json',{'ok':True})
            with pytest.raises(ValueError,match='leaves'):
                checkpoint.write_json(read/'forbidden.json',{'ok':False})
            with pytest.raises(ValueError,match='leaves'):
                common.write_json(write/'data/forbidden.json',{'ok':False})
            with pytest.raises(RuntimeError,match='already owns'):
                with follow.isolated_outputs(write,read):
                    pass
            raise RuntimeError('controlled failure')
    for module,output,writer in before:
        assert module.OUT == output and module.write_json is writer
    assert not (read/'forbidden.json').exists()
    assert not (read/'data/forbidden.json').exists()
    assert read_json(write/'allowed.json') == {'ok':True}
    with follow.isolated_outputs(write,read):
        assert trainer.OUT == write


def test_warm_latents_are_exact_and_original_cache_bytes_never_change(paths,monkeypatch):
    read,write = paths
    (read/'data/AAPL').mkdir()
    write_json(read/'data/AAPL/manifest.json',{'fixture':'development data'})
    np.save(read/'data/timestamps_ns.npy',np.arange(3,dtype=np.int64))
    panel = SimpleNamespace(stocks=['AAPL'],dates=np.array(['2025-01-01','2025-01-02','2026-07-01']))
    class Branch:
        mode='frozen';receipt={'revision':'fixed'}
        backbone=SimpleNamespace(config=SimpleNamespace(d_model=2))
        calls=0
        def latent(self,context):
            self.calls += len(context)
            return torch.stack([context.sum((1,2)),context.mean((1,2))],1)
    branch = Branch()
    config = {'lookback':48,'fp16_cache_admitted':True,'precision':'fp32'}
    batch = {'stock':torch.tensor([0,0]),'row':torch.tensor([0,1]),
             'foundation_context':torch.arange(12,dtype=torch.float32).reshape(2,2,3)/7,
             'local':torch.zeros(2,2,1)}
    monkeypatch.setattr(cache,'OUT',read)
    old = cache.RepresentationCache(panel,branch,config,storage_dtype='float16')
    old.get({name:value[:1] for name,value in batch.items()})
    hashes = {str(p.relative_to(read)):digest(p) for p in read.rglob('*') if p.is_file()}
    before_calls = branch.calls
    with follow.isolated_outputs(write,read) as binding:
        local = cache.RepresentationCache(panel,branch,config,storage_dtype='float16')
        via_trainer = trainer.RepresentationCache(panel,branch,config,storage_dtype='float16')
        ref = binding.caches[-1]
        assert ref() is via_trainer
        del via_trainer;gc.collect()
        assert ref() is None  # Do not retain a finished fit's CUDA backbone.
        warm = follow.ReadOnlyWarmCache(local,read/'representation_cache')
        actual = warm.get(batch)
        expected = branch.latent(batch['foundation_context']).half().float()
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        assert branch.calls-before_calls == 3  # one missing row plus two explicit reference rows
        assert warm.reused_rows == 1 and warm.local_rows == 1
        assert all(not values.flags.writeable and not flags.flags.writeable
                   for values,flags in warm.originals.values() if values is not None)
        with pytest.raises(ValueError,match='boundary'):
            warm.get({**batch,'row':torch.tensor([0,2])})
    assert hashes == {str(p.relative_to(read)):digest(p) for p in read.rglob('*') if p.is_file()}
    with pytest.raises(ValueError,match='outlive'):
        warm.get(batch)


@pytest.mark.parametrize('loss',['raw','scaled'])
def test_actual_no_foundation_fusion_resumes_exactly_in_the_new_namespace(paths,loss):
    read,write = paths
    frame,starts,ends = two_sessions()
    panel = make_panel(frame,starts,ends)
    config = {'family':'fusion','lookback':48,'hidden':32,'projection':64,
              'market_encoder':'transformer','foundation':False,'market':True,'residual_cap':.25,
              'lr':.001,'loss':loss,'batch_size':4,'total_updates':5,'precision':'fp32',
              'sampler_policy':CoverageSampler.policy,'prefetch_batches':2,'adaptation_fraction':.2}
    class Experts:
        available={0:np.ones(len(panel.dates),dtype=bool)}
        sources={'AAPL':'synthetic expert fixture, never used for market conclusions'}
        def predict(self,stocks,rows):
            return np.tile((np.sin(rows)*.0001)[:,None],(1,4)).astype(np.float32)
    engine = source_identity()
    with follow.isolated_outputs(write,read):
        def make(namespace):
            # The admitted neural_trial resets the seed before every model
            # reconstruction. Frozen inactive heads are regenerated, not saved.
            assert runtime(2026) == 'cpu'
            return trainer.Trainer(panel,NeuralFusion(config,12,12,1,device='cpu'),config,
                                   '2021-01-05',namespace,experts=Experts(),device='cpu')
        torch.manual_seed(91)
        uninterrupted=make('uninterrupted');uninterrupted.run_until(5)
        expected_rng=torch.random.get_rng_state().clone()
        expected_hash=state_hash(uninterrupted)
        torch.manual_seed(91)
        first=make('resumed');first.run_until(3)
        paired=make('paired_probe_recovery')
        clone_prefix(first.path,paired.path)
        paired.resume();paired.run_until(5)
        assert state_hash(paired) == expected_hash
        with pytest.raises(ValueError,match='Preserve'):
            clone_prefix(first.path,paired.path)
        with pytest.raises(ValueError,match='new training namespace'):
            clone_prefix(first.path,read/'forbidden.pt')
        restored=make('resumed');restored.resume();restored.run_until(5)
        assert restored.last_loss == uninterrupted.last_loss
        assert torch.equal(torch.random.get_rng_state(),expected_rng)
        assert restored.recipe == uninterrupted.recipe
        for name,value in uninterrupted.model.state_dict().items():
            torch.testing.assert_close(value,restored.model.state_dict()[name],rtol=0,atol=0,msg=name)
        for name,sampler in uninterrupted.samplers.items():
            assert sampler.state_dict() == restored.samplers[name].state_dict()
        assert restored.path.is_relative_to(write/'training')
    assert engine == source_identity()
    assert not (read/'training').exists()
