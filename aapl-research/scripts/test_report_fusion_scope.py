from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from fusion.next.common import OUT,digest,identity,write_json
from scripts import report_fusion_scope as report


def fixture(root):
    parent=root/'search_manifest.json';write_json(parent,{'identity':'parent'})
    runner=root/'performance/scope_100h_runner.txt';runner.parent.mkdir();runner.write_text('fixed runner\n')
    limit=root/'performance/scope_100h_time_limit.json';write_json(limit,{'total_wall_hours':100})
    scope={'parent_search_sha256':digest(parent),'parent_search_identity':'parent','source_identity':'source',
        'execution_script_sha256':digest(runner),'hard_limit_receipt_sha256':digest(limit),
        'authorization':'user approved compression','counts':{'fusion':2},'simple_count':12,
        'validation':['2025Q1','2025Q3','2026Q1'],'seeds':[2026,2027,2028],
        'training_budget_policy':{'total_coverage_equivalent_epochs':2},
        'mechanisms_retained':['gate_vs_plain_mlp','frozen_vs_lora'],
        'mechanisms_deferred':['no_foundation'],'projection':{'hours_with_margin_and_reserve':91.56}}
    scope['identity']=identity(scope);write_json(root/'performance/scope_100h.json',scope)
    return scope


def test_scope_report_keeps_incomplete_stages_incomplete_and_provenance_returnable(monkeypatch):
    with TemporaryDirectory(prefix='unit_scope_report_',dir=OUT) as temporary:
        root=Path(temporary);scope=fixture(root);monkeypatch.setattr(report,'source_identity',lambda:'source')
        value=report.inspect(root)
        assert not any(row['completed'] for row in value['stages'].values())
        assert not value['original_complete_scope_claimed']
        assert value['mechanisms_deferred']==['no_foundation']
        assert value['scope_provenance_passes_existing_return_allowlist']
        write_json(root/'F3_complete.json',{'scope_identity':scope['identity'],'source_identity':'source',
            'scope_amendment_sha256':digest(root/'performance/scope_100h.json'),
            'original_complete_scope_claimed':False})
        assert report.inspect(root)['stages']['F3']['completed']


def test_scope_report_rejects_completed_stage_from_the_old_full_scope(monkeypatch):
    with TemporaryDirectory(prefix='unit_scope_report_',dir=OUT) as temporary:
        root=Path(temporary);fixture(root);monkeypatch.setattr(report,'source_identity',lambda:'source')
        write_json(root/'F3_complete.json',{'source_identity':'source'})
        with pytest.raises(ValueError,match='Stage completion'):report.inspect(root)


@pytest.mark.parametrize('artifact',['search_manifest.json','performance/scope_100h_runner.txt',
                                   'performance/scope_100h_time_limit.json'])
def test_scope_report_rejects_modified_parent_runner_or_time_limit(monkeypatch,artifact):
    with TemporaryDirectory(prefix='unit_scope_report_',dir=OUT) as temporary:
        root=Path(temporary);fixture(root);monkeypatch.setattr(report,'source_identity',lambda:'source')
        (root/artifact).write_text('{}\n')
        with pytest.raises(ValueError):report.inspect(root)


def test_portable_stage_requires_the_declared_policy(monkeypatch):
    with TemporaryDirectory(prefix='unit_scope_report_',dir=OUT) as temporary:
        root=Path(temporary);scope=fixture(root);monkeypatch.setattr(report,'source_identity',lambda:'source')
        policy={'scope_identity':scope['identity'],'base_engine_source_identity':'source',
            'inference_precision':'float32','training_precision_changed':False}
        for field,name in [('runtime_sha256','portable_inference_runtime.py'),('CPU_entrypoint_sha256','portable_cpu_run.py'),
                           ('launcher_sha256','start_portable_fusion.py'),('rescore_script_sha256','prepare_portable_inference.py')]:
            path=root/'performance'/('portable_'+name+'.txt');path.write_text(name);policy[field]=digest(path)
        policy['identity']=identity(policy);write_json(root/'performance/portable_inference_policy.json',policy)
        stage={'scope_identity':scope['identity'],'source_identity':'source',
            'scope_amendment_sha256':digest(root/'performance/scope_100h.json'),'original_complete_scope_claimed':False}
        write_json(root/'F3_complete.json',stage)
        with pytest.raises(ValueError,match='lacks the portable inference policy'):report.inspect(root)
        stage.update(numerical_policy_identity=policy['identity'],evaluation_precision='float32')
        write_json(root/'F3_complete.json',stage)
        assert report.inspect(root)['stages']['F3']['completed']
        (root/'performance/portable_portable_inference_runtime.py.txt').write_text('changed')
        with pytest.raises(ValueError,match='implementation changed'):report.inspect(root)
