"""Freeze two post-hoc seeds before running monthly split-neutral repetitions."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

from multiscale.bars import file_digest
from multiscale.spec import MODELS,OUT


def main():
    previous=OUT/'private_split_neutral_monthly/20261107T145708694676Z'
    state=json.loads((previous/'PRIVATE_QUEUE_STATE.json').read_text())
    result=previous/'PRIVATE_RESULTS.json'
    source_plan=previous/'PRIVATE_PROTOCOL.json'
    uncertainty=OUT/'private_split_neutral_uncertainty/20261107T145946279609Z/PRIVATE_DESCRIPTIVE_INTERVALS.json'
    interval_data=json.loads(uncertainty.read_text())
    if (state['status']!='completed_posthoc_development_only' or state['completed']!=5 or
            state['results_sha256']!=file_digest(result) or interval_data['source_results_sha256']!=file_digest(result)):
        raise ValueError('The original exploratory comparison has not been verified')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    parent=OUT/'private_split_neutral_seed_replication';parent.mkdir(parents=True,exist_ok=True)
    directory=parent/stamp;directory.mkdir(exist_ok=False)
    exposure=OUT/'POST_CUTOFF_EXPOSURE.md';before=exposure.read_bytes();(directory/'EXPOSURE_BEFORE.md').write_bytes(before)
    plan={'status':'private_posthoc_seed_replication_predeclared',
          'created_at_utc':datetime.now(timezone.utc).isoformat(),
          'scope':'personal non-commercial local development; all results remain local',
          'task':'monthly H21 split-neutral price log return','baseline_seed':2026,'additional_seeds':[2027,2028],
          'models':list(MODELS),'folds_per_seed':5,'max_epochs':20,'patience':3,
          'batch_size':128,'learning_rate':0.001,'ridge_alphas':[0.1,1.0,10.0],
          'parallel_workers':2,'threads_per_worker':1,'maximum_wall_seconds_per_worker':600,
          'prior_protocol_sha256':file_digest(source_plan),'prior_results_sha256':file_digest(result),
          'prior_interval_sha256':file_digest(uncertainty),
          'posthoc_reason':'The 2026-seed result and its descriptive intervals were already observed.',
          'exposure_before_sha256':hashlib.sha256(before).hexdigest(),
          'script_sha256':file_digest(Path(__file__)),'independent_test':False,'Q3_values_read':False,
          'limitations':['All seeds reuse the same exposed time periods and do not constitute independent confirmation.',
                         'This sensitivity analysis was chosen after observing the first seed.',
                         'Cash dividends are excluded and publication rights remain unconfirmed.']}
    path=directory/'PRIVATE_REPLICATION_PROTOCOL.json'
    with path.open('x',encoding='utf8') as f:json.dump(plan,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    with exposure.open('a',encoding='utf8') as f:
        f.write(f'\n## 月度拆股中性收益：事后随机种子复跑\n\n{stamp}：观察 2026 种子结果后，'
                '固定追加 2027、2028 两个种子，对相同五折和八类方法作本机敏感性检查；'
                f'重复使用已曝光时期，不能作为独立确认。计划见 {path}。\n')
    print(json.dumps({'status':plan['status'],'path':str(path),'sha256':file_digest(path),
                      'additional_seeds':plan['additional_seeds']},ensure_ascii=False))


if __name__=='__main__':main()
