"""Generate a local, explicitly non-submission English results appendix."""
from datetime import datetime,timezone
import json
from pathlib import Path

from multiscale.bars import file_digest
from multiscale.spec import OUT


def main():
    root=OUT/'private_three_scale_development/20261107T134128870773Z'
    anchor=OUT/'private_anchored_price_exploration/20261107T135350353680Z'
    controls=OUT/'private_minute_full_controls/20261107T140007359046Z'
    neural=OUT/'private_minute_full_neural/20261107T140317864464Z'
    annual=OUT/'private_annual_short_development/20261107T143734746737Z'
    annual_state=json.loads((annual/'PRIVATE_QUEUE_STATE.json').read_text())
    annual_result_path=annual/'PRIVATE_ANNUAL_RESULTS.json'
    annual_result=json.loads(annual_result_path.read_text())
    annual_plan=json.loads((annual/'PRIVATE_PROTOCOL.json').read_text())
    annual_gru_by_fold={j['fold']:json.loads((annual/j['key']/'fold_receipt.json').read_text())['models']['gru']
                        ['against_last_close_or_zero']['relative_improvement'] for j in annual_plan['jobs']
                        if j['target']=='future_log_return'}
    aggregate_paths=sorted((OUT/'private_minute_h60_aggregate').glob('*/PRIVATE_PARTIAL_AGGREGATE.json'))
    if not aggregate_paths:raise ValueError('A verified H60 aggregate is required')
    minute_aggregate=json.loads(aggregate_paths[-1].read_text())
    first_paths=sorted((OUT/'private_minute_first_fold_reports').glob('*/PRIVATE_FIRST_FOLDS.json'))
    if not first_paths:raise ValueError('A verified minute first-fold snapshot is required')
    first_folds=json.loads(first_paths[-1].read_text())
    for item in first_folds['tasks']:
        verification=json.loads(Path(item['verification']).read_text())
        if (verification['fold_receipt_sha256']!=item['receipt_sha256'] or
                file_digest(Path(verification['fold_directory'])/'fold_receipt.json')!=item['receipt_sha256']):
            raise ValueError('A minute first-fold verification changed')
    split_root=OUT/'private_split_neutral_monthly/20261107T145708694676Z'
    split_state=json.loads((split_root/'PRIVATE_QUEUE_STATE.json').read_text())
    split_result_path=split_root/'PRIVATE_RESULTS.json'
    split_result=json.loads(split_result_path.read_text())
    split_intervals_path=OUT/'private_split_neutral_uncertainty/20261107T145946279609Z/PRIVATE_DESCRIPTIVE_INTERVALS.json'
    split_intervals=json.loads(split_intervals_path.read_text())
    seed_comparison_path=OUT/'private_split_neutral_seed_replication/20261107T150345851174Z/PRIVATE_SEED_COMPARISON.json'
    seed_comparison=json.loads(seed_comparison_path.read_text())
    if any(file_digest(Path(path)/'PRIVATE_RESULTS.json')!=seed_comparison['source_result_sha256'][seed]
           for seed,path in seed_comparison['sources'].items()):
        raise ValueError('A seed comparison source changed')
    split_fold_gru=[json.loads((split_root/f'fold_{i:02d}'/'fold_receipt.json').read_text())['models']['gru']
                    ['against_last_close_or_zero']['relative_improvement'] for i in range(1,6)]
    if (not any(v<0 for v in split_fold_gru) or
            any(v['confidence_interval'] is None or not v['confidence_interval'][0]<=0<=v['confidence_interval'][1]
                for v in split_intervals['models']['gru'])):
        raise ValueError('The stated split-neutral GRU interpretation is unsupported')
    aq=json.loads((root/'PRIVATE_AGGREGATE_INDEPENDENT_CHECK.json').read_text())
    bq=json.loads((anchor/'PRIVATE_ANCHORED_INDEPENDENT_CHECK.json').read_text())
    cq=json.loads((controls/'PRIVATE_INDEPENDENT_CHECK.json').read_text())
    nq=json.loads((neural/'PRIVATE_INDEPENDENT_CHECK.json').read_text())
    if (aq.get('folds')!=50 or bq.get('folds')!=25 or cq.get('validation_origins')!=75900 or
            annual_state.get('status')!='completed_short_history_development_only' or
            annual_state.get('completed')!=10 or annual_state.get('results_sha256')!=file_digest(annual_result_path) or
            split_state.get('status')!='completed_posthoc_development_only' or
            split_state.get('results_sha256')!=file_digest(split_result_path) or
            split_intervals.get('source_results_sha256')!=file_digest(split_result_path) or
            seed_comparison.get('status')!='private_posthoc_three_seed_comparison' or
            seed_comparison.get('same_targets_and_splits') is not True or
            seed_comparison.get('source_result_sha256',{}).get('2026')!=file_digest(split_result_path) or
            minute_aggregate.get('status')!='partial_h60_full_origin_development_only' or
            first_folds.get('status')!='private_first_fold_development_snapshot' or
            nq.get('status')!='private_full_minute_neural_independently_recomputed'):
        raise ValueError('Private result verifications are incomplete')
    tasks=json.loads((root/'PRIVATE_PROTOCOL.json').read_text())['task_order']
    model_names=[('drift','Drift'),('seasonal','Seasonal'),('ridge','Ridge'),('mlp','MLP'),('gru','GRU'),('lstm','LSTM'),('xpatch','xPatch')]
    scale={'hourly':'Hourly','monthly':'Monthly','quarterly':'Quarterly','annual':'Annual (short history)'}
    completed_folds=minute_aggregate['completed_folds']
    missing_folds=[fold for fold in range(1,6) if fold not in completed_folds]
    lines=['# Private Preliminary Results — Not for Submission','',
           '**Internal development only. Data publication rights, source conventions, independent confirmation, and the complete five-scale matrix remain unresolved. Do not distribute or submit this appendix.**','',
           '## Scope and evaluation','',
           f'This appendix records real, historical exploratory runs on the existing Alpaca SIP/raw candidate. Hourly, monthly, and quarterly tasks each use five chronological development folds. With the user’s approval, the annual task uses about ten years of daily data and a twelve-month lookback; its five folds depart from the original fifteen-year requirement. The minute H60 return task currently has {len(completed_folds)} complete full-origin fold(s). All results are already-exposed development evidence, not an independent test. The 2026 Q3 period remains sealed.','',
           'Every comparison uses matched origins within the task. The endpoint mean squared error (MSE) is pooled across folds in proportion to the number of validation origins. A reported percentage is 100 × (1 − model MSE / persistence MSE): positive values denote lower error than the baseline, and negative values denote higher error. These percentages are not investment returns. Each neural configuration uses one seed, at most 20 epochs, and an inner stopping partition.','',
           'An hourly horizon counts complete regular-session hourly bars and may span sessions. Monthly H21 counts trading days; quarterly H3 counts completed monthly bars. Minute H60 targets stay within a regular session.','']
    for target,title in [('future_log_return','Cumulative log-return target'),('future_close','Nominal close-price target')]:
        lines += ['## '+title,'','| Task | Origins | '+' | '.join(label for _,label in model_names)+' |',
                  '| --- | ---: | '+' | '.join(['---:']*len(model_names))+' |']
        for profile,horizon,t in tasks:
            if t!=target:continue
            d=json.loads((root/f'PRIVATE_TASK_{profile}_h{horizon}_{t}.json').read_text())
            vals=[f"{100*d['metrics'][name]['relative_improvement']:.2f}%" for name,_ in model_names]
            lines.append('| '+scale[profile]+' H'+str(horizon)+' | '+str(d['validation_origins'])+' | '+' | '.join(vals)+' |')
        d=annual_result[target]
        vals=[f"{100*d['metrics'][name]['relative_improvement']:.2f}%" for name,_ in model_names]
        lines.append('| Annual H12 (short history) | '+str(d['validation_origins'])+' | '+' | '.join(vals)+' |')
        lines.append('')
    lines += ['## Additional minute tasks: first-fold snapshot','',
              'The following tasks have completed their first full-origin development fold. Later folds are not included in this table; none of these rows establishes a stable or independent advantage.','',
              '| Task | Origins | '+' | '.join(label for _,label in model_names)+' |',
              '| --- | ---: | '+' | '.join(['---:']*len(model_names))+' |']
    for item in first_folds['tasks']:
        target='price' if item['target']=='future_close' else 'return'
        vals=[f"{100*item['relative_improvement'][name]:.2f}%" for name,_ in model_names]
        lines.append(f"| Minute H{item['horizon']} {target} | {item['validation_origins']} | "+' | '.join(vals)+' |')
    lines += ['',f'## Minute H60: {len(completed_folds)} of 5 full-origin folds','',
              f'This snapshot pools folds {", ".join(map(str,completed_folds))} across {minute_aggregate["validation_origins"]:,} validation origins. The first fold has 64,261 fitting, 11,370 stopping, and 75,900 validation origins. '+
              (f'Folds {", ".join(map(str,missing_folds))} are not included in this snapshot. ' if missing_folds else '')+
              'The earlier 1/30-origin pilot is not treated as a complete fold.','',
              '| Method | Relative endpoint MSE improvement |','| --- | ---: |']
    for name,label in [('last_close','Zero return')]+model_names:
        data=minute_aggregate['models'][name]
        lines.append('| '+label+' | '+f"{100*data['relative_improvement']:.2f}%"+' |')
    lines += ['', 'The pooled result is based on exposed development periods and overlapping paths; it does not establish an independent or stable improvement.','',
              'The annual short-history task has 47 overlapping validation origins across its five folds. Its count is not 47 independent annual observations. '+
              f"For annual returns, the GRU’s relative improvement is {100*annual_gru_by_fold[1]:.2f}% in fold 1 and {100*annual_gru_by_fold[2]:.2f}% in fold 2, illustrating variation across periods. "+
              'Nominal-price targets may be affected by stock splits, and the return target excludes cash dividends.','',
              '## Split sensitivity and label definition','',
              'The planned targets use nominal prices. Twenty-one overlapping monthly H21 origins cross the 2020 split, and they account for 74.01% of the zero-return baseline’s total squared error and 76.91% of the persistence-price baseline’s total squared error. These are overlapping windows around one event, not 21 independent events.','',
              'A subsequent label-only diagnostic adjusts realized log returns for the split while leaving the zero-return prediction unchanged. Under that different target, the same windows account for 1.58% of baseline squared error. This is not a model improvement or a retrained comparison. The adjusted price return excludes cash dividends and is not total return. Future corporate actions were not supplied as predictive features.','',
              'After the original results were inspected, all eight methods were also retrained on the same five monthly folds using a split-neutral price-return target. The past log-relative histories matched the original split-aware histories, while the price anchor and target changed. '+
              f"The pooled GRU endpoint MSE was {100*split_result['models']['gru']['relative_improvement']:.2f}% below the zero-return baseline, "+
              'but one fold was worse than baseline; descriptive paired-loss intervals using 21- and 42-session blocks both included zero. This post-hoc first-seed result is not independent confirmation.','',
              '| Method | Relative endpoint MSE improvement |','| --- | ---: |']
    for name,label in [('last_close','Zero return')]+model_names:
        lines.append('| '+label+' | '+f"{100*split_result['models'][name]['relative_improvement']:.2f}%"+' |')
    lines += ['', 'Two additional seeds were fixed after inspecting the first result and rerun on the same folds. All three seeds reuse the same exposed market periods. The GRU results change sign across seeds, so they do not establish a stable advantage.','',
              '| Method | Seed 2026 | Seed 2027 | Seed 2028 |','| --- | ---: | ---: | ---: |']
    for name,label in [('mlp','MLP'),('gru','GRU'),('lstm','LSTM'),('xpatch','xPatch')]:
        by_seed=seed_comparison['relative_mse_improvement_vs_zero_return'][name]
        lines.append('| '+label+' | '+' | '.join(f"{100*by_seed[str(seed)]:.2f}%" for seed in (2026,2027,2028))+' |')
    lines += ['',
              '## Post-hoc price anchoring','',
              'After inspecting the original price results, a separate matched comparison was run on the same 25 price folds. Each neural method starts from the current price and adds a trainable, zero-initialized residual head; the unchanged persistence forecast is eligible at epoch zero for inner stopping. The additional head and stopping option change the training recipe and capacity, so the comparison does not isolate a single causal architectural effect.','',
              '| Task | Origins | MLP | GRU | LSTM | xPatch |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for prof,h in [('hourly',8),('hourly',1),('hourly',4),('monthly',21),('quarterly',3)]:
        d=json.loads((anchor/f'PRIVATE_TASK_{prof}_h{h}.json').read_text())
        vals=[f"{100*d['models'][name]['anchored_improvement_vs_persistence']:.2f}%" for name in ('mlp','gru','lstm','xpatch')]
        lines.append('| '+scale[prof]+' H'+str(h)+' | '+str(d['validation_origins'])+' | '+' | '.join(vals)+' |')
    lines += ['', 'Across the 100 model-fold fits, inner stopping selected the epoch-zero persistence forecast 40 times. Any apparent quarterly advantage is based on only 52 overlapping validation origins; monthly pooled errors are strongly affected by the split. The present evidence is insufficient to establish an independent or stable improvement.','',
              '## Outstanding work','',
              '- Confirm account-specific data and publication rights and assess whether the short-history annual design is adequate for the intended venue.',
              '- Finish the remaining minute folds and horizons under an agreed resource budget; assess convergence and additional seeds.',
              '- Freeze a data and evaluation protocol, state the split treatment, and establish any genuinely independent confirmation before submission.',
              '- Verify the target venue, references, authorship, data-availability statement, and permission to share results or weights.','',
              'Local audit evidence includes the saved-fold checks, the annual array re-aggregation, and the partial minute aggregate. These checks do not establish licensing or independent significance.','']
    directory=OUT/'private_manuscript_appendices';directory.mkdir(parents=True,exist_ok=True)
    path=directory/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_PRIVATE_PRELIMINARY_RESULTS_EN.md')
    with path.open('x',encoding='utf8') as f:f.write('\n'.join(lines))
    print(json.dumps({'status':'private_english_appendix_created_not_for_submission','path':str(path),'sha256':file_digest(path)},ensure_ascii=False))


if __name__=='__main__':main()
