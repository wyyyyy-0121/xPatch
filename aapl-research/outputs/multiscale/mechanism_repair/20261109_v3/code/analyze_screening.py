"""Evaluate every registered screening contrast, including failed and degenerate ones."""
from study import *
import csv, torch

def same_parameters(a,b):
    return a.keys()==b.keys() and all(torch.equal(a[n],b[n]) for n in a)

def main():
    assert read(BASE/'audit/SCREENING_COMPLETE.json')['jobs']==52
    jobs=read(BASE/'protocol/JOBS.json')['jobs'];rows=[];done={}
    for cfg in jobs:
        out=BASE/'runs'/cfg['id'];r=read(out/'DONE.json');assert r['status']=='completed'
        assert sha(out/'predictions.npz')==r['prediction_sha256']
        assert sha(out/'best_trained.pt')==r['checkpoint_sha256']
        done[cfg['id']]=r
        for mode,m in r['metrics'].items():
            rows.append({'task':task_key(cfg['profile'],cfg['horizon'],cfg['target']),
                'model':cfg['model'],'variant':cfg['variant'],'fold':cfg['fold'],'seed':cfg['seed'],
                'mode':mode,**m,'epochs':r['epochs'],'best_epoch':r['trained_best_epoch'],
                'selected_baseline':r['selected_baseline'],'shadow_stop_epoch':r['shadow_stop_epoch'],
                'shadow_best_epoch':r['shadow_best_epoch']})
    groups={}
    for row in rows:groups.setdefault((row['task'],row['model']),[]).append(row)
    contrasts=[];replication=[]
    for (task,model),data in sorted(groups.items()):
        pairs=[('long_aligned','coupled_policy')] if model=='tiny' else (
            [('noanchor_random','anchor_random'),('noanchor_baseline','anchor_baseline'),
             ('anchor_baseline','anchor_random'),('noanchor_baseline','noanchor_random')]
            if task.endswith('future_log_return') else
            [('residual_random','price_random'),('residual_baseline','price_baseline'),
             ('price_baseline','price_random'),('residual_baseline','residual_random')])
        advance=False;reasons=[]
        for primary,reference in pairs:
            modes=('selected',) if model=='tiny' else ('trained','selected','coupled')
            for mode in modes:
                folded=[]
                for fn in (1,2):
                    a=next(r for r in data if r['variant']==primary and r['mode']==mode and r['fold']==fn)
                    b=next(r for r in data if r['fold']==fn and
                        (r['mode']=='coupled' if model=='tiny' else r['variant']==reference and r['mode']==mode))
                    folded.append({'fold':fn,'primary_mse':a['endpoint_mse'],'reference_mse':b['endpoint_mse'],
                        'improvement_over_reference_pct':100*(1-a['endpoint_mse']/b['endpoint_mse']),
                        'improvement_over_zero_or_persistence_pct':a['improvement_pct']})
                consistent=all(r['primary_mse']<r['reference_mse'] for r in folded)
                contrasts.append({'task':task,'model':model,'primary':primary,'reference':reference,
                    'mode':mode,'folds':folded,'two_fold_direction_consistent':consistent,
                    'predictive_gain_both_folds':all(r['improvement_over_zero_or_persistence_pct']>0 for r in folded),
                    'scope':'exposed development screening only'})
                # Only one registered training/selection mode governs advancement. Coupled-only positives do not.
                if consistent and mode==('selected' if model=='tiny' else 'trained'):
                    advance=True;reasons.append(primary+' vs '+reference)
        replication.append({'task':task,'model':model,'mechanism_replicate':advance,'reasons':reasons,
                            'full_four_cell_controls_required':model!='tiny'})
    # Check actual original-recipe restoration before treating new diagnostics as matched interventions.
    restoration=[];prefix=[]
    for cfg in jobs:
        r=done[cfg['id']];out=BASE/'runs'/cfg['id'];task=task_key(cfg['profile'],cfg['horizon'],cfg['target'])
        meta=read(PRIOR/'cache'/task/'TASK.json')['folds'][str(cfg['fold'])]
        if cfg['variant'] in ('anchor_random','price_random'):
            original=meta['models'][cfg['model']]
            assert sha(original['checkpoint'])==original['checkpoint_sha256']
            a=torch.load(out/'best_trained.pt',map_location='cpu',weights_only=True)
            b=torch.load(original['checkpoint'],map_location='cpu',weights_only=True)
            previous=read(original['summary'])
            restoration.append({'job':cfg['id'],'model_state_exact':same_parameters(a['model'],b['model']),
                'best_epoch_matches':r['trained_best_epoch']==previous['best_epoch'],
                'epochs_matches':r['epochs']==previous['epochs']})
        if cfg['model']=='tiny':
            oldjob=next(j for j in read(PRIOR/'protocol/SELECTED_JOBS.json')['jobs'] if
                j['profile']==cfg['profile'] and j['horizon']==cfg['horizon'] and j['target']==cfg['target'] and
                j['fold']==cfg['fold'] and j['seed']==cfg['seed'] and j['family']=='tiny' and
                j['field']=='long' and j['loss']=='aligned')
            old=read(PRIOR/'runs'/oldjob['id']/'DONE.json')
            a=torch.load(out/('initial.pt' if r['shadow_best_epoch']==0 else 'coupled_best.pt'),map_location='cpu',weights_only=True)
            b=torch.load(PRIOR/'runs'/oldjob['id']/'best.pt',map_location='cpu',weights_only=True)
            prefix.append({'job':cfg['id'],'prefix_selected_model_exact':same_parameters(a['model'],b['model']),
                'prefix_best_epoch_matches':r['shadow_best_epoch']==old['best_epoch'],
                'prefix_stop_epoch_matches':r['shadow_stop_epoch']==old['epochs'],
                'trained_only_epochs':r['epochs'],'old_coupled_epochs':old['epochs']})
    verified=all(all(v for k,v in x.items() if k.endswith(('_exact','_matches'))) for x in restoration+prefix)
    receipt={'status':'matched_screening_complete' if verified else 'restoration_discrepancy_requires_review',
        'at_utc':now(),'new_fits':len(jobs),'rows':rows,'contrasts':contrasts,
        'replication_groups':replication,'original_restoration':restoration,'tiny_prefix_checks':prefix,
        'matched_interventions_verified':verified,'Q3_values_read':False,'independent_confirmation':False,
        'method_success_claimed':False,'sum_recorded_fit_cpu_seconds':sum(r['training_cpu_seconds'] for r in done.values())}
    atomic(BASE/'analysis/SCREENING_RESULTS.json',receipt)
    with (BASE/'analysis/SCREENING_ROWS.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# 52 格配对筛查结果','',
      '全部预定格均保留；以下是已曝光历史时期的一种子两折筛查，不能称为独立确认。', '',
      '| 任务 / 模型 / 配方 | 折 | 已训练最佳改善 | 内层可选基准后改善 | 已训练停止轮 / 基准前缀停止轮 |',
      '| --- | ---: | ---: | ---: | --- |']
    for cfg in jobs:
        r=done[cfg['id']]
        lines.append(f"| {task_key(cfg['profile'],cfg['horizon'],cfg['target'])} / {cfg['model']} / {cfg['variant']} | {cfg['fold']} | {r['metrics']['trained']['improvement_pct']:+.4f}% | {r['metrics']['selected']['improvement_pct']:+.4f}% | {r['epochs']} / {r['shadow_stop_epoch']} |")
    lines+=['','## 可复核因素','', '| 任务 / 模型 | 两折方向一致的预定对比 |', '| --- | --- |']
    for r in replication:lines.append(f"| {r['task']} / {r['model']} | {', '.join(r['reasons']) or '无'} |")
    lines+=['',f'原配方与小模型前缀的匹配核对：{verified}。',
            '进入复核的因素只说明值得检验；超过弱原模型和超过零/持续性是不同要求。Goal 仍为 active，不能在此宣布方法成功。']
    (BASE/'analysis/SCREENING_REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':receipt['status'],'new_fits':len(jobs),'groups':replication,'matched':verified}),flush=True)
    if not verified:raise SystemExit(2)

if __name__=='__main__':main()
