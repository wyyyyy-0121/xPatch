"""Inventory real comparison differences without fitting or protected-data access."""
from study import *
import csv

def main():
    old=read(PRIOR/'protocol/DATA.json'); rows=[]
    assert len(old['task_files'])==20
    for task,digest in sorted(old['task_files'].items()):
        path=PRIOR/'cache'/task/'TASK.json';assert sha(path)==digest
        meta=read(path)
        for fn,fold in meta['folds'].items():
            assert sha(fold['scaler_file'])==fold['scaler_sha256']
            s=read(fold['scaler_file'])
            assert s['split_sha256']==fold['split_sha256']
            rows.append({'task':task,'fold':int(fn),'target':meta['target'],
                'counts':fold['counts'],'input_same_price_information':True,
                'original_input':'log(P_past_split_known/P_origin), scalar fit-only scaling plus standardized price anchor',
                'tiny_input':'same historical log closes, window demeaning then scalar fit-only scaling, no price anchor',
                'original_target':'raw nominal level' if meta['target']=='future_close' else 'cumulative log return',
                'tiny_target':'nominal dollar residual from origin' if meta['target']=='future_close' else 'cumulative log return',
                'endpoint_original_target_std':s['original']['y_std'][-1],
                'endpoint_tiny_target_std':s['tiny']['y_std'][-1],
                'std_ratio_original_over_tiny':s['original']['y_std'][-1]/s['tiny']['y_std'][-1],
                'original_initialization':'random head; no guaranteed persistence function',
                'tiny_initialization':'zero residual head; initial persistence function',
                'original_stopping':'patience against trained epoch best; no epoch-zero candidate',
                'tiny_stopping':'patience against best including epoch-zero; baseline competes with training',
                'split_hashes':fold['split_sha256']})
    original=list(csv.DictReader((ROOT/'outputs/multiscale/completed_core_evidence/20261108_parallel8_continuation/results/TRAINING_MODELS.csv').open()))
    tiny=[]
    for cfg in read(PRIOR/'protocol/SELECTED_JOBS.json')['jobs']:
        done=read(PRIOR/'runs'/cfg['id']/'DONE.json')
        if cfg['family']=='tiny':tiny.append(done)
    assert len(original)==400 and len(tiny)==390
    summary={'status':'comparison_differences_verified','at_utc':now(),'task_folds':len(rows),
        'tasks':20,'original_fits':400,'original_epoch_cap':sum(r['epoch_cap_reached']=='True' for r in original),
        'tiny_fits':390,'tiny_epoch_zero_selected':sum(r['best_epoch']==0 for r in tiny),
        'tiny_training_ends_at_three_epochs':sum(r['epochs']==3 for r in tiny),'rows':rows,
        'key_hypotheses':[
            'With identical architecture, does removing the price anchor alter error?',
            'Does baseline-function initialization explain apparent improvements over random original heads?',
            'At matched baseline initial functions, does raw-level versus dollar-residual scaling alter price learning?',
            'Does baseline-coupled patience terminate useful optimization before a trained-only policy?'],
        'claims_not_supported_yet':['No structural causal explanation has been established by this inventory.',
            'No new prediction gain or independent validation follows from source differences.'],
        'Q3_values_read':False,'new_market_fits':0,'previous_data_manifest_sha256':sha(PRIOR/'protocol/DATA.json')}
    atomic(BASE/'analysis/COMPARISON_AUDIT.json',summary)
    lines=['# 新旧配方差异审计','',
      '覆盖二十任务、一百个任务折。原四百次拟合和上一轮三百九十个小模型拟合均读取其保存记录，未启动训练。','',
      f"原拟合达到二十轮上限：{summary['original_epoch_cap']}/400；小模型选择初始基准：{summary['tiny_epoch_zero_selected']}/390；小模型训练恰好三轮终止：{summary['tiny_training_ends_at_three_epochs']}/390。",'',
      '价格锚点、初始函数、目标尺度和基准是否参与 patience 是同时存在的区别。小模型的改善不能只归于感受野或网络规模。', '',
      '| 价格任务 | 折 | 原目标标准差 / 残差标准差 | 拟合/停止/评价起点 |', '| --- | --- | ---: | --- |']
    for r in rows:
        if r['target']=='future_close':lines.append(f"| {r['task']} | {r['fold']} | {r['std_ratio_original_over_tiny']:.3f} | {r['counts']['fit']}/{r['counts']['stop']}/{r['counts']['validation']} |")
    lines+=['','以上是实现差异和量纲诊断，不能独自证明哪项造成预测改善。下一阶段按 PLAN_ZH.md 的 52 格拆开检验。']
    (BASE/'analysis/COMPARISON_AUDIT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('status','tasks','task_folds','original_epoch_cap','tiny_epoch_zero_selected','tiny_training_ends_at_three_epochs')}),flush=True)

if __name__=='__main__':main()
