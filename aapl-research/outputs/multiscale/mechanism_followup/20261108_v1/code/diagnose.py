from common import *
from datetime import datetime,timezone
import csv

def main():
    data=read(BASE/'protocol/DATA.json');cache=cache_arrays();bars,_,folds,_=bars_and_folds()
    models=['mlp','gru','lstm','xpatch'];pooled_p=[];pooled_y=[];out=[];histories=[]
    for f in folds:
        k=str(f['fold']);row=data['folds'][k];v=f['validation'];rr=rows_for(v)
        y=cache['actual'][rr,-1];threshold=float(np.median(cache['volatility'][rows_for(f['inner']['fit'])]))
        states=np.asarray(cache['volatility'][rr])>threshold;values=[]
        for m in models:
            record=row['models'][m];assert sha(record['prediction'])==record['prediction_sha256']
            with np.load(record['prediction'],allow_pickle=False) as a:
                assert np.array_equal(a['origins'],v);p=a['prediction'][:,-1].copy()
            values.append(p)
            history=read(record['summary']);histories.append({'fold':int(k),'model':m,**history})
            for label,mask in [('all',np.ones(len(v),dtype=bool)),('low_volatility',~states),('high_volatility',states)]:
                yy,pp=y[mask],p[mask];den=float(np.mean(yy**2));mse=float(np.mean((pp-yy)**2))
                out.append({'fold':int(k),'model':m,'state':label,'origins':len(yy),'mse':mse,
                    'baseline_mse':den,'improvement_pct':100*(1-mse/den),
                    'energy_ratio':float(np.mean(pp**2)/den),'alignment_ratio':float(2*np.mean(pp*yy)/den),
                    'prediction_bias':float(pp.mean()),'actual_mean':float(yy.mean()),
                    'prediction_actual_correlation':float(np.corrcoef(pp,yy)[0,1]),'threshold':threshold})
        pooled_p.append(np.column_stack(values));pooled_y.append(y)
    p=np.concatenate(pooled_p);y=np.concatenate(pooled_y);den=float(np.mean(y**2))
    result={'status':'four_model_diagnostics_completed','at_utc':datetime.now(timezone.utc).isoformat(),
        'origins':len(y),'models':models,'fold_state_metrics':out,'training_histories':histories,
        'forecast_correlations':np.corrcoef(p,rowvar=False).tolist(),
        'pooled':{m:{'improvement_pct':100*(1-np.mean((p[:,i]-y)**2)/den),
            'energy_ratio':float(np.mean(p[:,i]**2)/den),'alignment_ratio':float(2*np.mean(p[:,i]*y)/den),
            'correlation':float(np.corrcoef(p[:,i],y)[0,1])} for i,m in enumerate(models)},
        'equal_mean_improvement_pct':100*(1-float(np.mean((p.mean(1)-y)**2))/den),
        'Q3_values_read':False,'new_fits':0,'independent_confirmation':False}
    atomic(BASE/'diagnostics/DIAGNOSTICS.json',result)
    with (BASE/'diagnostics/FOUR_MODEL_METRICS.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(out[0]));writer.writeheader();writer.writerows(out)
    lines=['# 四模型诊断（已曝光历史开发数据）','',f'共同预测起点：{len(y):,}；五折；目标为 H60 累计对数收益。','',
        '| 模型 | 相对零基准 MSE 改善 | 预测能量 / 零基准误差 | 对应抵消项 / 零基准误差 | 预测与收益相关 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for m,r in result['pooled'].items():lines.append(f"| {m} | {r['improvement_pct']:.4f}% | {r['energy_ratio']:.6f} | {r['alignment_ratio']:.6f} | {r['correlation']:.5f} |")
    lines+=['',f"四模型等权平均的 MSE 改善为 {result['equal_mean_improvement_pct']:.4f}%。该组合未训练，也未使用独立确认资料。",'',
        '误差分解精确成立，但不能独立归因于某个模块。模型不同或预测相关性低都不保证融合增益。',
        '这 20 次 H60 原拟合均未触及 20 轮上限；训练路径损失与停止终点误差不是同一指标，不能直接将二者差异解释为过拟合。','',
        '待检验解释：MLP 价格锚点；GRU/LSTM 最后状态读出；xPatch 锚点与双分支；共同的预测幅度/校准问题。',
        '高低波动阈值分别只由每折内层拟合段确定；所有条件分析仍为开发诊断。']
    (BASE/'diagnostics/DIAGNOSTIC_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':result['status'],'equal_mean_improvement_pct':result['equal_mean_improvement_pct']}))

if __name__=='__main__':main()
