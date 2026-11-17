from common import *
import torch
from multiscale.features import CloseWindows,Scaler
from multiscale.models import make_model

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    assert len(os.sched_getaffinity(0))<=2
    rows=[];sensitivity=[];combinations=[]
    for task in sorted(read(BASE/'protocol/DATA.json')['task_files']):
        meta=read(BASE/'cache'/task/'TASK.json');profile=meta['profile'];h=meta['horizon'];target=meta['target']
        bars=admitted_bars(meta['frequency']);windows=CloseWindows(bars,meta['lookback'],read(SOURCE/'admission.json')['split_actions'])
        paths={m:[] for m in ('mlp','gru','lstm','xpatch')};truth=[]
        for fn,fold in meta['folds'].items():
            with np.load(fold['targets']) as a:y=a['actual'];anchor=a['anchor'];origins=a['origins']
            yy=y if target=='future_log_return' else y-anchor[:,None]
            baseline=float(np.mean(yy[:,-1]**2));truth.append(yy[:,-1])
            for model,rec in fold['models'].items():
                assert sha(rec['prediction'])==rec['prediction_sha256']
                with np.load(rec['prediction']) as a:
                    assert np.array_equal(origins,a['origins']);p=a['prediction']
                pp=p if target=='future_log_return' else p-anchor[:,None]
                energy=float(np.mean(pp[:,-1]**2));alignment=float(2*np.mean(pp[:,-1]*yy[:,-1]))
                summary=read(rec['summary']);mse=float(np.mean((p[:,-1]-y[:,-1])**2))
                assert np.isclose(mse,baseline+energy-alignment,rtol=1e-9,atol=1e-12)
                corr=float(np.corrcoef(pp[:,-1],yy[:,-1])[0,1]) if pp[:,-1].std()>0 and yy[:,-1].std()>0 else None
                rows.append({'task':task,'profile':profile,'target':target,'horizon':h,'fold':int(fn),'model':model,
                    'origins':len(y),'baseline_mse':baseline,'endpoint_mse':mse,
                    'relative_improvement_pct':100*(1-mse/baseline),'prediction_energy_over_baseline':energy/baseline,
                    'alignment_offset_over_baseline':alignment/baseline,'mean_prediction_change':float(pp[:,-1].mean()),
                    'mean_realized_change':float(yy[:,-1].mean()),'mean_error':float((p[:,-1]-y[:,-1]).mean()),
                    'prediction_truth_correlation':corr,'per_step_mse':np.mean((p-y)**2,axis=0).tolist(),
                    'epochs':summary['epochs'],'best_epoch':summary['best_epoch'],'epoch_cap_reached':summary['epochs']==20,
                    'training_objective':'standardized path MSE','stopping_objective':'raw endpoint MSE'})
                paths[model].append(pp[:,-1])
                if fn!='1':continue
                ck=torch.load(rec['checkpoint'],weights_only=True,map_location='cpu');assert ck['identity']==rec['identity']
                net=make_model(model,meta['lookback'],h);net.load_state_dict(ck['model']);net.eval()
                idx=origins[np.linspace(0,len(origins)-1,min(128,len(origins)),dtype=int)]
                x,a=windows.arrays(idx);scaler=Scaler(**ck['scaler']);x,a=scaler.transform_x(x,a)
                xx=torch.from_numpy(x[:,:,None]);aa=torch.from_numpy(a);L=xx.shape[1];cut=max(1,L//4)
                with torch.inference_mode():
                    original=net(xx,aa).numpy();variants={}
                    for label,part in [('recent_quarter',slice(-cut,None)),('oldest_quarter',slice(0,cut))]:
                        changed=xx.clone();changed[:,part]=0
                        variants[label]=net(changed,aa).numpy()
                    variants['anchor_to_fit_mean']=net(xx,torch.zeros_like(aa)).numpy()
                    if model in ('gru','lstm'):
                        hidden=net.rnn(xx)[0].mean(1);variants['mean_readout_no_retraining']=net.head(torch.cat((hidden,aa),1)).numpy()
                    elif model=='xpatch':
                        trend=net.ema(xx);res=xx-trend
                        variants['trend_input_zero']=(net.backbone(res,trend*0)[:,:,0]+net.anchor_head(aa))[:,:h].numpy()
                        variants['residual_input_zero']=(net.backbone(res*0,trend)[:,:,0]+net.anchor_head(aa))[:,:h].numpy()
                for label,v in variants.items():
                    raw_delta=(v[:,-1]-original[:,-1])*scaler.y_std[-1]
                    sensitivity.append({'task':task,'model':model,'fold':1,'origins':len(idx),'intervention':label,
                        'mean_square_output_change_over_baseline':float(np.mean(raw_delta**2)/baseline),
                        'interpretation':'fixed-checkpoint sensitivity; no retraining or causal attribution'})
        names=list(paths);p=np.column_stack([np.concatenate(paths[m]) for m in names]);y=np.concatenate(truth)
        average=p.mean(1);coef=np.linalg.lstsq(p,y,rcond=None)[0];oracle=p@coef
        combinations.append({'task':task,'models':names,'prediction_correlation':np.corrcoef(p,rowvar=False).tolist(),
            'equal_mean_improvement_pct':100*(1-np.mean((average-y)**2)/np.mean(y*y)),
            'posthoc_unconstrained_linear_oracle_improvement_pct':100*(1-np.mean((oracle-y)**2)/np.mean(y*y)),
            'oracle_coefficients':coef.tolist(),'oracle_scope':'fit and scored on same exposed outer labels; optimistic descriptive bound only; never a deployable result'})
        print(json.dumps({'diagnosed':task}),flush=True)
    atomic(BASE/'diagnostics/DIAGNOSTICS_20_TASKS.json',{'rows':rows,'sensitivity':sensitivity,
        'combinations':combinations,'Q3_values_read':False,'independent_confirmation':False})
    for name,data in [('ERROR_MECHANISMS.csv',rows),('INTERNAL_SENSITIVITY.csv',sensitivity)]:
        with (BASE/'diagnostics'/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    (BASE/'diagnostics/HYPOTHESIS_LEDGER_ZH.md').write_text('''# 假设与证据身份

1. 预测能量相对实际对应过高：误差恒等式可以直接核对；不能据此推断经济机制或校准必然有效。
2. 长距离局部关系利用不足：前轮局部感受野七根柱；新实验只比较参数匹配的十七/五百一十一根局部感受野。整体仍包含池化、输入变换和有限窗口，不能等同于彻底隔离长记忆。
3. 路径损失与终点评价不同：实现事实。其性能影响由原四模型损失探针和新小模型二乘二配对检验，不从现有曲线直接认定因果。
4. 专家可互补：低预测相关性不保证有效。等权和事后线性上界仅描述已有输出；有时间顺序的常数/门控融合才能提供可实现开发证据。
5. 长期任务存在少样本、标签重叠和公司行动：保留原标签，沿用既有事件分析，逐任务限定结论。

内部扰动只在预定第一折的最多一百二十八个均匀起点计算，使用原检查点固定预处理。均值读出在此不重训，只是敏感性；上一轮实际重训探针仍是另一个证据。
''',encoding='utf-8')

if __name__=='__main__':main()
