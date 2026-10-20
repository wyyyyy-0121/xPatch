"""Summarize completed validation reports and derive fold market-regime context."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from data_quality import aggregate_bars, load_ohlcv
from experiment import sequence_ids


def summarize(root):
    root=Path(root)
    here=Path(__file__).resolve().parent
    raw,_=load_ohlcv(here/'data'/'AAPL_1d.csv')
    lines=['# 修复后验证与交付状态','',
           '这些实验使用截至 2022-06-17 的过期、未独立验证 AAPL 镜像，只能作为工程回归，不能代表当前预测能力。',
           '本次没有运行新的最终测试。旧测试时期已登记为暴露。','',
           '| 时间尺度 | 状态 | 候选配置数 | 通过门槛数 | 选择 |',
           '|---|---|---:|---:|---|']
    for name in ('annual','quarterly','monthly','hourly','minute'):
        path=root/name/'report.json'
        if not path.exists():
            raise FileNotFoundError(path)
        result=json.loads(path.read_text())
        rows=result['rankings']
        choices=', '.join(sorted({r['model'] for r in result['selections']})) or '无排名'
        lines.append(f"| {name} | {result['status']} | {len(rows)} | {sum(r['accepted'] for r in rows)} | {choices} |")
        if not rows:
            continue
        bars=aggregate_bars(raw,result['bar_interval'])
        market=[]; seen=set()
        for row in rows:
            for fold in row['folds']:
                key=(row['horizon'],fold['fold'],fold['validation_start'],fold['validation_end'])
                if key in seen: continue
                seen.add(key)
                segment=bars.loc[pd.Timestamp(key[2]):pd.Timestamp(key[3])]
                ids=sequence_ids(segment,result['bar_interval'])
                returns=np.diff(np.log(segment.close.to_numpy()))[np.diff(ids)==1]
                change=float(segment.close.iloc[-1]/segment.close.iloc[0]-1)
                market.append(dict(horizon=key[0],fold=key[1],start=key[2],end=key[3],
                                   realized_return=change,realized_volatility=float(np.std(returns)),
                                   phase='up' if change>.02 else ('down' if change<-.02 else 'flat')))
        pd.DataFrame(market).to_csv(root/name/'regime_summary.csv',index=False)
    lines += ['', '候选数包括 2 种目标 × 4 个窗口 × 8 个模型（长期各 64 个）。全部使用五折、共同 origins 与共同 MASE 分母。',
              '每个长期报告包含完整窗口／模型比较；`regime_summary.csv` 给出各折实际涨跌与波动，供诊断市场状态变化。',
              '价格图和 CSV 是所选模型最终预测步的抽样展示；误差指标使用全部有效验证样本。',
              '小时／分钟数据未满足完整的数据与来源门槛，没有生成预测排名；具体失败项见各自报告。',
              '用户 ZIP 全量审计为线性插值，拒绝用作真实分钟数据。详情见 ../../data/audits/ARCHIVE_AUDIT.md。',
              '工程测试、GPU 训练和模型文件加载通过，均不等于模型达到预测效果门槛。', '']
    (root/'SUMMARY.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path(__file__).parent/'results'/'repair_validation')
    summarize(parser.parse_args().results)
