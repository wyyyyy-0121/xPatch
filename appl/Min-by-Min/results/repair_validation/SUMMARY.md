# 修复后验证与交付状态

这些实验使用截至 2022-06-17 的过期、未独立验证 AAPL 镜像，只能作为工程回归，不能代表当前预测能力。
本次没有运行新的最终测试。旧测试时期已登记为暴露。

| 时间尺度 | 状态 | 候选配置数 | 通过门槛数 | 选择 |
|---|---|---:|---:|---|
| annual | validation_complete | 64 | 0 | last_close |
| quarterly | validation_complete | 64 | 0 | last_close |
| monthly | validation_complete | 64 | 0 | last_close |
| hourly | insufficient_data | 0 | 0 | 无排名 |
| minute | insufficient_data | 0 | 0 | 无排名 |

候选数包括 2 种目标 × 4 个窗口 × 8 个模型（长期各 64 个）。全部使用五折、共同 origins 与共同 MASE 分母。
每个长期报告包含完整窗口／模型比较；`regime_summary.csv` 给出各折实际涨跌与波动，供诊断市场状态变化。
价格图和 CSV 是所选模型最终预测步的抽样展示；误差指标使用全部有效验证样本。
小时／分钟数据未满足完整的数据与来源门槛，没有生成预测排名；具体失败项见各自报告。
用户 ZIP 全量审计为线性插值，拒绝用作真实分钟数据。详情见 ../../data/audits/ARCHIVE_AUDIT.md。
工程测试、GPU 训练和模型文件加载通过，均不等于模型达到预测效果门槛。
