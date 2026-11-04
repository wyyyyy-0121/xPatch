# 旧收益实验定向审计

核查日期：2026-11-04。范围是只读查看配置、实现、已保存报告与汇总，不是重新训练或逐个重放260个运行。本轮没有重新验证全部原始响应/权重，也不把旧报告中的完整验收次数称作本轮已复核次数。

## 核实内容

| 项目 | 本轮直接读取的依据 | 已核实事实与含义 |
|---|---|---|
| 任务 | `/home/ayaka/xPatch/configs/aapl_v2.json`；`finance_v2/data.py` | 预测未来1/5/12分钟累计log收益；主要端点12分钟MSE，240个同日历史收益，1分钟origin步长 |
| 数据身份 | 配置、`data_acquisition/V2_SOURCE_AUDIT.md`、`finance_v2/provenance.py` | SIP/raw，bar-start，CSV登记SHA256 `dafb2d2ac05ae7652fea0334caea209457db6418368efdcad8b7d648cdfb5830`；原始响应逐字段验证的代码存在，但本轮未运行它 |
| 信息 | 配置、`finance_v2/models.py` | close与OHLCV面板；xPatch逐通道独立，不能把OHLCV输入当融合证据；Chronos cross_learning=false；Kronos close-only用重复close与零量等合成输入，OHLCV的amount是代理，均需披露 |
| 时间与切分 | `finance_v2/data.py`、`finance_v2/RESULTS.md` | 2021起扩展训练，2023Q1–2025Q2十季度；训练内20日epoch选择、20日校准和会话purge；同日连续窗口避免隔夜目标 |
| 样本范围 | `finance_v2/results_summary.json` | 每面板85,422起点、619有效交易日；bar-start13:30–15:47 ET，柱末约13:31–15:48 ET；不能推广成全天 |
| xPatch训练 | `finance_v2/models.py`、`training.py` | 原架构48/240两窗口；patch12/stride6、EMA alpha0.1、RevIN关闭；Adam lr0.001、batch128、最多30轮/patience5。训练用逐分钟MSE，epoch选择用12分钟累计MSE |
| 其余监督训练 | 配置、`training.py` | Ridge alpha1；LightGBM leaves31/lr0.05/300树等固定；这些是训练/epoch选择，不是广泛HPO |
| 预训练模型 | `models.py`、配置与model_lock；已安装API源码 | Kronos-small固定temperature1/top_p0.9/sample_count1，三个种子各一条路径；Chronos2固定上下文240、minute-return、冻结权重。其代码变量/API字段虽名为mean，当前2.3.2实现实际取0.5分位数，旧适配器求的是逐步中位数之和；不等同累计收益中位数或条件均值。没有LoRA或预测器微调证据 |
| 新模型身份 | `/home/ayaka/xPatch/configs/aapl_v2_model_lock.json` | Kronos-small与tokenizer、Chronos2登记固定revision/文件哈希；可只读引用，不复制权重；新实验仍需自行登记实际读取身份 |
| 旧结果 | 汇总与`RESULTS.md` | no_candidate_passed；所有候选合并12分钟MSE高于零；历史确认、交易成本研究与前瞻观察按旧门槛停止 |
| 研究曝光 | `research/AAPL_V2_AMENDMENTS.md` | 先前47个完整结果及1个中断目录保留；协议修订后的260运行也属于已看历史。不能重称盲测 |

旧主签名：`7f6cbded596aee585a8b65e4dad383195a8511bf8f69428670489f8d5df8f192`。修订前签名：`9fcaa4347dac53b3d115e58b0008796e5cff7739aae9937afd278de5f8ee33ad`。

## 结果核对摘录

下表来自直接读取的旧汇总/报告；正数表示比零收益基线的MSE更高。这不是新训练结果。

| 候选 | close误差增加 | OHLCV误差增加 |
|---|---:|---:|
| Ridge | 0.477% | 2.186% |
| LightGBM | 0.809% | 0.939% |
| xPatch48 | 0.351% | 0.421% |
| xPatch240 | 9.274% | 0.318% |
| Kronos-small | 58.621% | 57.544% |
| Chronos-2 | 21.600% | 24.906% |

零收益MSE为 `3.6245731809803527e-6`。不能据此断言认真HPO或实际微调后仍必然失败，也不能忽略阴性结果只强调某季度表现。

## 可以复用与必须重新完成

可复用的是交易所日历/同日连续窗口的设计、原始响应receipt与数据身份、既有权重身份、预测工件及失败记录、残差校准的训练内隔离思路。复用设计不代表直接import旧runner；其中 `ROOT` 和若干默认写路径指向旧仓库，验证函数还可能默认 `save=True` 回写，不能在只读要求下无检查地调用。

必须重新完成的是：新研究身份、目标对齐的损失对照、真正的超参搜索、Chronos实际微调、Kronos多路径与预测器微调的可行性、同等origin/时间信息条件、内外层隔离、当前端到端性能准入。旧代码已修改上游EMA和网络控制，需对照官方源码，不能把整个旧仓库称作未经修改的原版。

RV30是另一目标的历史证据；本轮不读取其当前观察累计成绩。交接记录中的RV30调参计数及结果，不作为收益任务已充分调参的证据。旧文稿和交付索引不在本轮修订范围内。
