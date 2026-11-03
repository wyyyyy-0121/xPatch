# RV30 最终交付证据核对表

依据 `GOAL.md` 的七项验收要求建立。本表是交付核对入口，不能代替实际工件核验，也不代表最终验收通过。2026-11-03 编写时，正式搜索仍在进行；最终 manifest、两轮正式复核、最终模型和最终验收均未完成。

2026-11-04 更新：120 组、12,200 个正式搜索任务已全部完成。首次开发复核的 42 个 xPatch 数值失败原地保留于 `verify_1/`。加载时恢复训练精度的修订已通过全部 60 个保存模型、1,620,216 个完整预测的逐位检查，以及 134 项工程测试，追加锁见 `rv30_replay_lock.json`。正式第一轮改用该锁指定的 `verify_1_replay_correction/`，第二轮仍为 `verify_2/`；下表的第一轮要求据此核对，完整两轮与最终验收仍须实际完成。

当前研究签名为 `65a4daf1ebacb8a7dedca5e866010bd94cda25acc7124509ec81d1c14dc2fbf8`，运行根目录为 `artifacts/rv30/<签名>/development/`。下表中的“运行目录”均指该目录。原研究源快照提交为 `b73f8993f18f739e81acaeb0092c485f5e9ab28a`；后处理修正与独立统计覆盖在两份追加锁中追溯。2026-11-04 已另以 `rv30_acceleration_lock.json` 和 `rv30_throughput_lock.json` 登记逐位一致的执行加速、恢复和性能证据，原冻结和前驱锁均未改写。

| 验收项 | 当前已具备的证据 | 最终仍须取得或核对的证据 |
|---|---|---|
| 历史封存 | `delivery/history.json` 引用旧 v1/v2 提交、原十二分钟收益阴性结论与逐文件哈希 | 验收时重新核对原文件、历史签名、数据和权重锁；旧 `aapl` 监控保持暂停 |
| 数据权限、来源和审计 | `delivery/data.json`、`data_acquisition/DELIVERY_SOURCE_AUDIT.md`、`delivery/provider_sources.json`，20 日与多年三资产审计 | 重新核对全部输入与缓存指纹；保留九根 SPY 缺柱和非点时历史的限制 |
| 文献与预注册 | `research/RV30_REVIEW.md`、`research/rv30_sources.json`、`RESEARCH_PROTOCOL.md`、`configs/rv30.json`、`configs/rv30_search.json`、`configs/rv30_model_lock.json`、原冻结与依赖锁 | 研究签名、全部研究核心源码与数值依赖应保持一致；不能因结果扩大搜索或降低门槛 |
| 性能与启动门槛 | `delivery/PERFORMANCE.md`、`performance.json`、两份执行追加锁；192 项逐位检查、六组种子续训、最大规模 30 轮、正式源码双向恢复及显存预载回退；六项启动门槛行为检查通过 | 当前测量仅准许既有 `development` 续跑；新增实验须补其具体工作负载及计算计划，不能自动沿用本轮最大规模资格；诊断和中断累计计费，不清零预算 |
| 1. 统一可运行入口 | `python -m finance --help` 覆盖 audit、prepare、train、evaluate、verify、confirm、freeze-model、infer、report、accept、finalize/deliver；边界状态已有工程测试 | 在真实冻结模型上运行 README 示例；JSON 必须包含起点、信息截止、目标窗口、单位、模型与资格状态，成功/不可用/错误的退出码应正确 |
| 2. 搜索和模型证据 | 固定三内折、各模型配置、三种子、xPatch 32→16→8 剪枝、恢复与失败保存已实现；工程烟测已完成 | 全部 120 模型组成功结束，200 个正式外层种子任务覆盖精确；`search_coverage.json` 无遗漏，所有阶段在累计预算内；最终 `delivery/model.json` 指向完整权重或明确基线及预处理/校准工件 |
| 3. 完整结果 | 主指标、辅助校准、分段与固定十项比较实现已冻结 | 运行目录的 `evaluation.json`、`daily_losses.csv` 和全部预测/种子/季度/年度/时段；`delivery/REPORT.md` 与 `results.json` 应如实呈现通过者、未通过者、失败、剪枝和预算停止证据 |
| 4. 工程正确性与恢复 | `delivery/model_checks.json`：原 49 项测试、16 个预 2023 训练/恢复任务、16 个独立回放、32 个原版 xPatch 配置前向/反向；后处理锁引用 85 项测试与另外 8 个预 2023 回放，最新执行追加锁引用 119 项工程检查及实际 22 线程继承证据 | 正式 `verify_1` 和 `verify_2` 的 `summary.json`、逐任务回放及 `statistics.json`；监督模型完整回放，Chronos 固定首尾批次回放且标签/分数全量重算；早期工程/环境验证不能代替正式复核 |
| 独立统计和阶段顺序 | `delivery/rv30_postprocessing_lock.json`、`delivery/rv30_statistics_lock.json` 及纠错记录、原 Git 快照 | 两轮独立重建集成、日等权 QLIKE/log-RV MSE、区间、MZ、HAC、两种 bootstrap、Holm 与稳定性门槛；精确模型/种子/来源/轮次/阶段及完成时间均匹配后才准入确认 |
| 条件历史确认 | 已实现开发两轮复核前置约束和 2025H2—2026H1 固定历史确认 | 无开发通过者时记录明确跳过；有通过者时仅用冻结的最后开发折配置，保留曝光声明，确认任务须再各独立复核两轮 |
| 条件前瞻与用途 | 当前及时 SIP 403、完整订单流不足、GitHub 未授权事项在 `delivery/external_dependencies.json` 登记；不下单 | 有历史确认通过者时仍需冻结合格模型、真实按时接收/发出账本与验证器，并完成至少 60 个合格交易日；缺任何一项都保持未验收。没有通过者则按门槛跳过依赖阶段 |
| 5. 自动验收器 | `finance accept` 会重算指标/门槛、核对签名、完整搜索和复核来源、冻结工件、推理与本地代码提交 | `acceptance.json` 各项通过、失败时非零退出；`artifact_index.json` 覆盖全部任务凭证及其逐文件哈希；`delivery/manifest.json` 索引报告、样例、验收及完整工件链 |
| 6. 可照做的使用文档 | `delivery/README.md` 的环境、密钥、数据/训练恢复、推理与验收说明；完整 68 包锁及带哈希离线恢复包；`environment_restore.json` 证明本机新独立环境离线安装、精确签名、8 个早期 CPU 回放及 PyTorch CPU/Chronos API 导入通过 | 在当前真实交付上执行最终推理与验收命令；异机恢复、在线重下与新环境 GPU 执行未测，不能扩大已验证范围；明确波动率与涨跌、盈利的区别以及数据/权重边界 |
| 7. 最终本地交付 | 主线为 `codex/aapl-delivery`，稳定目录为 `finance/`、`delivery/`，小修复用提交追溯 | 最终结果与文档做本地提交；重新核对 manifest 全部引用与代码提交可追溯；动态监控、生成时间和最终提交不得使已验收指纹失效；无未授权推送 |

最终结论分开记录：工程交付是否完成，以及候选是否通过全部预测门槛。充分搜索后无模型通过时，必须交付可运行 HAR 参考基线及同样完整的阴性证据包，标记 `completed_no_predictive_edge`；基线本身亦须独立完整回放两轮。预算耗尽、漏模型、复核未完成、仅有一个 passed 标志，都不足以支持完整阴性结论。

只有逐项证据核验通过，完整交付与本地提交完成后，才可将 Goal 标为 complete，并停止当前 `rv30` 监控。

## 已通过历史门槛后的状态

开发修正后第一轮和第二轮各完成 200 个模型回放及独立统计，历史确认两轮各完成四个模型。两个 Ridge 候选通过，当前不属于完整阴性交付。`delivery/prospective_lock.json` 冻结两个 Ridge 和各自 HAR 的确认模型，AAPL/Ridge 是原主信息集的默认历史回放候选；其 `prospective_qualified` 和 `active_predictive_edge_established` 均保持 false。

前瞻采集、完整日质量、不可补发的落盘确认、六十日前不披露分数和两轮完整重放已具备工程入口；实际资格必须由真实观察产生。`delivery/readiness.json` 只说明历史研究与前瞻工具就绪，`delivery/manifest.json` 当前若标为等待前瞻，不能当成最终通过。尚需及时 SIP 权限、最早六十个完整合格交易日、两轮真实前瞻验证、最终工具验收和本地交付提交。

前瞻实现通过 156 项工程检查；长历史输入下四模型、346 个起点的完整日 CPU 测量通过，最慢单起点约 0.97 秒。九项真实命令行为包括正常历史回放、非交易时段、模型可用前、实时不可用、六十日前禁止评分、提前验收拒绝及启动门槛，均留有实际退出码与日志。准入接口最新仍为403；它是外部等待，不能因工程就绪将最终验收置为通过。
