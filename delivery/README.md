# AAPL 分钟起点 RV30 工具

当前主线使用稳定入口 `finance/`。`GOAL.md` 是最终验收约定，`delivery/status.json` 记录实际进度；只有 `delivery/manifest.json` 和相应验收均通过时才构成最终交付。基础测试、烟测或训练结束本身都不代表验收完成。

每个起点预测未来三十分钟累计对数收益平方之和（RV），同时输出经验分位区间。它衡量波动程度，不是未来价格、涨跌方向或交易利润预测。历史回放始终标为回放；当前及时 SIP 权限与真实接收时间采集未准入，实时模式会返回不可用状态。

## 环境与输入

在 `/home/ayaka/xPatch` 运行。现有已验证环境为 Python 3.13：`/home/ayaka/.venvs/xpatch-v2/bin/python`。沿用这个环境不改变旧实验工件或创建新的产品大版本。正式环境清单在冻结后生成的 `delivery/rv30_dependencies.json`，包括全部已安装包版本。

Alpaca 凭据保留在 `/home/ayaka/.config/alpaca/credentials.env`，仅数据采集客户端通过环境读取；不把凭据加入命令参数、日志或 Git。已有数据接入、逐页原始记录与特殊日期审计见 `data_acquisition/DELIVERY_SOURCE_AUDIT.md`，统一数据身份见 `delivery/data.json`。

AAPL、SPY、QQQ 使用相同 SIP/raw 分钟柱与交易所日历。缺口保留，不插值；所有面板按相同的过去输入及 AAPL 标签资格配对。CSV、大权重、缓存和完整实验输出保留本地，不提交 Git。历史文件不能证明当时的发布和接收时刻。

## 固定工作流程

先审计与准备输入，再完成首次外折前的工程烟测与签名；只有冻结后才能启动正式搜索。当前计算机可使用以下命令：

```bash
cd /home/ayaka/xPatch
/home/ayaka/.venvs/xpatch-v2/bin/python -m finance audit
/home/ayaka/.venvs/xpatch-v2/bin/python -m finance prepare
/home/ayaka/.venvs/xpatch-v2/bin/python -m finance --help
```

正式运行与恢复均使用同一命令、同一运行名：

```bash
cd /home/ayaka/xPatch
/home/ayaka/.venvs/xpatch-v2/bin/python -m finance deliver --run-name development
```

此命令依次执行固定搜索、十季度外层预测和统计、开发结果两轮独立回放、条件历史确认、确认任务两轮回放及交付验收。没有候选通过时，冻结 HAR 参考基线，并对最终基线再完整回放两轮。任一阶段复核失败都停止其依赖步骤。已完成且签名和逐文件指纹一致的任务会复用；中断的 xPatch 会从相同请求的 optimizer、轮数与随机状态恢复。失败或超时试验保留为失败，不能换参数重试来增加搜索额度。另一个入口正在运行时会拒绝重复启动。

搜索与后处理也可以分开执行：`python -m finance train --run-name development` 仅训练；搜索全部结束后，`python -m finance finalize --run-directory artifacts/rv30/<签名>/development` 按上述顺序完成后续。当前运行采用这种方式，以保留原训练签名，并对后处理的顺序与验收覆盖纠错。原始研究冻结文件保持原样，纠错见 `delivery/amendments/postprocessing_correction.json`，独立纠错签名见 `delivery/rv30_postprocessing_lock.json`；它不能豁免数据、特征、模型拟合、搜索或门槛的修改。

输出在 `artifacts/rv30/<协议签名>/development/`。每个任务保存请求、资源、参数、拟合状态、校准、模型、预测及指纹；各模型组保存搜索阶段和内层选择。`progress.json`、`scheduler.json` 用于查看进度；`evaluation.json` 和 `daily_losses.csv` 在全部模型组结束后生成。正式搜索期间不根据已生成的外层分数改变空间。

所有实验进程共享最多 22 个逻辑 CPU：三个 CPU 工作进程各六个、一个 GPU 工作进程四个，BLAS 每进程一个，LightGBM 使用其分配的六个。调度器限制亲和性、嵌套线程和优先级，内存不足暂停新任务；WSL 亲和性不等于 Windows 独占物理线程。计算预算、超时和停止规则在 `configs/rv30.json` 与 `RESEARCH_PROTOCOL.md` 冻结。

助手每半小时检查一次长任务；本地调度器自行推进并保存进度。没有实质变化时不重复通知，也不因标准输出安静而重新启动。

## 推理与结果资格

完成冻结基线或最终模型后，可生成历史示例：

```bash
cd /home/ayaka/xPatch
/home/ayaka/.venvs/xpatch-v2/bin/python -m finance infer \
  --as-of '2026-06-30T10:00:00-04:00' \
  --output delivery/example_prediction.json
```

JSON 包含模型与协议身份、实际生成时间、起点、信息截止、目标窗口、RV 单位、均值、分位数和资格状态。信息截止固定为起点前一分钟。历史示例的 `received_at` 为 null，`eligible_signal` 为 false；不能将补算输出当作当时已发出的预测。

`--source-manifest` 可指定遵循相同审计格式的新离线三资产数据；不会改变已冻结的模型或拟合状态。`--mode live` 当前返回 `live_unavailable`，退出码 2；数据不足、非交易起点也返回相应状态，不伪造预测。成功历史回放退出码 0；签名、数据或工件核对失败返回非零。

模型未通过预设门槛时仍可交付完整、可运行的 HAR 参考基线和阴性证据。其 `active_predictive_edge_established` 与 `prospective_qualified` 为 false。若开发与历史确认均通过，仍须另外冻结事前接收/发出记录及前瞻验证器，累计至少六十个合格完整交易日；这些条件未完成时目标保持进行中。

## 验收与追溯

以下命令中的 `<签名>` 替换为 `delivery/rv30_freeze.json` 的 signature：

```bash
cd /home/ayaka/xPatch
/home/ayaka/.venvs/xpatch-v2/bin/python -m finance accept \
  --run-directory artifacts/rv30/<签名>/development
```

验收器检查历史封存、源代码/依赖/数据身份、完整搜索覆盖、日等权指标与门槛、各阶段两轮独立复核、冻结工件、推理示例和文档。复核按确切模型、种子、来源指纹、阶段和轮次逐项核对，数量相同但漏模型或重复任务也会失败；起点、时间戳、日期与校准样本必须等于预定网格。任何未完成项返回非零；结果在运行目录的 `acceptance.json`。监督模型完整回放校准与评价预测；Chronos 每个任务回放固定首尾批次，所有标签和主分数仍从原始 AAPL CSV 独立重算。独立复核入口为 `python -m finance verify --run-directory artifacts/rv30/<签名>/development --scope development --round 1`；`--scope` 另支持 `confirmation` 和 `delivery`，第二轮需先有该范围的第一轮成功证据。

`delivery/manifest.json` 是唯一最终索引，引用代码提交、原研究签名、后处理纠错签名、模型、报告、推理样例和验收证据。其 `artifact_index` 连接每份任务凭证与凭证下的模型/预测逐文件哈希，覆盖失败试验。小修复使用 Git 提交与运行编号，不复制 `finance_v3/v4` 等目录。`finance_ls/`、`finance_v2/` 及其签名、权重和结果作为历史保留；旧十二分钟收益实验的阴性结论继续有效。
