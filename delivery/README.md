# AAPL 分钟起点 RV30 工具

当前主线使用稳定入口 `finance/`。`GOAL.md` 是最终验收约定，`delivery/status.json` 记录实际进度；只有 `delivery/manifest.json` 和相应验收均通过时才构成最终交付。基础测试、烟测或训练结束本身都不代表验收完成。

每个起点预测未来三十分钟累计对数收益平方之和（RV），同时输出经验分位区间。它衡量波动程度，不是未来价格、涨跌方向或交易利润预测。历史回放始终标为回放；当前及时 SIP 权限与真实接收时间采集未准入，实时模式会返回不可用状态。

## 环境与输入

在 `/home/ayaka/xPatch` 运行。现有已验证环境为 Python 3.13：`/home/ayaka/.venvs/xpatch-v2/bin/python`。沿用这个环境不改变旧实验工件或创建新的产品大版本。正式环境清单在冻结后生成的 `delivery/rv30_dependencies.json`，包括全部已安装包版本。

需要在另一台 Linux/WSL x86_64 机器重建时，先准备 Python **3.13.13** 和兼容当前 PyTorch CUDA 构建的 NVIDIA 驱动，再在独立环境安装完整分发包快照：

```bash
cd /home/ayaka/xPatch
python3.13 -m venv /home/ayaka/.venvs/xpatch-replay
/home/ayaka/.venvs/xpatch-replay/bin/python -m pip install --no-deps -r delivery/requirements.lock.txt
/home/ayaka/.venvs/xpatch-replay/bin/python -m pip check
```

`requirements.lock.txt` 从已签名的依赖 JSON 导出，包含 68 个分发包；`--no-deps` 使用完整快照，避免解析器另装未登记的依赖。Python 补丁版本、包名与版本的完整集合都参与环境核验。在线重新下载因连接重置和大包传输不稳定而未完成验证。已在本机独立新环境 `/home/ayaka/.venvs/xpatch-replay` 验证离线安装、精确依赖/签名、8 个预 2023 CPU 模型回放以及 PyTorch CPU 运算和 Chronos API 导入，见 `delivery/environment_restore.json`；新环境 GPU 执行和异机恢复尚未测试。当前正式研究继续使用原环境。正式回放前需通过以下核对，不能通过重新冻结来绕过环境不匹配：

```bash
/home/ayaka/.venvs/xpatch-replay/bin/python -c 'from finance.io import ROOT,read_json; from finance.provenance import verify_freeze; v=verify_freeze(read_json(ROOT/"configs/rv30.json")); print(v["signature"])'
```

可核验的离线恢复包约 5.4 GiB，位于 `/home/ayaka/.cache/xpatch-rv30/wheels/locked`；完整文件名、来源及 SHA256 在 `delivery/environment_wheels.json`。它包含 26 个与 PyPI 发布哈希匹配的原始 wheel，以及 42 个由当前锁定环境文件重建的本地快照，不能把后者称作原始发布包。备份该目录并恢复到目标机器后，使用清单中的唯一 wheel 集合和哈希安装：

```bash
/home/ayaka/.venvs/xpatch-replay/bin/python -I -m pip --isolated install \
  --no-index --no-deps --only-binary=:all: --require-hashes \
  --find-links /home/ayaka/.cache/xpatch-rv30/wheels/locked \
  -r delivery/requirements.offline.txt
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 0-1 /home/ayaka/.venvs/xpatch-replay/bin/python -I \
  scripts/verify_rv30_environment.py \
  --output-directory artifacts/rv30/setup/environment_recheck
```

验证输出目录须为新目录；脚本核查所有包实际位于该独立环境，并只回放预 2023 的上一窗口 RV、HAR、Ridge、LightGBM 两个信息集，不能代替正式开发或确认的两轮复核。安装目录含其他同版本 wheel 时，pip 可能选择另一文件并因哈希不符停止；必须使用 `locked` 中精确的 68 个文件。恢复包生成、受检范围和限制见 `delivery/ENVIRONMENT_RESTORE.md`。

预训练权重沿用 `configs/rv30_model_lock.json` 中的精确模型目录、文件与 SHA256，需从原本地工件备份恢复，或取得该锁对应的同一份权重后逐文件核对。模型训练与回放不会自动改用最新权重。完整权重和行情数据未加入 Git，仅克隆代码不足以复现实验。

Alpaca 凭据保留在 `/home/ayaka/.config/alpaca/credentials.env`，仅数据采集客户端通过环境读取；不把凭据加入命令参数、日志或 Git。已有数据接入、逐页原始记录与特殊日期审计见 `data_acquisition/DELIVERY_SOURCE_AUDIT.md`，统一数据身份见 `delivery/data.json`。

AAPL、SPY、QQQ 使用相同 SIP/raw 分钟柱与交易所日历。缺口保留，不插值；所有面板按相同的过去输入及 AAPL 标签资格配对。CSV、大权重、缓存和完整实验输出保留本地，不提交 Git。历史文件不能证明当时的发布和接收时刻。

数据恢复先按 `delivery/data.json` 的路径还原原始分页、CSV 和审计凭证，再运行 `finance audit` 与 `finance prepare`。数据采集入口 `python -m data_acquisition.delivery_data audit20 --symbol SPY --output <新目录>` 对固定 20 日做审计，`history` 对固定的 2021-01-04—2026-06-30 采集；`--symbol` 支持三个标的，`verify` 与 `audit` 可重建已有目录的分页和日历核查。重新向供应商下载可能得到修订后的数据，须保留为独立来源，不能覆盖当前签名中的输入。

新增离线日期用于推理时，为三个标的准备相同 SIP/raw 口径的完整历史，按 `delivery/data.json` 的结构另存经过核验的清单，并用 `infer --source-manifest <新清单>` 指定；至少保留 22 个完整历史会话及 512 分钟输入所需的上下文。新增清单不会改变本轮训练、选择或统计样本。实时采集仍须先解决 `delivery/external_dependencies.json` 中的及时 SIP 与真实接收时间准入项。

## 固定工作流程

先审计与准备输入，再完成首次外折前的工程烟测、性能测试与签名；正式搜索需要匹配的性能证据和冻结。正确性烟测不能代替完整规模吞吐测试，新增实验的启动门槛见 `delivery/PERFORMANCE.md`。当前计算机可使用以下命令：

```bash
cd /home/ayaka/xPatch
/home/ayaka/.venvs/xpatch-v2/bin/python -m finance audit
/home/ayaka/.venvs/xpatch-v2/bin/python -m finance prepare
/home/ayaka/.venvs/xpatch-v2/bin/python -m finance --help
```

正式训练与恢复均使用性能检查入口和同一运行名；入口先核验 `delivery/performance.json`，证据不符则返回非零状态，不启动训练：

```bash
cd /home/ayaka/xPatch
/home/ayaka/.venvs/xpatch-v2/bin/python scripts/start_rv30_checked.py --run-name development
```

此命令执行固定搜索和十季度外层预测。已完成且签名和逐文件指纹一致的任务会复用；中断的 xPatch 会从相同请求的 optimizer、轮数与随机状态恢复。失败或超时试验保留为失败，不能换参数重试来增加搜索额度。另一个入口正在运行时会拒绝重复启动。仅检查性能资格可加 `--check-only`，不会创建训练进程。当前测量只允许已有 `development` 续跑；本轮 xPatch 最大规模和持续运行已验收，新实验仍需其具体模型与工作负载的性能证据，以及完整的新计算计划，不能沿用本轮资格自动放行。

当前执行修订包括完整前向/损失/反向 CUDA graph、DEMA 前向及反向融合和受限显存预载，分别由 `delivery/rv30_acceleration_lock.json`、`delivery/rv30_throughput_lock.json` 签名。两份锁引用各自验证证据、前驱锁和 Git 源快照，不覆盖原研究冻结。最新 119 项工程检查、192 项配置/信息集/批量逐位检查、六组种子续训、最大规模 30 轮及正式源码双向恢复通过。具体计时与中止风险见 `delivery/RUN_RISK.md`；最大配置的提速倍数不等于整个搜索的提速倍数。显存预载最多 1 GiB，并要求留出至少 1 GiB 空闲；不满足时走原数据路径。当前保持单 GPU worker。

搜索全部结束后，以下命令依次执行统计、开发结果两轮独立回放、条件历史确认、确认任务两轮回放及交付验收。当前运行的自动等待器身份见 `delivery/run.json.continuation`；它会完成衔接，存活时不另启最终化进程。只有确认等待器退出且尚未衔接时才手动执行：

```bash
cd /home/ayaka/xPatch
/home/ayaka/.venvs/xpatch-v2/bin/python -m finance finalize \
  --run-directory artifacts/rv30/65a4daf1ebacb8a7dedca5e866010bd94cda25acc7124509ec81d1c14dc2fbf8/development
```

没有候选通过时，冻结 HAR 参考基线，并对最终基线再完整回放两轮。任一阶段复核失败都停止其依赖步骤。当前运行将训练和后处理分开，以保留原训练签名，并对后处理的顺序与验收覆盖纠错。原始研究冻结文件保持原样，纠错见 `delivery/amendments/postprocessing_correction.json`，独立纠错签名见 `delivery/rv30_postprocessing_lock.json`；它不能豁免数据、特征、模型拟合、搜索或门槛的修改。

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

当前研究的完整验收命令如下；运行目录的签名必须与 `delivery/rv30_freeze.json` 一致：

```bash
cd /home/ayaka/xPatch
/home/ayaka/.venvs/xpatch-v2/bin/python -m finance accept \
  --run-directory artifacts/rv30/65a4daf1ebacb8a7dedca5e866010bd94cda25acc7124509ec81d1c14dc2fbf8/development
```

验收器检查历史封存、源代码/依赖/数据身份、完整搜索覆盖、日等权指标与门槛、各阶段两轮独立复核、冻结工件、推理示例和文档。复核按确切模型、种子、来源指纹、阶段和轮次逐项核对，数量相同但漏模型或重复任务也会失败；起点、时间戳、日期与校准样本必须等于预定网格。任何未完成项返回非零；结果在运行目录的 `acceptance.json`。监督模型完整回放校准与评价预测；Chronos 每个任务回放固定首尾批次，所有标签和主分数仍从原始 AAPL CSV 独立重算。独立复核入口为 `python -m finance verify --run-directory artifacts/rv30/<签名>/development --scope development --round 1`；`--scope` 另支持 `confirmation` 和 `delivery`，第二轮需先有该范围的第一轮成功证据。

每轮还保存 `statistics.json`，直接从原始预测与校准工件重建三种子集成、日等权损失、经验区间、时段/季度/年度分段、MZ、DM/HAC、日和五日块 bootstrap、Holm 及全部门槛；独立实现不调用主评价器的这些统计函数。汇总评价的指纹必须与两轮统计证据一致后才能进入确认。此补充覆盖的后处理签名为 `delivery/rv30_statistics_lock.json`，它引用并保留前一份后处理锁、原研究冻结与对应 Git 快照。

`delivery/manifest.json` 是唯一最终索引，引用代码提交、原研究签名、后处理纠错签名、模型、报告、推理样例和验收证据。其 `artifact_index` 连接每份任务凭证与凭证下的模型/预测逐文件哈希，覆盖失败试验。小修复使用 Git 提交与运行编号，不复制 `finance_v3/v4` 等目录。`finance_ls/`、`finance_v2/` 及其签名、权重和结果作为历史保留；旧十二分钟收益实验的阴性结论继续有效。

完整交付逐项核对表见 `delivery/ACCEPTANCE_CHECKLIST.md`。该表区分已有工程证据与仍待正式实验产生的验收证据。
