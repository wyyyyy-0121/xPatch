# 本轮实现与运行状态

本目录严格对应项目根目录的 `FUSION_IMPLEMENTATION_PLAN.md`，不替代该计划。旧 `fusion/*.py` 和旧实验产物保留原身份。本轮输出在 `outputs/fusion_next/`。

用户已授权后续研究变更持续推送 GitHub，取代计划书末尾原先的“不推送”限制；计划正文与哈希保持原身份。同步分支为 [`codex/aapl-return12-implementation`](https://github.com/wyyyyy-0121/xPatch/tree/codex/aapl-return12-implementation/aapl-research)。代码、计划和运行说明随完成的提交推送；行情、基础权重、账户凭据及本地输出不进入 Git。

阶段证据摘要见 [STATUS.md](STATUS.md)。当前 F1 全量独立审计、本地 CPU 预检和训练包验证已完成，云端 F2 正在实测。下文命令按阶段列出，不应据此重复启动已经完成或仍在运行的任务。

大型神经网络训练指定在用户的 AutoDL 单卡实例执行。本地负责数据、审计、打包和最终推理。当前云端连接及实际硬件已经核验；实例重启后须复核连接地址。GPU 暂不需要时由用户关闭实例，本工具不会创建或付费租赁实例。

## 已实现的入口

以下是现有命令；后续阶段只有在前置准入通过后才能执行。

在项目根目录运行本地检查：

```bash
AAPL_CPU_THREADS=2 AAPL_CPU_IDS=12,13 bash scripts/python.sh -m pytest fusion/next/tests_data.py fusion/next/tests_models.py fusion/next/tests_training.py fusion/next/tests_evaluation.py fusion/next/tests_experiment.py fusion/next/tests_stream.py fusion/next/tests_budget.py fusion/next/tests_confirmation.py fusion/next/tests_inference.py fusion/next/tests_freeze.py fusion/next/tests_final_test.py fusion/next/tests_cloud.py fusion/next/tests_delivery.py fusion/next/tests_accounting.py fusion/next/tests_serve.py -q
AAPL_CPU_THREADS=2 AAPL_CPU_IDS=12,13 bash scripts/python.sh -m fusion.next.audit
```

审计会从原始开发数据重建特征与标签，验证文件身份、会话边界、缺失规则和覆盖准入。存在未下载的股票时只报告部分审计，不能据此放行 F1。

下载与准备支持断点续传。启动前先核对已有进程，不重复运行：

```bash
AAPL_CPU_THREADS=2 AAPL_CPU_IDS=14-17 bash scripts/python.sh -m fusion.next.acquire --workers 6
AAPL_CPU_THREADS=2 AAPL_CPU_IDS=14,15 bash scripts/python.sh -m fusion.next.data --wait
```

两个进程都只处理截至 2026-06-30 的开发数据。下载任务共用速率上限，不随并发数相乘；原始响应分页有哈希与完成收据。

生成代码与固定权重包，隔离解包后检查全部文件和模块导入：

```bash
AAPL_CPU_THREADS=2 AAPL_CPU_IDS=12,13 bash scripts/python.sh -m fusion.next.design
AAPL_CPU_THREADS=2 AAPL_CPU_IDS=12,13 bash scripts/python.sh -m fusion.next.cloud --verify
```

`design` 只生成搜索草案。没有真实 F2 性能结果和完整预算投影时，不生成正式 `search_manifest.json`。代码权重包不含行情数据、账户凭据或保留数据。

全部股票准备完后，先完成 F1 独立审计，再打包开发数据：

```bash
AAPL_CPU_THREADS=2 AAPL_CPU_IDS=12,13 bash scripts/python.sh -m fusion.next.audit --require-complete
AAPL_CPU_THREADS=2 AAPL_CPU_IDS=12,13 bash scripts/python.sh -m fusion.next.cloud --include-data --verify
```

只有 F1 已通过才允许第二条命令成功。开发数据包采用明确文件清单；封存数据不在允许列表中。

## 云端 F2

用户开机并提供可用 SSH 认证后，上传已验证的训练包，解包为 `aapl-fusion/`。选择已安装兼容 RTX 5090 的 PyTorch 镜像，执行：

```bash
cd aapl-fusion
bash cloud/bootstrap.sh
bash cloud/run.sh -m fusion.next.performance --batch-size 64 --stress-seconds 900
```

Bootstrap 检查真实 CUDA 矩阵运算，建立隔离环境并记录依赖版本；不会下载基础模型。F2 拒绝在本地执行大型模型，拒绝未完成 F1 的数据，且只使用 2021—2022 年训练数据。

F2 包括冻结骨干、最大形状监督基线、真实 LoRA、缓存数值对比、精度对比、保存恢复及至少 15 分钟连续压力运行。端到端计时包含取窗、设备传输和缓存读写。压力运行期间每分钟记录可读取的温度、SM 时钟、功耗、利用率、显存和时钟事件原因，读取失败如实记录。该 GPU 检查完成后，仍须加入 CPU 树模型计时，以及所有季度、种子、机制、重训和部署工作的总时长投影；加入 50% 余量后符合预算，才能锁定正式搜索。

CPU 性能准入可在数据审计完成后先在本地执行，不读取正式验证分数：

```bash
AAPL_CPU_THREADS=8 AAPL_CPU_IDS=0-7 bash scripts/python.sh -m fusion.next.performance --cpu-only
```

如果最终在云端执行 CPU 树模型，必须在云端重新执行同一 CPU 准入，不能把不同机器的吞吐直接用于云端总时长投影。完成云端 GPU 与 CPU 实测后，运行：

```bash
bash cloud/run.sh -m fusion.next.budget
bash cloud/run.sh -m fusion.next.experiment F3
bash cloud/run.sh -m fusion.next.experiment F4
bash cloud/run.sh -m fusion.next.experiment F5
```

预算入口先根据实际速度确定固定更新预算，再依次减少宽度和配置数。它列入全部信息范围的 OOF、三种子、七组机制及最终拟合，不符合包级预算时拒绝冻结搜索。预算还保守保留第二次拟合额度，但最终检验明确使用 Q2 已确认的同一模型，不在 Q2 后重训。正式实验入口只接受通过 F2 且与冻结代码身份一致的配置。

标准组按前 1/3、再前 2 个晋级。若评分前的预算削减仅剩 2—3 个起始配置，则首轮至少保留 2 个，以满足两个完整预算候选和三种子复核；这项小组规则同时写入搜索清单和预算计算。性能投影包含训练段缩放器拟合，并对 TCN 与 Transformer 实测取较慢速度；LoRA 的正式 FP32 计算不能用 BF16 的试跑速度估价。

`freeze` 提供 F6 前的候选系统、最强单模型、简单融合及统计阈值冻结入口；`stream` 提供完成柱验证、延迟重放时间标记、重复与缺失输入处理和状态恢复；`inference` 提供固定四专家及三种子均值的加载接口。这些代码存在不表示已完成实际重放、延迟或预测验收。

F5 完成后，才允许冻结候选并做一次已知 Q2 的流程确认：

```bash
bash cloud/run.sh -m fusion.next.freeze
bash cloud/run.sh -m fusion.next.confirmation
```

冻结入口重新核验共同起点、真实标签、原始预测、三种子均值及最终单模型/简单融合排名，摘要与原始记录不一致时拒绝放行。开发门槛未通过时，确认入口仍按固定流程拟合和导出可运行系统，但不计算 Q2 分数，不开启 Q3。已完成的确认报告不能被新系统替换。上述命令均不授权访问 Q3。

F6 完成后，在云端生成回传包，保留本地复核所需的原始开发预测、OOF、检查点、冻结部署与预算记录：

```bash
bash cloud/run.sh -m fusion.next.cloud --return-results
```

将 `outputs/fusion_next/cloud/return_bundle.tar` 和单独的 `return_receipt.json` 带回本地，执行：

```bash
AAPL_CPU_THREADS=2 AAPL_CPU_IDS=12,13 bash scripts/python.sh -m fusion.next.restore --archive outputs/fusion_next/cloud/return_bundle.tar --receipt outputs/fusion_next/cloud/return_receipt.json
```

恢复入口先在隔离目录核验整包、全部成员、逐文件 SHA、代码和计划身份，再写入结果。已有不同模型或预测时拒绝整次恢复，保留冲突清单；完全相同的结果支持重复执行。云端 CPU 计时用于云端预算，本地预检计时另存哈希命名的原始副本。仅复制部署权重不足以执行 F7 的独立放行复核。回传包不含 Q3 或原始开发行情；本地原有 F1 数据保持原身份。

只有新开发门槛和固定 Q2 确认均通过后，才能在本地运行一次全天最终检验：

```bash
AAPL_CPU_THREADS=4 AAPL_CPU_IDS=0-3 bash scripts/python.sh -m fusion.next.final_test --deployment outputs/fusion_next/deployment/pre_Q2
```

该入口独立复算开发与 Q2 放行条件，核对冻结代码、模型和对照身份，再使用已知 Q2 样本验证真实本地模型加载及 CPU/云端输出数值一致性，全部通过后才获取 Q3 输入。旧下午封存 CSV 和旧负结果准入规则保持原身份；新入口不打开旧 CSV，按固定范围另行下载全天 AAPL 及已冻结的背景输入，所有数据留在本地。若旧访问记录显示已被模型使用，则拒绝声称独立最终检验。最终检验只能恢复同一模型、同一起点的已完成分片；技术失败保留记录，完成后拒绝替换系统或再次评分。没有通过门槛时不得运行本命令。

将固定部署产物带回本地后，可执行开发期延迟历史重放、恢复和延迟测量：

```bash
AAPL_CPU_THREADS=4 AAPL_CPU_IDS=0-3 bash scripts/python.sh -m fusion.next.delivery --deployment outputs/fusion_next/deployment/pre_Q2 --origins 256
AAPL_CPU_THREADS=2 AAPL_CPU_IDS=12,13 bash scripts/python.sh -m fusion.next.delivery --score-file outputs/fusion_next/confirmation_predictions.npz
```

第二条命令只有在一次 Q2 确认确实运行并生成预测文件后才有输入。默认重放命令先在子进程中完成前半段并退出，再用另一个进程重新加载模型、恢复状态和完成后半段；冷启动计入实际输入准备与首次输出，温启动计入当前柱接入、窗口、推理及持久化。进程 ID、各阶段耗时、CPU 时间和逐起点计时保存于验收记录。异常退出后可在同一条重放命令上加 `--resume` 恢复；已有完成报告不能被覆盖。

重放的接收/生成时间是显式模拟时间，实际输出的墙钟另存；不能据此声称预测在历史当时及时发出。任何缺失预测都会使验收失败，不选择性丢弃。流式状态的数据与校验信息一次原子发布；训练检查点以原子收据提交完整版本，即使在收据与便利路径更新之间退出也能恢复。最多保留当前和前一个完整版本，旧版本的删除数量、字节及哈希逐项留存。

完成实际重放后，生成独立 CPU 推理包并隔离执行示例与进程恢复：

```bash
AAPL_CPU_THREADS=4 AAPL_CPU_IDS=0-3 bash scripts/python.sh -m fusion.next.local_package --deployment outputs/fusion_next/deployment/pre_Q2 --verify
```

包内含冻结代码、四专家和三种子模型、所需固定骨干权重、版本锁、运行说明及已知 2026Q1 的少量完成柱示例；不包含账户凭据、保留期数值或完整行情。运行入口不依赖旧仓库和原虚拟环境，逐文件校验后接收 JSONL 的 `bar` / `predict` 事件。输入被拒或模型失败会记录并返回非零退出码。示例使用延迟历史重放时间，不能用作历史实时预测的证据。原始工程验收即使失败也随包保留，包的可运行性与预测研究门槛分别报告。

研究目录内也可直接使用 JSONL 入口：

```bash
AAPL_CPU_THREADS=4 AAPL_CPU_IDS=0-3 bash scripts/python.sh -m fusion.next.serve --deployment outputs/fusion_next/deployment/pre_Q2 --calendar-start 2026-10-01 --calendar-end 2026-10-31 --state outputs/fusion_next/streams/feed/state.json < feed.jsonl
```

日历必须涵盖输入历史和预测当日，输入需包含 AAPL 与固定背景的至少 242 根完成常规时段柱。事件含显式时区的 `bar_start`；bar 另含 `symbol`、`received_at` 和 OHLCV `values`，predict 可含 `generated_at`。默认延迟历史重放，及时模式须显式设置 `--mode timely_forecast` 并使用真实接收与生成时间。通用入口拒绝 2026Q3 输入，避免绕过研究专用最终检验入口。

汇总已经存在的研究、工程、成本和清理证据：

```bash
AAPL_CPU_THREADS=2 AAPL_CPU_IDS=12,13 bash scripts/python.sh -m fusion.next.research_report
```

报告保存在 `outputs/fusion_next/research_report.json` 和 `.md`；缺少的阶段明确标为未完成，不将单元测试替代真实模型验收。环境安装及失败重试另计耗时；CPU 时间按进程区间合并，避免同一进程的 CPU/GPU 并发被重复计数。费用仅按记录的计算时间与报价估算，实际租赁账单须另行核对。

若一次最终检验确实完成，可只读取保存的预测，独立重算并核对全部预定统计，不重新读取行情或运行模型：

```bash
AAPL_CPU_THREADS=2 AAPL_CPU_IDS=12,13 bash scripts/python.sh -m fusion.next.research_report --recompute-final
```

## 实现细节与仍待实测的项目

- 历史窗口为 48/96/240，所有主比较使用共同合格起点；开盘可接前一交易日历史，隔夜跳空单列。
- close-only 对照将会话边界的收盘到收盘变动与日内收益分开，边界跳空不进入分钟波动尺度；只使用收盘和会话/缺失标记，不保留量价字段。
- 缺失只向前填充，保留显式掩码；未来 12 根标签柱必须均被观测，且不跨收盘。
- 联合训练按股票、交易日均衡采样，统一覆盖 2021 年起的历史。同一批次中，2021—2022 年样本使用联合分支的辅助读出；此后样本接入此前季度的真实 OOF 专家。早期历史不设固定百分比，避免改变各交易日的权重。采样和更新顺序在 10%/30%/100% 训练前缀之间保持连续。
- 多股训练的末段可按冻结配置适配 AAPL；只训练 AAPL 的配对对照使用相同更新预算。
- 检查点保存可训练权重、持久缓冲、优化器、全部随机数流、采样器和训练段缩放器。冻结骨干通过固定权重身份重建；LoRA 不回退为全参数训练。前向计算中途失败时保留上一个完整检查点。
- 冻结表示缓存按权重、预处理与代码、股票、时间边界和输入范围分开。LoRA、随机骨干不能使用预训练骨干缓存；低精度存储必须先通过数值准入。
- 评价使用原始 12 分钟收益 MSE。若部署为三种子预测均值，评价同一均值预测系统；单独报告各个种子。
- 新的最终统计入口使用 5 日非循环 moving-block bootstrap、5000 次、零中心化单侧检验和三对照 Holm 校正，并提供 1/10 日敏感性。

当前代码校验不能证明预测优于零收益或其他模型。F2 真实性能、正式搜索、机制结论、Q2 确认、条件式 Q3 检验和部署验收均须取得相应实测产物；未通过前不宣称完成。
