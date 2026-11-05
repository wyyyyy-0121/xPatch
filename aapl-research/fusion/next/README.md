# 本轮实现与运行状态

本目录严格对应项目根目录的 `FUSION_IMPLEMENTATION_PLAN.md`，不替代该计划。旧 `fusion/*.py` 和旧实验产物保留原身份。本轮输出在 `outputs/fusion_next/`。

大型神经网络训练指定在用户的 AutoDL 单卡实例执行。本地继续数据下载、审计和打包。当前 SSH 地址只用于记录；实例重启后须确认实际连接地址。GPU 暂不需要时由用户关闭实例，本工具不会创建或付费租赁实例。

## 已实现的入口

以下是现有命令；后续阶段只有在前置准入通过后才能执行。

在项目根目录运行本地检查：

```bash
AAPL_CPU_THREADS=2 AAPL_CPU_IDS=12,13 bash scripts/python.sh -m pytest fusion/next/tests_data.py fusion/next/tests_models.py fusion/next/tests_training.py fusion/next/tests_evaluation.py fusion/next/tests_experiment.py fusion/next/tests_stream.py fusion/next/tests_budget.py -q
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

F2 包括冻结骨干、最大形状监督基线、真实 LoRA、缓存数值对比、精度对比、保存恢复及至少 15 分钟连续压力运行。端到端计时包含取窗、设备传输和缓存读写。该 GPU 检查完成后，仍须加入 CPU 树模型计时，以及所有季度、种子、机制、重训和部署工作的总时长投影；加入 50% 余量后符合预算，才能锁定正式搜索。

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

预算入口先根据实际速度确定固定更新预算，再依次减少宽度和配置数。它列入全部信息范围的 OOF、三种子、七组机制及条件式最终重训，不符合包级预算时拒绝冻结搜索。正式实验入口只接受通过 F2 且与冻结代码身份一致的配置。

`freeze` 提供 F6 前的候选系统、最强单模型、简单融合及统计阈值冻结入口；`stream` 提供完成柱验证、延迟重放时间标记、重复与缺失输入处理和状态恢复；`inference` 提供固定四专家及三种子均值的加载接口。这些代码存在不表示已完成实际重放、延迟或预测验收。

## 实现细节与仍待实测的项目

- 历史窗口为 48/96/240，所有主比较使用共同合格起点；开盘可接前一交易日历史，隔夜跳空单列。
- 缺失只向前填充，保留显式掩码；未来 12 根标签柱必须均被观测，且不跨收盘。
- 联合训练按股票、交易日均衡采样，统一覆盖 2021 年起的历史。同一批次中，2021—2022 年样本使用联合分支的辅助读出；此后样本接入此前季度的真实 OOF 专家。早期历史不设固定百分比，避免改变各交易日的权重。采样和更新顺序在 10%/30%/100% 训练前缀之间保持连续。
- 多股训练的末段可按冻结配置适配 AAPL；只训练 AAPL 的配对对照使用相同更新预算。
- 检查点保存可训练权重、持久缓冲、优化器、全部随机数流、采样器和训练段缩放器。冻结骨干通过固定权重身份重建；LoRA 不回退为全参数训练。前向计算中途失败时保留上一个完整检查点。
- 冻结表示缓存按权重、预处理与代码、股票、时间边界和输入范围分开。LoRA、随机骨干不能使用预训练骨干缓存；低精度存储必须先通过数值准入。
- 评价使用原始 12 分钟收益 MSE。若部署为三种子预测均值，评价同一均值预测系统；单独报告各个种子。
- 新的最终统计入口使用 5 日非循环 moving-block bootstrap、5000 次、零中心化单侧检验和三对照 Holm 校正，并提供 1/10 日敏感性。

当前代码校验不能证明预测优于零收益或其他模型。F2 真实性能、正式搜索、机制结论、Q2 确认、条件式 Q3 检验和部署验收均须取得相应实测产物；未通过前不宣称完成。
