# AAPL 分钟预测文献核查（2026-11-02）

本文件保存本轮实施前已经完成的文献核查，供仓库长期追溯。当前执行状态以根目录 `EXECUTION_PLAN.md` 为准，实际实验定义以 `PROTOCOL.md` 和 `configs/aapl_v2.json` 为准；下文没有依据本轮验证分数新增模型。

检索重点为 2025—2026 年的 Review of Financial Studies、Journal of Finance、Journal of Financial Economics，并向前追溯直接相关高频研究；模型侧核对 AAAI、ICML、ICLR 工作和作者官方发布。本次不是穷尽全部 2026 年论文的系统综述。金融期刊多篇文章受正文访问限制，使用 DOI 注册元数据和出版摘要；Kronos 读取到预印本实验与附录，Chronos-2 读取到技术报告的架构与评估部分。出版身份、作者报告结果和本地独立复现须分开。

原始核查报告、来源快照和本地实验复算证据保存在本机 `/home/ayaka/wslcode/xpatch-aapl-research-2026-11-02/`，未将外部快照正文或原始市场数据纳入 Git。下列 DOI 与官方代码链接保留，可独立访问核对。

**金融期刊给这个项目的启示**

1. Jensen、Kelly、Malamud、Pedersen，*Machine Learning and the Implementable Efficient Frontier*，Review of Financial Studies 39(10)，2026。[DOI](https://doi.org/10.1093/rfs/hhag022)。本次核到出版摘要和元数据。文章主张按扣除交易成本后的风险收益评价策略，指出忽视成本的预测策略会依赖短暂特征，净收益可能很差。它是投资组合研究，不能当成 AAPL 分钟 alpha 的直接证明；对本项目的作用是确定第二阶段评价目标，避免只比较预测 MSE。

2. *Man versus Machine Learning Revisited*，Review of Financial Studies 38(12)，2025。[DOI](https://doi.org/10.1093/rfs/hhaf066)。本次核到出版摘要和元数据。作者指出被复核研究中的前视偏差，摘要报告消除该偏差后原先的 alpha 消失。这不是关于 AAPL 的结论，也不意味着所有机器学习都无效；它与本地居中清洗、插值和测试复用问题高度相关。

3. Kelly、Malamud、Zhou，*The Virtue of Complexity in Return Prediction*，Journal of Finance 79(1)，2024，2023 年在线发表。[DOI](https://doi.org/10.1111/jofi.13298)。本次核到出版摘要和元数据。文章为复杂模型的收益预测提供理论与美股市场经验依据。因此不能从旧 LS 失败推导“深度模型对金融无用”；但同样不能把该研究转述为单只 AAPL 一分钟价格路径可预测。

4. Gu、Kelly、Xiu，*Empirical Asset Pricing via Machine Learning*，Review of Financial Studies 33(5)，2020。[DOI](https://doi.org/10.1093/rfs/hhaa009)。本次核到出版摘要和元数据。树与神经网络通过非线性特征交互改善风险溢价预测，是采用简单树模型/线性模型与神经模型共同对照的依据。这是资产定价研究，时间尺度和信息集与本项目有明显差异。

5. Kolm、Turiel、Westray，*Deep order flow imbalance: Extracting alpha at multiple horizons from the limit order book*，Mathematical Finance 33(4)，2023。[DOI](https://doi.org/10.1111/mafi.12413)。本次核到出版摘要和元数据。这篇专门金融期刊研究比通用时序榜单更贴近高频问题：使用 115 只 Nasdaq 股票的盘口信息，订单流衍生的平稳输入使较简单神经模型优于多数直接输入原始盘口的模型；摘要报告有效预测跨度约为两个平均价格变化事件。这个单位是事件，不是两分钟，也不是未来 12 分钟。它提示信息来源和特征表示可能比网络新旧重要。

6. Gao、Han、Li、Zhou，*Market intraday momentum*，Journal of Financial Economics，2018。[DOI](https://doi.org/10.1016/j.jfineco.2018.05.009)。本次核对正式出版身份，并读取同名工作论文摘要：SPY 的开盘相关半小时收益对尾盘半小时收益具有条件预测性。这说明日内可预测性可能集中在特定时段和市场状态；不支持任意分钟的价格路径都可预测。

7. Zhang、Zohren、Roberts，*DeepLOB*，IEEE Transactions on Signal Processing 67(11)，2019。[DOI](https://doi.org/10.1109/TSP.2019.2907260)。本次核对论文身份，未复核全文实验。保留作取得盘口数据后的经典对照，不把仅有 OHLCV 的项目称为已实现 DeepLOB 类预测。

前四篇属于金融核心期刊文献；Mathematical Finance、IEEE TSP 属于相关专业领域期刊。AAAI、ICML、ICLR 是机器学习会议；作者技术报告和产品新权重另行注明，避免都称作“顶刊模型”。

**比原版 xPatch 更新、值得核查的模型**

| 候选 | 已核实的来源与版本 | 对本项目的意义 | 证据边界 |
|---|---|---|---|
| xPatch | AAAI 2025，[论文](https://doi.org/10.1609/aaai.v39i19.34270) | 现有实现和原版对照，成本低，便于判断新模型的增量 | 通用时序 benchmark 优势不等于 AAPL 分钟收益优势 |
| Kronos | AAAI 2026 正式 DOI；官方仓库与预印本全文 | 专门金融 OHLCVA 预训练，优先级最高的新候选；可先看 small | 主要预测基准从 5 分钟起，投资模拟为 A 股组合；不是所需一分钟 AAPL 任务的现成验证 |
| Chronos-2 | 2025 技术报告；官方仓库于 2026 年仍维护；120M，另有 small 等变体 | 多变量、历史协变量、分位数预测，适合检验 AAPL 加市场信息 | 本次未确认 Chronos-2 的顶会正式发表；通用基准不是股票 alpha 证据 |
| TimesFM 3.0 | Google 官方仓库标注 2026 年 8 月发布，9 月更新 | 当前新版本，原生多变量和协变量，适合作通用预训练对照 | 本次版本事实来自官方 README，未独立复现新榜单；自托管 3.0 权重为非商业/非生产许可，2.5 权重仍为 Apache-2.0，选型时需区别 |
| Timer-XL | 作者官方仓库标注 ICLR 2025 | 长上下文、多个变量的 Transformer 对照 | 长上下文本身未证明对分钟收益有效 |
| TimeBridge | 作者官方仓库标注 ICML 2025 | 处理非平稳性；可作后续候选 | 官方示例集中在电力、交通、天气等；不是金融验证 |
| Moirai / Moirai-2.0 | 原论文 ICML 2024，官方仓库记录 2025 年 8 月新权重 | 通用概率预测候选 | 初轮不必把所有通用大模型全部加入，以免增加选择偏差 |

Kronos：[正式论文](https://doi.org/10.1609/aaai.v40i30.39730)、[预印本](https://arxiv.org/abs/2508.02739)、[代码](https://github.com/shiyu-coder/Kronos)。Chronos-2：[技术报告](https://arxiv.org/abs/2510.15821)、[代码](https://github.com/amazon-science/chronos-forecasting)。TimesFM：[官方仓库](https://github.com/google-research/timesfm)。Timer-XL：[代码](https://github.com/thuml/Timer-XL)。TimeBridge：[代码](https://github.com/Hank0626/TimeBridge)。Moirai：[代码](https://github.com/SalesforceAIResearch/uni2ts)。

Kronos 的论文细节尤其值得保留：预训练约 120 亿条、45 个交易所，含 Nasdaq/NYSE 和一分钟粒度；正文附录 D 说预训练截止到 2024 年 6 月，测试从 2024 年 7 月起。附录 Table 8 的主预测配置却从 `5min: 480 → 96` 开始，最短即未来 480 个交易分钟；没有列出 `1min → 1/5/12min` 的任务。价格路径 RankIC 是逐样本预测路径与真实路径的秩相关，不是 93% 方向正确率；收益预测 IC 则以各资产类别和频率的样本向量计算。投资回测是 A 股多股票组合，不能换算成 AAPL 单股短线效果。

预训练的“看没看过测试数据”与本项目的“有没有拿测试结果调过参”也要分开。论文的 2024-06 截止不能自动代表所有后续下载的权重；未来实验须冻结 checkpoint revision、哈希和发布身份。现有 2025-03 至 06 的项目测试已被使用，即使换成 Kronos 或更换行情商，也不能重新称为完全未触碰的新测试。其他预训练模型同样要核查训练数据时间边界；无法证实时，应明确标记污染边界未知，并采用后续前瞻样本。

本轮执行采用零收益、Ridge、LightGBM、原版 xPatch48/240、Kronos-small 和 Chronos-2；预训练权重版本、许可证、文件哈希与镜像传输说明见 `configs/aapl_v2_model_lock.json`。TimesFM、Timer-XL、TimeBridge、Moirai 保留为文献背景，不在看到本轮分数后追加候选。

旧 `finance_ls` 已完成 92 次运行，长短双流主方案未证明稳定优于原版；旧行情来源与极端跳变限制了旧结果的市场解释。新版已另取并验收 2021-01-04 至 2026-06-30 的 Alpaca SIP/raw 历史数据，但历史最终柱仍不能证明原始发布时间与修订延迟。新数据不消除项目对历史日期或预训练模型的曝光，2025H2–2026H1 仍只能作为历史确认。

日内累计收益误差改善只支持预测层面的结论；只有预声明稳定性门槛通过后，才允许进一步研究执行时点、价差、佣金、滑点和换手。只有 OHLCV 时，不声称复现订单流或 DeepLOB 文献结论。
