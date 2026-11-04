# AI 期刊原文阅读与写作参考

核查日期：2026-11-04。用途：为 [PAPER_PLAN.md](PAPER_PLAN.md) 提供论证和实验组织依据。本轮选读 JMLR、Nature Machine Intelligence 的四篇原始研究，重点为引言、设计、实验和讨论；没有独立复核其全部证明或复现作者结果。这不是截至 2026 年的完整模型综述。

原件、提取文本、请求记录和截图在 `artifacts/rv30/paper_framework_20261104/`；身份、哈希和读取范围见 [framework_evidence.json](framework_evidence.json)。该目录属于忽略的研究工件，不纳入 Git。下列 JMLR 页码按 PDF 页数定位。

## 1. AMLB: an AutoML Benchmark

Pieter Gijsbers, Marcos L. P. Bueno, Stefan Coors, Erin LeDell, Sébastien Poirier, Janek Thomas, Bernd Bischl, Joaquin Vanschoren. **JMLR 25(101):1–65, 2024.**

[期刊页面](https://www.jmlr.org/papers/v25/22-0493.html) · [原文 PDF](https://www.jmlr.org/papers/volume25/22-0493/22-0493.pdf)

实际读取：摘要与引言；§5.1 数据集、指标、缺失实验结果；§6 开头、§6.1 统计比较部分、§6.4 失败分析；§7 开头。设计从 PDF 第 13 页开始，结果第 20 页，失败分析第 28 页，结论第 30 页。

组织顺序：Introduction → Related Literature → AutoML Frameworks → Benchmarking Tool → Benchmark Design → Results → Conclusion；结论含 Limitations 和 Future Work。结果分别讨论预测表现、Bradley–Terry 分组比较、误差/推理速度、运行失败。

作者比较 9 个 AutoML 框架，覆盖 71 个分类和 33 个回归任务，贡献包含评价工具、比较设计和多维分析。§5.1.3 指出失败与任务属性有关，直接丢弃失败结果可能让系统的平均成绩显得更好。

对本项目的用法：保留强基线、调参条件、全部候选和失败门槛；通过分时期分析解释平均表现。调度、恢复和资源账本的细节放附录，只有研究问题涉及运行成本时才在正文分析成本。

边界：AMLB 的筛选排除了时间序列和重复测量任务。我们必须保留按日期切分、purge 和日级相关性处理，不能照搬普通交叉验证或跨任务排名推断；其 104 个任务也不能替 AAPL 单目标研究提供泛化证据。

## 2. Evaluation of post-hoc interpretability methods in time-series classification

Hugues Turbé, Mina Bjelogrlic, Christian Lovis, Gianmarco Mengaldo. **Nature Machine Intelligence 5(3):250–260, 2023. DOI: 10.1038/s42256-023-00620-w.**

[出版商原文](https://www.nature.com/articles/s42256-023-00620-w) · [原刊 PDF](https://www.nature.com/articles/s42256-023-00620-w.pdf)

实际读取：HTML 的 Abstract、Main、Results、Discussion，以及 Methods 中的合成数据说明；原刊 PDF 第 2 页（出版页码 251）已渲染并视觉核对双栏。PDF 含扩展内容共 24 页，完整下载并经 `pdfinfo` 解析。

组织顺序：Abstract → Main → Results → Discussion → Methods → Data Availability / Code Availability，另有补充与扩展材料。Methods 位于 Discussion 之后是该刊的安排，不必机械套用到其他模板。

作者指出人为判断、遮蔽后的分布变化等因素会混淆解释方法评价，随后提出评价方案、两个指标和可调复杂度的合成数据，比较六种解释方法、三类网络，以及合成、FordA 和 ECG 数据。Discussion 区分重要时间点的排序能力与所分配数值的相对贡献含义。

合成任务由两个局部正弦波的频率之和决定分类标签，判别信息的位置及特征已知。作者因而能检验解释方法识别了什么，再和真实数据结果对照。

对本项目的用法：先定义待诊断的现象，再设计能检验它的仿真。分别讨论点损失、稳定性和校准，避免混用不同概念。仿真应提供已知真值或受控条件。

边界：该文的贡献包含新评价方法、指标和合成任务，超出了模型重新排名。我们没有实现这些贡献；金融仿真不能继承其有效性结论，也不能替代真实及时市场验证。

## 3. Probabilistic Forecasting with Generative Networks via Scoring Rule Minimization

Lorenzo Pacchiardi, Rilwan A. Adewoyin, Peter Dueben, Ritabrata Dutta. **JMLR 25(45):1–64, 2024.**

[期刊页面](https://www.jmlr.org/papers/v25/23-0038.html) · [原文 PDF](https://www.jmlr.org/papers/volume25/23-0038/23-0038.pdf)

实际读取：摘要、引言，§5 实验设计、§5.1 Lorenz63/Lorenz96、§5.2 WeatherBench 的任务设置与指标、§6 结论和限制。§5 从 PDF 第 12 页开始，§5.1 第 13 页，§6 第 17 页。理论部分只核对作者主张及条件概要，未逐步验证证明。

组织顺序：Introduction → Background → 新训练目标与理论 → Related Works → Simulation Study → Conclusions。虽然实验节名为 Simulation Study，其中也包含 WeatherBench 真实气象观测。

作者以 prequential scoring rule 训练生成网络，并与对抗式训练比较。Lorenz63/Lorenz96 长轨迹按时间以前 60% 训练、随后 20% 验证、最后 20% 测试；验证段用于调参和停止选择，测试段报告确定性误差及概率校准。结论说明理论条件难以核实、尚未研究多步联合预测等限制。

对本项目的用法：让指标对应具体性质，保留时间顺序和选择边界；用受控过程检验机制，用真实历史数据报告应用表现。历史预测研究可以将真实前瞻部署作为后续工作，但结论须限于实际验证范围。

边界：本项目没有实现该训练目标或取得其一致性理论；RV30 与原文单步混沌/气象任务不同，不能借该文证明我们的区间或任务设定。

## 4. Conformal Inference for Online Prediction with Arbitrary Distribution Shifts

Isaac Gibbs, Emmanuel Candès. **JMLR 25(162):1–36, 2024.**

[期刊页面](https://www.jmlr.org/papers/v25/22-1218.html) · [原文 PDF](https://www.jmlr.org/papers/volume25/22-1218/22-1218.pdf)

实际读取：摘要、引言、方法开头；§4 实验，重点为 §4.1 模拟和 §4.2.1 股票波动任务及局部覆盖分析。经验结果从第 12 页开始，模拟第 13 页，股票例子第 16 页起；相关实证讨论见第 18—20 页。未独立核验全套 regret/coverage 证明。

组织顺序：Introduction → 动态自适应方法 → Coverage Properties → Empirical Results；经验部分含 Simulated Examples 和 Real Data Examples，后部提供证明及补充实验。

作者针对历史权重过大、步长依赖变化速度的问题提出 DtACI，并研究局部区间性质。仿真采用已知均值的正态序列，包含平稳、平滑变化和跳变情形；真实例子包括股票波动与 COVID-19 病例。

股票例子以过去 1,250 日收益拟合 GARCH(1,1)，预测下一步平方简单收益，并在四只股票上分析覆盖。该日级任务与本项目分钟起点的未来 30 分钟 log 收益平方和不同。

对本项目的用法：区间平均覆盖不足以说明变化期间的表现；为仿真预设变化和局部诊断，并同时报告宽度及点误差。

边界：我们没有实现 DtACI，现有残差区间不能使用该文的理论保证；引用也不等于已经解决分布变化。

## 5. 版式与访问边界

NMI 原刊 PDF 展示了所需双栏形态；选读 JMLR PDF 采用单栏。论证组织、期刊身份和最终栏数分别核对。NeurIPS、ICML、ICLR 是会议，没有在本轮混称为期刊。

TPAMI 的 *Deep Time Series Forecasting With Shape and Temporal Criteria* 仅取得正式元数据；NMI 的 *Lessons from a challenge on forecasting epileptic seizures from non-cerebral signals* 页面未提供完整正文。这两项均未作为已读核心原文，也未借其作具体实验结论。

NMI 首次 PDF 请求超时，只得到部分文件；一次断点续传返回 HTTP 206，补齐至 4,537,266 字节，并通过 PDF 解析。来源记录保留两次请求，最初的部分文件没有被当作完整原文。
