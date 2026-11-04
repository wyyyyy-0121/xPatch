# 直接方法文献与官方实现核查

核查日期：2026-11-04。这里只陈述实际阅读范围，不将模型排行榜或README替代方法原文。论文HTML/PDF获取失败与续传状态保存在 `sources` 的清单；正文方法段和完整下载是不同的核验范围。未运行任何下载源码。

## xPatch：主研究对象

- 原文：[xPatch: Dual-Stream Time Series Forecasting with Exponential Seasonal-Trend Decomposition](https://arxiv.org/abs/2412.17323)，本轮读取arXiv v1正文方法§3.1–3.2、实验设置及原生实现；方法发表于AAAI2025，正式链接见[AAAI proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/34270)。本轮未独立逐页比对会议最终版与arXiv版。HTML下载部分超时，已读核心方法完整，未声称审读全部附录。
- 官方实现：[stitsyuk/xPatch](https://github.com/stitsyuk/xPatch)。通过GitHub contents API读取 `models/xPatch.py`、`layers/network.py`、`layers/ema.py` 和LICENSE，记录实际blob SHA与SHA256。
- 机制：每个通道独立；递推EMA给出趋势，残差作为seasonal分量；趋势用MLP/池化支路，残差经patch及CNN支路，合并输出。这里的seasonal是分解残差名称，不能据此假设AAPL分钟收益一定有稳定季节性。
- 可调项与成本：上下文、patch/stride、EMA alpha、RevIN、优化器参数和损失；从头监督训练，无外部预训练。原文长跨度通用数据集结果不直接支持AAPL12分钟收益。
- 实现风险：网络的patch宽度影响中间宽度（代码中dim=patch_len²），搜索代价并不只由窗口决定；趋势支路依赖合法pred_len，保持12。官方EMA固定`.to('cuda')`且采用幂比值；旧仓库已改写它。新实现必须核对论文递推式及数值等价，明确上游/补丁身份。
- 许可：已读取上游Apache-2.0许可证。

## PatchTST：patch Transformer架构对照

- 原文：[A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/abs/2211.14730)，本轮读取v2 §3.1、§3.2及任务设置；官方仓库标注ICLR2023。
- 官方实现：[yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST)。本轮读取官方README及Apache-2.0 LICENSE，已逐模块获取官方监督 backbone、layers 和 RevIN；新包仅修改相对导入命名空间，并保存源文件及 Git blob 身份。
- 机制：时间patch作为Transformer token，通道独立共享权重，实例归一化；论文同时讨论监督与masked自监督。当前只选从头监督版本，不能把未执行的自监督迁移作为已用能力。
- 适用性：与xPatch共享patch/通道独立设计，用于检验双流CNN相对patch Transformer是否有增益。它不是“最新模型”，是机制对照。
- 成本：主要随patch token数、hidden维度、层数变化；32配置及同等rung预算，最终规模由端到端实测决定。

## Chronos-2：选中的通用预训练模型

- 原文：[Chronos-2: From Univariate to Universal Forecasting](https://arxiv.org/abs/2510.15821)，本轮读取v1摘要、§3的缩放/tokenization、time/group attention、quantile head和训练流程、§4训练数据、评价范围及§5.4污染相关说明。文献在已核查官方citation中为技术报告/arXiv预印本；本轮不声称独立确认期刊/会议录用。
- 官方实现：[amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting)，模型[amazon/chronos-2](https://huggingface.co/amazon/chronos-2)，官方README列120M参数。另有28M small与纯合成版本；本轮不再增加多个checkpoint搜索，以免资源预算失控。
- 机制：在标准化后做asinh变换，以patch嵌入进入encoder-only Transformer；time attention处理时间，group attention在同一任务的变量间交流；直接输出多个未来patch的分位数。论文§4区分真实/合成单变量语料与纯合成多变量结构训练。
- 适用性：单变量收益可用，OHLCV可作past-only covariates；未来OHLCV不得作为known covariates。跨origin cross-learning必须关闭，batch不能导致较早预测读取较晚行情。
- 实际API核查：本机 `chronos-forecasting==2.3.2` 的 `Chronos2Pipeline.fit` 确实提供full与LoRA；保存了方法源码供审计。**`predict_quantiles` 的mean字段实际返回中位数**，源码有明确注释。任何旧结果中的“mean”称呼需按实际返回值解释。主方案比较中位数与带明确尾部假设的分位积分均值近似，不将其当精确均值。
- 训练代价：冻结推理搜索与LoRA各列预算；LoRA更新rank4/8/16、注意力投影/官方输出层，100/300/900更新步。隔离环境已安装 PEFT 0.17.1；新适配器显式检查可训练参数，阻止静默全参数回退。使用官方原生前向和分位数目标的可恢复训练器；准入中已验证 LoRA 参数更新而主体权重不变。正式开发调参是否启动以执行状态为准。
- 预训练边界：论文对GIFT-Eval测试重叠有检查，并不自动证明对本项目AAPL所有历史无污染。本轮未独立核对checkpoint训练样本明细，因此不能将2023–2026回顾结果称zero-shot未见证券评价。
- 许可：已读取官方Apache-2.0代码许可证；旧模型锁登记checkpoint为Apache-2.0。正式准入仍需核对实际读取checkpoint的model card、revision与哈希。

## Kronos：选中的金融预训练模型，放在OHLCV扩展

- 原文：[Kronos: A Foundation Model for the Language of Financial Markets](https://arxiv.org/abs/2508.02739)，本轮读取v1 §3 tokenizer/自回归预训练、§4相关实验、test-time scaling，并核查附录中训练截止与交易所分布。官方仓库称AAAI2026接收；本轮未独立核对正式proceedings页，阅读版本仍是arXiv原文。
- 官方实现：[shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos)，模型[NeoQuasar/Kronos-small](https://huggingface.co/NeoQuasar/Kronos-small)。本轮读取官方README、`finetune/train_predictor.py` 与MIT许可证；small为24.7M、对应tokenizer-base、上下文上限512。mini的tokenizer不同，不得混用。
- 机制：OHLCVA向量经Transformer autoencoder/BSQ生成coarse/fine子token；预测器用因果自回归模型生成未来token，解码出连续路径；多次rollout平均改善点估计的方差，但增加推理成本。
- 适用性：原生金融量价序列与本任务相关；论文跨资产RankIC等结果不能直接替代单只AAPL的12分钟MSE。主close面板不采用重复close伪造OHLC的适配。
- 实际微调范围：官方脚本支持预测器训练，tokenizer编码在no_grad内；示例依赖DDP、Qlib数据流程及外部日志。新实现只采用本地单GPU训练逻辑，冻结tokenizer、更新预测器，以内层收益MSE决定epoch/checkpoint；token交叉熵可作训练损失但不能替代目标选择指标。不开外部日志、不开订单/付费服务。尚未实施新适配。
- 点预测：需要保留每条路径的第12步收盘价，先逐路径算log收益再平均。官方高层predictor若先平均OHLC后返回，必须在新适配器中取得路径层输出；否则不能声称实现了平均log收益。非正价格作为失败记录，禁止悄悄删路径。
- 预训练边界：原文附录称训练数据截止2024-06，评价从2024-07起；NASDAQ列为in-distribution交易所。它提示较早AAPL日期存在预训练重叠风险，不能独立证明AAPL每条序列是否出现；训练截止仍需与具体公开权重身份对应。2024-07以后也不能称项目未见历史。两类曝光分开记录。
- 成本：冻结推理12配置、多路径采样；预测器实际微调12→4→2配置，三个种子；tokenizer全流程微调未纳入。若不能通过性能准入，则记录“扩展未完成”，不夸大覆盖。
- 许可：已读取官方MIT许可证；实际checkpoint许可与revision在G2再次核对。

## TimesFM：已核查新版本，暂作待准入储备

- 官方资料：[google-research/timesfm](https://github.com/google-research/timesfm)。2026-11-04读取的README明确最新为**3.0**（2026-08发布），不再把2.5称最新。
- 原始TimesFM文献[A decoder-only foundation model for time-series forecasting](https://arxiv.org/abs/2310.10688)标为ICML2024；本轮未读其全文，也不能用2024方法原文代替3.0的完整方法说明。3.0进一步的方法/模型卡核验尚待G2，故不纳入已充分审读的主候选。
- 已核查能力声明与接口：官方说明3.0支持单/多变量及past-only和known covariates；README示例 `TimesFM3Evaluator.predict_batch(..., horizon=12)`；该点输出示例为median，不能默认条件均值。2.5的LoRA示例不等于3.0已验证支持同样流程。
- 许可须区分代码与权重：官方README说明源码Apache-2.0、≤2.5权重Apache-2.0，**3.0权重采用单独的非商业/非生产许可**。当前仅核查该官方说明，未逐条读取3.0权重许可正文；不得宣称已通过许可准入。若后续选用，仅考虑符合条款的本地研究，不使用付费云接口。
- 当前不下载权重、不安装包、不安排系统微调；只在核心研究和金融扩展之外最多预留4GPU小时，并在所有评分前决定是否纳入。可因资料/许可/性能未准入而排除，理由留档。

## 文献阅读怎样改变了方案

1. xPatch的通道独立意味着“加OHLCV”不能自动解释成融入公司/市场信息；主架构比较采用统一close面板。
2. xPatch原生12维输出与累计收益任务的训练损失不一致，使用保持结构的两种损失作配对对照，而不是无依据开发新结构。
3. Chronos-2的字段名不能代表统计量，必须按实际中位数解释旧结果，并在内层明确点估计规则。
4. Kronos是多路径生成模型；预测精度与rollout成本都需测试，不能拿一条路径的劣势外推到充分适配版本。
5. 模型论文的宏观benchmark、金融RankIC与本研究的单资产累计收益MSE不同，所有改善须重新在同一任务评价。
