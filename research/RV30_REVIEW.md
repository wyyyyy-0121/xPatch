# 30 分钟已实现波动率：实施前文献核查

核查日期：2026-11-03。范围为本轮 RV 目标、损失函数、强基线和证据边界；原收益预测及通用模型综述仍见 `AAPL_MINUTE_REVIEW.md`。尚未生成本轮正式外层模型分数。下述研究不直接证明 AAPL 分钟预测有效，最终以本地预注册评价为准。

## 来源与实际读取范围

本次通过 Crossref 查询并核对正式 DOI，实际读取返回的出版元数据和可用摘要；对 Journal of Finance、Journal of Financial Economics、Review of Financial Studies 作 2024-01-01 至 2026-11-03 的定向检索，每刊最多前五条相关结果。这不是穷尽检索，不把未命中的论文说成不存在。Crossref 查询初次命中了 Patton 的 SSRN 版本，随后另取正式 2011 年期刊 DOI，以下引用正式版本。

另外从 Andrew Patton 的 Duke 作者主页导航至研究页面，取得作者公开的正式论文 PDF，读取其摘要、损失定义及稳健排序的假设和相关讨论。HAR 出版社 PDF 返回访问错误，因此没有声称读完其全文。原始 JSON、网页、PDF 和提取文本在 `artifacts/rv30/literature/`；轻量来源身份见 `research/rv30_sources.json`。

| 来源 | 已核查身份与读取范围 | 对本轮的具体作用 |
|---|---|---|
| Andersen、Bollerslev、Diebold、Labys，*Modeling and Forecasting Realized Volatility*，Econometrica，2003 | [正式 DOI](https://doi.org/10.1111/1468-0262.00418)，本次核到出版元数据 | 把分钟收益聚合成 RV 是一个可检验的预测目标；不能从文章身份推断本项目达到其统计条件或复现实证结论 |
| Corsi，*A Simple Approximate Long-Memory Model of Realized Volatility*，Journal of Financial Econometrics 7(2)，174–196 | [正式 DOI](https://doi.org/10.1093/jjfinec/nbp001)，2008 年在线、2009 年卷期；本次全文受限 | 固定 HAR 多尺度基线；本地 30 分钟/日/五日/二十二日聚合与日内季节性是明确登记的任务适配，不宣称逐项复现原论文 |
| Patton，*Volatility forecast comparison using imperfect volatility proxies*，Journal of Econometrics 160(1)，246–256，2011 | [正式 DOI](https://doi.org/10.1016/j.jeconom.2010.03.034)；[作者公开 PDF](https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf)，本次读取相关原文 | 采用原方差尺度的 QLIKE；明确波动率代理噪声下的排序稳健性有条件，不能把 QLIKE 当作行情错误或系统偏差的修复器 |
| Andersen、Bollerslev，*Intraday periodicity and volatility persistence in financial markets*，Journal of Empirical Finance，1997 | [正式 DOI](https://doi.org/10.1016/S0927-5398(97)00004-2)，本次元数据 | 单列日内季节性；季节性估计必须位于每折训练内，不能由全历史先算再切分 |
| Patton、Sheppard，*Good Volatility, Bad Volatility: Signed Jumps and the Persistence of Volatility*，Review of Economics and Statistics，2015 | [正式 DOI](https://doi.org/10.1162/REST_a_00503)，本次元数据及作者研究页 | 在预注册基线中允许正负半方差；不根据外层结果事后挑符号或扩展新模型 |
| Diebold、Mariano，*Comparing Predictive Accuracy*，Journal of Business & Economic Statistics，1995 | [正式 DOI](https://doi.org/10.1080/07350015.1995.10524599)，本次元数据 | 对损失差作序列相关修正；本地以逐日损失差、固定 HAC 日滞后及块 bootstrap 明确实施，而不把相邻分钟当独立样本 |
| *Weak Identification of Long Memory with Implications for Volatility Modeling*，Review of Financial Studies，2025 | [正式 DOI](https://doi.org/10.1093/rfs/hhaf022)，本次读取注册摘要 | 摘要讨论 long-memory 与 rough 模型的弱识别和近观测等价；复杂度或机制命名不是稳定增益证据，需预注册比较与不确定性 |
| *Do Equity and Options Markets Agree about Volatility?*，Journal of Finance，2026 | [正式 DOI](https://doi.org/10.1111/jofi.70070)，本次读取注册摘要 | 摘要报告交易成本、估计风险和短期波动风险阻止可行的 pseudo-arbitrage；本项目先判预测增益，再讨论特定用途，不能把波动预测改善换算成交易收益 |

近期 JFE 命中的 idiosyncratic volatility、term-structure volatility 等文章，当前仅有元数据且目标与单股 30 分钟 RV 不同，没有把它们作为新增模型的直接依据。金融期刊、计量期刊、机器学习会议和作者技术报告各按实际出版身份记录。

## 冻结前落实的选择

主标签是观测到的分钟收盘收益平方和，不声称等于无微观结构噪声的潜在积分方差。Patton 的稳健排序论证涉及条件无偏等假设；本项目尚未证明一分钟股票 RV 满足这些条件，因此报告只针对所定义的 RV 预测目标，保留数据修订、市场微观结构和供应商边界。

论文 QLIKE 可写为 log(p)+y/p；本项目采用 y/p-log(y/p)-1，两者只差对所有候选相同的 y 项，比较排序一致。log-RV MSE 是另一个辅助指标，不能把 log-RV 直接当方差代入 QLIKE。log 预测转回原尺度时，使用隔离校准尾部的经验 smearing，并报告 Mincer–Zarnowitz 与区间覆盖；它不自动保证逐状态的条件均值无偏。

HAR 保留 level-RV 与 log 两类基线、季节性及正负半方差候选，在十二配置中用内部时间验证选择。这个修改在任何正式外层结果之前完成，配置总数不变。Ridge、LightGBM、xPatch、Chronos 采用固定预算充分搜索；不根据旧收益任务的失败分数挑选新参数。

原版 xPatch 的层数和 dropout 未开放为配置，宽度由 patch 决定；搜索只登记源码真正支持的选项。其通道独立结构在市场面板中实现共享参数辅助训练，不提供新跨通道融合层。Chronos-2 固定已有 Apache-2.0 权重 revision `29ec3766d36d6f73f0696f85560a422f50e8498c`，禁用不同预测起点间的 cross-learning；预训练污染边界未独立证实，结果仍为回顾性。

实际数据发布需要时间。协议预留一个完整分钟的信息缓冲，以 u-1min 的已收盘行情预测 [u,u+30min]，预测须在 u 前生成；深度序列预测相应走 31 步并取末步。不能在回测里默认刚结束的柱已瞬间收到，再把结果当作真实可执行性证据。

按共同起点评价两个信息集，保留 SPY 九根缺口，不插值；采用扩展季度外折、三内折、独立校准、双边界 purge、日/五日块、Holm、年度及种子一致性。主线数据足够，条件订单流线缺完整数据，实时 SIP 权限不通过；两项均保留明确状态，未经取得不外推研究结论。
