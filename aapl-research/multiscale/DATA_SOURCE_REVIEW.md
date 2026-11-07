# AAPL 多尺度数据来源核验

更新：2026-11-07。本文记录来源调查与质量诊断，**不是数据准入、模型评分或法律意见**。目前没有确认能同时满足十五年年度历史、可核验的价格口径及论文使用权的来源；不得把下列候选冒充正式实验数据。

## 截止后信息曝光

2026-11-07 查询数据商方案时，网页页眉意外显示了一次标为当前的 AAPL 报价，其生效时间未独立核实。没有请求 2026Q3 的逐日行情序列，该报价也未进入数据集、模型或评分；但研究者已接触潜在截止后信息，不能据此宣称对所有未来信息完全盲。此前融合研究还阅读过一篇 2026 年修订的金融预测论文，其既有阅读记录将实证样本记为 2013 年；这属于截止后的文献上下文，不应与已读取 Q3 AAPL 行情等同。已将本轮新下载且含行情栏的条款 HTML 隔离，仅保留无行情的法律正文摘录；具体曝光和旧文献阅读记录在本机 `outputs/multiscale/POST_CUTOFF_EXPOSURE.md`。封存期独立性须在任何相关评分前另行审计。

## 当前可用情况

| 来源 | 已核实的信息 | 尚缺的条件 |
| --- | --- | --- |
| 现有 Alpaca SIP/raw 分钟数据 | 2021 年至 2026 年 6 月的开发资料，既有身份及分钟覆盖已核对；可构建完整交易小时柱。[官网条款](https://s3.amazonaws.com/files.alpaca.markets/disclosures/library/TermsAndConditions.pdf)规定个人、非商业使用，并引用适用的额外数据协议。 | 本轮新任务的来源、口径、论文发表权限和切分仍须冻结；旧准入不自动证明多尺度预测有效。 |
| Alpaca 日线 | 实际返回 2016-01-04 至 2026-06-30；末日日柱在保护边界之后才完整。 | 约 10.4 年，低于年度原定十五年；季度首折样本也过少。 |
| [Kaggle：Kenil Patel](https://www.kaggle.com/datasets/kpatel00/apple-stock-prices-updated-15-august-2024) | 上传页标注 CC0，说明可用于教育和研究；日线为 1980-12-12 至 2024-09-20，另有分红和拆股表。 | 上传者仅称来自公开金融记录，未列出原始行情供应商；原始权利链和复权规则未完整核实，尚不能作为名义价格准入。 |
| [Kaggle：Rishabh K](https://www.kaggle.com/datasets/rishabhkmr/appleaapl-historical-share-price-since-listing) | 上传页标注 CC0，日线为 1980-12-12 至 2022-07-15。 | 与上项重合的 10,486 个交易日收盘价逐值相同；不能视为两份独立行情。原始来源未确认。 |
| [Kaggle：Shivam Dhiman](https://www.kaggle.com/datasets/shiivvvaam/apple-stock-market-historical-data-1980-2024) | 上传页标注 CC0，价格覆盖到 2024 年。 | 用 XNYS 日历作 Nasdaq 的近似代理时有 3 个缺失、2 个多余日期；早年价格精度较低。日历差异的真实原因仍待核查。 |
| [Kaggle：Alpha Vantage 历史快照](https://www.kaggle.com/datasets/prathamjyotsingh/apple-stocks)及[另一上传版本](https://www.kaggle.com/datasets/ayeshaaqib/apple-stock-data-1999-2024) | 上传者均称原始来源为 Alpha Vantage，覆盖 1999-11 至 2024 年；较长版本有 6,262 个日线，上传页分别标注 Apache 2.0 与 CC BY 4.0。 | 两份候选的 6,213 个重合日收盘价完全一致，不能当作独立来源。公开标签不证明上游允许转授权或论文发表；部分旧日期与另一来源不一致。 |
| [Twelve Data](https://twelvedata.com/terms) | 先前取得的开发截止内诊断数据覆盖 1980 年至 2026 年 6 月。 | 其条款区分内部使用、非展示使用、不可还原的派生数据和向第三方展示；是否允许当前订阅／演示方式用于论文训练与发布结果尚未确认。 |
| [Alpha Vantage](https://www.alphavantage.co/documentation/) | 官方文档称月线覆盖 25 年以上；[支持页面](https://www.alphavantage.co/support/)列出免费 API 额度和经过验证的教育项目。 | 月线端点是否可用、价格复权、论文发表权以及截止日期参数均未确认；其[条款](https://www.alphavantage.co/terms_of_service/)将个人许可限定为私人、个人性质的非商业使用。未注册账号，也未请求可能包含封存期行情的最新数据。 |
| [Massive 股票接口](https://massive.com/docs/rest/stocks/overview) | 官方概览说明可按自定义日期区间请求历史股票 OHLC；其[个人服务条款](https://massive.com/legal/individuals-terms-of-service)将使用限定为本人个人、非商业、非业务用途。 | 尚未确认 AAPL 日线的最早可得日期、适用方案及费用、价格调整口径和论文发表聚合结果的权利。仅检查网页片段，响应未完整读取；没有注册账户或请求市场行情，不能认定候选已准入。 |
| [FinMultiTime，论文 v1](https://arxiv.org/html/2506.05019v1) | 论文的资产表将 AAPL 标为具备时间序列；总体覆盖被描述为 2009—2025。美国股票价格序列在表 2 中标为日线，“分钟级”对应新闻文本，不是分钟价格柱；论文称美国日线来自 Yahoo Finance。 | 未核实 AAPL 自身的实际起止日期、调整口径及上游训练／发表授权。论文的 CC BY 4.0 不自动授权第三方行情；本次 Hugging Face 元数据访问超时，未下载数据，不能准入。 |
| [Kibot 官方访客接口](https://www.kibot.com/api/api-access.html) | 官方说明访客可获取美国股票日线；[历史接口](https://www.kibot.com/api/history-request.html)文档列出日期上下界及 `unadjusted=1` 选项。 | [许可](https://www.kibot.com/license.html)将数据和相关或派生信息用于许可人内部，论文发表聚合结果的权限未确认。实际历史价格口径仍需调查，内部诊断回执留在本机，尚未准入训练。 |
| [QuantConnect](https://www.quantconnect.com/terms) | 条款描述平台数据可在托管环境内用于研究与算法开发。 | 平台数据导出受到明确限制，学术用途不自动例外；购买的 CLI 数据或机构派生数据协议有单独规定。尚无适合本地研究及发表的适用许可，不取得或绕过平台数据。 |
| [SimFin](https://www.simfin.com/en/prices/) | 官网列出免费与付费计划，并声称更高级数据有 20 年以上历史；免费档列有 5 年图表和 5 年基本面批量历史。 | 免费档是否提供至少 15 年可下载 AAPL 日收盘价未证实；发表权和具体价格计划待确认。官网还写明取消订阅后需删除下载数据及备份，影响长期复现。 |
| [EODHD 学术方案](https://eodhd.com/lp/universities) | 官网介绍面向大学和学术研究的长期数据及学生折扣。 | 尚无适用本人的学术许可、价格与发表权确认；演示数据保留为诊断，不用于训练或论文结果。 |

另有一个 [Kaggle 长期数据集](https://www.kaggle.com/datasets/adamvakar/apple-comprehensive-financial-dataset-1980-2026) 标注 CC BY-NC-SA，但其说明明确称行情来自 Yahoo Finance，且保留原数据提供方权利。这说明上传页的开放许可标签不能单独证明上游行情权利；没有下载或准入该候选。

## 追加的公开数据集元数据核对

以下只检查了上传页元数据，没有下载价格行，也没有将其用于训练或评分。检索时间为 2026-11-07，原始元数据和哈希回执保存在本机 `outputs/multiscale/source_references/kaggle_metadata_20261107T072219536837Z/`。上传者给出的许可标签与用途描述不等于原始行情提供商的授权。

| 数据集 | 页面说明 | 目前判断 |
| --- | --- | --- |
| [MAANG Historical Stock Market Data](https://www.kaggle.com/datasets/soumendraprasad/stock) | 标注 CC0；说明包含 2001–2023 年苹果等公司数据，并提及研究用途。 | 页面未列出行情上游来源；该版本也在 2023 年结束，不能直接覆盖计划截止期。发表权和拼接口径均未确认。 |
| [Apple Stock Price Dataset](https://www.kaggle.com/datasets/mayankanand2701/apple-stock-price-dataset) | 标注 MIT；说明包含约 44 年苹果价格与调整收盘价。 | 上传页未说明行情上游提供商和调整规则；仅检查说明，没有验证具体日期与逐日质量。 |
| [US Stock Market Historical](https://www.kaggle.com/datasets/sadiajavedd/us-stock-market-historical) | 标注 CC0；描述美国股票日线及分析用途。 | 上游来源、AAPL 实际覆盖期和价格口径未在所查说明中明确。 |
| [New York Stock Exchange](https://www.kaggle.com/datasets/dgawlik/nyse) | 标注 CC0，但页面明确称价格来自 Yahoo Finance；大多数价格覆盖 2010–2016 年。 | 该例进一步说明上传者许可标签不能单独证明上游授权；年限也不足原年度任务要求。 |
| [S&P 500 stock data](https://www.kaggle.com/datasets/camnugent/sandp500) | 标注 CC0；作者称新版从 Investors Exchange API 获取，包含约五年数据至 2018 年 2 月。 | 不满足年度历史要求；原始供应商许可及论文发表条件未核实。 |

Stooq 的数据库说明页面在本次普通访问时返回浏览器验证页面，未取得可核对的许可正文；其使用权、历史覆盖及价格口径仍为未知，不能将它列为合格来源。

## 学术数据仓库的追加核对

本次通过数据仓库 API 查看具体版本的元数据，并仅对其中一个 2024 年发布的 AAPL 文件作历史区间内部诊断；没有读取封存期行情，也没有将任何候选送入训练或评分。仓库页面所标许可不自动证明上传者拥有上游行情的转授权。

| 来源 | 核验结果 | 准入判断 |
| --- | --- | --- |
| [Zenodo AAPL 历史数据，记录 12747720](https://doi.org/10.5281/zenodo.12747720) | 仓库标注 CC0；文件有 10,987 个日线日期，1980-12-12 至 2024-07-15，下载文件的 MD5 与仓库记录一致。记录说明未识别原始行情供应商。 | 截止 2024 年，须另接后续历史；上游权利链、复权和来源拼接尚未核实，不能准入正式研究。 |
| [Zenodo AAPL/MSFT 周线数据](https://doi.org/10.5281/zenodo.20573716) | 上传页标注 CC BY 4.0，说明原数据来自 Yahoo Finance，覆盖 2019-05 至 2024-05，提供每周调整收盘价。 | 不满足长历史日线和名义价格要求；上传页许可不能单独证明 Yahoo 上游授权。 |
| [Harvard Dataverse AAPL 文件](https://doi.org/10.7910/DVN/AL6IFR) | 仓库标注 CC0；说明覆盖 2024-02 至 2025-01，未明确行情上游。 | 约一年历史，不满足年度任务；未下载价格文件。 |
| [UNC Dataverse Apple 30-year Financial Data](https://doi.org/10.15139/S3/BTHPWT) | 仓库标注 CC0；三十年内容是资产负债表、利润表和现金流量表。 | 属于财务报表而非所需的股票日线价格；未下载数据文件。 |

Zenodo 日线与已检查的 Kenil Patel 候选有 10,987 个共同日期，`Close` 的最大相对差异约为 5.82×10⁻⁸，数值接近但不全为逐值相同；两份文件的 `Adj Close` 在全部共同日期的相对差异均超过 0.1%，原因尚未确定。这些相似性不能证明独立来源或授权，也不能把两份文件当作互相独立的价格真值。仓库元数据回执见本机 `outputs/multiscale/source_references/repository_metadata_20261107T084634213624Z/receipt.json`；文件和逐列比较回执见 `outputs/multiscale/daily_sources/zenodo_12747720_candidate_20261107T084543152731Z/`。

## 价格与公司行动检查

- Kenil Patel 候选的 11,035 个日期与所用 XNYS 代理日历一致。其收盘价与此前候选完全一致；2016–2024 的重叠部分和 Nasdaq/Alpaca 诊断也高度接近，但数据商之间可能共享上游，不能据此假设彼此独立。
- 这份数据的 `Close` 看起来已按后续拆股回溯调整；`Adj Close` 是另一字段。其标签说明、与 Alpaca 原始数据的换算比较以及以下 SEC 范围核对，均不足以证明所有历史值都可无误恢复为名义收盘价。不得把回溯调整列直接称为历史名义价格。
- Alpha Vantage 归因候选与 Alpaca 原始日线在 2016–2024 年有 2,194 个共同日期，其中至少 95% 的收盘价逐值一致；五天的相对差异超过 0.1%。用拆股表换算后，它与 Kenil Patel 候选有六天相差超过 0.1%，最大约 2.40%；其中 **2015-09-30 是月末，差异约 0.318%**，可能影响年度／季度标签。Twelve Data 的诊断值也与 Kenil Patel 候选高度相近，但它们可能共享上游。当前不能凭多数来源擅自选定正确值，须在评分前固定主来源及敏感性处理。
- 另外检查了 [Plotly 于 2017 年固定版本的公开样例](https://github.com/plotly/datasets/blob/6a88de1c9a95f0019566d3b568eb26444feda686/finance-charts-apple.csv)：该月末值与 Kenil Patel 候选换算后相差约百万分之一以内，与 Alpha 归因候选相差约 0.318%。该样例发布早于 2020 年拆股，可作为时间版本的额外线索；其原始行情提供商和数据许可未独立确认，也不能因此断言它是独立来源或判定哪一值正确。
- 本轮尝试寻找 2015 年第三季度机构 13F 申报，作为月末价格分歧的独立线索；四个 SEC 索引或申报元数据请求均返回 HTTP 403，未取得持仓或完成比较。未来若能取得公开申报，仍须核对证券身份、报告日、价值单位和取整；申报估值也不能证明行情发表权。回执见本机 `source_references/sec_13f_lookup_20261107T103634194415Z/receipt.json`。
- Alpha Vantage 归因候选中，2000 财年第二至第四季度的日期完整；把 2000 年拆股前的高低价换算成年报股数单位后，六个季度极值均与 SEC 年报相差不超过 0.01 美元。第一季度只有部分日期，不用于这一比较。这仍不能证明逐日收盘价或上游许可。
- 仅用上述候选与 Alpaca 的**日期并集**，配合人工常数价格模拟样本覆盖：年度可形成 248 个合法起点，普通五折首折只有 13 个内层拟合起点；假设排除有争议的 2015-09 月末，分别降为 176 个和 3 个。按 24 个拟合起点、4 个早停起点及每折至少 6 个月验证设定的热身五折候选可以生成，但 24 个仍很少。日期并集不是行情拼接或正式准入，热身方案也尚未冻结。
- 苹果 [2000 年 SEC 年报](https://www.sec.gov/Archives/edgar/data/320193/000091205700053623/a2032880z10-k.txt)列出 1999、2000 财年各季度股价高低范围。按候选表中 2005、2014、2020 年拆股换算后，八个季度的 16 个极值都与年报相差不到 0.01 美元；最大绝对差约 0.0075 美元。这验证的是两年间的季度极值，**不是逐日收盘价或全部复权链的证明**。
- 下表中 2000、2005、2014、2020 已由公司文件核对；1987 记录在候选表和另一供应商的公司行动表中一致，但尚未找到公司一手材料。供应商可能共享上游，不能把一致性当作独立来源权利证明。

| 生效日期 | 拆股比例 | 一手材料 |
| --- | ---: | --- |
| 1987-06-16 | 2:1 | 待核实 |
| 2000-06-21 | 2:1 | [SEC 年报](https://www.sec.gov/Archives/edgar/data/320193/000091205700053623/a2032880z10-k.txt) |
| 2005-02-28 | 2:1 | [苹果公告](https://www.apple.com/newsroom/2005/02/11Apple-Announces-Two-for-One-Stock-Split/) |
| 2014-06-09 | 7:1 | [苹果公告](https://www.apple.com/newsroom/2014/04/23Apple-Expands-Capital-Return-Program-to-Over-130-Billion/) |
| 2020-08-31 | 4:1 | [苹果公告](https://www.apple.com/newsroom/2020/07/apple-reports-third-quarter-results/) |

数据及来源原文、SHA-256 和诊断回执保留在本机 `outputs/multiscale/`，不上传行情或机器收据。Kaggle 三候选比较见 `kaggle_review/20261107T055249953534Z/review.json`，SEC 季度范围比较见 `kaggle_review/sec_ranges_20261107T060941942177Z.json`，Alpha 归因候选比较见 `kaggle_review/alpha_20261107T062504602213Z.json`，日期覆盖模拟见 `coverage_planning/20261107T063051882441Z.json`，2017 年样例见 `source_references/plotly_20261107T063711607888Z/receipt.json`。

Massive 的有界网页片段核查及限制见本机 `source_references/massive_metadata_20261107T095845862208Z/`；未保存整页，不能将其当作完整条款审读或行情数据授权。

FinMultiTime 的版本、模态频率、上游来源和访问结果见本机 `literature/finmultitime_v1_20261107T102614Z/receipt.json`；不能将不同模态的总体时间跨度当作 AAPL 日线覆盖证明。

直接访问 [Yahoo 官方服务条款地址](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html)时，当前网络虽收到 HTTP 200，页面实际显示地区服务不可用通知而非条款正文；因此未能核实 Yahoo 行情的学术发表条件。元数据回执见本机 `source_references/yahoo_terms_metadata_20261107T103402656493Z/receipt.json`。

## 准入前还需完成

1. 取得可引用的原始供应商和适用于本研究的使用／发表条件，区分公开下载、研究使用、结果发表和原始行情再分发；不把未知条件写成已获许可。
2. 核实完整拆股链、名义价格重建误差、交易日缺失和来源拼接；保留原始版本及哈希。
3. 按历史测试曝光记录确定各折身份。旧资料不会因更换供应商变成独立测试；2026Q3 继续封存。
4. 数据和季度切分合格后，再锁定二十任务配置、有限训练矩阵和计算预算。当前 `protocol_draft.py` 只生成不可运行草案，不授权正式评分或云端租赁。
