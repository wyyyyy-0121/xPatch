# AAPL 多尺度数据来源核验

更新：2026-11-07。本文记录来源调查与质量诊断，**不是数据准入、模型评分或法律意见**。目前没有确认能同时满足十五年年度历史、可核验的价格口径及论文使用权的来源；不得把下列候选冒充正式实验数据。

## 截止后信息曝光

2026-11-07 查询数据商方案时，网页页眉意外显示了一次标为当前的 AAPL 报价，其生效时间未独立核实。没有请求 2026Q3 的逐日行情序列，该报价也未进入数据集、模型或评分；但研究者已接触潜在截止后信息，不能据此宣称对所有未来信息完全盲。已将本轮新下载且含行情栏的条款 HTML 隔离，仅保留无行情的法律正文摘录；详细记录在本机 `outputs/multiscale/POST_CUTOFF_EXPOSURE.md`。封存期独立性须在任何相关评分前另行审计。

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
| [SimFin](https://www.simfin.com/en/prices/) | 官网列出免费与付费计划，并声称更高级数据有 20 年以上历史；免费档列有 5 年图表和 5 年基本面批量历史。 | 免费档是否提供至少 15 年可下载 AAPL 日收盘价未证实；发表权和具体价格计划待确认。官网还写明取消订阅后需删除下载数据及备份，影响长期复现。 |
| [EODHD 学术方案](https://eodhd.com/lp/universities) | 官网介绍面向大学和学术研究的长期数据及学生折扣。 | 尚无适用本人的学术许可、价格与发表权确认；演示数据保留为诊断，不用于训练或论文结果。 |

另有一个 [Kaggle 长期数据集](https://www.kaggle.com/datasets/adamvakar/apple-comprehensive-financial-dataset-1980-2026) 标注 CC BY-NC-SA，但其说明明确称行情来自 Yahoo Finance，且保留原数据提供方权利。这说明上传页的开放许可标签不能单独证明上游行情权利；没有下载或准入该候选。

## 价格与公司行动检查

- Kenil Patel 候选的 11,035 个日期与所用 XNYS 代理日历一致。其收盘价与此前候选完全一致；2016–2024 的重叠部分和 Nasdaq/Alpaca 诊断也高度接近，但数据商之间可能共享上游，不能据此假设彼此独立。
- 这份数据的 `Close` 看起来已按后续拆股回溯调整；`Adj Close` 是另一字段。其标签说明、与 Alpaca 原始数据的换算比较以及以下 SEC 范围核对，均不足以证明所有历史值都可无误恢复为名义收盘价。不得把回溯调整列直接称为历史名义价格。
- Alpha Vantage 归因候选与 Alpaca 原始日线在 2016–2024 年有 2,194 个共同日期，其中至少 95% 的收盘价逐值一致；五天的相对差异超过 0.1%。用拆股表换算后，它与 Kenil Patel 候选有六天相差超过 0.1%，最大约 2.40%；其中 **2015-09-30 是月末，差异约 0.318%**，可能影响年度／季度标签。Twelve Data 的诊断值也与 Kenil Patel 候选高度相近，但它们可能共享上游。当前不能凭多数来源擅自选定正确值，须在评分前固定主来源及敏感性处理。
- 另外检查了 [Plotly 于 2017 年固定版本的公开样例](https://github.com/plotly/datasets/blob/6a88de1c9a95f0019566d3b568eb26444feda686/finance-charts-apple.csv)：该月末值与 Kenil Patel 候选换算后相差约百万分之一以内，与 Alpha 归因候选相差约 0.318%。该样例发布早于 2020 年拆股，可作为时间版本的额外线索；其原始行情提供商和数据许可未独立确认，也不能因此断言它是独立来源或判定哪一值正确。
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

## 准入前还需完成

1. 取得可引用的原始供应商和适用于本研究的使用／发表条件，区分公开下载、研究使用、结果发表和原始行情再分发；不把未知条件写成已获许可。
2. 核实完整拆股链、名义价格重建误差、交易日缺失和来源拼接；保留原始版本及哈希。
3. 按历史测试曝光记录确定各折身份。旧资料不会因更换供应商变成独立测试；2026Q3 继续封存。
4. 数据和季度切分合格后，再锁定二十任务配置、有限训练矩阵和计算预算。当前 `protocol_draft.py` 只生成不可运行草案，不授权正式评分或云端租赁。
