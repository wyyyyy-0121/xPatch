# AAPL分钟数据获取与初审

核查日期：2026-10-29（Asia/Shanghai）。本次实际下载了供应商公开样本，未购买、注册、训练、提交或推送。

## 取得结果

| 来源入口 | 实际分钟范围（纽约时间） | 全时段行数 | 常规时段行数 | 常规交易日 | 观察覆盖率 |
|---|---|---:|---:|---:|---:|
| FirstRate Data 产品页样本 | 2026-10-13 04:00 至 2026-10-27 19:59 | 9,385 | 4,290 | 11 | 100% |
| 同供应商免费页样本 | 2026-10-12 04:00 至 2026-10-23 19:59 | 8,500 | 3,900 | 10 | 100% |

常规时段按XNYS日历、柱开始标签计算。两个版本重叠7,704根分钟柱，其OHLCV数值均一致；这是同供应商版本一致性检查，不是独立来源核验，不合并为正式数据。

两份样本均未发现重复时间、非有限值、负/零价格、负成交量或OHLC边界异常。常规时段最大绝对相邻分钟对数收益率约0.0061985（对应绝对简单收益率约0.62%），没有超过log(1.05)阈值的常规相邻分钟。样本只能说明这些日期的结构质量，不能证明长期价格真实性或点时可获得性。

按240个历史收益率、12个未来收益率、同日连续价格定义，11日版本有1,518个有效窗口，10日版本1,380个。样本没有跨夏令时切换或提前收盘日，远不足半年最低历史及完整开发实验。

## 版本身份与原始凭证

实际文件保存在本地被Git忽略的目录：

- [产品页原始ZIP](runs/20261029T133057666130Z/frd_sample_stock_AAPL.zip)
- [产品页原始分钟CSV](runs/20261029T133057666130Z/AAPL_1min_sample.csv)
- [产品页审计](runs/20261029T133057666130Z/quality_audit.json)
- [免费页原始ZIP](runs/20261029T133057666130Z/frd_free_page_AAPL.zip)
- [免费页原始分钟CSV](runs/20261029T133057666130Z/free_page_variant/AAPL_1min_vendor.csv)
- [免费页审计](runs/20261029T133057666130Z/free_page_variant/quality_audit.json)
- [下载收据](runs/20261029T133057666130Z/download_receipts.json)、[第二ZIP收据](runs/20261029T133057666130Z/additional_sample_receipt.json)
- [重叠版本比较](runs/20261029T133057666130Z/two_public_versions_comparison.json)

产品页ZIP SHA-256：4980f8bb772b5c5fa3360bb75a12f1c194c056322541698647c3395ada60b94e。

免费页ZIP SHA-256：e6d97a55d43fa985ba3852feeee55909fadab8725e93f06bbe8c00bbebab6bf0。

官网免费页所标样本日期仍为2026-09-21至10-02，与两份实际ZIP均不一致。产品页CSV有表头；免费页CSV无表头。免费页的派生审计输入只添加列名，原字节另存，见conversion.json。ZIP说明中的2000年起始日期属于全产品说明，不能当作样本覆盖27年。下载版本必须绑定哈希。

## 来源与处理边界

官方说明分钟柱是美国东部EST/EDT、柱开始标签，包含04:00–20:00盘前盘后，省略无成交量分钟。免费样本页面标注split-adjusted；ZIP说明列出三种复权产品但未唯一标识此CSV，故不认定为unadjusted。归一文件只给时间加America/New_York偏移，不做价格清洗或填补。

产品页描述多个交易场所，未由此独立证实为SIP或原始逐笔成交。供应商质量说明披露日终检查异常、重新取得数据或人工修正，是否依赖未来邻近信息仍未核实。文件来源比匿名CSV清楚，但不满足严格因果原始行情确认要求。

原始行情再分发受供应商许可限制；本次原始数据、网页快照和派生逐柱数据均留在runs目录，不上传GitHub。引用FirstRate Data。许可正文与供应商页面为准。

官方出处：

- https://firstratedata.com/i/stock/AAPL
- https://firstratedata.com/free-intraday-data
- https://firstratedata.com/about/dataquality
- https://firstratedata.com/about/license

## 长期开发数据的优先路线

优先尝试Alpaca历史SIP，显式feed=sip、timeframe=1Min、adjustment=raw。raw仅指公司行动调整，不等于逐笔、无修订或已证明点时因果。官方FAQ允许end至少落后15分钟的历史SIP查询无需实时SIP订阅；所有股票接口仍需账户认证。当前Windows与WSL进程均未检测到相应密钥，因此本次没有取得Alpaca行情，亦未验证具体账号权限。

已准备download_alpaca_sip.py：默认20个预先指定交易日，包括2024年3月夏令时前后、2024-10-28开发期异常日期和2024-11-29提前收盘。显式SIP、不回退IEX；保留分页原始响应、完整性、时区、哈希及请求身份，密钥只在请求头或本地隐藏输入，不写收据。取得后先审20日，再决定扩展多年开发历史（建议候选2021-01-04至2025-03-25）。

下一步必须完成：账户历史权限实测、20日覆盖/早收盘/DST审计、异常点的第二来源核查，以及公司行动和聚合规则核对。多年原始行情尚未取得。

Alpaca官方文档已按本次字节保存，在线出处：

- https://docs.alpaca.markets/us/docs/about-market-data-api
- https://docs.alpaca.markets/us/docs/market-data-faq
- https://docs.alpaca.markets/us/reference/stockbars

Massive是备选账户来源，未下载其行情。其官方文档说明无合格成交时可能不生成聚合柱，且默认拆股调整，不能默认等同原始价格：
https://massive.com/docs/rest/stocks/aggregates/custom-bars

其他核查：Yahoo官方chart和历史页返回地域服务限制403；未绕过访问控制。Dukascopy官方入口在工具中因域名解析受限未能核查；其AAPL产品是否为股票成交或CFD报价尚未确认。AlgoSeek未取得核验成功的公开AAPL文件直链，Databento需要账户密钥。均不列为已经取得的数据。

## 本次完成程度

已完成：公开样本取得、下载凭证、两个版本结构审计、SIP下载准备与离线关键测试。

未完成：20日SIP请求、独立第二供应商核验、多年分钟开发集、因果与点时可获得性证明。正式状态保持formal_eligible=false；全部新样本已用于供应商评价，不能作为独立确认。

旧实验、原数据与12个旧源码指纹不变。本次仅新增data_acquisition工作区，未提交、未推送。
