# 交付主线数据探针

2026-11-03，25 次只读行情请求完成。原始响应保留于 Git 忽略目录 `data_acquisition/runs/delivery_probe_20261103/`，请求参数、接收时间、HTTP 状态、字段、数量和 SHA256 见该目录 `probe.json`。

| 项目 | 实测结果 | 可支持的结论 |
|---|---|---|
| AAPL / SPY / QQQ 历史 SIP 分钟线 | 15/15 请求非空 HTTP 200；抽查 2021-01-04、2022-12-30、2023-01-03、2025-06-30、2026-06-30 | 这些抽样日期可访问；完整覆盖还要逐页下载及审计 |
| AAPL 历史 SIP 逐笔成交与最佳报价 | 8/8 请求非空 HTTP 200；涵盖首次外折前及评价期的抽样日期 | 存在访问能力，不证明已有完整订单流训练/评价集 |
| 最新 SIP 分钟线 | HTTP 403 | 当前无该接口访问权限 |
| 最近十分钟 SIP 分钟线 | HTTP 403 | 不能承诺及时的 SIP 分钟推理 |

两个 403 均返回 `subscription does not permit querying recent SIP data`。未调用账户、订阅、购买和交易接口，未改变套餐，未回退 IEX。实时权限问题按用户约定登记跳过，不阻塞历史研究。

报价响应只有 `bp/bs/ap/as` 等最佳买卖价与数量字段，不含多档盘口。没有完整历史采集和会话覆盖审计前，不将条件订单流研究列为可执行，也不声称复现 DeepLOB。

## 官方口径核对

以下页面均已实际获取 HTTP 200，快照及提取文本保存在 `data_acquisition/runs/delivery_provider_sources_20261103/`，身份清单见 `delivery/provider_sources.json`。

- [Market Data FAQ](https://docs.alpaca.markets/docs/market-data-faq)：分钟柱采用左端时间戳，14:52 的 bar 对应 `[14:52, 14:53)`；SIP 历史请求的 end 需至少落后十五分钟才能在无相应实时订阅时访问。历史最终柱不能反推首次发布时间或修订延迟。
- [About Market Data API](https://docs.alpaca.markets/docs/about-market-data-api)：官方文档区分历史与实时权限；本文仅报告本账户的请求实测，不推断具体套餐。
- [Historical bars](https://docs.alpaca.markets/reference/stockbars)：结果先按标的再按时间排序，须使用分页令牌取得完整结果；本次每个请求仅一个标的，下载器核对所有分页及全部导出字段。
- [Historical quotes](https://docs.alpaca.markets/reference/stockquotes-1)：使用历史股票报价接口；访问权限与完整覆盖、盘口档位是不同的验收项。

下一项是 SPY/QQQ 的 20 日 DST/提前收盘结构审计，再进行多年分钟线取得与全字段重建。AAPL 复用并只读复核已封存数据。最终数据准入以 `delivery/data.json` 和 `DELIVERY_SOURCE_AUDIT.md` 为准，本文不是准入通过声明。
