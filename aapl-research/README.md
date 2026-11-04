# AAPL 12 分钟收益预测研究

本研究按 [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) 执行。主目标是未来 12 分钟累计 log 收益，不加入新闻、政治或舆情输入。主评价时点为 13:30–15:47 ET。

P0—P5 已完成，68 项准入审计通过；协议已冻结，正式开发期搜索正在运行。当前状态以 [EXECUTION_STATUS.md](EXECUTION_STATUS.md) 和 `outputs` 中的机器报告为准。

- [实施计划](IMPLEMENTATION_PLAN.md)：三个开发验证块、2026Q1 流程确认、2026Q3 条件封存检验、本地 24 GPU 小时上限。
- [任务和方法说明](RESEARCH_PLAN.md)：标签、共同起点、信息时点和模型训练方式。
- [方法核查](METHOD_REVIEW.md)、[来源与只读审计](sources/evidence_manifest.json)：方法来源、许可和模型身份。
- [旧收益实验审计](LEGACY_AUDIT.md)：保留可复用的历史证据。
- [冻结协议](protocol_locked.json)、[搜索清单](trial_lists.json)：正式运行依据；原草案仅作来源记录。
- [封存清单](holdout_manifest.json)：64 个有效交易日、8,832 个共同起点及文件哈希，不含预测分数。
- [环境身份](environment_lock.json)、[模型可用时间核查](model_availability.json)。

正式运行使用 `scripts/python.sh`，其进程亲和性、依赖、缓存和临时目录限定在本项目中。主代码位于 `aapl/`，原生方法的小型源码位于 `aapl/vendor/`，并附来源和许可证。旧仓库 `/home/ayaka/xPatch` 只读，旧 Goal 和观察安排保持用户原有状态。

本地只执行零收益、Ridge、LightGBM、xPatch、PatchTST、Chronos-2 冻结和真实 LoRA 微调。Kronos、TimesFM 与全天面板属于未启用的云端扩展。研究比较完成后才判断论文方向；当前不写论文、不付费、不下单、不推送。
