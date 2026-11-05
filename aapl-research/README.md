# AAPL 12 分钟收益预测研究

下一阶段研发计划见 [基础模型与多专家神经融合计划书](FUSION_IMPLEMENTATION_PLAN.md)：覆盖全天预测、多股票联合训练、状态门控、基础模型适配、机制对照与部署验收。该文件是待执行计划，下面的结果描述属于已完成的上一轮比较实验。

本研究按 [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) 执行。主目标是未来 12 分钟累计 log 收益，不加入新闻、政治或舆情输入。主评价时点为 13:30–15:47 ET。

本地研究流程已完成，结果见 [RESULTS.md](RESULTS.md)。六类候选均完成预定搜索，主指标已从保存的预测独立重算。没有候选同时在开发验证和 D 块优于零收益，因此按计划结束本轮研究，2026 年 7—9 月数据保持封存。该结果适用于本轮 AAPL 下午时段、信息输入与训练预算。

工程准入、正式完成审计和补充计划核对均通过。当前状态以 [EXECUTION_STATUS.md](EXECUTION_STATUS.md) 和 `outputs` 中的机器报告为准。

- [实施计划](IMPLEMENTATION_PLAN.md)：三个开发验证块、2026Q1 流程确认、2026Q3 条件封存检验、本地 24 GPU 小时上限。
- [任务和方法说明](RESEARCH_PLAN.md)：标签、共同起点、信息时点和模型训练方式。
- [方法核查](METHOD_REVIEW.md)、[来源与只读审计](sources/evidence_manifest.json)：方法来源、许可和模型身份。
- [旧收益实验审计](LEGACY_AUDIT.md)：保留可复用的历史证据。
- [冻结协议](protocol_locked.json)、[搜索清单](trial_lists.json)：正式运行依据；原草案仅作来源记录。
- [封存清单](holdout_manifest.json)：64 个有效交易日、8,832 个共同起点及文件哈希，不含预测分数。
- [环境身份](environment_lock.json)、[模型可用时间核查](model_availability.json)。
- [独立主指标复核](outputs/formal/independent_verification.json)、[完成审计](outputs/completion_audit.json)、[补充计划核对与成本清单](outputs/formal/postrun_plan_verification.json)。

正式运行使用 `scripts/python.sh`，其进程亲和性、依赖、缓存和临时目录限定在本项目中。主代码位于 `aapl/`，原生方法的小型源码位于 `aapl/vendor/`，并附来源和许可证。旧仓库 `/home/ayaka/xPatch` 只读，旧 Goal 和观察安排保持用户原有状态。

上一轮本地执行零收益、Ridge、LightGBM、xPatch、PatchTST、Chronos-2 冻结和真实 LoRA 微调。新的全天面板与云端融合训练按页首链接的下一阶段计划执行。研究比较完成后才判断论文方向；当前不写论文、不付费、不下单。用户已授权后续变更持续推送，研究代码及运行说明同步至 [GitHub 研究分支](https://github.com/wyyyyy-0121/xPatch/tree/codex/aapl-return12-implementation/aapl-research)。
