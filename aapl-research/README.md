# AAPL 12 分钟收益预测研究

下一阶段研发计划见 [基础模型与多专家神经融合计划书](FUSION_IMPLEMENTATION_PLAN.md)：覆盖全天预测、多股票联合训练、状态门控、基础模型适配、机制对照与部署验收。该文件是待执行计划，下面的结果描述属于已完成的上一轮比较实验。

云端融合运行中的后续输入提速、实测范围和接入状态见 [速度优化修订](fusion/next/FUTURE_SPEED_AMENDMENT.md)。F4 完成后已核验交接，后续 F5 使用优化入口，完整训练提速比例尚未测定。

当前 Goal 的诊断与后续时间安排见 [11 月 8 日后续项目推进计划](FUSION_FOLLOWUP_PLAN.md)：11 月 6—7 日准备和小试，下一次大规模运算安排在 11 月 8 日，沿用原 100 小时绝对截止。开发诊断入口为 `scripts/review_fusion_development.py`，只读分析三个已知开发季度，不更改本轮正式选择。

本地准备、云端迁移和实际回传的操作顺序见 [后续运行交接步骤](FUSION_FOLLOWUP_OPERATIONS.md)。源码包已通过隔离导入、三项策略和命令入口检查；它仍需原实际交付、正式候选冻结及云端输入核验，不能作为 GPU 或部署通过证据。

下一轮的候选草案、隔离缓存与恢复组件在 `scripts/followup_runtime.py`，独立进程组截止组件在 `scripts/followup_guard.py`。草案仍为 `draft_not_runnable`；云端部署和 GPU 实测准入待本轮交付后完成。

本轮交付后，用 `scripts/followup_admission.py --freeze` 核验原阶段、正式回传与实际 CPU 验收，再冻结下一轮候选。该入口仍不启动训练；原运行、推理包或截止条件不符时拒绝冻结。

`scripts/run_fusion_followup.py` 提供受守护的训练段短测、11 月 8 日固定 9 次拟合和条件式种子扩展入口，绑定真实 GPU 计时、原绝对截止、累计阶段上限和 CPU/软件环境。实际 GPU 短测和正式拟合尚未执行。

下一轮结束后，`scripts/review_fusion_pilot.py` 复算九次拟合与 2×2 对照，按预定季度、分时段和去除贡献最高五日的敏感性检查，最多推荐一个配方进入预算复核。`--mode extend` 重算决定后，只给已选配方及固定配对对照补 seed 2027/2028，另守累计 6 小时上限；`scripts/review_fusion_extension.py` 重算三种子及原开发门槛，不授权最终集访问。

`scripts/fit_fusion_followup.py` 重算完整三种子复核后冻结独立工程配方，只复用配方、终点与字节完全相同的原四专家，再拟合三个新融合种子并导出。入口绑定实测计算/拷贝投影、累计 6 小时上限和精确守护；探索保留最后 8 小时，工程拟合保留最后 2 小时给本地交付。

`scripts/return_fusion_followup.py` 另提供受独立 CPU 守护约束的正式打包、下载与恢复。它支持无合格 pilot 时保留原包的路径，也支持完整三种子新模型；整体和逐文件校验后重算开发/模型链，拒绝覆盖不同本地字节，保留中断下载。网络测试采用模拟传输，实际 pilot、扩展、工程拟合与正式回传尚未执行。

正式回传后由 `scripts/deliver_fusion_followup.py` 进行独立 CPU 验收和打包：固定 256 起点、两个真实退出/恢复进程、FP32 与批流一致、异常输入持久记录、完整冷启动/p95，以及独立包的三个真实执行进程。合并 117 项 CPU 检查已通过；准备检查采用合成行情与小模型，真实研究模型验收仍待完成，见补充计划。打包副本修订单独记录在 `fusion/next/PORTABLE_PACKAGE_AMENDMENT.md`，仅接入未来本地子进程，当前云端训练保持原样。

实际交付及守护退出后，`scripts/report_fusion_followup.py` 使用同一冻结清单汇总原研究、后续开发对照、实际 CPU/包和失败尝试的预算证据。当前合并 **128 项 CPU 检查通过**，包括对隔离集成检查中真实重放/包产物的汇总核验；真实研究交付仍待取得。它重验完整链条，工程延迟未达标继续标未通过，不开启最终数据或自动完成 Goal；详细 JSON、原始证据及机器报告仍保留本地。

论文准备的直接来源、当前实现相对原方法的改动，以及尚缺的贡献证据见 [融合相关工作核查](FUSION_RELATED_WORK.md) 和 [引用条目](FUSION_REFERENCES.bib)。这是限定范围的准备记录，不增加模型试验，也不代表已经完成新颖性检索或获得预测改善。

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
