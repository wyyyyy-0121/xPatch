# 已授权执行入口：AAPL 概率预测有限筛查

用户于本对话明确要求“开新窗口做这个实验 然后把结果统计给回到这里”。按 6.1 Sol / Low 执行，不调用 Luna，不升档。原对话 ID：01a1b438-df6f-7f00-9f94-f7d72e9d59f3。

工作目录 /home/ayaka/wslcode/aapl-research。先读 FAST_DIRECTION_PLAN_ZH.md；具体模型公式、原五折身份、统计规则读取 RESEARCH_NEXT_STEP_PLAN_ZH.md 第2至5节。不要通读历史聊天或全部补充。下列澄清在本轮首次评分前生效，优先于旧计划的含糊措辞：

1. 时限按快速方案：T0从新任务开始执行即记录，第一阶段含开发/资格/数值恢复/拟合/评分最多3小时；只有两任务筛查均通过才允许追加最多6小时，其中剩余拟合最多4小时，总截止T0+9小时。不重置时钟，不自动延期。
2. 前两折按每个任务分别判断，绝不跨任务平均NLL：C在每折均比B0、B1的NLL低；合并两折按origin数量加权，相对B0和B1都至少降低0.01nat/origin；对B2合并NLL不得更差。C的90%合并覆盖必须在[87%,93%]；这是将“不能完全失控”事前量化。其余50%覆盖/CRPS照实报告，完整五折门用长计划第5节。先冻结后统一评分，不按结果调门槛。
3. 最多24/60个拟合位置；分钟A_gap=C直接复用。历史fit/stop/outer范围沿原折，不新造独立测试。执行JSON必须在读本轮评分标签前明确fit/评估origin清单、purge、共同mask、历史支持、数值截断和优化设置。沿用原outer作历史开发评分，fit仅原purged inner-fit，stop不用来调参。数值/来源资格不通过即停止，不为凑出市场结果修改配方。
4. 资源总计最多16逻辑CPU，最多8拟合进程每个2核。分析合计2核，若同时运行则预留2核并减少拟合槽。检查实际其他相关任务共同计数。GPU关闭、无付费、Q3封存，不重启foundation/multistock或任何已关闭旧候选。
5. 批处理脚本一次运行矩阵，控制BLAS线程，原子checkpoint，真实中断恢复短测计入时限，恢复保留原截止。先做成本短测，不搜索模型。独立数值复算一次，不使用子代理、不启动多轮审查。程序运行时有界等待，避免LLM频繁轮询。
6. 失败只能限制本配方/数据/期限的证据，不能断言收益或波动本质不可预测。即使某任务通过也不说两任务共同成功；B1/B2对B0的结果单列。不得把覆盖率或波动增益算作原均值目标成功。

## 必须交付

固定输出目录 outputs/multiscale/probability_direction_screen/20261111_fast_screen/，已有文件时先检查身份/状态，不覆盖未识别产物。

- EXECUTION_PROTOCOL.json、STATUS.json：T0、固定截止、输入身份、阶段/单元状态及恢复游标；STATUS至少在阶段结束/失败/检查点提交时更新。
- RESULTS.csv：task/fold/recipe/n_origins/NLL/CRPS/coverage50/coverage90/width50/width90；失败单元不能填成零分。
- DECISION.json：numerical_qualified、screen_passed、每门结果、completed/planned、stage、failure_reason、historical_development_only=true、Q3_read=false及实际资源/耗时。区分科学失败与工程未完成。
- FINAL_RESULTS_ZH.md：先给明确继续/停止，再给按任务分列的C vs B0/B1/B2表；经典模型本身有无信号；消融范围；是否值得后续研究；原均值目标仍未完成。
- RUN_COMMANDS.md：一条复现命令、环境、参数/输出身份、checkpoint/恢复说明；保存全部合格结果和失败记录。

失败或到点也必须写DECISION和FINAL_RESULTS，停止额外搜索。通过只补既定三折；本轮不读Q3、不开始确认、不自动改论文原稿。最后可另存简短英文增补。

## 回报

用户明确要求结果回到原对话。完成后用 send_message_to_thread 向上述原对话ID发一次精简结果：实际完成量、继续/停止、两任务对三个基准的数值、覆盖率、资源/耗时、数值资格和局限、绝对文件路径。来源授权是原对话用户的原话“开新窗口做这个实验 然后把结果统计给回到这里”。若工具拒绝回报，保留文件并在本任务最终回答明确说明，原对话的监控可读取文件。不要为普通中间状态频繁发消息。
