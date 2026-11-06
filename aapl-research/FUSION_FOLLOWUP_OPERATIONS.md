# 11 月 8 日后续运行的迁移与交接步骤

核查日期：2026-11-06。研究范围、候选和时间条件以 [后续计划](FUSION_FOLLOWUP_PLAN.md) 为准。本文补齐本地准备代码迁移到云端及正式结果返回的操作顺序，不增加训练或放宽门槛。

## 本地已经验证的范围

已在本地准备仅含源码、方法说明和三项数值/提速/打包策略的更新包，索引为 `outputs/fusion_followup/preparation/latest_source_overlay.json`。2026-11-06 17:10 的版本为 409,600 字节，更新包身份 `125e356972a40ab475e52d6701ffc9b3a4f696e16ba06fe9c4de49f248103986`，归档 SHA256 为 `8128cea3ae9dfd627a88e04c01d63cbcbbddef2cda8d36a37940764bdb6d5387`。

验证时将冻结引擎的原字节和更新包复制到临时目录，另启干净 Python 进程。16 个支持模块实际从该目录导入，FP32、输入准备提速和打包副本三项策略核验通过，10 个后续命令的 `--help` 入口正常退出，CUDA 未初始化。特殊的 CPU/训练启动脚本未作为普通模块导入，也未加载真实模型。首次清单遗漏 `scripts/portable_cpu_run.py` 的失败证据已保留；修正清单包含该入口及原策略所绑定的启动脚本。

该检查只验证源码迁移和入口依赖。它不是 GPU 计时、正式候选冻结、原模型 F8 或新模型部署验收。更新包未包含行情、预测、权重、凭据、正式冻结清单，也未上传云端。源码、后续计划或策略一旦改变，须重建更新包并重新核验，不能沿用旧哈希。

11/6 18:02 发现 WSL 向前校时后，系统换算的原接收者创建时间比旧收据增加 1,491 秒。日志同时包含 `Clock change detected` 和 `Forward time jump detected`，原进程仍持有接收锁；本地 `cloud/local_completion/receiver_identity_clock_review.json` 保存原收据、原进程目录时间、当前 boot ID、原始 start ticks 及日志哈希。旧 dispatch 和绑定保持原字节，未重启接收者。后续准入现额外拒绝“创建时间不符但该 PID 仍存活”的情形，直至真实退出；不能把时间偏移视为退出、重新启动或扩大预算的依据。十八项准入检查通过，包括正负两个方向的 1,491 秒偏移。此修订后的源码包须以 `latest_source_overlay.json` 的最新指向为准，前述版本保留为历史验证。

## 原任务完成后才能迁移

1. 核对云端原执行器的 PID/创建身份确已退出、原执行锁已释放；本地原接收者亦已正常退出。核查原正式回传、F3—F6、条件式 F7 和实际 F8 的完整证据。原 F8 未通过时先处理实际问题，不以本地准备检查代替。
2. 在本地执行 `scripts.followup_admission --freeze`，由真实原交付生成 `outputs/fusion_followup/preparation/frozen_latest.json` 及其指向的正式清单。`draft_not_runnable` 草案不能代替它。保留实际清单的相对路径和身份，后续所有入口使用同一清单。
3. 将更新包与该正式清单传至云端独立暂存目录，先校验归档 SHA256、成员路径、逐文件 SHA256、原引擎和主计划身份。更新包只提供新增准备组件；不重新传行情、权重或原最终数据。
4. 逐文件检查目标。同名同字节可复用；不同字节先保留并停止定位，不能直接覆盖冻结引擎、原执行器或原结果。特别核对 `scripts/review_fusion_development.py`、`scripts/finish_fusion_100h.py`、`scripts/portable_cpu_run.py` 及策略依赖，不能只复制新训练入口。
5. 原本地打包修订仅装在未来本地子进程。若云端尚无 `outputs/fusion_next/performance/portable_package_policy.json`，从已验证更新包复制同一策略字节；不在云端补造本地验收收据。保留本地原交付证据，云端仅核验正式清单所列的可迁移输入。
6. 在云端干净解释器内调用 `scripts.followup_admission.check_frozen`，核验相同清单、后续计划、全部准备实现、原数据/OOF/部署字节及原截止。此处不调用 `--freeze` 生成另一份清单。任何输入缺失或不同都先处理原因，不重算原预测来掩盖差异。

本地入口统一使用项目的线程包装器，例如：

```bash
AAPL_CPU_THREADS=2 AAPL_CPU_IDS=12,13 bash scripts/python.sh \
  -m scripts.followup_admission --freeze
```

云端的 `cloud/run.sh` 由原训练包提供；本地仓库不另设同名启动器。它运行项目虚拟环境并指定云端配置、权重及离线缓存。后续调度器要求干净解释器，不经原 `start_fast_fusion.py` 或本地接收者的 `sitecustomize` 启动；数值和提速适配器由后续工作进程在隔离输出环境内安装。

## 短测、正式训练与回传顺序

| 主机 | 入口 | 执行条件与输出 |
| --- | --- | --- |
| 云端 | `scripts.run_fusion_followup --mode probe --candidate 固定名称` | 原真实交付完成后，分别短测 A/B/C；最多五次预约，每次 600 秒，保留失败与部分尝试。正式清单通过仍不能代替实测。 |
| 云端 | `scripts.run_fusion_followup --mode pilot` | 11/8 目标窗口之后、三配方真实短测和投影通过；固定九次两轮拟合，累计六小时上限。 |
| 云端 | `scripts.review_fusion_pilot` | 重算完整九次结果并保存固定扩大决定；这是 CPU 复核，不额外拟合。 |
| 云端 | `scripts.run_fusion_followup --mode extend` | 仅由前一决定和固定配对关系放行；补齐种子，累计六小时上限。 |
| 云端 | `scripts.review_fusion_extension` | 重算完整三种子、季度及固定对照，保留开发未通过的事实。 |
| 云端 | `scripts.fit_fusion_followup` | 有扩大路径时按冻结工程配方拟合三个新融合种子，精确复用原专家字节；另守工程时限及 CPU 收尾余量。 |
| 云端 | `scripts.return_fusion_followup --mode bundle` | 无合格 pilot 时返回开发证据并绑定原包；新模型路径须先有完整扩展和工程导出。 |
| 本地 | `scripts.return_fusion_followup --mode fetch` | 使用同一正式清单下载、校验及恢复，保留中断文件与冲突证据。 |
| 本地 | `scripts.deliver_fusion_followup` | 有新模型时实际执行 256 起点双进程恢复、延迟、异常输入和独立包验收；无新模型时核验保留的原包。 |
| 本地 | `scripts.report_fusion_followup` | 实际 CPU 交付守护退出后汇总原研究、后续比较、包、失败与预算证据；不自动宣布 Goal 完成。 |

表中每个入口均加 `--manifest` 指向同一份真实正式清单。云端通过 `bash cloud/run.sh -m scripts.入口名` 运行，本地通过 `bash scripts/python.sh -m scripts.入口名` 运行；不能把上表中的说明文字作为参数。线程/亲和性及软件环境须与真实短测签名一致。长作业沿用精确进程身份、独立守护、单实例锁与持久日志，观察超时不触发重复启动。

原 100 小时绝对截止仍为 **2026-11-09 17:34:22（北京时间）**。准备、迁移、等待、失败与交付均计入。原训练未结束时不部署更新或插入 GPU 短测；11/8 前不启动大规模训练。开发结果不反向替换原冻结系统，不开放新的 Q2/Q3 用途。详细机器清单、归档和日志保留本地，GitHub 仅同步源码与必要说明。

原始 CPU 交付的完整冷启动计时和实际打包模块入口另见 `fusion/next/ORIGINAL_DELIVERY_TIMING_AMENDMENT.md`。后续准入必须核验完整进程计时策略及双进程测量链，仍以 60 秒为上限；函数入口之后的历史耗时不能代替。正式后续矩阵冻结前同步最新源码覆盖包和对应策略，不能使用修改前的草稿身份。
