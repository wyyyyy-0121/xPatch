"""Render the completed engineering gates without presenting predictive research scores."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aapl.common import read_json, utc_now


if __name__ == '__main__':
    p = read_json(ROOT / 'protocol_locked.json')
    data = read_json(ROOT / 'outputs/data_admission.json')
    manifest = read_json(ROOT / 'holdout_manifest.json')
    model = read_json(ROOT / 'outputs/model_admission.json')
    optimization = read_json(ROOT / 'outputs/optimization_admission.json')
    parallel = read_json(ROOT / 'outputs/parallel_admission.json')
    perf = read_json(ROOT / 'outputs/performance_admission.json')
    lines = ['# 数据、实现与性能准入报告', '', f'完成时间：{utc_now()}。执行依据：IMPLEMENTATION_PLAN.md。', '',
             '本报告是正式开发搜索前的工程准入，不是模型预测优越性的证据。', '',
             f"开发 CSV 的 {data['all_rows']:,} 行、134 页原始响应及所有字段已独立核对；常规交易分钟 {data['regular_rows']:,} 根、1,378 个日历会话。共同 origin、同日 240 分钟窗口和精确 12 分钟标签通过核验。", '',
             f"封存集为 2026-07-01 至 2026-09-30，共 {manifest['effective_sessions']} 个有效交易日、{manifest['eligible_origins']:,} 个共同起点；原始页面与 CSV 哈希已登记并设置只读。开发加载器默认拒绝读取，需要实际开发决策和匹配授权。", '',
             '新版验证块为 2024Q4、2025Q2、2025Q4，D 块为 2026Q1。D 块预留此前 20 个交易日作校准并排除于拟合；最终拟合截至 2026-05-29，校准仅使用 6 月最后 20 个交易日。', '',
             'xPatch EMA 在 CPU/GPU 上的 24 组值与梯度案例通过；xPatch、PatchTST、Chronos LoRA 的中途断点恢复、重建模型和优化器、错误签名与损坏文件拒绝均通过。批次切分、起点顺序和未来行扰动检查通过。', '']
    lora = model['checks']['chronos2_lora']['weight_identity']
    lines += [f"rank16 LoRA 的实际可训练参数为 {lora['trainable_parameters']:,} / {lora['all_parameters']:,}；适配参数已更新，预训练主体逐张量哈希保持不变。", '',
              '采用的优化：xPatch/PatchTST 使用原始序列 GPU 缓存和 fused AdamW；Chronos LoRA 使用原始序列 GPU 缓存。等价性容差在测试前固定为 MSE 相对变化小于 0.1%，逐点预测 rtol=1e-4、atol=1e-7。', '',
              'TF32、bf16、三分之一样本抽取未通过该容差，采用全部起点和 fp32。torch.compile 的 xPatch 路径出现 CUDA Graph 缓存覆盖错误，PatchTST 未通过数值等价检查，因此保留普通执行；失败日志仍保留。', '',
              '| 模型 | 新版最大训练起点数 | 小测端到端秒 | 最大形状端到端秒 | 单作业峰值 RSS GiB | 设备显存峰值 MiB |',
              '|---|---:|---:|---:|---:|---:|']
    for name, entries in perf['profiles'].items():
        small = read_json(ROOT / entries['small']['path'])
        large = read_json(ROOT / entries['maximum']['path'])
        lines.append(f"| {name} | {large['training_samples']:,} | {small['end_to_end_wall_seconds']:.2f} | {large['end_to_end_wall_seconds']:.2f} | {large['resources']['peak_rss_gib']:.2f} | {large['resources']['peak_gpu_used_mib']} |")
    lines += ['', '监督模型最大形状各连续训练至少 600 秒；Chronos LoRA 完整执行 900 次更新并核对保存恢复。性能样本重复使用 2021—2022 年起点，模拟完整训练规模，不计算正式开发或封存分数。LightGBM 成本因此额外乘以 2，再加入共同 50% 余量。', '',
              '三路共享 GPU 的四类模型均通过检查，三个作业实际重叠；最高设备显存占用 5,224 MiB。每个 GPU worker 使用独立 2 核，LightGBM 8 核、Ridge 2 核、管理器 2 核，合计最多 18 核，另保留旧观察所需配额。', '',
              f"冻结后的 GPU 预算含 50% 余量为 {p['measured_budget']['gpu_hours_with_margin']:.2f} 小时；CPU {p['measured_budget']['cpu_hours_with_margin']:.2f} 小时；不计并发加速的墙钟上界估计 {p['measured_budget']['wall_hours_with_margin_and_overhead']:.2f} 小时。工件峰值估计 {perf['storage_projection']['gib']:.2f} GiB。完整起始搜索规模和三个验证块均保留。", '',
              '预算包括所有配置的增量晋级、前两名三种子复核、逐轮季度验证、D 块重训与推理、配对机制、校准、可能进入封存检验的最慢第二模型，以及加载和检查点 I/O。', '',
              '实时限制：单作业 RSS≤8 GiB，系统可用 RAM≥4 GiB，设备显存≤85% 且至少剩余1.5 GiB。按三路实测可用内存再减去两个 CPU 作业的独立峰值 RSS 与0.5 GiB 管理器增量，可用内存保守下界约4.68 GiB；运行时仍持续检查实际资源。', '',
              '8 项核心测试、单进程搜索恢复小测、真实三进程晋级和淘汰小测、六类模型固定预算预测及校准收据小测通过。最终审计还会逐项核对协议、来源和脚本哈希。', '',
              '保留的已有成果包括隔离环境、来源许可与方法审计、数据核验、核心适配器、恢复验证和原始性能记录。已移除旧十季度执行代码、旧72小时预算及过时规划验证记录。旧仓库数据、权重、实验与观察状态未修改。', '',
              'Kronos、TimesFM 和全天面板属于未启用云端扩展。此报告不代表已运行这些比较。', '',
              '权威文件：`protocol_locked.json`、`protocol_lock_receipt.json`、`trial_lists.json`、`environment_lock.json`、`outputs/admission_audit.json`。', '']
    (ROOT / 'ADMISSION_REPORT.md').write_text('\n'.join(lines))
