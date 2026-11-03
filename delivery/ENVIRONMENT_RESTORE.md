# RV30 环境恢复证据

2026-11-03 在同一工作站建立独立、未继承系统 site-packages 的新环境 `/home/ayaka/.venvs/xpatch-replay`，以 Python 3.13.13 恢复当前研究依赖。机器索引为 `delivery/environment_restore.json`，逐 wheel 指纹为 `delivery/environment_wheels.json`，离线安装锁为 `delivery/requirements.offline.txt`。

## 已验证范围

- 68 个分发包的名称、精确版本及 Python 版本与原冻结一致，完整 `verify_freeze` 通过，原研究签名保持不变。
- 离线安装启用 `--no-index --no-deps --require-hashes`，依赖检查通过；所有包实际位于新环境内部，不依赖原环境的 Python 包路径。
- 预 2023 的上一窗口 RV、HAR、Ridge、LightGBM，在 AAPL 和市场两个信息集共 8 项已保存模型回放全部通过；重建标签、变换、预测和校准。这些是环境恢复检查，不是正式开发或确认的两轮复核。
- 新环境 PyTorch 2.14.1+cu130 的 CPU 张量运算及锁定 Chronos-2 API 导入通过。GPU 对该验证进程隐藏，正式搜索始终独占 GPU。

没有验证新环境 GPU 内核执行、另一台计算机安装或从公网重新下载所有包。相同包版本不消除驱动、系统动态库和硬件兼容性要求。

## 恢复包的来源

完整恢复包约 5.4 GiB，在 `/home/ayaka/.cache/xpatch-rv30/wheels/locked`。其中 26 个是与官方 PyPI 发布 SHA256 匹配的原始 wheel，42 个是从当前锁定解释器可见的已安装文件重建的本地快照。重建时核对原安装 RECORD 中的文件哈希，保留包内容、元数据及兼容标签，重建归档的 RECORD；pip 从 entry_points 元数据重新生成命令脚本，环境前缀中的手册等数据使用 wheel 的标准 data 布局恢复。

本地快照的文件指纹和逐文件来源库存放在证据目录中，不能将其当作原始发布归档或从源码独立构建的证明。当前解释器可见的依赖可能位于原 venv 或其 Python 基础环境；新环境核验要求全部 68 包都安装在新前缀内。

早先公网下载遇到重复连接重置、大包传输未完成，保留在 `artifacts/rv30/setup/environment_restore_20261103/` 的传输记录中。第一次离线安装亦因混合目录中另一个同版本 wheel 被 pip 优先选择而被哈希检查拒绝；随后固定仅含所选 68 个文件的 `locked` 目录，实际安装成功。没有放宽哈希检查或修改研究依赖。

## 恢复与检查

先备份 `locked` 中全部 68 个 wheel，以及代码仓库内两份清单。数据与预训练权重须按各自现有清单另行备份；恢复包不包含行情数据或模型权重。

准备相同 Python 后，在空的新目录创建 venv，再按 README 的离线命令安装。若恢复目录不同，调整 `--find-links` 路径即可，文件内容必须匹配清单。不要把下载缓存中的其他候选 wheel 加进 `locked`；同版本不同标签的文件也可能具有不同哈希。

`scripts/verify_rv30_environment.py --output-directory <新证据目录>` 执行依赖、研究冻结、新环境路径和 8 项 CPU 回放检查。为避免占用训练资源，README 使用 CPU 0–1、单线程 BLAS、隐藏 GPU；全部实验仍共享最多 22 逻辑线程。失败保留日志且返回非零，不能以人工 passed 标志替代执行结果。

在原冻结环境仍可用时，可重新生成恢复包：

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 0-1 /home/ayaka/.venvs/xpatch-v2/bin/python \
  scripts/backup_rv30_wheels.py \
  --wheelhouse /home/ayaka/.cache/xpatch-rv30/recovery-build \
  --evidence-directory artifacts/rv30/setup/environment_backup
```

此命令默认生成本地文件快照；证据目录若已包含核验过的官方 wheel 获取记录，会复用哈希一致的原始归档。最终用于安装的是生成目录下的 `locked` 子目录及对应的 `requirements.hashed.txt`。这些是恢复工件，不是产品大版本。公网 wheel 获取工具 `scripts/prepare_rv30_wheelhouse.py` 仅用于取得原始发布包，带有限次数重试；本次未以其结果声称公网重建通过。
