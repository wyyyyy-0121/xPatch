# AAPL 研究项目

## 当前论文

最新正文为六页英文双列版本，包含四种图表及补充材料：

- [论文 PDF](outputs/multiscale/manuscript/20261111_editorial_figure_revision/main.pdf)
- [完整 LaTeX 压缩包](outputs/multiscale/manuscript/AAPL_six_page_revised_LaTeX_20261111.zip)
- [编辑和编译说明](outputs/multiscale/manuscript/20261111_editorial_figure_revision/README_ZH.md)

论文题目：*Multi-Horizon Forecasting of Apple Stock Prices and Returns Across Temporal Scales*。作者邮箱由作者填写。

## 目录

| 位置 | 用途 |
| --- | --- |
| `multiscale/` | 多尺度研究代码 |
| `aapl/`、`fusion/`、`scripts/` | 数据处理、模型及运行工具 |
| `outputs/multiscale/manuscript/` | 当前稿件和历史版本 |
| `outputs/` 其余目录 | 实验结果、预测、检查点及核验记录 |
| `sources/` | 文献资料 |
| `holdout_sealed/` | 封存数据；日常清理排除该目录 |
| `tmp/`、`.cache/` | 临时文件和缓存 |
| `.venv/` | 原项目运行环境 |

根目录现有研究计划按原路径保留，便于核对实验决策。旧稿和失败实验属于研究历史，不等同于临时垃圾；保留原路径可维持证据索引和复现记录中的引用。

## 清理记录

2026-11-12 已清除测试临时输出、编译缓存、Python 字节码、LaTeX 中间文件，以及论文解压验收副本。最新交付包、实验结果、检查点、代码和封存数据保留。

详细记录：[清理结果](outputs/maintenance/cleanup_20261112/RESULT.json)。后续清理应检查文件用途；不要直接删除整个 `outputs/` 或 `tmp/`，其中可能保留有效的研究产物与辅助文件。
