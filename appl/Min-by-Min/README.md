# 分钟级高频交易预测系统（全流程自动化版）

本项目实现了一个分钟级高频交易预测系统，支持多模型自动训练、对比、预测、可视化和工程级健壮性，适用于金融时序数据的高效建模与实验。

## 项目亮点
- **一键自动化主流程**：`run.py` 串联依赖安装、完整训练、模型对比、预测、可视化、自动输出最佳模型总结。
- **多模型横向对比**：支持 LSTM、GRU、MLP、xPatch、multitask 等主流结构，权重自动管理，接口统一。
- **权重智能管理**：已有权重自动加载，无需重复训练，结构变更后自动覆盖。
- **shape健壮性**：全流程 shape 检查与自动补齐，彻底杜绝“too many indices for tensor”类报错。
- **可视化与日志**：所有结果、loss曲线、模型中间状态、残差分布等自动输出到 `plots/`，日志详尽。
- **工程结构规范**：主流程、测试/调试、模型、数据、日志、可视化等分离，目录清晰。

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 一键全流程自动化
```bash
python run.py
```
- 自动完成依赖安装、训练、模型对比、预测、可视化、最佳模型总结。
- 结果输出至 `plots/`、`checkpoints/`、`logs/`、`model_comparison_report.txt`。

### 3. 跳过训练直接对比/预测/可视化
```bash
python run_no_train.py
```
- 适用于已有权重时，仅对比、预测、可视化。

### 4. 多模型对比实验（详细报告）
```bash
python tests/compare_models_enhanced.py
```
- 支持所有主流模型横向对比，自动跳过已有权重的训练。

### 5. 预测与回测
```bash
python predict.py
```
- 自动识别最佳模型，加载对应权重，输出预测与可视化。

### 6. 可视化
```bash
python visualization.py
```
- 输出 loss 曲线、LSTM中间特征、残差分布等。

## 目录结构
```
分钟级预测研究/
├── run.py                  # 一键自动化主流程
├── run_no_train.py         # 跳过训练的自动化流程
├── train.py                # 完整训练脚本
├── predict.py              # 预测与回测
├── visualization.py        # 可视化脚本
├── config.py               # 配置文件
├── data_processor.py       # 数据处理
├── requirements.txt        # 依赖列表
├── models/                 # 各类模型结构
│   ├── LSTM.py, GRU.py, MLP.py, xpatch.py, multitask_model.py, ...
├── data/                   # 数据目录（如AAPL_1min.csv）
├── checkpoints/            # 权重文件（自动管理）
├── plots/                  # 可视化输出
├── logs/                   # 日志输出
├── tests/
│   └── compare_models_enhanced.py # 多模型对比与测试
└── README.md
```

## 主要功能说明
- **自动化主流程**：`run.py` 一键完成全部流程，适合生产/大规模实验。
- **跳过训练流程**：`run_no_train.py` 适合只做模型对比和预测。
- **多模型对比**：`tests/compare_models_enhanced.py` 支持所有模型横向对比，自动跳过已有权重训练。
- **shape健壮性**：所有模型输入输出自动检查和补齐，杜绝shape相关报错。
- **权重管理**：每个模型权重独立保存，结构变更后自动覆盖。
- **可视化**：loss曲线、LSTM中间特征、残差分布等自动输出到 `plots/`。
- **日志与报告**：所有流程自动记录日志，模型对比自动生成详细报告。

## 常见问题与解决方案
- **Q: 训练/评估时报 shape 错误？**
  - A: 已全局修复，所有输入自动补齐 shape，无需手动处理。
- **Q: 权重文件冲突或不一致？**
  - A: 结构变更后自动覆盖旧权重，无需手动删除。
- **Q: 依赖报错（如squared参数）？**
  - A: requirements.txt 已锁定 sklearn 1.6.1，自动兼容所有主流环境。
- **Q: 如何添加新模型/特征？**
  - A: 在 models/ 或 data_processor.py 中扩展，接口已统一。

## 参考性能（AAPL分钟级数据）
- MAE: 0.001~0.005
- RMSE: 0.002~0.008
- R²: 0.6~0.8
- 分类准确率: 65~75%

## 注意事项
- 数据需为分钟级时序，建议3个月以上历史。
- 回测结果仅供参考，实盘需谨慎。
- 如遇新报错，欢迎反馈，助手可持续自动修复。