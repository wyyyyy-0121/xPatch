# 基于xPatch的高频交易预测系统

本项目实现了一个基于xPatch架构的高频交易预测系统，用于预测股票价格的短期走势并生成交易信号。该系统特别适用于分钟级别的股票价格预测。

## 🚀 项目特点

- **基于xPatch架构的双流时间序列预测模型**
- **支持分钟级别的股票价格预测**
- **自动生成交易信号**
- **包含完整的回测系统**
- **可视化预测结果和交易信号**
- **多模型横向对比**：支持xPatch、MLP、LSTM、GRU等主流结构，均已升级为深层、正则化、支持LayerNorm/Dropout/残差等，接口统一，便于实验对比
- **多任务学习**：同时进行价格回归预测和涨跌方向分类
- **注意力融合机制**：多尺度特征注意力融合，替代简单拼接
- **可学习趋势提取**：替代固定EMA分解，支持多种可学习趋势提取方法
- **快速训练验证**：3-5轮快速验证，便于调试和实验

## 📋 快速开始

### 1. 环境准备
```bash
pip install -r requirements.txt
```

### 2. 快速验证（推荐）
```bash
python quick_train.py  # 快速验证训练效果（5轮训练，约5秒）
```

### 3. 完整训练
```bash
python train.py  # 完整训练（5轮训练，约10-30分钟）
```

### 4. 模型对比实验
```bash
python compare_models_enhanced.py  # 对比所有模型性能
```

### 5. 预测和回测
```bash
python predict.py  # 使用训练好的模型进行预测
```

## 🎯 重要说明

**训练完成后模型会自动保存到 `checkpoints/` 目录，无需每次都重新训练！**

- 模型文件：`checkpoints/best_model.pth`
- 快速验证模型：`checkpoints/quick_train_model.pth`
- 对比实验模型：`checkpoints/best_[model_name].pth`

## 📁 项目结构

```
分钟级预测研究/
├── config.py                    # 配置文件
├── data_processor.py            # 数据处理模块
├── train.py                     # 主训练脚本
├── quick_train.py               # 快速训练验证脚本 ⭐
├── compare_models_enhanced.py   # 增强模型对比脚本 ⭐
├── predict.py                   # 预测脚本
├── visualization.py             # 可视化模块
├── models/                      # 模型目录
│   ├── xpatch.py                # xPatch主模型
│   ├── LSTM.py                  # LSTM模型
│   ├── GRU.py                   # GRU模型
│   ├── MLP.py                   # MLP模型
│   ├── multitask_model.py       # 多任务学习模型 ⭐
│   ├── attention_fusion.py      # 注意力融合模块 ⭐
│   ├── learnable_trend.py       # 可学习趋势提取 ⭐
│   └── ...                      # 其它模型与工具
├── data/                        # 数据目录
│   └── AAPL_1min.csv           # 苹果公司分钟级数据
├── checkpoints/                 # 模型检查点目录
├── plots/                       # 图表输出目录
└── logs/                        # 日志目录
```

## 🔧 新增功能详解

### 1. 多任务学习（回归+分类）
- **功能**：同时预测价格（回归）和涨跌方向（分类）
- **模型**：`models/multitask_model.py`
- **优势**：联合训练，提高预测准确性

### 2. 注意力融合机制
- **功能**：多尺度特征注意力融合，替代简单拼接
- **模块**：`models/attention_fusion.py`
- **类型**：
  - `MultiScaleAttentionFusion`：多尺度特征注意力融合
  - `GatedAttentionFusion`：门控注意力融合
  - `CrossScaleAttention`：跨尺度注意力

### 3. 可学习趋势提取
- **功能**：替代固定EMA分解，支持多种可学习趋势提取方法
- **模块**：`models/learnable_trend.py`
- **类型**：
  - `LearnableTrendExtraction`：双向LSTM趋势提取
  - `GatedConvTrendExtraction`：门控卷积趋势提取
  - `AdaptiveTrendExtraction`：自适应趋势提取

### 4. 增强模型对比
- **功能**：统一接口，支持所有模型横向对比
- **脚本**：`compare_models_enhanced.py`
- **指标**：MAE、RMSE、R²、训练时间、分类准确率
- **输出**：对比图表、详细报告

### 5. 快速训练验证
- **功能**：快速验证训练效果，便于调试
- **脚本**：`quick_train.py`
- **特点**：5轮训练，1000个样本，约5秒完成

## ⚙️ 配置说明

主要配置参数（在config.py中）：

```python
# 数据配置
SEQUENCE_LENGTH = 48    # 输入序列长度（分钟）
PREDICTION_LENGTH = 12  # 预测长度（分钟）
BATCH_SIZE = 32        # 批次大小

# 模型配置
HIDDEN_SIZE = 128      # 隐藏层维度
NUM_LAYERS = 4         # 层数
NUM_HEADS = 8          # 注意力头数
D_FF = 256            # 前馈网络维度

# 训练配置
NUM_EPOCHS = 5         # 训练轮数（已优化为快速验证）
LEARNING_RATE = 0.001  # 学习率
```

## 📊 使用流程

### 新手用户推荐流程：
1. **快速验证**：`python quick_train.py`
2. **查看结果**：检查 `quick_train_report.txt` 和 `plots/quick_train_loss.png`
3. **模型对比**：`python compare_models_enhanced.py`
4. **完整训练**：`python train.py`
5. **预测应用**：`python predict.py`

### 高级用户流程：
1. **自定义配置**：修改 `config.py`
2. **多任务训练**：使用 `multitask_model.py`
3. **注意力融合**：集成 `attention_fusion.py`
4. **可学习趋势**：使用 `learnable_trend.py`
5. **完整对比**：`python compare_models_enhanced.py`

## 🎯 脚本用途说明

| 脚本 | 用途 | 运行时间 | 推荐度 |
|------|------|----------|--------|
| `quick_train.py` | 快速验证训练效果 | 5秒 | ⭐⭐⭐⭐⭐ |
| `train.py` | 完整训练 | 10-30分钟 | ⭐⭐⭐⭐ |
| `compare_models_enhanced.py` | 模型对比实验 | 15-30分钟 | ⭐⭐⭐⭐ |
| `predict.py` | 预测和回测 | 1-5分钟 | ⭐⭐⭐ |
| `visualization.py` | 可视化工具 | 1-2分钟 | ⭐⭐ |

## 🔍 常见问题与解决方案

### Q1: 训练时间太长怎么办？
**A**: 使用 `python quick_train.py` 进行快速验证，只需5秒。

### Q2: 如何选择最佳模型？
**A**: 运行 `python compare_models_enhanced.py` 进行横向对比。

### Q3: 训练完成后需要重新训练吗？
**A**: 不需要！模型会自动保存，除非更换数据或模型结构。

### Q4: 如何添加新的技术指标？
**A**: 修改 `data_processor.py` 中的 `feature_engineering` 方法。

### Q5: 如何调整预测长度？
**A**: 修改 `config.py` 中的 `PREDICTION_LENGTH` 参数。

## 📈 性能指标

基于AAPL分钟级数据的典型性能：
- **MAE**: 0.001-0.005
- **RMSE**: 0.002-0.008
- **R²**: 0.6-0.8
- **分类准确率**: 65-75%

## 🚨 注意事项

1. **数据质量**：
   - 确保数据质量，处理缺失值和异常值
   - 注意数据的时间对齐
   - 建议使用至少3个月的历史数据

2. **模型调优**：
   - 根据实际数据调整模型参数
   - 注意过拟合问题
   - 可以通过调整config.py中的参数来优化性能

3. **交易风险**：
   - 回测结果仅供参考
   - 实盘交易需谨慎
   - 建议先进行充分的回测验证

4. **内存使用**：
   - 如果内存不足，可以减小 `BATCH_SIZE`
   - 使用 `quick_train.py` 进行快速验证

## 📄 许可证

MIT License

## 🤝 联系方式

如有问题或建议，请提交Issue或Pull Request。

---

**💡 提示**: 首次使用建议先运行 `python quick_train.py` 验证环境配置！