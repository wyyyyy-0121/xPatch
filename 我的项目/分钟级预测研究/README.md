# 基于xPatch的高频交易预测系统

本项目实现了一个基于xPatch架构的高频交易预测系统，用于预测股票价格的短期走势并生成交易信号。该系统特别适用于分钟级别的股票价格预测。

## 项目特点

- 基于xPatch架构的双流时间序列预测模型
- 支持分钟级别的股票价格预测
- 自动生成交易信号
- 包含完整的回测系统
- 可视化预测结果和交易信号

## 项目结构

```
分钟级预测研究/
├── config.py              # 配置文件
├── data_processor.py      # 数据处理模块
├── models/               # 模型目录
│   ├── xpatch.py        # xPatch模型
│   ├── ema_decomposition.py # EMA分解模块
│   ├── patch_embedding.py   # Patch嵌入模块
│   ├── trend_stream.py  # 趋势流模型
│   └── residual_stream.py # 残差流模型
├── train.py             # 训练脚本
├── predict.py           # 预测脚本
├── visualization.py     # 可视化模块
├── data/               # 数据目录
│   └── AAPL_1min.csv   # 苹果公司分钟级数据
├── checkpoints/        # 模型检查点目录
└── plots/             # 图表输出目录
```

## 环境要求

- Python 3.8+
- PyTorch 1.9.0+
- pandas
- numpy
- scikit-learn
- ta (技术分析库)
- matplotlib
- seaborn

## 快速开始

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 训练模型：
```bash
python train.py
```

3. 预测和回测：
```bash
python predict.py
```

## 模型架构

本项目实现了一个基于xPatch架构的双流时间序列预测模型：

1. EMA分解：
   - 使用指数移动平均将时间序列分解为趋势和残差成分
   - 自适应平滑系数

2. Patch嵌入：
   - 多尺度patch切分
   - 位置编码
   - 特征投影

3. 双流处理：
   - 趋势流：处理长期趋势
   - 残差流：处理短期波动
   - 特征融合

4. 预测头：
   - 回归预测：预测未来价格
   - 交易信号生成

## 主要功能

1. 数据处理：
   - 分钟级数据加载和预处理
   - 技术指标计算（RSI、MACD、布林带等）
   - 序列数据生成
   - 数据标准化

2. 模型训练：
   - 多任务学习
   - 早停机制
   - 学习率自适应调整
   - 梯度裁剪

3. 预测和回测：
   - 价格预测
   - 交易信号生成
   - 策略回测
   - 性能评估（收益率、夏普比率、最大回撤等）

## 配置说明

主要配置参数（在config.py中）：

```python
# 数据配置
SEQUENCE_LENGTH = 96    # 输入序列长度（分钟）
PREDICTION_LENGTH = 24  # 预测长度（分钟）
BATCH_SIZE = 16        # 批次大小

# 模型配置
HIDDEN_SIZE = 128      # 隐藏层维度
NUM_LAYERS = 1         # 层数
NUM_HEADS = 4          # 注意力头数
D_FF = 256            # 前馈网络维度

# 训练配置
NUM_EPOCHS = 1         # 训练轮数
LEARNING_RATE = 0.001  # 学习率
```

## 输出结果

1. 训练过程：
   - 训练损失曲线
   - 验证损失曲线
   - 最佳模型权重

2. 预测结果：
   - 价格预测图表
   - 交易信号标记
   - 回测性能指标

## 注意事项

1. 数据质量：
   - 确保数据质量，处理缺失值和异常值
   - 注意数据的时间对齐
   - 建议使用至少3个月的历史数据

2. 模型调优：
   - 根据实际数据调整模型参数
   - 注意过拟合问题
   - 可以通过调整config.py中的参数来优化性能

3. 交易风险：
   - 回测结果仅供参考
   - 实盘交易需谨慎
   - 建议先进行充分的回测验证

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交Issue或Pull Request。 