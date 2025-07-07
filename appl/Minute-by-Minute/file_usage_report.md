# 文件使用情况报告

## 📁 核心文件（必需）

### 配置文件
- ✅ `config.py` - 配置文件，被多个脚本引用
- ✅ `requirements.txt` - 依赖包列表

### 数据处理
- ✅ `data_processor.py` - 数据处理模块，被train.py、quick_train.py等引用

### 主训练脚本
- ✅ `train.py` - 主训练脚本
- ✅ `quick_train.py` - 快速训练验证脚本（新增）
- ✅ `compare_models_enhanced.py` - 增强模型对比脚本（新增）

### 预测和可视化
- ✅ `predict.py` - 预测脚本
- ✅ `visualization.py` - 可视化模块

### 模型文件
- ✅ `models/xpatch.py` - xPatch主模型
- ✅ `models/LSTM.py` - LSTM模型
- ✅ `models/GRU.py` - GRU模型
- ✅ `models/MLP.py` - MLP模型
- ✅ `models/multitask_model.py` - 多任务学习模型（新增）
- ✅ `models/attention_fusion.py` - 注意力融合模块（新增）
- ✅ `models/learnable_trend.py` - 可学习趋势提取（新增）

### 数据目录
- ✅ `data/` - 数据目录
- ✅ `checkpoints/` - 模型检查点目录
- ✅ `plots/` - 图表输出目录
- ✅ `logs/` - 日志目录

## ⚠️ 可选文件（可删除或保留）

### 调试/测试文件
- ⚠️ `test_training.py` - 测试训练脚本，未被引用，可删除
- ⚠️ `check_setup.py` - 环境检查脚本，未被引用，可删除
- ⚠️ `process_data.py` - 数据处理脚本，未被引用，可删除

### 旧版本文件
- ⚠️ `run.py` - 旧版本运行脚本，未被引用，可删除

### 其他目录
- ⚠️ `layers/` - 层定义目录，可能未被使用
- ⚠️ `__pycache__/` - Python缓存目录，可删除

## 🗑️ 建议删除的文件

以下文件未被任何脚本引用，建议删除：

1. `test_training.py` - 已被 `quick_train.py` 替代
2. `check_setup.py` - 功能已集成到 `quick_train.py`
3. `process_data.py` - 功能已集成到 `data_processor.py`
4. `run.py` - 旧版本，功能已被其他脚本替代

## 📊 文件使用统计

- **核心文件**: 15个
- **可选文件**: 4个
- **建议删除**: 4个
- **总文件数**: 23个

## 🔧 清理建议

```bash
# 删除未使用的文件
rm test_training.py
rm check_setup.py
rm process_data.py
rm run.py

# 删除缓存目录
rm -rf __pycache__/
```

## 📝 说明

- ✅ 标记的文件是项目运行必需的
- ⚠️ 标记的文件是可选的，可以保留或删除
- 🗑️ 标记的文件建议删除，以保持项目整洁

删除建议文件后，项目将更加简洁，只保留核心功能文件。 