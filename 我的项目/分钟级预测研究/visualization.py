"""
可视化模块：用于展示模型训练结果
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

def plot_training_metrics(train_losses, val_losses, train_accs, val_accs, save_dir='plots'):
    """
    绘制训练指标
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        train_accs: 训练准确率列表
        val_accs: 验证准确率列表
        save_dir: 保存目录
    """
    # 创建保存目录
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 绘制损失曲线
    ax1.plot(train_losses, label='训练损失', color='blue')
    ax1.plot(val_losses, label='验证损失', color='red')
    ax1.set_title('训练和验证损失')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('损失')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制准确率曲线
    ax2.plot(train_accs, label='训练准确率', color='blue')
    ax2.plot(val_accs, label='验证准确率', color='red')
    ax2.set_title('训练和验证准确率')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('准确率')
    ax2.legend()
    ax2.grid(True)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(f'{save_dir}/training_metrics.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_dir='plots'):
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        save_dir: 保存目录
    """
    # 创建保存目录
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 计算混淆矩阵
    cm = pd.crosstab(y_true, y_pred, rownames=['真实标签'], colnames=['预测标签'])
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    
    # 保存图表
    plt.savefig(f'{save_dir}/confusion_matrix.png')
    plt.close()

def plot_feature_importance(model, feature_names, save_dir='plots'):
    """
    绘制特征重要性
    
    Args:
        model: 训练好的模型
        feature_names: 特征名称列表
        save_dir: 保存目录
    """
    # 创建保存目录
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取特征重要性
    importance = model.get_feature_importance()
    
    # 创建特征重要性DataFrame
    importance_df = pd.DataFrame({
        '特征': feature_names,
        '重要性': importance
    })
    importance_df = importance_df.sort_values('重要性', ascending=False)
    
    # 绘制特征重要性
    plt.figure(figsize=(12, 6))
    sns.barplot(x='重要性', y='特征', data=importance_df)
    plt.title('特征重要性')
    
    # 保存图表
    plt.savefig(f'{save_dir}/feature_importance.png')
    plt.close()

def plot_prediction_vs_actual(y_true, y_pred, save_dir='plots'):
    """
    绘制预测值vs实际值
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        save_dir: 保存目录
    """
    # 创建保存目录
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 绘制散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.title('预测值 vs 实际值')
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    
    # 保存图表
    plt.savefig(f'{save_dir}/prediction_vs_actual.png')
    plt.close()

if __name__ == '__main__':
    # 示例数据
    train_losses = [0.6924, 0.6910, 0.6908, 0.6903, 0.6902, 0.6898, 0.6899, 0.6901]
    val_losses = [0.6912, 0.6959, 0.6900, 0.6908, 0.6906, 0.6924, 0.6914, 0.6922]
    train_accs = [0.5331, 0.5377, 0.5391, 0.5388, 0.5388, 0.5401, 0.5400, 0.5382]
    val_accs = [0.5393, 0.4898, 0.5406, 0.5393, 0.5393, 0.5393, 0.5309, 0.5408]
    
    # 绘制训练指标
    plot_training_metrics(train_losses, val_losses, train_accs, val_accs) 