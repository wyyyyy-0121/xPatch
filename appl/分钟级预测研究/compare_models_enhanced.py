"""
增强的模型对比实验脚本
支持MLP/LSTM/GRU/xPatch等主流结构的横向对比
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report

from config import Config
from data_processor import DataProcessor
from models.xpatch import XPatch
from models.LSTM import LSTMModel
from models.GRU import GRUModel
from models.MLP import MLPModel
from models.multitask_model import MultiTaskLSTM, MultiTaskLoss

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelComparator:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.results = {}
        
    def get_model(self, model_name: str, input_dim: int) -> nn.Module:
        """获取指定模型"""
        if model_name == 'xpatch':
            return XPatch(input_dim, self.config)
        elif model_name == 'lstm':
            return LSTMModel(input_dim, self.config.HIDDEN_SIZE, self.config.NUM_LAYERS, 
                           self.config.PREDICTION_LENGTH, self.config.DROPOUT, use_layernorm=True)
        elif model_name == 'gru':
            return GRUModel(input_dim, self.config.HIDDEN_SIZE, self.config.NUM_LAYERS,
                          self.config.PREDICTION_LENGTH, self.config.DROPOUT, use_layernorm=True)
        elif model_name == 'mlp':
            return MLPModel(input_dim * self.config.SEQUENCE_LENGTH, self.config.HIDDEN_SIZE,
                          self.config.PREDICTION_LENGTH, self.config.DROPOUT, use_layernorm=True)
        elif model_name == 'multitask':
            return MultiTaskLSTM(input_dim, self.config.HIDDEN_SIZE, 2, self.config.PREDICTION_LENGTH)
        else:
            raise ValueError(f"未知模型: {model_name}")
    
    def prepare_classification_targets(self, y: np.ndarray) -> np.ndarray:
        """准备分类目标（涨跌方向）"""
        # 计算价格变化
        price_changes = np.diff(y[:, :, 3], axis=1)  # 假设第4列是收盘价
        
        # 分类标签：0=跌，1=平，2=涨
        threshold = 0.001  # 0.1%的阈值
        classification_targets = np.zeros_like(price_changes, dtype=int)
        classification_targets[price_changes > threshold] = 2   # 涨
        classification_targets[price_changes < -threshold] = 0  # 跌
        classification_targets[(price_changes >= -threshold) & (price_changes <= threshold)] = 1  # 平
        
        return classification_targets
    
    def train_model(self, model: nn.Module, train_loader, val_loader, model_name: str) -> Dict:
        """训练单个模型"""
        logger.info(f"开始训练模型: {model_name}")
        
        model.to(self.device)
        
        # 选择优化器和损失函数
        if model_name == 'multitask':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
            criterion = MultiTaskLoss(regression_weight=1.0, classification_weight=0.5)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
            criterion = nn.MSELoss()
        
        # 训练循环
        best_val_loss = float('inf')
        patience = 3
        counter = 0
        train_losses = []
        val_losses = []
        
        start_time = time.time()
        
        for epoch in range(self.config.NUM_EPOCHS):
            # 训练
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                if model_name == 'multitask':
                    # 多任务学习
                    outputs = model(batch_X)
                    classification_targets = self.prepare_classification_targets(batch_y.cpu().numpy())
                    classification_targets = torch.LongTensor(classification_targets).to(self.device)
                    
                    loss_dict = criterion(outputs, batch_y, classification_targets)
                    loss = loss_dict['total_loss']
                else:
                    # 单任务回归
                    if model_name == 'mlp':
                        outputs = model(batch_X.view(batch_X.size(0), -1))
                    else:
                        outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # 验证
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    if model_name == 'multitask':
                        outputs = model(batch_X)
                        classification_targets = self.prepare_classification_targets(batch_y.cpu().numpy())
                        classification_targets = torch.LongTensor(classification_targets).to(self.device)
                        loss_dict = criterion(outputs, batch_y, classification_targets)
                        loss = loss_dict['total_loss']
                    else:
                        if model_name == 'mlp':
                            outputs = model(batch_X.view(batch_X.size(0), -1))
                        else:
                            outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            logger.info(f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} - "
                       f"Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}")
            
            # 早停
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), f"checkpoints/best_{model_name}.pth")
            else:
                counter += 1
                if counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        training_time = time.time() - start_time
        
        return {
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'training_time': training_time
        }
    
    def evaluate_model(self, model: nn.Module, test_loader, model_name: str) -> Dict:
        """评估模型性能"""
        logger.info(f"评估模型: {model_name}")
        
        model.eval()
        all_predictions = []
        all_targets = []
        all_classification_preds = []
        all_classification_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                if model_name == 'multitask':
                    outputs = model(batch_X)
                    regression_pred = outputs['regression_pred']
                    classification_probs = outputs['classification_probs']
                    
                    # 分类预测
                    classification_pred = torch.argmax(classification_probs, dim=-1)
                    classification_targets = self.prepare_classification_targets(batch_y.cpu().numpy())
                    
                    all_classification_preds.extend(classification_pred.cpu().numpy().flatten())
                    all_classification_targets.extend(classification_targets.flatten())
                else:
                    if model_name == 'mlp':
                        regression_pred = model(batch_X.view(batch_X.size(0), -1))
                    else:
                        regression_pred = model(batch_X)
                
                all_predictions.extend(regression_pred.cpu().numpy().flatten())
                all_targets.extend(batch_y.cpu().numpy().flatten())
        
        # 计算回归指标
        mae = mean_absolute_error(all_targets, all_predictions)
        rmse = mean_squared_error(all_targets, all_predictions, squared=False)
        r2 = r2_score(all_targets, all_predictions)
        
        results = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
        
        # 如果是多任务模型，计算分类指标
        if model_name == 'multitask':
            accuracy = accuracy_score(all_classification_targets, all_classification_preds)
            results['Classification_Accuracy'] = accuracy
            
            # 分类报告
            class_report = classification_report(
                all_classification_targets, all_classification_preds,
                target_names=['跌', '平', '涨']
            )
            results['Classification_Report'] = class_report
        
        return results
    
    def compare_models(self, model_list: List[str] = ['xpatch', 'lstm', 'gru', 'mlp', 'multitask']) -> Dict:
        """对比多个模型"""
        logger.info("开始模型对比实验")
        
        # 准备数据
        processor = DataProcessor(self.config)
        df = processor.load_data()
        features, labels = processor.prepare_features(df)
        X, y = processor.create_sequences(features, labels)
        
        # 数据分割
        N = len(X)
        train_len = int(0.7 * N)
        val_len = int(0.15 * N)
        test_len = N - train_len - val_len
        
        X_train, y_train = X[:train_len], y[:train_len]
        X_val, y_val = X[train_len:train_len+val_len], y[train_len:train_len+val_len]
        X_test, y_test = X[train_len+val_len:], y[train_len+val_len:]
        
        # 创建数据加载器
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(y_val)
        )
        test_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_test), torch.FloatTensor(y_test)
        )
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        input_dim = features.shape[1]
        
        # 训练和评估每个模型
        for model_name in model_list:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"处理模型: {model_name}")
                logger.info(f"{'='*50}")
                
                # 获取模型
                model = self.get_model(model_name, input_dim)
                
                # 训练模型
                training_results = self.train_model(model, train_loader, val_loader, model_name)
                
                # 评估模型
                evaluation_results = self.evaluate_model(training_results['model'], test_loader, model_name)
                
                # 保存结果
                self.results[model_name] = {
                    'training': training_results,
                    'evaluation': evaluation_results
                }
                
                logger.info(f"{model_name} 评估结果:")
                logger.info(f"  MAE: {evaluation_results['MAE']:.6f}")
                logger.info(f"  RMSE: {evaluation_results['RMSE']:.6f}")
                logger.info(f"  R2: {evaluation_results['R2']:.6f}")
                if 'Classification_Accuracy' in evaluation_results:
                    logger.info(f"  分类准确率: {evaluation_results['Classification_Accuracy']:.6f}")
                
            except Exception as e:
                logger.error(f"模型 {model_name} 训练失败: {str(e)}")
                continue
        
        return self.results
    
    def plot_comparison_results(self, save_path: str = "plots/model_comparison.png"):
        """绘制对比结果"""
        if not self.results:
            logger.warning("没有结果可以绘制")
            return
        
        # 准备数据
        model_names = list(self.results.keys())
        mae_scores = [self.results[name]['evaluation']['MAE'] for name in model_names]
        rmse_scores = [self.results[name]['evaluation']['RMSE'] for name in model_names]
        r2_scores = [self.results[name]['evaluation']['R2'] for name in model_names]
        training_times = [self.results[name]['training']['training_time'] for name in model_names]
        
        # 创建子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # MAE对比
        ax1.bar(model_names, mae_scores, color='skyblue')
        ax1.set_title('MAE对比 (越低越好)')
        ax1.set_ylabel('MAE')
        ax1.tick_params(axis='x', rotation=45)
        
        # RMSE对比
        ax2.bar(model_names, rmse_scores, color='lightcoral')
        ax2.set_title('RMSE对比 (越低越好)')
        ax2.set_ylabel('RMSE')
        ax2.tick_params(axis='x', rotation=45)
        
        # R2对比
        ax3.bar(model_names, r2_scores, color='lightgreen')
        ax3.set_title('R²对比 (越高越好)')
        ax3.set_ylabel('R²')
        ax3.tick_params(axis='x', rotation=45)
        
        # 训练时间对比
        ax4.bar(model_names, training_times, color='gold')
        ax4.set_title('训练时间对比')
        ax4.set_ylabel('时间 (秒)')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"对比结果图已保存到: {save_path}")
    
    def generate_report(self, save_path: str = "model_comparison_report.txt"):
        """生成详细的对比报告"""
        if not self.results:
            logger.warning("没有结果可以生成报告")
            return
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("模型对比实验报告\n")
            f.write("="*50 + "\n\n")
            
            f.write("实验配置:\n")
            f.write(f"  序列长度: {self.config.SEQUENCE_LENGTH}\n")
            f.write(f"  预测长度: {self.config.PREDICTION_LENGTH}\n")
            f.write(f"  批次大小: {self.config.BATCH_SIZE}\n")
            f.write(f"  训练轮数: {self.config.NUM_EPOCHS}\n")
            f.write(f"  学习率: {self.config.LEARNING_RATE}\n\n")
            
            f.write("详细结果:\n")
            f.write("-"*30 + "\n")
            
            for model_name, result in self.results.items():
                f.write(f"\n模型: {model_name}\n")
                f.write(f"  训练时间: {result['training']['training_time']:.2f}秒\n")
                f.write(f"  最佳验证损失: {result['training']['best_val_loss']:.6f}\n")
                f.write(f"  MAE: {result['evaluation']['MAE']:.6f}\n")
                f.write(f"  RMSE: {result['evaluation']['RMSE']:.6f}\n")
                f.write(f"  R²: {result['evaluation']['R2']:.6f}\n")
                
                if 'Classification_Accuracy' in result['evaluation']:
                    f.write(f"  分类准确率: {result['evaluation']['Classification_Accuracy']:.6f}\n")
                    f.write(f"  分类报告:\n{result['evaluation']['Classification_Report']}\n")
            
            # 总结
            f.write("\n总结:\n")
            f.write("-"*30 + "\n")
            
            best_mae_model = min(self.results.keys(), key=lambda x: self.results[x]['evaluation']['MAE'])
            best_r2_model = max(self.results.keys(), key=lambda x: self.results[x]['evaluation']['R2'])
            fastest_model = min(self.results.keys(), key=lambda x: self.results[x]['training']['training_time'])
            
            f.write(f"最佳MAE模型: {best_mae_model}\n")
            f.write(f"最佳R²模型: {best_r2_model}\n")
            f.write(f"最快训练模型: {fastest_model}\n")
        
        logger.info(f"详细报告已保存到: {save_path}")

def main():
    """主函数"""
    config = Config()
    
    # 创建模型对比器
    comparator = ModelComparator(config)
    
    # 运行对比实验
    results = comparator.compare_models(['lstm', 'gru', 'mlp', 'multitask'])
    
    # 绘制对比结果
    comparator.plot_comparison_results()
    
    # 生成详细报告
    comparator.generate_report()
    
    logger.info("模型对比实验完成！")

if __name__ == "__main__":
    main() 