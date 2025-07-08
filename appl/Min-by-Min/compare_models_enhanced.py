"""
增强的模型对比实验脚本
支持MLP/LSTM/GRU/xPatch等主流结构的横向对比
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
import sklearn
from packaging import version

from config import Config
from data_processor import DataProcessor
from models.xpatch import XPatch
from models.LSTM import LSTMModel
from models.GRU import GRUModel
from models.MLP import MLPModel

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def safe_rmse(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return mean_squared_error(y_true, y_pred) ** 0.5

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
            return MLPModel(self.config.SEQUENCE_LENGTH, input_dim, self.config.HIDDEN_SIZE,
                          self.config.PREDICTION_LENGTH, self.config.DROPOUT, use_layernorm=True)
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
        """Train a single model (robust shape debug for MLP/multitask)"""
        logger.info(f"Start training model: {model_name}")
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        patience = 3
        counter = 0
        train_losses = []
        val_losses = []
        start_time = time.time()
        for epoch in range(self.config.NUM_EPOCHS):
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                if model_name == 'xpatch':
                    batch_y = batch_y[:, :, -1]  # 只取收盘价，shape=[batch, prediction_length]
                optimizer.zero_grad()
                outputs = model(batch_X)
                if isinstance(outputs, dict):
                    if 'regression_pred' in outputs:
                        outputs = outputs['regression_pred']
                    elif 'output' in outputs:
                        outputs = outputs['output']
                    else:
                        outputs = list(outputs.values())[0]
                if outputs.shape != batch_y.shape:
                    batch_y = batch_y.view_as(outputs)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    if model_name == 'xpatch':
                        batch_y = batch_y[:, :, -1]
                    outputs = model(batch_X)
                    if isinstance(outputs, dict):
                        if 'regression_pred' in outputs:
                            outputs = outputs['regression_pred']
                        elif 'output' in outputs:
                            outputs = outputs['output']
                        else:
                            outputs = list(outputs.values())[0]
                    if outputs.shape != batch_y.shape:
                        batch_y = batch_y.view_as(outputs)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            logger.info(f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
                torch.save(model.state_dict(), f"checkpoints/best_{model_name}.pth")
            else:
                counter += 1
                if counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        training_time = time.time() - start_time
        # 保存loss曲线
        import matplotlib.pyplot as plt
        import os
        os.makedirs('plots', exist_ok=True)
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title(f'{model_name} Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'plots/loss_curve_{model_name}.png')
        plt.close()
        return {
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss if best_val_loss != float('inf') else None,
            'training_time': training_time
        }
    
    def evaluate_model(self, model: nn.Module, test_loader, model_name: str) -> Dict:
        """评估模型性能"""
        logger.info(f"评估模型: {model_name}")
        
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = model(batch_X)
                if isinstance(outputs, dict):
                    if 'regression_pred' in outputs:
                        outputs = outputs['regression_pred']
                    elif 'output' in outputs:
                        outputs = outputs['output']
                    else:
                        outputs = list(outputs.values())[0]
                regression_pred = outputs
                
                all_predictions.extend(regression_pred.cpu().numpy().flatten())
                all_targets.extend(batch_y.cpu().numpy().flatten())
        
        # 计算回归指标
        mae = mean_absolute_error(all_targets, all_predictions)
        rmse = safe_rmse(all_targets, all_predictions)
        r2 = r2_score(all_targets, all_predictions)
        
        results = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
        
        return results
    
    def compare_models(self, model_list: List[str] = ['xpatch', 'lstm', 'gru', 'mlp']) -> Dict:
        """对比多个模型"""
        logger.info("开始模型对比实验")
        
        # 准备数据
        processor = DataProcessor(self.config)
        df = processor.load_data()
        features, labels = processor.prepare_features(df)
        logger.info(f"[DEBUG] features.shape: {features.shape}, labels.shape: {labels.shape}")
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
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False, drop_last=True)
        
        input_dim = features.shape[1]
        logger.info(f"[DEBUG] input_dim for model: {input_dim}")
        
        # 训练和评估每个模型
        for model_name in model_list:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"处理模型: {model_name}")
                logger.info(f"{'='*50}")
                
                # 获取模型
                model = self.get_model(model_name, input_dim)
                checkpoint_path = Path(f"checkpoints/best_{model_name}.pth")
                if checkpoint_path.exists():
                    logger.info(f"检测到已有权重文件 {checkpoint_path}，跳过训练，直接加载权重并评估。")
                    model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
                    training_results = {
                        'model': model,
                        'train_losses': [],
                        'val_losses': [],
                        'best_val_loss': None,
                        'training_time': 0.0
                    }
                else:
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
                
            except Exception as e:
                logger.error(f"模型 {model_name} 训练失败: {str(e)}")
                continue
        
        return self.results
    
    def plot_comparison_results(self, save_path: str = "plots/model_comparison.png"):
        """Draw comparison results (all English, no Chinese)"""
        if not self.results:
            logger.warning("No results to plot.")
            return

        model_names = list(self.results.keys())
        mae_scores = [self.results[name]['evaluation']['MAE'] for name in model_names]
        rmse_scores = [self.results[name]['evaluation']['RMSE'] for name in model_names]
        r2_scores = [self.results[name]['evaluation']['R2'] for name in model_names]
        training_times = [self.results[name]['training']['training_time'] for name in model_names]

        import matplotlib.pyplot as plt
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Model Comparison Results", fontsize=18, fontweight='bold')

        # MAE
        ax1.bar(model_names, mae_scores, color='skyblue')
        ax1.set_title('MAE Comparison (Lower is better)', fontsize=14)
        ax1.set_xlabel('Model')
        ax1.set_ylabel('MAE Value')
        for i, v in enumerate(mae_scores):
            ax1.text(i, v, f"{v:.3f}", ha='center', va='bottom', fontsize=10)

        # RMSE
        ax2.bar(model_names, rmse_scores, color='lightcoral')
        ax2.set_title('RMSE Comparison (Lower is better)', fontsize=14)
        ax2.set_xlabel('Model')
        ax2.set_ylabel('RMSE Value')
        for i, v in enumerate(rmse_scores):
            ax2.text(i, v, f"{v:.3f}", ha='center', va='bottom', fontsize=10)

        # R2
        ax3.bar(model_names, r2_scores, color='lightgreen')
        ax3.set_title('R² Comparison (Higher is better)', fontsize=14)
        ax3.set_xlabel('Model')
        ax3.set_ylabel('R² Value')
        for i, v in enumerate(r2_scores):
            ax3.text(i, v, f"{v:.3f}", ha='center', va='bottom', fontsize=10)

        # Training time
        if any(training_times):
            ax4.bar(model_names, training_times, color='gold')
            ax4.set_title('Training Time Comparison', fontsize=14)
            ax4.set_xlabel('Model')
            ax4.set_ylabel('Training Time (seconds)')
            for i, v in enumerate(training_times):
                ax4.text(i, v, f"{v:.1f}", ha='center', va='bottom', fontsize=10)
        else:
            ax4.set_title('No Training Time Data', fontsize=14)
            ax4.axis('off')

        for ax in [ax1, ax2, ax3, ax4]:
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Comparison plot saved to: {save_path}")
    
    def generate_report(self, save_path: str = "model_comparison_report.txt"):
        """Generate detailed comparison report (robust to NoneType)"""
        if not self.results:
            logger.warning("No results to generate report.")
            return
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("Model Comparison Report\n")
            f.write("="*50 + "\n\n")
            f.write("Experiment Config:\n")
            f.write(f"  Sequence Length: {self.config.SEQUENCE_LENGTH}\n")
            f.write(f"  Prediction Length: {self.config.PREDICTION_LENGTH}\n")
            f.write(f"  Batch Size: {self.config.BATCH_SIZE}\n")
            f.write(f"  Epochs: {self.config.NUM_EPOCHS}\n")
            f.write(f"  Learning Rate: {self.config.LEARNING_RATE}\n\n")
            f.write("Results:\n")
            f.write("-"*30 + "\n")
            for model_name, result in self.results.items():
                f.write(f"\nModel: {model_name}\n")
                train_time = result['training']['training_time']
                best_val_loss = result['training']['best_val_loss']
                f.write(f"  Training Time: {train_time if train_time is not None else '-'} s\n")
                f.write(f"  Best Val Loss: {best_val_loss if best_val_loss is not None else '-'}\n")
                f.write(f"  MAE: {result['evaluation']['MAE']:.6f}\n")
                f.write(f"  RMSE: {result['evaluation']['RMSE']:.6f}\n")
                f.write(f"  R²: {result['evaluation']['R2']:.6f}\n")
            # Summary
            f.write("\nSummary:\n")
            f.write("-"*30 + "\n")
            best_mae_model = min(self.results.keys(), key=lambda x: self.results[x]['evaluation']['MAE'])
            best_r2_model = max(self.results.keys(), key=lambda x: self.results[x]['evaluation']['R2'])
            fastest_model = min(self.results.keys(), key=lambda x: self.results[x]['training']['training_time'] if self.results[x]['training']['training_time'] is not None else float('inf'))
            f.write(f"Best MAE Model: {best_mae_model}\n")
            f.write(f"Best R² Model: {best_r2_model}\n")
            f.write(f"Fastest Model: {fastest_model}\n")
        logger.info(f"Detailed report saved to: {save_path}")

def main():
    """主函数"""
    config = Config()
    
    # 创建模型对比器
    comparator = ModelComparator(config)
    
    # 运行对比实验
    results = comparator.compare_models(['xpatch', 'lstm', 'gru', 'mlp'])
    
    # 绘制对比结果
    comparator.plot_comparison_results()
    
    # 生成详细报告
    comparator.generate_report()
    
    logger.info("模型对比实验完成！")

if __name__ == "__main__":
    main() 