"""
训练脚本：用于训练xPatch模型
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, List
import time
import logging
from data_processor import DataProcessor
from config import Config
import gc
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset):
    """时间序列数据集"""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

class Trainer:
    def __init__(self, config):
        self.config = config
        self.prediction_length = self.config.PREDICTION_LENGTH
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 初始化数据处理器
        self.processor = DataProcessor(config)
        
        # 初始化模型
        self.model = self.init_model()
        self.model.to(self.device)
        
        # 初始化优化器和损失函数
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.criterion = nn.MSELoss()
    
    def init_model(self):
        """初始化模型"""
        from models.LSTM import LSTMModel
        
        # 获取特征维度
        feature_dim = 37  # 根据实际特征数量调整
        
        model = LSTMModel(
            input_dim=feature_dim,
            hidden_size=self.config.HIDDEN_SIZE,
            num_layers=2,  # 减少层数
            prediction_length=self.config.PREDICTION_LENGTH,
            dropout=0.2
        )
        
        logger.info(f"模型输入维度: {feature_dim}, 隐藏维度: {self.config.HIDDEN_SIZE}")
        return model
    
    def prepare_data(self):
        """准备数据加载器"""
        # 获取数据
        X_train, y_train, X_val, y_val = self.processor.prepare_data()
        
        # 转换为PyTorch张量
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.BATCH_SIZE,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        # 使用tqdm创建进度条
        pbar = tqdm(train_loader, desc='Training', leave=False)
        for batch_idx, (batch_X, batch_y) in enumerate(pbar):
            try:
                self.optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                
                # 更新进度条
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})
                
                # 定期清理内存
                if batch_idx % 10 == 0:
                    del outputs, loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    else:
                        gc.collect()
                        
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"内存不足，跳过批次 {batch_idx}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        
        # 使用tqdm创建进度条
        pbar = tqdm(val_loader, desc='Validating', leave=False)
        with torch.no_grad():
            for batch_X, batch_y in pbar:
                outputs = self.model(batch_X)  # 直接传入3D张量，不展平
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                
                # 更新进度条
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        return total_loss / len(val_loader)
    
    def evaluate_metrics(self, y_true, y_pred):
        """计算多种评估指标"""
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2 = r2_score(y_true, y_pred)
        return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

    def train(self):
        """训练模型"""
        logger.info("开始训练...")
        
        # 准备数据
        train_loader, val_loader = self.prepare_data()
        
        # 训练循环
        best_val_loss = float('inf')
        patience = 5
        counter = 0
        best_epoch = 0
        all_train_loss, all_val_loss = [], []
        for epoch in range(self.config.NUM_EPOCHS):
            logger.info(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}")
            
            # 训练一个epoch
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss = self.validate(val_loader)
            all_train_loss.append(train_loss)
            all_val_loss.append(val_loss)
            logger.info(f"Epoch {epoch+1} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
            # EarlyStopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch+1
                counter = 0
                torch.save(self.model.state_dict(), self.config.BEST_MODEL_PATH)
                logger.info(f"保存最佳模型到 {self.config.BEST_MODEL_PATH}")
            else:
                counter += 1
                if counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        logger.info(f"训练结束，最佳Val Loss: {best_val_loss:.6f} (Epoch {best_epoch})")
        # 可视化损失曲线
        try:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(all_train_loss, label='Train Loss')
            plt.plot(all_val_loss, label='Val Loss')
            plt.legend()
            plt.title('Loss Curve')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.savefig('plots/loss_curve.png')
            plt.close()
        except Exception as e:
            logger.warning(f"损失曲线绘制失败: {str(e)}")

        # 新增：可视化LSTM中间特征统计
        try:
            if hasattr(self.model, 'debug_stats') and self.model.debug_stats["out_mean"]:
                plt.figure(figsize=(10,6))
                plt.plot(self.model.debug_stats["out_mean"], label="LSTM out mean")
                plt.plot(self.model.debug_stats["out_std"], label="LSTM out std")
                plt.plot(self.model.debug_stats["pred_mean"], label="Predictions mean")
                plt.plot(self.model.debug_stats["pred_std"], label="Predictions std")
                plt.legend()
                plt.title("LSTM中间特征均值/方差变化")
                plt.xlabel("Batch (累计)")
                plt.ylabel("Value")
                plt.tight_layout()
                Path("plots").mkdir(exist_ok=True)
                plt.savefig("plots/lstm_debug_stats.png")
                plt.close()
                logger.info("已保存LSTM中间特征统计可视化: plots/lstm_debug_stats.png")
        except Exception as e:
            logger.warning(f"LSTM中间特征统计可视化失败: {str(e)}")
    
    def save_model(self):
        """保存模型"""
        save_path = Path("checkpoints/best_model.pth")
        save_path.parent.mkdir(exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"模型已保存到: {save_path}")

def sliding_window_cv(config, window_size=0.6, val_size=0.2, test_size=0.2, n_splits=5):
    """
    滑动窗口交叉验证主控函数
    window_size: 每次训练窗口占总样本比例（如0.6）
    val_size: 验证集比例（如0.2）
    test_size: 测试集比例（如0.2）
    n_splits: 滑窗次数
    """
    from data_processor import DataProcessor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    logger.info(f"滑动窗口交叉验证: window_size={window_size}, val_size={val_size}, test_size={test_size}, n_splits={n_splits}")
    processor = DataProcessor(config)
    df = processor.load_data()
    features, labels = processor.prepare_features(df)
    N = len(features)
    train_len = int(window_size * N)
    val_len = int(val_size * N)
    test_len = int(test_size * N)
    step = (N - train_len - val_len - test_len) // max(n_splits-1,1)
    results = []
    for i in range(n_splits):
        start = i * step
        train_X = features[start : start+train_len]
        train_y = labels[start : start+train_len]
        val_X = features[start+train_len : start+train_len+val_len]
        val_y = labels[start+train_len : start+train_len+val_len]
        test_X = features[start+train_len+val_len : start+train_len+val_len+test_len]
        test_y = labels[start+train_len+val_len : start+train_len+val_len+test_len]
        logger.info(f"滑窗{i+1}/{n_splits}: 训练{len(train_X)} 验证{len(val_X)} 测试{len(test_X)}")
        # 初始化Trainer
        trainer = Trainer(config)
        trainer.processor = processor  # 复用同一数据处理器
        # 训练
        trainer.train()
        # 预测与评估
        trainer.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(test_X).to(trainer.device)
            y = torch.FloatTensor(test_y).to(trainer.device)
            print(f"[调试] LSTM输入X的shape: {X.shape}")
            outputs = trainer.model(X)  # [batch, prediction_length, input_dim]
            print(f"[调试] LSTM输出outputs的shape: {outputs.shape}")
            y_pred = outputs.cpu().numpy().flatten()
            y_true = y.cpu().numpy().flatten()
            mae = mean_absolute_error(y_true, y_pred)
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            r2 = r2_score(y_true, y_pred)
            logger.info(f"滑窗{i+1} MAE={mae:.6f} RMSE={rmse:.6f} R2={r2:.6f}")
            results.append([mae, rmse, r2])
    results = np.array(results)
    logger.info(f"滑动窗口交叉验证均值: MAE={results[:,0].mean():.6f}±{results[:,0].std():.6f} RMSE={results[:,1].mean():.6f}±{results[:,1].std():.6f} R2={results[:,2].mean():.6f}±{results[:,2].std():.6f}")
    return results

def get_model(model_name, input_dim, config):
    if model_name == 'xpatch':
        from models.xpatch import XPatch
        return XPatch(input_dim, config)
    elif model_name == 'lstm':
        from models.LSTM import LSTMModel
        return LSTMModel(input_dim, config.HIDDEN_SIZE, config.NUM_LAYERS, config.PREDICTION_LENGTH, config.DROPOUT, use_layernorm=True)
    elif model_name == 'gru':
        from models.GRU import GRUModel
        return GRUModel(input_dim, config.HIDDEN_SIZE, config.NUM_LAYERS, config.PREDICTION_LENGTH, config.DROPOUT, use_layernorm=True)
    elif model_name == 'mlp':
        from models.MLP import MLPModel
        return MLPModel(input_dim * config.SEQUENCE_LENGTH, config.HIDDEN_SIZE, config.PREDICTION_LENGTH, config.DROPOUT, use_layernorm=True)
    else:
        raise ValueError(f"未知模型: {model_name}")


def compare_models(config, model_list=['xpatch', 'lstm', 'gru', 'mlp']):
    from data_processor import DataProcessor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    processor = DataProcessor(config)
    df = processor.load_data()
    features, labels = processor.prepare_features(df)
    N = len(features)
    train_len = int(0.8 * N)
    test_len = N - train_len
    train_X = features[:train_len]
    train_y = labels[:train_len]
    test_X = features[train_len:]
    test_y = labels[train_len:]
    results = []
    for model_name in model_list:
        logger.info(f"\n==== 训练与评估模型: {model_name} ====")
        input_dim = features.shape[1]
        model = get_model(model_name, input_dim, config)
        model = model.to(config.DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        criterion = torch.nn.MSELoss()
        # 训练
        X = torch.FloatTensor(train_X).to(config.DEVICE)
        y = torch.FloatTensor(train_y).to(config.DEVICE)
        for epoch in range(10):  # 只训练10轮做对比
            model.train()
            optimizer.zero_grad()
            if model_name == 'mlp':
                out = model(X.reshape(X.size(0), -1))
            else:
                out = model(X)
            loss = criterion(out, y.reshape(out.shape))
            loss.backward()
            optimizer.step()
        # 测试
        model.eval()
        X_test = torch.FloatTensor(test_X).to(config.DEVICE)
        y_test = torch.FloatTensor(test_y).to(config.DEVICE)
        with torch.no_grad():
            if model_name == 'mlp':
                pred = model(X_test.reshape(X_test.size(0), -1)).cpu().numpy().flatten()
            else:
                pred = model(X_test).cpu().numpy().flatten()
            y_true = y_test.cpu().numpy().flatten()
            mae = mean_absolute_error(y_true, pred)
            rmse = mean_squared_error(y_true, pred, squared=False)
            r2 = r2_score(y_true, pred)
            logger.info(f"{model_name} MAE={mae:.6f} RMSE={rmse:.6f} R2={r2:.6f}")
            results.append([model_name, mae, rmse, r2])
    logger.info("\n==== 多模型对比结果 ====")
    for r in results:
        logger.info(f"{r[0]}: MAE={r[1]:.6f} RMSE={r[2]:.6f} R2={r[3]:.6f}")
    return results

def main():
    try:
        # 加载配置
        config = Config()
        
        # 初始化训练器
        trainer = Trainer(config)
        
        # 开始训练
        trainer.train()
        
    except Exception as e:
        logger.error(f"训练过程出错: {str(e)}")
        raise

if __name__ == "__main__":
    from config import Config
    # compare_models(Config(), model_list=['xpatch', 'lstm', 'gru', 'mlp'])  # 取消注释可直接运行多模型对比
    sliding_window_cv(Config(), window_size=0.6, val_size=0.2, test_size=0.2, n_splits=5)