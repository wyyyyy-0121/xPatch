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
        # 动态获取特征数
        df = self.processor.load_data()
        features, _ = self.processor.prepare_features(df)
        feature_dim = features.shape[1]  # 单步特征数
        input_dim = self.config.SEQUENCE_LENGTH * feature_dim
        output_dim = self.config.PREDICTION_LENGTH * feature_dim
        logger.info(f"模型输入维度: {input_dim}, 输出维度: {output_dim}")
        model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )
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
        for batch_X, batch_y in pbar:
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(batch_X.view(batch_X.size(0), -1))
            # reshape输出
            batch_size = outputs.size(0)
            outputs = outputs.view(batch_size, self.config.PREDICTION_LENGTH, -1)
            loss = self.criterion(outputs, batch_y)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
            
            # 清理内存
            del outputs, loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        
        # 使用tqdm创建进度条
        pbar = tqdm(val_loader, desc='Validating', leave=False)
        with torch.no_grad():
            for batch_X, batch_y in pbar:
                outputs = self.model(batch_X.view(batch_X.size(0), -1))
                batch_size = outputs.size(0)
                outputs = outputs.view(batch_size, self.config.PREDICTION_LENGTH, -1)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                
                # 更新进度条
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        return total_loss / len(val_loader)
    
    def train(self):
        """训练模型"""
        logger.info("开始训练...")
        
        # 准备数据
        train_loader, val_loader = self.prepare_data()
        
        # 训练循环
        best_val_loss = float('inf')
        for epoch in range(self.config.NUM_EPOCHS):
            logger.info(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}")
            
            # 训练一个epoch
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss = self.validate(val_loader)
            
            logger.info(f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} - "
                       f"Train Loss: {train_loss:.6f} - "
                       f"Val Loss: {val_loss:.6f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model()
                logger.info("保存最佳模型")
        
        logger.info("训练完成！")
    
    def save_model(self):
        """保存模型"""
        save_path = Path("checkpoints/best_model.pth")
        save_path.parent.mkdir(exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"模型已保存到: {save_path}")

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
    main() 