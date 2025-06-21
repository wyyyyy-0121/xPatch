"""
用于模型预测和结果可视化
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from config import Config
from data_processor import DataProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Predictor:
    def __init__(self, config: Config, model_path: str):
        """
        初始化预测器
        
        Args:
            config: 配置对象
            model_path: 模型文件路径
        """
        self.config = config
        self.model_path = model_path
        self.device = torch.device(config.DEVICE)
        logger.info(f"使用设备: {self.device}")
        self.model = self._load_model()
        self.data_processor = DataProcessor(config)
        
    def _load_model(self):
        """加载模型"""
        # 加载数据并设置时间索引
        df = pd.read_csv(self.config.DATA_PATH)
        df.columns = ['timestamp'] + list(df.columns[1:])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        features, _ = DataProcessor(self.config).prepare_features(df)
        feature_dim = features.shape[1]
        model = torch.nn.Sequential(
            torch.nn.Linear(self.config.SEQUENCE_LENGTH * feature_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.config.DROPOUT),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.config.DROPOUT),
            torch.nn.Linear(256, self.config.PREDICTION_LENGTH * feature_dim)
        )
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def predict(self, X: torch.Tensor) -> np.ndarray:
        """
        进行预测
        
        Args:
            X: 输入特征
            
        Returns:
            np.ndarray: 预测结果
        """
        try:
            batch_size = X.shape[0]
            X = X.reshape(batch_size, -1)
            with torch.no_grad():
                outputs = self.model(X)
                # reshape为(batch, 5, 特征数)
                feature_dim = outputs.shape[1] // self.config.PREDICTION_LENGTH
                predictions = outputs.reshape(batch_size, self.config.PREDICTION_LENGTH, feature_dim)
                return predictions.cpu().numpy()
        except Exception as e:
            logger.error(f"预测过程出错: {str(e)}")
            raise
    
    def plot_predictions(self, actual: np.ndarray, predicted: np.ndarray, save_path: str = None):
        """
        绘制预测结果
        
        Args:
            actual: 实际值
            predicted: 预测值
            save_path: 保存路径
        """
        plt.figure(figsize=(15, 8))
        
        # 绘制实际值
        plt.plot(actual[:, 0, 3], label='实际收盘价', color='blue')
        
        # 绘制预测值
        plt.plot(predicted[:, 0, 3], label='预测收盘价', color='red', linestyle='--')
        
        plt.title('股票价格预测结果')
        plt.xlabel('时间')
        plt.ylabel('价格')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"预测结果图已保存至: {save_path}")
        
        plt.close()

def main():
    # 加载配置
    config = Config()
    
    # 初始化预测器
    predictor = Predictor(config, config.BEST_MODEL_PATH)
    
    # 加载数据
    data_processor = DataProcessor(config)
    df = data_processor.load_data()
    
    # 准备特征
    logger.info("正在准备特征...")
    features, labels = data_processor.prepare_features(df)
    
    # 创建序列
    logger.info("开始创建序列数据...")
    X, y = data_processor.create_sequences(features, labels)
    
    # 进行预测
    logger.info("开始预测...")
    predictions = predictor.predict(torch.FloatTensor(X))
    
    # 绘制结果
    plot_path = config.PLOT_DIR / "predictions.png"
    predictor.plot_predictions(y, predictions, str(plot_path))

if __name__ == "__main__":
    main() 