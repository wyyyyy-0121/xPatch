"""
配置文件：包含所有模型和数据处理的配置参数
"""

import os
from pathlib import Path
import torch

class Config:
    # 基础路径配置
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / 'data'
    MODEL_DIR = BASE_DIR / 'models'
    CHECKPOINT_DIR = BASE_DIR / 'checkpoints'
    PLOT_DIR = BASE_DIR / 'plots'
    LOG_DIR = BASE_DIR / 'logs'
    
    # 数据配置
    DATA_PATH = str(DATA_DIR / "AAPL_1min.csv")
    SEQUENCE_LENGTH = 60  # 输入序列长度（分钟）
    PREDICTION_LENGTH = 5  # 预测长度（分钟）
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    
    # 模型配置
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    NUM_HEADS = 4
    D_FF = 256
    DROPOUT = 0.1
    PATCH_SIZES = [5, 10, 20]  # 更小的patch size
    EMA_ALPHA = 0.1
    
    # 训练配置
    BATCH_SIZE = 32
    NUM_EPOCHS = 5  # 设置为5个epoch
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 文件路径
    BEST_MODEL_PATH = f"{CHECKPOINT_DIR}/best_model.pth"
    
    # 技术指标参数
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    BB_PERIOD = 20
    BB_STD = 2
    ATR_PERIOD = 14
    EMA_PERIODS = [5, 10, 20, 50, 100]
    
    # 训练配置
    WARMUP_STEPS = 1000  # 预热步数
    MAX_GRAD_NORM = 1.0  # 梯度裁剪阈值
    
    # 早停配置
    EARLY_STOPPING_PATIENCE = 5  # 早停耐心值
    
    # 技术指标参数
    SMA_WINDOW = 20  # 简单移动平均窗口
    EMA_WINDOW = 20  # 指数移动平均窗口
    RSI_WINDOW = 14  # RSI窗口
    BB_WINDOW = 20  # 布林带窗口
    
    # 交易参数
    TRANSACTION_COST = 0.001  # 交易成本
    PRICE_THRESHOLD = 0.001  # 价格变动阈值
    
    # 特征工程配置
    FEATURE_WINDOW = 60  # 特征窗口大小（分钟）
    TECHNICAL_INDICATORS = [
        'RSI', 'MACD', 'BB', 'ATR'  # 技术指标列表
    ]
    
    # 交易配置
    STOP_LOSS = 0.02  # 止损比例（2%）
    TAKE_PROFIT = 0.03  # 止盈比例（3%）
    
    # 特征列配置
    FEATURE_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
    TARGET_COLUMN = 'close'
    
    def __init__(self):
        """初始化配置，创建必要的目录"""
        # 创建必要的目录
        for dir_path in [self.DATA_DIR, self.MODEL_DIR, self.CHECKPOINT_DIR, 
                        self.PLOT_DIR, self.LOG_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True) 